use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use parking_lot::Mutex;
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType,
    WindowFunction,
};
use std::collections::VecDeque;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

/// Cikis aygiti listeleme sonucu.
#[derive(Debug, Clone)]
pub struct OutputDeviceInfo {
    pub name: String,
    pub is_default: bool,
}

/// Tum cikis aygitlarini listeler (secim ekrani icin).
pub fn list_output_devices() -> Result<Vec<OutputDeviceInfo>> {
    let host = cpal::default_host();
    let default_name = host
        .default_output_device()
        .and_then(|d| d.name().ok());
    let mut out = Vec::new();
    for dev in host
        .output_devices()
        .context("Cikis aygitlari listelenemedi")?
    {
        if let Ok(name) = dev.name() {
            out.push(OutputDeviceInfo {
                is_default: Some(&name) == default_name.as_ref(),
                name,
            });
        }
    }
    if let Some(def) = default_name {
        if !out.iter().any(|d| d.name == def) {
            out.insert(
                0,
                OutputDeviceInfo {
                    name: def,
                    is_default: true,
                },
            );
        }
    }
    Ok(out)
}

/// Parcali beslenen durumlu yeniden ornekleyici.
/// SincFixedIn sabit 1024 cerceve ister; kalan parca bir sonrakine tasinir,
/// cumle sonunda flush() ile bosaltilir (sinir tikirtisi olmaz).
struct ResampleStream {
    in_rate: u32,
    resampler: SincFixedIn<f32>,
    pending: Vec<f32>,
}

impl ResampleStream {
    fn new(in_rate: u32, out_rate: u32) -> Result<Self> {
        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        };
        let resampler = SincFixedIn::<f32>::new(
            out_rate as f64 / in_rate as f64,
            2.0,
            params,
            1024,
            1,
        )
        .context("Ornekleyici kurulamadi")?;
        Ok(Self {
            in_rate,
            resampler,
            pending: Vec::new(),
        })
    }

    fn feed(&mut self, input: &[f32], out: &mut Vec<f32>) -> Result<()> {
        self.pending.extend_from_slice(input);
        while self.pending.len() >= 1024 {
            let chunk: Vec<f32> = self.pending.drain(..1024).collect();
            let mut waves = self
                .resampler
                .process(&[chunk], None)
                .context("Yeniden ornekleme olmadi")?;
            out.append(&mut waves.remove(0));
        }
        Ok(())
    }

    fn flush(&mut self, out: &mut Vec<f32>) -> Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let mut buf = vec![0.0f32; 1024];
        let n = self.pending.len();
        buf[..n].copy_from_slice(&self.pending);
        self.pending.clear();
        let mut waves = self
            .resampler
            .process(&[buf], None)
            .context("Yeniden ornekleme olmadi")?;
        out.append(&mut waves.remove(0));
        Ok(())
    }
}

type SharedQueue = Arc<Mutex<VecDeque<f32>>>;
type OwnerSlot = Arc<Mutex<Option<std::thread::Thread>>>;

/// Rust tarafi ses cikisi (UI + HTTP API okutmasi icin tek el).
///
/// `cpal::Stream` bilerek `Send` degildir; akis, onu olusturan ayri bir
/// owner thread'de yasar. Bu yapi yalnizca `Send + Sync` veri tasir ve
/// Tauri state'inde sorunsuz durur.
pub struct AudioPlayer {
    /// Istenen aygit anahtari ("": sistem varsayilani). Degisince yeniden kurulur.
    key: String,
    /// Cozumlenen gercek aygit adi.
    resolved: Option<String>,
    device_rate: u32,
    channels: usize,
    queue: SharedQueue,
    resample: Mutex<Option<ResampleStream>>,
    spawn_lock: Mutex<()>,
    started: AtomicBool,
    shutdown_flag: Arc<AtomicBool>,
    owner: OwnerSlot,
}

impl AudioPlayer {
    /// `preferred`: aygit adi; bos/`None` ise sistem varsayilani.
    /// Bulunamazsa varsayilana duser (hata vermez).
    pub fn new(preferred: Option<&str>) -> Result<Self> {
        let want = preferred.unwrap_or("").trim().to_string();
        let wanted = if want.is_empty() { None } else { Some(want.clone()) };
        let (rate, channels, actual) = resolve_device(wanted.as_deref())?;
        log::info!(
            "Ses aygiti: {} ({rate} Hz, {channels} kanal)",
            actual.as_deref().unwrap_or("<varsayilan>")
        );
        Ok(Self {
            key: want,
            resolved: actual,
            device_rate: rate,
            channels,
            queue: Arc::new(Mutex::new(VecDeque::new())),
            resample: Mutex::new(None),
            spawn_lock: Mutex::new(()),
            started: AtomicBool::new(false),
            shutdown_flag: Arc::new(AtomicBool::new(false)),
            owner: Arc::new(Mutex::new(None)),
        })
    }

    /// Istenen anahtar ("" = varsayilan). Aygit degisikligi karsilastirmasi icin.
    pub fn key(&self) -> &str {
        &self.key
    }

    /// Cozumlenen gercek aygit adi (yoksa sistem varsayilanidir).
    pub fn resolved_name(&self) -> Option<&str> {
        self.resolved.as_deref()
    }

    /// Owner thread'i bir kez baslatir (idempotent).
    fn ensure_started(&self) -> Result<()> {
        if self.shutdown_flag.load(Ordering::SeqCst) {
            anyhow::bail!("Oynatici kapatildi (aygit degisti)");
        }
        if self.started.load(Ordering::SeqCst) {
            return Ok(());
        }
        let _guard = self.spawn_lock.lock();
        if self.started.load(Ordering::SeqCst) {
            return Ok(());
        }
        if self.shutdown_flag.load(Ordering::SeqCst) {
            anyhow::bail!("Oynatici kapatildi (aygit degisti)");
        }
        let queue = Arc::clone(&self.queue);
        let shutdown = Arc::clone(&self.shutdown_flag);
        let owner_slot = Arc::clone(&self.owner);
        let wanted = self.resolved.clone();
        let (tx, rx) = std::sync::mpsc::channel::<Result<(), String>>();
        std::thread::Builder::new()
            .name("ema-audio".into())
            .spawn(move || {
                match build_stream(queue, &wanted) {
                    Ok(_stream) => {
                        if let Err(e) = _stream.play() {
                            let _ = tx.send(Err(format!("Ses akisi baslatilamadi: {e}")));
                            return;
                        }
                        *owner_slot.lock() = Some(std::thread::current());
                        let _ = tx.send(Ok(()));
                        // Akis (_stream) bu thread'de yasar.
                        loop {
                            std::thread::park();
                            if shutdown.load(Ordering::SeqCst) {
                                break;
                            }
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(format!("Ses akisi acilamadi: {e:#}")));
                    }
                }
            })
            .context("Ses threadi acilamadi")?;
        rx.recv()
            .map_err(|e| anyhow::anyhow!("Ses threadi yanit vermedi: {e}"))?
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        self.started.store(true, Ordering::SeqCst);
        Ok(())
    }

    /// Mono PCM'i (verilen hizda) kuyruga ekler; hiz farkliysa donusturur.
    /// Parcali cagrilabilir (streaming); cumle sonunda flush() cagrilmalidir.
    pub fn play(&self, samples: &[f32], in_rate: u32) -> Result<()> {
        if samples.is_empty() {
            return Ok(());
        }
        self.ensure_started()?;
        let ready: Vec<f32> = if in_rate == self.device_rate {
            samples.to_vec()
        } else {
            let mut rs = self.resample.lock();
            let need_new = rs.as_ref().map(|r| r.in_rate != in_rate).unwrap_or(true);
            if need_new {
                *rs = Some(ResampleStream::new(in_rate, self.device_rate)?);
            }
            let mut out = Vec::new();
            rs.as_mut()
                .expect("resampler kuruldu")
                .feed(samples, &mut out)?;
            out
        };
        let mut q = self.queue.lock();
        let ch = self.channels;
        q.reserve(ready.len() * ch);
        for &s in &ready {
            for _ in 0..ch {
                q.push_back(s);
            }
        }
        Ok(())
    }

    /// Kalan parcalari isleyip kuyruga ekler (cumle sonu).
    pub fn flush(&self) -> Result<()> {
        let mut extra = Vec::new();
        {
            let mut rs = self.resample.lock();
            if let Some(r) = rs.as_mut() {
                r.flush(&mut extra)?;
            }
        }
        if !extra.is_empty() {
            let mut q = self.queue.lock();
            let ch = self.channels;
            q.reserve(extra.len() * ch);
            for &s in &extra {
                for _ in 0..ch {
                    q.push_back(s);
                }
            }
        }
        Ok(())
    }

    /// Kuyrugu bosaltir (sesi keser). Akis acik kalir.
    pub fn stop(&self) {
        self.queue.lock().clear();
    }

    /// Kalici kapatma (aygit degisiminde): kuyruk bosalir, owner thread sonlanir.
    pub fn shutdown(&self) {
        self.shutdown_flag.store(true, Ordering::SeqCst);
        self.queue.lock().clear();
        self.started.store(false, Ordering::SeqCst);
        if let Some(t) = self.owner.lock().take() {
            t.unpark();
        }
    }

    pub fn is_playing(&self) -> bool {
        !self.queue.lock().is_empty()
    }

    /// Kuyruktaki sesin bitis suresi (sn).
    pub fn queued_seconds(&self) -> f32 {
        let q = self.queue.lock();
        q.len() as f32 / self.channels as f32 / self.device_rate as f32
    }

    /// Kuyruk bosalana kadar bekler (veya durdurulana). true = tamam caldi.
    pub fn wait_until_played(&self, expected_secs: f32, stop: &AtomicBool) -> bool {
        let start = Instant::now();
        loop {
            if stop.load(Ordering::SeqCst) {
                return false;
            }
            let elapsed = start.elapsed().as_secs_f32();
            if !self.is_playing() && elapsed >= expected_secs + 0.3 {
                return true;
            }
            if elapsed > expected_secs + 30.0 {
                log::warn!("Ses bitis sinyali gelmedi, guvenlik supabi devreye girdi");
                return true;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    }
}

/// Istenen ada uyan aygiti cozumler (yoksa varsayilan).
/// Donus: (hiz, kanal, gercek ad [yoksa None = sistem varsayilani]).
fn resolve_device(preferred: Option<&str>) -> Result<(u32, usize, Option<String>)> {
    let host = cpal::default_host();
    if let Some(want) = preferred {
        if let Ok(devices) = host.output_devices() {
            for dev in devices {
                if let Ok(name) = dev.name() {
                    if name == want {
                        if let Ok(supported) = dev.default_output_config() {
                            return Ok((
                                supported.sample_rate().0,
                                supported.channels() as usize,
                                Some(name),
                            ));
                        }
                    }
                }
            }
        }
        log::warn!("Istenen aygit bulunamadi ({want}), varsayilana dusuluyor");
    }
    let device = host
        .default_output_device()
        .context("Varsayilan ses cikis aygiti bulunamadi")?;
    let supported = device
        .default_output_config()
        .context("Ses aygiti yapilandirmasi okunamadi")?;
    Ok((
        supported.sample_rate().0,
        supported.channels() as usize,
        device.name().ok(),
    ))
}

/// Akisi OLUSTURAN thread'de cagrilir (Stream tasiyamaz).
/// `wanted`: `Some(ad)` ise o aygit, yoksa sistem varsayilani acilir.
fn build_stream(queue: SharedQueue, wanted: &Option<String>) -> Result<cpal::Stream> {
    let host = cpal::default_host();
    let device = match wanted {
        Some(want) => {
            let mut found = None;
            if let Ok(devices) = host.output_devices() {
                for dev in devices {
                    if let Ok(name) = dev.name() {
                        if &name == want {
                            found = Some(dev);
                            break;
                        }
                    }
                }
            }
            found.or_else(|| host.default_output_device()).context(
                "Ne istenen ne varsayilan ses aygiti bulunamadi",
            )?
        }
        None => host
            .default_output_device()
            .context("Varsayilan ses cikis aygiti bulunamadi")?,
    };
    let supported = device
        .default_output_config()
        .context("Ses aygiti yapilandirmasi okunamadi")?;
    let config = supported.config();
    let err_fn = |err| log::error!("Ses akis hatasi: {err}");
    let ch = supported.channels() as usize;

    let stream = match supported.sample_format() {
        cpal::SampleFormat::F32 => device.build_output_stream(
            &config,
            move |data: &mut [f32], _| {
                let mut q = queue.lock();
                for frame in data.chunks_mut(ch) {
                    for s in frame.iter_mut() {
                        *s = q.pop_front().unwrap_or(0.0);
                    }
                }
            },
            err_fn,
            None,
        ),
        cpal::SampleFormat::I16 => device.build_output_stream(
            &config,
            move |data: &mut [i16], _| {
                let mut q = queue.lock();
                for frame in data.chunks_mut(ch) {
                    for s in frame.iter_mut() {
                        let v = q.pop_front().unwrap_or(0.0).clamp(-1.0, 1.0);
                        *s = (v * 32767.0) as i16;
                    }
                }
            },
            err_fn,
            None,
        ),
        cpal::SampleFormat::U16 => device.build_output_stream(
            &config,
            move |data: &mut [u16], _| {
                let mut q = queue.lock();
                for frame in data.chunks_mut(ch) {
                    for s in frame.iter_mut() {
                        let v = q.pop_front().unwrap_or(0.0).clamp(-1.0, 1.0);
                        *s = ((v * 0.5 + 0.5) * 65535.0) as u16;
                    }
                }
            },
            err_fn,
            None,
        ),
        fmt => anyhow::bail!("Desteklenmeyen ses formati: {fmt:?}"),
    }
    .context("Ses akisi acilamadi")?;
    Ok(stream)
}

/// Tek kanalli yeniden ornekleme (tek seferlik; testler icin).
#[allow(dead_code)]
fn resample_mono(input: &[f32], in_rate: u32, out_rate: u32) -> Result<Vec<f32>> {
    if in_rate == out_rate {
        return Ok(input.to_vec());
    }
    let mut rs = ResampleStream::new(in_rate, out_rate)?;
    let mut out = Vec::new();
    rs.feed(input, &mut out)?;
    rs.flush(&mut out)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_passthrough_same_rate() {
        let x: Vec<f32> = (0..5000).map(|i| (i as f32 * 0.01).sin()).collect();
        let y = resample_mono(&x, 48000, 48000).unwrap();
        assert_eq!(x, y);
    }

    #[test]
    fn resample_changes_length_proportionally() {
        let x = vec![0.5f32; 48000]; // 1 sn @48k
        let y = resample_mono(&x, 48000, 44100).unwrap();
        let ratio = y.len() as f32 / x.len() as f32;
        assert!((ratio - 44100.0 / 48000.0).abs() < 0.02, "oran {ratio}");
        assert!(y.iter().all(|v| (v - 0.5).abs() < 0.05));
    }

    #[test]
    fn chunked_feed_matches_oneshot() {
        // Parcali besleme + flush, tek seferle ayni sonucu vermeli (sinir tikirtisi yok).
        let x: Vec<f32> = (0..20000).map(|i| (i as f32 * 0.02).sin()).collect();
        let one = resample_mono(&x, 48000, 44100).unwrap();
        let mut rs = ResampleStream::new(48000, 44100).unwrap();
        let mut chunked = Vec::new();
        for piece in x.chunks(1024) {
            rs.feed(piece, &mut chunked).unwrap();
        }
        rs.flush(&mut chunked).unwrap();
        assert_eq!(one.len(), chunked.len());
        let diff = one
            .iter()
            .zip(chunked.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(diff < 1e-6, "fark {diff}");
    }
}
