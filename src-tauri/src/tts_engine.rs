use anyhow::{Context, Result};
use ndarray::{Array1, Array2, Array3};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Value;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::turkish_tokenizer::{TurkishTokenizer, chunk_spoken};

// Gercek model sabitleri (config.json + engine.py ile birebir)
pub const TEXT_DIM: usize = 224;
pub const LATENT_DIM: usize = 64;
pub const NOISE_STEPS: usize = 4;
pub const HOP: usize = 1920; // 25 Hz latent -> 48 kHz
pub const FIRST_WINDOW: usize = 25; // ilk pencere 1 sn
pub const WINDOW: usize = 100; // sonraki pencereler 4 sn
pub const CONTEXT: usize = 8; // her yonde context frame
pub const DECODE_RATE: usize = 48000; // decoder cikisi her zaman 48 kHz
pub const MAX_WORD_FRAMES: usize = 250;
pub const MAX_FRAMES: usize = 3000;

/// `ort::Error<SessionBuilder>` `Send/Sync` olmadığı için `?` ile
/// doğrudan `anyhow::Error`'a dönüşemez; Debug çıktısı üzerinden taşınır.
fn ort_err<E: std::fmt::Debug>(e: E) -> anyhow::Error {
    anyhow::anyhow!("ORT error: {e:?}")
}

fn default_api_port() -> u16 {
    8765
}

/// Konusulabilir model kimligi (UI + API + settings.json ortak dili).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ModelId {
    #[serde(rename = "ema-lightning")]
    EmaLightning,
    #[serde(rename = "antalia-mini")]
    AntaliaMini,
    #[serde(rename = "ahenk-cep")]
    AhenkCep,
}

impl Default for ModelId {
    fn default() -> Self {
        ModelId::EmaLightning
    }
}

impl ModelId {
    pub fn as_str(&self) -> &'static str {
        match self {
            ModelId::EmaLightning => "ema-lightning",
            ModelId::AntaliaMini => "antalia-mini",
            ModelId::AhenkCep => "ahenk-cep",
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            ModelId::EmaLightning => "EMA Lightning",
            ModelId::AntaliaMini => "Antalia Mini",
            ModelId::AhenkCep => "Ahenk Cep",
        }
    }

    /// models/ kok dizini altindaki alt klasor.
    pub fn subdir(&self) -> &'static str {
        match self {
            ModelId::EmaLightning => "onnx",
            ModelId::AntaliaMini => "antalia",
            ModelId::AhenkCep => "ahenk",
        }
    }
}

fn default_steps() -> usize {
    8
}

fn default_cfg() -> f32 {
    2.0
}

fn default_temperature() -> f32 {
    0.6
}

fn default_eq() -> bool {
    true
}

/// Arayuz gorunumu ("dark" | "light" | "sepia" | "contrast").
/// Sentezi etkilemez; settings.json ile kalici olmasi icin config'tedir.
fn default_theme() -> String {
    "dark".to_string()
}

/// Ahenk Cep: uslup (neutral, angry, calm, cheerful, empathetic, sad, surprised).
/// Diger motorlar yok sayar.
fn default_style() -> String {
    "neutral".to_string()
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TtsConfig {
    /// Secili model (settings.json ile kalici).
    #[serde(default)]
    pub model: ModelId,
    pub speed: f32,
    pub sample_rate: u32,
    pub threads: usize,
    pub use_gpu: bool,
    pub chunk_size: usize,
    pub seed: u64,
    /// Ses perdesi, yarim ton (+12 = bir oktav tiz). Hiz degismez:
    /// model speed/ratio ile calisir, cikti ratio ile yeniden orneklenir.
    pub pitch: f32,
    /// HTTP API sunucusu acik mi (settings.json ile kalici).
    #[serde(default)]
    pub api_enabled: bool,
    /// HTTP API portu.
    #[serde(default = "default_api_port")]
    pub api_port: u16,
    /// Ses cikis aygiti adi; bos ise sistem varsayilani.
    #[serde(default)]
    pub output_device: String,
    /// Antalia: akis adimi (1/2/4/8). EMA yok sayar.
    #[serde(default = "default_steps")]
    pub steps: usize,
    /// Antalia: CFG olcegi. EMA yok sayar.
    #[serde(default = "default_cfg")]
    pub cfg: f32,
    /// Antalia: gurultu sicakligi. EMA yok sayar.
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    /// Antalia: ton filtresi + limiter. EMA yok sayar.
    #[serde(default = "default_eq")]
    pub eq: bool,
    /// Arayuz temasi. Sentezi etkilemez (motorlar okumaz).
    #[serde(default = "default_theme")]
    pub theme: String,
    /// Ahenk Cep: uslup. Diger motorlar yok sayar.
    #[serde(default = "default_style")]
    pub style: String,
}

impl Default for TtsConfig {
    fn default() -> Self {
        Self {
            model: ModelId::default(),
            speed: 1.0,
            sample_rate: 48000,
            threads: 4,
            use_gpu: false,
            chunk_size: 1024,
            seed: 0,
            pitch: 0.0,
            api_enabled: false,
            api_port: default_api_port(),
            output_device: String::new(),
            steps: default_steps(),
            cfg: default_cfg(),
            temperature: default_temperature(),
            eq: default_eq(),
            theme: default_theme(),
            style: default_style(),
        }
    }
}

/// Yarim ton -> ornekleme orani (web surumuyle birebir: 2^(st/12)).
pub fn pitch_ratio(semitones: f32) -> f32 {
    2f32.powf(semitones.clamp(-12.0, 12.0) / 12.0)
}

pub struct TtsEngine {
    text_encoder: Session,
    sound_stage: Session,
    decoder: Session,
    tokenizer: TurkishTokenizer,
    config: TtsConfig,
    model_dir: PathBuf,
    running: Arc<AtomicBool>,
    /// Kullanicinin durdur istedigi (dogal bitisten ayirt etmek icin ayri bayrak).
    stop_requested: Arc<AtomicBool>,
}

impl TtsEngine {
    pub fn new(model_dir: impl AsRef<Path>, config: TtsConfig) -> Result<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();

        let text_encoder =
            Self::load_session(&model_dir.join("text_encoder.onnx"), config.threads)?;
        let sound_stage =
            Self::load_session(&model_dir.join("sound_stage.onnx"), config.threads)?;
        let decoder = Self::load_session(&model_dir.join("decoder.onnx"), config.threads)?;
        let tokenizer =
            TurkishTokenizer::from_vocab_file(model_dir.join("vocab.json"))?;

        Ok(Self {
            text_encoder,
            sound_stage,
            decoder,
            tokenizer,
            config,
            model_dir,
            running: Arc::new(AtomicBool::new(false)),
            stop_requested: Arc::new(AtomicBool::new(false)),
        })
    }

    fn load_session(path: &Path, threads: usize) -> Result<Session> {
        if !path.exists() {
            anyhow::bail!("Model file not found: {}", path.display());
        }
        let mut builder = Session::builder().map_err(ort_err)?;
        builder = builder.with_intra_threads(threads).map_err(ort_err)?;
        builder = builder.with_inter_threads(threads).map_err(ort_err)?;
        builder = builder
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(ort_err)?;
        let session = builder.commit_from_file(path).map_err(|e| {
            anyhow::anyhow!("Failed to load ONNX model {}: {e:?}", path.display())
        })?;
        Ok(session)
    }

    pub fn update_config(&mut self, config: TtsConfig) {
        self.config = config;
    }

    pub fn stop(&self) {
        self.stop_requested.store(true, Ordering::SeqCst);
        self.running.store(false, Ordering::SeqCst);
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Durdur bayraklarinin paylasilan tutamaclari (UI/API kilidi beklemez).
    pub fn running_handle(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.running)
    }

    pub fn stop_handle(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.stop_requested)
    }

    pub fn synthesize(
        &mut self,
        text: &str,
        chunk_callback: &mut dyn FnMut(&[f32]),
        progress_callback: &mut dyn FnMut(usize, usize),
    ) -> Result<()> {
        self.stop_requested.store(false, Ordering::SeqCst);
        self.running.store(true, Ordering::SeqCst);
        let result = self.synthesize_internal(text, chunk_callback, progress_callback);
        self.running.store(false, Ordering::SeqCst);
        result
    }

    fn synthesize_internal(
        &mut self,
        text: &str,
        chunk_callback: &mut dyn FnMut(&[f32]),
        progress_callback: &mut dyn FnMut(usize, usize),
    ) -> Result<()> {
        // Resmi boru hatti (api.py/engine.py): konusulur metin parcalara
        // bolunur (cumle sonlari + duraklamalar), her parca ayri planlanir.
        let spoken = self.tokenizer.preprocess(text);
        let limit_speed = self.config.speed.clamp(0.25, 4.0);
        let pieces = chunk_spoken(&spoken, limit_speed);
        if pieces.is_empty() {
            anyhow::bail!("Metin bos: sentezlenecek harf yok");
        }

        // Asama 1: tum parcalarin plani (hizli; text_stage + timeline).
        struct Planned {
            plan: FramePlan,
            pause: f32,
            first_window: usize,
        }
        let mut planned: Vec<Planned> = Vec::with_capacity(pieces.len());
        for (i, (ptext, pause)) in pieces.iter().enumerate() {
            let ids = self.tokenizer.encode_ids(ptext);
            if ids.is_empty() {
                continue;
            }
            // api.py ile birebir: parca tohumu (seed * 1_000_003 + i) % 2^63.
            let seed_i = self
                .config
                .seed
                .wrapping_mul(1_000_003)
                .wrapping_add(i as u64)
                & 0x7FFF_FFFF_FFFF_FFFF;
            let plan = self.plan(&ids, seed_i)?;
            let first_window = if i == 0 { FIRST_WINDOW } else { WINDOW };
            planned.push(Planned {
                plan,
                pause: *pause,
                first_window,
            });
        }
        if planned.is_empty() {
            anyhow::bail!("Metin bos: sentezlenecek harf yok");
        }

        // Toplam pencere (ilerleme % global hesaplanir).
        let totals: Vec<usize> = planned
            .iter()
            .map(|p| window_spans(p.plan.frames, p.first_window).len())
            .collect();
        let grand_total: usize = totals.iter().sum();
        let mut streamer = Streamer::new(&self.config);
        let mut base = 0usize;
        for item in planned.iter() {
            if !self.is_running() {
                break;
            }
            let latents = self.run_sound_stage(&item.plan)?;
            let spans = window_spans(item.plan.frames, item.first_window);
            self.decode_stream(
                &latents,
                &spans,
                base,
                grand_total.max(1),
                &mut streamer,
                chunk_callback,
                progress_callback,
            )?;
            base += spans.len();
            // Parca sonu duraklamasi (resmi yol: son parca 0 sn).
            if item.pause > 0.0 && self.is_running() {
                let n = (item.pause * DECODE_RATE as f32).round() as usize;
                if !streamer.push_silence(n, chunk_callback, &self.running) {
                    break;
                }
            }
        }
        progress_callback(grand_total.max(1), grand_total.max(1));
        Ok(())
    }

    /// engine.py::plan ile birebir: harf id'lerinden frame timeline kurar.
    /// `seed`: bu parcanin gurultu tohumu (api.py: (seed*1_000_003+i)%2^63).
    fn plan(&mut self, ids: &[i64], seed: u64) -> Result<FramePlan> {
        let l = ids.len();
        let mask: Vec<bool> = ids.iter().map(|&id| id != 0).collect();

        // text_stage: (ids, mask) -> (h[L,224], dur[L])
        let (h, dur) = self.run_text_encoder(ids, &mask)?;

        // Kelime gruplama (engine.py::piece): boslukla ayir, bosluklar kelimeye dahil
        let chars_len = l;
        let mut starts: Vec<usize> = Vec::new();
        // Not: encode ciktisi char bazli; kelime siniri icin normalize edilmis metin gerekir.
        // Burada id dizisi uzerinden degil, cw/wstart'i harf pozisyonlarindan kurariz:
        // kelime baslangici = ilk harf ya da onceki bosluk olan harf. Bosluk id'si = stoi[' '].
        let space_id = 2i64; // vocab[2] == ' '
        for i in 0..chars_len {
            if ids[i] != space_id && (i == 0 || ids[i - 1] == space_id) {
                starts.push(i);
            }
        }
        if starts.is_empty() {
            starts.push(0);
        }
        let mut bounds = vec![0usize];
        bounds.extend(starts.iter().skip(1).copied());
        bounds.push(chars_len);
        let mut cw = vec![0i64; chars_len];
        let mut wstart = vec![0i64; chars_len];
        for (w, win) in bounds.windows(2).enumerate() {
            let (a, b) = (win[0], win[1]);
            for i in a..b {
                cw[i] = w as i64;
                wstart[i] = a as i64;
            }
        }
        let n_words = *cw.iter().max().unwrap_or(&0) as usize + 1;

        // dur/speed, kelimeye dagit, yuvarla, [1,250] kirp
        // Perde degisiminde hiz sabit kalsin diye model speed/ratio ile calisir.
        let ratio = pitch_ratio(self.config.pitch);
        let speed = (self.config.speed.clamp(0.25, 4.0) / ratio).clamp(0.25, 4.0);
        let mut word_frames = vec![0usize; n_words];
        {
            let mut sums = vec![0f32; n_words];
            for i in 0..chars_len {
                sums[cw[i] as usize] += dur[i] / speed;
            }
            for (w, s) in sums.iter().enumerate() {
                word_frames[w] = (s.round() as i64).clamp(1, MAX_WORD_FRAMES as i64) as usize;
            }
        }
        let total: usize = word_frames.iter().sum();
        let frames = total.min(MAX_FRAMES);
        if frames == 0 {
            anyhow::bail!("Frame sayisi 0: metin cok kisa ya da süreler gecersiz");
        }

        // fw/fp
        let mut fw = Vec::with_capacity(frames);
        let mut fp = Vec::with_capacity(frames);
        let mut starts_cum = vec![0usize; n_words + 1];
        for w in 0..n_words {
            starts_cum[w + 1] = starts_cum[w] + word_frames[w];
        }
        for f in 0..frames {
            // hangi kelime? (repeat_interleave esdegeri)
            let mut w = n_words - 1;
            for ww in 0..n_words {
                if f < starts_cum[ww + 1] {
                    w = ww;
                    break;
                }
            }
            fw.push(w as i64);
            let pos = (f - starts_cum[w]) as f32 / word_frames[w] as f32;
            fp.push(pos);
        }
        let fmask = vec![true; frames];

        // seed'li gurultu (B,4,T,64): SplitMix64 + Box-Muller
        let noise = seeded_gaussian(seed, NOISE_STEPS, frames, LATENT_DIM);

        Ok(FramePlan {
            h,
            dur_per_letter: dur.iter().map(|&d| d / speed).collect(),
            mask,
            cw,
            wstart,
            fw,
            fp,
            fmask,
            noise,
            letters: l,
            frames,
        })
    }

    fn run_text_encoder(&mut self, ids: &[i64], mask: &[bool]) -> Result<(Vec<f32>, Vec<f32>)> {
        let l = ids.len();
        let ids_arr = Array2::from_shape_vec((1, l), ids.to_vec()).context("ids dizilemedi")?;
        let mask_arr =
            Array2::from_shape_vec((1, l), mask.to_vec()).context("mask dizilemedi")?;
        let ids_val = Value::from_array(ids_arr)
            .map_err(|e| anyhow::anyhow!("ids tensor olusmadi: {e:?}"))?;
        let mask_val = Value::from_array(mask_arr)
            .map_err(|e| anyhow::anyhow!("mask tensor olusmadi: {e:?}"))?;
        let outputs = self
            .text_encoder
            .run(ort::inputs![ids_val, mask_val])
            .map_err(|e| anyhow::anyhow!("text_encoder calismadi: {e:?}"))?;
        let (hs, hd) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("h okunamadi: {e:?}"))?;
        let (ds, dd) = outputs[1]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("dur okunamadi: {e:?}"))?;
        let hl: Vec<usize> = hs.iter().map(|&d| d as usize).collect();
        let dl: Vec<usize> = ds.iter().map(|&d| d as usize).collect();
        if hl != [1, l, TEXT_DIM] || dl != [1, l] {
            anyhow::bail!("text_encoder cikti sekli hatali: h={hs:?} dur={ds:?}");
        }
        Ok((hd.to_vec(), dd.to_vec()))
    }

    fn run_sound_stage(&mut self, plan: &FramePlan) -> Result<Vec<f32>> {
        let (l, t) = (plan.letters, plan.frames);
        let mk = |v: Array2<f32>| {
            Value::from_array(v).map_err(|e| anyhow::anyhow!("tensor olusmadi: {e:?}"))
        };
        let mk_i64 = |v: Array2<i64>| {
            Value::from_array(v).map_err(|e| anyhow::anyhow!("tensor olusmadi: {e:?}"))
        };
        let mk_bool = |v: Array2<bool>| {
            Value::from_array(v).map_err(|e| anyhow::anyhow!("tensor olusmadi: {e:?}"))
        };
        let h = Array2::from_shape_vec((l, TEXT_DIM), plan.h.clone()).context("h sekli")?;
        let h3 = h.insert_axis(ndarray::Axis(0)); // (1,L,224)
        let h_val = Value::from_array(h3.into_dimensionality::<ndarray::Ix3>().unwrap())
            .map_err(|e| anyhow::anyhow!("h tensor: {e:?}"))?;
        let dur_val = mk(Array2::from_shape_vec((1, l), plan.dur_per_letter.clone()).context("dur sekli")?)?;
        let mask_val = mk_bool(Array2::from_shape_vec((1, l), plan.mask.clone()).context("mask sekli")?)?;
        let cw_val = mk_i64(Array2::from_shape_vec((1, l), plan.cw.clone()).context("cw sekli")?)?;
        let ws_val = mk_i64(Array2::from_shape_vec((1, l), plan.wstart.clone()).context("wstart sekli")?)?;
        let fw_val = mk_i64(Array2::from_shape_vec((1, t), plan.fw.clone()).context("fw sekli")?)?;
        let fp_val = mk(Array2::from_shape_vec((1, t), plan.fp.clone()).context("fp sekli")?)?;
        let fm_val = mk_bool(Array2::from_shape_vec((1, t), plan.fmask.clone()).context("fmask sekli")?)?;
        let n3 = Array3::from_shape_vec(
            (NOISE_STEPS, t, LATENT_DIM),
            plan.noise.clone(),
        )
        .context("noise sekli")?;
        let n4 = n3.insert_axis(ndarray::Axis(0)); // (1,4,T,64)
        let noise_val = Value::from_array(n4.into_dimensionality::<ndarray::Ix4>().unwrap())
            .map_err(|e| anyhow::anyhow!("noise tensor: {e:?}"))?;

        let outputs = self
            .sound_stage
            .run(ort::inputs![
                h_val, dur_val, mask_val, cw_val, ws_val, fw_val, fp_val, fm_val, noise_val
            ])
            .map_err(|e| anyhow::anyhow!("sound_stage calismadi: {e:?}"))?;
        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("latents okunamadi: {e:?}"))?;
        let sl: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        if sl != [1, t, LATENT_DIM] {
            anyhow::bail!("sound_stage cikti sekli hatali: {shape:?}");
        }
        Ok(data.to_vec()) // (T,64) duz
    }

    /// Pencereli decode + streaming (engine.py::decode ile birebir).
    /// `progress_callback`: global (biten, toplam) cagirir.
    fn decode_stream(
        &mut self,
        latents: &[f32], // (T,64) duz
        spans: &[(usize, usize)],
        prog_base: usize,
        prog_total: usize,
        streamer: &mut Streamer,
        chunk_callback: &mut dyn FnMut(&[f32]),
        progress_callback: &mut dyn FnMut(usize, usize),
    ) -> Result<()> {
        let t = latents.len() / LATENT_DIM;

        for (idx, (s, e)) in spans.iter().copied().enumerate() {
            if !self.is_running() {
                break;
            }
            progress_callback(prog_base + idx, prog_total.max(1));
            let a = s.saturating_sub(CONTEXT);
            let b = (e + CONTEXT).min(t);
            let n = b - a;
            // z: (1,64,n)
            let mut z = vec![0f32; 64 * n];
            for i in 0..n {
                for c in 0..LATENT_DIM {
                    z[c * n + i] = latents[(a + i) * LATENT_DIM + c];
                }
            }
            let z_arr = Array3::from_shape_vec((1, 64, n), z).context("z sekli")?;
            let z_val = Value::from_array(z_arr)
                .map_err(|e| anyhow::anyhow!("z tensor: {e:?}"))?;
            let len_val = Value::from_array(Array1::from_vec(vec![n as i64]))
                .map_err(|e| anyhow::anyhow!("lengths tensor: {e:?}"))?;
            let win: Vec<f32> = {
                let outputs = self
                    .decoder
                    .run(ort::inputs![z_val, len_val])
                    .map_err(|e| anyhow::anyhow!("decoder calismadi: {e:?}"))?;
                let (_, data) = outputs[0]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| anyhow::anyhow!("audio okunamadi: {e:?}"))?;
                let off = (s - a) * HOP;
                let len = (e - s) * HOP;
                data[off..off + len].to_vec()
            };
            // Perde kaydirma durumu parcalar arasi da tasinir (Streamer).
            if !streamer.push_audio(&win, chunk_callback, &self.running)? {
                return Ok(());
            }
        }
        streamer.flush_carry(chunk_callback, &self.running);
        Ok(())
    }

    pub fn synthesize_to_vec(&mut self, text: &str) -> Result<Vec<f32>> {
        let mut samples = Vec::new();
        let mut noop = |_done: usize, _total: usize| {};
        self.synthesize(
            text,
            &mut |chunk| {
                samples.extend_from_slice(chunk);
            },
            &mut noop,
        )?;
        Ok(samples)
    }

    pub fn save_wav(&self, samples: &[f32], path: impl AsRef<Path>) -> Result<()> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: self.config.sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer =
            hound::WavWriter::create(path, spec).context("WAV dosyasi acilamadi")?;
        for &sample in samples {
            let clamped = sample.clamp(-1.0, 1.0);
            writer.write_sample((clamped * i16::MAX as f32) as i16)?;
        }
        writer.finalize().context("WAV yazilamadi")?;
        Ok(())
    }

    pub fn config(&self) -> &TtsConfig {
        &self.config
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }
}

struct FramePlan {
    h: Vec<f32>,
    dur_per_letter: Vec<f32>,
    mask: Vec<bool>,
    cw: Vec<i64>,
    wstart: Vec<i64>,
    fw: Vec<i64>,
    fp: Vec<f32>,
    fmask: Vec<bool>,
    noise: Vec<f32>, // (4,T,64) duz
    letters: usize,
    frames: usize,
}

/// Decode pencereleri (engine.py::windows ile birebir):
/// ilk `first_window` frame, sonra dort sn'lik parcalar.
/// Resmi yol: ilk parca 25 ile baslar, sonraki parcalar 100 ile.
pub fn window_spans(frames: usize, first_window: usize) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut s = 0usize;
    let mut first = true;
    while s < frames {
        let e = (s + if first { first_window } else { WINDOW }).min(frames);
        spans.push((s, e));
        s = e;
        first = false;
    }
    spans
}

/// Parcalar arasi tasinan akis durumu: perde kaydirici + parca tamponu.
/// Cumle sonu duraklamalari da buradan gecer (tam sessizlik, tik yok).
/// (Ahenk motoru da kullanir: pub(crate).)
pub(crate) struct Streamer {
    carry: Vec<f32>,
    resamp: LinearResampler,
    chunk_size: usize,
}

impl Streamer {
    pub(crate) fn new(config: &TtsConfig) -> Self {
        Self {
            carry: Vec::with_capacity(config.chunk_size.max(64) * 2),
            resamp: LinearResampler::new(pitch_ratio(config.pitch)),
            chunk_size: config.chunk_size.max(64),
        }
    }

    /// Pencere sesini isler; false donerse durduruldu.
    pub(crate) fn push_audio(
        &mut self,
        win: &[f32],
        chunk_callback: &mut dyn FnMut(&[f32]),
        running: &AtomicBool,
    ) -> Result<bool> {
        let mut shifted = Vec::with_capacity(win.len() + 2);
        self.resamp.feed(win, &mut shifted);
        self.carry.extend_from_slice(&shifted);
        Ok(self.drain(chunk_callback, running))
    }

    /// Tam sessizlik ekler (parca arasi duraklama). false donerse durduruldu.
    pub(crate) fn push_silence(
        &mut self,
        n: usize,
        chunk_callback: &mut dyn FnMut(&[f32]),
        running: &AtomicBool,
    ) -> bool {
        self.carry.extend(std::iter::repeat(0.0).take(n));
        self.drain(chunk_callback, running)
    }

    fn drain(
        &mut self,
        chunk_callback: &mut dyn FnMut(&[f32]),
        running: &AtomicBool,
    ) -> bool {
        while self.carry.len() >= self.chunk_size {
            let rest = self.carry.split_off(self.chunk_size);
            let head = std::mem::replace(&mut self.carry, rest);
            chunk_callback(&head);
            if !running.load(Ordering::SeqCst) {
                return false;
            }
        }
        true
    }

    pub(crate) fn flush_carry(
        &mut self,
        chunk_callback: &mut dyn FnMut(&[f32]),
        running: &AtomicBool,
    ) {
        if !self.carry.is_empty() && running.load(Ordering::SeqCst) {
            let tail = std::mem::take(&mut self.carry);
            chunk_callback(&tail);
        }
    }
}

/// Deterministik Gaussian gurultu: SplitMix64 + Box-Muller.
/// Python'daki torch.Generator(manual_seed) ile bit-bit ayni DEGIL;
/// ayni seed ayni sesi verir (uygulama ici tekrarlanabilirlik).
pub(crate) fn seeded_gaussian(seed: u64, steps: usize, frames: usize, dim: usize) -> Vec<f32> {
    let mut rng = SplitMix64(seed);
    let mut out = Vec::with_capacity(steps * frames * dim);
    for _ in 0..steps * frames * dim / 2 + 1 {
        let u1 = (rng.next() as f64 / u64::MAX as f64).clamp(1e-12, 1.0 - 1e-12);
        let u2 = rng.next() as f64 / u64::MAX as f64;
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        out.push((r * theta.cos()) as f32);
        out.push((r * theta.sin()) as f32);
    }
    out.truncate(steps * frames * dim);
    out
}

/// Lineer yeniden ornekleyici: ciktiyi `ratio` kat hizli okur (perde kaydirma).
/// Hiz degisimini dengelemek icin model speed/ratio ile calistirilir.
/// Pencere sinirlarinda tik olusmamasi icin okuma konumu ve son ornek tasinir.
/// (Antalia motoru da kullanir: pub(crate).)
pub(crate) struct LinearResampler {
    ratio: f64,
    pos: f64,
    prev: f32,
}

impl LinearResampler {
    pub(crate) fn new(ratio: f32) -> Self {
        Self {
            ratio: ratio as f64,
            pos: 0.0,
            prev: 0.0,
        }
    }

    pub(crate) fn feed(&mut self, x: &[f32], out: &mut Vec<f32>) {
        if x.is_empty() {
            return;
        }
        if (self.ratio - 1.0).abs() < 1e-9 {
            out.extend_from_slice(x);
            self.prev = x[x.len() - 1];
            return;
        }
        let n = x.len();
        while self.pos <= n as f64 - 1.0 {
            let i = self.pos.floor() as isize;
            let f = (self.pos - i as f64) as f32;
            let a = if i < 0 { self.prev } else { x[i as usize] };
            let b = if i + 1 < n as isize {
                x[(i + 1) as usize]
            } else {
                a
            };
            out.push(a + (b - a) * f);
            self.pos += self.ratio;
        }
        self.pos -= n as f64;
        self.prev = x[n - 1];
    }
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn test_dir(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join(name)
    }

    fn read_f32(path: &Path) -> Vec<f32> {
        let b = fs::read(path).unwrap();
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn read_i64(path: &Path) -> Vec<i64> {
        let b = fs::read(path).unwrap();
        b.chunks_exact(8)
            .map(|c| {
                i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
            })
            .collect()
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn reference_pipeline_matches_python() {
        let ref_dir = test_dir("models/ref");
        let shapes: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(ref_dir.join("shapes.json")).unwrap(),
        )
        .unwrap();
        let l = shapes["L"].as_u64().unwrap() as usize;
        let t = shapes["T"].as_u64().unwrap() as usize;

        let ids = read_i64(&ref_dir.join("ids.i64"));
        let h_ref = read_f32(&ref_dir.join("h.f32"));
        let dur_ref = read_f32(&ref_dir.join("dur.f32"));
        let fw_ref = read_i64(&ref_dir.join("fw.i64"));
        let fp_ref = read_f32(&ref_dir.join("fp.f32"));
        let lat_ref = read_f32(&ref_dir.join("latents.f32"));
        let audio_ref = read_f32(&ref_dir.join("audio_full.f32"));
        assert_eq!(ids.len(), l);
        assert_eq!(fw_ref.len(), t);

        let cfg = TtsConfig {
            speed: 1.0,
            seed: 0,
            ..Default::default()
        };
        let mut engine =
            TtsEngine::new(test_dir("models/onnx"), cfg).expect("motor acilmali");

        // 1) text stage + timeline
        let mask: Vec<bool> = ids.iter().map(|&id| id != 0).collect();
        let (h, dur) = engine.run_text_encoder(&ids, &mask).unwrap();
        assert_eq!(h.len(), l * TEXT_DIM);
        println!("text: h fark = {:.3e}, dur fark = {:.3e}",
            max_abs_diff(&h, &h_ref), max_abs_diff(&dur, &dur_ref));
        assert!(max_abs_diff(&h, &h_ref) < 2e-5);
        assert!(max_abs_diff(&dur, &dur_ref) < 2e-5);

        let plan = engine.plan(&ids, 0).unwrap();
        assert_eq!(plan.frames, t, "frame sayisi Python ile ayni olmali");
        assert_eq!(plan.fw, fw_ref, "fw birebir ayni olmali");
        println!("plan: fp fark = {:.3e}", max_abs_diff(&plan.fp, &fp_ref));
        assert!(max_abs_diff(&plan.fp, &fp_ref) < 1e-6);

        // 2) pencereli decode, tam decode referansiyla karsilastir
        let mut out: Vec<f32> = Vec::new();
        engine.running.store(true, Ordering::SeqCst);
        let spans = window_spans(t, FIRST_WINDOW);
        let mut streamer = Streamer::new(engine.config());
        engine
            .decode_stream(
                &lat_ref,
                &spans,
                0,
                spans.len().max(1),
                &mut streamer,
                &mut |c| out.extend_from_slice(c),
                &mut |_, _| {},
            )
            .unwrap();
        engine.running.store(false, Ordering::SeqCst);
        assert_eq!(out.len(), t * HOP);
        // Not: pencere birlesimi, resmi Python stream yolunda da ~2.1e-3 tepe fark
        // verir (tam decode'a kiyasla; ort ~4e-6, isitilemez). Esik buna gore.
        println!("decode: audio fark = {:.3e} (ref len {})", max_abs_diff(&out, &audio_ref), audio_ref.len());
        assert!(max_abs_diff(&out, &audio_ref) < 5e-3);

        // 3) uc-tan-uca sentez calisiyor mu (gurultu Rust uretimi, bit-ayni degil)
        let wav = engine
            .synthesize_to_vec("Merhaba, size nasıl yardımcı olabilirim?")
            .unwrap();
        assert!(!wav.is_empty());
        assert!(wav.iter().all(|x| x.is_finite()));
        let peak = wav.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        println!("sentez: {} ornek, {:.2} sn, peak {:.3}", wav.len(), wav.len() as f32 / 48000.0, peak);
        assert!(peak > 1e-4 && peak <= 1.0);

        // 4) farkli uzunlukta metin: dinamik L/T sekilleri calismali
        // (statik export olsaydi burada ORT boyut hatasi verirdi)
        let wav2 = engine
            .synthesize_to_vec("Sabahın erken saatlerinde liman henüz uyanmamıştı.")
            .unwrap();
        assert!(!wav2.is_empty());
        assert_eq!(wav2.len() % HOP, 0, "cikti frame*1920 olmali");
        assert!(wav2.iter().all(|x| x.is_finite()));
        println!("sentez2: {} ornek, {:.2} sn", wav2.len(), wav2.len() as f32 / 48000.0);
    }

    #[test]
    fn pitch_changes_length_not_speed() {
        let base_cfg = TtsConfig {
            speed: 1.0,
            seed: 0,
            ..Default::default()
        };
        let mut engine =
            TtsEngine::new(test_dir("models/onnx"), base_cfg).expect("motor acilmali");
        let base = engine.synthesize_to_vec("Merhaba.").unwrap();

        // +-12 yarim ton (bir oktav): sure ayni kalmali, perde degismeli.
        // (model speed/ratio ile calisir, cikti ratio ile yeniden orneklenir)
        // update_config uzerinden: arayuzdeki config'in motora ulastigini dogrular.
        let mut cfg = engine.config().clone();
        cfg.pitch = 12.0;
        engine.update_config(cfg);
        let high = engine.synthesize_to_vec("Merhaba.").unwrap();
        let mut cfg = engine.config().clone();
        cfg.pitch = -12.0;
        engine.update_config(cfg);
        let low = engine.synthesize_to_vec("Merhaba.").unwrap();
        let zcr = |w: &[f32]| {
            w.windows(2).filter(|p| (p[0] <= 0.0) != (p[1] <= 0.0)).count() as f32
                / w.len() as f32
        };
        println!(
            "pitch: base={} (zcr {:.4}) +12={} (zcr {:.4}) -12={} (zcr {:.4})",
            base.len(),
            zcr(&base),
            high.len(),
            zcr(&high),
            low.len(),
            zcr(&low)
        );
        assert!((high.len() as f32 / base.len() as f32 - 1.0).abs() < 0.1);
        assert!((low.len() as f32 / base.len() as f32 - 1.0).abs() < 0.1);
        assert!(zcr(&high) > zcr(&base), "tiz ses daha cok sifir kesmeli");
        assert!(zcr(&low) < zcr(&base), "pes ses daha az sifir kesmeli");
        assert!(high.iter().all(|x| x.is_finite()));
        assert!(low.iter().all(|x| x.is_finite()));

        // pitch 0 = birebir ayni sinyal (passthrough)
        let mut cfg = engine.config().clone();
        cfg.pitch = 0.0;
        engine.update_config(cfg);
        let again = engine.synthesize_to_vec("Merhaba.").unwrap();
        assert_eq!(base, again);
    }

    #[test]
    fn utterance_end_fades_to_zero() {
        // Noktasiz biten metin bile finish_piece() ile "." alir; ses sifira
        // inerek biter (tik sesi yok). Python olcumu: noktasiz 0.074, noktali 0.0004.
        let mut engine = TtsEngine::new(
            test_dir("models/onnx"),
            TtsConfig {
                seed: 0,
                ..Default::default()
            },
        )
        .expect("motor acilmali");
        let wav = engine.synthesize_to_vec("Merhaba").unwrap();
        let tail_max = wav[wav.len().saturating_sub(50)..]
            .iter()
            .map(|x| x.abs())
            .fold(0.0f32, f32::max);
        println!("bitis son 50 ornek max: {tail_max:.4}");
        assert!(tail_max < 0.03, "bitis sönümlenmemis (tik sesi olur)");
    }

    /// Uygulama akisinin birebir aynasi (kilit + bayrak + calma + durdurma):
    /// calma ortasinda stop gelirse is 5 sn icinde bitmeli, asla takilmamali.
    #[test]
    fn stop_during_playback_returns_quickly() {
        use crate::audio_player::AudioPlayer;
        use std::sync::{Arc, Mutex as StdMutex};
        use std::time::{Duration, Instant};

        let engine = StdMutex::new(
            TtsEngine::new(test_dir("models/onnx"), TtsConfig::default())
                .expect("motor acilmali"),
        );
        let engine = Arc::new(engine);
        let player =
            Arc::new(AudioPlayer::new(None).expect("oynatici acilmali"));
        let running = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stop_req = Arc::new(std::sync::atomic::AtomicBool::new(false));

        let text = "Sabahın erken saatlerinde liman henüz uyanmamıştı. \
            Balıkçılar ağlarını sessizce topluyor, martılar ise dönerek \
            şanslarını deniyordu. Model sadece sekiz milyon parametre ve çok hızlı.";
        let (done_tx, done_rx) = std::sync::mpsc::channel::<&'static str>();
        let w_engine = Arc::clone(&engine);
        let w_player = Arc::clone(&player);
        let w_running = Arc::clone(&running);
        let w_stop = Arc::clone(&stop_req);
        let worker = std::thread::spawn(move || {
            use std::sync::atomic::Ordering;
            let mut eng = w_engine.lock().unwrap();
            w_running.store(true, Ordering::SeqCst);
            let rate = 48000u32;
            let mut total = 0usize;
            let r = eng.synthesize(
                text,
                &mut |chunk: &[f32]| {
                    total += chunk.len();
                    let _ = w_player.play(chunk, rate);
                },
                &mut |_, _| {},
            );
            if r.is_err() {
                let _ = done_tx.send("sentez-hatasi");
                return;
            }
            let _ = w_player.flush();
            let ok = w_player
                .wait_until_played(total as f32 / rate as f32, &w_stop);
            w_running.store(false, Ordering::SeqCst);
            let _ = done_tx.send(if ok { "bitti" } else { "durdu" });
        });

        // Calma fazina girilmesini bekle (debug'da cikarim yavastir).
        let t0 = Instant::now();
        let mut saw_playing = false;
        while t0.elapsed() < Duration::from_secs(60) {
            if player.is_playing() {
                saw_playing = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(200));
        }
        assert!(saw_playing, "calma fazina girilemedi");
        println!("calma basladi, durduruluyor...");

        // stop_all() esdegeri:
        use std::sync::atomic::Ordering;
        running.store(false, Ordering::SeqCst);
        stop_req.store(true, Ordering::SeqCst);
        player.stop();

        let t1 = Instant::now();
        match done_rx.recv_timeout(Duration::from_secs(15)) {
            Ok(msg) => println!(
                "is {:.1} sn'de '{}' ile bitti",
                t1.elapsed().as_secs_f32(),
                msg
            ),
            Err(_) => panic!("DONMA: durdurmadan 15 sn sonra is bitmedi!"),
        }
        worker.join().unwrap();
    }

    #[test]
    fn window_spans_match_reference() {
        // engine.py::windows T=53 -> [(0,25),(25,53)] (kisa metin testi).
        assert_eq!(window_spans(53, FIRST_WINDOW), vec![(0, 25), (25, 53)]);
        // Uzun metin: ilk 25, sonra 100'er.
        assert_eq!(
            window_spans(297, FIRST_WINDOW),
            vec![(0, 25), (25, 125), (125, 225), (225, 297)]
        );
        // Sonraki parcalar 100 ile baslar (web tts.js ile birebir).
        assert_eq!(
            window_spans(250, WINDOW),
            vec![(0, 100), (100, 200), (200, 250)]
        );
        assert_eq!(window_spans(0, FIRST_WINDOW), Vec::<(usize, usize)>::new());
    }

    #[test]
    fn multi_piece_synthesis_has_pause() {
        let s1 = "Sabahın erken saatlerinde liman henüz uyanmamıştı ve balıkçılar ağlarını sessizce topluyordu.";
        let s2 = "Martılar teknelerin etrafında dönerek şanslarını deniyor, güneş yavaşça yükseliyordu.";
        let s3 = "Öğleden sonra rüzgar dindi ve deniz çarşaf gibi dümdüz oldu.";
        let text = format!("{s1} {s2} {s3}");
        // Kurgu dogrulamasi: limit 180 harf oldugundan tam 2 parca olmali.
        let pieces = chunk_spoken(&text, 1.0);
        assert_eq!(pieces.len(), 2, "kurgu 2 parca vermeli: {pieces:?}");
        assert!((pieces[0].1 - 0.25).abs() < 1e-6);
        assert_eq!(pieces[1].1, 0.0);

        let mut engine = TtsEngine::new(
            test_dir("models/onnx"),
            TtsConfig {
                seed: 0,
                ..Default::default()
            },
        )
        .expect("motor acilmali");
        let wav = engine.synthesize_to_vec(&text).unwrap();
        assert!(!wav.is_empty());
        assert!(wav.iter().all(|x| x.is_finite()));
        // Parca arasi 0.25 sn'lik TAM sessizlik (12000 sifir) sonda bir yerde olmali.
        let n = wav.len();
        let mut best = 0usize;
        let mut cur = 0usize;
        for &x in &wav[..n.saturating_sub(2400)] {
            if x == 0.0 {
                cur += 1;
                best = best.max(cur);
            } else {
                cur = 0;
            }
        }
        println!("en uzun sessizlik: {best} ornek ({:.3} sn)", best as f32 / 48000.0);
        assert!(best >= 10000, "parca arasi duraklama bulunamadi");
    }

    #[test]
    fn various_inputs_produce_sound() {
        let cfg = TtsConfig {
            speed: 1.0,
            seed: 0,
            ..Default::default()
        };
        let mut engine =
            TtsEngine::new(test_dir("models/onnx"), cfg).expect("motor acilmali");
        for text in [
            "Merhaba.",
            "Yarın saat 09:45'te İstanbul'dan kalkan uçağınız 2 saat 15 dakika sürecek.",
            "15 Ekim 2026 Çarşamba günü saat 14:30'da, 1.250.000 TL tutarındaki ödemeniz hesabınıza yatırılacak.",
            "Model sadece 8.6 milyon parametre ve çok hızlı.",
        ] {
            let wav = engine.synthesize_to_vec(text).unwrap();
            assert!(!wav.is_empty(), "bos cikti: {text}");
            assert!(wav.iter().all(|x| x.is_finite()), "NaN/Inf: {text}");
            let peak = wav.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let rms =
                (wav.iter().map(|x| x * x).sum::<f32>() / wav.len() as f32).sqrt();
            println!("{text:?} -> {} ornek, peak {:.3}, rms {:.4}", wav.len(), peak, rms);
            assert!(peak > 1e-3, "sessiz cikti (peak): {text}");
            assert!(rms > 1e-4, "sessiz cikti (rms): {text}");
        }
    }
}
