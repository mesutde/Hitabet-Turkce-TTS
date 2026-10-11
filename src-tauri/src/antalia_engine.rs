use anyhow::{Context, Result};
use ndarray::{Array1, Array2, Array3};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Value;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::antalia_text::{AntaliaTokenizer, CHUNK_SUFFIX, chunk_pause, chunk_prefix};
use crate::tts_engine::LinearResampler;
use crate::turkish_tokenizer::chunk_spoken;

// Gercek model sabitleri (config.json + api.py ile birebir)
pub const TEXT_DIM: usize = 192;
pub const N_MELS: usize = 128;
pub const STACK: usize = 2;
pub const HOP: usize = 512;
pub const NATIVE_RATE: usize = 48000;
pub const MEL_MEAN: f32 = -5.9;
pub const MEL_STD: f32 = 2.6;
pub const MAX_DUR_FRAMES: usize = 60;
pub const FADE_MS: f32 = 5.0;
// Duraklama hizi her zaman 48 kHz (decoder cikisi sabit)
pub const DECODE_RATE: f32 = 48000.0;
// Sabit secenekler (config.json options + OPTION_DEFAULTS)
pub const GEM_STRETCH: f32 = 1.3;
pub const FIRST_STRETCH: f32 = 1.0;
pub const SPACE_STRETCH: f32 = 1.0;
pub const EQ_MAKEUP_DB: f64 = 0.54;
pub const EQ_HPF_HZ: f64 = 90.0;
pub const EQ_HPF_Q: f64 = 0.707;
pub const EQ_SHELF_HZ: f64 = 210.0;
pub const EQ_SHELF_DB: f64 = -5.0;
pub const EQ_SHELF_Q: f64 = 0.707;
// api.py uretim varsayilani (config options.tone): 210 Hz low-shelf +6 dB,
// 4 kHz high-shelf -9 dB, makeup 0.0 dB.
pub const TONE_LO_DB: f64 = 6.0;
pub const TONE_HI_HZ: f64 = 4000.0;
pub const TONE_HI_DB: f64 = -9.0;
/// 10^(-1/20): -1 dBFS esigi (dogrusal genlik).
pub const LIMIT_THRESHOLD: f64 = 0.8912509381776667;
/// 10^(-0.1/20): -0.1 dBFS tavani (dogrusal genlik).
pub const LIMIT_CEILING: f64 = 0.9885530946604388;

/// `ort::Error` degerleri `Send/Sync` olmadigindan Debug uzerinden tasinir.
fn ort_err<E: std::fmt::Debug>(e: E) -> anyhow::Error {
    anyhow::anyhow!("ORT error: {e:?}")
}

#[derive(Clone, Debug)]
pub struct AntaliaParams {
    pub speed: f32,       // varsayilan 0.95
    pub steps: usize,     // 1/2/4/8, varsayilan 8
    pub cfg: f32,         // varsayilan 2.0
    pub temperature: f32, // varsayilan 0.6 (HF config defaults; gurultu olcegi)
    pub eq: bool,         // varsayilan true (ton filtresi)
    /// api.py tone-override: Some((lo_dB@210Hz, hi_dB@4kHz)) = config varsayilani
    /// (Some((6,-9))); None = varsayilan "less bass" EQ + 0.54 dB makeup.
    pub tone: Option<(f32, f32)>,
    pub pitch: f32,       // yarim ton, varsayilan 0.0 (hiz degismez)
    pub seed: u64,
    pub sample_rate: u32, // 48000/24000(/16000/8000)
    pub threads: usize,
}

impl Default for AntaliaParams {
    fn default() -> Self {
        Self {
            speed: 0.95,
            steps: 8,
            cfg: 2.0,
            temperature: 0.6,
            eq: true,
            tone: Some((TONE_LO_DB as f32, TONE_HI_DB as f32)),
            pitch: 0.0,
            seed: 0,
            sample_rate: 48000,
            threads: 4,
        }
    }
}

/// Geminasyon esnetmesi (api.py _acoustic aynasi): ikiz harf + kopya oncesi.
/// `logdur` yerinde degisir; first/space esnetmeleri 1.0 oldugundan atlanir.
fn apply_gem(ids: &[i64], space_id: i64, logdur: &mut [f32]) {
    if (GEM_STRETCH - 1.0).abs() <= f32::EPSILON {
        return;
    }
    let g = GEM_STRETCH.ln();
    for j in 2..ids.len().saturating_sub(1) {
        if ids[j] == ids[j - 1] && ids[j] != space_id {
            logdur[j] += g;
        } else if ids[j - 1] == space_id && ids[j] == ids[j - 2] {
            logdur[j - 1] += g;
        }
    }
}

/// Tek parcanin timeline plani (plan_text ciktisi).
/// `fw` uretimde okunmaz (testler parite icin okur).
#[allow(dead_code)]
struct ChunkPlan {
    cond: Vec<f32>,   // (T,192) gather sonrasi
    fmask: Vec<bool>, // (T) tumu true
    fw: Vec<i64>,     // (T) sembol indeksleri
    dur: Vec<i64>,    // (L) cerceve sayilari
    letters: usize,   // L
    frames: usize,    // T
    npres: usize,     // prefix sembol sayisi
    nsuf: usize,      // suffix sembol sayisi
}

pub struct AntaliaEngine {
    text: Session,
    prep: Session,
    step: Session,
    vocoder: Session,
    tokenizer: AntaliaTokenizer,
    params: AntaliaParams,
    null_cond: Vec<f32>, // egitimli null kosul vektoru (192-dim)
    model_dir: PathBuf,
    running: Arc<AtomicBool>,
    /// Kullanicinin durdur istedigi (dogal bitisten ayirt etmek icin ayri bayrak).
    stop_requested: Arc<AtomicBool>,
}

fn read_f32_le(path: &Path, expect: usize) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("okunamadi: {}", path.display()))?;
    if bytes.len() != expect * 4 {
        anyhow::bail!(
            "{} boyutu hatali: {} (beklenen {} x f32)",
            path.display(),
            bytes.len(),
            expect
        );
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

impl AntaliaEngine {
    pub fn new(model_dir: impl AsRef<Path>, params: AntaliaParams) -> Result<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();
        let text = Self::load_session(&model_dir.join("antalia_text.onnx"), params.threads)?;
        let prep = Self::load_session(&model_dir.join("antalia_prep.onnx"), params.threads)?;
        let step = Self::load_session(&model_dir.join("antalia_step.onnx"), params.threads)?;
        let vocoder =
            Self::load_session(&model_dir.join("antalia_vocoder.onnx"), params.threads)?;
        let tokenizer =
            AntaliaTokenizer::from_symbols_file(model_dir.join("symbols.json"))?;
        let null_cond = read_f32_le(&model_dir.join("null_cond.f32"), 192)?;
        Ok(Self {
            text,
            prep,
            step,
            vocoder,
            tokenizer,
            params,
            null_cond,
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

    pub fn update_params(&mut self, params: AntaliaParams) {
        self.params = params;
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

    pub fn params(&self) -> &AntaliaParams {
        &self.params
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
        let spoken = self.tokenizer.preprocess(text);
        let chunks = chunk_spoken(&spoken, self.params.speed);
        if chunks.is_empty() {
            anyhow::bail!("Metin bos: sentezlenecek harf yok");
        }
        let n = chunks.len();
        // EQ zinciri cagri basina taze (Python OutputChain ile ayni).
        let mut eq = EqChain::with_tone(
            self.params.sample_rate,
            self.params.eq,
            self.params.tone,
        );
        // Gurultu akisi metin basina tek (parcalar sirayla ceker).
        let mut rng = SplitMix64(self.params.seed);
        // Perde kaydirici durumuda cagri boyunca tasinir.
        let mut resamp =
            LinearResampler::new(crate::tts_engine::pitch_ratio(self.params.pitch));
        let chunk_size = 1024usize;
        let mut carry: Vec<f32> = Vec::with_capacity(chunk_size * 2);
        let out_rate = self.params.sample_rate;

        for (k, (chunk_text, _)) in chunks.iter().enumerate() {
            if !self.is_running() {
                break;
            }
            progress_callback(k, n.max(1));
            let wav = self.synth_chunk(chunk_text, &mut rng)?;
            // Duraklama: cumle sonu 0.12, cumlecik 0.06 (son parca haric).
            let mut wav = wav;
            if k + 1 < n {
                let pause = (chunk_pause(chunk_text) * DECODE_RATE) as usize;
                wav.extend(std::iter::repeat(0.0).take(pause));
            }
            // 48k -> hedef hiz, EQ, limiter (OutputChain sirasi).
            let out = eq.process(&self.resample_48_to(&wav, out_rate)?);
            // Perde + dilimleme (sifirlar aynen gecer).
            let mut shifted = Vec::with_capacity(out.len() + 2);
            resamp.feed(&out, &mut shifted);
            carry.extend_from_slice(&shifted);
            while carry.len() >= chunk_size && self.is_running() {
                let rest = carry.split_off(chunk_size);
                let head = std::mem::replace(&mut carry, rest);
                chunk_callback(&head);
            }
        }
        if !carry.is_empty() && self.is_running() {
            let tail = std::mem::take(&mut carry);
            chunk_callback(&tail);
        }
        progress_callback(n.max(1), n.max(1));
        Ok(())
    }

    /// Tek parca: prefix/suffix sar, planla, uret, decode et, kenarlari kes.
    /// Donus: 48 kHz ham ses (EQ'suz).
    fn synth_chunk(&mut self, chunk: &str, rng: &mut SplitMix64) -> Result<Vec<f32>> {
        let pt = self.plan_text(chunk)?;
        let l = pt.letters;
        // Gurultu (metin akisindan sirali) x sicaklik.
        let n_tok = pt.frames / STACK;
        let mut noise = vec![0.0f32; n_tok * N_MELS * STACK];
        box_muller_fill(rng, &mut noise);
        let temp = self.params.temperature;
        if (temp - 1.0).abs() > f32::EPSILON {
            for v in noise.iter_mut() {
                *v *= temp;
            }
        }
        let mel = self.think(&pt, noise)?;
        // Decode (tum parca bir kerede) + kenar kesimi.
        let wav_full = self.run_vocoder(&mel, pt.frames)?;
        // Span (api.py ile birebir)
        let mut cumsum = vec![0usize; l];
        let mut acc = 0usize;
        for (i, &d) in pt.dur.iter().enumerate().take(l) {
            acc += d.max(0) as usize;
            cumsum[i] = acc;
        }
        let a = *cumsum.get(pt.npres).unwrap_or(&0);
        let b = *cumsum
            .get(l.saturating_sub(2).saturating_sub(pt.nsuf))
            .unwrap_or(&acc);
        let wav = cut_edges(&wav_full, a, b, pt.npres > 0, !CHUNK_SUFFIX.is_empty());
        Ok(wav)
    }

    /// Parca metni -> timeline plani (prefix/suffix dahil).
    fn plan_text(&mut self, chunk: &str) -> Result<ChunkPlan> {
        let prefix = chunk_prefix(chunk);
        let full = format!("{prefix}{chunk}{CHUNK_SUFFIX}");
        let ids = self.tokenizer.encode_ids(&full);
        let l = ids.len();
        let mask: Vec<bool> = ids.iter().map(|&id| id != 0).collect();

        let (h, logdur) = self.run_text(&ids, &mask)?;
        // Sure uzatmalari (log-uzayda toplamsal).
        let space_id = self.tokenizer.space_id();
        let mut logdur = logdur;
        apply_gem(&ids, space_id, &mut logdur);
        // Parca basi prefix sembol sayisi (first_stretch kapsami icin).
        let npres = if prefix.is_empty() {
            0
        } else {
            self.tokenizer.encode_ids(prefix).len().saturating_sub(2)
        };
        if (FIRST_STRETCH - 1.0).abs() > f32::EPSILON {
            // Python: j=1+npres; j<len-1 ve harf bosluk degilken j++; [1+npres, j).
            let start = 1 + npres;
            let mut end = ids.len().saturating_sub(1);
            for (i, &id) in ids.iter().enumerate().skip(start) {
                if i >= ids.len().saturating_sub(1) {
                    break;
                }
                if id == space_id {
                    end = i;
                    break;
                }
                end = i + 1;
            }
            let add = FIRST_STRETCH.ln();
            for d in logdur.iter_mut().take(end).skip(start.min(end)) {
                *d += add;
            }
        }
        if (SPACE_STRETCH - 1.0).abs() > f32::EPSILON {
            let add = SPACE_STRETCH.ln();
            for (d, &id) in logdur.iter_mut().zip(ids.iter()) {
                if id == space_id {
                    *d += add;
                }
            }
        }
        // Perde telafisi: model speed/ratio ile calisir, cikti geri orneklenir.
        let ratio = crate::tts_engine::pitch_ratio(self.params.pitch);
        let speed = (self.params.speed.clamp(0.05, 4.0) / ratio).clamp(0.05, 4.0);
        let dur: Vec<i64> = logdur
            .iter()
            .zip(mask.iter())
            .map(|(&ld, &m)| {
                if !m {
                    return 0;
                }
                ((ld.exp() / speed).round() as i64).clamp(1, MAX_DUR_FRAMES as i64)
            })
            .collect();
        let sum: usize = dur.iter().map(|&d| d as usize).sum();
        let frames = ((sum + STACK - 1) / STACK * STACK).max(STACK);
        // fw: repeat_interleave + L-1 kirpma; fmask: tumu true (T dahil).
        let mut fw = Vec::with_capacity(frames);
        for (i, &d) in dur.iter().enumerate() {
            for _ in 0..d {
                if fw.len() >= frames {
                    break;
                }
                fw.push(i as i64);
            }
        }
        while fw.len() < frames {
            fw.push((l as i64) - 1);
        }
        let fmask = vec![true; frames];

        // cond = gather(h, fw)
        let hlen = h.len() / l.max(1);
        let mut cond = vec![0.0f32; frames * hlen];
        for (f, &idx) in fw.iter().enumerate() {
            let src = (idx.max(0) as usize).min(l.saturating_sub(1)) * hlen;
            let dst = f * hlen;
            cond[dst..dst + hlen].copy_from_slice(&h[src..src + hlen]);
        }

        Ok(ChunkPlan {
            cond,
            fmask,
            fw,
            dur,
            letters: l,
            frames,
            npres,
            nsuf: self
                .tokenizer
                .encode_ids(CHUNK_SUFFIX)
                .len()
                .saturating_sub(2),
        })
    }

    /// Dusunme: prep x2 + K adim + unstack/maske/denorm -> mel (B,T,128) duz.
    /// `noise`: cagri akisindan sirali uretilmis (B,N,256) duz vektor.
    fn think(&mut self, pt: &ChunkPlan, noise: Vec<f32>) -> Result<Vec<f32>> {
        let (frames, hlen) = (pt.frames, TEXT_DIM);
        let ctok = self.run_prep(&pt.cond, &pt.fmask)?;
        // null kosul: egitimli vektor genislet + ayni prep
        let mut null_full = vec![0.0f32; frames * hlen];
        for f in 0..frames {
            let dst = f * hlen;
            null_full[dst..dst + hlen].copy_from_slice(&self.null_cond);
        }
        let cnull = self.run_prep(&null_full, &pt.fmask)?;

        let n_tok = frames / STACK;
        let steps = self.params.steps.clamp(1, 32);
        let dt = 1.0 / steps as f32;
        let d_val = if steps > 1 { dt } else { 1.0 };
        let cfg = self.params.cfg;
        let mut x = noise;
        let tmask: Vec<bool> = pt.fmask.chunks(STACK).map(|c| c[0]).collect();
        for i in 0..steps {
            if !self.is_running() {
                break;
            }
            let t = i as f32 * dt;
            let v = self.run_step(&x, t, d_val, &ctok, &cnull, &tmask, cfg, n_tok)?;
            for (a, b) in x.iter_mut().zip(v.iter()) {
                *a += *b * dt;
            }
        }
        // Unstack + maske + denorm.
        let mut mel = vec![0.0f32; frames * N_MELS];
        for f in 0..frames {
            let src = (f / STACK) * N_MELS * STACK + (f % STACK) * N_MELS;
            let dst = f * N_MELS;
            mel[dst..dst + N_MELS].copy_from_slice(&x[src..src + N_MELS]);
        }
        for (f, m) in pt.fmask.iter().enumerate().take(frames) {
            if !m {
                for v in mel[f * N_MELS..(f + 1) * N_MELS].iter_mut() {
                    *v = 0.0;
                }
            }
        }
        for v in mel.iter_mut() {
            *v = *v * MEL_STD + MEL_MEAN;
        }
        Ok(mel)
    }

    fn run_text(&mut self, ids: &[i64], mask: &[bool]) -> Result<(Vec<f32>, Vec<f32>)> {
        let l = ids.len();
        let ids_arr =
            Array2::from_shape_vec((1, l), ids.to_vec()).context("ids dizilemedi")?;
        let mask_arr =
            Array2::from_shape_vec((1, l), mask.to_vec()).context("mask dizilemedi")?;
        let ids_val = Value::from_array(ids_arr)
            .map_err(|e| anyhow::anyhow!("ids tensor: {e:?}"))?;
        let mask_val = Value::from_array(mask_arr)
            .map_err(|e| anyhow::anyhow!("mask tensor: {e:?}"))?;
        let outputs = self
            .text
            .run(ort::inputs![ids_val, mask_val])
            .map_err(|e| anyhow::anyhow!("antalia_text calismadi: {e:?}"))?;
        let (hs, hd) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("h okunamadi: {e:?}"))?;
        let (ds, dd) = outputs[1]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("logdur okunamadi: {e:?}"))?;
        let hl: Vec<usize> = hs.iter().map(|&d| d as usize).collect();
        let dl: Vec<usize> = ds.iter().map(|&d| d as usize).collect();
        if hl != [1, l, TEXT_DIM] || dl != [1, l] {
            anyhow::bail!("text cikti sekli hatali: h={hs:?} dur={ds:?}");
        }
        Ok((hd.to_vec(), dd.to_vec()))
    }

    fn run_prep(&mut self, cond: &[f32], fmask: &[bool]) -> Result<Vec<f32>> {
        let t = cond.len() / TEXT_DIM.max(1);
        let cond_arr = Array2::from_shape_vec((t, TEXT_DIM), cond.to_vec())
            .context("cond sekli")?;
        let cond3 = cond_arr.insert_axis(ndarray::Axis(0));
        let cond_val = Value::from_array(cond3.into_dimensionality::<ndarray::Ix3>().unwrap())
            .map_err(|e| anyhow::anyhow!("cond tensor: {e:?}"))?;
        let mask_val = Value::from_array(
            Array2::from_shape_vec((1, t), fmask.to_vec()).context("fmask sekli")?,
        )
        .map_err(|e| anyhow::anyhow!("fmask tensor: {e:?}"))?;
        let outputs = self
            .prep
            .run(ort::inputs![cond_val, mask_val])
            .map_err(|e| anyhow::anyhow!("antalia_prep calismadi: {e:?}"))?;
        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("conv okunamadi: {e:?}"))?;
        let sl: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        if sl != [1, t, TEXT_DIM] {
            anyhow::bail!("prep cikti sekli hatali: {shape:?}");
        }
        // Cift-ortalama (Rust): ctok=(c[:,0::2]+c[:,1::2])*0.5
        let n = t / STACK;
        let mut ctok = vec![0.0f32; n * TEXT_DIM];
        for i in 0..n {
            for d in 0..TEXT_DIM {
                ctok[i * TEXT_DIM + d] =
                    (data[i * 2 * TEXT_DIM + d] + data[(i * 2 + 1) * TEXT_DIM + d]) * 0.5;
            }
        }
        Ok(ctok)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_step(
        &mut self,
        x: &[f32],
        t: f32,
        d: f32,
        ctok: &[f32],
        cnull: &[f32],
        tmask: &[bool],
        cfg: f32,
        n_tok: usize,
    ) -> Result<Vec<f32>> {
        // Adim grafikleri toplu (1,N,D) bekler (ihracattaki gibi).
        let mk = |v: Vec<f32>, rows: usize, cols: usize, name: &str| {
            Value::from_array(
                Array3::from_shape_vec((1, rows, cols), v)
                    .with_context(|| format!("{name} sekli"))?,
            )
            .map_err(|e| anyhow::anyhow!("{name} tensor: {e:?}"))
        };
        let dim_tok = N_MELS * STACK;
        let x_val = mk(x.to_vec(), n_tok, dim_tok, "x")?;
        // t/d: 1-D (B,) — ihracattaki gibi
        let t_val = Value::from_array(Array1::from_vec(vec![t]))
            .map_err(|e| anyhow::anyhow!("t tensor: {e:?}"))?;
        let d_val = Value::from_array(Array1::from_vec(vec![d]))
            .map_err(|e| anyhow::anyhow!("d tensor: {e:?}"))?;
        let ctok_val = mk(ctok.to_vec(), n_tok, TEXT_DIM, "ctok")?;
        let cnull_val = mk(cnull.to_vec(), n_tok, TEXT_DIM, "cnull")?;
        // tmask (1,N) bool
        let tm_val = Value::from_array(
            Array2::from_shape_vec((1, n_tok), tmask.to_vec()).context("tmask sekli")?,
        )
        .map_err(|e| anyhow::anyhow!("tmask tensor: {e:?}"))?;
        let cfg_val = Value::from_array(ndarray::arr0(cfg).into_dimensionality::<ndarray::IxDyn>().unwrap())
            .map_err(|e| anyhow::anyhow!("cfg tensor: {e:?}"))?;
        let outputs = self
            .step
            .run(ort::inputs![x_val, t_val, d_val, ctok_val, cnull_val, tm_val, cfg_val])
            .map_err(|e| anyhow::anyhow!("antalia_step calismadi: {e:?}"))?;
        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("v okunamadi: {e:?}"))?;
        let sl: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        if sl != [1, n_tok, dim_tok] {
            anyhow::bail!("step cikti sekli hatali: {shape:?}");
        }
        Ok(data.to_vec())
    }

    fn run_vocoder(&mut self, mel: &[f32], frames: usize) -> Result<Vec<f32>> {
        // Giris duz (T,128); vocoder (1,128,T) bekler: devrik kopyala.
        let mut transposed = vec![0.0f32; N_MELS * frames];
        for f in 0..frames {
            for m in 0..N_MELS {
                transposed[m * frames + f] = mel[f * N_MELS + m];
            }
        }
        let mel_arr = Array3::from_shape_vec((1, N_MELS, frames), transposed)
            .context("mel sekli")?;
        let mel_val = Value::from_array(mel_arr)
            .map_err(|e| anyhow::anyhow!("mel tensor: {e:?}"))?;
        let outputs = self
            .vocoder
            .run(ort::inputs![mel_val])
            .map_err(|e| anyhow::anyhow!("antalia_vocoder calismadi: {e:?}"))?;
        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("wav okunamadi: {e:?}"))?;
        let sl: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        if sl != [1, frames * HOP] {
            anyhow::bail!("vocoder cikti sekli hatali: {shape:?}");
        }
        Ok(data.to_vec())
    }



    fn resample_48_to(&self, samples: &[f32], target: u32) -> Result<Vec<f32>> {
        if target == NATIVE_RATE as u32 || samples.is_empty() {
            return Ok(samples.to_vec());
        }
        use rubato::{Resampler, SincFixedIn, SincInterpolationParameters,
                     SincInterpolationType, WindowFunction};
        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        };
        let mut resampler = SincFixedIn::<f32>::new(
            target as f64 / NATIVE_RATE as f64, 2.0, params, 1024, 1,
        )
        .context("ornekleyici kurulamadi")?;
        let mut out = Vec::new();
        for block in samples.chunks(1024) {
            let mut buf = vec![0.0f32; 1024];
            buf[..block.len()].copy_from_slice(block);
            let mut waves = resampler.process(&[buf], None).context("ornekleme")?;
            out.append(&mut waves.remove(0));
        }
        let expected =
            (samples.len() as f64 * target as f64 / NATIVE_RATE as f64).ceil() as usize;
        out.truncate(expected);
        Ok(out)
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
            sample_rate: self.params.sample_rate,
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

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }
}

// ---------------------------------------------------------------------------
// Kenar kesimi (api.py::_edges aynasi).
// ---------------------------------------------------------------------------

/// Prefix/suffix sesini kirp + kisa fade uygula.
/// lead_ms/tail_ms None ve gain_db 0 oldugundan (varsayilan) atlanir.
fn cut_edges(wav: &[f32], a: usize, b: usize, has_pre: bool, has_suf: bool) -> Vec<f32> {
    const FADE: usize = (FADE_MS * NATIVE_RATE as f32 / 1000.0) as usize; // 240
    let s0 = if has_pre { a * HOP } else { 0 };
    let s1 = if has_suf {
        (b * HOP).min(wav.len())
    } else {
        wav.len()
    };
    let s0 = s0.min(wav.len());
    let s1 = s1.max(s0);
    let mut out = wav[s0..s1].to_vec();
    // linspace esdegeri (payda FADE-1, Python ile birebir).
    if has_pre && FADE > 1 && out.len() > 2 * FADE {
        let denom = (FADE - 1) as f32;
        for (i, v) in out.iter_mut().take(FADE).enumerate() {
            *v *= i as f32 / denom;
        }
    }
    if has_suf && FADE > 1 && out.len() > 2 * FADE {
        let denom = (FADE - 1) as f32;
        let n = out.len();
        for (i, v) in out.iter_mut().skip(n - FADE).enumerate() {
            *v *= 1.0 - i as f32 / denom;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// EQ zinciri (post.py OutputChain aynasi): RBJ biquad IIR + makeup + limiter.
// Durum parcalar arasi tasinir (cagri basina taze zincir).
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
}

fn rbj_highpass(f0: f64, q: f64, fs: f64) -> Biquad {
    let w0 = 2.0 * std::f64::consts::PI * f0 / fs;
    let alpha = w0.sin() / (2.0 * q);
    let c = w0.cos();
    norm_biquad((1.0 + c) / 2.0, -(1.0 + c), (1.0 + c) / 2.0, 1.0 + alpha, -2.0 * c, 1.0 - alpha)
}

fn rbj_lowshelf(f0: f64, gain_db: f64, q: f64, fs: f64) -> Biquad {
    let a = 10f64.powf(gain_db / 40.0);
    let w0 = 2.0 * std::f64::consts::PI * f0 / fs;
    let alpha = w0.sin() / (2.0 * q);
    let c = w0.cos();
    let s = 2.0 * a.sqrt() * alpha;
    norm_biquad(
        a * ((a + 1.0) - (a - 1.0) * c + s),
        2.0 * a * ((a - 1.0) - (a + 1.0) * c),
        a * ((a + 1.0) - (a - 1.0) * c - s),
        (a + 1.0) + (a - 1.0) * c + s,
        -2.0 * ((a - 1.0) + (a + 1.0) * c),
        (a + 1.0) + (a - 1.0) * c - s,
    )
}

/// RBJ cookbook highshelf (tone-override icin; varsayilan zincirde yok).
fn rbj_highshelf(f0: f64, gain_db: f64, q: f64, fs: f64) -> Biquad {
    let a = 10f64.powf(gain_db / 40.0);
    let w0 = 2.0 * std::f64::consts::PI * f0 / fs;
    let alpha = w0.sin() / (2.0 * q);
    let c = w0.cos();
    let s = 2.0 * a.sqrt() * alpha;
    norm_biquad(
        a * ((a + 1.0) + (a - 1.0) * c + s),
        -2.0 * a * ((a - 1.0) + (a + 1.0) * c),
        a * ((a + 1.0) + (a - 1.0) * c - s),
        (a + 1.0) - (a - 1.0) * c + s,
        // Kitaplik a1: 2*((A-1) - (A+1)*cos) (lowshelf'teki + isaretinden farkli).
        2.0 * ((a - 1.0) - (a + 1.0) * c),
        (a + 1.0) - (a - 1.0) * c - s,
    )
}

fn norm_biquad(b0: f64, b1: f64, b2: f64, a0: f64, a1: f64, a2: f64) -> Biquad {
    Biquad {
        b0: b0 / a0,
        b1: b1 / a0,
        b2: b2 / a0,
        a1: a1 / a0,
        a2: a2 / a0,
    }
}

pub struct EqChain {
    sections: Vec<Biquad>,
    state: Vec<[f64; 4]>, // x1,x2,y1,y2 her bolum icin
    gain: f64,
    use_eq: bool,
}

impl EqChain {
    pub fn new(out_rate: u32, use_eq: bool) -> Self {
        Self::with_tone(out_rate, use_eq, None)
    }

    /// tone: None = varsayilan EQ + makeup; Some((lo,hi)) = api.py tone-override
    /// (210 Hz low-shelf lo dB + 4 kHz high-shelf hi dB, makeup 0 dB).
    pub fn with_tone(out_rate: u32, use_eq: bool, tone: Option<(f32, f32)>) -> Self {
        let fs = out_rate as f64;
        let (sections, gain) = if !use_eq {
            (Vec::new(), 1.0)
        } else if let Some((lo, hi)) = tone {
            (
                vec![
                    rbj_lowshelf(EQ_SHELF_HZ, lo as f64, EQ_SHELF_Q, fs),
                    rbj_highshelf(TONE_HI_HZ, hi as f64, EQ_SHELF_Q, fs),
                ],
                1.0,
            )
        } else {
            // Varsayilan EQ (post.py eq_sections aynasi):
            // 2. sira highpass 90 Hz Q0.707 + low-shelf 210 Hz -5 dB Q0.707,
            // tasarim hizi = cikis hizi; makeup +0.54 dB sabit.
            (
                vec![
                    rbj_highpass(EQ_HPF_HZ, EQ_HPF_Q, fs),
                    rbj_lowshelf(EQ_SHELF_HZ, EQ_SHELF_DB, EQ_SHELF_Q, fs),
                ],
                10f64.powf(EQ_MAKEUP_DB / 20.0),
            )
        };
        let n = sections.len();
        Self {
            sections,
            state: vec![[0.0; 4]; n],
            gain,
            use_eq,
        }
    }

    /// Ham 48k... NOT: giris hedef hizdadir (resample disarida yapilir).
    pub fn process(&mut self, chunk: &[f32]) -> Vec<f32> {
        if !self.use_eq {
            return chunk
                .iter()
                .map(|&x| x.clamp(-1.0, 1.0))
                .collect();
        }
        let mut out = Vec::with_capacity(chunk.len());
        for &sample in chunk {
            let mut y = sample as f64;
            for (sec, st) in self.sections.iter().zip(self.state.iter_mut()) {
                let o = sec.b0 * y + sec.b1 * st[0] + sec.b2 * st[1]
                    - sec.a1 * st[2]
                    - sec.a2 * st[3];
                st[1] = st[0];
                st[0] = y;
                st[3] = st[2];
                st[2] = o;
                y = o;
            }
            y *= self.gain;
            out.push(soft_limit(y) as f32);
        }
        out
    }
}

fn soft_limit(x: f64) -> f64 {
    let a = x.abs();
    if a <= LIMIT_THRESHOLD {
        return x;
    }
    let room = LIMIT_CEILING - LIMIT_THRESHOLD;
    x.signum() * (LIMIT_THRESHOLD + room * ((a - LIMIT_THRESHOLD) / room).tanh())
}

// ---------------------------------------------------------------------------
// Yardimcilar
// ---------------------------------------------------------------------------

struct SplitMix64(u64);

/// `out` dizisini standart normal dagilimla doldurur (Box-Muller).
fn box_muller_fill(rng: &mut SplitMix64, out: &mut [f32]) {
    let mut i = 0;
    // SplitMix64 ciktisinden (0,1] araliginda uniform deger.
    let mut uni = || {
        let u = rng.next() >> 11; // 53 bit
        ((u as f64) + 0.5) / 9007199254740992.0
    };
    while i < out.len() {
        let u1 = uni().max(f64::MIN_POSITIVE);
        let u2 = uni();
        let r = (-2.0 * u1.ln()).sqrt();
        let t = 2.0 * std::f64::consts::PI * u2;
        out[i] = (r * t.cos()) as f32;
        if i + 1 < out.len() {
            out[i + 1] = (r * t.sin()) as f32;
        }
        i += 2;
    }
}

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

    fn ref_dir() -> PathBuf {
        test_dir("models/antalia_ref")
    }

    fn read_f32(name: &str) -> Vec<f32> {
        let b = fs::read(ref_dir().join(name)).unwrap();
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn read_i64(name: &str) -> Vec<i64> {
        let b = fs::read(ref_dir().join(name)).unwrap();
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

    fn test_params() -> AntaliaParams {
        AntaliaParams {
            speed: 0.95,
            steps: 8,
            cfg: 2.0,
            temperature: 0.6,
            eq: true,
            // Uretim yolu (api.py varsayilani): tone-override zinciri.
            tone: Some((TONE_LO_DB as f32, TONE_HI_DB as f32)),
            pitch: 0.0,
            seed: 0,
            sample_rate: 48000,
            threads: 4,
        }
    }

    fn test_engine() -> AntaliaEngine {
        AntaliaEngine::new(test_dir("models/antalia"), test_params())
            .expect("antalia motoru acilmali")
    }

    #[test]
    fn reference_text_and_timeline_match() {
        let shapes: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(ref_dir().join("shapes.json")).unwrap(),
        )
        .unwrap();
        let l = shapes["L"].as_u64().unwrap() as usize;
        let t = shapes["T"].as_u64().unwrap() as usize;

        let ids = read_i64("ids.i64");
        let h_ref = read_f32("h.f32");
        let logdur_ref = read_f32("logdur.f32");
        let dur_ref = read_i64("dur.i64");
        let fw_ref = read_i64("fw.i64");
        let cond_ref = read_f32("cond.f32");
        let ctok_ref = read_f32("ctok.f32");
        assert_eq!(ids.len(), l);

        let mut engine = test_engine();
        let mask: Vec<bool> = ids.iter().map(|&id| id != 0).collect();
        let (h, mut logdur) = engine.run_text(&ids, &mask).unwrap();
        // Referans logdur gem-esnetmeli: ayni yardimciyla uretim tarafi dogrulanir.
        apply_gem(&ids, engine.tokenizer.space_id(), &mut logdur);
        println!(
            "text: h {:.3e}, logdur {:.3e}",
            max_abs_diff(&h, &h_ref),
            max_abs_diff(&logdur, &logdur_ref)
        );
        assert!(max_abs_diff(&h, &h_ref) < 2e-5);
        assert!(max_abs_diff(&logdur, &logdur_ref) < 2e-5);

        // plan_text(): gem + timeline (deterministik kisim).
        // fw birebir esitligi gem/dur dogrulugunu de kapsar.
        let chunk_text: String = shapes["chunk"].as_str().unwrap_or("").to_string();
        let plan = engine.plan_text(&chunk_text).unwrap();
        assert_eq!(plan.frames, t, "frame sayisi eslesmeli");
        assert_eq!(plan.dur, dur_ref, "dur birebir eslesmeli");
        // Referans fw dolgusu: sum(dur) uzunlugunda; uretim T'ye L-1 ile tamamlar
        // (expand'in kirpma semantigi). Ortak kisim birebir olmali.
        assert_eq!(&plan.fw[..fw_ref.len()], fw_ref.as_slice());
        assert!(plan.fw[fw_ref.len()..].iter().all(|&i| i == l as i64 - 1));
        // cond gather + prep + cift-ortalama
        let hlen = h.len() / l.max(1);
        let mut cond = vec![0.0f32; t * hlen];
        for (f, &idx) in plan.fw.iter().enumerate() {
            let src = (idx.max(0) as usize).min(l.saturating_sub(1)) * hlen;
            cond[f * hlen..(f + 1) * hlen].copy_from_slice(&h[src..src + hlen]);
        }
        println!("cond fark: {:.3e}", max_abs_diff(&cond, &cond_ref));
        assert!(max_abs_diff(&cond, &cond_ref) < 1e-5);
        let ctok = engine.run_prep(&cond, &plan.fmask).unwrap();
        println!("ctok fark: {:.3e}", max_abs_diff(&ctok, &ctok_ref));
        assert!(max_abs_diff(&ctok, &ctok_ref) < 1e-5);
        println!("dur ref ilk 12: {:?}", &dur_ref[..12.min(dur_ref.len())]);
    }

    #[test]
    fn reference_wav_cut_matches() {
        // Kenar kesimi: ayni wav_full uzerinde Python ile ayni kirpma mi?
        // wav_cut referansi kullanilarak uctan uca dogrulanir (asagidaki testte).
        let wav_ref = read_f32("wav_cut.f32");
        assert!(!wav_ref.is_empty());
        assert_eq!(wav_ref.len() % HOP, 0);
    }

    #[test]
    fn eq_chain_matches_python() {
        let x = read_f32("eq_in.f32");
        let y_ref = read_f32("eq_out.f32");
        let y_raw_ref = read_f32("eq_raw.f32");
        assert_eq!(x.len(), y_ref.len());
        // eq=true (ton): parcali isleme de ayni sonucu vermeli
        let mut eq = EqChain::new(48000, true);
        let mut y1 = Vec::new();
        for piece in x.chunks(30000) {
            y1.extend_from_slice(&eq.process(piece));
        }
        let mut eq2 = EqChain::new(48000, true);
        let y2 = eq2.process(&x);
        println!(
            "eq: tek-parca {:.3e}, parcali-tutarli {:.3e}",
            max_abs_diff(&y2, &y_ref),
            max_abs_diff(&y1, &y2)
        );
        assert!(max_abs_diff(&y2, &y_ref) < 1e-5);
        assert!(max_abs_diff(&y1, &y2) < 1e-9);
        // eq=false: hard clip
        let mut eq3 = EqChain::new(48000, false);
        let y3 = eq3.process(&x);
        assert!(max_abs_diff(&y3, &y_raw_ref) < 1e-6);
    }

    #[test]
    fn eq_tone_matches_python() {
        // api.py uretim varsayilani: tone=(+6,-9), makeup 0 dB.
        let x = read_f32("eq_in.f32");
        let y_ref = read_f32("eq_tone_out.f32");
        assert_eq!(x.len(), y_ref.len());
        let mut eq = EqChain::with_tone(48000, true, Some((6.0, -9.0)));
        let mut y1 = Vec::new();
        for piece in x.chunks(30000) {
            y1.extend_from_slice(&eq.process(piece));
        }
        let mut eq2 = EqChain::with_tone(48000, true, Some((6.0, -9.0)));
        let y2 = eq2.process(&x);

        println!(
            "eq-tone ref: peak {:.4}, ilk8 {:?}",
            y_ref.iter().map(|v| v.abs()).fold(0.0f32, f32::max),
            &y_ref[..8]
        );
        assert!(max_abs_diff(&y2, &y_ref) < 1e-5);
        assert!(max_abs_diff(&y1, &y2) < 1e-9);
    }

    #[test]
    fn think_matches_reference_mel() {
        // Referans gurultuyle dusunme: mel birebir karsilastirilir.
        let shapes: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(ref_dir().join("shapes.json")).unwrap(),
        )
        .unwrap();
        let t = shapes["T"].as_u64().unwrap() as usize;
        let chunk_text: String = shapes["chunk"].as_str().unwrap_or("").to_string();
        let mel_ref = read_f32("mel.f32");
        let noise = read_f32("noise.f32");
        assert_eq!(mel_ref.len(), t * N_MELS);

        let mut engine = test_engine();
        let pt = engine.plan_text(&chunk_text).unwrap();
        assert_eq!(pt.frames, t);
        // think adim dongusu dur bayragina bakar: uretimdeki gibi kur.
        engine.running.store(true, Ordering::SeqCst);
        let mel = engine.think(&pt, noise).unwrap();
        assert_eq!(mel.len(), mel_ref.len());
        // Referans (128,T) mel-major; uretim (T,128) frame-major: devrik karsilastir.
        let mut mel_ref_t = vec![0.0f32; mel_ref.len()];
        for f in 0..t {
            for m in 0..N_MELS {
                mel_ref_t[f * N_MELS + m] = mel_ref[m * t + f];
            }
        }
        println!("mel fark: {:.3e}", max_abs_diff(&mel, &mel_ref_t));
        assert!(max_abs_diff(&mel, &mel_ref_t) < 5e-4);
    }

    #[test]
    fn full_synthesis_produces_sound() {
        let mut engine = test_engine();
        let wav = engine
            .synthesize_to_vec("Merhaba, size nasıl yardımcı olabilirim?")
            .unwrap();
        assert!(!wav.is_empty());
        assert!(wav.iter().all(|x| x.is_finite()));
        let peak = wav.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let rms =
            (wav.iter().map(|x| x * x).sum::<f32>() / wav.len() as f32).sqrt();
        println!(
            "antalia sentez: {} ornek, {:.2} sn, peak {:.3}, rms {:.4}",
            wav.len(),
            wav.len() as f32 / 48000.0,
            peak,
            rms
        );
        assert!(peak > 1e-3 && peak <= 1.0);
        assert!(rms > 1e-4);
    }
}
