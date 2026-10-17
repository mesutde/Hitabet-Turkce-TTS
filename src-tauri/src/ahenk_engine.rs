use anyhow::{Context, Result};
use ndarray::{Array1, Array2, Array3};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Value;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::tts_engine::{
    self, CONTEXT, DECODE_RATE, FIRST_WINDOW, HOP, LATENT_DIM, MAX_FRAMES, MAX_WORD_FRAMES,
    Streamer, TEXT_DIM, TtsConfig, WINDOW, seeded_gaussian,
};
use crate::turkish_tokenizer::{TurkishTokenizer, chunk_spoken};

// Ahenk Cep (ozgurguler/ahenk-cep-tr): EMA Lightning mimarisi + stil gommesi.
// ONNX dort parcaya ayrilmistir (tarayici demosu): text -> h,dur; cond (aligner);
// step (TEK Euler adimi, CFG iceride); decoder (tek pencere). Euler dongusu Rust'ta.
pub const STYLES: [&str; 8] = [
    "neutral",
    "angry",
    "calm",
    "cheerful",
    "empathetic",
    "sad",
    "surprised",
    "laughter",
];

pub const DEFAULT_STEPS: usize = 16;
pub const DEFAULT_CFG: f32 = 2.0;

/// Stil adini (Turkce ya da Ingilizce, buyuk/kucuk duyarsiz) indekse cevirir.
/// Bilinmeyen ad -> neutral (0). "gulme" egitilmedi ama listede durur.
pub fn style_index(name: &str) -> usize {
    let n = name.trim().to_lowercase();
    let tr = [
        ("notr", "neutral"),
        ("kizgin", "angry"),
        ("sakin", "calm"),
        ("neseli", "cheerful"),
        ("empatik", "empathetic"),
        ("uzgun", "sad"),
        ("saskin", "surprised"),
        ("gulme", "laughter"),
    ];
    let key = tr
        .iter()
        .find(|(t, _)| *t == n)
        .map(|(_, e)| *e)
        .unwrap_or(n.as_str());
    STYLES.iter().position(|s| *s == key).unwrap_or(0)
}

fn ort_err<E: std::fmt::Debug>(e: E) -> anyhow::Error {
    anyhow::anyhow!("ORT error: {e:?}")
}

pub struct AhenkEngine {
    text: Session,
    cond: Session,
    step: Session,
    decoder: Session,
    tokenizer: TurkishTokenizer,
    config: TtsConfig,
    model_dir: PathBuf,
    running: Arc<AtomicBool>,
    stop_requested: Arc<AtomicBool>,
}

impl AhenkEngine {
    pub fn new(model_dir: impl AsRef<Path>, config: TtsConfig) -> Result<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();
        let text = Self::load_session(&model_dir.join("text.onnx"), config.threads)?;
        let cond = Self::load_session(&model_dir.join("cond.onnx"), config.threads)?;
        let step = Self::load_session(&model_dir.join("step.onnx"), config.threads)?;
        let decoder = Self::load_session(&model_dir.join("decoder.onnx"), config.threads)?;
        let tokenizer = TurkishTokenizer::from_vocab_file(model_dir.join("vocab.json"))?;
        Ok(Self {
            text,
            cond,
            step,
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
        builder
            .commit_from_file(path)
            .map_err(|e| anyhow::anyhow!("Failed to load ONNX model {}: {e:?}", path.display()))
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

    pub fn running_handle(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.running)
    }

    pub fn stop_handle(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.stop_requested)
    }

    pub fn config(&self) -> &TtsConfig {
        &self.config
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
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
        let limit_speed = self.config.speed.clamp(0.25, 4.0);
        let pieces = chunk_spoken(&spoken, limit_speed);
        if pieces.is_empty() {
            anyhow::bail!("Metin bos: sentezlenecek harf yok");
        }

        let style = style_index(&self.config.style) as i64;
        let steps = self.config.steps.clamp(1, 64);
        let cfg = self.config.cfg.clamp(0.1, 8.0);

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
            let seed_i = self
                .config
                .seed
                .wrapping_mul(1_000_003)
                .wrapping_add(i as u64)
                & 0x7FFF_FFFF_FFFF_FFFF;
            let plan = self.plan(&ids, seed_i, style)?;
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

        let totals: Vec<usize> = planned
            .iter()
            .map(|p| tts_engine::window_spans(p.plan.frames, p.first_window).len())
            .collect();
        let grand_total: usize = totals.iter().sum();
        let mut streamer = Streamer::new(&self.config);
        let mut base = 0usize;
        for item in planned.iter() {
            if !self.is_running() {
                break;
            }
            let cond = self.run_cond(&item.plan)?;
            let latents = self.run_euler(&item.plan.noise, &cond, item.plan.frames, steps, cfg)?;
            let spans = tts_engine::window_spans(item.plan.frames, item.first_window);
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

    /// engine.py::plan aynasi: text stage + harf id'lerinden frame timeline.
    fn plan(&mut self, ids: &[i64], seed: u64, style: i64) -> Result<FramePlan> {
        let l = ids.len();
        let (h, dur) = self.run_text(ids, style)?;

        let chars_len = l;
        let mut starts: Vec<usize> = Vec::new();
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

        let ratio = tts_engine::pitch_ratio(self.config.pitch);
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

        let mut fw = Vec::with_capacity(frames);
        let mut fp = Vec::with_capacity(frames);
        let mut starts_cum = vec![0usize; n_words + 1];
        for w in 0..n_words {
            starts_cum[w + 1] = starts_cum[w] + word_frames[w];
        }
        for f in 0..frames {
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

        // Tek Euler adimi icin gurultu (T,64): adim=1.
        let noise = seeded_gaussian(seed, 1, frames, LATENT_DIM);

        Ok(FramePlan {
            h,
            dur: dur.iter().map(|&d| d / speed).collect(),
            cw,
            wstart,
            fw,
            fp,
            noise,
            frames,
        })
    }

    /// text.onnx: (ids[1,L], style[1]) -> (h[1,L,224], dur[1,L]).
    fn run_text(&mut self, ids: &[i64], style: i64) -> Result<(Vec<f32>, Vec<f32>)> {
        let l = ids.len();
        let ids_arr = Array2::from_shape_vec((1, l), ids.to_vec()).context("ids dizilemedi")?;
        let ids_val =
            Value::from_array(ids_arr).map_err(|e| anyhow::anyhow!("ids tensor: {e:?}"))?;
        let style_val = Value::from_array(Array1::from_vec(vec![style]))
            .map_err(|e| anyhow::anyhow!("style tensor: {e:?}"))?;
        let outputs = self
            .text
            .run(ort::inputs![ids_val, style_val])
            .map_err(|e| anyhow::anyhow!("text calismadi: {e:?}"))?;
        let (hs, hd) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("h okunamadi: {e:?}"))?;
        let (_ds, dd) = outputs[1]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("dur okunamadi: {e:?}"))?;
        let hl: Vec<usize> = hs.iter().map(|&d| d as usize).collect();
        if hl != [1, l, TEXT_DIM] {
            anyhow::bail!("text cikti sekli hatali: h={hs:?}");
        }
        Ok((hd.to_vec(), dd.to_vec()))
    }

    /// cond.onnx: (h,dur,cw,wstart,fw,fp) -> cond[1,T,224].
    fn run_cond(&mut self, plan: &FramePlan) -> Result<Vec<f32>> {
        let (l, t) = (plan.cw.len(), plan.frames);
        let mk_f = |v: Array2<f32>| {
            Value::from_array(v).map_err(|e| anyhow::anyhow!("tensor: {e:?}"))
        };
        let mk_i = |v: Array2<i64>| {
            Value::from_array(v).map_err(|e| anyhow::anyhow!("tensor: {e:?}"))
        };
        let h = Array2::from_shape_vec((l, TEXT_DIM), plan.h.clone()).context("h sekli")?;
        let h3 = h.insert_axis(ndarray::Axis(0));
        let h_val = Value::from_array(h3.into_dimensionality::<ndarray::Ix3>().unwrap())
            .map_err(|e| anyhow::anyhow!("h tensor: {e:?}"))?;
        let dur_val = mk_f(Array2::from_shape_vec((1, l), plan.dur.clone()).context("dur sekli")?)?;
        let cw_val = mk_i(Array2::from_shape_vec((1, l), plan.cw.clone()).context("cw sekli")?)?;
        let ws_val = mk_i(Array2::from_shape_vec((1, l), plan.wstart.clone()).context("wstart sekli")?)?;
        let fw_val = mk_i(Array2::from_shape_vec((1, t), plan.fw.clone()).context("fw sekli")?)?;
        let fp_val = mk_f(Array2::from_shape_vec((1, t), plan.fp.clone()).context("fp sekli")?)?;
        let outputs = self
            .cond
            .run(ort::inputs![h_val, dur_val, cw_val, ws_val, fw_val, fp_val])
            .map_err(|e| anyhow::anyhow!("cond calismadi: {e:?}"))?;
        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("cond okunamadi: {e:?}"))?;
        let sl: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        if sl != [1, t, TEXT_DIM] {
            anyhow::bail!("cond cikti sekli hatali: {shape:?}");
        }
        Ok(data.to_vec())
    }

    /// Euler dongusu: `steps` kez step.onnx (x,cond,t,dt,cfg) -> x.
    /// noise: (T,64) duz. Donus: latents (T,64) duz.
    fn run_euler(
        &mut self,
        noise: &[f32],
        cond: &[f32],
        frames: usize,
        steps: usize,
        cfg: f32,
    ) -> Result<Vec<f32>> {
        let t = frames;
        let x0 = Array3::from_shape_vec((1, t, LATENT_DIM), noise.to_vec())
            .context("noise sekli")?;
        let mut x = Value::from_array(x0)
            .map_err(|e| anyhow::anyhow!("x tensor: {e:?}"))?;
        // cond her adimda yeniden kurulur (Value `run`'a tasinir; ucuz: T*224).
        for k in 0..steps {
            let cond_arr = Array3::from_shape_vec((1, t, TEXT_DIM), cond.to_vec())
                .context("cond sekli")?;
            let cond_val = Value::from_array(cond_arr)
                .map_err(|e| anyhow::anyhow!("cond tensor: {e:?}"))?;
            let tv = Value::from_array(Array1::from_vec(vec![k as f32 / steps as f32]))
                .map_err(|e| anyhow::anyhow!("t tensor: {e:?}"))?;
            let dtv = Value::from_array(Array1::from_vec(vec![1.0 / steps as f32]))
                .map_err(|e| anyhow::anyhow!("dt tensor: {e:?}"))?;
            let cfgv = Value::from_array(Array1::from_vec(vec![cfg]))
                .map_err(|e| anyhow::anyhow!("cfg tensor: {e:?}"))?;
            let outputs = self
                .step
                .run(ort::inputs![x, cond_val, tv, dtv, cfgv])
                .map_err(|e| anyhow::anyhow!("step calismadi: {e:?}"))?;
            let (_, data) = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| anyhow::anyhow!("step ciktisi okunamadi: {e:?}"))?;
            let next = Array3::from_shape_vec((1, t, LATENT_DIM), data.to_vec())
                .context("step cikti sekli")?;
            x = Value::from_array(next)
                .map_err(|e| anyhow::anyhow!("x tensor: {e:?}"))?;
        }
        let (_, data) = x
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("latents okunamadi: {e:?}"))?;
        Ok(data.to_vec())
    }

    /// Pencereli decode (EMA ile ayni pencereler); Ahenk decoder'i yalniz z alir.
    fn decode_stream(
        &mut self,
        latents: &[f32],
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
            let mut z = vec![0f32; LATENT_DIM * n];
            for i in 0..n {
                for c in 0..LATENT_DIM {
                    z[c * n + i] = latents[(a + i) * LATENT_DIM + c];
                }
            }
            let z_arr = Array3::from_shape_vec((1, LATENT_DIM, n), z).context("z sekli")?;
            let z_val =
                Value::from_array(z_arr).map_err(|e| anyhow::anyhow!("z tensor: {e:?}"))?;
            let win: Vec<f32> = {
                let outputs = self
                    .decoder
                    .run(ort::inputs![z_val])
                    .map_err(|e| anyhow::anyhow!("decoder calismadi: {e:?}"))?;
                let (_, data) = outputs[0]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| anyhow::anyhow!("audio okunamadi: {e:?}"))?;
                let off = (s - a) * HOP;
                let len = (e - s) * HOP;
                data[off..off + len].to_vec()
            };
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
            &mut |chunk| samples.extend_from_slice(chunk),
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
        let mut writer = hound::WavWriter::create(path, spec).context("WAV dosyasi acilamadi")?;
        for &sample in samples {
            let clamped = sample.clamp(-1.0, 1.0);
            writer.write_sample((clamped * i16::MAX as f32) as i16)?;
        }
        writer.finalize().context("WAV yazilamadi")?;
        Ok(())
    }
}

struct FramePlan {
    h: Vec<f32>,        // (L,224) duz
    dur: Vec<f32>,      // (L) duz (zaten speed'e bolunmus)
    cw: Vec<i64>,
    wstart: Vec<i64>,
    fw: Vec<i64>,
    fp: Vec<f32>,
    noise: Vec<f32>, // (T,64) duz
    frames: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn test_dir(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join(name)
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
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect()
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn style_index_maps_names() {
        assert_eq!(style_index("neutral"), 0);
        assert_eq!(style_index("angry"), 1);
        assert_eq!(style_index("kizgin"), 1);
        assert_eq!(style_index("SAD"), 5);
        assert_eq!(style_index("neseli"), 3);
        assert_eq!(style_index("bilinmeyen"), 0);
    }

    #[test]
    fn reference_pipeline_matches_python() {
        let ref_dir = test_dir("models/ahenk_ref");
        let shapes: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(ref_dir.join("shapes.json")).unwrap())
                .unwrap();
        let l = shapes["L"].as_u64().unwrap() as usize;
        let t = shapes["T"].as_u64().unwrap() as usize;
        let style = shapes["style"].as_i64().unwrap();
        let steps = shapes["steps"].as_u64().unwrap() as usize;
        let cfg = shapes["cfg"].as_f64().unwrap() as f32;

        let ids = read_i64(&ref_dir.join("ids.i64"));
        let h_ref = read_f32(&ref_dir.join("h.f32"));
        let dur_ref = read_f32(&ref_dir.join("dur.f32"));
        let fw_ref = read_i64(&ref_dir.join("fw.i64"));
        let cond_ref = read_f32(&ref_dir.join("cond.f32"));
        let noise = read_f32(&ref_dir.join("noise.f32"));
        let lat_ref = read_f32(&ref_dir.join("latents.f32"));
        let audio_ref = read_f32(&ref_dir.join("audio_full.f32"));
        assert_eq!(ids.len(), l);

        let mut engine =
            AhenkEngine::new(test_dir("models/ahenk"), TtsConfig::default()).expect("motor acilmali");

        // Tokenizer, Python referansiyla ayni id'leri uretmelidir (L=34).
        let spoken = engine.tokenizer.preprocess("merhaba dünya bugün hava çok güzel");
        assert_eq!(
            engine.tokenizer.encode_ids(&spoken),
            ids,
            "tokenizer id'leri Python referansiyla ayni olmali"
        );

        let (h, dur) = engine.run_text(&ids, style).unwrap();
        println!(
            "text: h fark={:.3e} dur fark={:.3e}",
            max_abs_diff(&h, &h_ref),
            max_abs_diff(&dur, &dur_ref)
        );
        assert!(max_abs_diff(&h, &h_ref) < 2e-5);
        assert!(max_abs_diff(&dur, &dur_ref) < 2e-5);

        let plan = engine.plan(&ids, 0, style).unwrap();
        assert_eq!(plan.frames, t, "frame sayisi Python ile ayni olmali");
        assert_eq!(plan.fw, fw_ref, "fw birebir ayni olmali");

        let cond = engine.run_cond(&plan).unwrap();
        println!("cond fark={:.3e}", max_abs_diff(&cond, &cond_ref));
        assert!(max_abs_diff(&cond, &cond_ref) < 2e-5);

        let lat = engine.run_euler(&noise, &cond, t, steps, cfg).unwrap();
        println!("latents fark={:.3e}", max_abs_diff(&lat, &lat_ref));
        assert!(max_abs_diff(&lat, &lat_ref) < 2e-4);

        // Pencereli decode (Python da ayni pencerelerle uretti).
        let mut out: Vec<f32> = Vec::new();
        engine.running.store(true, Ordering::SeqCst);
        let spans = tts_engine::window_spans(t, FIRST_WINDOW);
        let mut streamer = Streamer::new(engine.config());
        engine
            .decode_stream(
                &lat,
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
        println!("decode: audio fark={:.3e}", max_abs_diff(&out, &audio_ref));
        assert!(max_abs_diff(&out, &audio_ref) < 5e-3);

        // Uc-tan-uca sentez calisiyor mu (gurultu Rust uretimi, bit-ayni degil).
        let wav = engine
            .synthesize_to_vec("Merhaba, size nasıl yardımcı olabilirim?")
            .unwrap();
        assert!(!wav.is_empty());
        assert!(wav.iter().all(|x| x.is_finite()));
        let peak = wav.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        println!("sentez: {} ornek, {:.2} sn, peak {:.3}", wav.len(), wav.len() as f32 / 48000.0, peak);
        assert!(peak > 1e-4 && peak <= 1.0);
    }

    #[test]
    fn styles_change_synthesis() {
        let mut engine =
            AhenkEngine::new(test_dir("models/ahenk"), TtsConfig::default()).expect("motor acilmali");
        let mut cfg = engine.config().clone();
        cfg.style = "angry".to_string();
        engine.update_config(cfg);
        let a = engine.synthesize_to_vec("Bu bir deneme cümlesi.").unwrap();
        let mut cfg = engine.config().clone();
        cfg.style = "sad".to_string();
        engine.update_config(cfg);
        let s = engine.synthesize_to_vec("Bu bir deneme cümlesi.").unwrap();
        assert!(a.iter().all(|x| x.is_finite()));
        assert!(s.iter().all(|x| x.is_finite()));
        assert!(a.iter().map(|x| x.abs()).fold(0.0f32, f32::max) > 1e-3);
        // Ayni metin, farkli stil: sinyaller ayni olmamali (stil gommesi isliyor).
        let differ = if a.len() == s.len() {
            max_abs_diff(&a, &s) > 1e-4
        } else {
            true
        };
        assert!(differ, "farkli stiller ayni sinyali verdi");
    }
}
