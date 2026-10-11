// Hizli arastirma araci: verilen metni sentezler, WAV yazar, istatistik basar.
// Kullanim: cargo run --example synth -- "METIN" out.wav [speed] [pitch] [seed]
use ema_lightning_tts_lib::tts_engine::{TtsConfig, TtsEngine};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("kullanim: synth -- \"METIN\" out.wav [speed] [pitch] [seed]");
        std::process::exit(2);
    }
    let text = args[1].clone();
    let out = args[2].clone();
    let speed: f32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1.0);
    let pitch: f32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let seed: u64 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(0);

    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let model_dir = manifest.join("..").join("models").join("onnx");
    let cfg = TtsConfig {
        speed,
        pitch,
        seed,
        ..Default::default()
    };
    let mut engine = TtsEngine::new(&model_dir, cfg).expect("motor acilmadi");
    let wav = engine.synthesize_to_vec(&text).expect("sentez olmadi");
    engine.save_wav(&wav, &out).expect("wav yazilmadi");

    let peak = wav.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let mut max_jump = 0.0f32;
    let mut max_idx = 0usize;
    for (i, p) in wav.windows(2).enumerate() {
        let j = (p[1] - p[0]).abs();
        if j > max_jump {
            max_jump = j;
            max_idx = i;
        }
    }
    let tail: f32 =
        wav[wav.len().saturating_sub(4800)..].iter().map(|x| x.abs()).sum::<f32>() / 4800.0;
    println!(
        "ornek={} sn={:.2} peak={:.3} max_jump={:.4} @ornek {} ({:.3}sn) tail100ms={:.4}",
        wav.len(),
        wav.len() as f32 / 48000.0,
        peak,
        max_jump,
        max_idx,
        max_idx as f32 / 48000.0,
        tail
    );
}

use std::path::PathBuf;
