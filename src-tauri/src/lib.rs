pub mod tts_engine;
pub mod turkish_tokenizer;
pub mod antalia_text;
pub mod antalia_engine;
mod api_server;
mod audio_player;

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use tauri::{AppHandle, Emitter, Manager, State};
use tts_engine::{ModelId, TtsConfig, TtsEngine};

use antalia_engine::{AntaliaEngine, AntaliaParams};
use audio_player::AudioPlayer;

/// Calisabilir motor (iki modelden biri). Imzalar birebir oldugundan
/// UI + HTTP API tek elden konusur; config eslemesi `apply_config` icindedir.
pub(crate) enum Engine {
    Ema(TtsEngine),
    Antalia(AntaliaEngine),
}

impl Engine {
    pub(crate) fn model(&self) -> ModelId {
        match self {
            Engine::Ema(_) => ModelId::EmaLightning,
            Engine::Antalia(_) => ModelId::AntaliaMini,
        }
    }

    /// Paylasilan arayuz ayarini yuklu motora yazar (modele ozgu esleme).
    pub(crate) fn apply_config(&mut self, config: &TtsConfig) {
        match self {
            Engine::Ema(e) => e.update_config(config.clone()),
            Engine::Antalia(a) => a.update_params(antalia_params_from(config)),
        }
    }

    pub(crate) fn synthesize(
        &mut self,
        text: &str,
        chunk_callback: &mut dyn FnMut(&[f32]),
        progress_callback: &mut dyn FnMut(usize, usize),
    ) -> anyhow::Result<()> {
        match self {
            Engine::Ema(e) => e.synthesize(text, chunk_callback, progress_callback),
            Engine::Antalia(a) => a.synthesize(text, chunk_callback, progress_callback),
        }
    }

    pub(crate) fn synthesize_to_vec(&mut self, text: &str) -> anyhow::Result<Vec<f32>> {
        match self {
            Engine::Ema(e) => e.synthesize_to_vec(text),
            Engine::Antalia(a) => a.synthesize_to_vec(text),
        }
    }

    pub(crate) fn save_wav(
        &self,
        samples: &[f32],
        path: impl AsRef<std::path::Path>,
    ) -> anyhow::Result<()> {
        match self {
            Engine::Ema(e) => e.save_wav(samples, path),
            Engine::Antalia(a) => a.save_wav(samples, path),
        }
    }

    pub(crate) fn stop(&self) {
        match self {
            Engine::Ema(e) => e.stop(),
            Engine::Antalia(a) => a.stop(),
        }
    }

    pub(crate) fn running_handle(&self) -> Arc<AtomicBool> {
        match self {
            Engine::Ema(e) => e.running_handle(),
            Engine::Antalia(a) => a.running_handle(),
        }
    }

    pub(crate) fn stop_handle(&self) -> Arc<AtomicBool> {
        match self {
            Engine::Ema(e) => e.stop_handle(),
            Engine::Antalia(a) => a.stop_handle(),
        }
    }
}

/// Paylasilan ayardan Antalia parametresi uretir (uretim tone-override sabittir).
pub(crate) fn antalia_params_from(config: &TtsConfig) -> AntaliaParams {
    AntaliaParams {
        speed: config.speed,
        steps: config.steps.clamp(1, 32),
        cfg: config.cfg,
        temperature: config.temperature,
        eq: config.eq,
        tone: Some((
            antalia_engine::TONE_LO_DB as f32,
            antalia_engine::TONE_HI_DB as f32,
        )),
        pitch: config.pitch,
        seed: config.seed,
        sample_rate: config.sample_rate,
        threads: config.threads,
    }
}

pub(crate) struct AppState {
    engine: Mutex<Option<Engine>>,
    /// Yuklu motorun modeli (None = soguk). Gecis karari buradan verilir.
    loaded_model: Mutex<Option<ModelId>>,
    config: Mutex<TtsConfig>,
    /// Motor kilidini beklemeden "Durdur" calissin diye paylasilan bayrak.
    running: Mutex<Arc<AtomicBool>>,
    /// Dogal bitis ile kullanici durdurmasini ayirt eder.
    stop_req: Mutex<Arc<AtomicBool>>,
    /// Modellerin kok dizini (altinda `onnx/` + `antalia/`; settings.json ile kalici).
    model_dir: Mutex<String>,
    /// Rust tarafi ses calici (UI + HTTP API okutmasi icin tek el).
    player: parking_lot::Mutex<Option<Arc<AudioPlayer>>>,
    /// Calisan API sunucusu (kapatma kanali + port).
    api: parking_lot::Mutex<Option<ApiRuntime>>,
}

/// Indirilebilen/dogrulanabilen her dosya (katalog degerleri uretim aninda olculdu).
struct CatalogFile {
    name: &'static str,
    size: u64,
    sha256: &'static str,
}

/// Model dosya listesi: motor acilisi icin zorunlu dosyalar (indiriciyle ayni liste).
pub(crate) fn required_files(model: ModelId) -> &'static [&'static str] {
    match model {
        ModelId::EmaLightning => &[
            "text_encoder.onnx",
            "text_encoder.onnx.data",
            "sound_stage.onnx",
            "sound_stage.onnx.data",
            "decoder.onnx",
            "decoder.onnx.data",
            "vocab.json",
        ],
        ModelId::AntaliaMini => &[
            "antalia_text.onnx",
            "antalia_text.onnx.data",
            "antalia_prep.onnx",
            "antalia_prep.onnx.data",
            "antalia_step.onnx",
            "antalia_step.onnx.data",
            "antalia_vocoder.onnx",
            "antalia_vocoder.onnx.data",
            "symbols.json",
            "null_cond.f32",
        ],
    }
}

/// Calisma aninda kullanilan katalog girdisi (gomulu ya da paketten).
#[derive(Debug, Clone)]
struct ActiveFile {
    name: String,
    size: u64,
    sha256: String,
}

fn embedded_active(model: ModelId) -> Vec<ActiveFile> {
    download_catalog(model)
        .iter()
        .map(|f| ActiveFile {
            name: f.name.to_string(),
            size: f.size,
            sha256: f.sha256.to_string(),
        })
        .collect()
}

/// Gecerli katalog: derleme anindaki gomulu liste (cevrimdisi).
/// NOT: HuggingFace kaynak repolarinda (.pt/.safetensors) ONNX yoktur;
/// o yuzden agdan indirme/guncelleme yoktur. Kurulum modelleri gomulu
/// getirir; onarim USB/klasorden yapilir.
fn active_catalog(model: ModelId) -> Vec<ActiveFile> {
    embedded_active(model)
}

/// Indirme katalogu: dosya + bayt + sha256 (eksik/bozuk dosya bastan iner).
pub(crate) fn download_catalog(model: ModelId) -> &'static [CatalogFile] {
    match model {
        ModelId::EmaLightning => &[
            CatalogFile { name: "text_encoder.onnx", size: 318419, sha256: "dff4c4ecb9fd83b728311d79db456b0d55265341c7587e5c8b49c7c793b3040c" },
            CatalogFile { name: "text_encoder.onnx.data", size: 4788864, sha256: "cfcdbb56d03aba285c7c2a7b98dcd8a2816b0a1e57c5646a3020845e46b3cc19" },
            CatalogFile { name: "sound_stage.onnx", size: 4960187, sha256: "f50fd7161c7910705d5579e1dcda482179edf5834f5373b21a89b3da95692d88" },
            CatalogFile { name: "sound_stage.onnx.data", size: 17588224, sha256: "31d5978a83ace50404a4f573d5203c44d39fc602d08a5b2591d6e3b1b3ecac04" },
            CatalogFile { name: "decoder.onnx", size: 1371806, sha256: "a17108aef48ccc8d9534b8b3d7955e09495866c570f393891a236cfc4fa94717" },
            CatalogFile { name: "decoder.onnx.data", size: 11993088, sha256: "00d0994495ed90c3c9715c550d57f2dfc0e7ac7e906226b60e5a52f47228597f" },
            CatalogFile { name: "vocab.json", size: 260, sha256: "415f285e2decbd76d73e57d5881d19d7c32718529f5c850c45afe98c95c53e9f" },
        ],
        ModelId::AntaliaMini => &[
            CatalogFile { name: "antalia_text.onnx", size: 491604, sha256: "2f83287f41148cb4aec2a04040ba9fe86db2ac9f7e2c89ae6733062c091c705f" },
            CatalogFile { name: "antalia_text.onnx.data", size: 4006144, sha256: "23c92002833c76604c9f3a95e7d3eb45f63307d2a10b92b24e55ca50c210041f" },
            CatalogFile { name: "antalia_prep.onnx", size: 50202, sha256: "ec977c27933ba8e3562b89a6e19ceca028ef8b7e1220569509cbf03daa79df46" },
            CatalogFile { name: "antalia_prep.onnx.data", size: 600576, sha256: "f0263073186dd43117f7f6d5d95f750a1516bd207f397f3aaf807951688dbc27" },
            CatalogFile { name: "antalia_step.onnx", size: 2827493, sha256: "ba2bf52504dd7b6d8da23a5bab8a1339453ace82d88a1fd25f52f2f234de1a36" },
            CatalogFile { name: "antalia_step.onnx.data", size: 10164224, sha256: "65d4daabc9ad82e28c530180ad2d12dc97525b15d9dc1706025ea11ea21c6058" },
            CatalogFile { name: "antalia_vocoder.onnx", size: 339858, sha256: "91cc30d18c0a9ddd960dfd79bd6de9251d301494b7d879ef8edde09630255513" },
            CatalogFile { name: "antalia_vocoder.onnx.data", size: 32653312, sha256: "23cf126beab8acaeae6ba81736d19eb79e6f04d0431f54db394eaebbbea662a2" },
            CatalogFile { name: "symbols.json", size: 256, sha256: "ae4f9cba585c5ded5fdffa030057c74519ae874302a3aed344b5d11758172a0e" },
            CatalogFile { name: "null_cond.f32", size: 768, sha256: "01818b95856c4aab302906d81964099828c2565bd28bb805148908099e73376d" },
        ],
    }
}

/// Istenen aygita uyan oynaticiyi dondurur (gerekirse eskisini kapatip kurar).
/// Kilit kisa tutulur; dondurulen Arc kilitsiz kullanilir.
pub(crate) fn player_for(app: &AppHandle, device: &str) -> Result<Arc<AudioPlayer>, String> {
    let state = app.state::<AppState>();
    let mut slot = state.player.lock();
    let mismatch = slot
        .as_ref()
        .map(|p| p.key() != device.trim())
        .unwrap_or(true);
    if mismatch {
        if let Some(old) = slot.take() {
            old.shutdown();
        }
        let want = device.trim();
        let player = AudioPlayer::new(if want.is_empty() { None } else { Some(want) })
            .map_err(|e| format!("Ses aygiti acilamadi: {e}"))?;
        slot.replace(Arc::new(player));
    }
    Ok(Arc::clone(slot.as_ref().expect("oynatici kuruldu")))
}

pub(crate) struct ApiRuntime {
    port: u16,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
}

pub(crate) const DEFAULT_API_PORT: u16 = 8765;

/// Eski tek modelli ayardan kok dizine gecis: "models/onnx" -> "models".
fn migrate_model_dir(dir: &str) -> String {
    let t = dir.trim();
    if t == "models/onnx" || t == "models\\onnx" {
        return "models".to_string();
    }
    for sep in ["/onnx", "\\onnx"] {
        if let Some(root) = t.strip_suffix(sep) {
            if !root.is_empty() {
                return root.to_string();
            }
        }
    }
    t.to_string()
}

/// Aday kok dizinleri uretir (saf: test edilebilir).
/// Eski NSIS kurulumlarinda kaynaklar `_up_` adli ara klasore yerlesmisti
/// (`../models` -> `_up_/models`); geriye uyumluluk icin o da taranir.
/// Yeni kurulumlarda hedef daima `<yer>/models` duzenindedir.
fn candidate_roots(
    model_root: &str,
    cwd: Option<PathBuf>,
    exe_dir: Option<PathBuf>,
    res_dir: Option<PathBuf>,
) -> Vec<PathBuf> {
    let given = PathBuf::from(model_root);
    let mut out: Vec<PathBuf> = Vec::new();
    let mut push = |p: PathBuf| {
        if !out.contains(&p) {
            out.push(p);
        }
    };
    push(given.clone());
    if let Some(cwd) = cwd {
        push(cwd.join(&given));
        push(cwd.join("_up_").join(&given));
    }
    if let Some(exe) = exe_dir {
        push(exe.join(&given));
        // Eski NSIS duzeni: kaynaklar exe yaninda `_up_` altindaydi.
        push(exe.join("_up_").join(&given));
        if let Some(parent) = exe.parent() {
            push(parent.join(&given));
        }
    }
    if let Some(res) = res_dir {
        push(res.join(&given));
        push(res.join("_up_").join(&given));
        // resource_dir bazi surumlerde dogrudan kok degil alt dizindir.
        push(res.clone());
        if let Some(parent) = res.parent() {
            push(parent.join(&given));
        }
    }
    out
}

/// Model dizinini cozumle: aday koklerin altinda `<alt-klasor>/<ana-onnx>`
/// bulunan ilk aday kullanilir.
fn resolve_model_dir(app: &AppHandle, model_root: &str, model: ModelId) -> PathBuf {
    use tauri::Manager;
    let probe = match model {
        ModelId::EmaLightning => "text_encoder.onnx",
        ModelId::AntaliaMini => "antalia_text.onnx",
    };
    let cwd = std::env::current_dir().ok();
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|e| e.parent().map(|p| p.to_path_buf()));
    let res_dir = app.path().resource_dir().ok();
    let roots = candidate_roots(model_root, cwd, exe_dir, res_dir);
    for r in &roots {
        let c = r.join(model.subdir());
        if c.join(probe).exists() {
            log::info!("model dizini: {} ({})", c.display(), model.as_str());
            return c;
        }
        // Kimi yerlesimde kok dogrudan model klasorudur.
        if r.join(probe).exists() {
            log::info!("model dizini (kok): {} ({})", r.display(), model.as_str());
            return r.clone();
        }
    }
    // Hicbiri yoksa ilk adayi dondur ki hata mesaji anlasilir olsun.
    roots
        .into_iter()
        .next()
        .map(|r| r.join(model.subdir()))
        .unwrap_or_else(|| PathBuf::from(model_root).join(model.subdir()))
}

/// Istenen modelin motorunu kurar (agir is; cagiran thread'de calisir).
fn build_engine(
    app: &AppHandle,
    model_root: &str,
    model: ModelId,
    config: &TtsConfig,
) -> Result<(Engine, PathBuf), String> {
    let resolved = resolve_model_dir(app, model_root, model);
    let engine = match model {
        ModelId::EmaLightning => TtsEngine::new(&resolved, config.clone())
            .map(Engine::Ema)
            .map_err(|e| {
                format!(
                    "Motor baslatilamadi (aranan dizin: {}): {}. Once python scripts/export_onnx.py ile modelleri uretin.",
                    resolved.display(),
                    e
                )
            })?,
        ModelId::AntaliaMini => AntaliaEngine::new(&resolved, antalia_params_from(config))
            .map(Engine::Antalia)
            .map_err(|e| {
                format!(
                    "Motor baslatilamadi (aranan dizin: {}): {}. Once python scripts/export_antalia_onnx.py ile modelleri uretin.",
                    resolved.display(),
                    e
                )
            })?,
    };
    Ok((engine, resolved))
}

/// Motoru state'e yerlestirir + bayrak tutamaclarini paylasir.
fn store_engine(state: &tauri::State<'_, AppState>, engine: Engine, model: ModelId) {
    let handle = engine.running_handle();
    let stop_handle = engine.stop_handle();
    if let Ok(mut guard) = state.engine.try_lock() {
        *guard = Some(engine);
    }
    if let Ok(mut m) = state.loaded_model.lock() {
        *m = Some(model);
    }
    if let Ok(mut r) = state.running.lock() {
        *r = handle;
    }
    if let Ok(mut s) = state.stop_req.lock() {
        *s = stop_handle;
    }
}

#[tauri::command]
async fn initialize_engine(
    app: AppHandle,
    model_dir: String,
    model: ModelId,
    config: TtsConfig,
) -> Result<String, String> {
    // Agir is ayri thread'de: komut dagitimi asla bloklanmaz (Durdur hep calisir).
    tokio::task::spawn_blocking(move || {
        let root = migrate_model_dir(&model_dir);
        let state = app.state::<AppState>();
        let mut engine_guard = match state.engine.try_lock() {
            Ok(g) => g,
            Err(_) => {
                return Err(
                    "Motor şu an başka bir iş yapıyor; bitmesini bekleyin ya da Durdur'a basın."
                        .to_string(),
                )
            }
        };
        let mut config_guard = state.config.lock().map_err(|e| e.to_string())?;

        let mut cfg = config.clone();
        cfg.model = model;
        *config_guard = cfg.clone();
        if let Ok(mut d) = state.model_dir.lock() {
            *d = root.clone();
        }

        let (engine, resolved) = build_engine(&app, &root, model, &cfg)?;
        let handle = engine.running_handle();
        let stop_handle = engine.stop_handle();
        *engine_guard = Some(engine);
        if let Ok(mut m) = state.loaded_model.lock() {
            *m = Some(model);
        }
        if let Ok(mut r) = state.running.lock() {
            *r = handle;
        }
        if let Ok(mut s) = state.stop_req.lock() {
            *s = stop_handle;
        }
        // persist_settings kilit ister; elde tutulan kilitler
        // birakilmazsa ayni thread kendini kilitler (donma).
        drop(engine_guard);
        drop(config_guard);
        persist_settings(&app);
        Ok(format!(
            "Motor hazir ({}): {}",
            model.display_name(),
            resolved.display()
        ))
    })
    .await
    .map_err(|e| format!("Motor gorevi yarida kaldi: {e}"))?
}

/// Istenen model yuklu degilse kurar (soguk baslatma + model gecisi).
/// Kilitliyse beklemez; o anki isi bitiren taraf kazanir.
pub(crate) fn ensure_engine_ready(app: &AppHandle, model: ModelId) -> Result<(), String> {
    {
        let state = app.state::<AppState>();
        let guard = match state.engine.try_lock() {
            Ok(g) => g,
            Err(_) => {
                return Err(
                    "Motor şu an başka bir iş yapıyor; bitmesini bekleyin ya da Durdur'a basın."
                        .into(),
                )
            }
        };
        let loaded = state
            .loaded_model
            .lock()
            .map(|m| *m)
            .unwrap_or(None);
        if guard.is_some() && loaded == Some(model) {
            return Ok(());
        }
    }
    let (model_root, config) = {
        let state = app.state::<AppState>();
        let d = state
            .model_dir
            .lock()
            .map(|d| migrate_model_dir(&d))
            .unwrap_or_else(|_| "models".into());
        let mut c = state
            .config
            .lock()
            .map(|c| c.clone())
            .unwrap_or_default();
        c.model = model;
        (d, c)
    };
    let (engine, resolved) = build_engine(app, &model_root, model, &config).map_err(|e| {
        format!("{e} Arayuzden 'Motoru Baslat'a basin.")
    })?;
    let _ = resolved;
    let state = app.state::<AppState>();
    match state.engine.try_lock() {
        Ok(_) => (),
        // Bu sirada baskasi basladi/bitiriyor: motor birazdan hazir olur.
        Err(_) => return Ok(()),
    };
    store_engine(&state, engine, model);
    Ok(())
}

/// Her seyi susturur: cikarim bayragi + durdur-istegi + Rust calma + arayuz calma.
pub(crate) fn stop_all(app: &AppHandle) {
    if let Some(state) = app.try_state::<AppState>() {
        if let Ok(running) = state.running.lock() {
            running.store(false, Ordering::SeqCst);
        }
        if let Ok(stop_req) = state.stop_req.lock() {
            stop_req.store(true, Ordering::SeqCst);
        }
        // Kilit musaitse motora da dogrudan soyle (paylasilan bayrak zaten yeterli).
        if let Ok(guard) = state.engine.try_lock() {
            if let Some(engine) = guard.as_ref() {
                engine.stop();
            }
        }
        let player = state.player.lock();
        if let Some(p) = player.as_ref() {
            p.stop();
        }
    }
    let _ = app.emit("stop-playback", ());
}

#[tauri::command]
async fn synthesize(
    app: AppHandle,
    text: String,
    config: TtsConfig,
) -> Result<(), String> {
    // Agir is ayri thread'de: komut dagitimi asla bloklanmaz.
    tokio::task::spawn_blocking(move || {
        // Tek doğruluk kaynağı: arayüzdeki tüm config (hız, perde, seed, ...)
        // her sentezde motora yazılır. Eskiden sadece speed gidiyordu.
        let state = app.state::<AppState>();
        {
            let mut config_guard = state.config.lock().map_err(|e| e.to_string())?;
            *config_guard = config.clone();
        }

        // Onceki calma varsa sustur (ust uste binmesin).
        {
            let player = state.player.lock();
            if let Some(p) = player.as_ref() {
                p.stop();
            }
        }
        persist_settings(&app);

        // Secili aygita uygun oynatici (degistiyse eskisi kapatilir).
        let player = player_for(&app, &config.output_device)?;
        let rate = config.sample_rate;

        // Secili model yuklu degilse simdi kur (soguk baslatma + model gecisi).
        ensure_engine_ready(&app, config.model)?;

        // Kilitliyse BEKLEME: arayuzun takilma ihtimali sifirlanir.
        let mut engine_guard = match state.engine.try_lock() {
            Ok(g) => g,
            Err(_) => {
                return Err(
                    "Motor şu an başka bir iş yapıyor; bitmesini bekleyin ya da Durdur'a basın."
                        .to_string(),
                )
            }
        };
        let engine = engine_guard.as_mut().ok_or_else(|| {
            "Motor hazir degil; 'Motoru Baslat'a basin.".to_string()
        })?;
        engine.apply_config(&config);

        // Uretildikce cal: ilk ses icin sentezin bitmesi beklenmez.
        let app_progress = app.clone();
        let mut total = 0usize;
        let mut play_err: Option<String> = None;
        engine
            .synthesize(
                &text,
                &mut |chunk: &[f32]| {
                    if play_err.is_some() {
                        return;
                    }
                    total += chunk.len();
                    if let Err(e) = player.play(chunk, rate) {
                        play_err = Some(format!("Calma olmadi: {e}"));
                    }
                },
                &mut |done: usize, total: usize| {
                    let _ = app_progress.emit(
                        "synthesis-progress",
                        serde_json::json!({ "done": done, "total": total }),
                    );
                },
            )
            .map_err(|e| format!("Synthesis failed: {}", e))?;
        if let Some(e) = play_err {
            return Err(e);
        }
        player
            .flush()
            .map_err(|e| format!("Calma olmadi: {e}"))?;

        // Kuyruk bosalana kadar bekle (motor kilidi elde: gercek sira).
        // Durdurulursa erken cikilir; arayuz "Durduruldu" gosterir.
        let stop_flag = state
            .stop_req
            .lock()
            .map(|r| Arc::clone(&r))
            .unwrap_or_else(|_| Arc::new(AtomicBool::new(false)));
        player.wait_until_played(total as f32 / rate as f32, &stop_flag);

        app.emit("synthesis-complete", ()).ok();
        Ok(())
    })
    .await
    .map_err(|e| format!("Sentez gorevi yarida kaldi: {e}"))?
}

#[tauri::command]
async fn stop_synthesis(app: AppHandle) -> Result<(), String> {
    // Motor kilidini beklemez: uzun sentez ortasinda bile aninda durur.
    // Rust calma + arayuz calma da susar.
    stop_all(&app);
    Ok(())
}

#[tauri::command]
async fn save_wav(
    app: AppHandle,
    text: String,
    config: TtsConfig,
    output_path: String,
) -> Result<(), String> {
    tokio::task::spawn_blocking(move || {
        let state = app.state::<AppState>();
        {
            let mut config_guard = state.config.lock().map_err(|e| e.to_string())?;
            *config_guard = config.clone();
        }

        // API caliyorsa sustur (ust uste binmesin).
        {
            let player = state.player.lock();
            if let Some(p) = player.as_ref() {
                p.stop();
            }
        }
        persist_settings(&app);

        // Secili model yuklu degilse simdi kur (soguk baslatma + model gecisi).
        ensure_engine_ready(&app, config.model)?;

        let mut engine_guard = match state.engine.try_lock() {
            Ok(g) => g,
            Err(_) => {
                return Err(
                    "Motor şu an başka bir iş yapıyor; bitmesini bekleyin ya da Durdur'a basın."
                        .to_string(),
                )
            }
        };
        let engine = engine_guard.as_mut().ok_or_else(|| {
            "Motor hazir degil; 'Motoru Baslat'a basin.".to_string()
        })?;
        engine.apply_config(&config);

        let samples = engine
            .synthesize_to_vec(&text)
            .map_err(|e| format!("Synthesis failed: {}", e))?;

        engine
            .save_wav(&samples, &output_path)
            .map_err(|e| format!("Failed to save WAV: {}", e))?;

        Ok(())
    })
    .await
    .map_err(|e| format!("Kayit gorevi yarida kaldi: {e}"))?
}

#[tauri::command]
async fn get_config(state: State<'_, AppState>) -> Result<TtsConfig, String> {
    let config_guard = state.config.lock().map_err(|e| e.to_string())?;
    Ok(config_guard.clone())
}

#[tauri::command]
async fn update_config(
    app: AppHandle,
    config: TtsConfig,
) -> Result<(), String> {
    let state = app.state::<AppState>();
    {
        let mut config_guard = state.config.lock().map_err(|e| e.to_string())?;
        *config_guard = config.clone();
    }
    // Motor mesgulse beklemeden gec: paylasilan ayar saklandi, motor
    // bir sonraki sentezde guncel ayari zaten aliyor.
    if let Ok(mut engine_guard) = state.engine.try_lock() {
        if let Some(engine) = engine_guard.as_mut() {
            engine.apply_config(&config);
        }
    }
    persist_settings(&app);
    Ok(())
}

#[derive(Debug, Clone, serde::Serialize)]
struct AudioDeviceDto {
    name: String,
    is_default: bool,
}

#[tauri::command]
async fn list_audio_devices() -> Result<Vec<AudioDeviceDto>, String> {
    audio_player::list_output_devices()
        .map(|ds| {
            ds.into_iter()
                .map(|d| AudioDeviceDto {
                    name: d.name,
                    is_default: d.is_default,
                })
                .collect()
        })
        .map_err(|e| format!("Ses aygitlari listelenemedi: {e}"))
}

/// Cikis aygitini secer (bos metin = sistem varsayilani). Motor kilidi gerekmez.
#[tauri::command]
async fn set_output_device(app: AppHandle, name: String) -> Result<String, String> {
    let name = name.trim().to_string();
    {
        let state = app.state::<AppState>();
        let mut config_guard = state.config.lock().map_err(|e| e.to_string())?;
        config_guard.output_device = name.clone();
    }
    persist_settings(&app);
    Ok(if name.is_empty() {
        "Sistem varsayılanı".into()
    } else {
        name
    })
}

#[tauri::command]
async fn check_model_files(
    app: AppHandle,
    model_dir: String,
    model: ModelId,
) -> Result<Vec<String>, String> {
    let root = migrate_model_dir(&model_dir);
    let path = resolve_model_dir(&app, &root, model);
    let mut missing = Vec::new();

    for file in required_files(model) {
        if !path.join(file).exists() {
            missing.push(format!("{}/{}", path.display(), file));
        }
    }

    Ok(missing)
}

#[derive(Debug, Clone, serde::Serialize)]
struct ModelStatusDto {
    id: ModelId,
    name: String,
    ready: bool,
    missing: Vec<String>,
    dir: String,
    size_mb: f64,
}

/// Her iki modelin durumunu dondurur (arayuz kartlari icin tek cagri).
#[tauri::command]
async fn model_status(app: AppHandle) -> Result<Vec<ModelStatusDto>, String> {
    let root = app
        .state::<AppState>()
        .model_dir
        .lock()
        .map(|d| migrate_model_dir(&d))
        .unwrap_or_else(|_| "models".into());
    let mut out = Vec::new();
    for model in [ModelId::EmaLightning, ModelId::AntaliaMini] {
        let path = resolve_model_dir(&app, &root, model);
        let mut missing = Vec::new();
        let mut bytes = 0u64;
        for file in required_files(model) {
            let p = path.join(file);
            match std::fs::metadata(&p) {
                Ok(m) if m.is_file() => bytes += m.len(),
                _ => missing.push(file.to_string()),
            }
        }
        out.push(ModelStatusDto {
            id: model,
            name: model.display_name().to_string(),
            ready: missing.is_empty(),
            missing,
            dir: path.display().to_string(),
            size_mb: (bytes as f64) / (1024.0 * 1024.0),
        });
    }
    Ok(out)
}

/// Tek tikla model hazirlama (diyalog yok, ag yok): hedef `models/<alt>`
/// dizinini yoksa yaratir, dosyalari gomulu katalogla dogrular.
/// Tamamsa "Hazir" doner; eksik varsa hangi dosyalarin eksik oldugunu ve
/// cozumu (kurulumu onarma / Klasorden Kur) soyleyen hata doner.
#[tauri::command]
async fn download_model(app: AppHandle, model: ModelId) -> Result<String, String> {
    tokio::task::spawn_blocking(move || {
        let state = app.state::<AppState>();
        let root = state
            .model_dir
            .lock()
            .map(|d| migrate_model_dir(&d))
            .unwrap_or_else(|_| "models".into());
        let target_root = resolve_download_root(&app, &root, model);
        std::fs::create_dir_all(&target_root).map_err(|e| {
            format!("Hedef dizin acilamadi ({}): {e}", target_root.display())
        })?;
        let mut missing = Vec::new();
        for f in active_catalog(model) {
            if !file_matches(&target_root.join(&f.name), f.size, &f.sha256) {
                missing.push(f.name);
            }
        }
        if missing.is_empty() {
            Ok(format!(
                "{} hazir: {}",
                model.display_name(),
                target_root.display()
            ))
        } else {
            Err(format!(
                "{} icin {} dosya eksik/bozuk ({}): kurulumu onarin ya da 'Klasorden Kur' ile yedek klasorden yukleyin.",
                model.display_name(),
                missing.len(),
                missing.join(", ")
            ))
        }
    })
    .await
    .map_err(|e| format!("Kurulum gorevi yarida kaldi: {e}"))?
}

/// Secili modelin eksik/bozuk dosyalarini kullanicinin sectigi paket
/// klasorunden kurar/dogrular (cevrimdisi: USB, yedek klasor).
/// Paket duzeni: `<klasor>/ema-lightning/<dosya>` ve `<klasor>/antalia-mini/<dosya>`.
/// Ilerleme `model-download-progress`, bitis `model-download-complete` olayiyla duyurulur.
#[tauri::command]
async fn install_model_files(
    app: AppHandle,
    model: ModelId,
    src_dir: String,
) -> Result<String, String> {
    tokio::task::spawn_blocking(move || {
        install_model_blocking(&app, model, &src_dir)
    })
    .await
    .map_err(|e| format!("Kurulum gorevi yarida kaldi: {e}"))?
}

fn install_model_blocking(
    app: &AppHandle,
    model: ModelId,
    src_dir: &str,
) -> Result<String, String> {
    let src_root = src_dir.strip_prefix("file://").unwrap_or(src_dir);
    if src_root.trim().is_empty() {
        return Err("Paket klasoru secilmedi.".to_string());
    }
    let root = {
        let state = app.state::<AppState>();
        state
            .model_dir
            .lock()
            .map(|d| migrate_model_dir(&d))
            .unwrap_or_else(|_| "models".into())
    };
    // Hedef: cozumlenebilir ilk adayin kok + alt klasor (yoksa exe yani).
    let target_root = resolve_download_root(app, &root, model);
    std::fs::create_dir_all(&target_root)
        .map_err(|e| format!("Hedef dizin acilamadi ({}): {e}", target_root.display()))?;
    let files = active_catalog(model);
    let total = files.len();
    for (i, f) in files.iter().enumerate() {
        let dest = target_root.join(&f.name);
        // Saglam dosya varsa atla (boyut + sha256).
        if file_matches(&dest, f.size, &f.sha256) {
            emit_dl(app, model, &f.name, i, total, f.size, f.size);
            continue;
        }
        let part = target_root.join(format!("{}.part", f.name));
        copy_local(app, model, src_root, f, &part, i, total)?;
        if !file_matches(&part, f.size, &f.sha256) {
            let _ = std::fs::remove_file(&part);
            return Err(format!(
                "Dogrulama tutmadi: {} (paketteki dosya farkli/bozuk, parca silindi)",
                f.name
            ));
        }
        std::fs::rename(&part, &dest)
            .map_err(|e| format!("Dosya yerine konulamadi ({}): {e}", f.name))?;
        emit_dl(app, model, &f.name, i, total, f.size, f.size);
    }
    let dir = target_root.display().to_string();
    let _ = app.emit(
        "model-download-complete",
        serde_json::json!({ "model": model, "dir": dir }),
    );
    Ok(format!("{} hazir: {}", model.display_name(), dir))
}

/// Yerel dosyalari gomulu katalogla karsilastirir; eksik/degismis
/// dosya adlarini dondurur (bos = guncel). Tamamen cevrimdisi calisir.
#[tauri::command]
async fn check_updates(app: AppHandle, model: ModelId) -> Result<Vec<String>, String> {
    tokio::task::spawn_blocking(move || {
        let state = app.state::<AppState>();
        let root = state
            .model_dir
            .lock()
            .map(|d| migrate_model_dir(&d))
            .unwrap_or_else(|_| "models".into());
        let dir = resolve_model_dir(&app, &root, model);
        let mut changed = Vec::new();
        for f in active_catalog(model) {
            if !file_matches(&dir.join(&f.name), f.size, &f.sha256) {
                changed.push(f.name);
            }
        }
        Ok::<Vec<String>, String>(changed)
    })
    .await
    .map_err(|e| format!("Denetim gorevi yarida kaldi: {e}"))?
}

/// Indirme hedefi: cozumlenen dizin varsa orasi (onarim/guncelleme),
/// yoksa CWD/exe yani/ust sirayla ilk yazilabilir kok.
fn resolve_download_root(app: &AppHandle, root: &str, model: ModelId) -> PathBuf {
    let probed = resolve_model_dir(app, root, model);
    if probed.exists() {
        return probed;
    }
    // Aday henuz yok: yazilabilir ilk kok secilir.
    let mut cands: Vec<PathBuf> = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        cands.push(cwd.join(root).join(model.subdir()));
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            cands.push(dir.join(root).join(model.subdir()));
            // NSIS `_up_` duzeniyle tutarli: ust dizin de olabilir.
            if let Some(parent) = dir.parent() {
                cands.push(parent.join(root).join(model.subdir()));
            }
        }
    }
    cands.push(PathBuf::from(root).join(model.subdir()));
    for c in cands {
        if let Some(p) = c.parent() {
            if p.exists() {
                return c;
            }
        } else {
            return c;
        }
    }
    PathBuf::from(root).join(model.subdir())
}

fn emit_dl(
    app: &AppHandle,
    model: ModelId,
    file: &str,
    index: usize,
    total: usize,
    done: u64,
    expected: u64,
) {
    let _ = app.emit(
        "model-download-progress",
        serde_json::json!({
            "model": model,
            "file": file,
            "index": index,
            "total": total,
            "done": done,
            "expected": expected,
        }),
    );
}

fn file_matches(path: &PathBuf, size: u64, sha256: &str) -> bool {
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(_) => return false,
    };
    if data.len() as u64 != size {
        return false;
    }
    use sha2::Digest;
    let mut h = sha2::Sha256::new();
    h.update(&data);
    hex::encode(h.finalize()) == sha256
}

/// Yerel kurulum (file:// ya da dizin yolu): ayni dogrulamali hat uzerinden kopyalar.
/// Cevrimdisi tasiyici (USB) senaryosu icin birebirdir.
fn copy_local(
    app: &AppHandle,
    model: ModelId,
    base: &str,
    f: &ActiveFile,
    part: &PathBuf,
    index: usize,
    total: usize,
) -> Result<(), String> {
    let src_root = base.strip_prefix("file://").unwrap_or(base);
    let src = PathBuf::from(src_root)
        .join(model.as_str())
        .join(&f.name);
    let mut inp =
        std::fs::File::open(&src).map_err(|e| format!("Kaynak acilamadi ({}): {e}", src.display()))?;
    let mut out =
        std::fs::File::create(part).map_err(|e| format!("Parca yazilamadi ({}): {e}", part.display()))?;
    let mut buf = vec![0u8; 1 << 20];
    let mut done = 0u64;
    loop {
        use std::io::Read;
        let n = inp
            .read(&mut buf)
            .map_err(|e| format!("Kaynak okunamadi: {e}"))?;
        if n == 0 {
            break;
        }
        use std::io::Write;
        out.write_all(&buf[..n])
            .map_err(|e| format!("Parca yazilamadi: {e}"))?;
        done += n as u64;
        if done % (4 << 20) < (1 << 20) as u64 {
            emit_dl(app, model, &f.name, index, total, done, f.size);
        }
    }
    use std::io::Write;
    out.flush().map_err(|e| format!("Parca yazilamadi: {e}"))?;
    Ok(())
}



#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct SavedSettings {
    model_dir: String,
    config: TtsConfig,
}

impl Default for SavedSettings {
    fn default() -> Self {
        Self {
            model_dir: "models".into(),
            config: TtsConfig::default(),
        }
    }
}

fn settings_path() -> Option<PathBuf> {
    std::env::current_exe()
        .ok()
        .and_then(|e| e.parent().map(|p| p.join("settings.json")))
}

fn load_settings_file() -> Option<SavedSettings> {
    let path = settings_path()?;
    let data = std::fs::read_to_string(&path).ok()?;
    serde_json::from_str(&data).ok()
}

/// Ayarlari exe yanina yazar (tasinabilir). Hata olursa sessiz gecilir.
pub(crate) fn persist_settings(app: &AppHandle) {
    let Some(state) = app.try_state::<AppState>() else {
        return;
    };
    let (model_dir, config) = {
        let d = state
            .model_dir
            .lock()
            .map(|d| migrate_model_dir(&d))
            .unwrap_or_else(|_| "models".into());
        let c = state
            .config
            .lock()
            .map(|c| c.clone())
            .unwrap_or_default();
        (d, c)
    };
    if let Some(path) = settings_path() {
        let saved = SavedSettings { model_dir, config };
        if let Ok(json) = serde_json::to_string_pretty(&saved) {
            if let Err(e) = std::fs::write(&path, json) {
                log::debug!("settings.json yazilamadi: {e}");
            }
        }
    }
}

/// Kayitli ses profili (exe yanindaki presets.json'da durur, tasinabilir).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct VoicePreset {
    name: String,
    created_ms: u64,
    config: TtsConfig,
}

fn presets_path() -> Option<PathBuf> {
    std::env::current_exe()
        .ok()
        .and_then(|e| e.parent().map(|p| p.join("presets.json")))
}

fn load_presets_file() -> Vec<VoicePreset> {
    let Some(path) = presets_path() else {
        return Vec::new();
    };
    let data = std::fs::read_to_string(&path).unwrap_or_default();
    if data.trim().is_empty() {
        return Vec::new();
    }
    serde_json::from_str(&data).unwrap_or_default()
}

fn store_presets_file(list: &[VoicePreset]) -> Result<(), String> {
    let Some(path) = presets_path() else {
        return Err("presets.json yolu bulunamadi".to_string());
    };
    let json =
        serde_json::to_string_pretty(list).map_err(|e| format!("Profil yazilamadi: {e}"))?;
    std::fs::write(&path, json).map_err(|e| format!("presets.json yazilamadi: {e}"))?;
    Ok(())
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Profil adini temizler: kirpar, bos/uzun adi reddeder.
fn sanitize_preset_name(raw: &str) -> Result<String, String> {
    let name = raw.trim().replace(['\r', '\n', '\t'], " ");
    let name = name.split_whitespace().collect::<Vec<_>>().join(" ");
    if name.is_empty() {
        return Err("Profil adi bos olamaz".to_string());
    }
    if name.chars().count() > 60 {
        return Err("Profil adi en fazla 60 harf olabilir".to_string());
    }
    Ok(name)
}

/// Listeye ekler ya da ayni isimliyi gunceller (uzerine yazma semantiği).
/// Test edilebilir saf cekirdek: dosya giris-cikisi disarida.
fn upsert_preset(
    list: &mut Vec<VoicePreset>,
    name: String,
    config: TtsConfig,
    created_ms: u64,
) -> VoicePreset {
    if let Some(p) = list.iter_mut().find(|p| p.name == name) {
        p.config = config.clone();
        p.clone()
    } else {
        let preset = VoicePreset {
            name,
            created_ms,
            config,
        };
        list.push(preset.clone());
        preset
    }
}

/// Tum profilleri dondurur (isme gore sirali; "Varsayilan" arayuzde eklenir).
#[tauri::command]
async fn list_presets() -> Result<Vec<VoicePreset>, String> {
    let mut list = load_presets_file();
    list.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    Ok(list)
}

/// Profili kaydeder (ayni isim varsa uzerine yazar) ve kaydedileni dondurur.
#[tauri::command]
async fn save_preset(
    _app: AppHandle,
    name: String,
    config: TtsConfig,
) -> Result<VoicePreset, String> {
    let name = sanitize_preset_name(&name)?;
    let mut list = load_presets_file();
    let preset = upsert_preset(&mut list, name, config, now_ms());
    store_presets_file(&list)?;
    Ok(preset)
}

/// Profili siler (bulunamazsa hata).
#[tauri::command]
async fn delete_preset(_app: AppHandle, name: String) -> Result<String, String> {
    let name = sanitize_preset_name(&name)?;
    let mut list = load_presets_file();
    let before = list.len();
    list.retain(|p| p.name != name);
    if list.len() == before {
        return Err(format!("Profil bulunamadi: {name}"));
    }
    store_presets_file(&list)?;
    Ok(format!("Profil silindi: {name}"))
}

#[derive(Debug, Clone, serde::Serialize)]
struct ApiStatusDto {
    running: bool,
    port: u16,
    url: Option<String>,
    engine: String,
    /// Yuklu model (None = soguk).
    model: Option<ModelId>,
    speaking: bool,
}

fn api_status_inner(app: &AppHandle) -> ApiStatusDto {
    let state = app.state::<AppState>();
    let (running_api, port) = {
        let api = state.api.lock();
        match api.as_ref() {
            Some(rt) => (true, rt.port),
            None => {
                let p = state
                    .config
                    .lock()
                    .map(|c| c.api_port)
                    .unwrap_or(DEFAULT_API_PORT);
                (false, p)
            }
        }
    };
    // Kilitliyse bekleme: tutan bir istir, motor calisiyor demektir.
    let (engine, model) = match state.engine.try_lock().map(|g| g.as_ref().map(|e| e.model())) {
        Ok(Some(m)) => ("ready".to_string(), Some(m)),
        Ok(None) => {
            let m = state
                .loaded_model
                .lock()
                .map(|m| *m)
                .unwrap_or(None);
            (if m.is_some() { "ready" } else { "cold" }.to_string(), m)
        }
        Err(_) => {
            let m = state
                .loaded_model
                .lock()
                .map(|m| *m)
                .unwrap_or(None);
            ("ready".to_string(), m)
        }
    };
    let running_flag = state
        .running
        .lock()
        .map(|r| r.load(Ordering::SeqCst))
        .unwrap_or(false);
    let playing = {
        let p = state.player.lock();
        p.as_ref().map(|p| p.is_playing()).unwrap_or(false)
    };
    ApiStatusDto {
        running: running_api,
        port,
        url: running_api.then(|| format!("http://127.0.0.1:{port}")),
        engine,
        model,
        speaking: running_flag || playing,
    }
}

/// API sunucusunu acar (önceki varsa kapatilir). Port hatasi hemen doner.
fn start_api_server(app: &AppHandle, port: u16) -> Result<(), String> {
    if !(1024..=65535).contains(&port) {
        return Err("Port 1024-65535 araliginda olmali".into());
    }
    stop_api_server(app);
    let std_listener = std::net::TcpListener::bind(("127.0.0.1", port))
        .map_err(|e| format!("Port {port} dinlenemedi (kullanimda olabilir): {e}"))?;
    std_listener
        .set_nonblocking(true)
        .map_err(|e| e.to_string())?;
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let app_clone = app.clone();
    std::thread::spawn(move || {
        let rt = match tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
        {
            Ok(rt) => rt,
            Err(e) => {
                log::error!("API async calisma ortami kurulamadi: {e}");
                return;
            }
        };
        rt.block_on(async move {
            let listener = match tokio::net::TcpListener::from_std(std_listener) {
                Ok(l) => l,
                Err(e) => {
                    log::error!("API dinleyici devralinamadi: {e}");
                    return;
                }
            };
            log::info!("HTTP API dinleniyor: http://127.0.0.1:{port}");
            if let Err(e) = api_server::serve(listener, app_clone, port, rx).await {
                log::error!("HTTP API hatasi: {e:#}");
            }
        });
    });
    app.state::<AppState>().api.lock().replace(ApiRuntime {
        port,
        shutdown: Some(tx),
    });
    Ok(())
}

fn stop_api_server(app: &AppHandle) {
    let tx = app
        .state::<AppState>()
        .api
        .lock()
        .take()
        .and_then(|rt| rt.shutdown);
    if let Some(tx) = tx {
        let _ = tx.send(());
    }
}

#[tauri::command]
async fn get_settings(app: AppHandle) -> Result<SavedSettings, String> {
    let state = app.state::<AppState>();
    let model_dir = state
        .model_dir
        .lock()
        .map(|d| migrate_model_dir(&d))
        .unwrap_or_else(|_| "models".into());
    let config = state
        .config
        .lock()
        .map(|c| c.clone())
        .unwrap_or_default();
    Ok(SavedSettings { model_dir, config })
}

#[tauri::command]
async fn set_api_enabled(
    app: AppHandle,
    enabled: bool,
    port: u16,
) -> Result<ApiStatusDto, String> {
    // Motor acma islemi saniyeler surebilir: ayri thread'de yapilir.
    tokio::task::spawn_blocking(move || {
        {
            let state = app.state::<AppState>();
            let mut config_guard = state.config.lock().map_err(|e| e.to_string())?;
            config_guard.api_enabled = enabled;
            config_guard.api_port = port;
        }
        persist_settings(&app);
        if enabled {
            // Secili model soguksa simdi ac (ilk istekte de denenir).
            let model = app
                .state::<AppState>()
                .config
                .lock()
                .map(|c| c.model)
                .unwrap_or_default();
            if let Err(e) = ensure_engine_ready(&app, model) {
                log::warn!("API acilirken motor hazirlanamadi: {e}");
            }
            start_api_server(&app, port)?;
        } else {
            stop_api_server(&app);
        }
        Ok(api_status_inner(&app))
    })
    .await
    .map_err(|e| format!("API gorevi yarida kaldi: {e}"))?
}

#[tauri::command]
async fn api_status(app: AppHandle) -> Result<ApiStatusDto, String> {
    Ok(api_status_inner(&app))
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    env_logger::init();

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(AppState {
            engine: Mutex::new(None),
            loaded_model: Mutex::new(None),
            config: Mutex::new(TtsConfig::default()),
            running: Mutex::new(Arc::new(AtomicBool::new(false))),
            stop_req: Mutex::new(Arc::new(AtomicBool::new(false))),
            model_dir: Mutex::new("models".into()),
            player: parking_lot::Mutex::new(None),
            api: parking_lot::Mutex::new(None),
        })
        .setup(|app| {
            let handle = app.handle().clone();
            // Kayitli ayarlari yukle (eski "models/onnx" kok'e gocer).
            if let Some(saved) = load_settings_file() {
                if let Some(state) = handle.try_state::<AppState>() {
                    if let Ok(mut c) = state.config.lock() {
                        *c = saved.config;
                    }
                    if let Ok(mut d) = state.model_dir.lock() {
                        *d = migrate_model_dir(&saved.model_dir);
                    }
                }
            }
            // Komut satiri bayraklari (headless kullanim): --api [--api-port N]
            let args: Vec<String> = std::env::args().collect();
            let cli_api = args.iter().any(|a| a == "--api");
            let cli_port = args
                .iter()
                .position(|a| a == "--api-port")
                .and_then(|i| args.get(i + 1))
                .and_then(|s| s.parse::<u16>().ok());
            if cli_api || cli_port.is_some() {
                if let Some(state) = handle.try_state::<AppState>() {
                    if let Ok(mut c) = state.config.lock() {
                        c.api_enabled = true;
                        if let Some(p) = cli_port {
                            c.api_port = p;
                        }
                    }
                }
                persist_settings(&handle);
            }
            // API aciksa motor + sunucuyu baslat.
            let (enabled, port, model) = match handle.try_state::<AppState>() {
                Some(state) => match state.config.lock() {
                    Ok(c) => (c.api_enabled, c.api_port, c.model),
                    Err(_) => (false, DEFAULT_API_PORT, ModelId::default()),
                },
                None => (false, DEFAULT_API_PORT, ModelId::default()),
            };
            if enabled {
                if let Err(e) = ensure_engine_ready(&handle, model) {
                    log::warn!("Baslangicta motor hazirlanamadi: {e}");
                }
                if let Err(e) = start_api_server(&handle, port) {
                    log::warn!("Baslangicta API baslatilamadi: {e}");
                }
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            initialize_engine,
            synthesize,
            stop_synthesis,
            save_wav,
            get_config,
            update_config,
            check_model_files,
            model_status,
            download_model,
            install_model_files,
            check_updates,
            list_presets,
            save_preset,
            delete_preset,
            get_settings,
            set_api_enabled,
            api_status,
            list_audio_devices,
            set_output_device,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_id_serializes_to_api_names() {
        assert_eq!(
            serde_json::to_string(&ModelId::EmaLightning).unwrap(),
            "\"ema-lightning\""
        );
        assert_eq!(
            serde_json::to_string(&ModelId::AntaliaMini).unwrap(),
            "\"antalia-mini\""
        );
        assert_eq!(
            serde_json::from_str::<ModelId>("\"antalia-mini\"").unwrap(),
            ModelId::AntaliaMini
        );
        // Eski settings.json (model alani yok) EMA varsayar.
        let cfg: TtsConfig = serde_json::from_str(
            r#"{"speed":1.0,"sample_rate":48000,"threads":4,"use_gpu":false,"chunk_size":1024,"seed":0,"pitch":0.0}"#,
        )
        .unwrap();
        assert_eq!(cfg.model, ModelId::EmaLightning);
        assert_eq!(cfg.steps, 8);
        assert_eq!(cfg.cfg, 2.0);
        assert_eq!(cfg.temperature, 0.6);
        assert!(cfg.eq);
        assert_eq!(cfg.theme, "dark");
    }

    #[test]
    fn migrate_old_single_model_dir() {
        assert_eq!(migrate_model_dir("models/onnx"), "models");
        assert_eq!(migrate_model_dir("models\\onnx"), "models");
        assert_eq!(migrate_model_dir("C:\\paket\\models\\onnx"), "C:\\paket\\models");
        assert_eq!(migrate_model_dir("models"), "models");
        assert_eq!(migrate_model_dir("D:\\tts\\kök"), "D:\\tts\\kök");
    }

    #[test]
    fn catalog_covers_required_files() {
        for model in [ModelId::EmaLightning, ModelId::AntaliaMini] {
            let cat = download_catalog(model);
            let names: Vec<&str> = cat.iter().map(|f| f.name).collect();
            for req in required_files(model) {
                assert!(
                    names.contains(req),
                    "{model:?} icin {req} katalogda yok"
                );
            }
            // Boyut/ozet bos olmamali (0 baytlik dosya indirilemez).
            for f in cat {
                assert!(f.size > 0, "{} boyutu sifir", f.name);
                assert_eq!(f.sha256.len(), 64, "{} ozeti 64 hane olmali", f.name);
            }
        }
    }

    #[test]
    fn shared_config_maps_to_antalia_params() {
        let mut cfg = TtsConfig::default();
        cfg.model = ModelId::AntaliaMini;
        cfg.speed = 1.1;
        cfg.steps = 4;
        cfg.cfg = 2.5;
        cfg.temperature = 0.6;
        cfg.eq = false;
        cfg.pitch = 2.0;
        cfg.seed = 42;
        cfg.sample_rate = 24000;
        cfg.threads = 2;
        let p = antalia_params_from(&cfg);
        assert_eq!(p.speed, 1.1);
        assert_eq!(p.steps, 4);
        assert_eq!(p.cfg, 2.5);
        assert_eq!(p.temperature, 0.6);
        assert!(!p.eq);
        assert_eq!(p.pitch, 2.0);
        assert_eq!(p.seed, 42);
        assert_eq!(p.sample_rate, 24000);
        assert_eq!(p.threads, 2);
        // Adim kisgaci paylasilan tarafta da gecerli.
        cfg.steps = 99;
        assert_eq!(antalia_params_from(&cfg).steps, 32);
    }

    #[test]
    fn preset_name_is_sanitized() {
        assert_eq!(sanitize_preset_name("  Gece  Anlatimi ").unwrap(), "Gece Anlatimi");
        assert_eq!(
            sanitize_preset_name("a\tb\nc").unwrap(),
            "a b c"
        );
        assert!(sanitize_preset_name("   ").is_err());
        assert!(sanitize_preset_name(&"x".repeat(61)).is_err());
        assert!(sanitize_preset_name(&"x".repeat(60)).is_ok());
    }

    #[test]
    fn preset_upsert_overwrites_same_name() {
        let mut list = Vec::new();
        let mut cfg = TtsConfig::default();
        cfg.speed = 1.2;
        let p1 = upsert_preset(&mut list, "Hizli".into(), cfg.clone(), 100);
        assert_eq!(p1.created_ms, 100);
        assert_eq!(list.len(), 1);
        // Ayni isim: uzerine yazar, created korunur, liste buyumez.
        cfg.speed = 0.8;
        let p2 = upsert_preset(&mut list, "Hizli".into(), cfg.clone(), 200);
        assert_eq!(list.len(), 1);
        assert_eq!(p2.created_ms, 100);
        assert_eq!(list[0].config.speed, 0.8);
        // Yeni isim: eklenir.
        upsert_preset(&mut list, "Yavas".into(), TtsConfig::default(), 300);
        assert_eq!(list.len(), 2);
        // Silme: retain semantiği.
        list.retain(|p| p.name != "Hizli");
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].name, "Yavas");
    }

    #[test]
    fn preset_roundtrips_through_json() {
        let mut list = Vec::new();
        upsert_preset(&mut list, "Antalia Cocuk".into(), TtsConfig::default(), 1);
        let json = serde_json::to_string_pretty(&list).unwrap();
        let back: Vec<VoicePreset> = serde_json::from_str(&json).unwrap();
        assert_eq!(back.len(), 1);
        assert_eq!(back[0].name, "Antalia Cocuk");
        assert_eq!(back[0].config.model, ModelId::EmaLightning);
    }

    #[test]
    fn candidate_roots_cover_installed_layouts() {
        let roots = candidate_roots(
            "models",
            Some(PathBuf::from("C:\\is")),
            Some(PathBuf::from("C:\\is\\Hitabet")),
            Some(PathBuf::from("C:\\is\\Hitabet\\kaynak")),
        );
        let has = |p: &str| roots.iter().any(|r| r == &PathBuf::from(p));
        // Verilen + CWD + exe yani + eski `_up_` duzenleri + kaynak.
        assert!(has("models"));
        assert!(has("C:\\is\\models"));
        assert!(has("C:\\is\\Hitabet\\models"));
        assert!(has("C:\\is\\Hitabet\\_up_\\models"));
        assert!(has("C:\\is\\Hitabet\\kaynak\\models"));
        assert!(has("C:\\is\\Hitabet\\kaynak"));
        assert!(has("C:\\is\\Hitabet\\kaynak\\_up_\\models"));
        // Tekrar yok.
        let mut sorted = roots.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), roots.len());
    }

    #[test]
    fn active_catalog_is_embedded_and_offline() {
        // Katalog derleme aninda gomuludur; ag gerektirmez.
        for model in [ModelId::EmaLightning, ModelId::AntaliaMini] {
            let got = active_catalog(model);
            assert_eq!(got.len(), required_files(model).len());
            assert!(got.iter().all(|f| !f.sha256.is_empty() && f.size > 0));
        }
    }

    #[test]
    fn file_matches_verifies_size_and_hash() {
        let dir = std::env::temp_dir().join("ema-dl-test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("probe.bin");
        std::fs::write(&path, b"merhaba").unwrap();
        use sha2::Digest;
        let mut h = sha2::Sha256::new();
        h.update(b"merhaba");
        let hex = hex::encode(h.finalize());
        assert!(file_matches(&path, 7, &hex));
        assert!(!file_matches(&path, 8, &hex));
        assert!(!file_matches(&path, 7, &"0".repeat(64)));
        assert!(!file_matches(&dir.join("yok.bin"), 7, &hex));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
