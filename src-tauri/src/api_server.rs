use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use tauri::{AppHandle, Emitter, Manager};
use tower_http::cors::CorsLayer;

use crate::tts_engine::{ModelId, TtsConfig};
use crate::AppState;

/// Tek istekte en fazla harf (bellek/sure siniri).
pub const MAX_TEXT_CHARS: usize = 5000;

#[derive(Clone)]
pub struct ApiShared {
    pub app: AppHandle,
    pub port: u16,
}

// ---------------------------------------------------------------------------
// Istek / yanit tipleri
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct SpeakRequest {
    pub text: String,
    /// Model secimi ("ema-lightning" | "antalia-mini"). Verilmezse kayitli model.
    pub model: Option<ModelId>,
    pub speed: Option<f32>,
    pub pitch: Option<f32>,
    pub seed: Option<u64>,
    /// Antalia: akis adimi (1/2/4/8). EMA yok sayar.
    pub steps: Option<usize>,
    /// Antalia: CFG olcegi. EMA yok sayar.
    pub cfg: Option<f32>,
    /// Antalia: gurultu sicakligi. EMA yok sayar.
    pub temperature: Option<f32>,
    /// Antalia: ton filtresi + limiter. EMA yok sayar.
    pub eq: Option<bool>,
    /// Dogrudan konus (varsayilan true). false ise sadece kaydeder.
    pub speak: Option<bool>,
    /// Dosyaya kaydet (varsayilan false). Klasor: Videolar/Hitabet.
    pub save: Option<bool>,
    /// Dosya adi (uzantisiz da olur). Verilmezse zaman damgali uretilir.
    pub filename: Option<String>,
}

#[derive(Debug, Serialize)]
struct SpeakOk {
    status: String,
    seconds: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    file_url: Option<String>,
}

#[derive(Debug, Serialize)]
struct StatusBody {
    engine: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<&'static str>,
    speaking: bool,
    queue_secs: f32,
    port: u16,
    #[serde(skip_serializing_if = "Option::is_none")]
    device: Option<String>,
}

#[derive(Debug, Serialize)]
struct HealthBody {
    status: &'static str,
    engine: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<&'static str>,
    version: &'static str,
}

#[derive(Debug, Serialize)]
struct ErrorBody {
    error: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    hint: Option<String>,
}

fn err_body(error: impl Into<String>, hint: Option<&str>) -> ErrorBody {
    ErrorBody {
        error: error.into(),
        hint: hint.map(|s| s.to_string()),
    }
}

// ---------------------------------------------------------------------------
// Saf dogrulama (birim testli)
// ---------------------------------------------------------------------------

/// Metni temizler: kirpar, bos ve asiri uzun girisi reddeder.
pub fn clean_text(raw: &str) -> Result<String, String> {
    let t = raw.trim().to_string();
    if t.is_empty() {
        return Err("text bos olamaz".into());
    }
    if t.chars().count() > MAX_TEXT_CHARS {
        return Err(format!(
            "text cok uzun (en fazla {MAX_TEXT_CHARS} harf)"
        ));
    }
    Ok(t)
}

/// Istekteki opsiyonel degerleri motor ayarlarina isler (kisgacli).
pub fn apply_overrides(mut cfg: TtsConfig, req: &SpeakRequest) -> TtsConfig {
    if let Some(m) = req.model {
        cfg.model = m;
    }
    if let Some(s) = req.speed {
        cfg.speed = s.clamp(0.25, 4.0);
    }
    if let Some(p) = req.pitch {
        cfg.pitch = p.clamp(-12.0, 12.0);
    }
    if let Some(s) = req.seed {
        cfg.seed = s;
    }
    if let Some(s) = req.steps {
        cfg.steps = s.clamp(1, 32);
    }
    if let Some(c) = req.cfg {
        cfg.cfg = c.clamp(0.1, 8.0);
    }
    if let Some(t) = req.temperature {
        cfg.temperature = t.clamp(0.0, 2.0);
    }
    if let Some(e) = req.eq {
        cfg.eq = e;
    }
    cfg
}

/// Konus/kaydet karari. Varsayilan: konus, kaydetme.
/// Ikisi de kapaliysa hata (yapilacak is yok).
pub fn resolve_action(req: &SpeakRequest) -> Result<(bool, bool), String> {
    let speak = req.speak.unwrap_or(true);
    let save = req.save.unwrap_or(false);
    if !speak && !save {
        return Err("speak veya save parametrelerinden en az biri true olmali".into());
    }
    Ok((speak, save))
}

/// Dosya adini guvenli hale getirir (klasor atlama + yasak karakter yok).
pub fn sanitize_filename(raw: &str) -> String {
    let mut s: String = raw
        .chars()
        .map(|c| {
            if matches!(c, '<' | '>' | ':' | '"' | '/' | '\\' | '|' | '?' | '*')
                || c.is_control()
            {
                '_'
            } else {
                c
            }
        })
        .collect();
    s = s.trim().trim_matches('.').trim().to_string();
    if s.len() > 100 {
        s.truncate(100);
    }
    if s.is_empty() {
        s = timestamp_name();
    }
    if !s.to_lowercase().ends_with(".wav") {
        s.push_str(".wav");
    }
    s
}

fn timestamp_name() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    format!("tts_{ms}")
}

/// Kayit klasoru: Videolar/Hitabet (yoksa olusturulur).
fn save_dir(app: &AppHandle) -> Result<PathBuf, String> {
    use tauri::Manager;
    let base = app
        .path()
        .video_dir()
        .map_err(|e| format!("Video klasoru bulunamadi: {e}"))?;
    let dir = base.join("Hitabet");
    std::fs::create_dir_all(&dir)
        .map_err(|e| format!("Kayit klasoru acilamadi ({}): {e}", dir.display()))?;
    Ok(dir)
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

pub fn router(shared: ApiShared) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/status", get(status))
        .route("/speak", post(speak))
        .route("/stop", post(stop))
        .layer(CorsLayer::permissive())
        .with_state(shared)
}

pub async fn serve(
    listener: tokio::net::TcpListener,
    app: AppHandle,
    port: u16,
    shutdown: tokio::sync::oneshot::Receiver<()>,
) -> anyhow::Result<()> {
    let shared = ApiShared { app, port };
    axum::serve(listener, router(shared))
        .with_graceful_shutdown(async move {
            let _ = shutdown.await;
        })
        .await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Handler'lar
// ---------------------------------------------------------------------------

async fn health(State(sh): State<ApiShared>) -> impl IntoResponse {
    let engine = engine_state(&sh.app);
    (
        StatusCode::OK,
        Json(HealthBody {
            status: "ok",
            engine,
            model: model_state(&sh.app),
            version: env!("CARGO_PKG_VERSION"),
        }),
    )
}

async fn status(State(sh): State<ApiShared>) -> impl IntoResponse {
    let state = sh.app.state::<AppState>();
    // Kilitliyse bekleme: tutan bir istir, motor calisiyor demektir.
    let engine = {
        match state.engine.try_lock() {
            Ok(g) => g.is_some(),
            Err(_) => true,
        }
    };
    let running = state
        .running
        .lock()
        .map(|r| r.load(Ordering::SeqCst))
        .unwrap_or(false);
    let (playing, queue_secs, device) = {
        let p = state.player.lock();
        match p.as_ref() {
            Some(player) => (
                player.is_playing(),
                player.queued_seconds(),
                player.resolved_name().map(|s| s.to_string()),
            ),
            None => (false, 0.0, None),
        }
    };
    (
        StatusCode::OK,
        Json(StatusBody {
            engine: if engine { "ready" } else { "cold" },
            model: model_state(&sh.app),
            speaking: running || playing,
            queue_secs,
            port: sh.port,
            device,
        }),
    )
}

async fn stop(State(sh): State<ApiShared>) -> impl IntoResponse {
    crate::stop_all(&sh.app);
    (
        StatusCode::OK,
        Json(serde_json::json!({ "status": "stopped" })),
    )
}

async fn speak(
    State(sh): State<ApiShared>,
    Json(req): Json<SpeakRequest>,
) -> impl IntoResponse {
    let text = match clean_text(&req.text) {
        Ok(t) => t,
        Err(e) => {
            let code = if e.contains("uzun") {
                StatusCode::PAYLOAD_TOO_LARGE
            } else {
                StatusCode::BAD_REQUEST
            };
            return (code, Json(err_body(e, None))).into_response();
        }
    };
    match tokio::task::spawn_blocking(move || do_speak_blocking(&sh.app, &req, &text))
        .await
    {
        Ok(Ok(ok)) => (StatusCode::OK, Json(ok)).into_response(),
        Ok(Err((code, body))) => (code, Json(body)).into_response(),
        Err(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(err_body("is parcasi calismadi", None)),
        )
            .into_response(),
    }
}

type ApiFail = (StatusCode, ErrorBody);

fn fail(code: StatusCode, msg: impl Into<String>, hint: Option<&str>) -> ApiFail {
    (code, err_body(msg, hint))
}

/// Tum agir is burada (blocking thread): kilit -> sentez -> cal -> bekle.
fn do_speak_blocking(
    app: &AppHandle,
    req: &SpeakRequest,
    text: &str,
) -> Result<SpeakOk, ApiFail> {
    let state = app.state::<AppState>();

    // Arayuzde calan varsa sustur (ust uste binmesin).
    let _ = app.emit("stop-playback", ());

    // Istekteki model (yoksa kayitli) yuklu degilse kur: soguk baslatma + gecis.
    // Kilitliyse BEKLEME: tutan vardir, yani motor mesgul (soguk degil).
    let base: TtsConfig = state
        .config
        .lock()
        .map(|c| c.clone())
        .unwrap_or_default();
    let cfg = apply_overrides(base, req);
    crate::ensure_engine_ready(app, cfg.model).map_err(|e| {
        fail(
            StatusCode::SERVICE_UNAVAILABLE,
            format!("Motor hazir degil: {e}"),
            Some("Once arayuzden 'Motoru Baslat'a basin ya da model dizinini kontrol edin."),
        )
    })?;

    let mut guard = match state.engine.try_lock() {
        Ok(g) => g,
        Err(_) => {
            return Err(fail(
                StatusCode::SERVICE_UNAVAILABLE,
                "Motor şu an başka bir iş yapıyor; bitmesini bekleyin ya da /stop ile durdurun",
                None,
            ))
        }
    };
    let engine = guard.as_mut().ok_or_else(|| {
        fail(
            StatusCode::SERVICE_UNAVAILABLE,
            "Motor hazir degil",
            Some("Once arayuzden 'Motoru Baslat'a basin."),
        )
    })?;

    let rate = cfg.sample_rate;
    engine.apply_config(&cfg);

    let (do_speak, do_save) = resolve_action(req).map_err(|e| fail(StatusCode::BAD_REQUEST, e, None))?;

    // Secili aygita uygun oynatici (konusma istenmiyorsa kurulmaz).
    let player = if do_speak {
        Some(crate::player_for(app, &cfg.output_device).map_err(|e| {
            fail(
                StatusCode::SERVICE_UNAVAILABLE,
                format!("Ses aygiti acilamadi: {e}"),
                Some("Arayuzdeki ses aygiti listesini kontrol edin."),
            )
        })?)
    } else {
        None
    };

    // Arayuzde ilerleme gorunsun; uretildikce cal (streaming).
    // Kayit istenirse tamponda da birikir (en fazla ~23 MB).
    let app_ev = app.clone();
    let mut samples: Vec<f32> = Vec::new();
    let mut total = 0usize;
    let mut play_err: Option<String> = None;
    let player_ref = player.as_ref();
    let synth = engine.synthesize(
        text,
        &mut |chunk: &[f32]| {
            total += chunk.len();
            if do_save {
                samples.extend_from_slice(chunk);
            }
            if play_err.is_some() {
                return;
            }
            if let Some(p) = player_ref {
                if let Err(e) = p.play(chunk, rate) {
                    play_err = Some(format!("Calma olmadi: {e}"));
                }
            }
        },
        &mut |done: usize, total: usize| {
            let _ = app_ev.emit(
                "api-synthesis-progress",
                serde_json::json!({ "done": done, "total": total }),
            );
        },
    );
    if let Err(e) = synth {
        return Err(fail(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Sentez olmadi: {e}"),
            None,
        ));
    }
    if let Some(e) = play_err {
        return Err(fail(StatusCode::INTERNAL_SERVER_ERROR, e, None));
    }
    if let Some(p) = player_ref {
        if let Err(e) = p.flush() {
            return Err(fail(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Calma olmadi: {e}"),
                None,
            ));
        }
    }

    // Durdurulduysa yarim isi birak. NOT: `running` bayragi dogal bitiste
    // de false olur; kullanici istegi ayri `stop_req` bayragindadir.
    let stop_flag: Arc<std::sync::atomic::AtomicBool> = state
        .stop_req
        .lock()
        .map(|r| Arc::clone(&r))
        .unwrap_or_else(|_| Arc::new(std::sync::atomic::AtomicBool::new(false)));
    let stopped = || stop_flag.load(Ordering::SeqCst);
    if stopped() {
        let _ = app.emit("api-synthesis-complete", serde_json::json!({"status":"stopped"}));
        return Ok(SpeakOk {
            status: "stopped".into(),
            seconds: 0.0,
            file: None,
            file_url: None,
        });
    }

    // Kayit istenirse dosyaya yaz (durdurulmamissa).
    let mut file: Option<String> = None;
    let mut file_url: Option<String> = None;
    if do_save {
        let dir = save_dir(app)
            .map_err(|e| fail(StatusCode::INTERNAL_SERVER_ERROR, e, None))?;
        let name = match req.filename.as_deref() {
            Some(n) if !n.trim().is_empty() => sanitize_filename(n),
            _ => sanitize_filename(""),
        };
        let path = dir.join(&name);
        engine
            .save_wav(&samples, &path)
            .map_err(|e| fail(StatusCode::INTERNAL_SERVER_ERROR, format!("Kayit olmadi: {e}"), None))?;
        file_url = Some(format!(
            "file:///{}",
            path.to_string_lossy().replace('\\', "/")
        ));
        file = Some(path.to_string_lossy().into_owned());
    }
    let seconds = total as f32 / rate as f32;

    // Konusma istenmiyorsa bekleme yok.
    if do_speak {
        if let Some(p) = player_ref {
            if !p.wait_until_played(seconds, &stop_flag) {
                let _ = app.emit("api-synthesis-complete", serde_json::json!({"status":"stopped"}));
                return Ok(SpeakOk {
                    status: "stopped".into(),
                    seconds: 0.0,
                    file: None,
                    file_url: None,
                });
            }
        }
    }

    let status = match (do_speak, do_save) {
        (true, true) => "spoken+saved",
        (true, false) => "spoken",
        (false, true) => "saved",
        (false, false) => unreachable!("resolve_action garanti eder"),
    };
    let _ = app.emit(
        "api-synthesis-complete",
        serde_json::json!({ "status": status }),
    );
    Ok(SpeakOk {
        status: status.into(),
        seconds,
        file,
        file_url,
    })
}

fn engine_state(app: &AppHandle) -> &'static str {
    // Gecici state borrow omur sorunundan kacinmak icin deger ile alinir.
    // Kilitliyse bekleme: tutan bir istir, motor calisiyor demektir.
    let ready = match app.state::<AppState>().engine.try_lock() {
        Ok(g) => g.is_some(),
        Err(_) => true,
    };
    if ready {
        "ready"
    } else {
        "cold"
    }
}

/// Yuklu modelin API adi (None = soguk).
fn model_state(app: &AppHandle) -> Option<&'static str> {
    let state = app.state::<AppState>();
    if let Ok(g) = state.engine.try_lock() {
        if let Some(e) = g.as_ref() {
            return Some(e.model().as_str());
        }
    }
    state
        .loaded_model
        .lock()
        .map(|m| m.map(|id| id.as_str()))
        .unwrap_or(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(text: &str) -> SpeakRequest {
        SpeakRequest {
            text: text.into(),
            model: None,
            speed: None,
            pitch: None,
            seed: None,
            steps: None,
            cfg: None,
            temperature: None,
            eq: None,
            speak: None,
            save: None,
            filename: None,
        }
    }

    #[test]
    fn rejects_empty_and_blank_text() {
        assert!(clean_text("").is_err());
        assert!(clean_text("   \n\t  ").is_err());
    }

    #[test]
    fn trims_valid_text() {
        assert_eq!(clean_text("  merhaba  ").unwrap(), "merhaba");
    }

    #[test]
    fn rejects_oversize_text() {
        let big = "a".repeat(MAX_TEXT_CHARS + 1);
        assert!(clean_text(&big).is_err());
        assert!(clean_text(&"a".repeat(MAX_TEXT_CHARS)).is_ok());
    }

    #[test]
    fn overrides_clamp_and_apply() {
        let base = TtsConfig::default();
        let r = SpeakRequest {
            text: "x".into(),
            model: Some(ModelId::AntaliaMini),
            speed: Some(99.0),
            pitch: Some(-99.0),
            seed: Some(7),
            steps: Some(99),
            cfg: Some(99.0),
            temperature: Some(99.0),
            eq: Some(false),
            speak: None,
            save: None,
            filename: None,
        };
        let cfg = apply_overrides(base, &r);
        assert_eq!(cfg.model, ModelId::AntaliaMini);
        assert_eq!(cfg.speed, 4.0);
        assert_eq!(cfg.pitch, -12.0);
        assert_eq!(cfg.seed, 7);
        assert_eq!(cfg.steps, 32);
        assert_eq!(cfg.cfg, 8.0);
        assert_eq!(cfg.temperature, 2.0);
        assert!(!cfg.eq);
        let cfg2 = apply_overrides(TtsConfig::default(), &req("x"));
        assert_eq!(cfg2.model, ModelId::EmaLightning);
        assert_eq!(cfg2.speed, 1.0);
        assert_eq!(cfg2.pitch, 0.0);
        assert_eq!(cfg2.steps, 8);
        assert_eq!(cfg2.cfg, 2.0);
        assert_eq!(cfg2.temperature, 0.6);
        assert!(cfg2.eq);
    }

    #[test]
    fn action_defaults_to_speak_only() {
        assert_eq!(resolve_action(&req("x")).unwrap(), (true, false));
    }

    #[test]
    fn action_save_without_speak() {
        let mut r = req("x");
        r.speak = Some(false);
        r.save = Some(true);
        assert_eq!(resolve_action(&r).unwrap(), (false, true));
    }

    #[test]
    fn action_rejects_both_off() {
        let mut r = req("x");
        r.speak = Some(false);
        r.save = Some(false);
        assert!(resolve_action(&r).is_err());
    }

    #[test]
    fn filename_is_sanitized() {
        // Ayiricilar gider, klasor disina cikilamaz (ayni klasorde kalir).
        // (.ini uzantisi korunur, .wav eklenir: her zaman WAV yazilir.)
        assert_eq!(sanitize_filename("..\\..\\win.ini"), "_.._win.ini.wav");
        assert_eq!(sanitize_filename("kayit: 12/05?"), "kayit_ 12_05_.wav");
        assert_eq!(sanitize_filename("not.wav"), "not.wav");
        assert!(sanitize_filename("  ").starts_with("tts_"));
        assert!(sanitize_filename("").starts_with("tts_"));
    }
}
