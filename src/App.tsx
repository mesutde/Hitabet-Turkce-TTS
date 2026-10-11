import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";
import { open as openUrl } from "@tauri-apps/plugin-shell";
import type {
  TtsConfig,
  ApiStatus,
  SavedSettings,
  ModelId,
  ModelStatus,
  DownloadProgress,
  VoicePreset,
} from "./types";

const DEFAULT_PORT = 8765;

export interface AudioDevice {
  name: string;
  is_default: boolean;
}

const DEFAULT_TEXT =
  "Dr. Ayşe Yılmaz ile randevunuz 3 Kasım Pazartesi 10:15'te, 4. kattaki 412 numaralı odada olacak. randevunuza geç kalacaksanız önceden haber vermeniz rica olunur.";

export default function App() {
  const [text, setText] = useState(DEFAULT_TEXT);
  const [isSynthesizing, setIsSynthesizing] = useState(false);
  const [activeTab, setActiveTab] = useState<"text" | "api">("text");
  const [devices, setDevices] = useState<AudioDevice[]>([]);
  const [isEngineReady, setIsEngineReady] = useState(false);
  const [status, setStatus] = useState("Hazır");
  const [config, setConfig] = useState<TtsConfig>({
    model: "ema-lightning",
    speed: 1.0,
    sample_rate: 48000,
    threads: 4,
    use_gpu: false,
    chunk_size: 1024,
    seed: 0,
    pitch: 0,
    api_enabled: false,
    api_port: DEFAULT_PORT,
    output_device: "",
    steps: 8,
    cfg: 2.0,
    temperature: 0.6,
    eq: true,
    theme: "dark",
  });
  const [apiStatus, setApiStatus] = useState<ApiStatus | null>(null);
  const [modelDir, setModelDir] = useState("models");
  const [models, setModels] = useState<ModelStatus[]>([]);
  const [downloading, setDownloading] = useState<DownloadProgress | null>(null);
  const [presets, setPresets] = useState<VoicePreset[]>([]);
  const [presetName, setPresetName] = useState("");
  const [activePreset, setActivePreset] = useState<string>("__default");
  const finishedRef = useRef(true);
  const stopRequestedRef = useRef(false);

  const THEME_LIST = ["dark", "light", "sepia", "contrast"] as const;
  const THEME_LABEL: Record<string, string> = {
    dark: "Koyu",
    light: "Açık",
    sepia: "Sepya",
    contrast: "Kontrast",
  };

  // Görünüm: html[data-theme] üzerinden CSS değişkenleri değişir.
  // Kalıcılık config ile settings.json'a yazılır (update_config).
  useEffect(() => {
    const t = (THEME_LIST as readonly string[]).includes(config.theme)
      ? config.theme
      : "dark";
    document.documentElement.setAttribute("data-theme", t);
  }, [config.theme]);

  const handleThemeChange = async (theme: string) => {
    const next = { ...config, theme };
    setConfig(next);
    try {
      await invoke("update_config", { config: next });
    } catch (err) {
      setStatus(`Hata: ${err}`);
    }
  };

  // Calma Rust tarafinda (secili aygit); arayuz yalnizca ilerleme/durum izler.
  useEffect(() => {
    const unlistenProgress = listen<{ done: number; total: number }>(
      "synthesis-progress",
      (event) => {
        const { done, total } = event.payload;
        if (total > 1) {
          setStatus(`Sentezleniyor... %${Math.round((done / total) * 100)}`);
        }
      }
    );

    const unlistenComplete = listen("synthesis-complete", () => {
      finishedRef.current = true;
      setIsSynthesizing(false);
      if (stopRequestedRef.current) {
        setStatus("Durduruldu");
        return;
      }
      setStatus("Tamamlandı");
    });

    const unlistenApiProgress = listen<{ done: number; total: number }>(
      "api-synthesis-progress",
      (event) => {
        const { done, total } = event.payload;
        if (total > 1) {
          setStatus(`API okutuyor... %${Math.round((done / total) * 100)}`);
        } else {
          setStatus("API okutuyor...");
        }
      }
    );

    const unlistenApiComplete = listen<{ status: string }>(
      "api-synthesis-complete",
      (event) => {
        setStatus(
          event.payload?.status === "stopped" ? "Durduruldu" : "Tamamlandı (API)"
        );
      }
    );

    const unlistenDlProgress = listen<DownloadProgress>(
      "model-download-progress",
      (event) => setDownloading(event.payload)
    );

    const unlistenDlComplete = listen<{ model: ModelId; dir: string }>(
      "model-download-complete",
      (event) => {
        setDownloading(null);
        setStatus(`${event.payload?.model} indirildi: ${event.payload?.dir}`);
        refreshModels().catch(() => {});
      }
    );

    // Kayitli ayarlari yukle (model dizini + config + API durumu).
    invoke<SavedSettings>("get_settings")
      .then((s) => {
        if (s?.model_dir) setModelDir(s.model_dir);
        if (s?.config) setConfig((prev) => ({ ...prev, ...s.config }));
      })
      .catch(() => {});
    invoke<ApiStatus>("api_status")
      .then((st) => setApiStatus(st))
      .catch(() => {});
    refreshDevices().catch(() => {});
    refreshModels().catch(() => {});
    refreshPresets().catch(() => {});

    return () => {
      unlistenProgress.then((f) => f());
      unlistenComplete.then((f) => f());
      unlistenApiProgress.then((f) => f());
      unlistenApiComplete.then((f) => f());
      unlistenDlProgress.then((f) => f());
      unlistenDlComplete.then((f) => f());
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Ses cikis aygitlari (Rust/cpal listeler). Secim motora aninda yazilir.
  const refreshDevices = async () => {
    try {
      setDevices(await invoke<AudioDevice[]>("list_audio_devices"));
    } catch (err) {
      setStatus(`Hata: ${err}`);
    }
  };

  const selectDevice = async (name: string) => {
    try {
      await invoke("set_output_device", { name });
      setConfig((prev) => ({ ...prev, output_device: name }));
    } catch (err) {
      setStatus(`Hata: ${err}`);
    }
  };

  const refreshModels = async () => {
    try {
      setModels(await invoke<ModelStatus[]>("model_status"));
    } catch (err) {
      setStatus(`Hata: ${err}`);
    }
  };

  const refreshPresets = async () => {
    try {
      setPresets(await invoke<VoicePreset[]>("list_presets"));
    } catch (err) {
      setStatus(`Hata: ${err}`);
    }
  };

  const pushConfig = async (next: TtsConfig, msg: string) => {
    const modelChanged = next.model !== config.model;
    setConfig(next);
    if (modelChanged) {
      setIsEngineReady(false);
    }
    try {
      await invoke("update_config", { config: next });
      setStatus(msg + (modelChanged ? " (model değişti — Motoru Başlat)" : ""));
    } catch (err) {
      setStatus(`Hata: ${err}`);
    }
  };

  const applyPreset = async (name: string) => {
    setActivePreset(name);
    if (name === "__default") {
      // Fabrika varsayılanı: ortam alanları (API/aygıt) korunur.
      const next: TtsConfig = {
        ...config,
        model: "ema-lightning",
        speed: 1.0,
        pitch: 0,
        seed: 0,
        sample_rate: 48000,
        threads: 4,
        use_gpu: false,
        chunk_size: 1024,
        steps: 8,
        cfg: 2.0,
        temperature: 0.6,
        eq: true,
      };
      await pushConfig(next, "Varsayılan profil uygulandı");
      return;
    }
    const p = presets.find((x) => x.name === name);
    if (!p) return;
    // Kaydedilmiş profildeki ortam alanları yok sayılır, o anki ortam korunur.
    const next: TtsConfig = {
      ...p.config,
      api_enabled: config.api_enabled,
      api_port: config.api_port,
      output_device: config.output_device,
      theme: config.theme,
    };
    await pushConfig(next, `'${name}' profili uygulandı`);
  };

  const handleSavePreset = async () => {
    try {
      const saved = await invoke<VoicePreset>("save_preset", {
        name: presetName,
        config,
      });
      setPresetName("");
      await refreshPresets();
      setActivePreset(saved.name);
      setStatus(`Profil kaydedildi: ${saved.name}`);
    } catch (err) {
      setStatus(`Hata: ${err}`);
    }
  };

  const handleDeletePreset = async (name: string) => {
    if (!window.confirm(`'${name}' profili silinsin mi?`)) return;
    try {
      await invoke("delete_preset", { name });
      if (activePreset === name) setActivePreset("__default");
      await refreshPresets();
      setStatus(`Profil silindi: ${name}`);
    } catch (err) {
      setStatus(`Hata: ${err}`);
    }
  };

  const handleSelectModel = (model: ModelId) => {
    setConfig((prev) => ({ ...prev, model }));
    setIsEngineReady(false);
    setStatus("Model değişti — Motoru Başlat'a basın");
  };

  const handleInitialize = async () => {
    try {
      setStatus("Motor başlatılıyor...");
      const msg = await invoke<string>("initialize_engine", {
        modelDir,
        model: config.model,
        config,
      });
      setIsEngineReady(true);
      setStatus(typeof msg === "string" ? msg : "Motor hazır");
      refreshApiStatus();
    } catch (err) {
      setStatus(`Hata: ${err}`);
    }
  };

  const handleCheckUpdates = async (model: ModelId) => {
    try {
      setStatus(`${model} denetleniyor...`);
      const changed = await invoke<string[]>("check_updates", { model });
      await refreshModels();
      setStatus(
        changed.length === 0
          ? `${model} güncel (tüm dosyalar sağlam)`
          : `${model}: ${changed.length} dosya güncel değil — ${changed.join(", ")}`
      );
    } catch (err) {
      setStatus(`Hata: ${err}`);
    }
  };

  // Tek tık: klasör sormaz, kurulu dizindeki models/ altına kurar/doğrular.
  const handleDownload = async (model: ModelId) => {
    try {
      setStatus(`${model} hazırlanıyor...`);
      const msg = await invoke<string>("download_model", { model });
      setStatus(typeof msg === "string" ? msg : "Model hazır");
      await refreshModels();
      refreshApiStatus();
    } catch (err) {
      setStatus(`Hata: ${err}`);
    }
  };

  // Yedek/USB klasöründen kurulum (klasör seçicili).
  const handleInstallFromFolder = async (model: ModelId) => {
    try {
      const picked = await open({
        directory: true,
        multiple: false,
        title: "Model paket klasörünü seçin (altında ema-lightning/ ve antalia-mini/)",
      });
      if (typeof picked !== "string" || !picked) {
        return;
      }
      setStatus(`${model} kuruluyor...`);
      const msg = await invoke<string>("install_model_files", {
        model,
        srcDir: picked,
      });
      setDownloading(null);
      setStatus(typeof msg === "string" ? msg : "Kurulum tamamlandı");
      await refreshModels();
    } catch (err) {
      setDownloading(null);
      setStatus(`Hata: ${err}`);
    }
  };

  const handleSynthesize = async () => {
    if (!text.trim() || isSynthesizing) return;
    setIsSynthesizing(true);
    setStatus("Sentezleniyor...");
    finishedRef.current = false;
    stopRequestedRef.current = false;
    try {
      // Calma Rust tarafinda, secili aygitta, uretildikce baslar.
      await invoke("synthesize", { text, config });
      // Komut döndü ama tamamlanma olayı gecikebilir/kayıpsa 5 sn yedek:
      setTimeout(() => {
        if (!finishedRef.current && !stopRequestedRef.current) {
          finishedRef.current = true;
          setIsSynthesizing(false);
          setStatus("Tamamlandı");
        }
      }, 5000);
    } catch (err) {
      finishedRef.current = true;
      setStatus(`Hata: ${err}`);
      setIsSynthesizing(false);
    }
  };

  // Backend yavas/kilitli bile olsa arayuz ASLA takilmaz: 2.5 sn'de vazgecilir.
  const invokeWithTimeout = async <T,>(p: Promise<T>, ms: number): Promise<T> => {
    let timer: ReturnType<typeof setTimeout> | undefined;
    try {
      return await Promise.race([
        p,
        new Promise<T>((_, rej) => {
          timer = setTimeout(() => rej(new Error("zaman aşımı")), ms);
        }),
      ]);
    } finally {
      if (timer) clearTimeout(timer);
    }
  };

  const handleStop = async () => {
    stopRequestedRef.current = true;
    finishedRef.current = true;
    try {
      await invokeWithTimeout(invoke("stop_synthesis"), 2500);
    } catch {
      /* backend donmus olsa bile arayuz serbest kalir */
    }
    setIsSynthesizing(false);
    setStatus("Durduruldu");
  };

  const handleSaveWav = async () => {
    if (!text.trim()) return;
    try {
      const path = `output_${Date.now()}.wav`;
      await invoke("save_wav", { text, config, outputPath: path });
      setStatus(`WAV kaydedildi: ${path}`);
    } catch (err) {
      setStatus(`Hata: ${err}`);
    }
  };

  const handleConfigChange = (
    key: keyof TtsConfig,
    value: number | boolean | string
  ) => {
    setConfig((prev) => ({ ...prev, [key]: value }));
  };

  // Ses kimligini sifirlar: hiz 1.0, perde Dogal (0), seed 0 + Antalia
  // varsayilanlari (adim 8, cfg 2.0, sicaklik 0.6, EQ acik). Kalici yazilir.
  const handleResetVoice = async () => {
    const next = {
      ...config,
      speed: 1.0,
      pitch: 0,
      seed: 0,
      steps: 8,
      cfg: 2.0,
      temperature: 0.6,
      eq: true,
    };
    setConfig(next);
    try {
      await invoke("update_config", { config: next });
      setStatus("Varsayılan ses geri yüklendi (hız 1.0, ton Doğal, seed 0)");
    } catch (err) {
      setStatus(`Hata: ${err}`);
    }
  };

  const refreshApiStatus = async () => {
    try {
      setApiStatus(await invoke<ApiStatus>("api_status"));
    } catch {
      /* yoksay */
    }
  };

  const handleApiToggle = async (enabled: boolean) => {
    try {
      setStatus(enabled ? "API başlatılıyor..." : "API kapatılıyor...");
      const st = await invoke<ApiStatus>("set_api_enabled", {
        enabled,
        port: config.api_port,
      });
      setApiStatus(st);
      setConfig((prev) => ({ ...prev, api_enabled: enabled }));
      setStatus(enabled ? `API dinliyor: ${st.url}` : "API kapalı");
    } catch (err) {
      setStatus(`Hata: ${err}`);
      refreshApiStatus();
    }
  };

  const handleApiPortApply = async () => {
    if (!config.api_enabled) {
      // Sunucu kapalı: portu kaydet, açılınca kullanılsın.
      try {
        await invoke("update_config", { config });
        setStatus("Port kaydedildi (API açılınca kullanılacak)");
      } catch (err) {
        setStatus(`Hata: ${err}`);
      }
      return;
    }
    // Sunucu açık: yeni portla yeniden başlat.
    await handleApiToggle(true);
  };

  const copyText = async (value: string) => {
    try {
      await navigator.clipboard.writeText(value);
      setStatus("Panoya kopyalandı");
    } catch {
      setStatus("Hata: pano erişimi engellendi");
    }
  };

  return (
    <div className="app">
      <header className="header">
        <h1>
          Hitabet <span className="subtitle">Türkçe Yapay Ses Stüdyosu</span>
        </h1>
        <div className="header-right">
          <select
            className="theme-select"
            value={
              (THEME_LIST as readonly string[]).includes(config.theme)
                ? config.theme
                : "dark"
            }
            onChange={(e) => handleThemeChange(e.target.value)}
            title="Görünüm: Koyu / Açık / Sepya / Kontrast"
            aria-label="Görünüm teması"
          >
            {(THEME_LIST as readonly string[]).map((t) => (
              <option key={t} value={t}>
                {THEME_LABEL[t]}
              </option>
            ))}
          </select>
          <button
            type="button"
            className="dev-link"
            title="Geliştirici LinkedIn profili"
            onClick={() => openUrl("https://www.linkedin.com/in/mesutdemirci/")}
          >
            <svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor" aria-hidden="true">
              <path d="M20.45 20.45h-3.55v-5.57c0-1.33-.03-3.04-1.85-3.04-1.86 0-2.14 1.45-2.14 2.94v5.67H9.35V9h3.41v1.56h.05c.47-.9 1.63-1.85 3.36-1.85 3.6 0 4.27 2.37 4.27 5.46v6.28zM5.34 7.43a2.06 2.06 0 1 1 0-4.12 2.06 2.06 0 0 1 0 4.12zM7.12 20.45H3.56V9h3.56v11.45z" />
            </svg>
            Mesut Demirci
          </button>
          <span className={`status-badge ${isEngineReady ? "ready" : ""}`}>
            {status}
          </span>
        </div>
      </header>

      <main className="main">
        <section className="panel">
          <h2>Model Ayarları</h2>
          <div className="form-group">
            <label>Modellerin Kök Dizini (altında onnx/ + antalia/)</label>
            <div className="input-row">
              <input
                type="text"
                value={modelDir}
                onChange={(e) => setModelDir(e.target.value)}
                placeholder="models"
              />
              <button
                type="button"
                className="btn btn-secondary btn-icon"
                title="Klasör seç"
                aria-label="Model klasörünü seç"
                onClick={async () => {
                  try {
                    const picked = await open({
                      directory: true,
                      multiple: false,
                      title: "Modellerin kök klasörünü seçin (altında onnx/ ve antalia/ olmalı)",
                    });
                    if (typeof picked === "string" && picked) {
                      setModelDir(picked);
                      refreshModels().catch(() => {});
                    }
                  } catch (err) {
                    setStatus(`Hata: ${err}`);
                  }
                }}
              >
                <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
                </svg>
              </button>
            </div>
          </div>
          <div className="model-cards">
            {models.map((m) => (
              <div
                key={m.id}
                className={`model-card${config.model === m.id ? " selected" : ""}`}
              >
                <label className="model-pick">
                  <input
                    type="radio"
                    name="tts-model"
                    checked={config.model === m.id}
                    onChange={() => handleSelectModel(m.id)}
                  />
                  <span className="model-name">{m.name}</span>
                </label>
                <span className={`model-state ${m.ready ? "ready" : "missing"}`}>
                  {m.ready
                    ? `Hazır (${m.size_mb.toFixed(1)} MB)`
                    : `Eksik: ${m.missing.length} dosya`}
                </span>
                {!m.ready && (
                  <span className="hint-text">{m.missing.join(", ")}</span>
                )}
                <div className="button-row">
                  <button
                    type="button"
                    className="btn btn-secondary btn-small"
                    onClick={() => handleDownload(m.id)}
                    title="Kurulu dizindeki models/ altına kur/doğrula (klasör sormaz, çevrimdışı)"
                  >
                    Modeli Kur
                  </button>
                  {!m.ready && (
                    <button
                      type="button"
                      className="btn btn-secondary btn-small"
                      disabled={downloading?.model === m.id}
                      onClick={() => handleInstallFromFolder(m.id)}
                      title="USB/yedek klasörden eksik dosyaları kur (sha256 doğrulamalı, çevrimdışı)"
                    >
                      {downloading?.model === m.id ? "Kuruluyor..." : "Klasörden Kur"}
                    </button>
                  )}
                  <button
                    type="button"
                    className="btn btn-secondary btn-small"
                    onClick={() => handleCheckUpdates(m.id)}
                    title="Yerel dosyaları gömülü listeyle karşılaştır (çevrimdışı bütünlük denetimi)"
                  >
                    Güncellemeleri Denetle
                  </button>
                </div>
              </div>
            ))}
          </div>
          {downloading && (
            <div className="form-group">
              <label>
                {downloading.model} · {downloading.file} ({downloading.index + 1}/
                {downloading.total})
              </label>
              <div className="progress-track">
                <div
                  className="progress-fill"
                  style={{
                    width: `${
                      downloading.expected > 0
                        ? Math.min(
                            100,
                            (downloading.done / downloading.expected) * 100
                          )
                        : 0
                    }%`,
                  }}
                />
              </div>
            </div>
          )}
          <div className="form-group">
            <span className="hint-text">
              Kurulum iki modeli de gömülü getirir; internet gerekmez.
              "Modeli Kur" kurulu dizindeki models/ klasörünü hazırlar
              (yoksa yaratır, klasör sormaz). Dosya silinir/bozulursa
              "Klasörden Kur" ile USB'deki ya da yedek klasörden
              onarabilirsiniz (paket kökünde ema-lightning/ ve antalia-mini/
              beklenir, sha256 doğrulamalı).
            </span>
          </div>
          <div className="button-row">
            <button
              className="btn btn-primary"
              onClick={handleInitialize}
              disabled={isEngineReady}
            >
              {isEngineReady ? "Motor Hazır" : "Motoru Başlat"}
            </button>
            <button
              type="button"
              className="btn btn-secondary"
              onClick={() => refreshModels()}
              title="Dosya durumunu yeniden tara"
            >
              Yenile
            </button>
          </div>
        </section>

        <nav className="tabs" aria-label="Bölümler">
          {(
            [
              ["text", "1. Metin Girişi"],
              ["api", "2. HTTP API"],
            ] as const
          ).map(([key, label]) => (
            <button
              key={key}
              type="button"
              className={`tab${activeTab === key ? " active" : ""}`}
              aria-selected={activeTab === key}
              role="tab"
              onClick={() => setActiveTab(key)}
            >
              {label}
            </button>
          ))}
        </nav>

        {activeTab === "text" && (
          <>
        <section className="panel panel-wide">
          <h2>Metin Girişi</h2>
          <div className="form-group">
            <label>Ses Çıkış Aygıtı</label>
            <div className="input-row">
              <select
                value={
                  config.output_device === "" || devices.some((d) => d.name === config.output_device)
                    ? config.output_device
                    : ""
                }
                onChange={(e) => selectDevice(e.target.value)}
                title="Sesin çalınacağı aygıt"
              >
                <option value="">Sistem varsayılanı{devices.some((d) => d.is_default) ? ` (${devices.find((d) => d.is_default)?.name})` : ""}</option>
                {devices
                  .filter((d) => !(config.output_device === "" && d.is_default))
                  .map((d, i) => (
                    <option key={`${d.name}-${i}`} value={d.name}>
                      {d.name}
                    </option>
                  ))}
              </select>
              <button
                type="button"
                className="btn btn-secondary btn-icon"
                onClick={() => refreshDevices()}
                title="Aygıt listesini yenile"
                aria-label="Aygıt listesini yenile"
              >
                <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <path d="M21 12a9 9 0 1 1-2.64-6.36M21 3v6h-6" />
                </svg>
              </button>
            </div>
            <span className="hint-text">
              Seslendirilen ses seçili aygıttan çıkar.
            </span>
          </div>
          <textarea
            className="text-input"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Seslendirmek istediğiniz metni yazın..."
            rows={8}
          />
          <div className="button-row">
            <button
              className="btn btn-primary"
              onClick={handleSynthesize}
              disabled={!isEngineReady || isSynthesizing || !text.trim()}
            >
              {isSynthesizing ? "Sentezleniyor..." : "Seslendir"}
            </button>
              <button
                className="btn btn-danger"
                onClick={handleStop}
                disabled={!isSynthesizing}
              >
                Durdur
              </button>
            <button
              className="btn btn-secondary"
              onClick={handleSaveWav}
              disabled={!isEngineReady || !text.trim()}
            >
              WAV Kaydet
            </button>
          </div>
        </section>
          </>
        )}

        {activeTab === "text" && (
        <section className="panel">
          <h2>Sentez Parametreleri</h2>
          <div className="form-group">
            <label>Ses Profili</label>
            <select
              value={activePreset}
              onChange={(e) => applyPreset(e.target.value)}
              title="Kaydedilmiş ses ayarlarını uygula"
            >
              <option value="__default">Varsayılan</option>
              {presets.map((p) => (
                <option key={p.name} value={p.name}>
                  {p.name}
                </option>
              ))}
            </select>
            <span className="hint-text">
              Model + hız + perde + seed + Antalia ayarlarını tek tıkla uygular.
            </span>
          </div>
          <div className="form-group">
            <label>Profili Kaydet</label>
            <div className="input-row">
              <input
                type="text"
                value={presetName}
                onChange={(e) => setPresetName(e.target.value)}
                placeholder="örn. Gece Anlatımı"
                maxLength={60}
              />
              <button
                type="button"
                className="btn btn-secondary"
                onClick={handleSavePreset}
                disabled={!presetName.trim()}
                title="O anki ayarları bu isimle kaydet (aynı isim üzerine yazar)"
              >
                Kaydet
              </button>
            </div>
          </div>
          {presets.length > 0 && (
            <div className="form-group">
              <label>Kayıtlı Profiller</label>
              <div className="preset-list">
                {presets.map((p) => (
                  <div key={p.name} className="preset-row">
                    <button
                      type="button"
                      className={`btn btn-preset${activePreset === p.name ? " active" : ""}`}
                      onClick={() => applyPreset(p.name)}
                      title={`${p.config.model} · hız ${p.config.speed} · seed ${p.config.seed}`}
                    >
                      {p.name}
                    </button>
                    <button
                      type="button"
                      className="btn btn-secondary btn-icon"
                      onClick={() => handleDeletePreset(p.name)}
                      title={`'${p.name}' profilini sil`}
                      aria-label={`'${p.name}' profilini sil`}
                    >
                      <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                        <path d="M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2m3 0v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6" />
                      </svg>
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}
          <div className="form-group">
            <label>Hız: {config.speed.toFixed(2)}x (0.25 – 4.0)</label>
            <input
              type="range"
              min="0.25"
              max="4.0"
              step="0.05"
              value={config.speed}
              onChange={(e) => handleConfigChange("speed", parseFloat(e.target.value))}
            />
          </div>
          <div className="form-group">
            <label>
              Ses tonu:{" "}
              {config.pitch === 0
                ? "Doğal"
                : `${config.pitch > 0 ? "+" : "−"}${Math.abs(config.pitch)} yarım ton`}
            </label>
            <div className="button-row presets">
              {[
                { label: "Kalın", value: -2.5 },
                { label: "Normal", value: 0 },
                { label: "Çocuk", value: 3 },
              ].map((p) => (
                <button
                  key={p.label}
                  type="button"
                  className={`btn btn-preset${config.pitch === p.value ? " active" : ""}`}
                  onClick={() => handleConfigChange("pitch", p.value)}
                >
                  {p.label}
                </button>
              ))}
            </div>
            <input
              type="range"
              min="-4"
              max="4"
              step="0.5"
              value={config.pitch}
              onChange={(e) => handleConfigChange("pitch", parseFloat(e.target.value))}
              onDoubleClick={() => handleConfigChange("pitch", 0)}
            />
            <span className="hint-text">
              Aynı sesin perdesi değişir, hız aynı kalır. ±3'ün ötesi yapay duyulur.
            </span>
          </div>
          <div className="form-group">
            <label>Seed (aynı seed aynı sesi verir)</label>
            <div className="input-row">
              <input
                type="number"
                min="0"
                max="999999"
                step="1"
                value={config.seed}
                onChange={(e) => handleConfigChange("seed", parseInt(e.target.value || "0"))}
              />
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() =>
                  handleConfigChange("seed", Math.floor(Math.random() * 1000000))
                }
              >
                Rastgele
              </button>
            </div>
          </div>
          {config.model === "antalia-mini" && (
            <>
              <div className="form-group">
                <label>Akış Adımı (steps)</label>
                <select
                  value={config.steps}
                  onChange={(e) =>
                    handleConfigChange("steps", parseInt(e.target.value))
                  }
                  title="Az adım = hızlı, çok adım = doğal"
                >
                  <option value={1}>1 (en hızlı)</option>
                  <option value={2}>2</option>
                  <option value={4}>4</option>
                  <option value={8}>8 (varsayılan)</option>
                </select>
                <span className="hint-text">
                  Az adım hızlı ama daha yapay duyulur.
                </span>
              </div>
              <div className="form-group">
                <label>CFG Ölçeği: {config.cfg.toFixed(1)} (1.0 – 3.0)</label>
                <input
                  type="range"
                  min="1.0"
                  max="3.0"
                  step="0.1"
                  value={config.cfg}
                  onChange={(e) =>
                    handleConfigChange("cfg", parseFloat(e.target.value))
                  }
                />
                <span className="hint-text">
                  Metne sadakat; yüksek değer daha keskin ama riskli.
                </span>
              </div>
              <div className="form-group">
                <label>
                  Sıcaklık (temperature): {config.temperature.toFixed(2)} (0 – 1.5)
                </label>
                <input
                  type="range"
                  min="0"
                  max="1.5"
                  step="0.05"
                  value={config.temperature}
                  onChange={(e) =>
                    handleConfigChange("temperature", parseFloat(e.target.value))
                  }
                />
                <span className="hint-text">
                  Tonlama çeşitliliği; 0 sabit, yüksek değer oynak.
                </span>
              </div>
              <div className="form-group checkbox">
                <label>
                  <input
                    type="checkbox"
                    checked={config.eq}
                    onChange={(e) => handleConfigChange("eq", e.target.checked)}
                  />
                  Ton filtresi + limiter (EQ)
                </label>
              </div>
            </>
          )}
          {config.model === "ema-lightning" && (
            <span className="hint-text">
              Adım/CFG/sıcaklık/EQ yalnız Antalia Mini içindir; EMA Lightning
              hız + perde + seed ile çalışır.
            </span>
          )}
          <div className="form-group">
            <label>Örnekleme Hızı</label>
            <select
              value={config.sample_rate}
              onChange={(e) => handleConfigChange("sample_rate", parseInt(e.target.value))}
            >
              <option value={24000}>24000 Hz</option>
              <option value={48000}>48000 Hz</option>
            </select>
          </div>
          <div className="form-group">
            <label>Thread Sayısı: {config.threads}</label>
            <input
              type="range"
              min="1"
              max="16"
              step="1"
              value={config.threads}
              onChange={(e) => handleConfigChange("threads", parseInt(e.target.value))}
            />
          </div>
          <div className="form-group checkbox">
            <label>
              <input
                type="checkbox"
                checked={config.use_gpu}
                onChange={(e) => handleConfigChange("use_gpu", e.target.checked)}
              />
              GPU İvmeleme (CUDA)
            </label>
          </div>
          <div className="form-group">
            <label>Chunk Boyutu: {config.chunk_size}</label>
            <input
              type="range"
              min="256"
              max="4096"
              step="256"
              value={config.chunk_size}
              onChange={(e) => handleConfigChange("chunk_size", parseInt(e.target.value))}
            />
          </div>
          <div className="form-group">
            <button
              type="button"
              className="btn btn-secondary"
              onClick={handleResetVoice}
              title="Hız 1.0, ton Doğal (0), seed 0 yapar ve kaydeder"
            >
              Varsayılanlara Dön
            </button>
          </div>
        </section>
        )}

        {activeTab === "api" && (
        <section className="panel panel-wide">
          <h2>HTTP API</h2>
          <div className="form-group checkbox">
            <label>
              <input
                type="checkbox"
                checked={config.api_enabled}
                onChange={(e) => handleApiToggle(e.target.checked)}
              />
              HTTP API'yi etkinleştir (dış programlar metin gönderip sesli okutabilir)
            </label>
          </div>
          <div className="form-group">
            <label>Port (1024 – 65535)</label>
            <div className="input-row">
              <input
                type="number"
                min="1024"
                max="65535"
                step="1"
                value={config.api_port}
                onChange={(e) =>
                  handleConfigChange("api_port", parseInt(e.target.value || "8765"))
                }
              />
              <button type="button" className="btn btn-secondary" onClick={handleApiPortApply}>
                Uygula
              </button>
              {apiStatus?.url && (
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => copyText(apiStatus.url as string)}
                >
                  URL'yi Kopyala
                </button>
              )}
            </div>
          </div>
          <div className="api-status">
            <span className={`dot ${apiStatus?.running ? "on" : "off"}`} />
            {apiStatus?.running
              ? `Dinleniyor: ${apiStatus.url} · Motor: ${apiStatus.engine}${
                  apiStatus.model ? ` · Model: ${apiStatus.model}` : ""
                }${apiStatus.speaking ? " · (konuşuyor)" : ""}`
              : "API kapalı — yalnızca bu bilgisayar erişebilir (127.0.0.1), ağa açılmaz."}
          </div>

          <details className="api-docs">
            <summary>API belgeleri ve örnekler</summary>
            <p className="hint-text">
              Uygulama açıkken dış programlar HTTP ile metin gönderip doğrudan
              sesli okutabilir. Yalnızca bu bilgisayardan erişilir; şifre gerekmez.
            </p>

            <h3>Sesli okut</h3>
            <div className="code-block">
              <button
                type="button"
                className="btn btn-secondary btn-small"
                onClick={() =>
                  copyText(
                    `curl -X POST http://127.0.0.1:${config.api_port}/speak -H "Content-Type: application/json" -d '{"text":"Merhaba, kargonuz yola çıktı.","speed":1.0,"pitch":0,"seed":0}'`
                  )
                }
              >
                Kopyala
              </button>
              <pre>{`curl -X POST http://127.0.0.1:${config.api_port}/speak -H "Content-Type: application/json" -d '{"text":"Merhaba, kargonuz yola çıktı.","speed":1.0,"pitch":0,"seed":0}'`}</pre>
            </div>

            <table className="param-table">
              <thead>
                <tr>
                  <th>Parametre</th>
                  <th>Tür / Aralık</th>
                  <th>Varsayılan</th>
                  <th>Anlamı</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><code>text</code></td>
                  <td>metin (zorunlu, ≤5000 harf)</td>
                  <td>—</td>
                  <td>Okutulacak Türkçe metin. Sayı, tarih, saat ve para otomatik seslendirilir (<code>14:30</code> → "on dört otuz", <code>1.250.000 TL</code> → "bir milyon iki yüz elli bin lira")</td>
                </tr>
                <tr>
                  <td><code>speed</code></td>
                  <td>0.25 – 4.0</td>
                  <td><code>1.0</code></td>
                  <td>Konuşma hızı. <code>1.0</code> normal; <code>0.5</code> yarı hız, <code>2.0</code> iki katı</td>
                </tr>
                <tr>
                  <td><code>pitch</code></td>
                  <td>−4.0 – +4.0 (yarım ton)</td>
                  <td><code>0</code></td>
                  <td><b>Ses tonu/perde.</b> <code>−2.5</code> kalın, <code>0</code> doğal, <code>+3</code> çocuk sesi. Hız değişmez; ±3'ün ötesi yapay duyulur</td>
                </tr>
                <tr>
                  <td><code>seed</code></td>
                  <td>≥0 tam sayı</td>
                  <td><code>0</code></td>
                  <td>Aynı seed aynı sesi verir; farklı seed farklı tonlama varyasyonu</td>
                </tr>
                <tr>
                  <td><code>model</code></td>
                  <td><code>"ema-lightning"</code> / <code>"antalia-mini"</code></td>
                  <td>seçili model</td>
                  <td>Hangi model okutsun. Verilmezse arayüzde seçili model kullanılır (gerekirse otomatik yüklenir)</td>
                </tr>
                <tr>
                  <td><code>steps</code></td>
                  <td>1 – 32 (1/2/4/8 önerilir)</td>
                  <td><code>8</code></td>
                  <td>Yalnız Antalia: akış adımı. EMA yok sayar</td>
                </tr>
                <tr>
                  <td><code>cfg</code></td>
                  <td>0.1 – 8.0</td>
                  <td><code>2.0</code></td>
                  <td>Yalnız Antalia: metne sadakat. EMA yok sayar</td>
                </tr>
                <tr>
                  <td><code>temperature</code></td>
                  <td>0.0 – 2.0</td>
                  <td><code>0.6</code></td>
                  <td>Yalnız Antalia: tonlama çeşitliliği. EMA yok sayar</td>
                </tr>
                <tr>
                  <td><code>eq</code></td>
                  <td><code>true</code> / <code>false</code></td>
                  <td><code>true</code></td>
                  <td>Yalnız Antalia: ton filtresi + limiter. EMA yok sayar</td>
                </tr>
                <tr>
                  <td><code>speak</code></td>
                  <td><code>true</code> / <code>false</code></td>
                  <td><code>true</code></td>
                  <td>Doğrudan seslendir. <code>false</code> ise ses çalınmaz (kaydetmek için <code>save</code> ile birlikte kullanılır)</td>
                </tr>
                <tr>
                  <td><code>save</code></td>
                  <td><code>true</code> / <code>false</code></td>
                  <td><code>false</code></td>
                  <td>WAV olarak kaydet. Klasör: <code>Videolar/Hitabet</code> (yoksa otomatik açılır)</td>
                </tr>
                <tr>
                  <td><code>filename</code></td>
                  <td>metin (isteğe bağlı)</td>
                  <td>zaman damgalı</td>
                  <td>Dosya adı (<code>.wav</code> otomatik eklenir). Örn: <code>"duyuru"</code> → <code>duyuru.wav</code></td>
                </tr>
              </tbody>
            </table>
            <p className="hint-text">
              Yanıt, ses bitince döner: <code>{'{"status":"spoken","seconds":2.12}'}</code>
            </p>

            <h3>Antalia Mini ile okut</h3>
            <div className="code-block">
              <button
                type="button"
                className="btn btn-secondary btn-small"
                onClick={() =>
                  copyText(
                    `curl -X POST http://127.0.0.1:${config.api_port}/speak -H "Content-Type: application/json" -d '{"text":"Merhaba.","model":"antalia-mini","steps":8,"cfg":2.0,"temperature":0.6}'`
                  )
                }
              >
                Kopyala
              </button>
              <pre>{`curl -X POST http://127.0.0.1:${config.api_port}/speak -H "Content-Type: application/json" -d '{"text":"Merhaba.","model":"antalia-mini","steps":8,"cfg":2.0,"temperature":0.6}'`}</pre>
            </div>

            <h3>Konuşmadan kaydet</h3>
            <div className="code-block">
              <button
                type="button"
                className="btn btn-secondary btn-small"
                onClick={() =>
                  copyText(
                    `curl -X POST http://127.0.0.1:${config.api_port}/speak -H "Content-Type: application/json" -d '{"text":"Merhaba.","speak":false,"save":true,"filename":"karsilama"}'`
                  )
                }
              >
                Kopyala
              </button>
              <pre>{`curl -X POST http://127.0.0.1:${config.api_port}/speak -H "Content-Type: application/json" -d '{"text":"Merhaba.","speak":false,"save":true,"filename":"karsilama"}'`}</pre>
            </div>
            <p className="hint-text">
              Yanıt, dosyanın açık yolunu da içerir:{" "}
              <code>{'{"status":"saved","seconds":1.24,"file":"C:/Users/Ad/Videos/Hitabet/karsilama.wav","file_url":"file:///C:/Users/Ad/Videos/Hitabet/karsilama.wav"}'}</code>
            </p>

            <h3>Durdur</h3>
            <div className="code-block">
              <button
                type="button"
                className="btn btn-secondary btn-small"
                onClick={() => copyText(`curl -X POST http://127.0.0.1:${config.api_port}/stop`)}
              >
                Kopyala
              </button>
              <pre>{`curl -X POST http://127.0.0.1:${config.api_port}/stop`}</pre>
            </div>
            <p className="hint-text">
              Çalan/üretilen sesi anında keser (arayüzdeki Durdur ile aynı).
              Yanıt: <code>{'{"status":"stopped"}'}</code>
            </p>

            <h3>Durum</h3>
            <div className="code-block">
              <button
                type="button"
                className="btn btn-secondary btn-small"
                onClick={() => copyText(`curl http://127.0.0.1:${config.api_port}/status`)}
              >
                Kopyala
              </button>
              <pre>{`curl http://127.0.0.1:${config.api_port}/status`}</pre>
            </div>
            <table className="param-table">
              <thead>
                <tr>
                  <th>Alan</th>
                  <th>Anlamı</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><code>engine</code></td>
                  <td><code>ready</code> = model yüklü, <code>cold</code> = henüz yüklenmedi (ilk istekte otomatik yüklenir)</td>
                </tr>
                <tr>
                  <td><code>model</code></td>
                  <td>Yüklü model (<code>"ema-lightning"</code> / <code>"antalia-mini"</code>, yoksa alan çıkmaz)</td>
                </tr>
                <tr>
                  <td><code>speaking</code></td>
                  <td>Şu an ses çalınıyor/üretiliyor mu</td>
                </tr>
                <tr>
                  <td><code>queue_secs</code></td>
                  <td>Sırada bekleyen sesin süresi (sn)</td>
                </tr>
                <tr>
                  <td><code>port</code></td>
                  <td>Dinlenen port</td>
                </tr>
              </tbody>
            </table>
            <p className="hint-text">
              Hatalar: boş metin → <code>400</code>, çok uzun metin → <code>413</code>,
              model bulunamazsa → <code>503</code> (açıklamalı mesajla).
            </p>
          </details>
        </section>
        )}

      </main>
    </div>
  );
}
