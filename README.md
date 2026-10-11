# Hitabet-Turkce-TTS

Çevrimdışı çalışan, iki modelli Türkçe metin-konuşma (TTS) masaüstü stüdyosu. İnternet gerekmez: modeller uygulamayla birlikte gelir, seslendirme tamamen cihazda yapılır.

- **İki model, tek arayüz:** EMA Lightning (hızlı, akıcı) + Antalia Mini (doğal tonlama, ayarlanabilir adımlar)
- **Taşınabilir (portable):** kurulum yok, exe yanına ayarlar (`settings.json`) ve ses profilleri (`presets.json`) yazılır
- **Türkçe'ye özel:** sayı/tarih/saat/para genişletme, geminasyon, cümle-içi duraklamalar
- **Yerel HTTP API:** dış programlar `POST /speak` ile okutabilir (`127.0.0.1`, varsayılan kapalı)
- **Ses profilleri:** hız/perde/seed + model ayarlarını isimle kaydedip tek tıkla uygulama
- **4 görünüm:** Koyu / Açık / Sepya / Kontrast

**Hazır kurulum:** [Releases](../../releases) sayfasından güncel `Hitabet_*_x64-setup.exe` indirilir; iki model gömülü gelir.

## Kullanılan Modeller ve Teşekkür

Bu uygulama aşağıdaki açık modellerin ONNX'e dönüştürülmüş halleriyle çalışır. Modellerin sahiplerine ve emeği geçenlere teşekkürler:

| Model | Kaynak | Lisans |
|---|---|---|
| **EMA Lightning** (`ema.pt` + `decoder.pt`) | [canberkkkkkk/ema-lightning](https://huggingface.co/canberkkkkkk/ema-lightning) | Apache-2.0 |
| **Antalia Mini** (`acoustic` + `vocoder` safetensors) | [cloud0day3/antalia-mini](https://huggingface.co/cloud0day3/antalia-mini) | Apache-2.0 |

Ayrıca EMA Lightning'in referans implementasyonu için [canberk7/ema-lightning](https://github.com/canberk7/ema-lightning) reposuna teşekkürler.

> Not: HuggingFace repolarında yalnızca ham ağırlıklar (`.pt`/`.safetensors`) bulunur; uygulamanın çalıştırdığı ONNX dosyaları `scripts/export_onnx.py` ve `scripts/export_antalia_onnx.py` ile yerelde üretilir (aşağıya bakın). Bu repo lisansı (MIT) uygulama kaynak kodunu kapsar; model ağırlıkları kendi lisanslarına (Apache-2.0) tabidir.

## Kaynaktan Derleme

### Gereksinimler
- [Node.js](https://nodejs.org/) 18+
- [Rust](https://rustup.rs/) 1.77+
- Windows'ta [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/); Linux'ta `build-essential` + WebKitGTK (`libwebkit2gtk-4.1-dev` vb., ayrıntı için [Tauri önkoşulları](https://v2.tauri.app/start/prerequisites/)); macOS'ta Xcode Command Line Tools
- Model export için (bir kez): Python 3.10+ ve `pip install torch onnx onnxruntime onnxscript huggingface_hub`

### Adımlar (Windows / Linux / macOS ortak)

```bash
npm install

# 1) EMA Lightning ONNX üretimi (~40 MB, tests: models/onnx)
pip install torch onnx onnxruntime onnxscript huggingface_hub
python scripts/export_onnx.py --output_dir models/onnx

# 2) Antalia Mini ONNX üretimi (~49 MB, tests: models/antalia)
python scripts/export_antalia_onnx.py --output_dir models/antalia

# 3) Geliştirme / üretim
npm run tauri dev      # geliştirme
npm run tauri build    # paketler (NSIS/MSI/portable dizini)
```

Notlar:
- `export_*.py` betikleri ağırlıkları HuggingFace'ten indirir, ONNX'e çevirir ve PyTorch çıktısıyla sayısal olarak doğrular (`cargo test --lib` parite testleri: ~1e-6 mertebesi).
- CUDA/GPU yalnızca NVIDIA + uyumlu ONNX Runtime derlemesinde; varsayılan CPU'dur ve iki model de CPU'da gerçek zamanlı çalışır.
- Linux/macOS paketleri Tauri'nin ilgili hedeflerini üretir (`deb`/`AppImage`, `dmg`); bu repo Windows portable akışına odaklanır, ancak kod platformdan bağımsızdır (cpal/rubato/ort hepsi çok platformludur).

## Mimari

```
┌─────────────────────────────────────────────────────┐
│  Tauri v2 UI (React + TypeScript)                   │
│  - Metin girişi, model kartları, ses profilleri      │
│  - Event tabanlı streaming (audio-chunk)             │
└──────────────────────┬──────────────────────────────┘
                       │ Tauri Commands / Events
┌──────────────────────▼──────────────────────────────┐
│  Rust Backend (src-tauri)                           │
│  - tts_engine.rs: EMA Lightning inference            │
│  - antalia_engine.rs: Antalia Mini inference         │
│  - turkish_tokenizer.rs / antalia_text.rs: normaliz. │
│  - lib.rs: motor seçimi, API, indirici, profiller    │
└──────────────────────┬──────────────────────────────┘
                       │ ort crate (ONNX Runtime)
┌──────────────────────▼──────────────────────────────┐
│  Modeller (48 kHz, tamamı cihazda)                  │
│  - models/onnx: text_encoder/sound_stage/decoder     │
│  - models/antalia: text/prep/step/vocoder            │
└─────────────────────────────────────────────────────┘
```

## Konfigürasyon

`settings.json` (exe yanı, taşınabilir) veya UI üzerinden:

| Parametre     | Tip    | Varsayılan | Açıklama                    |
|---------------|--------|------------|-----------------------------|
| `model`       | metin  | `ema-lightning` | Seçili model (`antalia-mini` de olur) |
| `speed`       | float  | 1.0        | Konuşma hızı (0.25x - 4.0x) |
| `sample_rate` | int    | 48000      | Örnekleme hızı (Hz)        |
| `threads`     | int    | 4          | ONNX Runtime thread sayısı |
| `use_gpu`     | bool   | false      | CUDA ivmelendirme           |
| `chunk_size`  | int    | 1024       | Streaming buffer boyutu    |
| `seed`        | int    | 0          | Ses tohumu (aynı seed aynı ses) |
| `steps`       | int    | 8          | Yalnız Antalia: akış adımı (1/2/4/8) |
| `cfg`         | float  | 2.0        | Yalnız Antalia: metne sadakat |
| `temperature` | float  | 0.6        | Yalnız Antalia: tonlama çeşitliliği |
| `eq`          | bool   | true       | Yalnız Antalia: ton filtresi + limiter |
| `theme`       | metin  | `dark`     | Görünüm (`light`/`sepia`/`contrast`) |

## ONNX Tensor Boyutları

| Model                  | Input | Output |
|------------------------|-------|--------|
| `text_encoder.onnx`    | `ids (B,L) int64`, `mask (B,L) bool` | `h (B,L,224) f32`, `dur (B,L) f32` |
| `sound_stage.onnx`     | `h`, `dur/speed`, `mask`, `cw`/`wstart`/`fw` int64, `fp` f32, `fmask` bool, `noise (B,4,T,64)` | `latents (B,T,64) f32` |
| `decoder.onnx`         | `z (B,64,T) f32`, `lengths (B,) int64` | `audio (B,T*1920) f32` @48kHz |

Artı `vocab.json` (49 token). Antalia tarafı: `antalia_text/prep/step/vocoder.onnx` + `symbols.json` (47 token) + `null_cond.f32`; ayrıntılar `scripts/export_antalia_onnx.py` içindedir.

## HTTP API (yerel)

Uygulama açıkken dış programlar HTTP ile metin gönderip doğrudan sesli okutabilir.
Arayüzdeki **HTTP API** panelinden açılır/kapanır (varsayılan: kapalı, port `8765`).
Yalnızca bu bilgisayardan erişilir (`127.0.0.1`), ağa açılmaz. Ayarlar `settings.json`
ile kalıcıdır; `ema-lightning-tts.exe --api [--api-port 8765]` ile headless başlatılabilir.

```powershell
# Sesli okut (ses bitince döner): text zorunlu, diğerleri opsiyonel
curl -X POST http://127.0.0.1:8765/speak `
  -H "Content-Type: application/json" `
  -d '{"text":"Merhaba, kargonuz yola çıktı.","speed":1.0,"pitch":0,"seed":0}'
# -> {"status":"spoken","seconds":2.12}

# Antalia Mini ile okut
curl -X POST http://127.0.0.1:8765/speak `
  -H "Content-Type: application/json" `
  -d '{"text":"Merhaba.","model":"antalia-mini","steps":8,"cfg":2.0,"temperature":0.6}'

# Konuşmadan kaydet (dosya yolu sonuç json'da döner)
curl -X POST http://127.0.0.1:8765/speak `
  -H "Content-Type: application/json" `
  -d '{"text":"Merhaba.","speak":false,"save":true,"filename":"karsilama"}'
# -> {"status":"saved","seconds":1.24,
#     "file":"C:/Users/Ad/Videos/Hitabet/karsilama.wav",
#     "file_url":"file:///C:/Users/Ad/Videos/Hitabet/karsilama.wav"}

# Durdur (çalan/üretilen sesi keser)
curl -X POST http://127.0.0.1:8765/stop
# -> {"status":"stopped"}

# Durum / sağlık
curl http://127.0.0.1:8765/status
# -> {"engine":"ready","model":"ema-lightning","speaking":false,"queue_secs":0.0,"port":8765}
curl http://127.0.0.1:8765/health
```

| Parametre | Aralık | Varsayılan | Anlamı |
|---|---|---|---|
| `text` | ≤5000 harf (zorunlu) | — | Okutulacak metin; sayı/tarih/saat/para otomatik seslendirilir |
| `model` | `ema-lightning` / `antalia-mini` | seçili model | Hangi model okutsun |
| `speed` | 0.25 – 4.0 | `1.0` | Konuşma hızı |
| `pitch` | −4.0 – +4.0 (yarım ton) | `0` | Ses tonu. Hız değişmez |
| `seed` | ≥0 | `0` | Aynı seed aynı sesi verir |
| `steps` / `cfg` / `temperature` / `eq` | model aralığı | `8` / `2.0` / `0.6` / `true` | Yalnız Antalia (EMA yok sayar) |
| `speak` | true/false | `true` | Doğrudan seslendir |
| `save` | true/false | `false` | WAV olarak kaydet. Klasör: `Videolar/Hitabet` |
| `filename` | metin (opsiyonel) | zaman damgalı | Dosya adı (`.wav` otomatik eklenir) |

Hatalar: boş metin → `400`, çok uzun → `413`, model yoksa → `503`.

## Modeller (çok modelli yapı ve çevrimdışı onarım)

Uygulama iki Türkçe TTS modelini aynı arayüzden çalıştırır; `models/` kökünün
altında `onnx/` (EMA Lightning) ve `antalia/` (Antalia Mini) klasörleri beklenir.
Kurulum (NSIS/MSI) iki modeli de gömülü getirir; normal kullanımda internet gerekmez.

Bir dosya silinir/bozulursa, **Model Ayarları** kartındaki **Modeli Kur** tek tıkla
kurulu dizindeki `models/` altına kurar/doğrular (klasör sormaz); **Klasörden Kur**
USB'deki ya da yedek klasörden onarır (sha256 doğrulamalı). **Güncellemeleri Denetle**
yerel dosyaları gömülü listeyle karşılaştırır (çevrimdışı bütünlük denetimi).

## Lisans

MIT — ayrıntı için `LICENSE` dosyasına bakın.
