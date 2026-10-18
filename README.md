
EMA Lightning : [output_1791712412039.wav](https://github.com/user-attachments/files/33300596/output_1791712412039.wav)
Antalia Mini  : [output_1791712468199.wav](https://github.com/user-attachments/files/33300605/output_1791712468199.wav)

<img width="954" height="845" alt="image" src="https://github.com/user-attachments/assets/75a85cd2-c79e-4948-9b17-5eea423ada9e" />




<img width="976" height="857" alt="image" src="https://github.com/user-attachments/assets/db5ab57d-8a3d-4d39-93b5-dff4526ce147" />
<img width="958" height="604" alt="image" src="https://github.com/user-attachments/assets/d5e16556-0603-4d9d-9f41-b873c686e74f" />
<img width="958" height="844" alt="image" src="https://github.com/user-attachments/assets/fbb92962-ae03-41a5-bf97-6ad2764357c5" />
<img width="958" height="856" alt="image" src="https://github.com/user-attachments/assets/3b1e8107-1ef1-44a3-950a-0af28fbd0b1f" />


# Hitabet-Turkce-TTS

Çevrimdışı çalışan, üç modelli Türkçe metin-konuşma (TTS) masaüstü stüdyosu. İnternet gerekmez: modeller uygulamayla birlikte gelir, seslendirme tamamen cihazda yapılır.

- **Üç model, tek arayüz:** EMA Lightning (hızlı, akıcı) + Antalia Mini (doğal tonlama, ayarlanabilir adımlar) + Ahenk Cep (küçük ve hafif, 7 duygu üslubu)
- **Taşınabilir (portable):** kurulum yok, exe yanına ayarlar (`settings.json`) ve ses profilleri (`presets.json`) yazılır
- **Türkçe'ye özel:** sayı/tarih/saat/para genişletme, geminasyon, cümle-içi duraklamalar
- **Yerel HTTP API:** dış programlar `POST /speak` ile okutabilir (`127.0.0.1`, varsayılan kapalı)
- **Ses profilleri:** hız/perde/seed + model ayarlarını isimle kaydedip tek tıkla uygulama
- **4 görünüm:** Koyu / Açık / Sepya / Kontrast

**Hazır kurulum:** [Releases](../../releases) sayfasından güncel `Hitabet_*_x64-setup.exe` indirilir; üç model gömülü gelir.

## Kullanılan Modeller ve Teşekkür

Bu uygulama aşağıdaki açık modellerin ONNX'e dönüştürülmüş halleriyle çalışır. Modellerin sahiplerine ve emeği geçenlere teşekkürler:

| Model | Kaynak | Lisans |
|---|---|---|
| **EMA Lightning** (`ema.pt` + `decoder.pt`) | [canberkkkkkk/ema-lightning](https://huggingface.co/canberkkkkkk/ema-lightning) | Apache-2.0 |
| **Antalia Mini** (`acoustic` + `vocoder` safetensors) | [cloud0day3/antalia-mini](https://huggingface.co/cloud0day3/antalia-mini) | Apache-2.0 |
| **Ahenk Cep** (`onnx/` hazır) | [ozgurguler/ahenk-cep-tr](https://huggingface.co/ozgurguler/ahenk-cep-tr) | CC-BY-4.0 |

Ayrıca EMA Lightning'in referans implementasyonu için [canberk7/ema-lightning](https://github.com/canberk7/ema-lightning), Ahenk Cep kodu için [ozguradmin/ahenk-turkish-tts](https://github.com/ozguradmin/ahenk-turkish-tts) reposuna teşekkürler.

> Not: HuggingFace repolarında yalnızca ham ağırlıklar (`.pt`/`.safetensors`) bulunur; EMA ve Antalia'nın ONNX dosyaları `scripts/export_onnx.py` ve `scripts/export_antalia_onnx.py` ile yerelde üretilir (aşağıya bakın). Ahenk Cep ONNX olarak yayında olduğundan `scripts/fetch_ahenk_onnx.py` ile indirilir. Bu repo lisansı (MIT) uygulama kaynak kodunu kapsar; model ağırlıkları kendi lisanslarına (Apache-2.0 ve CC-BY-4.0) tabidir. Ahenk Cep **CC-BY-4.0** olduğundan atıf zorunludur; yukarıdaki tablo bu atfı sağlar.

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

# 3) Ahenk Cep ONNX indirimi (~34 MB, tests: models/ahenk; sha256 doğrulamalı)
python scripts/fetch_ahenk_onnx.py --out models/ahenk

# 4) Geliştirme / üretim
npm run tauri dev      # geliştirme
npm run tauri build    # paketler (NSIS/MSI/portable dizini)
```

Notlar:
- `export_*.py` betikleri ağırlıkları HuggingFace'ten indirir, ONNX'e çevirir ve PyTorch çıktısıyla sayısal olarak doğrular; `fetch_ahenk_onnx.py` hazır ONNX'i indirip sha256 doğrular (`cargo test --lib` parite testleri: ~1e-5 mertebesi).
- CUDA/GPU yalnızca NVIDIA + uyumlu ONNX Runtime derlemesinde; varsayılan CPU'dur ve üç model de CPU'da gerçek zamanlı çalışır.
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
│  - ahenk_engine.rs: Ahenk Cep inference (+ üslup)    │
│  - turkish_tokenizer.rs / antalia_text.rs: normaliz. │
│  - lib.rs: motor seçimi, API, indirici, profiller    │
└──────────────────────┬──────────────────────────────┘
                       │ ort crate (ONNX Runtime)
┌──────────────────────▼──────────────────────────────┐
│  Modeller (tamamı cihazda)                          │
│  - models/onnx: text_encoder/sound_stage/decoder     │
│  - models/antalia: text/prep/step/vocoder            │
│  - models/ahenk: text/cond/step/decoder              │
└─────────────────────────────────────────────────────┘
```

## Konfigürasyon

`settings.json` (exe yanı, taşınabilir) veya UI üzerinden:

| Parametre     | Tip    | Varsayılan | Açıklama                    |
|---------------|--------|------------|-----------------------------|
| `model`       | metin  | `ema-lightning` | Seçili model (`antalia-mini` / `ahenk-cep` de olur) |
| `speed`       | float  | 1.0        | Konuşma hızı (0.25x - 4.0x) |
| `sample_rate` | int    | 48000      | Örnekleme hızı (Hz)        |
| `threads`     | int    | 4          | ONNX Runtime thread sayısı |
| `use_gpu`     | bool   | false      | CUDA ivmelendirme           |
| `chunk_size`  | int    | 1024       | Streaming buffer boyutu    |
| `seed`        | int    | 0          | Ses tohumu (aynı seed aynı ses) |
| `steps`       | int    | 8          | Antalia: akış adımı (1/2/4/8). Ahenk Cep: Euler adımı (16 önerilir) |
| `cfg`         | float  | 2.0        | Antalia: metne sadakat. Ahenk Cep: üslup gücü (2.0) |
| `temperature` | float  | 0.6        | Yalnız Antalia: tonlama çeşitliliği |
| `eq`          | bool   | true       | Yalnız Antalia: ton filtresi + limiter |
| `style`       | metin  | `neutral`  | Yalnız Ahenk Cep: üslup |
| `theme`       | metin  | `dark`     | Görünüm (`light`/`sepia`/`contrast`) |

## ONNX Tensor Boyutları

| Model                  | Input | Output |
|------------------------|-------|--------|
| `text_encoder.onnx`    | `ids (B,L) int64`, `mask (B,L) bool` | `h (B,L,224) f32`, `dur (B,L) f32` |
| `sound_stage.onnx`     | `h`, `dur/speed`, `mask`, `cw`/`wstart`/`fw` int64, `fp` f32, `fmask` bool, `noise (B,4,T,64)` | `latents (B,T,64) f32` |
| `decoder.onnx`         | `z (B,64,T) f32`, `lengths (B,) int64` | `audio (B,T*1920) f32` @48kHz |

Artı `vocab.json` (49 token). Antalia tarafı: `antalia_text/prep/step/vocoder.onnx` + `symbols.json` (47 token) + `null_cond.f32`; ayrıntılar `scripts/export_antalia_onnx.py` içindedir.

Ahenk Cep (EMA mimarisi + üslup gömme, aynı 49 token; Euler döngüsü Rust'ta):

| Model             | Input | Output |
|-------------------|-------|--------|
| `text.onnx`       | `ids (1,L) int64`, `style (1,) int64` | `h (1,L,224) f32`, `dur (1,L) f32` |
| `cond.onnx`       | `h`, `dur`, `cw`/`wstart`/`fw` int64, `fp` f32 | `cond (1,T,224) f32` |
| `step.onnx`       | `x (1,T,64)`, `cond`, `t`/`dt`/`cfg` f32 | `x_next (1,T,64) f32` (tek Euler adımı) |
| `decoder.onnx`    | `z (1,64,F) f32` | `audio (1,F*1920) f32` @48kHz |

Üsluplar: `neutral`, `angry`, `calm`, `cheerful`, `empathetic`, `sad`, `surprised` (Türkçe adlar da geçerli: `notr`/`kizgin`/…). `laughter` eğitilmemiştir.

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

# Ahenk Cep ile okut (üslup: cheerful)
curl -X POST http://127.0.0.1:8765/speak `
  -H "Content-Type: application/json" `
  -d '{"text":"Harika haber!","model":"ahenk-cep","style":"cheerful","steps":16,"cfg":2.0}'

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
| `model` | `ema-lightning` / `antalia-mini` / `ahenk-cep` | seçili model | Hangi model okutsun |
| `speed` | 0.25 – 4.0 | `1.0` | Konuşma hızı |
| `pitch` | −4.0 – +4.0 (yarım ton) | `0` | Ses tonu. Hız değişmez |
| `seed` | ≥0 | `0` | Aynı seed aynı sesi verir |
| `steps` / `cfg` | model aralığı | `8` / `2.0` (Antalia), `16` / `2.0` (Ahenk Cep) | Antalia akış / Ahenk Cep Euler adımı + üslup gücü (EMA yok sayar) |
| `temperature` / `eq` | — | `0.6` / `true` | Yalnız Antalia (diğerleri yok sayar) |
| `style` | 7 üslup adı | `neutral` | Yalnız Ahenk Cep (diğerleri yok sayar) |
| `speak` | true/false | `true` | Doğrudan seslendir |
| `save` | true/false | `false` | WAV olarak kaydet. Klasör: `Videolar/Hitabet` |
| `filename` | metin (opsiyonel) | zaman damgalı | Dosya adı (`.wav` otomatik eklenir) |

Hatalar: boş metin → `400`, çok uzun → `413`, model yoksa → `503`.

## Modeller (çok modelli yapı ve çevrimdışı onarım)

Uygulama üç Türkçe TTS modelini aynı arayüzden çalıştırır; `models/` kökünün
altında `onnx/` (EMA Lightning), `antalia/` (Antalia Mini) ve `ahenk/` (Ahenk Cep)
klasörleri beklenir. Kurulum (NSIS/MSI) üç modeli de gömülü getirir; normal
kullanımda internet gerekmez.

Bir dosya silinir/bozulursa, **Model Ayarları** kartındaki **Kur** tek tıkla
kurulu dizindeki `models/` altına kurar/doğrular (klasör sormaz); **Klasörden**
USB'deki ya da yedek klasörden onarır (sha256 doğrulamalı). **Denetle**
yerel dosyaları gömülü listeyle karşılaştırır (çevrimdışı bütünlük denetimi).
Model kartları yan yana gösterilir; dar pencerede otomatik olarak alt alta sarar.

## Sürüm Geçmişi

- **v0.3.0** — Ahenk Cep (üçüncü model, 7 duygu üslubu) eklendi; model kartları
  yan yana ve daha kompakt (Kur / Klasörden / Denetle).
- **v0.2.0** — Başlıkta sürüm rozeti; uzun metinlerde "Durdur" pasif kalma hatası
  düzeltildi (geri sayım + takılma dedektörü).
- **v0.1.0** — İlk sürüm: iki modelli çevrimdışı Türkçe TTS (EMA Lightning + Antalia Mini).

## Lisans

MIT — uygulama kaynak kodu için; ayrıntı için `LICENSE` dosyasına bakın.
Model ağırlıkları kendi lisanslarına tabidir: EMA Lightning ve Antalia Mini
**Apache-2.0**, Ahenk Cep **CC-BY-4.0** (atıf zorunlu).
