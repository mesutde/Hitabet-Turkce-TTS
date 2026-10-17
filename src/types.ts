export type ModelId = "ema-lightning" | "antalia-mini" | "ahenk-cep";

export interface TtsConfig {
  /** Seçili model (settings.json ile kalıcı). */
  model: ModelId;
  speed: number;
  sample_rate: number;
  threads: number;
  use_gpu: boolean;
  chunk_size: number;
  seed: number;
  /** Ses perdesi, yarım ton. Hız değişmez. */
  pitch: number;
  /** HTTP API sunucusu açık mı. */
  api_enabled: boolean;
  /** HTTP API portu. */
  api_port: number;
  /** Ses çıkış aygıtı adı; boş ise sistem varsayılanı. */
  output_device: string;
  /** Antalia: akış adımı (1/2/4/8). EMA yok sayar. */
  steps: number;
  /** Antalia: CFG ölçeği. EMA yok sayar. */
  cfg: number;
  /** Antalia: gürültü sıcaklığı. EMA yok sayar. */
  temperature: number;
  /** Antalia: ton filtresi + limiter. EMA yok sayar. */
  eq: boolean;
  /** Ahenk Cep: üslup (neutral | angry | calm | cheerful | empathetic | sad | surprised).
   *  Diğer modeller yok sayar. */
  style: string;
  /** Arayüz teması: dark | light | sepia | contrast. Sentezi etkilemez. */
  theme: string;
}

export interface ApiStatus {
  running: boolean;
  port: number;
  url: string | null;
  engine: string;
  /** Yüklü model (yoksa null = soğuk). */
  model: ModelId | null;
  speaking: boolean;
}

export interface SavedSettings {
  model_dir: string;
  config: TtsConfig;
}

export interface ModelStatus {
  id: ModelId;
  name: string;
  ready: boolean;
  missing: string[];
  dir: string;
  size_mb: number;
}

export interface DownloadProgress {
  model: ModelId;
  file: string;
  index: number;
  total: number;
  done: number;
  expected: number;
}

export interface VoicePreset {
  name: string;
  created_ms: number;
  config: TtsConfig;
}
