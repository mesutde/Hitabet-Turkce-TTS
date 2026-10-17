// Tasinabilir kullanim icin guncel modelleri exe'nin yanina kopyalar.
// Calistir: npm run sync-models   (tauri build oncesinde otomatik de calisir)
import { rmSync, cpSync, existsSync, mkdirSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
// Faz C: uc model de paketlenir (models/ kok yapisi korunur).
const subdirs = ["onnx", "antalia", "ahenk"];
for (const profile of ["release", "debug"]) {
  for (const sub of subdirs) {
    const src = join(root, "models", sub);
    const dest = join(root, "src-tauri", "target", profile, "models", sub);
    if (!existsSync(src)) {
      console.warn(`[sync-models] kaynak yok, atlaniyor: ${src}`);
      continue;
    }
    mkdirSync(join(root, "src-tauri", "target", profile, "models"), { recursive: true });
    rmSync(dest, { recursive: true, force: true });
    cpSync(src, dest, { recursive: true });
    console.log(`[sync-models] ${profile}/${sub}: modeller tazelendi`);
  }
}
