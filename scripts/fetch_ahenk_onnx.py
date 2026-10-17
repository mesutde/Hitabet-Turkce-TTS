"""Ahenk Cep (ozgurguler/ahenk-cep-tr) ONNX dosyalarini indirir.

Kullanici sozlesmesi: model ONNX olarak yayinda oldugu icin export gerekmez;
yalniz indirme + sha256 dogrulama yapilir. `vocab.json`, `ahenk-cep.json`
icindeki `vocab` alanindan cikarilir (49 token).

Kullanim: python scripts/fetch_ahenk_onnx.py [--out models/ahenk]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = "ozgurguler/ahenk-cep-tr"
BASE = f"https://huggingface.co/{REPO}/resolve/main"

# ad -> (hf yolu, boyut, sha256)
FILES = {
    "ahenk-cep.json": ("onnx/ahenk-cep.json", 2628,
                       "5df4d66c71f1dd9442fd973f3b8637861243f7b01e3d17f44cf76c53f3e3d2f6"),
    "text.onnx": ("onnx/text.onnx", 4815801,
                  "9a1f49d898d485ef1226a6cde101a5da69c8d7ef4fb5891132a882eb6eaa253a"),
    "cond.onnx": ("onnx/cond.onnx", 829432,
                  "a9cfe2ed7793c210af98dc859a8e1e2cf02d81cb827f4a545a92f3666c96056c"),
    "step.onnx": ("onnx/step.onnx", 16801473,
                  "8f772fb89815e6cc71b23b753a25f7c79cc4280af6c3edb3ee3b2c19dfc07acb"),
    "decoder.onnx": ("onnx/decoder.onnx", 12058685,
                     "f4fc10af62b84a867aff3fe4daf3f962f479dc5b9a809d860f3adc0050f381bb"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    print(f"indiriliyor: {url}")
    with urllib.request.urlopen(url) as r, dest.open("wb") as f:
        f.write(r.read())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "models" / "ahenk"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for name, (remote, size, digest) in FILES.items():
        dest = out / name
        if dest.exists() and dest.stat().st_size == size and sha256(dest) == digest:
            print(f"zaten saglam: {name}")
            continue
        download(f"{BASE}/{remote}", dest)
        got = sha256(dest)
        if got != digest:
            raise SystemExit(f"sha256 uyusmadi: {name}\n beklenen {digest}\n gelen    {got}")
        print(f"tamam: {name} ({size} bayt)")

    cfg = json.loads((out / "ahenk-cep.json").read_text(encoding="utf-8"))
    vocab = cfg["vocab"]
    if len(vocab) != 49:
        raise SystemExit(f"beklenmeyen vocab uzunlugu: {len(vocab)}")
    # write_bytes: Windows'ta LF'\nin CRLF'e cevrilmesini onler (katalog boyutu/sha256 sabit).
    (out / "vocab.json").write_bytes(
        (json.dumps(vocab, ensure_ascii=False) + "\n").encode("utf-8")
    )
    print(f"vocab.json yazildi ({len(vocab)} token)")
    print("hazir:", out)


if __name__ == "__main__":
    main()
