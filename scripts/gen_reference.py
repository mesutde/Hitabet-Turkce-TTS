#!/usr/bin/env python3
"""Rust motoru icin referans verisi uretir (models/ref/).

Kaydedilenler (raw little-endian + shapes.json):
  ids.i64, h.f32, dur.f32, fw.i64, fp.f32, latents.f32, audio_full.f32
"""
import json
from pathlib import Path

import numpy as np
import torch

TEXT = "Merhaba, size nasıl yardımcı olabilirim?"


def save(path: Path, arr: np.ndarray):
    path.write_bytes(np.ascontiguousarray(arr).tobytes())


def main():
    from huggingface_hub import snapshot_download
    from ema_lightning.model import load_acoustic
    from ema_lightning.decoder import load_decoder
    from ema_lightning.frontend import Frontend
    from ema_lightning.engine import Engine

    snap = Path(snapshot_download(
        "canberkkkkkk/ema-lightning", allow_patterns=["ema.pt", "decoder.pt"]))
    acoustic = load_acoustic(str(snap / "ema.pt"), "cpu")
    decoder = load_decoder(str(snap / "decoder.pt"), "cpu")
    ck = torch.load(str(snap / "ema.pt"), map_location="cpu", weights_only=True)
    frontend = Frontend(list(ck["vocab"]))

    engine = Engine(acoustic, decoder, "cpu")
    piece = engine.piece(frontend(TEXT), pause=0.0, seed=0)
    engine.plan([piece], speed=1.0)
    engine.think([piece])

    L, T = piece.letters, piece.frames
    print(f"text={frontend(TEXT)!r} L={L} T={T}")

    z = piece.latents.unsqueeze(0).transpose(1, 2).contiguous()
    with torch.no_grad():
        audio_full = decoder(z, None)[0].numpy()

    out = Path("models/ref")
    out.mkdir(parents=True, exist_ok=True)
    save(out / "ids.i64", piece.ids.numpy().astype(np.int64))
    save(out / "h.f32", piece.h.numpy().astype(np.float32))
    save(out / "dur.f32", piece.dur.numpy().astype(np.float32))
    save(out / "fw.i64", piece.fw.numpy().astype(np.int64))
    save(out / "fp.f32", piece.fp.numpy().astype(np.float32))
    save(out / "latents.f32", piece.latents.numpy().astype(np.float32))
    save(out / "audio_full.f32", audio_full.astype(np.float32))
    (out / "shapes.json").write_text(json.dumps(
        {"L": L, "T": T, "text": frontend(TEXT)}, ensure_ascii=False))
    print(f"referans yazildi: {out.absolute()} audio_len={len(audio_full)}")


if __name__ == "__main__":
    main()
