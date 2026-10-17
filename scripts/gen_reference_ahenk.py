"""Ahenk Cep referans cikarim: ONNX hatlarini Python'da kosup Rust testi
icin ara sonuclari (ids/h/dur/fw/cond/noise/latents/audio) diske yazar.

Kullanim: python scripts/gen_reference_ahenk.py [--out models/ahenk_ref]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "models" / "ahenk"

SPOKEN = "merhaba dünya bugün hava çok güzel"  # rakam/kisaltma yok: front-end fark etmez
STYLE = 0          # neutral
STEPS = 16
CFG = 2.0
SPEED = 1.0


def build_plan(ids, dur, speed, max_word_frames, max_frames):
    L = len(ids)
    SPACE = 2
    starts = [i for i in range(L) if ids[i] != SPACE and (i == 0 or ids[i - 1] == SPACE)] or [0]
    bounds = [0] + starts[1:] + [L]
    cw = [0] * L
    ws = [0] * L
    for w, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        for i in range(a, b):
            cw[i] = w
            ws[i] = a
    nw = max(cw) + 1
    wf = []
    for w in range(nw):
        s = sum(dur[i] / speed for i in range(L) if cw[i] == w)
        wf.append(int(min(max(round(s), 1), max_word_frames)))
    total = min(sum(wf), max_frames)
    cum = [0]
    for x in wf:
        cum.append(cum[-1] + x)
    fw, fp = [], []
    for f in range(total):
        w = nw - 1
        for ww in range(nw):
            if f < cum[ww + 1]:
                w = ww
                break
        fw.append(w)
        fp.append((f - cum[w]) / wf[w])
    return cw, ws, fw, fp, total


def windows(T, first=25, window=100):
    out, s, fst = [], 0, True
    while s < T:
        e = min(s + (first if fst else window), T)
        out.append((s, e))
        s, fst = e, False
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "models" / "ahenk_ref"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((D / "ahenk-cep.json").read_text(encoding="utf-8"))
    vocab = cfg["vocab"]
    stoi = {c: i for i, c in enumerate(vocab) if len(c) == 1}
    HOP = cfg["hop"]
    LAT = cfg["latent_dim"]

    ids = np.array([[stoi.get(c, 1) for c in SPOKEN]], dtype=np.int64)

    sess = {n: ort.InferenceSession(str(D / f"{n}.onnx"), providers=["CPUExecutionProvider"])
            for n in ("text", "cond", "step", "decoder")}

    h, dur = sess["text"].run(None, {"ids": ids, "style": np.array([STYLE], dtype=np.int64)})
    cw, ws, fw, fp, T = build_plan(ids[0].tolist(), dur[0].tolist(), SPEED,
                                   cfg["limits"]["max_word_frames"], cfg["limits"]["max_frames"])
    cond = sess["cond"].run(None, {
        "h": h, "dur": dur,
        "cw": np.array([cw], dtype=np.int64),
        "wstart": np.array([ws], dtype=np.int64),
        "fw": np.array([fw], dtype=np.int64),
        "fp": np.array([fp], dtype=np.float32),
    })[0]

    noise = np.random.default_rng(0).standard_normal((1, T, LAT)).astype(np.float32)
    x = noise.copy()
    for k in range(STEPS):
        x = sess["step"].run(None, {
            "x": x, "cond": cond,
            "t": np.array([k / STEPS], dtype=np.float32),
            "dt": np.array([1.0 / STEPS], dtype=np.float32),
            "cfg": np.array([CFG], dtype=np.float32),
        })[0]

    audio = []
    for (s, e) in windows(T):
        a, b = max(0, s - 8), min(T, e + 8)
        z = x[0, a:b, :].T[None, :, :].astype(np.float32)
        w = sess["decoder"].run(None, {"z": z})[0].reshape(-1)
        audio.append(w[(s - a) * HOP:(s - a) * HOP + (e - s) * HOP])
    audio = np.concatenate(audio).astype(np.float32)

    def save(name, arr):
        arr.astype(np.float32).tofile(out / f"{name}.f32")

    (out / "ids.i64").write_bytes(ids.astype(np.int64).tobytes())
    (out / "fw.i64").write_bytes(np.array(fw, dtype=np.int64).tobytes())
    save("h", h)
    save("dur", dur)
    save("cond", cond)
    save("noise", noise)
    save("latents", x)
    save("audio_full", audio)
    (out / "shapes.json").write_text(json.dumps({
        "L": len(SPOKEN), "T": T, "style": STYLE, "steps": STEPS, "cfg": CFG, "speed": SPEED,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"L={len(SPOKEN)} T={T} audio={audio.shape} peak={float(np.abs(audio).max()):.3f}")
    print("yazildi:", out)


if __name__ == "__main__":
    main()
