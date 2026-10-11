# -*- coding: utf-8 -*-
"""Antalia Rust parite testi icin referans verisi (models/antalia_ref/).

api.py::_acoustic + _edges adimlarini birebir kosar, ara degerleri kaydeder.
Parametreler: speed=0.95, steps=8, cfg=2.0, temperature=0.6, seed=0.
"""
import json
from pathlib import Path

import numpy as np
import torch

TEXT = "Merhaba, size nasıl yardımcı olabilirim?"


def save(path: Path, arr: np.ndarray):
    path.write_bytes(np.ascontiguousarray(arr).tobytes())


def main():
    torch.set_grad_enabled(False)
    from huggingface_hub import snapshot_download
    from antalia_mini.weights import load_release
    from antalia_mini.text import encode, to_model_text
    from antalia_mini.api import split_model_text, OPTION_DEFAULTS
    import math

    snap = Path(snapshot_download(
        "cloud0day3/antalia-mini",
        allow_patterns=["acoustic.safetensors", "vocoder.safetensors", "config.json"]))
    acoustic, vocoder, config = load_release(str(snap), "fp32")
    acoustic.eval()
    vocoder.eval()

    o = {**OPTION_DEFAULTS, **config.get("options", {})}
    speed, steps, cfg, temp, seed = 0.95, 8, 2.0, 0.6, 0
    print("options:", {k: o[k] for k in
                       ("prefix", "suffix", "prefix_initials", "gem_stretch", "tone",
                        "fade_ms", "gain_db", "mel_sharpen")}, flush=True)

    spoken = to_model_text(TEXT)
    chunks = split_model_text(spoken)
    print(f"spoken={spoken!r} chunks={len(chunks)}", flush=True)
    assert len(chunks) == 1
    chunk = chunks[0]

    # _acoustic birebir (tek chunk, seed 0)
    am, dev, c = acoustic, "cpu", acoustic.c
    pre, suf = o["prefix"], o["suffix"]
    pres = [pre if pre and (o["prefix_initials"] is None or t[:1] in o["prefix_initials"]) else ""
            for t in [chunk]]
    # m harfi prefix_initials disinda -> pres bos
    npres = [0]
    nsuf = len(encode(suf)) - 2 if suf else 0
    ids = torch.tensor([encode(f"{pres[0]}{chunk}{suf}")], dtype=torch.long)
    L = ids.shape[1]
    il = torch.tensor([L])
    from antalia_mini.modules import sequence_mask
    xmask = sequence_mask(il, L)
    h = am.encoder(ids, xmask)
    logd = am.duration(h, xmask)
    # gem_stretch (first/space 1.0 -> atla)
    sp = encode(" ")[1]
    g = math.log(o["gem_stretch"])
    xs = ids[0].tolist()
    gem_idx = []
    for j in range(2, len(xs) - 1):
        if xs[j] == xs[j - 1] and xs[j] != sp:
            logd[0, j] += g
            gem_idx.append(j)
        elif xs[j - 1] == sp and xs[j] == xs[j - 2]:
            logd[0, j - 1] += g
            gem_idx.append(j - 1)
    print(f"gem uygulanan harfler: {gem_idx}", flush=True)
    dur = torch.round(torch.exp(logd) / speed).clamp(min=1, max=c.max_dur_frames).long() * xmask
    mel_len = (dur.sum(1) + c.stack - 1) // c.stack * c.stack
    T = int(mel_len.max())
    ymask = sequence_mask(mel_len, T)
    cond = am.expand(h, dur, T) * ymask[..., None]
    gg = am.generator
    ctok = gg.prep_cond(cond, ymask)
    tmask = ymask[:, :: c.stack]
    gen = torch.Generator().manual_seed(seed)
    n = int(mel_len[0]) // c.stack
    x = torch.zeros(1, ctok.shape[1], c.n_mels * c.stack)
    noise = torch.randn(n, c.n_mels * c.stack, generator=gen) * temp
    x[0, :n] = noise
    Path("models/antalia_ref/noise.f32").write_bytes(
        np.ascontiguousarray(noise.numpy().astype(np.float32)).tobytes())
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((1,), i * dt)
        d = torch.full_like(t, dt)
        vc = gg(x, t, d, ctok, tmask)
        cnull = gg.prep_cond(gg.null_cond.expand_as(cond) * ymask[..., None], ymask)
        vu = gg(x, t, d, cnull, tmask)
        v = vc + (cfg - 1.0) * (vc - vu)
        x = x + v * dt
    y = am._unstack(x) * ymask[..., None]
    mel = am.denorm_mel(y).transpose(1, 2)
    Tm = int(mel_len[0])

    # vocoder + edges
    with torch.no_grad():
        wav_full = vocoder(mel[:, :, :Tm])  # referans tam decode de degil, edges icin
    wav_full = wav_full[0].numpy()
    # _edges: span=(cum[npres], cum[L-2-nsuf])
    cum = torch.cumsum(dur, 1)[0]
    a, b = int(cum[0]), int(cum[L - 2 - nsuf])
    hop = 512
    fade = int(o["fade_ms"] * 48000 / 1000)
    s0 = 0
    s1 = min(len(wav_full), b * hop)
    wav = wav_full[s0:s1].copy()
    if o["suffix"] and fade and len(wav) > 2 * fade:
        wav[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    print(f"L={L} T={Tm} span=({a},{b}) wav_full={len(wav_full)} wav_cut={len(wav)}",
          flush=True)

    out = Path("models/antalia_ref")
    out.mkdir(parents=True, exist_ok=True)
    save(out / "ids.i64", ids.numpy().astype(np.int64))
    save(out / "h.f32", h[0].numpy().astype(np.float32))
    save(out / "logdur.f32", logd[0].numpy().astype(np.float32))
    save(out / "dur.i64", dur[0].numpy().astype(np.int64))
    # fw: repeat_interleave esdegeri
    fw = np.repeat(np.arange(L), dur[0].numpy())
    save(out / "fw.i64", fw.astype(np.int64))
    save(out / "cond.f32", cond[0].numpy().astype(np.float32))
    save(out / "ctok.f32", ctok[0].numpy().astype(np.float32))
    save(out / "mel.f32", mel[0].numpy().astype(np.float32))
    save(out / "wav_cut.f32", wav.astype(np.float32))
    (out / "shapes.json").write_text(json.dumps(
        {"L": L, "T": Tm, "spoken": spoken, "chunk": chunk,
         "span": [a, b], "nsuf": nsuf,
         "params": {"speed": speed, "steps": steps, "cfg": cfg, "temp": temp, "seed": seed}},
        ensure_ascii=False))
    print(f"referans yazildi: {out.absolute()}", flush=True)


if __name__ == "__main__":
    main()
