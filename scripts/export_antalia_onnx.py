# -*- coding: utf-8 -*-
"""Antalia Mini -> ONNX export (gercek mimari, cloud0day3/antalia-mini).

Uretilen dosyalar (models/antalia/):
  antalia_text.onnx : (ids[B,L] int64, mask[B,L] bool) -> (h[B,L,192] f32, logdur[B,L] f32)
  antalia_prep.onnx : (cond[B,T,192] f32, fmask[B,T] bool) -> (ctok[B,N,192] f32)
  antalia_step.onnx : (x[B,N,256], t[B,], d[B,], ctok, cnull[B,N,192],
                       tmask[B,N] bool, cfg scalar) -> (v[B,N,256] f32, CFG harimli)
  antalia_vocoder.onnx : (mel[B,128,T] f32) -> (wav[B,T*512] f32 @48kHz, manuel iSTFT)
  symbols.json, tensor_shapes.txt

Kullanim:
  python scripts/export_antalia_onnx.py --output_dir models/antalia
Gereksinim: torch onnx onnxruntime onnxscript huggingface_hub safetensors antalia-mini
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO = "cloud0day3/antalia-mini"
TEST_TEXT = "Merhaba, size nasıl yardımcı olabilirim?"


def load_all():
    from huggingface_hub import snapshot_download
    from antalia_mini.weights import load_release
    snap = Path(snapshot_download(
        REPO, allow_patterns=["acoustic.safetensors", "vocoder.safetensors", "config.json"]))
    acoustic, vocoder, config = load_release(str(snap), "fp32")
    return acoustic.eval(), vocoder.eval(), config


def build_istft_weights(n_fft=2048, hop=512):
    n_bin = n_fft // 2 + 1
    hann = torch.hann_window(n_fft)
    W = torch.zeros(n_fft, 2 * n_bin)
    n = torch.arange(n_fft).float()
    for k in range(n_bin):
        ang = 2 * math.pi * k * n / n_fft
        if k == 0:
            W[:, k] = 1.0 / n_fft
        elif k == n_bin - 1:
            W[:, k] = ((-1.0) ** n) / n_fft
        else:
            W[:, k] = 2 * torch.cos(ang) / n_fft
            W[:, n_bin + k] = -2 * torch.sin(ang) / n_fft
    W = W * hann[:, None]
    return W.t().unsqueeze(1).contiguous(), (hann ** 2).unsqueeze(0).unsqueeze(0).contiguous()


class TextWrapper(torch.nn.Module):
    def __init__(self, acoustic):
        super().__init__()
        self.acoustic = acoustic

    def forward(self, ids, mask):
        h = self.acoustic.encoder(ids, mask)
        return h, self.acoustic.duration(h, mask)


class PrepWrapper(torch.nn.Module):
    def __init__(self, generator):
        super().__init__()
        self.g = generator

    def forward(self, cond, fmask):
        # prep_cond'un conv kismi (cift-uzunluk guard'i gerektiren
        # cift-ortalamasi Rust'ta yapilir: (a+b)/2, bit-esit).
        m = fmask[:, None].float()
        return self.g.cond_conv(cond.transpose(1, 2) * m, m).transpose(1, 2)


class StepWrapper(torch.nn.Module):
    """Tek DiT adimi, CFG harimi gomulu (cfg=1'de v_cond'a esit)."""

    def __init__(self, generator):
        super().__init__()
        self.g = generator

    def forward(self, x, t, d, ctok, cnull, tmask, cfg):
        vc = self.g(x, t, d, ctok, tmask)
        vu = self.g(x, t, d, cnull, tmask)
        return vc + (cfg - 1.0) * (vc - vu)


class VocoderWrapper(torch.nn.Module):
    """Conv yigini + manuel iSTFT (polar/istft complex op yok)."""

    def __init__(self, vocoder):
        super().__init__()
        self.voc = vocoder
        feat_w, env_w = build_istft_weights()
        self.register_buffer("istft_w", feat_w)
        self.register_buffer("env_w", env_w)

    def forward(self, mel):
        import torch.nn.functional as F
        v = self.voc
        x = v.inp(mel)
        x = v.norm0(x.transpose(1, 2)).transpose(1, 2)
        for b in v.blocks:
            x = b(x)
        x = v.head(v.norm1(x.transpose(1, 2))).transpose(1, 2)
        mag, ph = x.chunk(2, 1)
        mag = torch.exp(mag).clamp(max=1e2)
        feat = torch.cat([mag * torch.cos(ph), mag * torch.sin(ph)], dim=1)
        B, _, T = feat.shape
        full = F.conv_transpose1d(feat, self.istft_w, stride=512)
        env = F.conv_transpose1d(torch.ones(B, 1, T, device=mel.device),
                                 self.env_w, stride=512)
        s = 1024
        e = s + (T - 1) * 512
        wav = full[:, 0, s:e] / env[:, 0, s:e].clamp_min(1e-10)
        return F.pad(wav, (0, 512))


def export_strict(wrapper, args, dynamic_shapes, path, input_names, output_names):
    """Strict torch.export (dinamik boylar korunur) + ONNX cevirisi."""
    import torch.onnx
    ep = torch.export.export(wrapper, args, dynamic_shapes=dynamic_shapes)
    print(f"  range: {ep.range_constraints}", flush=True)
    torch.onnx.export(ep, (), str(path), input_names=input_names,
                      output_names=output_names, dynamo=True, opset_version=18)
    print(f"  [OK] {path.name}", flush=True)


def check(path, feeds, refs, tol, label):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(None, feeds)
    ok = True
    for i, (g, r) in enumerate(zip(got, refs)):
        d = float(np.abs(np.asarray(g) - np.asarray(r)).max())
        good = d < tol
        ok = ok and good
        print(f"  [{'OK ' if good else 'FAIL'}] {label}{i}: max fark {d:.3e} (esik {tol:.0e})",
              flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="models/antalia")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    acoustic, vocoder, config = load_all()
    from antalia_mini.text import SYMBOLS, encode, to_model_text
    from antalia_mini.api import split_model_text
    g = acoustic.generator

    spoken = to_model_text(TEST_TEXT)
    print(f"frontend: {TEST_TEXT!r} -> {spoken!r}", flush=True)
    chunks = split_model_text(spoken)
    print(f"chunks: {len(chunks)}", flush=True)
    assert chunks, "test parcasi bos olmamali"

    ok_all = True
    Ld = torch.export.Dim("letters", min=1, max=2000)
    Td = torch.export.Dim("frames", min=1, max=4000)
    Nd = torch.export.Dim("tokens", min=1, max=2000)

    # --- text ---
    ids = torch.tensor([encode(chunks[0])])
    L = ids.shape[1]
    mask = ids != 0
    with torch.no_grad():
        h_ref, logdur_ref = TextWrapper(acoustic)(ids, mask)
    print(f"text: L={L}", flush=True)
    export_strict(TextWrapper(acoustic).eval(), (ids, mask),
                   {"ids": {1: Ld}, "mask": {1: Ld}}, out / "antalia_text.onnx",
                   ["ids", "mask"], ["h", "logdur"])
    ok_all &= check(out / "antalia_text.onnx",
                    {"ids": ids.numpy(), "mask": mask.numpy()},
                    [h_ref.numpy(), logdur_ref.numpy()], 1e-5, "text")

    # --- prep (cond: Rust gather esdegeri + fmask) -> conv ozellikleri;
    #     cift-ortalama Rust'ta (a+b)/2 (bit-esit, guard gerektirmez).
    T = 64
    torch.manual_seed(1)
    cond = torch.randn(1, T, 192)
    fmask = torch.ones(1, T, dtype=torch.bool)
    with torch.no_grad():
        ctok_ref = g.prep_cond(cond, fmask)
    export_strict(PrepWrapper(g).eval(), (cond, fmask),
                   {"cond": {1: Td}, "fmask": {1: Td}}, out / "antalia_prep.onnx",
                   ["cond", "fmask"], ["conv"])
    # Dogrulama: ONNX conv ciktisi + numpy cift-ortalama vs torch ctok.
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out / "antalia_prep.onnx"),
                                providers=["CPUExecutionProvider"])
    conv_got = sess.run(None, {"cond": cond.numpy(), "fmask": fmask.numpy()})[0]
    pair = (conv_got[:, 0::2, :] + conv_got[:, 1::2, :]) * 0.5
    dprep = float(np.abs(pair - ctok_ref.numpy()).max())
    print(f"  prep conv+cift-ortalama: max fark {dprep:.3e} (esik 1e-5)", flush=True)
    ok_all &= dprep < 1e-5

    # --- step ---
    N = T // 2
    x = torch.randn(1, N, 256)
    ctok = torch.randn(1, N, 192)
    cnull = torch.randn(1, N, 192)
    tmask = torch.ones(1, N, dtype=torch.bool)
    t = torch.tensor([0.0])
    d = torch.tensor([0.125])
    cfg = torch.tensor(2.0)
    with torch.no_grad():
        vc = g(x, t, d, ctok, tmask)
        vu = g(x, t, d, cnull, tmask)
        v_ref = vc + (cfg.item() - 1.0) * (vc - vu)
    export_strict(StepWrapper(g).eval(), (x, t, d, ctok, cnull, tmask, cfg),
                   {"x": {1: Nd}, "t": {}, "d": {}, "ctok": {1: Nd},
                    "cnull": {1: Nd}, "tmask": {1: Nd}, "cfg": {}},
                   out / "antalia_step.onnx",
                   ["x", "t", "d", "ctok", "cnull", "tmask", "cfg"], ["v"])
    ok_all &= check(out / "antalia_step.onnx",
                    {"x": x.numpy(), "t": t.numpy(), "d": d.numpy(),
                     "ctok": ctok.numpy(), "cnull": cnull.numpy(),
                     "tmask": tmask.numpy(), "cfg": np.array(2.0, dtype=np.float32)},
                    [v_ref.numpy()], 5e-5, "step")

    # --- vocoder (gercek mel ile) ---
    with torch.no_grad():
        mel_ref, ml = acoustic.synthesize(ids, torch.tensor([L]), steps=8,
                                          speed=0.95, temperature=0.6, cfg=2.0,
                                          generator=torch.Generator().manual_seed(0))
    Tm = int(ml[0])
    mel = mel_ref[:, :, :Tm].contiguous()
    with torch.no_grad():
        wav_ref = vocoder(mel).numpy()
    print(f"vocoder: Tmel={Tm}", flush=True)
    export_strict(VocoderWrapper(vocoder).eval(), (mel,),
                   {"mel": {2: Td}}, out / "antalia_vocoder.onnx",
                   ["mel"], ["wav"])
    sess_ok = check(out / "antalia_vocoder.onnx", {"mel": mel.numpy()},
                    [wav_ref], 5e-3, "vocoder")
    # Ek guvence: ortalama + SNR (bilinen dagilim: tipik 95dB, tek bolge ~59dB)
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out / "antalia_vocoder.onnx"),
                                providers=["CPUExecutionProvider"])
    got = sess.run(None, {"mel": mel.numpy()})[0]
    dd = np.abs(got - wav_ref)
    snr = 10 * np.log10((wav_ref ** 2).mean() / (dd ** 2).mean())
    print(f"  vocoder SNR: {snr:.1f} dB, ort fark: {dd.mean():.3e}", flush=True)
    ok_all &= sess_ok and snr > 50.0 and float(dd.mean()) < 1e-4

    with open(out / "symbols.json", "w", encoding="utf-8") as f:
        json.dump(SYMBOLS, f, ensure_ascii=False)
    print(f"[OK] symbols.json ({len(SYMBOLS)} token)", flush=True)
    # null_cond egitimli parametresi (Rust prep-null yolu icin ham deger)
    null_cond = g.null_cond.detach().cpu().numpy().astype(np.float32)
    assert null_cond.shape == (192,), null_cond.shape
    Path(out / "null_cond.f32").write_bytes(np.ascontiguousarray(null_cond).tobytes())
    print("[OK] null_cond.f32 (192, f32 LE)", flush=True)
    with open(out / "tensor_shapes.txt", "w", encoding="utf-8") as f:
        f.write(
            "antalia_text.onnx:\n"
            "  ids (1,L) int64 ([BOS]...[EOS], PAD=0) | mask (1,L) bool\n"
            "  h (1,L,192) f32 | logdur (1,L) f32\n"
            "antalia_prep.onnx:\n"
            "  cond (1,T,192) f32 (=gather(h,fw)), fmask (1,T) bool\n"
            "  conv (1,T,192) f32; cift-ortalama Rust'ta: ctok=(c[:,0::2]+c[:,1::2])*0.5\n"
            "antalia_step.onnx:\n"
            "  x (1,N,256), t (1,) f32, d (1,) f32, ctok/cnull (1,N,192),\n"
            "  tmask (1,N) bool, cfg scalar f32 -> v (1,N,256) (CFG harimli)\n"
            "antalia_vocoder.onnx:\n"
            "  mel (1,128,T) f32 (denorm log-mel) -> wav (1,T*512) f32 @48kHz\n"
            "Rust timeline (api.py::_acoustic ile birebir):\n"
            "  dur=round(exp(logdur+stretches)/speed) clamp[1,60]; T=ceil(sum/2)*2\n"
            "  fw=repeat_interleave(arange(L), dur)[:T] -> cond=gather(h,fw)\n"
            "  gürültü (1,N,256)*sıcaklık; K adim: t=i*dt, d=dt (K==1 ise 1.0)\n"
            "  prefix/suffix span kesimi + fade + duraklama; EQ/limiter/resample Rust'ta\n"
        )
    print("[OK] tensor_shapes.txt", flush=True)
    print("=== EXPORT", "BASARILI" if ok_all else "BASARISIZ", "===")
    return ok_all


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
