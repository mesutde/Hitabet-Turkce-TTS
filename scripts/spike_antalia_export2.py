# -*- coding: utf-8 -*-
"""Faz A spike bolum 2: vocoder (native iSTFT) + text + tek DiT adimi export."""
import sys
from pathlib import Path

import numpy as np
import torch


def load_all():
    from huggingface_hub import snapshot_download
    from antalia_mini.weights import load_release
    snap = Path(snapshot_download(
        "cloud0day3/antalia-mini",
        allow_patterns=["acoustic.safetensors", "vocoder.safetensors", "config.json"]))
    acoustic, vocoder, config = load_release(str(snap), "fp32")
    return acoustic.eval(), vocoder.eval(), config


def export_strict(wrapper, args, dynamic_shapes, path, input_names, output_names):
    import torch.onnx
    ep = torch.export.export(wrapper, args, dynamic_shapes=dynamic_shapes)
    print(f"  range: {ep.range_constraints}")
    torch.onnx.export(ep, (), str(path), input_names=input_names,
                      output_names=output_names, dynamo=True, opset_version=18)
    print(f"  [OK] {path.name}")


def check(path, feeds, refs, tol):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(None, feeds)
    ok = True
    for i, (g, r) in enumerate(zip(got, refs)):
        d = float(np.abs(np.asarray(g) - np.asarray(r)).max())
        good = d < tol
        ok = ok and good
        print(f"  [{'OK ' if good else 'FAIL'}] cikti{i}: max fark {d:.3e} (esik {tol:.0e})")
    return ok


class VocoderWrapper(torch.nn.Module):
    def __init__(self, vocoder):
        super().__init__()
        self.vocoder = vocoder

    def forward(self, mel):
        return self.vocoder(mel)


class TextWrapper(torch.nn.Module):
    def __init__(self, acoustic):
        super().__init__()
        self.acoustic = acoustic

    def forward(self, ids, mask):
        h = self.acoustic.encoder(ids, mask)
        logdur = self.acoustic.duration(h, mask)
        return h, logdur


class StepWrapper(torch.nn.Module):
    """Tek DiT adimi, CFG harimi gomulu (cfg=1'de v_cond'a esit)."""

    def __init__(self, generator):
        super().__init__()
        self.g = generator

    def forward(self, x, t, d, ctok, cnull, tmask, cfg):
        vc = self.g(x, t, d, ctok, tmask)
        vu = self.g(x, t, d, cnull, tmask)
        return vc + (cfg - 1.0) * (vc - vu)


def main():
    out = Path("models/antalia_spike")
    out.mkdir(parents=True, exist_ok=True)
    acoustic, vocoder, config = load_all()
    from antalia_mini.text import encode

    results = {}
    B, Ld, Td = (torch.export.Dim("batch", min=1, max=4),
                 torch.export.Dim("letters", min=1, max=1000),
                 torch.export.Dim("frames", min=1, max=3000))

    # --- T3: text encoder + duration ---
    ids = torch.tensor([encode("merhaba, size nasıl yardımcı olabilirim?")])
    L = ids.shape[1]
    mask = ids != 0
    print(f"T3: L={L}")
    with torch.no_grad():
        h_ref, logdur_ref = TextWrapper(acoustic)(ids, mask)
    try:
        export_strict(TextWrapper(acoustic).eval(), (ids, mask),
                       {"ids": {1: Ld}, "mask": {1: Ld}},
                       out / "t3_text.onnx", ["ids", "mask"], ["h", "logdur"])
        results["T3-text"] = check(
            out / "t3_text.onnx",
            {"ids": ids.numpy(), "mask": mask.numpy()},
            [h_ref.numpy(), logdur_ref.numpy()], 1e-5)
    except Exception as e:
        print(f"  [FAIL] T3 export hatasi: {type(e).__name__}: {str(e)[:300]}")
        results["T3-text"] = False

    # --- T4: tek DiT adimi ---
    g = acoustic.generator
    T = 64
    torch.manual_seed(0)
    x = torch.randn(1, T // 2, 256)
    ctok = torch.randn(1, T // 2, 192)
    cnull = torch.randn(1, T // 2, 192)
    tmask = torch.ones(1, T // 2, dtype=torch.bool)
    t = torch.tensor([0.0])
    d = torch.tensor([0.125])
    cfg = torch.tensor(2.0)
    with torch.no_grad():
        vc = g(x, t, d, ctok, tmask)
        vu = g(x, t, d, cnull, tmask)
        v_ref = vc + (cfg.item() - 1.0) * (vc - vu)
    print("T4: N=", T // 2)
    try:
        export_strict(StepWrapper(g).eval(), (x, t, d, ctok, cnull, tmask, cfg),
                       {"x": {1: Td}, "t": {}, "d": {}, "ctok": {1: Td},
                        "cnull": {1: Td}, "tmask": {1: Td}, "cfg": {}},
                       out / "t4_step.onnx",
                       ["x", "t", "d", "ctok", "cnull", "tmask", "cfg"], ["v"])
        results["T4-step"] = check(
            out / "t4_step.onnx",
            {"x": x.numpy(), "t": t.numpy(), "d": d.numpy(), "ctok": ctok.numpy(),
             "cnull": cnull.numpy(), "tmask": tmask.numpy(), "cfg": np.array(2.0, dtype=np.float32)},
            [v_ref.numpy()], 1e-5)
    except Exception as e:
        print(f"  [FAIL] T4 export hatasi: {type(e).__name__}: {str(e)[:300]}")
        results["T4-step"] = False

    # --- T1: vocoder native iSTFT ---
    with torch.no_grad():
        mel_ref, ml = acoustic.synthesize(ids, torch.tensor([L]), steps=8,
                                          speed=0.95, temperature=0.6, cfg=2.0,
                                          generator=torch.Generator().manual_seed(0))
    Tmel = int(ml[0])
    print(f"T1: Tmel={Tmel}")
    mel = mel_ref[:, :, :Tmel].contiguous()
    with torch.no_grad():
        wav_ref = vocoder(mel)
    try:
        export_strict(VocoderWrapper(vocoder).eval(), (mel,),
                       {"mel": {2: Td}},
                       out / "t1_vocoder.onnx", ["mel"], ["wav"])
        results["T1-vocoder"] = check(
            out / "t1_vocoder.onnx", {"mel": mel.numpy()}, [wav_ref.numpy()], 1e-4)
    except Exception as e:
        print(f"  [FAIL] T1 export hatasi: {type(e).__name__}: {str(e)[:500]}")
        results["T1-vocoder"] = False

    # --- T2: vocoder manuel iSTFT (yedek yol) ---
    import sys as _sys
    _sys.path.insert(0, "C:/Users/Mesut/AppData/Local/Temp/opencode")
    from manual_istft import build_istft_weights

    class VocoderManual(torch.nn.Module):
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

    print("T2: manuel iSTFT vocoder")
    try:
        export_strict(VocoderManual(vocoder).eval(), (mel,),
                       {"mel": {2: Td}},
                       out / "t2_vocoder.onnx", ["mel"], ["wav"])
        results["T2-vocoder-manuel"] = check(
            out / "t2_vocoder.onnx", {"mel": mel.numpy()}, [wav_ref.numpy()], 1e-4)
    except Exception as e:
        print(f"  [FAIL] T2 export hatasi: {type(e).__name__}: {str(e)[:500]}")
        results["T2-vocoder-manuel"] = False

    print("\n=== SPIKE SONUC ===")
    for k, v in results.items():
        print(f"  {k}: {'GO' if v else 'NO-GO'}")
    return all(results.values())


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
