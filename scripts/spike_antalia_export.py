# -*- coding: utf-8 -*-
"""Faz A spike: Antalia Mini ONNX cevrilebilirlik kaniti.

Testler (sirasiyla, ilki batarsa sonrakine gecilir):
  T1: vocoder native export (torch.polar + torch.istft oldugu gibi)
  T2: vocoder manuel iSTFT (ConvTranspose1d esdegeri) export
  T3: text encoder + duration export
  T4: tek DiT adimi (CFG harimli) export
Hepsi onnxruntime ile PyTorch'a karsi dogrulanir.
"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO_SNAP = None


def load_all():
    global REPO_SNAP
    from huggingface_hub import snapshot_download
    from antalia_mini.weights import load_release
    snap = Path(snapshot_download(
        "cloud0day3/antalia-mini",
        allow_patterns=["acoustic.safetensors", "vocoder.safetensors", "config.json"]))
    REPO_SNAP = snap
    acoustic, vocoder, config = load_release(str(snap), "fp32")
    return acoustic.eval(), vocoder.eval(), config


def export_strict(wrapper, args, dynamic_shapes, path, input_names, output_names):
    """Iki asama: strict torch.export (dinamik boylar korunur) + ONNX."""
    import torch.onnx
    ep = torch.export.export(wrapper, args, dynamic_shapes=dynamic_shapes)
    print(f"  range_constraints: {ep.range_constraints}")
    torch.onnx.export(
        ep, (), str(path),
        input_names=input_names, output_names=output_names,
        dynamo=True, opset_version=18,
    )
    print(f"  [OK] {path.name}")


def check(path, feeds, ref_outs, tol=2e-5):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(None, feeds)
    ok = True
    for i, (g, r) in enumerate(zip(got, ref_outs)):
        d = float(np.abs(g - r).max())
        status = "OK " if d < tol else "FAIL"
        if d >= tol:
            ok = False
        print(f"  [{status}] cikti{i}: max fark {d:.3e} (esik {tol:.0e})")
    return ok


def main():
    acoustic, vocoder, config = load_all()
    from antalia_mini.api import model_text, split_model_text
    spoken = model_text("Merhaba, size nasıl yardımcı olabilirim?")
    print("spoken:", spoken)
    chunks = split_model_text(spoken)
    print("chunks:", len(chunks), [c[:40] for c in chunks])

    # Referans: tek chunk uzerinden tam boru hatti (say() ile ayni matematik).
    from antalia_mini.api import Antalia
    import io
    tts = Antalia.__new__(Antalia)
    # __init__'siz minimal kurulum: load_release sonuclari dogrudan bagla
    import antalia_mini.api as api_mod
    tts.acoustic = acoustic
    tts.vocoder = vocoder
    tts.config = config
    tts.native_rate = 48000
    tts.defaults = {**{"steps": 8, "cfg": 2.0, "speed": 0.95, "temperature": 0.6},
                    **config.get("defaults", {})}
    from antalia_mini.api import OPTION_DEFAULTS
    tts.options = {**OPTION_DEFAULTS, **config.get("options", {})}
    tts.device = torch.device("cpu")
    tts._generator = acoustic.generator
    tts._voc = vocoder
    import threading
    tts._lock = threading.Lock()

    ch = chunks[0]
    gen = torch.Generator().manual_seed(0)
    wav_ref = tts._synth([ch], [gen], speed=0.95, steps=8, cfg=2.0)[0]
    print("referans wav:", wav_ref.shape, wav_ref.dtype)

    # Mel ara degeri (vocoder girisi icin): _acoustic'i ayri cagir
    from antalia_mini.text import encode
    print("encode ornek:", encode("merhaba.")[:12])
    print("SPIKE-HAZIR")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
