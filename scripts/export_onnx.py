#!/usr/bin/env python3
"""
EMA Lightning -> ONNX export (gercek mimari).

Kaynak: https://huggingface.co/canberkkkkkk/ema-lightning
  ema.pt     : TextEncoder(d=224) + Duration + Aligner + DiT(6 blok, 4-step DMD2)
  decoder.pt : HiFi-GAN vocoder (64-dim latent @25Hz -> 48kHz, hop=1920)

Uretilen dosyalar (models/onnx/):
  text_encoder.onnx : (ids[B,L] int64, mask[B,L] bool) -> (h[B,L,224] f32, dur[B,L] f32)
  sound_stage.onnx  : (h, dur, mask, cw int64, wstart int64, fw int64, fp, fmask bool,
                       noise[B,4,T,64]) -> (latents[B,T,64] f32)
  decoder.onnx      : (z[B,64,T] f32, lengths[B] int64) -> (audio[B,T*1920] f32)
  vocab.json        : Rust tokenizer icin id->karakter listesi

Kullanim:
  python scripts/export_onnx.py                      # HF'ten indirir, cevirir, dogrular
  python scripts/export_onnx.py --verify-only        # mevcut ONNX'lari PyTorch ile karsilastirir

Gereksinim: pip install torch onnx onnxruntime onnxscript huggingface_hub normalizer-tr ema-lightning
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = "canberkkkkkk/ema-lightning"
TEST_TEXT = "Merhaba, size nasıl yardımcı olabilirim?"


# ---------------------------------------------------------------------------
# Gercek model yukleme (pip paketi ema-lightning'in kendi modulleri)
# ---------------------------------------------------------------------------

def load_models(device="cpu"):
    from huggingface_hub import snapshot_download
    from ema_lightning.model import load_acoustic
    from ema_lightning.decoder import load_decoder

    snap = Path(snapshot_download(REPO, allow_patterns=["ema.pt", "decoder.pt", "config.json"]))
    acoustic = load_acoustic(str(snap / "ema.pt"), device)
    decoder = load_decoder(str(snap / "decoder.pt"), device)

    ck = torch.load(str(snap / "ema.pt"), map_location="cpu", weights_only=True)
    vocab = list(ck["vocab"])
    times = [float(t) for t in ck["cfg"]["distilled"]["times"]]
    print(f"vocab={len(vocab)} times={times} d={ck['cfg']['d']} latent={ck['cfg']['latent_dim']}")
    return acoustic, decoder, vocab


def build_piece(acoustic, decoder, frontend, text, speed, seed, device):
    """engine.py::plan/think ile birebir ayni timeline kurulumu (tek parca)."""
    from ema_lightning.engine import Engine
    engine = Engine(acoustic, decoder, device)
    piece = engine.piece(frontend(text), pause=0.0, seed=seed)
    engine.plan([piece], speed)
    return engine, piece


def piece_tensors(piece, speed, device):
    L, T = piece.letters, piece.frames
    ids = piece.ids.to(device)
    mask = ids != 0
    h = piece.h.to(device)
    dur = (piece.dur / speed).to(device)
    cw = piece.cw.to(device)
    wstart = piece.wstart.to(device)
    fw = piece.fw.to(device)
    fp = piece.fp.to(device)
    fmask = torch.ones(T, dtype=torch.bool, device=device)
    g = torch.Generator(device="cpu").manual_seed(piece.seed)
    noise = torch.randn((len(acoustic_times()), T, 64), generator=g).unsqueeze(0).to(device)
    # (1, 4, T, 64)
    return {
        "ids": ids.unsqueeze(0), "mask": mask.unsqueeze(0),
        "h": h.unsqueeze(0), "dur": dur.unsqueeze(0),
        "mask_l": mask.unsqueeze(0), "cw": cw.unsqueeze(0), "wstart": wstart.unsqueeze(0),
        "fw": fw.unsqueeze(0), "fp": fp.unsqueeze(0), "fmask": fmask.unsqueeze(0),
        "noise": noise,
    }


def acoustic_times():
    return [0.0, 0.25, 0.5, 0.75]


# ---------------------------------------------------------------------------
# Export sarmalayicilari
# ---------------------------------------------------------------------------

class TextWrapper(torch.nn.Module):
    def __init__(self, acoustic):
        super().__init__()
        self.acoustic = acoustic

    def forward(self, ids, mask):
        h, dur = self.acoustic.text_stage(ids, mask)
        return h, dur


class SoundWrapper(torch.nn.Module):
    def __init__(self, acoustic):
        super().__init__()
        self.acoustic = acoustic

    def forward(self, h, dur, mask, cw, wstart, fw, fp, fmask, noise):
        return self.acoustic.sound_stage(h, dur, mask, cw, wstart, fw, fp, fmask, noise)


class DecoderWrapper(torch.nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, z, lengths):
        return self.decoder(z, lengths)


def export_dynamo(wrapper, args, input_names, output_names, dynamic_shapes, path):
    """Iki asama: once strict torch.export (dinamik L/T korunur), sonra ONNX cevirisi.

    torch.onnx'un dahili draft_export yolu T boyutlarini ornekteki degere
    sabitledigi icin (fw/fmask/noise 53'te kaliyordu) bu ayrim sart.
    """
    import torch.onnx
    ep = torch.export.export(wrapper, args, dynamic_shapes=dynamic_shapes)
    got = {k: dict(v) for k, v in dynamic_shapes.items()}
    print(f"  range_constraints: {ep.range_constraints} (istenen: {got})")
    torch.onnx.export(
        ep,
        (),
        str(path),
        input_names=input_names,
        output_names=output_names,
        dynamo=True,
        opset_version=18,
    )
    print(f"[OK] {path.name}")


# ---------------------------------------------------------------------------
# Ana akis
# ---------------------------------------------------------------------------

def do_export(out_dir: Path, opset=18):
    device = "cpu"
    acoustic, decoder, vocab = load_models(device)
    from ema_lightning.frontend import Frontend
    frontend = Frontend(vocab)

    text = frontend(TEST_TEXT)
    print(f"frontend: {TEST_TEXT!r} -> {text!r}")
    engine, piece = build_piece(acoustic, decoder, frontend, TEST_TEXT, speed=1.0, seed=0, device=device)
    t = piece_tensors(piece, speed=1.0, device=device)
    L, T = piece.letters, piece.frames
    print(f"letters={L} frames={T}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Not: batch hep 1 (uygulama tek cumle calistirir) -> statik; L/T dinamik.
    Ldim = torch.export.Dim("letters", min=1, max=2000)
    Tdim = torch.export.Dim("frames", min=1, max=3000)

    export_dynamo(
        TextWrapper(acoustic).eval(),
        (t["ids"], t["mask"]),
        ["ids", "mask"], ["h", "dur"],
        {"ids": {1: Ldim}, "mask": {1: Ldim}},
        out_dir / "text_encoder.onnx",
    )
    export_dynamo(
        SoundWrapper(acoustic).eval(),
        (t["h"], t["dur"], t["mask_l"], t["cw"], t["wstart"], t["fw"], t["fp"], t["fmask"], t["noise"]),
        ["h", "dur", "mask", "cw", "wstart", "fw", "fp", "fmask", "noise"], ["latents"],
        {"h": {1: Ldim}, "dur": {1: Ldim}, "mask": {1: Ldim},
         "cw": {1: Ldim}, "wstart": {1: Ldim},
         "fw": {1: Tdim}, "fp": {1: Tdim}, "fmask": {1: Tdim},
         "noise": {2: Tdim}},
        out_dir / "sound_stage.onnx",
    )

    engine.think([piece])
    z = piece.latents.unsqueeze(0).transpose(1, 2).contiguous()  # (1,64,T)
    lengths = torch.tensor([T], dtype=torch.int64)
    export_dynamo(
        DecoderWrapper(decoder).eval(),
        (z, lengths),
        ["z", "lengths"], ["audio"],
        {"z": {2: Tdim}, "lengths": {}},
        out_dir / "decoder.onnx",
    )

    with open(out_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    print(f"[OK] vocab.json ({len(vocab)} token)")

    with open(out_dir / "tensor_shapes.txt", "w", encoding="utf-8") as f:
        f.write(
            "text_encoder.onnx:\n"
            "  ids   (B, L) int64   | mask (B, L) bool\n"
            "  h     (B, L, 224) float32 | dur (B, L) float32\n"
            "sound_stage.onnx:\n"
            "  h (B,L,224) f32, dur (B,L) f32 (=dur_text/speed), mask (B,L) bool\n"
            "  cw (B,L) int64, wstart (B,L) int64, fw (B,T) int64, fp (B,T) f32, fmask (B,T) bool\n"
            "  noise (B,4,T,64) f32 (seed'li uretici)\n"
            "  latents (B,T,64) f32\n"
            "decoder.onnx:\n"
            "  z (B,64,T) f32, lengths (B,) int64\n"
            "  audio (B,T*1920) f32 @48kHz\n"
            "Timeline (Rust'ta kurulur, engine.py::plan ile birebir):\n"
            "  counts[w] = round(sum(dur/speed | word==w)) clamp[1,250]; frames=min(sum,3000)\n"
            "  fw = repeat_interleave(word_idx, counts)[:frames]\n"
            "  fp = (arange - (cumsum(counts)-counts)[fw]) / counts[fw]\n"
            "  pencereler: ilk 25 frame, sonra 100'er; +-8 frame context ile decode, kirp\n"
        )
    print(f"[OK] tensor_shapes.txt")
    return True


def do_verify(out_dir: Path):
    import onnxruntime as ort
    device = "cpu"
    acoustic, decoder, vocab = load_models(device)
    from ema_lightning.frontend import Frontend
    frontend = Frontend(vocab)
    engine, piece = build_piece(acoustic, decoder, frontend, TEST_TEXT, speed=1.0, seed=0, device=device)
    t = piece_tensors(piece, speed=1.0, device=device)
    T = piece.frames

    sess = {n: ort.InferenceSession(str(out_dir / f"{n}.onnx"), providers=["CPUExecutionProvider"])
            for n in ("text_encoder", "sound_stage", "decoder")}

    def feed(name, *arrs):
        return {i.name: a for i, a in zip(sess[name].get_inputs(), arrs)}

    h, dur = sess["text_encoder"].run(
        None, feed("text_encoder", t["ids"].numpy(), t["mask"].numpy()))
    print(f"text: h max-abs-fark = {np.abs(h - piece.h.numpy()).max():.3e}, "
          f"dur max-abs-fark = {np.abs(dur - piece.dur.numpy()).max():.3e}")

    lat = sess["sound_stage"].run(None, feed(
        "sound_stage", h, dur / 1.0,
        t["mask_l"].numpy(), t["cw"].numpy(), t["wstart"].numpy(),
        t["fw"].numpy(), t["fp"].numpy(), t["fmask"].numpy(), t["noise"].numpy()))[0]
    engine.think([piece])
    print(f"sound: latents max-abs-fark = {np.abs(lat[0, :T] - piece.latents.numpy()).max():.3e}")

    z = lat.transpose(0, 2, 1)
    audio = sess["decoder"].run(
        None, feed("decoder", z, np.array([T], dtype=np.int64)))[0]
    ref = decoder(torch.from_numpy(z), None).numpy()
    print(f"decode: audio max-abs-fark = {np.abs(audio - ref).max():.3e}")
    print("Dogrulama tamam. Farklar ~1e-5 civarindaysa export saglikli.")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="models/onnx")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.output_dir)
    if args.verify_only:
        sys.exit(0 if do_verify(out_dir) else 1)
    do_export(out_dir)
    do_verify(out_dir)


if __name__ == "__main__":
    main()
