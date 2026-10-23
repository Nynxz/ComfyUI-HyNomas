"""Parity: ComfyUI-Nomas PixelUMMDecoder vs the reference DCAEVPixelUMMDecoder (first LAYERS layers),
on a grid where sliding-tile local attention is active, and the windowed decode on a longer clip."""
import os
import sys

SRC = os.environ["NOMAS_SRC"]  # a Hy-Nomas checkout
COMFY = os.environ["COMFY"]  # a ComfyUI checkout
PACK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CKPT = os.environ["NOMAS_VAE"]  # VAE/PixelUMM_FT/diffusion_pytorch_model.safetensors
sys.path[:0] = [COMFY, PACK, SRC, os.path.join(SRC, "hymm/models/vae/pixelumm_decoder")]

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

import comfy.ops  # noqa: E402
import model as ref_model  # noqa: E402  (the reference's pixelumm_decoder/model.py, torch only)
from nomas import pixelumm  # noqa: E402

LAYERS = int(os.environ.get("LAYERS", 2))
dev = torch.device("cuda")
sd = {}
with safe_open(CKPT, "pt") as f:
    for k in f.keys():
        if not k.startswith("decoder."):
            continue
        parts = k.split(".")
        if parts[1] == "layers" and int(parts[2]) >= LAYERS:
            continue
        sd[k[len("decoder."):]] = f.get_tensor(k).to(torch.bfloat16)

arch = ref_model.GenExpertArch(num_layers=LAYERS)
ref = ref_model.DCAEVPixelUMMDecoder(arch, local_attention_tile=8)
print("ref", ref.load_state_dict(sd, strict=True))
ref = ref.to(dev, torch.bfloat16).eval()

pixelumm.LAYERS = LAYERS
ours = pixelumm.PixelUMMDecoder(dtype=torch.bfloat16, device=dev, operations=comfy.ops.disable_weight_init)
print("ours", ours.load_state_dict(sd, strict=True))
ours.eval()


def cosine(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return float((a @ b) / (a.norm() * b.norm())), float((a - b).abs().max() / b.abs().max())


torch.manual_seed(0)
with torch.no_grad():
    for L, h, w in [(2, 14, 8), (3, 5, 6)]:  # local attention (4x2 tiles), then full attention
        z = torch.randn(1, 32, L, h, w, device=dev, dtype=torch.bfloat16)
        xt = torch.randn(1, 3, 4 * L, 32 * h, 32 * w, device=dev)
        r = ref(z, xt)
        o = ours(z, xt)
        print(f"decoder forward L={L} {32 * h}x{32 * w}", cosine(o, r))

    # windowed decode of a 13-latent clip: the reference's _merge_overlapping over its own windows
    from streaming import _merge_overlapping, _window_starts  # noqa: E402
    L, h, w = 13, 3, 4
    z = torch.randn(1, 32, L, h, w, device=dev, dtype=torch.bfloat16)
    starts = _window_starts(L, 10, 5)
    assert starts == pixelumm.window_starts(L)
    noise = pixelumm.window_noise(1, h, w, 0, L, dev)
    parts = [ref(z[:, :, s:s + 10], noise[:, :, 4 * s:4 * (s + 10)]) for s in starts]
    r = _merge_overlapping(parts, [4 * s for s in starts], dim=2)
    o = pixelumm.decode_clip(ours, z)
    print("windowed decode", tuple(o.shape), starts, cosine(o, r))
