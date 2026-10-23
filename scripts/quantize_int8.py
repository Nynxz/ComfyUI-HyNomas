"""int8 + ConvRot quantization of a converted Hy-Nomas file (the DiT or the PixelUMM VAE).

    <comfy>/.venv/bin/python scripts/quantize_int8.py dit nomas_dit_distilled_bf16.safetensors nomas_dit_distilled_int8_convrot.safetensors \
        --tool <comfy-model-tools dir>
    <comfy>/.venv/bin/python scripts/quantize_int8.py vae nomas_pixelumm_vae_bf16.safetensors nomas_pixelumm_vae_int8_convrot.safetensors --tool ...

Uses quant_int8_convrot.py's recipe (fp32 upcast, block-Hadamard rotation, per-channel absmax
scale, ``<layer>.comfy_quant`` config), which ComfyUI's int8 ops load natively.
Quantized: the DiT's double-block linears (attention, MLP, modulation); the decoder's attention
and MLP linears. Kept bf16: embeddings, the text refiner, timestep MLPs, the final layer, the
camera branch (small, and the reference keeps it bf16 too), all norms, the decoder's pixel
embed / latent projection / head, and the DCAE-V encoder.
"""

import argparse
import json
import os
import re
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert import write_safetensors  # noqa: E402

PATTERNS = {
    "dit": re.compile(r"^double_blocks\.\d+\.(img|txt)_(attn_[qkv]|attn_proj|mlp\.fc[12]|mod\.linear)\.weight$"),
    "vae": re.compile(r"^decoder\.layers\.\d+\.(attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)\.weight$"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=list(PATTERNS))
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--tool", default=os.path.join(os.path.dirname(__file__), "..", "..", "references", "comfy-model-tools"),
                    help="directory with quant_int8_convrot.py")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.tool))
    import quant_int8_convrot as q

    pattern = PATTERNS[args.kind]
    entries, errs = [], []
    f = safe_open(args.src, "pt")
    meta = dict(f.metadata() or {})

    def quantized(key, shape):
        gs = q.best_gs(shape[-1])
        memo = {}

        def get(kind):
            if not memo:
                w = f.get_tensor(key)
                qd, scale = q.quantize_convrot(w, gs, mseclip=False, device=args.device)
                errs.append(q.recon_metrics(qd, scale, w, gs, device=args.device)[1])
                memo["weight"], memo["scale"] = qd.cpu(), scale.cpu()
            out = memo[kind]
            if kind == "scale":
                memo.clear()
            return out
        return gs, get

    for key in f.keys():
        s = f.get_slice(key)
        shape = tuple(s.get_shape())
        dtype = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}[s.get_dtype()]
        if pattern.match(key) and len(shape) == 2 and q.best_gs(shape[-1]):
            base = key[: -len(".weight")]
            gs, run = quantized(key, shape)
            entries.append((key, shape, torch.int8, lambda run=run: run("weight")))
            entries.append((f"{base}.weight_scale", shape[:-1] + (1,), torch.float32, lambda run=run: run("scale")))
            blob = torch.tensor(list(json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": gs}).encode()),
                                dtype=torch.uint8)
            entries.append((f"{base}.comfy_quant", tuple(blob.shape), torch.uint8, lambda b=blob: b))
        else:
            entries.append((key, shape, dtype, lambda k=key: f.get_tensor(k)))
    meta["nomas_quantization"] = "int8_convrot"
    write_safetensors(args.dst, entries, meta)
    if errs:
        errs = torch.tensor(errs)
        print(f"wrote {args.dst}: {len(errs)} quantized matrices, weight relerr mean {errs.mean():.2f}% max {errs.max():.2f}%")


if __name__ == "__main__":
    main()
