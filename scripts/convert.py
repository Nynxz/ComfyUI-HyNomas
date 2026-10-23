"""Convert the released Hy-Nomas weights into ComfyUI files (bf16, released key names).

    python scripts/convert.py dit  <Hy-Nomas>/DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors nomas_dit_distilled_bf16.safetensors
    python scripts/convert.py vae  <Hy-Nomas>/VAE/PixelUMM_FT/diffusion_pytorch_model.safetensors nomas_pixelumm_vae_bf16.safetensors
    python scripts/convert.py lora <Hy-Nomas>/DiT/lora_weights/cinematic nomas_lora_cinematic.safetensors

dit:  fp32 -> bf16, keys unchanged (Load Diffusion Model). A LoRA can be merged in first with
      --merge <adapter dir> (the reference merges into the distilled checkpoint: W += 2 * B @ A).
vae:  the PixelUMM file holds the PixelUMM decoder and the frozen DCAE-V encoder; fp32 -> bf16,
      keys unchanged (Load VAE).
lora: the PEFT adapter -> ComfyUI LoRA keys (diffusion_model.<module>.lora_A/B.weight + alpha).
Tensors are streamed to disk one at a time.
"""

import argparse
import json
import os
import struct

import torch
from safetensors import safe_open

DTYPES = {torch.float32: "F32", torch.bfloat16: "BF16", torch.float16: "F16", torch.int8: "I8", torch.uint8: "U8",
          torch.int64: "I64"}


def write_safetensors(path, entries, metadata=None):
    """entries: [(name, shape, dtype, produce)], ``produce()`` returning the tensor when it is written."""
    header, offset = {}, 0
    for name, shape, dtype, _ in entries:
        size = torch.tensor([], dtype=dtype).element_size()
        for d in shape:
            size *= d
        header[name] = {"dtype": DTYPES[dtype], "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    if metadata:
        header["__metadata__"] = {k: str(v) for k, v in metadata.items()}
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for name, shape, dtype, produce in entries:
            t = produce().to(dtype).contiguous()
            if tuple(t.shape) != tuple(shape):
                raise ValueError(f"{name}: produced {tuple(t.shape)}, declared {tuple(shape)}")
            f.write(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    os.replace(tmp, path)


def load_adapter(path):
    """A PEFT adapter dir (adapter.safetensors + adapter_config.json) -> ({module: (A, B)}, scale)."""
    with open(os.path.join(path, "adapter_config.json")) as f:
        cfg = json.load(f)
    scale = float(cfg.get("lora_alpha", cfg.get("alpha"))) / float(cfg.get("r", cfg.get("rank")))
    pairs = {}
    with safe_open(os.path.join(path, "adapter.safetensors"), "pt") as f:
        for key in f.keys():
            name = key.replace("._checkpoint_wrapped_module", "")
            for part in (".lora_A.weight", ".lora_B.weight"):
                if name.endswith(part):
                    pairs.setdefault(name[: -len(part)], {})[part[6]] = key
        return {m: (f.get_tensor(p["A"]), f.get_tensor(p["B"])) for m, p in pairs.items()}, scale, cfg


def convert_float(src, dst, dtype, merge=None, meta=None):
    lora, scale = ({}, 1.0) if merge is None else load_adapter(merge)[:2]
    f = safe_open(src, "pt")
    entries = []
    for key in f.keys():
        s = f.get_slice(key)
        shape = tuple(s.get_shape())

        def produce(key=key):
            t = f.get_tensor(key)
            stem = key[: -len(".weight")] if key.endswith(".weight") else None
            if stem in lora:
                a, b = lora[stem]
                t = t.float() + scale * (b.float() @ a.float())
            return t
        entries.append((key, shape, dtype, produce))
    meta = dict(meta or {})
    if merge is not None:
        meta["nomas_merged_lora"] = os.path.basename(os.path.normpath(merge))
        missing = set(lora) - {k[:-7] for k in f.keys() if k.endswith(".weight")}
        if missing:
            raise ValueError(f"LoRA modules not in the base: {sorted(missing)[:5]}")
    write_safetensors(dst, entries, meta)


def convert_lora(src, dst):
    lora, scale, cfg = load_adapter(src)
    entries = []
    for module, (a, b) in sorted(lora.items()):
        p = f"diffusion_model.{module}"
        entries.append((f"{p}.lora_A.weight", tuple(a.shape), torch.bfloat16, lambda a=a: a))
        entries.append((f"{p}.lora_B.weight", tuple(b.shape), torch.bfloat16, lambda b=b: b))
        alpha = torch.tensor(scale * a.shape[0])
        entries.append((f"{p}.alpha", (), torch.float32, lambda alpha=alpha: alpha))
    write_safetensors(dst, entries, {"nomas_lora": os.path.basename(os.path.normpath(src)), "rank": cfg.get("r")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["dit", "vae", "lora"])
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--merge", help="PEFT adapter dir to merge into the DiT")
    args = ap.parse_args()
    if args.kind == "lora":
        convert_lora(args.src, args.dst)
    else:
        convert_float(args.src, args.dst, torch.bfloat16, args.merge, {"nomas_source": os.path.basename(args.src)})
    print("wrote", args.dst)


if __name__ == "__main__":
    main()
