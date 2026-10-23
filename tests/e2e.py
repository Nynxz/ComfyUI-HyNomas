"""End-to-end: stock loaders -> Nomas World (+ a camera turn) -> VAE Decode -> mp4, outside the server.

    python tests/e2e.py --comfy <ComfyUI> --model <dit .safetensors> --clip <qwen_2.5_vl_7b .safetensors> \
        --vae <pixelumm .safetensors> --out out [--width 640 --height 352 --seconds 2]
"""

import argparse
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--comfy", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--clip", required=True)
ap.add_argument("--vae", required=True)
ap.add_argument("--out", default="out")
ap.add_argument("--width", type=int, default=1280)
ap.add_argument("--height", type=int, default=704)
ap.add_argument("--seconds", type=float, default=3.0)
ap.add_argument("--camera", default="w 1.5\nw+left 1.5")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()
sys.path[:0] = [args.comfy, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")]

import torch  # noqa: E402

import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402
from nomas.integration import register  # noqa: E402

register()
from nomas import nodes  # noqa: E402

os.makedirs(args.out, exist_ok=True)
t0 = time.time()
model = comfy.sd.load_diffusion_model(args.model)
clip = comfy.sd.load_clip(ckpt_paths=[args.clip], clip_type=comfy.sd.CLIPType.QWEN_IMAGE)
vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(args.vae))
print(f"loaded in {time.time() - t0:.0f}s: {type(model.model.diffusion_model).__name__}, vae {type(vae.first_stage_model).__name__}")

turn = nodes.NomasWorldPrompt.execute("third-person follow", "real-world footage", "a small orange fox",
                                      "Orange fur, a white chest and a fluffy tail.",
                                      "<|Subject1|> stands on a stone path in a luminous forest at dusk. Blue fireflies drift "
                                      "between tall mossy trees, and soft mist hangs over the ground.",
                                      "<|Subject1|> walks slowly along the path.", args.seconds).result[0]
t0 = time.time()
latent, world = nodes.NomasWorldGenerate.execute(model, clip, turn, args.seed, args.width, args.height, 8, "window").result
print(f"turn 1: {tuple(latent['samples'].shape)} in {time.time() - t0:.0f}s")
t0 = time.time()
ticks = nodes.NomasCameraPath.execute(args.camera).result[0]
latent, world = nodes.NomasWorldContinue.execute(model, clip, world, "", "", args.seconds, 8, ticks).result
print(f"turn 2 (camera): {tuple(latent['samples'].shape)} in {time.time() - t0:.0f}s")
t0 = time.time()
latent, world = nodes.NomasWorldContinue.execute(model, clip, world, "a wooden bridge appears over a glowing stream ahead",
                                                 "<|Subject1|> trots onto the bridge.", args.seconds, 8).result
print(f"turn 3 (text): {tuple(latent['samples'].shape)} in {time.time() - t0:.0f}s")
torch.save(latent["samples"], os.path.join(args.out, "latent.pt"))

t0 = time.time()
frames = vae.decode(latent["samples"])
print(f"decoded {tuple(frames.shape)} in {time.time() - t0:.0f}s")
frames = frames.reshape(-1, *frames.shape[-3:])
import av  # noqa: E402
import numpy as np  # noqa: E402

path = os.path.join(args.out, "world.mp4")
with av.open(path, "w") as c:
    s = c.add_stream("libx264", rate=24)
    s.width, s.height, s.pix_fmt = frames.shape[2], frames.shape[1], "yuv420p"
    s.options = {"crf": "18"}
    for f in (frames.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy():
        c.mux(s.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format="rgb24")))
    c.mux(s.encode())
print("wrote", path)
