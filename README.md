# ComfyUI-HyNomas

Native ComfyUI support for **[Hy-Nomas](https://github.com/Tencent-Hunyuan/Hy-Nomas)**, Tencent Hunyuan's interactive
world model. It generates a world segment by segment: each segment attends to a cache of the earlier ones, and you
steer it with text, with a camera move (WASD / arrows / mouse), or with both.

- **No requirements.** Pure ComfyUI and torch. The reference runtime's MagiAttention, FlashAttention 3, CUTLASS and
  DeepGEMM kernels are not needed.
- **Stock loaders.**
  - **Load Diffusion Model** loads the DiT.
  - **Load CLIP** loads ComfyUI's own Qwen2.5-VL-7B (type `qwen_image`).
  - **Load VAE** loads the PixelUMM decoder, and the stock **VAE Decode** decodes the latents.
  - Offloading, dtype options and int8 work as for any model.
- **Nomas Explore** walks through a world interactively. Click its viewer and hold W/A/S/D and the arrow keys (or
  drag). The move is recorded at 24 fps and queued, and the next turn plays back in the node.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Nynxz/ComfyUI-HyNomas
```

## Models

From [nynxz/Hy-Nomas](https://huggingface.co/nynxz/Hy-Nomas):

| File | Folder | Size |
|---|---|---|
| `nomas_dit_distilled_int8_convrot.safetensors` | `models/diffusion_models` | 9.0 GB |
| `nomas_dit_distilled_bf16.safetensors` (alternative) | `models/diffusion_models` | 17.1 GB |
| `nomas_pixelumm_vae_bf16.safetensors` | `models/vae` | 15.8 GB |
| `qwen_2.5_vl_7b.safetensors` (or `_fp8_scaled`) | `models/text_encoders` | from [Comfy-Org/Qwen-Image_ComfyUI](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI) |

On an RTX 5090 with the int8 DiT, a 3 s turn at 1280×704 takes about 21 s (8 steps plus the cache write). Decoding
takes about 4 s per 10-latent window.

## Workflows

In [`example_workflows`](example_workflows), also under **Templates → Custom Nodes**:

- [`nomas_explore.json`](example_workflows/nomas_explore.json): the interactive explorer.
- [`nomas_world.json`](example_workflows/nomas_world.json): a scripted world: an opening turn with a camera path, a
  camera turn and a text turn, decoded to a video.

## Nodes

- **Nomas World Prompt**: the opening turn, in the format Hy-Nomas's prompt enhancer writes. It takes a perspective,
  a style, one subject (written `<|Subject1|>` in the text), a scene description, an optional action, and a length.
- **Nomas Prompt (JSON)**: a whole multi-segment prompt with timestamps, as in the release's `examples/prompts.json`.
- **Nomas Camera Path**: a camera move as keys held for a time, one `<keys> <seconds>` per line (`w 2`,
  `w+left 1.5`, `wait 1`). `look <dx> <dy> <seconds>` is a mouse drag.
- **Nomas World**: generates the opening turn(s). It outputs a LATENT (the whole world) and a WORLD to continue.
- **Nomas World Continue**: adds a turn. The turn can be a change in words ("a wooden bridge crosses a stream
  ahead"), an action, a camera path, or a mix of them.
- **Nomas Explore**: the interactive node.
  - While you record a move, a live 3D fly-through (grid floor, back wall, subject marker) shows it as you make it,
    using the same movement rules the server integrates.
  - The timeline plays the whole world back to back, one coloured segment per turn (opening, camera, text, both),
    with a playhead to scrub, and the current turn's label underneath.
  - A corner view draws the camera's path through the world in 3D, with the camera's frustum following playback.
  - **Text turn** runs a turn from the change / action fields, **Undo** drops the last turn, **New world** starts over,
    and **Save video** writes the whole world to `output/video` as an mp4 (from the frames already decoded).
  - Each turn decodes only its new frames, plus 5 latents of overlap cross-faded into the previous turn.

**Camera.** Only the direction and the relative speed of a move reach the model, because translations are normalised
per turn. The turning speed does matter: rotations are not normalised, and the model scales them by 2.5×. A camera
path sets the turn's length.

**History.** In `window` mode the cache keeps 18 latent frames of the past (always the first and the newest, denser
near the present), as in the reference's single-GPU setting. `full` mode keeps everything, which costs about 0.2 GB of
VRAM per latent frame. The cache is int8, one scale per token and head. It is held in memory-mapped files in
ComfyUI's temp folder, so a long world costs page cache rather than RAM.

**Resolution.** The model was trained at 1280×704, which is the release's "720p". There is no image-to-world input:
Hy-Nomas starts every world from text.

## How it works

- **The DiT** is a HunyuanVideo-1.5-sized MMDiT: 54 double blocks, hidden 2048, 3-D RoPE on the image tokens only.
  - It runs one segment per call. The current segment is fully bidirectional, and earlier segments come from the
    cache.
  - After a segment is denoised (8 Euler steps on a shift-5 flow schedule), its clean latent is re-noised to
    σ = 6/29. One more pass then writes its keys and values into the cache.
  - Later segments see the earlier segments' video and the part of their text before `contained_actions`.
  - Inside a segment, the video in an action's time window sees only that action's text.
  - The camera enters each block as a per-frame residual computed from four 4×4 poses per latent frame.
- **The PixelUMM decoder** is a 36-layer, Qwen3-8B-shaped transformer. It predicts clean pixels from noise and the
  latent in a single step.
  - Each token is a 4×16×16 RGB tubelet, and each latent cell gives 2×2 tokens.
  - Attention is local in space (8×8-token tiles, each seeing its 3×3 neighbourhood) and full in time.
  - Clips are decoded in 10-latent windows at stride 5 and cross-faded.
- **The text encoder** is Qwen2.5-VL-7B's layer-25 output on a fixed video-JSON chat template. ComfyUI's Qwen-Image
  text encoder returns exactly that with `clip_layer(-3)`.

The int8 DiT quantizes the double-block linears with int8 + ConvRot, at 0.86% mean and 1.13% max weight error. The
camera branch, text refiner, embeddings and final layer stay in bf16.

## Verification

| Check (against the reference code) | Result |
|---|---|
| DiT, 2 real blocks, bf16: first segment, cache write, then a segment with history, camera and two-action masks | cos 0.99998, max 0.7% (bf16 rounding); camera tokens bit-identical |
| PixelUMM decoder, 2 real layers: full attention / sliding-tile local attention | bit-identical / cos 0.999997 |
| Windowed decode with cosine blending | bit-identical |
| Camera conditioning (resampling, framewise normalisation, rotation gain) | ≤ 1e-5 |
| Per-segment prompt strings, history frame selection | identical |

```bash
cd tests
PYTHONPATH=<ComfyUI>:.. NOMAS_SRC=<Hy-Nomas checkout> python -m pytest -q test_units.py
COMFY=<ComfyUI> NOMAS_SRC=<Hy-Nomas checkout> NOMAS_DIT=<fp32 DiT> python parity_dit.py      # CUDA
COMFY=<ComfyUI> NOMAS_SRC=<Hy-Nomas checkout> NOMAS_VAE=<PixelUMM_FT> python parity_vae.py   # CUDA
python e2e.py --comfy <ComfyUI> --model <dit> --clip <qwen_2.5_vl_7b> --vae <pixelumm> --out out
```

`parity_dit.py` runs the reference DiT on stock torch by stubbing its CUDA-only imports (`ref_stubs.py`).

## Making the weights

```bash
hf download tencent/Hy-Nomas --include "DiT/distilled/*" "VAE/PixelUMM_FT/*" --local-dir Hy-Nomas
python scripts/convert.py dit Hy-Nomas/DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors nomas_dit_distilled_bf16.safetensors
python scripts/convert.py vae Hy-Nomas/VAE/PixelUMM_FT/diffusion_pytorch_model.safetensors nomas_pixelumm_vae_bf16.safetensors
# int8 + ConvRot (comfy-model-tools' quant_int8_convrot.py recipe; run with ComfyUI's python for comfy-kitchen)
python scripts/quantize_int8.py dit nomas_dit_distilled_bf16.safetensors nomas_dit_distilled_int8_convrot.safetensors --tool <comfy-model-tools>
# LoRAs: merge into the DiT (--merge <adapter dir>), or convert to a ComfyUI LoRA
python scripts/convert.py lora Hy-Nomas/DiT/lora_weights/cinematic nomas_lora_cinematic.safetensors
```

Tencent's own int8 file (`distilled_int8`) uses a different format, which this pack refuses with a message pointing
here.

## Credits & licence

Hy-Nomas by [Tencent Hunyuan](https://github.com/Tencent-Hunyuan/Hy-Nomas) (Apache-2.0). This pack reimplements the
model code on ComfyUI's building blocks and is licensed Apache-2.0; see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
The PixelUMM decoder weights are a fine-tune of NVIDIA PixelUMM and are under the **NVIDIA One-Way Noncommercial
License** (non-commercial use only).
