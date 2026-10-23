"""Registers Hy-Nomas with ComfyUI's native loaders.

After ``register()``:
  Load Diffusion Model -> detects the Nomas DiT (bf16, fp32 or int8 ConvRot files with the
                          released key names) and returns a MODEL
  Load VAE             -> detects the PixelUMM decoder file (with or without the DCAE-V
                          encoder it ships with); VAE Decode turns Nomas latents
                          [B, 32, L, h, w] into 4L frames of 32h x 32w
  Load CLIP            -> the text encoder is ComfyUI's own Qwen2.5-VL-7B (type qwen_image)
Nothing in ComfyUI is replaced: detection is chained in front of the built-in function, the
model config is appended to ``comfy.supported_models.models``, and the VAE constructor is
wrapped to build the PixelUMM decoder when (and only when) it sees one.
"""

import logging

import torch

import comfy.latent_formats
import comfy.model_base
import comfy.model_detection
import comfy.model_management
import comfy.ops
import comfy.sd
import comfy.supported_models
import comfy.supported_models_base

from . import pixelumm
from .dit import NomasDiT


class NomasLatent(comfy.latent_formats.LatentFormat):
    latent_channels = 32
    latent_dimensions = 3
    scale_factor = 1.0


class NomasModel(comfy.model_base.BaseModel):
    """The Nomas DiT under ComfyUI's model management (offload, dtypes, LoRA). A world is
    generated segment by segment with a cache, so sampling is driven by the Nomas nodes."""

    def __init__(self, model_config, model_type=comfy.model_base.ModelType.FLOW, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=NomasDiT)


class NomasConfig(comfy.supported_models_base.BASE):
    unet_config = {"image_model": "hy_nomas"}
    unet_extra_config = {}
    sampling_settings = {"shift": 5.0, "multiplier": 1000.0}
    latent_format = NomasLatent
    memory_usage_factor = 4.0
    supported_inference_dtypes = [torch.bfloat16, torch.float32]

    def get_model(self, state_dict, prefix="", device=None):
        return NomasModel(self, device=device)


def detect_nomas_config(state_dict, key_prefix="", metadata=None):
    keys = state_dict.keys()
    p = key_prefix
    if f"{p}double_blocks.0.img_attn_q.bias" not in keys or f"{p}txt_in.c_embedder.linear_1.bias" not in keys:
        return None
    if f"{p}double_blocks.0.img_attn_q.weight_q" in keys:
        raise ValueError("This is Tencent's own int8 Nomas file (nomas-int8-bf16-single-v1), which ComfyUI can't load. "
                         "Use the bf16 or int8_convrot files made with ComfyUI-Nomas/scripts (see its README).")
    return {
        "image_model": "hy_nomas",
        "in_channels": state_dict[f"{p}img_in.proj.weight"].shape[1],
        "hidden_size": state_dict[f"{p}img_in.proj.weight"].shape[0],
        "depth": comfy.model_detection.count_blocks(keys, f"{p}double_blocks." + "{}."),
        "text_dim": state_dict[f"{p}txt_in.input_embedder.weight"].shape[1],
        "camera": f"{p}action_modules.0.proj_keyboard.weight" in keys,
    }


# ---- VAE -------------------------------------------------------------------------------------

def is_pixelumm(sd):
    return sd is not None and "decoder.cond_proj.weight" in sd and "decoder.layers.0.attn.q_norm_t.weight" in sd


class PixelUMMVAE(torch.nn.Module):
    """ComfyUI's ``first_stage_model`` for Nomas: decode = PixelUMM over 10-latent windows."""

    comfy_decode_estimate_is_reliable = False

    def __init__(self, seed=pixelumm.NOISE_SEED):
        super().__init__()
        self.decoder = pixelumm.PixelUMMDecoder(dtype=torch.bfloat16, operations=comfy.ops.manual_cast)
        self.seed = seed

    @torch.no_grad()
    def decode(self, z, **kwargs):
        return pixelumm.decode_clip(self.decoder, z, seed=kwargs.get("seed", self.seed))

    def encode(self, x):
        raise NotImplementedError("Nomas has no image-to-world input; the DCAE-V encoder is not ported yet")


def setup_pixelumm(vae):
    vae.latent_channels = 32
    vae.latent_dim = 3
    vae.output_channels = 3
    vae.upscale_ratio = (lambda a: a * 4, 32, 32)
    vae.downscale_ratio = (lambda a: a // 4, 32, 32)
    vae.upscale_index_formula = (4, 32, 32)
    vae.downscale_index_formula = (4, 32, 32)
    vae.working_dtypes = [torch.bfloat16, torch.float32]
    vae.handles_tiling = True
    # one 10-latent window at a time, whatever the clip length
    vae.memory_used_decode = lambda shape, dtype: (min(shape[2], pixelumm.WINDOW) * shape[3] * shape[4] * 4 * 4096 * 2 * 14
                                                   + 512 * 1024 * 1024)
    vae.memory_used_encode = lambda shape, dtype: 0
    # PixelUMM already decodes window by window; ComfyUI's spatial tiler would break its attention
    vae._decode_tiled_owned = lambda samples, **kw: vae.first_stage_model.decode(
        samples.to(vae.device, vae.vae_dtype)).to(vae.output_device, vae.vae_output_dtype()).add_(1.0).div_(2.0).clamp_(0.0, 1.0)


def register():
    detect = comfy.model_detection.detect_unet_config
    if not getattr(detect, "_nomas", False):
        def detect_unet_config(state_dict, key_prefix, metadata=None):
            config = detect_nomas_config(state_dict, key_prefix, metadata)
            if config is not None:
                return config
            return detect(state_dict, key_prefix, metadata=metadata)
        detect_unet_config._nomas = True
        comfy.model_detection.detect_unet_config = detect_unet_config

    if not any(getattr(m, "unet_config", {}).get("image_model") == "hy_nomas" for m in comfy.supported_models.models):
        comfy.supported_models.models.insert(0, NomasConfig)

    init = comfy.sd.VAE.__init__
    if not getattr(init, "_nomas", False):
        def vae_init(self, sd=None, device=None, config=None, dtype=None, metadata=None):
            if config is not None or not is_pixelumm(sd):
                return init(self, sd=sd, device=device, config=config, dtype=dtype, metadata=metadata)
            # the PixelUMM file also carries the frozen DCAE-V encoder (encoder.*), unused for now
            sd = {k: v for k, v in sd.items() if k.startswith("decoder.")}
            autoencoder = comfy.sd.AutoencoderKL
            comfy.sd.AutoencoderKL = lambda **params: PixelUMMVAE()
            try:
                init(self, sd=sd, device=device, config={"params": {}}, dtype=dtype, metadata=metadata)
            finally:
                comfy.sd.AutoencoderKL = autoencoder
            setup_pixelumm(self)
        vae_init._nomas = True
        comfy.sd.VAE.__init__ = vae_init

    logging.info("Hy-Nomas: registered the DiT with Load Diffusion Model and the PixelUMM decoder with Load VAE")
