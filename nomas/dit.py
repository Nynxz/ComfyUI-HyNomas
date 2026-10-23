"""Hy-Nomas DiT (HY-Nomas-G/2-32324-MMDiT) on ComfyUI ops.

Reimplements the inference path of hymm/models/dit/nomas_cpt.py (``forward_with_cache``) of
Tencent-Hunyuan/Hy-Nomas: a HunyuanVideo-1.5-sized MMDiT (54 double blocks, hidden 2048,
16 heads, 3-D RoPE on image tokens only) run one prompt segment per call against a cache of
earlier segments, with a camera residual added to the image stream after every block.

Three exact simplifications of the reference:

* Text padding is dropped. The reference pads every segment's text to 1280 tokens and masks
  the padding as keys everywhere, and text has no RoPE, so only the valid tokens matter.
* The history holds only keys the reference leaves visible (the selected frames' image tokens
  and each segment's head text), so attention needs no mask. Intra-segment action masking
  (video in an action window sees only the head text and that action's text) is done by
  grouping queries by the text they may see and running one attention per group.
* The camera cross-attention has one key per frame (per-chunk, chunk size 1), so its softmax is
  1 and it reduces to ``img[frame f] += proj_keyboard(V(keyboard_embed(cam_f)))``.
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from comfy.ldm.flux.layers import EmbedND, timestep_embedding
from comfy.ldm.flux.math import apply_rope1
from comfy.ldm.modules.attention import optimized_attention

EPS = 1e-6


def rms_norm(x, weight, eps=EPS):
    """The reference RMSNorm: normalise in fp32, cast back, then scale."""
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
    return y.to(x.dtype) * weight.to(x.dtype)


def modulate_ln(x, shift, scale):
    """LayerNorm (no affine, eps 1e-6) then ``x*(1+scale)+shift``, in fp32 as under autocast."""
    y = F.layer_norm(x.float(), (x.shape[-1],), eps=EPS)
    return (y * (1 + scale.float()[:, None]) + shift.float()[:, None]).to(x.dtype)


class QKNorm(nn.Module):
    def __init__(self, dim, dtype=None, device=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim, dtype=dtype, device=device))

    def forward(self, x):
        import comfy.model_management
        return rms_norm(x, comfy.model_management.cast_to_device(self.weight, x.device, x.dtype))


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden, dtype=None, device=None, operations=None):
        super().__init__()
        self.mlp = nn.Sequential(operations.Linear(256, hidden, dtype=dtype, device=device), nn.SiLU(),
                                 operations.Linear(hidden, hidden, dtype=dtype, device=device))

    def forward(self, t, dtype):
        # t is already sigma * 1000; cos before sin, as in the reference
        return self.mlp(timestep_embedding(t, 256, time_factor=1.0).to(dtype))


class TextProjection(nn.Module):
    def __init__(self, in_dim, hidden, dtype=None, device=None, operations=None):
        super().__init__()
        self.linear_1 = operations.Linear(in_dim, hidden, dtype=dtype, device=device)
        self.linear_2 = operations.Linear(hidden, hidden, dtype=dtype, device=device)

    def forward(self, x):
        return self.linear_2(F.silu(self.linear_1(x)))


class RefinerBlock(nn.Module):
    def __init__(self, hidden, heads, dtype=None, device=None, operations=None):
        super().__init__()
        self.heads = heads
        self.norm1 = operations.LayerNorm(hidden, elementwise_affine=True, eps=EPS, dtype=dtype, device=device)
        self.self_attn_qkv = operations.Linear(hidden, hidden * 3, dtype=dtype, device=device)
        self.self_attn_proj = operations.Linear(hidden, hidden, dtype=dtype, device=device)
        self.norm2 = operations.LayerNorm(hidden, elementwise_affine=True, eps=EPS, dtype=dtype, device=device)
        self.mlp = nn.Module()
        self.mlp.fc1 = operations.Linear(hidden, hidden * 4, dtype=dtype, device=device)
        self.mlp.fc2 = operations.Linear(hidden * 4, hidden, dtype=dtype, device=device)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), operations.Linear(hidden, hidden * 2, dtype=dtype, device=device))

    def forward(self, x, c, transformer_options={}):
        gate_msa, gate_mlp = self.adaLN_modulation(c).chunk(2, dim=1)
        b, n, _ = x.shape
        qkv = self.self_attn_qkv(self.norm1(x)).view(b, n, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        a = optimized_attention(qkv[0], qkv[1], qkv[2], self.heads, skip_reshape=True, transformer_options=transformer_options)
        x = x + gate_msa[:, None] * self.self_attn_proj(a)
        return x + gate_mlp[:, None] * self.mlp.fc2(F.silu(self.mlp.fc1(self.norm2(x))))


class TokenRefiner(nn.Module):
    """``txt_in`` (SingleTokenRefinerPerBlock on the 1-D timestep path)."""

    def __init__(self, text_dim, hidden, heads, depth=2, dtype=None, device=None, operations=None):
        super().__init__()
        self.input_embedder = operations.Linear(text_dim, hidden, dtype=dtype, device=device)
        self.t_embedder = TimestepEmbedder(hidden, dtype=dtype, device=device, operations=operations)
        self.c_embedder = TextProjection(text_dim, hidden, dtype=dtype, device=device, operations=operations)
        self.individual_token_refiner = nn.Module()
        self.individual_token_refiner.blocks = nn.ModuleList(
            RefinerBlock(hidden, heads, dtype=dtype, device=device, operations=operations) for _ in range(depth))

    def forward(self, x, t, transformer_options={}):
        # x holds only the valid tokens, so the reference's masked mean is a plain mean (fp32)
        c = self.t_embedder(t, x.dtype) + self.c_embedder(x.float().mean(1).to(x.dtype))
        x = self.input_embedder(x)
        for block in self.individual_token_refiner.blocks:
            x = block(x, c, transformer_options)
        return x


class MLP(nn.Module):
    def __init__(self, hidden, width, dtype=None, device=None, operations=None):
        super().__init__()
        self.fc1 = operations.Linear(hidden, width, dtype=dtype, device=device)
        self.fc2 = operations.Linear(width, hidden, dtype=dtype, device=device)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class Modulation(nn.Module):
    def __init__(self, hidden, factor, dtype=None, device=None, operations=None):
        super().__init__()
        self.linear = operations.Linear(hidden, hidden * factor, dtype=dtype, device=device)

    def forward(self, vec):
        return self.linear(F.silu(vec))


@dataclass
class Attend:
    """How one call's attention is laid out.

    ``groups`` is None for plain attention (every query sees every key), or a list of
    ``(query_index, text_keep)``: the query rows (image then text, over the call's
    ``image + text`` tokens) and the current-text key rows they may see.
    """
    groups: list = None


@dataclass
class History:
    """Per block, the earlier segments' visible keys and values ``[B, heads, N, D]``, or None."""
    k: list = field(default_factory=list)
    v: list = field(default_factory=list)

    def get(self, i):
        return (self.k[i], self.v[i]) if self.k else (None, None)


class DoubleBlock(nn.Module):
    def __init__(self, hidden, heads, mlp_ratio=4, dtype=None, device=None, operations=None):
        super().__init__()
        self.heads = heads
        dh = hidden // heads
        for side in ("img", "txt"):
            setattr(self, f"{side}_mod", Modulation(hidden, 6, dtype=dtype, device=device, operations=operations))
            for p in ("q", "k", "v"):
                setattr(self, f"{side}_attn_{p}", operations.Linear(hidden, hidden, dtype=dtype, device=device))
            setattr(self, f"{side}_attn_q_norm", QKNorm(dh, dtype=dtype, device=device))
            setattr(self, f"{side}_attn_k_norm", QKNorm(dh, dtype=dtype, device=device))
            setattr(self, f"{side}_attn_proj", operations.Linear(hidden, hidden, dtype=dtype, device=device))
            setattr(self, f"{side}_mlp", MLP(hidden, hidden * mlp_ratio, dtype=dtype, device=device, operations=operations))

    def qkv(self, side, x, shift, scale):
        b, n, c = x.shape
        h = modulate_ln(x, shift, scale)
        q, k, v = (getattr(self, f"{side}_attn_{p}")(h).view(b, n, self.heads, -1).transpose(1, 2) for p in "qkv")
        return getattr(self, f"{side}_attn_q_norm")(q), getattr(self, f"{side}_attn_k_norm")(k), v

    def forward(self, img, txt, vec, pe, attend, history, on_kv=None, transformer_options={}):
        s1, sc1, g1, s2, sc2, g2 = self.img_mod(vec).chunk(6, dim=-1)
        ts1, tsc1, tg1, ts2, tsc2, tg2 = self.txt_mod(vec).chunk(6, dim=-1)
        iq, ik, iv = self.qkv("img", img, s1, sc1)
        iq, ik = apply_rope1(iq, pe), apply_rope1(ik, pe)
        tq, tk, tv = self.qkv("txt", txt, ts1, tsc1)
        if on_kv is not None:
            on_kv(ik, iv, tk, tv)

        hk, hv = history
        q = torch.cat((iq, tq), dim=2)
        k = torch.cat(([hk] if hk is not None else []) + [ik, tk], dim=2)
        v = torch.cat(([hv] if hv is not None else []) + [iv, tv], dim=2)
        del iq, tq
        if attend.groups is None:
            out = optimized_attention(q, k, v, self.heads, skip_reshape=True, transformer_options=transformer_options)
        else:
            n_fixed = k.shape[2] - tk.shape[2]
            out = q.new_zeros(q.shape[0], q.shape[2], q.shape[1] * q.shape[3])
            for rows, text_keep in attend.groups:
                keys = torch.cat((torch.arange(n_fixed, device=k.device), n_fixed + text_keep))
                out[:, rows] = optimized_attention(q[:, :, rows], k[:, :, keys], v[:, :, keys], self.heads,
                                                   skip_reshape=True, transformer_options=transformer_options)
        del q, k, v
        n_img = img.shape[1]
        img = img + g1[:, None] * self.img_attn_proj(out[:, :n_img])
        img = img + g2[:, None] * self.img_mlp(modulate_ln(img, s2, sc2))
        txt = txt + tg1[:, None] * self.txt_attn_proj(out[:, n_img:])
        txt = txt + tg2[:, None] * self.txt_mlp(modulate_ln(txt, ts2, tsc2))
        return img, txt


class ActionModule(nn.Module):
    """The camera branch of one block. Only the V half of ``keyboard_attn_kv`` and
    ``proj_keyboard`` reach the output (see the module docstring); the other weights are
    kept so the checkpoint loads strictly."""

    def __init__(self, hidden, cam_dim=64, cam_hidden=128, kb_hidden=1024, heads=16, dtype=None, device=None, operations=None):
        super().__init__()
        self.keyboard_embed = nn.Sequential(operations.Linear(cam_dim, cam_hidden, dtype=dtype, device=device), nn.SiLU(),
                                            operations.Linear(cam_hidden, cam_hidden, dtype=dtype, device=device))
        self.keyboard_attn_kv = operations.Linear(cam_hidden, 2 * kb_hidden, bias=False, dtype=dtype, device=device)
        self.img_attn_q = operations.Linear(hidden, kb_hidden, bias=False, dtype=dtype, device=device)
        self.key_attn_q_norm = QKNorm(kb_hidden // heads, dtype=dtype, device=device)
        self.key_attn_k_norm = QKNorm(kb_hidden // heads, dtype=dtype, device=device)
        self.proj_keyboard = operations.Linear(kb_hidden, hidden, bias=False, dtype=dtype, device=device)
        self.kb_hidden = kb_hidden

    def residual(self, cam):
        """``cam [B, T, 64]`` -> the per-frame image residual ``[B, T, hidden]``."""
        value = self.keyboard_attn_kv(self.keyboard_embed(cam))[..., self.kb_hidden:]
        return self.proj_keyboard(value)


class FinalLayer(nn.Module):
    def __init__(self, hidden, out_channels, dtype=None, device=None, operations=None):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), operations.Linear(hidden, 2 * hidden, dtype=dtype, device=device))
        self.linear = operations.Linear(hidden, out_channels, dtype=dtype, device=device)

    def forward(self, x, vec):
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        return self.linear(modulate_ln(x, shift, scale))


class NomasDiT(nn.Module):
    def __init__(self, image_model=None, in_channels=32, out_channels=32, hidden_size=2048, num_heads=16, depth=54,
                 mlp_ratio=4, text_dim=3584, axes_dim=(16, 56, 56), theta=256, camera=True,
                 dtype=None, device=None, operations=None, **kwargs):
        super().__init__()
        self.dtype = dtype
        self.hidden_size, self.num_heads = hidden_size, num_heads
        self.out_channels = out_channels
        self.img_in = nn.Module()
        self.img_in.proj = operations.Conv3d(in_channels, hidden_size, kernel_size=1, dtype=dtype, device=device)
        self.txt_in = TokenRefiner(text_dim, hidden_size, num_heads, dtype=dtype, device=device, operations=operations)
        self.time_in = TimestepEmbedder(hidden_size, dtype=dtype, device=device, operations=operations)
        self.double_blocks = nn.ModuleList(
            DoubleBlock(hidden_size, num_heads, mlp_ratio, dtype=dtype, device=device, operations=operations) for _ in range(depth))
        self.final_layer = FinalLayer(hidden_size, out_channels, dtype=dtype, device=device, operations=operations)
        self.pe_embedder = EmbedND(dim=hidden_size // num_heads, theta=theta, axes_dim=list(axes_dim))
        self.camera = camera
        if camera:
            self.extrinsics_embedding_camctrl = operations.Linear(64, 64, dtype=dtype, device=device)
            self.c2ws_hidden_states_layer1 = operations.Linear(64, 64, dtype=dtype, device=device)
            self.c2ws_hidden_states_layer2 = operations.Linear(64, 64, dtype=dtype, device=device)
            self.action_modules = nn.ModuleList(
                ActionModule(hidden_size, dtype=dtype, device=device, operations=operations) for _ in range(depth))

    # -- camera -------------------------------------------------------------------------------

    def camera_tokens(self, extrinsics):
        """``[B, F, 4, 4]`` normalised c2w poses (F pixel frames) -> ``[B, ceil(F/4), 64]``.

        Front-pads with the first pose to a multiple of 4, then packs 4 consecutive row-major
        poses per latent frame (latent 0 = [P0]*4, latent i = P[4i-3 .. 4i]).
        """
        dtype = self.dtype or torch.bfloat16
        e = extrinsics.to(dtype)
        b, f = e.shape[:2]
        pad = (4 - f % 4) % 4
        e = torch.cat((e[:, :1].repeat(1, pad, 1, 1), e), dim=1).reshape(b, -1, 64)
        e = self.extrinsics_embedding_camctrl(e)
        return e + self.c2ws_hidden_states_layer2(F.silu(self.c2ws_hidden_states_layer1(e)))

    # -- forward --------------------------------------------------------------------------------

    def img_ids(self, t, h, w, frame_offset, device):
        ids = torch.zeros(t, h, w, 3, device=device, dtype=torch.float32)
        ids[..., 0] = torch.arange(frame_offset, frame_offset + t, device=device, dtype=torch.float32)[:, None, None]
        ids[..., 1] = torch.arange(h, device=device, dtype=torch.float32)[None, :, None]
        ids[..., 2] = torch.arange(w, device=device, dtype=torch.float32)[None, None, :]
        return ids.reshape(1, -1, 3)

    def forward_segment(self, x, timestep, text, frame_offset=0, camera=None, camera_on=True, attend=None,
                        history=None, on_kv=None, transformer_options={}):
        """One denoising (or cache-write) call on a whole segment.

        x          [B, 32, T, h, w] latent of the current segment
        timestep   [B] sigma * 1000
        text       [B, S, 3584] the segment's Qwen2.5-VL states, valid tokens only
        frame_offset  absolute latent index of the segment's first frame (RoPE time)
        camera     [B, T, 64] camera tokens for these frames (``camera_tokens``), or None
        attend     an :class:`Attend` (intra-segment action windows), or None for plain
        history    a :class:`History` of the earlier segments, or None
        on_kv      ``on_kv(block, img_k, img_v, txt_k, txt_v)`` receives each block's keys (after
                   QK-norm and RoPE) and values ``[B, heads, N, D]`` (cache-write pass)
        Returns the flow velocity ``[B, 32, T, h, w]``.
        """
        b, _, t, h, w = x.shape
        attend = attend or Attend()
        history = history or History()
        img = self.img_in.proj(x).flatten(2).transpose(1, 2)
        dtype = img.dtype
        vec = self.time_in(timestep, dtype)
        txt = self.txt_in(text.to(dtype), timestep, transformer_options)
        pe = self.pe_embedder(self.img_ids(t, h, w, frame_offset, x.device))

        cam = None
        if self.camera and camera is not None and camera_on:
            cam = camera.to(device=x.device, dtype=dtype)
        for i, block in enumerate(self.double_blocks):
            kv = None if on_kv is None else (lambda *a, i=i: on_kv(i, *a))
            img, txt = block(img, txt, vec, pe, attend, history.get(i), kv, transformer_options)
            if cam is not None:
                res = self.action_modules[i].residual(cam)  # [B, T, hidden]
                img = (img.view(b, t, h * w, -1) + res[:, :, None]).view(b, t * h * w, -1)

        out = self.final_layer(img, vec)
        return out.view(b, t, h, w, -1).permute(0, 4, 1, 2, 3)

    def forward(self, x, timestep, context, transformer_options={}, **kwargs):
        """Plain single-segment call (no history, no camera), e.g. from a stock sampler."""
        return self.forward_segment(x, timestep, context, camera=kwargs.get("nomas_camera"),
                                    transformer_options=transformer_options)
