"""Hy-Nomas PixelUMM decoder (DCAE-V f32t4c32 latents -> RGB video) on ComfyUI ops.

Reimplements hymm/models/vae/pixelumm_decoder/{model,autoencoder,streaming}.py of
Tencent-Hunyuan/Hy-Nomas: a Qwen3-8B-shaped transformer (36 layers, hidden 4096, 32 query /
8 KV heads) that predicts clean pixels x0 from a noisy pixel clip ``xt`` and the latent, one
4x16x16 RGB tubelet per token and 2x2 tokens per latent cell. Decoding is one call per
10-latent window on pure noise (the distilled 1-step setting), windows at stride 5 plus a
right-aligned tail, merged with a cosine cross-fade.

Attention is sliding-tile local in space (8x8-token tiles, each query tile sees the 3x3 tiles
around it, shifted inward at the border) and full in time. The reference runs it with
FlexAttention on a tile-major permutation; here each query tile gathers its keys and tiles of
equal shape are batched, which is the same attention without a mask.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.model_management
from comfy.ldm.modules.attention import optimized_attention

TUBELET = (4, 16, 16)
TUBE_DIM = 4 * 16 * 16 * 3
LATENT_CHANNELS = 32
GRID = 2  # tokens per latent cell along each spatial axis
HIDDEN, INTER, LAYERS, HEADS, KV_HEADS, HEAD_DIM = 4096, 12288, 36, 32, 8, 128
EPS = 1e-6
THETA_T, THETA_HW = 1_000_000.0, 10_000.0
SECTIONS = (64, 32, 32)
TILE = 8  # local attention tile in tokens (128 px)
WINDOW, STRIDE = 10, 5  # temporal decode window / stride in latents
NOISE_SEED = 42


def patchify(video):
    b, c, t, h, w = video.shape
    x = video.reshape(b, c, t // 4, 4, h // 16, 16, w // 16, 16)
    return x.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(b, -1, TUBE_DIM)


def unpatchify(tokens, grid):
    b = tokens.shape[0]
    nt, nh, nw = grid
    x = tokens.reshape(b, nt, nh, nw, 4, 16, 16, 3)
    return x.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(b, 3, nt * 4, nh * 16, nw * 16)


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def mrope_tables(grid, device):
    """cos/sin ``[1, 1, N, 128]`` for a T,H,W-ordered grid, laid out [T 64 | H 32 | W 32]."""
    positions = torch.meshgrid(*(torch.arange(n, device=device) for n in grid), indexing="ij")
    cos, sin = [], []
    for pos, dim, theta in zip(positions, SECTIONS, (THETA_T, THETA_HW, THETA_HW)):
        inv = theta ** (-torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
        ang = pos.flatten().float()[:, None] * inv[None]
        ang = torch.cat((ang, ang), dim=-1)
        cos.append(ang.cos())
        sin.append(ang.sin())
    return torch.cat(cos, -1)[None, None], torch.cat(sin, -1)[None, None]


def apply_mrope(x, cos, sin):
    rotated = torch.cat([_rotate_half(p) for p in x.split(SECTIONS, dim=-1)], dim=-1)
    return x * cos + rotated * sin


class RMSNorm(nn.Module):
    """Qwen3RMSNorm: normalise in fp32, cast back, then scale (the weight in the input dtype)."""

    def __init__(self, dim, dtype=None, device=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim, dtype=dtype, device=device))

    def forward(self, x):
        w = comfy.model_management.cast_to_device(self.weight, x.device, x.dtype)
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + EPS)
        return w * y.to(x.dtype)


def local_groups(grid, device):
    """Sliding-tile attention as ``[(query_index [G, Lq], key_index [G, Lk])]`` over the
    T,H,W-ordered tokens of one call, or None for full attention (at most 3x3 tiles)."""
    frames, height, width = grid
    th, tw = -(-height // TILE), -(-width // TILE)
    if th <= 3 and tw <= 3:
        return None
    idx = torch.arange(frames * height * width, device=device).view(frames, height, width)

    def tile_rows(a, n, size):
        return slice(a * TILE, min((a + 1) * TILE, size))

    def neighbours(a, n):
        if n <= 3:
            return list(range(n))
        c = min(max(a, 1), n - 2)
        return [c - 1, c, c + 1]

    buckets = {}
    for a in range(th):
        for bb in range(tw):
            q = idx[:, tile_rows(a, th, height), tile_rows(bb, tw, width)].reshape(-1)
            rows = neighbours(a, th)
            cols = neighbours(bb, tw)
            r0, r1 = rows[0] * TILE, min((rows[-1] + 1) * TILE, height)
            c0, c1 = cols[0] * TILE, min((cols[-1] + 1) * TILE, width)
            k = idx[:, r0:r1, c0:c1].reshape(-1)
            buckets.setdefault((q.numel(), k.numel()), ([], []))
            buckets[(q.numel(), k.numel())][0].append(q)
            buckets[(q.numel(), k.numel())][1].append(k)
    return [(torch.stack(qs), torch.stack(ks)) for qs, ks in buckets.values()]


class Attention(nn.Module):
    def __init__(self, dtype=None, device=None, operations=None):
        super().__init__()
        self.q_proj = operations.Linear(HIDDEN, HEADS * HEAD_DIM, bias=False, dtype=dtype, device=device)
        self.k_proj = operations.Linear(HIDDEN, KV_HEADS * HEAD_DIM, bias=False, dtype=dtype, device=device)
        self.v_proj = operations.Linear(HIDDEN, KV_HEADS * HEAD_DIM, bias=False, dtype=dtype, device=device)
        self.o_proj = operations.Linear(HEADS * HEAD_DIM, HIDDEN, bias=False, dtype=dtype, device=device)
        self.q_norm_t = RMSNorm(HEAD_DIM // 2, dtype=dtype, device=device)
        self.q_norm_hw = RMSNorm(HEAD_DIM // 2, dtype=dtype, device=device)
        self.k_norm_t = RMSNorm(HEAD_DIM // 2, dtype=dtype, device=device)
        self.k_norm_hw = RMSNorm(HEAD_DIM // 2, dtype=dtype, device=device)

    @staticmethod
    def _prepare(x, cos, sin, norm_t, norm_hw, dtype):
        # fp32 through the split QK-norm and RoPE, in token chunks to bound the fp32 copies
        out = torch.empty_like(x, dtype=dtype)
        step = max(1, (64 << 20) // (x.shape[1] * x.shape[3] * 4))
        for s in range(0, x.shape[2], step):
            part = x[:, :, s:s + step].float()
            t_part, hw_part = part.chunk(2, dim=-1)
            part = torch.cat((norm_t(t_part), norm_hw(hw_part)), dim=-1)
            out[:, :, s:s + step] = apply_mrope(part, cos[:, :, s:s + step], sin[:, :, s:s + step]).to(dtype)
        return out

    def forward(self, x, cos, sin, groups, transformer_options={}):
        b, n, _ = x.shape
        q = self.q_proj(x).view(b, n, HEADS, HEAD_DIM).transpose(1, 2)
        k = self.k_proj(x).view(b, n, KV_HEADS, HEAD_DIM).transpose(1, 2)
        v = self.v_proj(x).view(b, n, KV_HEADS, HEAD_DIM).transpose(1, 2)
        q = self._prepare(q, cos, sin, self.q_norm_t, self.q_norm_hw, v.dtype)
        k = self._prepare(k, cos, sin, self.k_norm_t, self.k_norm_hw, v.dtype)
        rep = HEADS // KV_HEADS
        if groups is None:
            out = optimized_attention(q, k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1), HEADS,
                                      skip_reshape=True, transformer_options=transformer_options)
        else:
            out = x.new_empty(b, n, HEADS * HEAD_DIM)
            for qi_all, ki_all in groups:
                lq, lk = qi_all.shape[1], ki_all.shape[1]
                # tiles per call: keep the gathered (head-expanded) keys and values near 256 MB
                per = max(1, (256 << 20) // (b * HEADS * lk * HEAD_DIM * q.element_size() * 2))
                for s in range(0, qi_all.shape[0], per):
                    qi, ki = qi_all[s:s + per], ki_all[s:s + per]
                    g = qi.shape[0]
                    # [B, heads, G*L, D] -> [B*G, heads, L, D]
                    qg = q[:, :, qi.reshape(-1)].view(b, HEADS, g, lq, HEAD_DIM).transpose(1, 2).reshape(b * g, HEADS, lq, HEAD_DIM)
                    kg = k[:, :, ki.reshape(-1)].view(b, KV_HEADS, g, lk, HEAD_DIM).transpose(1, 2).reshape(b * g, KV_HEADS, lk, HEAD_DIM)
                    vg = v[:, :, ki.reshape(-1)].view(b, KV_HEADS, g, lk, HEAD_DIM).transpose(1, 2).reshape(b * g, KV_HEADS, lk, HEAD_DIM)
                    o = optimized_attention(qg, kg.repeat_interleave(rep, 1), vg.repeat_interleave(rep, 1), HEADS,
                                            skip_reshape=True, transformer_options=transformer_options)
                    out[:, qi.reshape(-1)] = o.view(b, g, lq, -1).reshape(b, g * lq, -1)
                    del qg, kg, vg, o
        return self.o_proj(out)


class Layer(nn.Module):
    def __init__(self, dtype=None, device=None, operations=None):
        super().__init__()
        self.input_norm = RMSNorm(HIDDEN, dtype=dtype, device=device)
        self.attn = Attention(dtype=dtype, device=device, operations=operations)
        self.post_attn_norm = RMSNorm(HIDDEN, dtype=dtype, device=device)
        self.mlp = nn.Module()
        self.mlp.gate_proj = operations.Linear(HIDDEN, INTER, bias=False, dtype=dtype, device=device)
        self.mlp.up_proj = operations.Linear(HIDDEN, INTER, bias=False, dtype=dtype, device=device)
        self.mlp.down_proj = operations.Linear(INTER, HIDDEN, bias=False, dtype=dtype, device=device)

    def forward(self, x, cos, sin, groups, transformer_options={}):
        x = x + self.attn(self.input_norm(x), cos, sin, groups, transformer_options)
        h = self.post_attn_norm(x)
        # SwiGLU in row chunks: the 12288-wide intermediates are the largest activations
        out = torch.empty_like(x)
        step = 8192
        for s in range(0, x.shape[1], step):
            hs = h[:, s:s + step]
            out[:, s:s + step] = self.mlp.down_proj(F.silu(self.mlp.gate_proj(hs)) * self.mlp.up_proj(hs))
        return x + out


class PixelHead(nn.Module):
    def __init__(self, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm = operations.RMSNorm(HIDDEN, eps=EPS, dtype=dtype, device=device)
        self.linear = operations.Linear(HIDDEN, TUBE_DIM, dtype=dtype, device=device)

    def forward(self, x):
        return self.linear(self.norm(x))


class PixelUMMDecoder(nn.Module):
    """``forward(latent [B,32,L,h,w], xt [B,3,4L,32h,32w]) -> x0``."""

    def __init__(self, dtype=None, device=None, operations=None):
        super().__init__()
        self.embed = operations.Linear(TUBE_DIM, HIDDEN, dtype=dtype, device=device)
        self.cond_proj = operations.Linear(LATENT_CHANNELS, GRID * GRID * HIDDEN, dtype=dtype, device=device)
        self.layers = nn.ModuleList(Layer(dtype=dtype, device=device, operations=operations) for _ in range(LAYERS))
        self.final_norm = RMSNorm(HIDDEN, dtype=dtype, device=device)
        self.head = PixelHead(dtype=dtype, device=device, operations=operations)

    def condition_tokens(self, latent):
        b, _, t, h, w = latent.shape
        x = self.cond_proj(latent.permute(0, 2, 3, 4, 1))
        x = x.view(b, t, h, w, GRID, GRID, HIDDEN)
        return x.permute(0, 1, 2, 4, 3, 5, 6).reshape(b, -1, HIDDEN)

    def forward(self, latent, xt, transformer_options={}):
        dtype = latent.dtype
        _, _, t, h, w = latent.shape
        grid = (t, h * GRID, w * GRID)
        tokens = self.embed(patchify(xt.to(dtype))) + self.condition_tokens(latent)
        cos, sin = mrope_tables(grid, latent.device)
        groups = local_groups(grid, latent.device)
        for layer in self.layers:
            tokens = layer(tokens, cos, sin, groups, transformer_options)
        return unpatchify(self.head(self.final_norm(tokens)), grid)


# -- windowed decode ------------------------------------------------------------------------

def window_starts(length, window=WINDOW, stride=STRIDE):
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, stride))
    if starts[-1] != length - window:
        starts.append(length - window)
    return starts


def cosine_weights(overlap, device, dtype):
    k = torch.arange(1, overlap + 1, device=device, dtype=torch.float32)
    return (0.5 - 0.5 * torch.cos(math.pi * k / (overlap + 1))).to(dtype).view(1, 1, -1, 1, 1)


def window_noise(b, h, w, first, last, device, seed=NOISE_SEED):
    """The reference's pixel noise: one CUDA generator per latent frame ``i``, seeded
    ``seed * 1000003 + i``, so overlapping windows see the same noise."""
    parts = []
    for i in range(first, last):
        g = torch.Generator(device=device)
        g.manual_seed(seed * 1_000_003 + i)
        parts.append(torch.randn((b, 3, 4, h * 32, w * 32), generator=g, device=device, dtype=torch.float32))
    return torch.cat(parts, dim=2)


def decode_windows(decoder, latent, starts, seed=NOISE_SEED, pad=None, offset=0):
    """Decode the windows starting at ``starts`` (latent frames) and cross-fade them in order.
    Returns ``(x0 [B,3,F,H,W], first pixel frame)``. ``pad(start, real)`` supplies latent
    frames past the end of ``latent`` for windows that would overrun it (preview decoding).
    ``offset`` is the absolute index of ``latent``'s first frame (the noise is per absolute frame)."""
    b, _, length, h, w = latent.shape
    span = min(WINDOW, length) if pad is None else WINDOW
    out, end = None, 0
    for s in starts:
        z = latent[:, :, s:s + span]
        real = z.shape[2]
        if real < span:
            z = torch.cat((z, pad(s, span - real).to(z)), dim=2)
        x0 = decoder(z, window_noise(b, h, w, offset + s, offset + s + span, latent.device, seed))[:, :, :4 * real]
        p = 4 * s
        if out is None:
            out, first, end = x0, p, p + x0.shape[2]
            continue
        overlap = end - p
        wgt = cosine_weights(overlap, x0.device, x0.dtype)
        out[:, :, p - first:end - first] = out[:, :, p - first:end - first] * (1 - wgt) + x0[:, :, :overlap] * wgt
        out = torch.cat((out, x0[:, :, overlap:]), dim=2)
        end = p + x0.shape[2]
    return out, first


def decode_clip(decoder, latent, seed=NOISE_SEED):
    """Whole-clip decode (the reference's ``decode`` with num_steps 1): ``[B,3,4L,32h,32w]``
    x0 in about [-1, 1]."""
    return decode_windows(decoder, latent, window_starts(latent.shape[2]), seed)[0]


def decode_tail(decoder, latent, first, seed=NOISE_SEED, context=STRIDE):
    """Decode the frames from latent ``first`` on, with ``context`` latents before it decoded
    again for continuity. Returns ``(x0, start)``: pixels from latent ``start`` = first - context
    (clamped) to the end; the caller cross-fades ``[4*start, 4*first)`` into what it already shows."""
    start = max(0, first - context)
    part = latent[:, :, start:]
    x0, _ = decode_windows(decoder, part, window_starts(part.shape[2]), seed, offset=start)
    return x0, start
