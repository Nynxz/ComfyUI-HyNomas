"""A Hy-Nomas world: segment-by-segment generation with a cache of the earlier segments.

Follows hymm/pipeline/causal_ar.py (seg_ar rollout) of Tencent-Hunyuan/Hy-Nomas:

* every segment starts from the noise bank slice at its frames and is denoised with Euler on
  the flow velocity over the shift-5 schedule (8 steps for the distilled DiT);
* afterwards its clean latent is re-noised to sigma 6/29 and run once more at t = 206.897 to
  produce its keys and values, which go into the history (int8, one scale per token and head,
  as the reference's low-memory preset stores them);
* later segments attend to the history: the selected frames' image tokens and each segment's
  head text (everything before its ``contained_actions``). ``window`` keeps 18 frames picked
  by the reference's temporal-context rule (always frame 0 and the newest, denser near the
  present); ``full`` keeps everything.
"""

import logging
import os
import uuid
import weakref

import torch

import comfy.model_management
import comfy.utils

from . import prompt as prompts
from .dit import History

SHIFT = 5.0
CACHE_SIGMA = 0.206897
BANK_FRAMES = (120 * 24 - 1) // 4 + 1  # the reference's interactive noise bank: 120 s
WINDOW_FRAMES = 18


def sigmas(steps, shift=SHIFT):
    s = torch.linspace(1, 0, steps + 1, dtype=torch.float32)
    return shift * s / (1 + (shift - 1) * s)


def temporal_context_frames(total, keep=WINDOW_FRAMES, gamma=2.0):
    """The reference's ``temporal_context_frame_indices``: ``keep`` of ``total`` frames, oldest first."""
    n = min(keep, total)
    if n == total:
        return list(range(total))
    out = set()
    for i in range(n):
        u = i / (n - 1)
        d = 0 if i == 0 else total - 1 if i == n - 1 else i + int(round((total - n) * u ** gamma))
        out.add(total - 1 - d)
    return sorted(out)


def quantize(x):
    """int8 per (token, head) over the head dim: absmax/127, round half away from zero."""
    x = x.float()
    scale = x.abs().amax(-1, keepdim=True) / 127.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    n = x / scale
    q = (torch.sign(n) * torch.floor(n.abs() + 0.5)).clamp_(-127, 127).to(torch.int8)
    return q, scale


def archive_dir():
    try:
        import folder_paths
        base = folder_paths.get_temp_directory()
    except ImportError:
        import tempfile
        base = tempfile.gettempdir()
    path = os.path.join(base, "nomas_history")
    os.makedirs(path, exist_ok=True)
    return path


def _remove(path):
    try:
        os.remove(path)
    except OSError:
        pass


class Segment:
    """One finished segment: its latent and its per-block history keys/values (int8).

    The keys/values (~0.2 GB per latent frame at 1280x704) are written to ComfyUI's temp folder
    and memory-mapped back, so a long world costs page cache, not RAM; the file goes when the
    segment does."""

    def __init__(self, latent, frame_start, prompt):
        self.latent = latent
        self.frame_start = frame_start
        self.frames = latent.shape[2]
        self.prompt = prompt
        self.img = []  # per block: (k, k_scale, v, v_scale), [B, heads, frames*tpf, D]
        self.txt = []  # per block: the head text, same layout

    def spill(self):
        path = os.path.join(archive_dir(), f"{uuid.uuid4().hex}.pt")
        torch.save({"img": self.img, "txt": self.txt}, path)
        data = torch.load(path, mmap=True, weights_only=True)
        self.img, self.txt = data["img"], data["txt"]
        weakref.finalize(self, _remove, path)


class CachedHistory(History):
    """The GPU working set: per block, int8 keys/values dequantised to bf16 when the block runs."""

    def __init__(self, blocks, dtype):
        super().__init__()
        self.blocks, self.dtype = blocks, dtype

    def get(self, i):
        if not self.blocks:
            return None, None
        k, ks, v, vs = self.blocks[i]
        return (k.float() * ks).to(self.dtype), (v.float() * vs).to(self.dtype)


class World:
    """The state of one generated world, passed between ComfyUI nodes and turns."""

    def __init__(self, seed, height, width, history="window"):
        self.seed = seed
        self.h, self.w = height // 32, width // 32
        self.history = history
        self.generator = torch.Generator("cpu").manual_seed(seed)
        self.bank = torch.randn((1, 32, BANK_FRAMES, self.h, self.w), generator=self.generator, dtype=torch.bfloat16)
        self.segments = []
        self.pose = None  # the camera's end pose, so a camera turn continues where the last one stopped

    @property
    def frames(self):
        return sum(s.frames for s in self.segments)

    def latent(self):
        """All generated latents ``[1, 32, L, h, w]``."""
        return torch.cat([s.latent for s in self.segments], dim=2)

    def noise(self, start, count):
        while self.bank.shape[2] < start + count:  # extend in 30 s steps, from the same generator
            more = torch.randn((1, 32, (30 * 24) // 4, self.h, self.w), generator=self.generator, dtype=torch.bfloat16)
            self.bank = torch.cat((self.bank, more), dim=2)
        return self.bank[:, :, start:start + count]

    def fork(self):
        """A copy that shares the finished segments (for re-rolling the next turn)."""
        other = World.__new__(World)
        other.__dict__.update(self.__dict__)
        other.generator = torch.Generator("cpu")
        other.generator.set_state(self.generator.get_state())
        other.segments = list(self.segments)
        return other

    # -- history -----------------------------------------------------------------------------

    def history_blocks(self, device):
        cached = [s for s in self.segments if s.img]
        if not cached:
            return []
        tpf = self.h * self.w
        total = self.frames
        keep = set(range(total)) if self.history == "full" else set(temporal_context_frames(total))
        n_blocks = len(cached[0].img)
        parts = [[[] for _ in range(4)] for _ in range(n_blocks)]
        for seg in cached:
            local = [f - seg.frame_start for f in range(seg.frame_start, seg.frame_start + seg.frames) if f in keep]
            if not local:
                continue  # the reference drops a segment's text with its last frame
            idx = torch.cat([torch.arange(f * tpf, (f + 1) * tpf) for f in local])
            for b in range(n_blocks):
                for j in range(4):
                    parts[b][j].append(seg.img[b][j][:, :, idx])
                    parts[b][j].append(seg.txt[b][j])
        # one block at a time to the GPU, so the selection is never all in RAM at once
        return [tuple(torch.cat([t.to(device) for t in p], dim=2) for p in block) for block in parts]

    # -- generation --------------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, model, seg, latent_frames, camera=None, steps=8, write_cache=True, progress=None):
        """Generate one segment.

        seg            a :class:`prompt.EncodedSegment`
        latent_frames  the segment's length in latent frames (4 video frames each)
        camera         ``[4L-3, 4, 4]`` normalised poses (``camera.condition``), or None
        write_cache    run the cache-write pass so later segments can follow this one
        Returns the segment's latent ``[1, 32, L, h, w]`` (CPU).
        """
        dit = model.model.diffusion_model
        tpf = self.h * self.w
        tokens = latent_frames * tpf + seg.states.shape[1]
        history_tokens = (WINDOW_FRAMES if self.history == "window" else self.frames) * tpf + 1280 * len(self.segments)
        # weights stay managed by ComfyUI; leave room for the activations and the history
        work = (tokens * 2048 * 2 * 40 + history_tokens * 2048 * 2 * 2 * len(dit.double_blocks)) * 1.2
        comfy.model_management.load_models_gpu([model], memory_required=work)
        device = model.load_device
        dtype = model.model.manual_cast_dtype or model.model.get_dtype()
        if dtype not in (torch.bfloat16, torch.float16, torch.float32):
            dtype = torch.bfloat16

        start = self.frames
        history = CachedHistory(self.history_blocks(device), dtype)
        attend = None
        groups = prompts.attention_groups(seg, latent_frames, tpf, device)
        if groups is not None:
            from .dit import Attend
            attend = Attend(groups)
        text = seg.states.to(device, dtype)
        cam = None
        if camera is not None:
            poses = torch.as_tensor(camera, dtype=torch.float32)[None].to(device)
            cam = dit.camera_tokens(poses)
            if cam.shape[1] != latent_frames:
                raise ValueError(f"the camera covers {cam.shape[1]} latent frames, the segment {latent_frames}")

        sig = sigmas(steps)
        x = self.noise(start, latent_frames).to(device)
        pbar = comfy.utils.ProgressBar(steps + int(write_cache))
        for i in range(steps):
            comfy.model_management.throw_exception_if_processing_interrupted()
            t = torch.full((1,), float(sig[i]) * 1000.0, device=device)
            v = dit.forward_segment(x.to(dtype), t, text, start, cam, attend=attend, history=history)
            x = (x.float() + v.float() * float(sig[i + 1] - sig[i])).to(torch.bfloat16)
            pbar.update(1)
            if progress is not None:
                progress(i + 1, steps + int(write_cache), x)

        done = Segment(x.cpu(), start, seg.prompt)
        if write_cache:
            eps = torch.randn(x.shape, generator=self.generator, dtype=torch.bfloat16)
            xc = ((1 - CACHE_SIGMA) * x.float() + CACHE_SIGMA * eps.float().to(device)).to(torch.bfloat16)
            head = seg.head

            def keep(block, ik, iv, tk, tv):
                done.img.append(tuple(t.cpu() for t in (*quantize(ik), *quantize(iv))))
                done.txt.append(tuple(t.cpu() for t in (*quantize(tk[:, :, :head]), *quantize(tv[:, :, :head]))))

            t = torch.full((1,), CACHE_SIGMA * 1000.0, device=device)
            dit.forward_segment(xc.to(dtype), t, text, start, cam, attend=attend, history=history, on_kv=keep)
            done.spill()
            pbar.update(1)
        del history
        self.segments.append(done)
        logging.info("Nomas: segment %d done (%d latent frames, %d in the world)", len(self.segments), latent_frames, self.frames)
        return done.latent
