"""Parity: ComfyUI-Nomas NomasDiT vs the reference CPTHYNomasDiffusionTransformer (2 real blocks).

Segment 0 (no history, camera), cache write, then segment 1 (history, camera, intra-segment
action windows, cross-segment head-only text)."""
import os
import sys
import types

SRC = os.environ["NOMAS_SRC"]  # a Hy-Nomas checkout
COMFY = os.environ["COMFY"]  # a ComfyUI checkout
PACK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CKPT = os.environ["NOMAS_DIT"]  # DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors
sys.path[:0] = [os.path.dirname(__file__), SRC, COMFY, PACK]
import comfy.ldm.modules.attention  # noqa: F401,E402  (before the stubs)
import ref_stubs  # noqa: F401,E402

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

DEPTH = int(os.environ.get("DEPTH", 2))
dev = torch.device("cuda")
torch.manual_seed(0)

# ---- weights (first DEPTH blocks) ----
sd = {}
with safe_open(CKPT, "pt") as f:
    for k in f.keys():
        parts = k.split(".")
        if parts[0] in ("double_blocks", "action_modules") and int(parts[1]) >= DEPTH:
            continue
        sd[k] = f.get_tensor(k)

# ---- reference ----
import hymm.models.dit.modules as M  # noqa: E402

M.MODEL_CONFIG["HY-Nomas-G/2-32324-MMDiT"] = dict(M.MODEL_CONFIG["HY-Nomas-G/2-32324-MMDiT"], mm_double_blocks_depth=DEPTH)
ns = types.SimpleNamespace
args = ns(dit=ns(attn_mode="magi_mp_causal", seg_ar_attention_backend="flex", ba_attn_chunk_size=512, ba_attn_topk=3,
                 ba_attn_causal=False, ba_attn_slice_method="3D", ba_attn_full_attn_block_num=0, ba_attn_full_attn_every_n_blocks=0,
                 use_moe=False, unpad_zero_tokens=True),
          text_encoder=ns(text_states_dim=3584),
          camera=ns(use_camera_control=True, injection_mode="extrinsics_attn", per_segment_camera_attn=False,
                    per_chunk_camera_attn=True, camera_chunk_size=1),
          text_seg_pos_emb_num=0)
ref = M.load_model("HY-Nomas-G/2-32324-MMDiT", args, {"rope_theta": 256})
missing, unexpected = ref.load_state_dict(sd, strict=False)
print("ref missing", [m for m in missing if "freqs" not in m][:5], "unexpected", unexpected[:5])
ref = ref.to(dev, torch.bfloat16).eval()

# ---- ours ----
import comfy.ops  # noqa: E402
from nomas.dit import NomasDiT, Attend  # noqa: E402
from nomas.world import CachedHistory  # noqa: E402

ours = NomasDiT(depth=DEPTH, dtype=torch.bfloat16, device=dev, operations=comfy.ops.disable_weight_init)
m, u = ours.load_state_dict(sd, strict=False)
print("ours missing", m[:5], "unexpected", u[:5])
ours.eval()

T, H, W, S = 3, 4, 6, 1280
tpf = H * W


def cosine(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return float((a @ b) / (a.norm() * b.norm())), float((a - b).abs().max() / b.abs().max())


def text(valid):
    x = torch.randn(1, S, 3584, device=dev, dtype=torch.bfloat16) * 3
    m = torch.zeros(1, S, device=dev, dtype=torch.bfloat16)
    m[:, :valid] = 1
    return x, m


def camera(n_lat):
    from nomas import camera as cam
    ticks = cam.parse_script("w+left 0.5\nd 0.3\nup 0.2")
    c2ws, _ = cam.integrate(ticks)
    return torch.tensor(cam.condition(c2ws, n_lat))[None].to(dev)


with torch.no_grad():
    # camera tokens
    poses = camera(T)
    with torch.autocast("cuda", torch.bfloat16):
        ref_cam = ref._preprocess_camera(poses, None, None, H, W, torch.bfloat16, dev)
    our_cam = ours.camera_tokens(poses)
    if isinstance(ref_cam, (tuple, list)):
        ref_cam = ref_cam[0]
    print("camera tokens", tuple(our_cam.shape), cosine(our_cam, ref_cam))

    cache = [{"k": torch.zeros(1, 4096, 16, 128, device=dev, dtype=torch.bfloat16),
              "v": torch.zeros(1, 4096, 16, 128, device=dev, dtype=torch.bfloat16)} for _ in range(DEPTH)]
    cmask = [torch.zeros(1, 4096, dtype=torch.bool, device=dev) for _ in range(DEPTH)]

    # ---- segment 0 ----
    x0 = torch.randn(1, 32, T, H, W, device=dev, dtype=torch.bfloat16)
    txt0, m0 = text(41)
    t = torch.tensor([937.5], device=dev)
    with torch.autocast("cuda", torch.bfloat16):
        r0, _ = ref.forward_with_cache(x0, t, txt0, m0, kv_cache_list=cache, cache_valid_len=0, update_cache=False,
                                       global_frame_counter=0, cache_mask_list=cmask, camera_embedding=ref_cam)
    o0 = ours.forward_segment(x0, t, txt0[:, :41], 0, our_cam)
    print("segment 0 velocity", cosine(o0, r0))

    # ---- cache write of segment 0 (head = first 30 text tokens) ----
    head0 = 30
    kept = []
    tc = torch.tensor([206.897], device=dev)
    with torch.autocast("cuda", torch.bfloat16):
        _, n_new = ref.forward_with_cache(x0, tc, txt0, m0, kv_cache_list=cache, cache_valid_len=0, update_cache=True,
                                          global_frame_counter=0, cache_mask_list=cmask, camera_embedding=ref_cam)
    for cm in cmask:  # pipeline: hide segment 0's action tail from later segments
        cm[:, T * tpf + head0:T * tpf + S] = False
    ours.forward_segment(x0, tc, txt0[:, :41], 0, our_cam,
                         on_kv=lambda i, ik, iv, tk, tv: kept.append((ik, iv, tk[:, :, :head0], tv[:, :, :head0])))
    # our history: bf16 keys (no int8) to compare the attention maths exactly
    ones = lambda x: torch.ones(*x.shape[:-1], 1, device=dev)  # noqa: E731
    blocks = []
    for ik, iv, tk, tv in kept:
        k, v = torch.cat((ik, tk), 2), torch.cat((iv, tv), 2)
        blocks.append((k, ones(k), v, ones(v)))
    hist = CachedHistory(blocks, torch.bfloat16)
    ref_k = cache[0]["k"][0, :T * tpf, 0, :8]
    print("cached image keys", cosine(kept[0][0][0, 0, :, :8], ref_k))

    # ---- segment 1: history + camera + two actions ----
    T1 = 4
    x1 = torch.randn(1, 32, T1, H, W, device=dev, dtype=torch.bfloat16)
    txt1, m1 = text(57)
    poses1 = camera(T1)
    with torch.autocast("cuda", torch.bfloat16):
        ref_cam1 = ref._preprocess_camera(poses1, None, None, H, W, torch.bfloat16, dev)
    if isinstance(ref_cam1, (tuple, list)):
        ref_cam1 = ref_cam1[0]
    offs = [20, 33, 57]
    windows = [[0, 1, -1], [1, 3, 0], [3, 4, 1]]
    t1 = torch.tensor([750.0], device=dev)
    with torch.autocast("cuda", torch.bfloat16):
        r1, _ = ref.forward_with_cache(x1, t1, txt1, m1, kv_cache_list=cache, cache_valid_len=n_new, update_cache=False,
                                       global_frame_counter=T, cache_mask_list=cmask, camera_embedding=ref_cam1,
                                       intra_seg_txt_offsets=[offs], intra_seg_vid_windows=windows)
    from nomas.prompt import EncodedSegment, attention_groups
    seg = EncodedSegment("", txt1[:, :57], 20, offs, [])
    import nomas.prompt as P
    P.action_windows = lambda acts, n: windows
    groups = attention_groups(seg, T1, tpf, dev)
    o1 = ours.forward_segment(x1, t1, txt1[:, :57], T, ours.camera_tokens(poses1), attend=Attend(groups), history=hist)
    print("segment 1 velocity (history, camera, actions)", cosine(o1, r1))
    o1p = ours.forward_segment(x1, t1, txt1[:, :57], T, ours.camera_tokens(poses1), history=hist)
    print("  same without our action groups (should be worse)", cosine(o1p, r1))
