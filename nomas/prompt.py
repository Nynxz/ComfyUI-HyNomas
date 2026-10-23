"""Hy-Nomas prompts: the per-segment JSON strings, their Qwen2.5-VL encoding, and the token and
frame ranges the DiT's action masks need.

Follows hymm/inference.py (``_build_whole_item_prompts``, ``_keep_keys_in_dict``,
``_normalize_contained_action_timestamps``), hymm/models/text_encoder/encoder.py and
hymm/pipeline/base.py (``_compute_action_start``, ``_compute_action_text_offsets``,
``_compute_action_video_windows``) of Tencent-Hunyuan/Hy-Nomas.

Each segment is serialised on its own (``json.dumps``, ``ensure_ascii=False``): the global
fields (perspective_view, Style, subject_list) only in the first one, ``contained_actions``
always last, action times rebased to the segment start. The text encoder sees the video-JSON
chat template; its states are Qwen2.5-VL layer 25's raw output (``hidden_states[-3]``) after
the ``<|im_start|>user\\n`` marker, which is exactly what ComfyUI's Qwen-Image text encoder
returns with ``clip_layer(-3)``.
"""

import copy
import json
from dataclasses import dataclass, field

import torch

FPS = 24
SYSTEM = ("You are a helpful assistant. Describe the video by detailing the following aspects:         "
          "1. The main content and theme of the video.         "
          "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects.         "
          "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects.         "
          "4. background environment, light, style and atmosphere.         "
          "5. camera angles, movements, and transitions used in the video.")
TEMPLATE = "<|im_start|>system\n" + SYSTEM + "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
TEXT_LEN = 1280

KEEP_KEYWORDS = ["Style", "perspective_view", "subject_list", "subject_list.subject_id", "subject_list.display_name",
                 "subject_list.subject_attribute", "segments", "segments.visual_description_long",
                 "segments.description_change", "segments.contained_actions",
                 "segments.contained_actions.start_time", "segments.contained_actions.end_time",
                 "segments.contained_actions.action_description"]
PERSPECTIVES = ["first-person", "second-person", "third-person follow", "third-person observer"]
STYLES = ["real-world footage", "animation/anime", "CG", "video game",
          "software/web/app/computer screen recording", "slideshow"]
CAMERA_ONLY = '{"segments": [{}]}'


def parse_timestamp(ts):
    if ts is None:
        return 0.0
    if isinstance(ts, (int, float)):
        return float(ts)
    parts = ts.strip().split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(ts)


def format_timestamp(seconds):
    seconds = max(float(seconds), 0.0)
    minutes = int(seconds // 60)
    return f"{minutes:02d}:{seconds - minutes * 60:06.3f}"


def keep_keys(d, keys):
    """The reference's dotted-path whitelist (single-token keys match at every depth)."""
    tree = {}
    for key in keys:
        parts = key.split(".")
        if len(parts) == 1:
            tree.setdefault(None, set()).add(parts[0])
        else:
            node = tree
            for part in parts[:-1]:
                node.setdefault(None, set()).add(part)
                node = node.setdefault(part, {})
            node.setdefault(None, set()).add(parts[-1])
    inherited = tree.get(None, set())

    def apply(obj, t):
        keep = (t.get(None, set()) if t else set()) | inherited
        out = {}
        for k, v in obj.items():
            if k not in keep:
                continue
            sub = t.get(k) if t else None
            if sub:
                if isinstance(v, dict):
                    v = apply(v, sub)
                elif isinstance(v, list):
                    v = [apply(i, sub) if isinstance(i, dict) else i for i in v]
            out[k] = v
        return out
    return apply(copy.deepcopy(d), tree)


def segment_prompts(item):
    """A world item (perspective_view, Style, subject_list, segments with start/end times) ->
    ``(prompts, frame_boundaries)``: one JSON string per segment and the pixel-frame boundaries
    ``[0, b1, ..., bN]`` (None if a segment has no timestamps)."""
    item = dict(item)
    if "events" in item and "segments" not in item:
        item["segments"] = item.pop("events")
    if "subjects_lifecycle" in item and "subject_list" not in item:
        item["subject_list"] = item.pop("subjects_lifecycle")
    segments = item.get("segments") or []
    if not segments:
        raise ValueError("the prompt has no segments")
    context = {k: v for k, v in keep_keys(item, KEEP_KEYWORDS).items() if k != "segments"}
    prompts = []
    for i, seg in enumerate(segments):
        seg = dict(seg)
        seg_idx = seg.pop("_segment_idx", i)
        if i > 0:
            seg.pop("visual_description_long", None)
        if seg_idx == 0:
            seg.pop("description_change", None)
        if "contained_actions" in seg:
            seg["contained_actions"] = seg.pop("contained_actions")
        obj = {"segments": [seg]} if i > 0 else {**context, "segments": [seg]}
        start = parse_timestamp(seg.get("start_time", 0.0))
        for s in obj["segments"]:
            if isinstance(s.get("contained_actions"), list):
                s["contained_actions"] = [
                    {**a, **{k: format_timestamp(parse_timestamp(a[k]) - start) for k in ("start_time", "end_time") if k in a}}
                    if isinstance(a, dict) else a for a in s["contained_actions"]]
        prompts.append(json.dumps(keep_keys(obj, KEEP_KEYWORDS), ensure_ascii=False))

    if all("start_time" in s and "end_time" in s for s in segments):
        bounds = [int(parse_timestamp(segments[0]["start_time"]) * FPS)]
        bounds += [int(parse_timestamp(s["end_time"]) * FPS) for s in segments]
    else:
        bounds = None
    return prompts, bounds


def latent_boundaries(pixel_bounds, latent_frames):
    """Pixel-frame segment boundaries -> latent boundaries (the reference's rule)."""
    out = [0]
    for b in pixel_bounds[1:]:
        out.append(max(out[-1] + 1, min(b // 4, latent_frames)))
    out[-1] = latent_frames
    return out


# -- text encoding --------------------------------------------------------------------------

def token_ids(clip, prompt):
    """Qwen2.5 token ids of the full templated prompt (as the encoder sees it)."""
    tokens = clip.tokenize(TEMPLATE.format(prompt if prompt else " "))
    pairs = next(iter(tokens.values()))[0]
    return [int(p[0]) for p in pairs if not torch.is_tensor(p[0]) and not isinstance(p[0], dict)]


def crop_start(clip):
    """Tokens up to and including ``<|im_start|>user\\n``."""
    ids = token_ids(clip, "x")
    for i in range(len(ids) - 2):
        if ids[i:i + 3] == [151644, 872, 198]:
            return i + 3
    raise ValueError("the text encoder's tokenizer is not Qwen2.5 (no <|im_start|>user marker)")


def _common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _segment0(prompt):
    try:
        obj = json.loads(prompt)
    except Exception:
        return None, None
    segs = obj.get("segments") if isinstance(obj, dict) else None
    if not (isinstance(segs, list) and segs and isinstance(segs[0], dict)):
        return None, None
    return obj, segs[0]


def _variant(obj, seg0, actions):
    seg = dict(seg0)
    if actions is None:
        seg.pop("contained_actions", None)
    else:
        seg["contained_actions"] = actions
    return json.dumps({k: ([seg] if k == "segments" else v) for k, v in obj.items()}, ensure_ascii=False)


@dataclass
class EncodedSegment:
    """One segment's text, ready for the DiT."""
    prompt: str
    states: torch.Tensor  # [1, S, 3584], valid tokens only
    head: int  # tokens other segments may see (all of them when the segment has no actions)
    offsets: list = None  # [o_0, ..., o_A] action text ranges, or None
    actions: list = field(default_factory=list)  # the parsed contained_actions


_CACHE = {}
_CACHE_SIZE = 64


def encode(clip, prompt):
    """Encode one segment's JSON string with a Qwen2.5-VL-7B CLIP (Load CLIP, type qwen_image).

    Results are cached per text encoder and prompt, so repeated turns (every camera-only turn
    has the same text) don't load the 7B encoder again."""
    key = (id(clip.cond_stage_model), prompt)
    hit = _CACHE.pop(key, None)
    if hit is None:
        hit = _encode(clip, prompt)
    _CACHE[key] = hit  # most recent last
    while len(_CACHE) > _CACHE_SIZE:
        _CACHE.pop(next(iter(_CACHE)))
    return hit


@torch.no_grad()
def _encode(clip, prompt):
    clip = clip.clone()
    clip.clip_layer(-3)  # hidden_states[-3]: decoder layer 25's raw output, no final norm
    tokens = clip.tokenize(TEMPLATE.format(prompt if prompt else " "))
    out = clip.encode_from_tokens(tokens, return_dict=True)
    states = out["cond"]
    mask = out.get("attention_mask")
    if mask is not None:
        states = states[:, mask[0].bool()]
    states = states[:, :TEXT_LEN].cpu()
    valid = states.shape[1]

    ids = token_ids(clip, prompt)
    cs = crop_start(clip)
    obj, seg0 = _segment0(prompt)
    acts = seg0.get("contained_actions") if seg0 else None
    head, offsets = valid, None
    if isinstance(acts, list) and acts:
        clamp = lambda o: min(max(o - cs, 0), TEXT_LEN - 1)  # noqa: E731
        head = clamp(_common_prefix(ids, token_ids(clip, _variant(obj, seg0, None))))
        offsets, prev = [], 0
        for k in range(len(acts)):
            o = clamp(_common_prefix(ids, token_ids(clip, _variant(obj, seg0, acts[:k] if k else None))))
            prev = min(max(o, prev), TEXT_LEN)
            offsets.append(prev)
        offsets.append(min(valid, TEXT_LEN))
    return EncodedSegment(prompt, states, head, offsets, acts if isinstance(acts, list) else [])


def action_windows(actions, seg_len):
    """Segment-relative ``[vs, ve, action]`` latent windows (gaps are action -1), or None."""
    spans = []
    for k, a in enumerate(actions):
        if not isinstance(a, dict) or "start_time" not in a or "end_time" not in a:
            return None
        vs = max(0, min(int(parse_timestamp(a["start_time"]) * FPS) // 4, seg_len))
        ve = max(0, min(int(parse_timestamp(a["end_time"]) * FPS) // 4, seg_len))
        if ve > vs:
            spans.append([vs, ve, k])
    if not spans:
        return None
    spans.sort()
    for p in range(1, len(spans)):
        if spans[p][0] < spans[p - 1][1]:
            return None  # overlapping actions: the reference falls back to plain attention
    windows, cursor = [], 0
    for vs, ve, k in spans:
        if vs > cursor:
            windows.append([cursor, vs, -1])
        windows.append([vs, ve, k])
        cursor = ve
    if cursor < seg_len:
        windows.append([cursor, seg_len, -1])
    if len(windows) == 1 and windows[0][2] == 0 and len(actions) == 1:
        return None
    return windows


def attention_groups(seg, seg_len, tokens_per_frame, device):
    """The DiT's intra-segment action mask as query groups (see ``dit.Attend``), or None."""
    if seg.offsets is None:
        return None
    windows = action_windows(seg.actions, seg_len)
    if windows is None:
        return None
    offs = seg.offsets
    n_img = seg_len * tokens_per_frame
    head = torch.arange(offs[0], device=device)
    groups = {}

    def add(k, rows):
        groups.setdefault(k, []).append(rows)
    for vs, ve, k in windows:
        add(k, torch.arange(vs * tokens_per_frame, ve * tokens_per_frame, device=device))
    add(-1, n_img + head)
    for k in range(len(offs) - 1):
        add(k, n_img + torch.arange(offs[k], offs[k + 1], device=device))
    out = []
    for k, rows in groups.items():
        text = head if k < 0 else torch.cat((head, torch.arange(offs[k], offs[k + 1], device=device)))
        out.append((torch.cat(rows), text))
    return out
