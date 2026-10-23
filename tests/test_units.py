"""CPU checks of the ported pieces against the reference's own code (no model weights needed).

    cd tests
    PYTHONPATH=<ComfyUI>:.. NOMAS_SRC=<Hy-Nomas checkout> python -m pytest -q test_units.py

The reference functions are loaded straight from the Hy-Nomas checkout: whole modules where
they only need numpy/scipy, single functions cut out of hymm/inference.py otherwise.
"""

import ast
import importlib.util
import json
import math
import os
import re

import numpy as np
import pytest
import torch

from nomas import camera, pixelumm, prompt, world

SRC = os.environ.get("NOMAS_SRC", "")
needs_src = pytest.mark.skipif(not os.path.isdir(SRC), reason="set NOMAS_SRC to a Hy-Nomas checkout")


def load_module(rel):
    spec = importlib.util.spec_from_file_location(rel.replace("/", "_"), os.path.join(SRC, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_functions(rel, names):
    """Cut top-level functions/assignments out of a module that can't be imported whole."""
    tree = ast.parse(open(os.path.join(SRC, rel)).read())
    body = [n for n in tree.body if (isinstance(n, (ast.FunctionDef,)) and n.name in names)
            or (isinstance(n, (ast.Assign, ast.AnnAssign)) and any(
                getattr(t, "id", None) in names for t in (n.targets if isinstance(n, ast.Assign) else [n.target])))]
    import typing
    ns = {"re": re, "json": json, **{n: getattr(typing, n) for n in ("Any", "Dict", "List", "Optional", "Tuple")}}
    exec(compile(ast.Module(body=body, type_ignores=[]), rel, "exec"), ns)
    return ns


# -- camera ----------------------------------------------------------------------------------

def random_path(seed, n=50):
    rng = np.random.default_rng(seed)
    ticks = []
    for _ in range(n):
        keys = tuple(k for k in camera.KEYS if rng.random() < 0.3)
        ticks.append((keys, float(rng.normal() * 3), float(rng.normal() * 2)))
    return camera.integrate(ticks)[0]


@needs_src
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_camera_condition_matches_reference(seed):
    demo = load_module("hymm/models/camera/geometry/demo_precomputed.py")
    rot = load_module("hymm/models/camera/geometry/rotation_scale.py")
    c2ws = random_path(seed)
    latent = 9
    count = (latent - 1) * 4 + 1
    ref = demo.resample(c2ws, np.repeat(np.eye(3)[None], len(c2ws), 0), count)[0]
    ref = demo.normalize_poses(ref, [0, count], normalize_displacement_per_video=True)
    ref = rot.scale_framewise_rotations(ref, 2.5)
    ours = camera.condition(c2ws, latent)
    assert ours.shape == (count, 4, 4)
    assert np.abs(ours - ref).max() < 1e-5


@needs_src
def test_three_to_opencv_matches_reference():
    from scipy.spatial.transform import Rotation
    p = camera.Pose((1.0, 2.0, 3.0), yaw=0.7, pitch=-0.3)
    quat = Rotation.from_matrix(p.rotation_three()).as_quat()
    ref = np.eye(4)
    ref[:3, :3] = Rotation.from_quat(quat).as_matrix() @ np.diag([1.0, -1.0, -1.0])
    ref[:3, 3] = p.position
    assert np.abs(p.c2w_opencv() - ref).max() < 1e-12


def test_camera_controls():
    c2ws, end = camera.integrate(camera.parse_script("w 1\nleft 1"))
    assert len(c2ws) == 49
    # W moves along -Z (Three.js forward) at 2.8 u/s; ArrowLeft turns left (yaw +) at 20 deg/s
    assert np.allclose(c2ws[24, :3, 3], [0, 1.5, 10 - 2.8])
    assert math.isclose(end.yaw, math.radians(20), rel_tol=1e-9)
    # OpenCV camera +Z (forward) is world -Z at the start
    assert np.allclose(c2ws[0, :3, 2], [0, 0, -1])


# -- prompts ---------------------------------------------------------------------------------

REF_FUNCS = ["_keep_keys_in_dict", "_remove_keys_from_dict", "_parse_timestamp", "_format_relative_timestamp",
             "_normalize_contained_action_timestamps", "_strip_camera_text_blocks_for_camera_segment", "_resolve_subject_ids",
             "_build_whole_item_prompts", "_BENCH_META_TOP_LEVEL_KEYS", "_BENCH_META_SEGMENT_KEYS", "_BENCH_STRIP_KEYS",
             "PROMPT_SEPARATOR", "_CAMERA_TEXT_BLOCK_RE"]


@needs_src
def test_segment_prompts_match_reference():
    ns = load_functions("hymm/inference.py", REF_FUNCS)
    ns.setdefault("PROMPT_SEPARATOR", "|||")
    ns.setdefault("_CAMERA_TEXT_BLOCK_RE", re.compile(r"<Camera\d+>.*?</Camera\d+>", re.S))
    item = {
        "perspective_view": "third-person follow", "Style": "real-world footage", "test_id": "x",
        "subject_list": [{"subject_id": "<|Subject1|>", "display_name": "a fox", "subject_attribute": "orange", "extra": 1}],
        "segments": [
            {"start_time": "00:00.000", "end_time": "00:04.000", "visual_description_long": "A forest.",
             "description_change": None,
             "contained_actions": [{"start_time": "00:01.000", "end_time": "00:03.500", "action_description": "<Action1>walks</Action1>"}]},
            {"start_time": "00:04.000", "end_time": "00:09.000", "visual_description_long": "dropped",
             "contained_actions": [{"start_time": "00:05.000", "end_time": "00:09.000", "action_description": "<Action1>runs</Action1>"},
                                   {"start_time": "00:04.000", "end_time": "00:05.000", "action_description": "<Action2>stops</Action2>"}],
             "description_change": "Compared to the previous segment, it rains."},
        ]}
    ref, _ = ns["_build_whole_item_prompts"](item, item["segments"], prompt.KEEP_KEYWORDS, False, True, True, True)
    ours, bounds = prompt.segment_prompts(item)
    assert ours == ref.split("|||")
    assert bounds == [0, 96, 216]


def test_action_windows():
    acts = [{"start_time": "00:01.000", "end_time": "00:03.000"}, {"start_time": "00:03.000", "end_time": "00:05.000"}]
    assert prompt.action_windows(acts, 31) == [[0, 6, -1], [6, 18, 0], [18, 30, 1], [30, 31, -1]]
    assert prompt.action_windows(acts[:1] and [{"start_time": 0, "end_time": 6}], 10) is None


def test_latent_boundaries():
    assert prompt.latent_boundaries([0, 120, 240], 60) == [0, 30, 60]


# -- history ---------------------------------------------------------------------------------

@needs_src
@pytest.mark.parametrize("total", [5, 18, 19, 30, 60, 211])
def test_temporal_context_matches_reference(total):
    ns = load_functions("hymm/models/dit/modules/uniform_history_cache.py", ["temporal_context_frame_indices"])
    ns.update({"Real": (int, float), "math": math, "_validate_frame_budget": lambda *a: None})
    assert world.temporal_context_frames(total) == list(ns["temporal_context_frame_indices"](total, 18, 2.0))


def test_int8_kv_rounding():
    x = torch.tensor([[[[1.0, -0.5, 0.25, 127 / 254]]]])
    q, s = world.quantize(x)
    assert q.tolist() == [[[[127, -64, 32, 64]]]]  # half away from zero
    assert torch.allclose(s, torch.tensor(1 / 127))


# -- PixelUMM local attention -----------------------------------------------------------------

@pytest.mark.parametrize("grid", [(2, 44, 80), (1, 20, 36), (3, 9, 40), (2, 24, 24)])
def test_local_groups_match_reference_mask(grid):
    frames, height, width = grid
    tile = pixelumm.TILE
    th, tw = -(-height // tile), -(-width // tile)
    t, h, w = torch.meshgrid(*(torch.arange(n) for n in grid), indexing="ij")
    tile_h, tile_w = (h // tile).flatten(), (w // tile).flatten()
    groups = pixelumm.local_groups(grid, "cpu")
    n = frames * height * width
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 2, n, 16) for _ in range(3))
    if groups is None:
        assert th <= 3 and tw <= 3
        return
    ch, cw = tile_h.clamp(1, max(th - 2, 1)), tile_w.clamp(1, max(tw - 2, 1))
    mask = ((ch[:, None] - tile_h[None]).abs() <= 1) & ((cw[:, None] - tile_w[None]).abs() <= 1)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    out = torch.zeros_like(ref)
    seen = torch.zeros(n, dtype=torch.int64)
    for qi, ki in groups:
        for a, b in zip(qi, ki):
            out[:, :, a] = torch.nn.functional.scaled_dot_product_attention(q[:, :, a], k[:, :, b], v[:, :, b])
            seen[a] += 1
            assert mask[a][:, b].all() and mask[a].sum(1).eq(len(b)).all()  # exactly the allowed keys
    assert seen.eq(1).all()
    assert (out - ref).abs().max() < 1e-5


def test_window_starts():
    assert pixelumm.window_starts(31) == [0, 5, 10, 15, 20, 21]
    assert pixelumm.window_starts(7) == [0]
    assert pixelumm.window_starts(20) == [0, 5, 10]
