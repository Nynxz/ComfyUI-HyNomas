"""Nomas Explore: walk through a Hy-Nomas world with WASD and the mouse, one turn per queue.

The node's viewer (web/nomas_explore.js) records the keys held and the mouse drag at 24 fps,
writes them into the ``controls`` widget and queues the graph. Each run continues the world
held in this process (keyed by the viewer's session id), decodes only the new frames (plus a
few latents of overlap, cross-faded into what was shown), and sends the new turn back to the
viewer. ``undo`` drops the last turn; ``reset`` starts a new world from the ``turn`` prompt.
"""

import json
import logging
import os
import time
import uuid

import torch

import comfy.model_management
import folder_paths
from comfy_api.latest import IO

from . import camera as cam
from . import pixelumm
from .nodes import NomasTurn, NomasWorld, follow_up_prompt, run_turn, turn_frames
from . import prompt as prompts
from .world import World

SESSIONS = {}
MAX_SESSIONS = 2
FPS = 24


class Turn:
    """One turn as the viewer shows it: its frames in the world, what kind of turn it was, the
    camera path, and its preview clip (re-encoded when the next turn re-blends its tail)."""

    def __init__(self, start, kind, label, path):
        self.start, self.kind, self.label, self.path = start, kind, label, path
        self.end = start
        self.clip = None

    def info(self):
        return {"start": self.start, "frames": self.end - self.start, "kind": self.kind, "label": self.label,
                "path": self.path, "clip": self.clip}


class Session:
    def __init__(self, world):
        self.world = world
        self.undo = []  # (world, latent frames, turns) before each turn
        self.frames = None  # uint8 [F, H, W, 3] on the CPU
        self.decoded = 0  # latent frames already decoded
        self.turns = []

    def state(self, op, saved=None):
        return {"session": None, "op": op, "turns": [t.info() for t in self.turns], "frames": int(self.frames.shape[0]),
                "fps": FPS, "can_undo": bool(self.undo), "saved": saved}


def describe(change, action, ticks, opening=None):
    """``(kind, label)`` of a turn, for the timeline."""
    words = " · ".join(x for x in (change.strip(), action.strip().replace("<|Subject1|>", "they")) if x)
    if opening is not None:
        return "opening", opening
    moves = []
    for keys, dx, dy in ticks:
        k = "+".join(keys) if keys else ("look" if dx or dy else "wait")
        if moves and moves[-1][0] == k:
            moves[-1][1] += 1
        else:
            moves.append([k, 1])
    camera = ", ".join(f"{k.upper()} {n / FPS:.1f}s" for k, n in moves if k != "wait")
    if ticks and words:
        return "both", f"{camera} — {words}"
    if ticks:
        return "camera", camera or "camera"
    return "text", words or "camera-free turn"


def camera_path(world, ticks):
    """The camera over a turn as ``[x, y, z, yaw, pitch]`` points (Three.js world axes), for the
    viewer's 3D path view; one point per 4 frames, ending at the turn's last pose."""
    pose = (world.pose or cam.Pose()).copy()

    def point(p):
        return [round(float(v), 3) for v in p.position] + [round(p.yaw, 4), round(p.pitch, 4)]
    points = [point(pose)]
    for i, (keys, dx, dy) in enumerate(ticks):
        pose.step(set(keys), dx, dy)
        if i % 4 == 3 or i == len(ticks) - 1:
            points.append(point(pose))
    return points


@torch.no_grad()
def decode_new(vae, session):
    """Decode the latents added since the last call and splice them into ``session.frames``."""
    world = session.world
    latent = world.latent()
    comfy.model_management.load_models_gpu([vae.patcher], memory_required=vae.memory_used_decode(latent.shape, vae.vae_dtype))
    decoder = vae.first_stage_model.decoder
    with torch.no_grad():
        x0, start = pixelumm.decode_tail(decoder, latent.to(vae.device, vae.vae_dtype), session.decoded)
    new = ((x0[0].float() * 0.5 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 3, 0).cpu()  # [F, H, W, 3]
    keep = 4 * start
    if session.frames is not None and keep < session.frames.shape[0]:
        old = session.frames[keep:]
        n = old.shape[0]
        w = pixelumm.cosine_weights(n, "cpu", torch.float32).view(-1, 1, 1, 1)
        new[:n] = (old.float() * (1 - w) + new[:n].float() * w).round().to(torch.uint8)
    session.frames = new if session.frames is None else torch.cat((session.frames[:keep], new))
    session.decoded = latent.shape[2]


def save_clip(frames, fps=24, output=False):
    """Write uint8 frames as an mp4: a temp preview, or (``output``) a file in output/video."""
    import av
    import numpy as np

    if output:
        folder, sub, kind = folder_paths.get_output_directory(), "video", "output"
        name = f"nomas_explore_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    else:
        folder, sub, kind = folder_paths.get_temp_directory(), "", "temp"
        name = f"nomas_explore_{uuid.uuid4().hex[:10]}.mp4"
    path = os.path.join(folder, sub, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with av.open(path, "w") as c:
        s = c.add_stream("libx264", rate=fps)
        s.width, s.height, s.pix_fmt = frames.shape[2], frames.shape[1], "yuv420p"
        s.options = {"crf": "20", "preset": "veryfast", "movflags": "+faststart"}
        for f in frames.numpy():
            c.mux(s.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format="rgb24")))
        c.mux(s.encode())
    return {"filename": name, "subfolder": sub, "type": kind}


class NomasExplore(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="NomasExplore",
            display_name="Nomas Explore",
            category="video/nomas",
            description="Explore a Hy-Nomas world interactively: click the viewer, hold W/A/S/D and the arrow keys (or drag "
                        "with the mouse) to record a camera move, and each move becomes the next turn of the world. Type a "
                        "change before recording to steer the scene. Holds the world in memory between runs.",
            is_output_node=True,
            inputs=[
                IO.Model.Input("model"),
                IO.Clip.Input("clip"),
                IO.Vae.Input("vae", tooltip="The PixelUMM VAE: each turn's new frames are decoded right away for the viewer."),
                NomasTurn.Input("turn", tooltip="The opening turn (Nomas World Prompt)."),
                IO.Int.Input("seed", default=42, min=0, max=0xFFFFFFFFFFFFFFFF),
                IO.Int.Input("width", default=1280, min=256, max=2048, step=32),
                IO.Int.Input("height", default=704, min=256, max=2048, step=32),
                IO.String.Input("change", default="",
                                tooltip="Optional: what changes in the next turn, e.g. 'rain starts streaking down the window' "
                                        "(cleared by the viewer after use)."),
                IO.String.Input("action", default="", tooltip="Optional: what <|Subject1|> does in the next turn."),
                IO.Float.Input("seconds", default=3.0, min=0.5, max=15.0, step=0.5,
                               tooltip="Length of a turn without a camera move (the recording sets it otherwise)."),
                IO.Int.Input("steps", default=8, min=1, max=16, advanced=True),
                IO.Combo.Input("history", options=["window", "full"], default="window", advanced=True),
                IO.String.Input("controls", default="", advanced=True,
                                tooltip="Written by the viewer: the session, the operation and the recorded keys."),
            ],
            outputs=[IO.Image.Output(display_name="turn_frames", tooltip="The frames of the latest turn. The whole world is in the "
                                                             "latent (VAE Decode) or the viewer's history."), IO.Latent.Output(), NomasWorld.Output("world")],
        )

    @classmethod
    def fingerprint_inputs(cls, controls, **kwargs):
        return controls

    @classmethod
    def execute(cls, model, clip, vae, turn, seed, width, height, change, action, seconds, steps, history, controls) -> IO.NodeOutput:
        ctl = json.loads(controls) if controls.strip() else {}
        sid = ctl.get("session") or "default"
        op = ctl.get("op", "turn")
        ticks = [(tuple(k), float(dx), float(dy)) for k, dx, dy in ctl.get("ticks", [])]
        session = SESSIONS.get(sid)
        size = (height // 32, width // 32)
        if session is not None and ((session.world.h, session.world.w) != size or session.world.seed != seed):
            session = None  # a new size or seed starts over

        if op in ("view", "save"):
            if session is None:
                raise ValueError("no world to show yet: queue a turn first")
            saved = None
            if op == "save":
                clip = save_clip(session.frames, output=True)
                saved = f"{clip['subfolder']}/{clip['filename']}"
            return cls.output(session, sid, op, 0, saved)

        if op == "undo":
            if session is None or not session.undo:
                raise ValueError("nothing to undo")
            session.world, n, session.turns = session.undo.pop()
            session.frames, session.decoded = session.frames[:4 * n], n
            last = session.turns[-1]
            last.end = session.frames.shape[0]
            last.clip = save_clip(session.frames[last.start:last.end])
            return cls.output(session, sid, op, last.start)

        if session is None or op == "reset":
            world = World(seed, height, width, history)
            segs, _ = prompts.segment_prompts(turn["item"])
            latent_frames = (turn_frames(turn.get("seconds", seconds), ticks) - 1) // 4 + 1
            path = camera_path(world, ticks)
            scene = turn["item"]["segments"][0].get("visual_description_long", "")
            run_turn(world, model, clip, segs[0], latent_frames, ticks or None, steps)
            session = Session(world)
            session.turns.append(Turn(0, *describe("", "", ticks, opening=scene.replace("<|Subject1|>", "the subject")[:160]), path))
            decode_new(vae, session)
            SESSIONS[sid] = session  # only once its first turn is decoded, so a failed start leaves nothing broken
            while len(SESSIONS) > MAX_SESSIONS:
                SESSIONS.pop(next(iter(SESSIONS)))
        else:
            before = session.world.frames
            session.undo.append((session.world, before, [t for t in session.turns]))
            session.undo = session.undo[-8:]
            world = session.world.fork()
            frames = turn_frames(seconds, ticks)
            prompt = follow_up_prompt(change, action, frames / FPS)
            path = camera_path(world, ticks)
            run_turn(world, model, clip, prompt, (frames - 1) // 4 + 1, ticks or None, steps)
            session.world = world
            prev = session.turns[-1]
            session.turns = session.turns[:-1] + [Turn(prev.start, prev.kind, prev.label, prev.path)]  # undo keeps the old one
            session.turns.append(Turn(4 * before, *describe(change, action, ticks), path))
            decode_new(vae, session)
        # the new turn's clip, and the previous one's (its tail was re-blended with the new frames)
        for t, nxt in zip(session.turns, session.turns[1:] + [None]):
            end = nxt.start if nxt is not None else session.frames.shape[0]
            if t.clip is None or t.end != end:
                t.end = end
                t.clip = save_clip(session.frames[t.start:t.end])
        return cls.output(session, sid, op, session.turns[-1].start)

    @classmethod
    def output(cls, session, sid, op, first, saved=None):
        state = session.state(op, saved)
        state["session"] = sid
        logging.info("Nomas Explore: %s, %d turns, %.1f s", sid, len(session.turns), session.frames.shape[0] / FPS)
        images = session.frames[first:].float() / 255.0  # this turn only (the whole world for view/save)
        return IO.NodeOutput(images, {"samples": session.world.latent()}, session.world, ui={"nomas_state": [state]})
