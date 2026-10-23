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


class Session:
    def __init__(self, world):
        self.world = world
        self.undo = []  # (world, frame count) before each turn
        self.frames = None  # uint8 [F, H, W, 3] on the CPU
        self.decoded = 0  # latent frames already decoded


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
                IO.String.Input("change", multiline=True, default="",
                                tooltip="Optional: what changes in the next turn (cleared by the viewer after use)."),
                IO.String.Input("action", multiline=True, default="", tooltip="Optional: what <|Subject1|> does in the next turn."),
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

        clip_start = 0
        if op in ("view", "save"):
            if session is None:
                raise ValueError("no world to show yet: queue a turn first")
            full = session.frames
            clip = save_clip(full, output=op == "save")
            ui = {"nomas_clip": [clip], "nomas_state": [{
                "session": sid, "turns": len(session.world.segments), "frames": int(full.shape[0]),
                "seconds": round(full.shape[0] / 24, 2), "can_undo": bool(session.undo),
                "saved": f"{clip['subfolder']}/{clip['filename']}" if op == "save" else None, "op": op}]}
            return IO.NodeOutput(full.float() / 255.0, {"samples": session.world.latent()}, session.world, ui=ui)
        if op == "undo":
            if session is None or not session.undo:
                raise ValueError("nothing to undo")
            session.world, n = session.undo.pop()
            session.frames, session.decoded = session.frames[:4 * n], n
        elif session is None or op == "reset":
            world = World(seed, height, width, history)
            segs, _ = prompts.segment_prompts(turn["item"])
            latent_frames = (turn_frames(turn.get("seconds", seconds), ticks) - 1) // 4 + 1
            run_turn(world, model, clip, segs[0], latent_frames, ticks or None, steps)
            session = Session(world)
            SESSIONS[sid] = session
            while len(SESSIONS) > MAX_SESSIONS:
                SESSIONS.pop(next(iter(SESSIONS)))
            decode_new(vae, session)
        else:
            before = session.world.frames
            session.undo.append((session.world, before))
            session.undo = session.undo[-8:]
            world = session.world.fork()
            frames = turn_frames(seconds, ticks)
            prompt = follow_up_prompt(change, action, frames / 24)
            run_turn(world, model, clip, prompt, (frames - 1) // 4 + 1, ticks or None, steps)
            session.world = world
            clip_start = 4 * before
            decode_new(vae, session)

        frames = session.frames
        shown = frames[max(0, clip_start - 12):]  # the new turn, with half a second of lead-in
        ui = {"nomas_clip": [save_clip(shown)], "nomas_state": [{
            "session": sid, "turns": len(session.world.segments), "frames": int(frames.shape[0]),
            "seconds": round(frames.shape[0] / 24, 2), "can_undo": bool(session.undo), "op": op}]}
        logging.info("Nomas Explore: %s, %d turns, %.1f s", sid, len(session.world.segments), frames.shape[0] / 24)
        images = frames[clip_start:].float() / 255.0  # this turn only; the latent holds the whole world
        return IO.NodeOutput(images, {"samples": session.world.latent()}, session.world, ui=ui)
