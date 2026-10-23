"""Hy-Nomas nodes. Models load through the stock loaders (see integration.py): Load Diffusion
Model -> MODEL (the DiT), Load CLIP (qwen_2.5_vl_7b, type qwen_image) -> CLIP, Load VAE ->
VAE (PixelUMM); frames come out through the stock VAE Decode."""

import json

from comfy_api.latest import IO

from . import camera as cam
from . import prompt as prompts
from .world import World

NomasWorld = IO.Custom("NOMAS_WORLD")
NomasTurn = IO.Custom("NOMAS_TURN")

SUBJECT = "<|Subject1|>"


def latent_out(world):
    return {"samples": world.latent()}


def turn_frames(seconds, ticks):
    """Pixel frames of a turn: from the camera path when there is one, else from ``seconds``."""
    frames = len(ticks) + 1 if ticks else round(seconds * 24)
    return max(frames, 5)


def run_turn(world, model, clip, prompt, latent_frames, ticks, steps):
    seg = prompts.encode(clip, prompt)
    pose = None
    if ticks:
        c2ws, world.pose = cam.integrate(ticks, world.pose)
        pose = cam.condition(c2ws, latent_frames)
    world.generate(model, seg, latent_frames, pose, steps=steps)


class NomasWorldPrompt(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="NomasWorldPrompt",
            display_name="Nomas World Prompt",
            category="video/nomas",
            description="The first turn of a Hy-Nomas world, in the format its prompt enhancer writes: perspective, style, "
                        "one subject and a scene description, with an optional action.",
            inputs=[
                IO.Combo.Input("perspective", options=prompts.PERSPECTIVES, default="third-person follow"),
                IO.Combo.Input("style", options=prompts.STYLES, default="real-world footage"),
                IO.String.Input("subject", default="a small orange fox",
                                tooltip=f"Who the camera follows. Written as {SUBJECT} in the scene and action text."),
                IO.String.Input("subject_details", multiline=True, default="Orange fur, a white chest and a fluffy tail."),
                IO.String.Input("scene", multiline=True,
                                default=f"{SUBJECT} stands on a stone path in a luminous forest at dusk. Blue fireflies drift between "
                                        "tall mossy trees, and soft mist hangs over the ground.",
                                tooltip=f"The opening shot. Use {SUBJECT} for the subject."),
                IO.String.Input("action", multiline=True, default=f"{SUBJECT} walks slowly along the path.",
                                tooltip="What happens during the turn (optional). Leave empty for a camera-driven turn."),
                IO.Float.Input("seconds", default=5.0, min=0.5, max=30.0, step=0.5,
                               tooltip="Length of the turn, unless a camera path sets it."),
            ],
            outputs=[NomasTurn.Output("turn")],
        )

    @classmethod
    def execute(cls, perspective, style, subject, subject_details, scene, action, seconds) -> IO.NodeOutput:
        seg = {"start_time": 0.0, "end_time": seconds, "visual_description_long": scene.strip()}
        seg["contained_actions"] = [{"start_time": 0.0, "end_time": seconds, "action_description": f"<Action1>{action.strip()}</Action1>"}] \
            if action.strip() else []
        item = {"perspective_view": perspective, "Style": style,
                "subject_list": [{"subject_id": SUBJECT, "display_name": subject, "subject_attribute": subject_details}],
                "segments": [seg]}
        return IO.NodeOutput({"item": item, "seconds": seconds})


class NomasPromptJSON(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="NomasPromptJSON",
            display_name="Nomas Prompt (JSON)",
            category="video/nomas",
            description="A whole Hy-Nomas prompt as JSON (perspective_view, Style, subject_list, segments with start_time / "
                        "end_time), as in the release's examples/prompts.json. Each segment becomes one turn.",
            inputs=[IO.String.Input("json", multiline=True, default=json.dumps({
                "perspective_view": "third-person follow", "Style": "real-world footage",
                "subject_list": [{"subject_id": SUBJECT, "display_name": "a small orange fox",
                                  "subject_attribute": "Orange fur, a white chest and a fluffy tail."}],
                "segments": [
                    {"start_time": "00:00.000", "end_time": "00:05.000",
                     "visual_description_long": f"{SUBJECT} stands on a stone path in a luminous forest. Blue fireflies drift between the trees.",
                     "contained_actions": [{"start_time": "00:00.000", "end_time": "00:05.000",
                                            "action_description": f"<Action1>{SUBJECT} walks slowly along the path.</Action1>"}]},
                    {"start_time": "00:05.000", "end_time": "00:10.000",
                     "description_change": "Compared to the previous segment, glowing blue flowers open beside the path.",
                     "contained_actions": [{"start_time": "00:05.000", "end_time": "00:10.000",
                                            "action_description": f"<Action1>{SUBJECT} passes the flowers and looks around.</Action1>"}]},
                ]}, indent=2))],
            outputs=[NomasTurn.Output("turn")],
        )

    @classmethod
    def execute(cls, json) -> IO.NodeOutput:
        import json as _json
        item = _json.loads(json)
        if isinstance(item, list):
            item = item[0]
        return IO.NodeOutput({"item": item})


class NomasCameraPath(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="NomasCameraPath",
            display_name="Nomas Camera Path",
            category="video/nomas",
            description="A camera move for one turn, as keys held for a time (the demo's controls at 24 fps). W/S move along "
                        "the view, A/D strafe, the arrows turn 20 deg/s. Only the direction and the relative speed of moves "
                        "reach the model; turning speed matters.",
            inputs=[IO.String.Input("script", multiline=True, default="w 2\nw+left 1.5\nw 1.5",
                                    tooltip="One '<keys> <seconds>' per line, keys joined with + (w a s d up down left right, "
                                            "or wait). 'look <dx> <dy> <seconds>' is a mouse drag in pixels.")],
            outputs=[IO.Custom("NOMAS_CAMERA").Output("camera")],
        )

    @classmethod
    def execute(cls, script) -> IO.NodeOutput:
        return IO.NodeOutput(cam.parse_script(script))


class NomasWorldGenerate(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="NomasWorldGenerate",
            display_name="Nomas World",
            category="video/nomas",
            description="Generates a Hy-Nomas world segment by segment. Each segment attends to a cache of the earlier ones. "
                        "Decode the latent with VAE Decode and the PixelUMM VAE; continue it with Nomas World Continue.",
            inputs=[
                IO.Model.Input("model", tooltip="The Nomas DiT (Load Diffusion Model)."),
                IO.Clip.Input("clip", tooltip="qwen_2.5_vl_7b from Load CLIP, type qwen_image."),
                NomasTurn.Input("turn"),
                IO.Custom("NOMAS_CAMERA").Input("camera", optional=True, tooltip="A camera path for the first turn."),
                IO.Int.Input("seed", default=42, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True),
                IO.Int.Input("width", default=1280, min=256, max=2048, step=32,
                             tooltip="Trained at 1280x704 (the release's '720p' is 704 rows)."),
                IO.Int.Input("height", default=704, min=256, max=2048, step=32),
                IO.Int.Input("steps", default=8, min=1, max=16, advanced=True,
                             tooltip="Euler steps on the shift-5 schedule. The distilled DiT was made for 4, 6 or 8."),
                IO.Combo.Input("history", options=["window", "full"], default="window", advanced=True,
                               tooltip="window: 18 frames of the past (always the first and the newest, denser near the "
                                       "present), the reference's single-GPU setting. full: everything (memory grows "
                                       "~0.2 GB of VRAM per latent frame)."),
            ],
            outputs=[IO.Latent.Output(), NomasWorld.Output("world")],
        )

    @classmethod
    def execute(cls, model, clip, turn, seed, width, height, steps, history, camera=None) -> IO.NodeOutput:
        world = World(seed, height, width, history)
        segs, bounds = prompts.segment_prompts(turn["item"])
        if camera and len(segs) == 1:
            lengths = [(turn_frames(0, camera) - 1) // 4 + 1]
        elif bounds:
            frames = bounds[-1] - bounds[0]
            lb = prompts.latent_boundaries([b - bounds[0] for b in bounds], (frames - 1) // 4 + 1)
            lengths = [b - a for a, b in zip(lb[:-1], lb[1:])]
        else:
            lengths = [(turn_frames(turn.get("seconds", 5.0), None) - 1) // 4 + 1] * len(segs)
        for i, prompt in enumerate(segs):
            run_turn(world, model, clip, prompt, lengths[i], camera if i == 0 else None, steps)
        return IO.NodeOutput(latent_out(world), world)


class NomasWorldContinue(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="NomasWorldContinue",
            display_name="Nomas World Continue",
            category="video/nomas",
            description="Adds one turn to a world: a change in words, a camera move, or both. The latent holds the whole world.",
            inputs=[
                IO.Model.Input("model"),
                IO.Clip.Input("clip"),
                NomasWorld.Input("world"),
                IO.String.Input("change", multiline=True, default="",
                                tooltip="What changes, e.g. 'the fox reaches a wooden bridge over a stream'. Empty for a "
                                        "camera-only turn. Written as 'Compared to the previous segment, ...'."),
                IO.String.Input("action", multiline=True, default="", tooltip=f"What {SUBJECT} does (optional)."),
                IO.Custom("NOMAS_CAMERA").Input("camera", optional=True),
                IO.Float.Input("seconds", default=5.0, min=0.5, max=30.0, step=0.5, tooltip="Unless the camera path sets it."),
                IO.Int.Input("steps", default=8, min=1, max=16, advanced=True),
            ],
            outputs=[IO.Latent.Output(), NomasWorld.Output("world")],
        )

    @classmethod
    def execute(cls, model, clip, world, change, action, seconds, steps, camera=None) -> IO.NodeOutput:
        world = world.fork()
        frames = turn_frames(seconds, camera)
        prompt = follow_up_prompt(change, action, frames / 24)
        run_turn(world, model, clip, prompt, (frames - 1) // 4 + 1, camera, steps)
        return IO.NodeOutput(latent_out(world), world)


def follow_up_prompt(change, action, seconds):
    change, action = change.strip(), action.strip()
    if not change and not action:
        return prompts.CAMERA_ONLY
    seg = {}
    if change:
        prefix = "Compared to the previous segment, "
        seg["description_change"] = change if change.startswith(prefix) else prefix + change[0].lower() + change[1:]
    seg["contained_actions"] = [{"start_time": prompts.format_timestamp(0), "end_time": prompts.format_timestamp(seconds),
                                 "action_description": f"<Action1>{action}</Action1>"}] if action else []
    return json.dumps({"segments": [seg]}, ensure_ascii=False)


def node_list():
    from .explore import NomasExplore
    return [NomasWorldPrompt, NomasPromptJSON, NomasCameraPath, NomasWorldGenerate, NomasWorldContinue, NomasExplore]
