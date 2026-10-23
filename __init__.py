"""ComfyUI-Nomas: native Hy-Nomas interactive world generation (Tencent-Hunyuan/Hy-Nomas).

Importing the pack registers the DiT with Load Diffusion Model and the PixelUMM decoder with
Load VAE; the text encoder is ComfyUI's own Qwen2.5-VL-7B (Load CLIP, type qwen_image). The
nodes build prompts and camera paths and generate a world turn by turn.
"""

from typing_extensions import override

from comfy_api.latest import ComfyExtension

from .nomas.integration import register

register()

from .nomas.nodes import node_list  # noqa: E402

WEB_DIRECTORY = "./web"


class NomasExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return node_list()


async def comfy_entrypoint():
    return NomasExtension()
