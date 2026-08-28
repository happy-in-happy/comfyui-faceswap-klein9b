"""
Happyin Image Gate — blocks fake 1x1 black placeholders with ExecutionBlocker.

Place at the START of each processing branch (before the cycle).
Real image → passes through → cycle executes.
1x1 black pixel → ExecutionBlocker → cycle does NOT execute (GPU saved).

Pair with BlackImage node on MergeStreams to fill blocked slots.
"""

import torch


class HappyinImageGate:
    """Blocks execution if input is a fake 1x1 black placeholder.

    Real image → passes through unchanged.
    1x1 black pixel → ExecutionBlocker → entire downstream branch skipped.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "Happyin/Router"

    DESCRIPTION = (
        "Image Gate — block fake images, pass real ones\n\n"
        "Real image passes through unchanged.\n"
        "1x1 black placeholder → ExecutionBlocker (branch skipped, no GPU cost).\n"
        "Use BlackImage node on MergeStreams for blocked slots."
    )

    def run(self, image):
        from comfy_execution.graph_utils import ExecutionBlocker

        if not torch.is_tensor(image):
            print(f"[HappyinImageGate] NOT a tensor → BLOCK")
            return (ExecutionBlocker(None),)

        h, w = image.shape[1], image.shape[2]

        if h <= 1 or w <= 1:
            print(f"[HappyinImageGate] {w}x{h} → BLOCK")
            return (ExecutionBlocker(None),)

        print(f"[HappyinImageGate] {w}x{h} → PASS")
        return (image,)
