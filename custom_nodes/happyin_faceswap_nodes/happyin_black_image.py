"""
Happyin Black Image — fallback node for blocked branches.

trigger is the boolean signal:
  - trigger = 1x1 black → this branch INACTIVE → output 1x1 black
  - trigger = real image → this branch ACTIVE → request lazy image → pass through

Batch-safe: handles batches of any size transparently.

Scheme:
  NSFWDeg.safe_bad ──→ ImageGate(EB) → [cycle] → BlackImage.image
  NSFWDeg.safe_bad ──→ BlackImage.trigger
                                         ↓
                                    MergeStreams.image_1
"""

import torch


class HappyinBlackImage:
    """Fallback: trigger = black means inactive, trigger = real means active.

    trigger (required IMAGE) — from NSFWDeg output.
        1x1 black = branch inactive → output 1x1 black.
        Real image(s) = branch active → request image from cycle.
    image (optional lazy IMAGE) — from cycle output.

    Batch-safe: passes through batches transparently.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": ("IMAGE",),
            },
            "optional": {
                "image": ("IMAGE", {"lazy": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "Happyin/Utils"

    DESCRIPTION = (
        "Black Image — fallback for blocked branches\n\n"
        "trigger: from NSFWDeg output.\n"
        "  Real image = branch active → request cycle image.\n"
        "  1x1 black = branch inactive → output 1x1 black.\n"
        "image: from cycle output (lazy, only evaluated when active).\n\n"
        "Batch-safe: handles batches of any size."
    )

    @staticmethod
    def _is_real_image(t):
        if t is None or not torch.is_tensor(t):
            return False
        h, w = t.shape[1], t.shape[2]
        return h > 1 and w > 1

    def check_lazy_status(self, trigger, **kwargs):
        image = kwargs.get("image", None)

        b = trigger.shape[0] if torch.is_tensor(trigger) else 0
        if self._is_real_image(trigger):
            if image is None:
                print(f"[HappyinBlackImage] trigger REAL (batch={b}) "
                      f"→ requesting image from cycle")
                return ["image"]
            print(f"[HappyinBlackImage] trigger REAL (batch={b}), "
                  f"image available")
        else:
            print(f"[HappyinBlackImage] trigger BLACK → output 1x1 black")

        return []

    def run(self, trigger, image=None):
        b = trigger.shape[0] if torch.is_tensor(trigger) else 0

        if not self._is_real_image(trigger):
            print(f"[HappyinBlackImage] trigger BLACK → output 1x1 black")
            return (torch.zeros(1, 1, 1, 3),)

        if image is not None and self._is_real_image(image):
            bi, h, w = image.shape[0], image.shape[1], image.shape[2]
            print(f"[HappyinBlackImage] image {w}x{h} batch={bi} → PASS through")
            return (image,)

        print(f"[HappyinBlackImage] trigger REAL (batch={b}) "
              f"but no image → output 1x1 black")
        return (torch.zeros(1, 1, 1, 3),)
