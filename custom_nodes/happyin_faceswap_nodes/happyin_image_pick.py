"""
Happyin Image Pick — picks the first real image from multiple inputs.

Simple combiner: takes outputs from multiple MergeStreams nodes
and picks the one that has a real image (not 1x1 black, not 64x64 empty).

Batch-safe: handles batches of any size. If multiple inputs are real,
picks the first and passes the entire batch through.
"""

import torch


class HappyinImagePick:
    """Picks the first real image from up to 8 inputs.

    Connect outputs from MergeStreams nodes.
    Returns the first input that contains a real image.

    Batch-safe: handles batches transparently.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "image_7": ("IMAGE",),
                "image_8": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "pick"
    CATEGORY = "Happyin/Utils"

    DESCRIPTION = (
        "Image Pick — picks the first real image from multiple inputs.\n\n"
        "Connect outputs from MergeStreams or any IMAGE sources.\n"
        "Returns the first real image (skips 1x1 black, 64x64 empty, etc).\n\n"
        "Batch-safe: handles batches of any size."
    )

    def pick(self, image_1=None, image_2=None, image_3=None, image_4=None,
             image_5=None, image_6=None, image_7=None, image_8=None):

        images = {
            1: image_1, 2: image_2, 3: image_3, 4: image_4,
            5: image_5, 6: image_6, 7: image_7, 8: image_8,
        }

        print(f"[HappyinImagePick] === pick ===")

        for idx in range(1, 9):
            img = images[idx]
            if img is None:
                continue
            if not torch.is_tensor(img):
                print(f"[HappyinImagePick]   image_{idx} = "
                      f"{type(img).__name__} (skip)")
                continue
            b, h, w = img.shape[0], img.shape[1], img.shape[2]
            print(f"[HappyinImagePick]   image_{idx} = "
                  f"{w}x{h} batch={b}")
            # Skip 1x1 placeholder
            if h <= 1 or w <= 1:
                continue
            print(f"[HappyinImagePick]   RESULT: image_{idx} "
                  f"({w}x{h} batch={b})")
            return (img,)

        # Nothing found — 1x1 black placeholder
        print(f"[HappyinImagePick]   RESULT: no real image found")
        return (torch.zeros(1, 1, 1, 3),)
