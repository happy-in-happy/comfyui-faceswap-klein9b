"""Happyin face-swap node set for the FLUX.2 Klein 9B workflow.

This package carries exactly the twelve node classes the published workflow
(`workflow/face_swap_klein_9b.json`) instantiates, plus the helper modules they
import. It is a working subset of a larger private pack, extracted so the
workflow is reproducible outside its original machine -- nothing here reaches
back to that machine.

Registered class names are byte-identical to the ones stored in the workflow
JSON, so the graph loads without any node remapping.
"""

from .happyin_auto_upscale import HappyinAutoUpscale
from .happyin_black_image import HappyinBlackImage
from .happyin_bw_gate import HappyinBWGate
from .happyin_color_transfer import HappyinColorTransfer
from .happyin_contact_sheet import HappyinContactSheet
from .happyin_headwear_gate import HappyinHeadwearGate
from .happyin_image_gate import HappyinImageGate
from .happyin_image_pick import HappyinImagePick
from .happyin_mask_expand import HappyinMaskExpand
from .happyin_paste_back import HappyinPasteBack
from .happyin_person_mask import HappyinPersonMask
from .nodes import LatentSizeSnap

NODE_CLASS_MAPPINGS = {
    "Happyin_Camera_ContactSheet": HappyinContactSheet,
    "Happyin_Color_ColorTransfer": HappyinColorTransfer,
    "Happyin_Detect_PasteBack": HappyinPasteBack,
    "Happyin_Image_AutoUpscale": HappyinAutoUpscale,
    "Happyin_Mask_MaskExpand": HappyinMaskExpand,
    "Happyin_Mask_PersonMask": HappyinPersonMask,
    "Happyin_PixelAlign_LatentSizeSnap": LatentSizeSnap,
    "Happyin_Router_HeadwearGate": HappyinHeadwearGate,
    "Happyin_Router_ImageGate": HappyinImageGate,
    "Happyin_Utils_BlackImage": HappyinBlackImage,
    "Happyin_Utils_BWGate": HappyinBWGate,
    "Happyin_Utils_ImagePick": HappyinImagePick,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Happyin_Camera_ContactSheet": "Happyin Camera: Contact Sheet",
    "Happyin_Color_ColorTransfer": "Happyin Color: Color Transfer",
    "Happyin_Detect_PasteBack": "Happyin Detect: Paste Back",
    "Happyin_Image_AutoUpscale": "Happyin Image: Auto Upscale",
    "Happyin_Mask_MaskExpand": "Happyin Mask: Expand",
    "Happyin_Mask_PersonMask": "Happyin Mask: Person Mask",
    "Happyin_PixelAlign_LatentSizeSnap": "Happyin PixelAlign: Latent Size Snap",
    "Happyin_Router_HeadwearGate": "Happyin Router: Headwear Gate",
    "Happyin_Router_ImageGate": "Happyin Router: Image Gate",
    "Happyin_Utils_BlackImage": "Happyin Utils: Black Image",
    "Happyin_Utils_BWGate": "Happyin Utils: BW Gate",
    "Happyin_Utils_ImagePick": "Happyin Utils: Image Pick",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
