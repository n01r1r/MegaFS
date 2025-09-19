"""
Utilities to convert CelebAMask-HQ per-part annotations (mask-anno)
into single-channel labeled PNGs expected by the original MegaFS inference.

Usage example:

from utils.mask_converter import build_labeled_png, batch_convert

# single id
build_labeled_png(
    mask_root_anno="/content/CelebAMask-HQ/CelebAMask-HQ-mask-anno",
    img_id=2107,
    out_root="/content/CelebAMask-HQ/CelebAMaskHQ-mask"
)

# batch
batch_convert(
    mask_root_anno="/content/CelebAMask-HQ/CelebAMask-HQ-mask-anno",
    ids=[2332, 2107],
    out_root="/content/CelebAMask-HQ/CelebAMaskHQ-mask"
)
"""

import os
import cv2
import numpy as np
from typing import List

# Map part name -> class index used by original encode_segmentation_rgb()
PART_TO_ID = {
    "skin":1, "nose":2, "glass":3, "l_eye":4, "r_eye":5,
    "l_brow":6, "r_brow":7, "l_ear":8, "r_ear":9,
    "mouth":10, "u_lip":11, "l_lip":12, "hair":17
}

def _anno_path(mask_root_anno: str, img_id: int, part: str) -> str:
    gid = int(img_id)
    sub = gid // 2000  # CelebAMask-HQ groups by thousands
    return os.path.join(mask_root_anno, str(sub), f"{gid:05d}_{part}.png")

def build_labeled_png(mask_root_anno: str, img_id: int, out_root: str) -> str:
    """Build a single-channel labeled PNG for one image id.
    Returns output file path.
    """
    gid = int(img_id)
    os.makedirs(out_root, exist_ok=True)

    # CelebAMask-HQ masks are 1024x1024
    H = W = 1024
    parse = np.zeros((H, W), np.uint8)

    for part, cid in PART_TO_ID.items():
        p = _anno_path(mask_root_anno, gid, part)
        if not os.path.exists(p):
            continue
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        parse[m > 0] = cid

    out_path = os.path.join(out_root, f"{gid:05d}.png")
    cv2.imwrite(out_path, parse)
    return out_path

def batch_convert(mask_root_anno: str, ids: List[int], out_root: str) -> None:
    for img_id in ids:
        build_labeled_png(mask_root_anno, img_id, out_root)


