"""
utils/mask_utils.py
Mask I/O and conversion utilities.
"""

import numpy as np
from PIL import Image


def save_mask(mask: np.ndarray, path: str, binary: bool = True):
    """
    Save a mask array as a PNG.
    If binary=True, values >0 are written as 255.
    """
    if binary:
        out = (mask > 0).astype(np.uint8) * 255
    else:
        out = mask.astype(np.uint8)
    Image.fromarray(out, mode="L").save(path)


def load_mask(path: str, binary: bool = True) -> np.ndarray:
    """Load a PNG mask. If binary, returns 0/1 array."""
    arr = np.array(Image.open(path).convert("L"))
    if binary:
        return (arr > 127).astype(np.uint8)
    return arr


def mask_to_rle(mask: np.ndarray) -> dict:
    """Encode a binary mask to COCO RLE (simple uncompressed version)."""
    flat   = mask.flatten(order="F").tolist()
    counts = []
    val, cnt = flat[0], 0
    for v in flat:
        if v == val:
            cnt += 1
        else:
            counts.append(cnt)
            val, cnt = v, 1
    counts.append(cnt)
    return {"counts": counts, "size": list(mask.shape)}


def rle_to_mask(rle: dict) -> np.ndarray:
    """Decode a simple COCO RLE to a binary mask."""
    h, w  = rle["size"]
    counts = rle["counts"]
    flat   = []
    for i, c in enumerate(counts):
        flat.extend([i % 2] * c)
    return np.array(flat, dtype=np.uint8).reshape((h, w), order="F")


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute IoU for two binary numpy masks."""
    pred = (pred > 0).astype(bool)
    gt   = (gt   > 0).astype(bool)
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return float(inter) / float(union) if union > 0 else 1.0


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Dice score for two binary numpy masks."""
    pred = (pred > 0).astype(bool)
    gt   = (gt   > 0).astype(bool)
    return 2 * (pred & gt).sum() / max(pred.sum() + gt.sum(), 1)
