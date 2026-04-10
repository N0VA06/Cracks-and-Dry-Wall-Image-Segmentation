"""
preprocessing/coco_dataset.py
Loads COCO-format annotations → binary masks + text prompts.

Key design decisions for text-conditioned segmentation:
  • Category name → foreground via FOREGROUND_KEYWORDS (config)
  • Category name → prompt bucket via PROMPT_TEMPLATES (config)
  • Training : prompt randomly sampled from matching bucket (augmentation)
  • Val/Test : canonical prompt (bucket index 0) — deterministic
  • Tokenizer must have build_vocab(ALL_PROMPTS) called BEFORE dataset creation
"""

import json
import random
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset

from config import FOREGROUND_KEYWORDS, PROMPT_TEMPLATES, DEFAULT_PROMPT, SKIP_BBOX_MASKS
from models.text_conditioning import SimpleTokenizer


# ══════════════════════════════════════════════════════
#  Polygon / RLE → binary mask
# ══════════════════════════════════════════════════════

def polygons_to_mask(polygons: list, height: int, width: int) -> np.ndarray:
    from PIL import ImageDraw
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for poly in polygons:
        if isinstance(poly, list) and len(poly) >= 6:
            xy = list(zip(poly[0::2], poly[1::2]))
            if len(xy) >= 3:
                draw.polygon(xy, fill=1)
    return np.array(mask, dtype=np.uint8)


def rle_to_mask(rle: dict, height: int, width: int) -> np.ndarray:
    try:
        from pycocotools import mask as coco_mask
        return (coco_mask.decode(rle) > 0).astype(np.uint8)
    except ImportError:
        return np.zeros((height, width), dtype=np.uint8)


def bbox_to_mask(bbox: list, height: int, width: int) -> np.ndarray:
    """
    Convert COCO bbox [x, y, w, h] → binary rectangular mask.

    Used as fallback when the segmentation field is empty (bounding-box-only
    datasets like Drywall-Join-Detect). Better than all-zeros: the model at
    least learns where taping areas are located, even if the shape is coarse.
    For tight bounding boxes around thin tape joints this is a good proxy.
    Replace with SAM-generated masks (scripts/generate_sam_masks.py) for
    higher-quality training.
    """
    if len(bbox) != 4:
        return np.zeros((height, width), dtype=np.uint8)
    x, y, w, h = [int(round(v)) for v in bbox]
    mask = np.zeros((height, width), dtype=np.uint8)
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(width,  x + w)
    y2 = min(height, y + h)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 1
    return mask


# ══════════════════════════════════════════════════════
#  Prompt routing
# ══════════════════════════════════════════════════════

def _category_to_bucket(category_name: str) -> str | None:
    """Map a COCO category name → PROMPT_TEMPLATES bucket key."""
    name = category_name.lower()
    if any(k in name for k in ("crack",)):
        return "crack"
    if any(k in name for k in ("tape", "taping", "joint", "join", "seam", "drywall")):
        return "taping"
    return None


def _is_foreground(category_name: str) -> bool:
    name = category_name.lower()
    return any(kw in name for kw in FOREGROUND_KEYWORDS)


def sample_prompt(category_name: str, train: bool = True) -> str:
    """
    Given a COCO category name return a prompt string.
    train=True  → random choice from template bucket (augmentation)
    train=False → canonical prompt (templates[0])
    """
    bucket = _category_to_bucket(category_name)
    if bucket and bucket in PROMPT_TEMPLATES:
        templates = PROMPT_TEMPLATES[bucket]
        return random.choice(templates) if train else templates[0]
    return DEFAULT_PROMPT


# ══════════════════════════════════════════════════════
#  Dataset
# ══════════════════════════════════════════════════════

class COCOSegmentationDataset(Dataset):
    """
    COCO binary segmentation dataset with text-conditioned prompts.

    Returns per sample:
        image       : FloatTensor [3, H, W]
        mask        : FloatTensor [H, W]   values 0.0 or 1.0
        token_ids   : LongTensor  [max_token_len]
        prompt_text : str
    """

    def __init__(
        self,
        annotation_file: str,
        image_dir: str,
        transform=None,
        tokenizer: SimpleTokenizer | None = None,
        train: bool = True,
        max_token_len: int = 16,
    ):
        super().__init__()
        # Resolve image_dir before anything else — if it's a splits directory
        # (contains only JSON, no images), auto-correct to the real image folder.
        self.image_dir     = Path(self._resolve_image_dir(annotation_file, image_dir))
        self.transform     = transform
        self.tokenizer     = tokenizer
        self.train         = train
        self.max_token_len = max_token_len

        with open(annotation_file, "r") as f:
            coco = json.load(f)

        self.images     = {img["id"]: img for img in coco["images"]}
        self.categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

        # Group annotations by image id, respecting mask quality filter
        self.anno_by_image: dict[int, list] = {}
        for ann in coco.get("annotations", []):
            quality = ann.get("mask_quality", "unknown")
            if SKIP_BBOX_MASKS and quality == "bbox":
                continue    # exclude coarse bbox-fill annotations
            self.anno_by_image.setdefault(ann["image_id"], []).append(ann)

        # Keep only images that contain at least one foreground category
        self.image_ids = [
            img_id for img_id, annots in self.anno_by_image.items()
            if any(
                _is_foreground(self.categories.get(a["category_id"], ""))
                for a in annots
            )
        ]

        if not self.image_ids:
            cats = set(self.categories.values())
            raise RuntimeError(
                f"No foreground annotations found in:\n  {annotation_file}\n"
                f"Categories present: {cats}\n"
                f"FOREGROUND_KEYWORDS checked: {FOREGROUND_KEYWORDS}\n"
                f"Ensure category names overlap with FOREGROUND_KEYWORDS in config.py"
            )

    @staticmethod
    def _resolve_image_dir(annotation_file: str, image_dir: str) -> str:
        """
        Bulletproof image directory resolver.

        The split JSON lives in  datasets/splits/cracks/cracks_train.json
        but images live in       datasets/cracks.coco/train/

        If `image_dir` contains no image files (e.g. it's the splits folder),
        this walks up the path and searches DATASET_ROOTS for the real folder.
        Fixes FileNotFoundError regardless of which code path set image_dir.
        """
        import os
        from pathlib import Path

        img_dir = Path(image_dir)

        # Fast path: image_dir already has images
        has_images = any(
            list(img_dir.glob("*.jpg"))[:1] or
            list(img_dir.glob("*.png"))[:1] or
            list(img_dir.glob("*.jpeg"))[:1]
        )
        if has_images:
            return str(img_dir)

        # image_dir has no images — try common siblings and parents
        ann_path = Path(annotation_file)
        candidates = [
            ann_path.parent,                          # same dir as JSON
            ann_path.parent.parent / "train",         # ../train
            ann_path.parent.parent / "valid",         # ../valid
            ann_path.parent.parent / "images",        # ../images
        ]

        # Also search DATASET_ROOTS for any dataset folder whose name appears
        # anywhere in the annotation_file path
        try:
            from config import DATASET_ROOTS
            ann_str = str(annotation_file)
            for ds_name, ds_root in DATASET_ROOTS.items():
                if ds_name in ann_str or os.path.basename(ds_root) in ann_str:
                    for sub in ["train", "valid", "images"]:
                        candidates.append(Path(ds_root) / sub)
        except ImportError:
            pass

        for candidate in candidates:
            if candidate.is_dir():
                imgs = (
                    list(candidate.glob("*.jpg"))[:1] or
                    list(candidate.glob("*.png"))[:1] or
                    list(candidate.glob("*.jpeg"))[:1]
                )
                if imgs:
                    return str(candidate)

        # Nothing found — return original and let PIL raise a clear error
        return str(img_dir)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.image_ids)

    # ------------------------------------------------------------------
    def _build_mask(self, img_info: dict) -> np.ndarray:
        H, W = img_info["height"], img_info["width"]
        mask = np.zeros((H, W), dtype=np.uint8)
        for ann in self.anno_by_image.get(img_info["id"], []):
            cat_name = self.categories.get(ann["category_id"], "")
            if not _is_foreground(cat_name):
                continue

            seg     = ann.get("segmentation", [])
            bbox    = ann.get("bbox", [])
            quality = ann.get("mask_quality", "unknown")

            if isinstance(seg, list) and len(seg) > 0:
                m = polygons_to_mask(seg, H, W)
            elif isinstance(seg, dict):
                m = rle_to_mask(seg, H, W)
            elif bbox and len(bbox) == 4:
                # Coarse bbox fill: only reached if SKIP_BBOX_MASKS=False
                # (quality == "bbox" annotations are filtered in __init__ when True)
                m = bbox_to_mask(bbox, H, W)
            else:
                continue

            mask = np.maximum(mask, m)
        return mask   # uint8, 0 or 1

    # ------------------------------------------------------------------
    def _select_prompt(self, img_id: int) -> str:
        for ann in self.anno_by_image.get(img_id, []):
            cat_name = self.categories.get(ann["category_id"], "")
            if _is_foreground(cat_name):
                return sample_prompt(cat_name, train=self.train)
        return DEFAULT_PROMPT

    # ------------------------------------------------------------------
    def _encode(self, prompt: str) -> torch.Tensor:
        if self.tokenizer is not None:
            ids = self.tokenizer.encode(prompt)
        else:
            ids = [0] * self.max_token_len
        ids = (ids + [0] * self.max_token_len)[: self.max_token_len]
        return torch.tensor(ids, dtype=torch.long)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):
        img_id   = self.image_ids[idx]
        img_info = self.images[img_id]

        # Image
        img_path = self.image_dir / img_info["file_name"]
        image    = np.array(Image.open(img_path).convert("RGB"))

        # Mask
        mask = self._build_mask(img_info)

        # Augment
        if self.transform is not None:
            aug   = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask  = aug["mask"]

        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask)
        mask = mask.float()

        # Prompt
        prompt_text = self._select_prompt(img_id)
        token_ids   = self._encode(prompt_text)

        return image, mask, token_ids, prompt_text


# ══════════════════════════════════════════════════════
#  Factory
# ══════════════════════════════════════════════════════

def build_dataset(
    annotation_file: str,
    image_dir: str,
    transform=None,
    tokenizer: SimpleTokenizer | None = None,
    train: bool = True,
) -> COCOSegmentationDataset:
    return COCOSegmentationDataset(
        annotation_file=annotation_file,
        image_dir=image_dir,
        transform=transform,
        tokenizer=tokenizer,
        train=train,
    )