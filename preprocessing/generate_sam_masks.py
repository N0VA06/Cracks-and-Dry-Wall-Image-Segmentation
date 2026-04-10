"""
scripts/generate_sam_masks.py

High-quality segmentation mask generation for bounding-box-only COCO datasets
(e.g. Drywall-Join-Detect) using SAM 2 + smart point prompts + GrabCut refinement.

Why the simple bbox-fill was wrong:
  The drywall dataset bboxes annotate the PANEL or JOINT REGION, not just the
  tape line itself. Filling the entire bbox marks background wall as foreground,
  giving the segmentation model wrong targets.

This script does three things:
  1. Runs SAM 2 large with the bbox + intelligent point prompts
     - Positive points: sampled along the expected joint axis (center line)
     - Negative points: sampled at bbox corners (prevents selecting whole panel)
     - Selects the TIGHTEST valid mask from SAM's 3 candidates
  2. Refines the mask with GrabCut (OpenCV) for crisp boundaries
  3. Writes an updated COCO JSON with proper polygon segmentations

Setup:
  # Install SAM 2
  pip install git+https://github.com/facebookresearch/segment-anything-2.git

  # Download SAM 2.1 large checkpoint (~900 MB, best quality)
  python scripts/generate_sam_masks.py --download-only

Usage:
  python scripts/generate_sam_masks.py [--dataset drywall] [--device cuda]
  python main.py --mode split
  python main.py --mode train --model pretrained --dataset all
"""

import os
import sys
import json
import copy
import argparse
import urllib.request
import numpy as np
from pathlib import Path

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

# ══════════════════════════════════════════════════════
#  SAM 2 checkpoints
# ══════════════════════════════════════════════════════

SAM2_MODELS = {
    "large": {
        "url" : "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        "file": "sam2.1_hiera_large.pt",
        "cfg" : "configs/sam2.1/sam2.1_hiera_l.yaml",
        "size": "~900 MB",
    },
    "base_plus": {
        "url" : "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
        "file": "sam2.1_hiera_base_plus.pt",
        "cfg" : "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "size": "~300 MB",
    },
}


def download_sam2(model_key: str = "large") -> str:
    from config import CHECKPOINT_DIR
    info = SAM2_MODELS[model_key]
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    dest = os.path.join(CHECKPOINT_DIR, info["file"])
    if os.path.isfile(dest):
        print(f"Already exists: {dest}")
        return dest
    print(f"Downloading SAM 2.1 {model_key} ({info['size']}) ...")

    def hook(b, bs, total):
        pct = min(100, b * bs * 100 // total) if total > 0 else 0
        bar = "█" * (pct // 2) + "░" * (50 - pct // 2)
        sys.stdout.write(f"\r  [{bar}] {pct}%"); sys.stdout.flush()

    urllib.request.urlretrieve(info["url"], dest, hook)
    print(f"\nSaved → {dest}")
    return dest


# ══════════════════════════════════════════════════════
#  SAM 2 predictor loader
# ══════════════════════════════════════════════════════

def load_sam2_predictor(checkpoint: str, device: str):
    """Load SAM 2 image predictor. Falls back to SAM 1 if SAM 2 not installed."""
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        # build_sam2 takes a config NAME relative to the sam2 package,
        # NOT an absolute path. Hydra resolves it via pkg://sam2.
        # Map checkpoint filename → correct Hydra config name.
        cfg_name = None
        for info in SAM2_MODELS.values():
            if info["file"] in checkpoint:
                cfg_name = info["cfg"]   # e.g. "configs/sam2.1/sam2.1_hiera_l.yaml"
                break

        if cfg_name is None:
            # Fallback: guess from filename
            fname = os.path.basename(checkpoint)
            if "large" in fname or "hiera_l" in fname:
                cfg_name = "configs/sam2.1/sam2.1_hiera_l.yaml"
            elif "base_plus" in fname or "hiera_b+" in fname:
                cfg_name = "configs/sam2.1/sam2.1_hiera_b+.yaml"
            elif "small" in fname or "hiera_s" in fname:
                cfg_name = "configs/sam2.1/sam2.1_hiera_s.yaml"
            elif "tiny" in fname or "hiera_t" in fname:
                cfg_name = "configs/sam2.1/sam2.1_hiera_t.yaml"
            else:
                cfg_name = "configs/sam2.1/sam2.1_hiera_l.yaml"

        print(f"[SAM 2] config: {cfg_name}")
        print(f"[SAM 2] checkpoint: {checkpoint}")

        model     = build_sam2(cfg_name, checkpoint, device=device)
        predictor = SAM2ImagePredictor(model)
        print(f"[SAM 2] loaded successfully on {device}")
        return predictor, "sam2"

    except ImportError:
        print("[warn] SAM 2 not installed, falling back to SAM 1")

    # SAM 1 fallback
    from segment_anything import sam_model_registry, SamPredictor
    model_type = "vit_h" if "vit_h" in checkpoint else "vit_b"
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device); sam.eval()
    print(f"[SAM 1] loaded: {os.path.basename(checkpoint)}")
    return SamPredictor(sam), "sam1"


# ══════════════════════════════════════════════════════
#  Smart point prompt generation
# ══════════════════════════════════════════════════════

def generate_point_prompts(bbox_xywh: list, n_pos: int = 5, n_neg: int = 4):
    """
    Generate positive + negative point prompts for a drywall joint bbox.

    Strategy:
      Positive: points sampled along the JOINT AXIS (center line of the bbox).
        - For elongated bboxes (tape running vertically or horizontally):
          sample along the long-axis center line at equal intervals.
        - For square-ish bboxes (tape patch): sample a cross pattern.
      Negative: bbox corner regions — tells SAM "not the whole panel wall".

    Returns:
        pos_points : [N, 2]  (x, y)
        neg_points : [M, 2]  (x, y)
    """
    x, y, w, h = [float(v) for v in bbox_xywh]
    cx, cy = x + w / 2, y + h / 2
    aspect = w / max(h, 1)

    pos_pts = []
    if aspect > 1.5:
        # Wide bbox → joint runs horizontally → sample along horizontal center
        xs = np.linspace(x + w * 0.1, x + w * 0.9, n_pos)
        pos_pts = [[float(xi), cy] for xi in xs]
    elif aspect < 0.67:
        # Tall bbox → joint runs vertically → sample along vertical center
        ys = np.linspace(y + h * 0.1, y + h * 0.9, n_pos)
        pos_pts = [[cx, float(yi)] for yi in ys]
    else:
        # Square-ish bbox → tape patch → cross pattern
        pos_pts = [
            [cx, cy],
            [cx - w * 0.2, cy],
            [cx + w * 0.2, cy],
            [cx, cy - h * 0.2],
            [cx, cy + h * 0.2],
        ][:n_pos]

    # Negative points at the four corners (avoid selecting whole panel)
    margin = 0.15
    neg_pts = [
        [x + w * margin,       y + h * margin      ],  # top-left corner region
        [x + w * (1 - margin), y + h * margin      ],  # top-right
        [x + w * margin,       y + h * (1 - margin)],  # bottom-left
        [x + w * (1 - margin), y + h * (1 - margin)],  # bottom-right
    ][:n_neg]

    return np.array(pos_pts, dtype=np.float32), np.array(neg_pts, dtype=np.float32)


# ══════════════════════════════════════════════════════
#  Mask selection: tightest valid mask
# ══════════════════════════════════════════════════════

def select_best_mask(
    masks: np.ndarray,
    scores: np.ndarray,
    bbox_xywh: list,
    min_fg_ratio: float = 0.01,   # lowered: 1% of bbox area minimum
    max_fg_ratio: float = 0.85,   # raised: allow up to 85% of bbox (joints can be large)
) -> tuple[np.ndarray | None, str]:
    """
    From SAM's 3 candidate masks, return (tightest_valid_mask, reason).
    reason is "ok" or a skip reason string for logging.

    Selection order:
      1. Masks within [min_fg_ratio, max_fg_ratio] of bbox area → pick highest score
      2. If none: pick smallest mask that is at least min_fg_ratio (relax upper bound)
      3. If still none: return (None, reason) → caller falls back to bbox fill
    """
    _, _, w, h = bbox_xywh
    bbox_area  = float(w * h)
    min_px     = min_fg_ratio * bbox_area
    max_px     = max_fg_ratio * bbox_area

    valid = []
    reasons = []
    for i, (mask, score) in enumerate(zip(masks, scores)):
        n_fg = mask.sum()
        if n_fg < min_px:
            reasons.append(f"mask{i}: too_small ({n_fg:.0f}px < {min_px:.0f})")
        elif n_fg > max_px:
            reasons.append(f"mask{i}: too_large ({n_fg:.0f}px > {max_px:.0f})")
        else:
            valid.append((float(score), i, mask, n_fg))

    if valid:
        valid.sort(key=lambda t: -t[0])   # highest SAM score
        return valid[0][2].astype(np.uint8), "ok"

    # Relax: just take smallest mask above min threshold
    above_min = [(mask.sum(), mask) for mask in masks if mask.sum() >= min_px]
    if above_min:
        above_min.sort()
        return above_min[0][1].astype(np.uint8), "ok_relaxed"

    # All masks too small → fallback to bbox fill handled upstream
    return None, " | ".join(reasons) if reasons else "all_masks_empty"


# ══════════════════════════════════════════════════════
#  GrabCut refinement
# ══════════════════════════════════════════════════════

def grabcut_refine(
    image_bgr: np.ndarray,
    mask_init: np.ndarray,
    bbox_xywh: list,
    n_iter: int = 5,
) -> np.ndarray:
    """
    Refine a binary mask using OpenCV GrabCut.
    Uses the SAM mask as initialization so GrabCut starts from a good guess.
    Returns refined binary mask (uint8, 0/1).
    """
    try:
        import cv2
    except ImportError:
        return mask_init

    x, y, w, h = [int(v) for v in bbox_xywh]
    H, W = image_bgr.shape[:2]
    x  = max(0, x);  y  = max(0, y)
    x2 = min(W, x + w); y2 = min(H, y + h)
    if x2 <= x or y2 <= y:
        return mask_init

    # Build GrabCut initialisation mask from SAM output
    gc_mask = np.full((H, W), cv2.GC_BGD, dtype=np.uint8)
    gc_mask[y:y2, x:x2] = cv2.GC_PR_BGD   # bbox interior = probable background
    gc_mask[mask_init == 1] = cv2.GC_PR_FGD  # SAM foreground = probable foreground

    # Definite background: outside the bbox (with padding)
    pad = 10
    outside = np.ones((H, W), dtype=bool)
    outside[max(0,y-pad):min(H,y2+pad), max(0,x-pad):min(W,x2+pad)] = False
    gc_mask[outside] = cv2.GC_BGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(image_bgr, gc_mask, (x, y, w, h),
                    bgd_model, fgd_model, n_iter, cv2.GC_INIT_WITH_MASK)
        refined = ((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)).astype(np.uint8)
        # Only keep refined pixels that are inside the bbox + SAM region overlap
        result = np.zeros_like(refined)
        result[y:y2, x:x2] = refined[y:y2, x:x2]
        return result if result.sum() > 50 else mask_init
    except Exception:
        return mask_init


# ══════════════════════════════════════════════════════
#  Mask → COCO polygon
# ══════════════════════════════════════════════════════

def mask_to_polygon(mask: np.ndarray) -> list:
    """Binary mask → list of COCO polygon [x0,y0,x1,y1,…] lists.

    Uses BOTH area AND arc length as acceptance criteria so thin linear
    features (tape joints, 2-5px wide) are not rejected. A 3px × 200px
    tape line has contourArea ≈ 0 but arcLength ≈ 406 — length check catches it.
    """
    try:
        import cv2
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        polys = []
        for c in contours:
            area   = cv2.contourArea(c)
            length = cv2.arcLength(c, closed=True)
            # Accept if area > 4px² (blobs) OR perimeter > 12px (thin lines)
            if area < 4 and length < 12:
                continue
            flat = c.flatten().tolist()
            if len(flat) >= 6:
                polys.append(flat)
        return polys
    except ImportError:
        # Fallback when cv2 not available
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            return []
        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()), int(ys.max())
        # Only return if it's a meaningful region
        if (x2 - x1) < 2 and (y2 - y1) < 2:
            return []
        return [[x1, y1, x2, y1, x2, y2, x1, y2]]


# ══════════════════════════════════════════════════════
#  Per-annotation mask prediction
# ══════════════════════════════════════════════════════

def predict_mask_for_annotation(
    predictor,
    predictor_type: str,
    image_rgb: np.ndarray,
    image_bgr: np.ndarray,
    bbox_xywh: list,
    use_grabcut: bool = True,
) -> tuple[np.ndarray | None, str]:
    """
    Run SAM with bbox + point prompts, select tightest valid mask,
    optionally refine with GrabCut.
    Returns (mask, reason) where reason is "ok", "ok_relaxed", or a skip reason.
    """
    x, y, w, h = [float(v) for v in bbox_xywh]
    box_xyxy = np.array([x, y, x + w, y + h])

    pos_pts, neg_pts = generate_point_prompts(bbox_xywh)
    all_pts    = np.concatenate([pos_pts, neg_pts], axis=0)
    all_labels = np.array([1]*len(pos_pts) + [0]*len(neg_pts), dtype=np.int32)

    try:
        predictor.set_image(image_rgb)
        masks, scores, _ = predictor.predict(
            point_coords=all_pts,
            point_labels=all_labels,
            box=box_xyxy[None, :],
            multimask_output=True,
        )
    except Exception as e:
        return None, f"sam_error: {e}"

    mask, reason = select_best_mask(masks, scores, bbox_xywh)
    if mask is None or mask.sum() == 0:
        return None, reason

    if use_grabcut:
        mask = grabcut_refine(image_bgr, mask, bbox_xywh)

    return mask, reason


# ══════════════════════════════════════════════════════
#  Main pipeline
# ══════════════════════════════════════════════════════

def generate_masks(
    dataset_name: str = "drywall",
    sam2_model: str = "large",
    device: str = "cuda",
    use_grabcut: bool = True,
    log_every: int = 25,
):
    from config import DATASET_ROOTS, CHECKPOINT_DIR

    root = DATASET_ROOTS.get(dataset_name)
    if not root:
        print(f"Dataset '{dataset_name}' not in config.DATASET_ROOTS"); sys.exit(1)

    # Find checkpoint (SAM2 preferred, SAM1 fallback)
    checkpoint = None
    for info in SAM2_MODELS.values():
        p = os.path.join(CHECKPOINT_DIR, info["file"])
        if os.path.isfile(p):
            checkpoint = p
            break
    if checkpoint is None:
        # Try SAM 1 checkpoints
        for fname in ["sam_vit_h_4b8939.pth", "sam_vit_b_01ec64.pth"]:
            p = os.path.join(CHECKPOINT_DIR, fname)
            if os.path.isfile(p):
                checkpoint = p
                break
    if checkpoint is None:
        print("No SAM checkpoint found. Run:")
        print("  python scripts/generate_sam_masks.py --download-only")
        sys.exit(1)

    # Find source COCO JSON
    src_json = img_dir = None
    for folder in ["train", "valid"]:
        for name in ["_annotations.coco.json", "instances_default.json"]:
            p = os.path.join(root, folder, name)
            if os.path.isfile(p):
                src_json = p; img_dir = os.path.join(root, folder); break
        if src_json: break
    if not src_json:
        print(f"No COCO JSON found under {root}"); sys.exit(1)

    print(f"\nDataset  : {dataset_name}")
    print(f"JSON     : {src_json}")
    print(f"Images   : {img_dir}")
    print(f"SAM ckpt : {checkpoint}")
    print(f"GrabCut  : {use_grabcut}")
    print()

    predictor, pred_type = load_sam2_predictor(checkpoint, device)

    with open(src_json) as f:
        coco = json.load(f)
    images_by_id = {img["id"]: img for img in coco["images"]}

    from PIL import Image as PILImage
    try:
        import cv2
        _CV2 = True
    except ImportError:
        _CV2 = False
        print("[warn] opencv not installed — GrabCut disabled, contour polygons disabled")

    updated_annots = []
    n_filled = n_kept = n_skipped = 0
    skip_reasons: dict[str, int] = {}

    for i, ann in enumerate(coco["annotations"]):
        ann_out = copy.deepcopy(ann)
        seg     = ann.get("segmentation", [])
        bbox    = ann.get("bbox", [])
        seg_empty = isinstance(seg, list) and len(seg) == 0

        if not (seg_empty and bbox and len(bbox) == 4):
            # Already has polygon/RLE segmentation — mark as high quality
            if not seg_empty:
                ann_out["mask_quality"] = "polygon"
            updated_annots.append(ann_out)
            n_kept += 1
            if (i + 1) % log_every == 0:
                print(f"  [{i+1}/{len(coco['annotations'])}] kept={n_kept} filled={n_filled} skipped={n_skipped}")
            continue

        img_info = images_by_id.get(ann["image_id"])
        if img_info is None:
            updated_annots.append(ann_out); n_skipped += 1
            skip_reasons["no_img_info"] = skip_reasons.get("no_img_info", 0) + 1
            continue

        img_path = os.path.join(img_dir, img_info["file_name"])
        if not os.path.isfile(img_path):
            updated_annots.append(ann_out); n_skipped += 1
            skip_reasons["file_not_found"] = skip_reasons.get("file_not_found", 0) + 1
            continue

        try:
            img_pil = PILImage.open(img_path).convert("RGB")
            img_rgb = np.array(img_pil)
            img_bgr = img_rgb[:, :, ::-1].copy() if _CV2 else img_rgb
        except Exception as e:
            print(f"  [warn] load failed {img_path}: {e}")
            updated_annots.append(ann_out); n_skipped += 1
            skip_reasons["load_error"] = skip_reasons.get("load_error", 0) + 1
            continue

        result = predict_mask_for_annotation(
            predictor, pred_type, img_rgb, img_bgr, bbox,
            use_grabcut=use_grabcut and _CV2,
        )
        mask, reason = result if isinstance(result, tuple) else (result, "unknown")

        if mask is not None and mask.sum() > 0:
            polys = mask_to_polygon(mask)
            if polys:
                ann_out["segmentation"]  = polys
                ann_out["area"]          = int(mask.sum())
                ann_out["mask_quality"]  = "sam"   # precise SAM + GrabCut mask
                n_filled += 1
            else:
                ann_out["mask_quality"]  = "bbox"  # bbox fill fallback
                n_skipped += 1
                skip_reasons["no_contour"] = skip_reasons.get("no_contour", 0) + 1
        else:
            # ann_out still has bbox → coco_dataset.py uses bbox_to_mask fallback
            ann_out["mask_quality"]  = "bbox"      # bbox fill fallback
            n_skipped += 1
            bucket = reason.split(":")[0] if ":" in reason else reason
            skip_reasons[bucket] = skip_reasons.get(bucket, 0) + 1

        updated_annots.append(ann_out)

        if (i + 1) % log_every == 0:
            reasons_str = "  ".join(f"{k}:{v}" for k, v in sorted(skip_reasons.items()))
            print(
                f"  [{i+1}/{len(coco['annotations'])}] "
                f"kept={n_kept} filled={n_filled} skipped={n_skipped}"
                + (f"  ({reasons_str})" if skip_reasons else "")
            )

    # Write output
    out_path = src_json.replace(".json", "_sam2_segmented.json")
    coco_out = dict(coco)
    coco_out["annotations"] = updated_annots
    with open(out_path, "w") as f:
        json.dump(coco_out, f)

    print(f"\n{'='*55}")
    print(f"Done")
    print(f"  Filled  : {n_filled} / {n_filled+n_skipped} bbox-only annotations")
    print(f"  Kept    : {n_kept} (already had segmentation)")
    print(f"  Skipped : {n_skipped} → will use bbox-fill fallback in training")
    if skip_reasons:
        print(f"  Skip breakdown:")
        for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason:30s}: {count}")
    print(f"  Output  : {out_path}")
    print(f"\nNext steps:")
    print(f"  1. Replace the original JSON:")
    print(f"     cp \"{out_path}\" \"{src_json}\"")
    print(f"  2. python main.py --mode split")
    print(f"  3. python main.py --mode train --model pretrained --dataset all")


# ══════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SAM 2 mask generation for bbox-only COCO datasets")
    p.add_argument("--dataset",       default="drywall",
                   help="Key in config.DATASET_ROOTS")
    p.add_argument("--sam2-model",    default="large",    choices=list(SAM2_MODELS))
    p.add_argument("--device",        default="cuda")
    p.add_argument("--no-grabcut",    action="store_true",
                   help="Skip GrabCut refinement step")
    p.add_argument("--download-only", action="store_true",
                   help="Only download SAM 2 checkpoint, then exit")
    args = p.parse_args()

    if args.download_only:
        download_sam2(args.sam2_model)
        sys.exit(0)

    generate_masks(
        dataset_name=args.dataset,
        sam2_model=args.sam2_model,
        device=args.device,
        use_grabcut=not args.no_grabcut,
    )