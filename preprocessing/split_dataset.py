"""
preprocessing/split_dataset.py
Splits a COCO JSON dataset 80/10/10 (train/val/test) and saves three new
JSON files, preserving all annotation fields.

Usage:
  python main.py --mode split
"""

import json
import random
import os
from pathlib import Path

from config import DATASET_ROOTS, SPLIT_OUTPUT_DIR, SPLIT_RATIOS, SPLIT_SEED


def split_coco_json(
    src_json: str,
    output_dir: str,
    split_name: str,
    ratios: tuple = SPLIT_RATIOS,
    seed: int = SPLIT_SEED,
) -> dict[str, str]:
    """
    Split a COCO JSON file into train/val/test subsets.

    Args:
        src_json   : path to source COCO JSON
        output_dir : directory to write split JSONs
        split_name : prefix used in output filenames
        ratios     : (train_ratio, val_ratio, test_ratio)
        seed       : random seed for reproducibility

    Returns:
        dict mapping split name → output path
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(src_json, "r") as f:
        coco = json.load(f)

    images      = coco["images"]
    annotations = coco.get("annotations", [])
    categories  = coco.get("categories", [])
    info        = coco.get("info", {})
    licenses    = coco.get("licenses", [])

    # Shuffle images with fixed seed
    rng = random.Random(seed)
    shuffled = images.copy()
    rng.shuffle(shuffled)

    n      = len(shuffled)
    n_train = int(n * ratios[0])
    n_val   = int(n * ratios[1])

    splits = {
        "train": shuffled[:n_train],
        "val"  : shuffled[n_train : n_train + n_val],
        "test" : shuffled[n_train + n_val :],
    }

    # Index annotations by image_id
    anno_by_img: dict[int, list] = {}
    for ann in annotations:
        anno_by_img.setdefault(ann["image_id"], []).append(ann)

    output_paths: dict[str, str] = {}

    for subset, imgs in splits.items():
        img_ids = {img["id"] for img in imgs}
        subset_annos = [
            ann for ann in annotations if ann["image_id"] in img_ids
        ]

        subset_coco = {
            "info"       : info,
            "licenses"   : licenses,
            "categories" : categories,
            "images"     : imgs,
            "annotations": subset_annos,
        }

        out_path = os.path.join(output_dir, f"{split_name}_{subset}.json")
        with open(out_path, "w") as f:
            json.dump(subset_coco, f)

        output_paths[subset] = out_path
        print(f"  [{subset:5s}] {len(imgs):4d} images, {len(subset_annos):5d} annotations → {out_path}")

    return output_paths


def split_all_datasets():
    """Split every dataset registered in config.DATASET_ROOTS."""
    print("=" * 60)
    print("Dataset splitting")
    print("=" * 60)

    all_splits: dict[str, dict[str, str]] = {}

    for name, root in DATASET_ROOTS.items():
        root = Path(root)
        # Look for _annotations.coco.json or instances_default.json
        candidate_dirs = [root / "train", root, root / "valid"]
        json_path = None
        for d in candidate_dirs:
            for pattern in ["_annotations.coco.json", "instances_default.json", "annotations.json"]:
                p = d / pattern
                if p.exists():
                    json_path = str(p)
                    break
            if json_path:
                break

        if json_path is None:
            print(f"[SKIP] No COCO JSON found for dataset: {name}")
            continue

        print(f"\nDataset: {name}")
        print(f"  Source: {json_path}")

        out_dir = os.path.join(SPLIT_OUTPUT_DIR, name)
        paths   = split_coco_json(json_path, out_dir, split_name=name)
        all_splits[name] = paths

    print("\nDone. Split files saved to:", SPLIT_OUTPUT_DIR)
    return all_splits


# ── CLI entry point (also called from main.py) ───────────────────────
if __name__ == "__main__":
    split_all_datasets()
