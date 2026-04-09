"""
preprocessing/combined_dataset.py

Combines both COCO datasets (cracks + drywall) into a single DataLoader
for joint text-conditioned training.

Why this is essential:
  The model must see "segment crack" paired with crack masks AND
  "segment taping area" paired with taping masks IN THE SAME TRAINING RUN,
  otherwise the FiLM text conditioning has no signal to distinguish them.
"""

import os
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from config import (
    DATASET_ROOTS, SPLIT_OUTPUT_DIR,
    BATCH_SIZE, NUM_WORKERS,
)
from preprocessing.coco_dataset import build_dataset
from preprocessing.transforms import get_train_transforms, get_val_transforms
from models.text_conditioning import SimpleTokenizer


def _find_json(dataset_name: str, subset: str) -> tuple[str, str] | None:
    """
    Resolve (json_path, image_dir) for a given dataset + split subset.

    Split JSONs live in  datasets/splits/<name>/<name>_<subset>.json
    but images always live in the ORIGINAL dataset folder (e.g. datasets/cracks.coco/train/).
    The two must never be confused — that's what caused the FileNotFoundError.
    """
    root = DATASET_ROOTS.get(dataset_name, "")

    # ── Check split output dir first ──────────────────────────────────
    split_json = os.path.join(SPLIT_OUTPUT_DIR, dataset_name, f"{dataset_name}_{subset}.json")
    if os.path.isfile(split_json):
        # Images are in the original dataset's train/ folder regardless of subset,
        # because Roboflow exports everything under train/.
        img_dir = os.path.join(root, "train")
        if not os.path.isdir(img_dir):
            # Some exports use a different top-level folder; search for any image dir.
            for candidate in ["train", "valid", "images"]:
                d = os.path.join(root, candidate)
                if os.path.isdir(d):
                    img_dir = d
                    break
        return split_json, img_dir

    # ── Fall back to raw dataset folder (no split done yet) ───────────
    folder_map = {"train": ["train"], "val": ["valid", "val"], "test": ["test"]}
    for folder in folder_map.get(subset, [subset]):
        d = os.path.join(root, folder)
        for name in ["_annotations.coco.json", "instances_default.json", "annotations.json"]:
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return p, d

    return None


def build_combined_loaders(
    tokenizer: SimpleTokenizer,
    dataset_names: list[str] | None = None,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders that combine all registered datasets.

    Args:
        tokenizer      : pre-built SimpleTokenizer (vocab already populated)
        dataset_names  : list of dataset keys; defaults to all in DATASET_ROOTS
        batch_size     : override batch size
        num_workers    : dataloader workers

    Returns:
        train_loader, val_loader, test_loader
    """
    if dataset_names is None:
        dataset_names = list(DATASET_ROOTS.keys())

    train_tf = get_train_transforms()
    val_tf   = get_val_transforms()

    train_sets, val_sets, test_sets = [], [], []
    missing = []

    for ds_name in dataset_names:
        for subset, tf, collect, is_train in [
            ("train", train_tf, train_sets, True),
            ("val",   val_tf,   val_sets,  False),
            ("test",  val_tf,   test_sets, False),
        ]:
            result = _find_json(ds_name, subset)
            if result is None:
                missing.append(f"{ds_name}/{subset}")
                continue
            json_path, img_dir = result
            try:
                ds = build_dataset(
                    annotation_file=json_path,
                    image_dir=img_dir,
                    transform=tf,
                    tokenizer=tokenizer,
                    train=is_train,
                )
                collect.append(ds)
                print(f"  [combined] {ds_name}/{subset}: {len(ds)} samples")
            except RuntimeError as e:
                print(f"  [combined] SKIP {ds_name}/{subset}: {e}")

    if missing:
        print(
            f"\n  [combined] Not found (run --mode split first): {missing}"
        )

    def _loader(datasets: list[Dataset], shuffle: bool) -> DataLoader | None:
        if not datasets:
            return None
        combined = ConcatDataset(datasets)
        return DataLoader(
            combined,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=shuffle and len(combined) > batch_size,
        )

    train_loader = _loader(train_sets, shuffle=True)
    val_loader   = _loader(val_sets,   shuffle=False)
    test_loader  = _loader(test_sets,  shuffle=False)

    total_train = sum(len(d) for d in train_sets)
    total_val   = sum(len(d) for d in val_sets)
    print(
        f"\n  [combined] Ready — "
        f"train: {total_train}, val: {total_val} samples across {len(dataset_names)} dataset(s)"
    )

    return train_loader, val_loader, test_loader
