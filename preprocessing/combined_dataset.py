"""
preprocessing/combined_dataset.py

Combines cracks + drywall for joint text-conditioned training.

Minority-class strategy (drywall, 655 images):
  1. Strong augmentation pipeline (elastic deform, CLAHE, grid distortion…)
  2. RepeatedDataset wraps it MINORITY_REPEAT=6 times
     → 655 × 6 = 3,930 effective drywall samples per epoch
     → each repetition gets a DIFFERENT random augmentation (transforms
        are stateless; random state differs per __getitem__ call)
  3. WeightedRandomSampler then balances to ~50/50 per batch

Combined train split before sampler:
  cracks : 4,295
  drywall: 3,930 (655 × 6)
  total  : 8,225  — near-balanced without any disk writes
"""

import os
import numpy as np
import torch
from torch.utils.data import (
    ConcatDataset, DataLoader, Dataset, WeightedRandomSampler,
)

from config import (
    DATASET_ROOTS, SPLIT_OUTPUT_DIR,
    BATCH_SIZE, NUM_WORKERS, MINORITY_REPEAT,
)
from preprocessing.coco_dataset import build_dataset
from preprocessing.transforms import (
    get_train_transforms,
    get_minority_train_transforms,
    get_val_transforms,
)
from models.text_conditioning import SimpleTokenizer


# ══════════════════════════════════════════════════════
#  Repeated dataset wrapper
# ══════════════════════════════════════════════════════

class RepeatedDataset(Dataset):
    """
    Wraps a dataset to appear n_repeats times larger.
    Because the underlying dataset applies random transforms each call,
    every repetition produces a genuinely different augmented sample.
    """

    def __init__(self, dataset: Dataset, n_repeats: int):
        self.dataset   = dataset
        self.n_repeats = n_repeats

    def __len__(self) -> int:
        return len(self.dataset) * self.n_repeats

    def __getitem__(self, idx: int):
        return self.dataset[idx % len(self.dataset)]


# ══════════════════════════════════════════════════════
#  Path resolution
# ══════════════════════════════════════════════════════

def _find_json(dataset_name: str, subset: str) -> tuple[str, str] | None:
    """
    Return (json_path, image_dir) for a dataset + split.
    Split JSONs are in datasets/splits/; images stay in original dataset folder.
    """
    root = DATASET_ROOTS.get(dataset_name, "")

    split_json = os.path.join(
        SPLIT_OUTPUT_DIR, dataset_name, f"{dataset_name}_{subset}.json"
    )
    if os.path.isfile(split_json):
        img_dir = None
        for candidate in ["train", "valid", "images"]:
            d = os.path.join(root, candidate)
            if os.path.isdir(d):
                img_dir = d
                break
        return split_json, (img_dir or os.path.join(root, "train"))

    folder_map = {"train": ["train"], "val": ["valid", "val"], "test": ["test"]}
    for folder in folder_map.get(subset, [subset]):
        d = os.path.join(root, folder)
        for name in ["_annotations.coco.json", "instances_default.json", "annotations.json"]:
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return p, d

    return None


# ══════════════════════════════════════════════════════
#  Weighted sampler
# ══════════════════════════════════════════════════════

def _make_weighted_sampler(dataset_sizes: list[int]) -> WeightedRandomSampler:
    """1 / dataset_size weight per sample → equal expected draws per dataset."""
    weights = []
    for size in dataset_sizes:
        w = 1.0 / size if size > 0 else 0.0
        weights.extend([w] * size)
    wt = torch.tensor(weights, dtype=torch.float64)
    return WeightedRandomSampler(weights=wt, num_samples=int(wt.shape[0]), replacement=True)


# ══════════════════════════════════════════════════════
#  Which datasets are "minority"
# ══════════════════════════════════════════════════════

def _is_minority(dataset_name: str, all_sizes: dict[str, int]) -> bool:
    """True if this dataset is significantly smaller than the largest."""
    if not all_sizes:
        return False
    max_size = max(all_sizes.values())
    return all_sizes.get(dataset_name, max_size) < max_size * 0.5


# ══════════════════════════════════════════════════════
#  Main entry point
# ══════════════════════════════════════════════════════

def build_combined_loaders(
    tokenizer: SimpleTokenizer,
    dataset_names: list[str] | None = None,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders combining all datasets.

    Minority datasets (< 50% of majority size) receive:
      • Strong augmentation (elastic deform, CLAHE, grid distortion)
      • MINORITY_REPEAT repetitions per epoch (different aug each time)
    Then WeightedRandomSampler balances per batch.
    """
    if dataset_names is None:
        dataset_names = list(DATASET_ROOTS.keys())

    val_tf = get_val_transforms()

    # ── First pass: discover raw training sizes ────────────────────────
    raw_sizes: dict[str, int] = {}
    for ds_name in dataset_names:
        result = _find_json(ds_name, "train")
        if result is None:
            continue
        json_path, img_dir = result
        try:
            ds = build_dataset(json_path, img_dir, transform=None, tokenizer=None, train=True)
            raw_sizes[ds_name] = len(ds)
        except RuntimeError:
            pass

    # ── Second pass: build datasets with appropriate transforms ────────
    train_sets:      list[Dataset] = []
    train_set_sizes: list[int]     = []
    val_sets:        list[Dataset] = []
    test_sets:       list[Dataset] = []
    missing:         list[str]     = []

    for ds_name in dataset_names:
        is_min = _is_minority(ds_name, raw_sizes)

        for subset, collect, is_train in [
            ("train", train_sets, True),
            ("val",   val_sets,  False),
            ("test",  test_sets, False),
        ]:
            result = _find_json(ds_name, subset)
            if result is None:
                missing.append(f"{ds_name}/{subset}")
                continue
            json_path, img_dir = result

            # Choose augmentation
            if is_train and is_min:
                tf = get_minority_train_transforms()
            elif is_train:
                tf = get_train_transforms()
            else:
                tf = val_tf

            try:
                ds = build_dataset(
                    annotation_file=json_path,
                    image_dir=img_dir,
                    transform=tf,
                    tokenizer=tokenizer,
                    train=is_train,
                )

                # Apply repeats for minority train split only
                if is_train and is_min and MINORITY_REPEAT > 1:
                    n_before = len(ds)
                    ds = RepeatedDataset(ds, MINORITY_REPEAT)
                    print(
                        f"  [combined] {ds_name:10s}/train: {n_before:5d} samples "
                        f"× {MINORITY_REPEAT} repeats = {len(ds):5d} effective  "
                        f"img_dir={img_dir}"
                    )
                else:
                    print(
                        f"  [combined] {ds_name:10s}/{subset}: {len(ds):5d} samples"
                        f"  img_dir={img_dir}"
                    )

                collect.append(ds)
                if is_train:
                    train_set_sizes.append(len(ds))

            except RuntimeError as e:
                print(f"  [combined] SKIP {ds_name}/{subset}: {e}")

    if missing:
        print(f"  [combined] Not found (run --mode split): {', '.join(missing)}")

    # ── Train loader with WeightedRandomSampler ───────────────────────
    train_loader = None
    if train_sets:
        combined_train = ConcatDataset(train_sets)
        sampler        = _make_weighted_sampler(train_set_sizes)

        total = sum(train_set_sizes)
        for ds_name, size in zip(dataset_names[:len(train_set_sizes)], train_set_sizes):
            print(
                f"  [sampler] {ds_name:10s}: {size:5d} effective "
                f"({100*size/total:.0f}% → ~50% per batch after rebalancing)"
            )

        train_loader = DataLoader(
            combined_train,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

    # ── Val / test loaders (no resampling) ───────────────────────────
    def _seq_loader(datasets: list[Dataset]) -> DataLoader | None:
        if not datasets:
            return None
        return DataLoader(
            ConcatDataset(datasets),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    val_loader  = _seq_loader(val_sets)
    test_loader = _seq_loader(test_sets)

    n_train = sum(train_set_sizes)
    n_val   = sum(len(d) for d in val_sets)
    print(f"\n  [combined] train {n_train} effective | val {n_val} — {len(dataset_names)} dataset(s)\n")

    return train_loader, val_loader, test_loader