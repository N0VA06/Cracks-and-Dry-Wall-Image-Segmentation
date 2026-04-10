"""
evaluate.py
===========
Per-prompt, per-model evaluation on the held-out test split.

Metrics reported
----------------
  Correctness  : mIoU and Dice for each prompt ("segment crack", "segment taping area")
  Consistency  : per-image IoU std within each dataset (lower = more stable)

Usage
-----
  # Evaluate all trained models
  python evaluate.py

  # Evaluate one model only
  python evaluate.py --model pretrained

  # Use a specific seed for reproducibility
  python evaluate.py --seed 42
"""

import os
import json
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader

import config
from config import (
    IMAGE_SIZE, MEAN, STD, INFERENCE_THRESHOLD,
    CHECKPOINT_DIR, LOG_DIR, PREDICTION_DIR,
    SPLIT_OUTPUT_DIR, DATASET_ROOTS,
    TEXT_VOCAB_SIZE, TEXT_EMBED_DIM, ALL_PROMPTS,
    PROMPT_TEMPLATES, NUM_WORKERS,
)
from models.text_conditioning import SimpleTokenizer, TextEncoder
from preprocessing.coco_dataset import build_dataset, _is_foreground
from preprocessing.transforms import get_val_transforms


# ══════════════════════════════════════════════════════
#  Seed
# ══════════════════════════════════════════════════════

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ══════════════════════════════════════════════════════
#  Canonical prompt per dataset
# ══════════════════════════════════════════════════════

DATASET_CANONICAL_PROMPT = {
    "cracks" : PROMPT_TEMPLATES["crack"][0],     # "segment crack"
    "drywall": PROMPT_TEMPLATES["taping"][0],    # "segment taping area"
}


# ══════════════════════════════════════════════════════
#  Model loading helpers  (mirrors main.py factories)
# ══════════════════════════════════════════════════════

def load_model(model_key: str, device: torch.device):
    ckpt_map = {
        "pretrained": "pretrained_best.pth",
        "custom"    : "custom_best.pth",
        "active"    : "active_best.pth",
        "sam"       : "sam_best.pth",
    }
    ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_map[model_key])

    if model_key == "pretrained":
        from models.pretrained_deeplabv3 import PretrainedDeepLabV3
        model = PretrainedDeepLabV3(
            backbone="resnet50", freeze_backbone=False,
            use_text_prompt=True, text_embed_dim=TEXT_EMBED_DIM,
        )
    elif model_key in ("custom", "active"):
        from models.custom_deeplabv3 import CustomDeepLabV3
        model = CustomDeepLabV3(
            base_channels=config.CUSTOM_BASE_CHANNELS,
            use_dropout=(model_key == "active"),
            use_text_prompt=True, text_embed_dim=TEXT_EMBED_DIM,
        )
    elif model_key == "sam":
        from models.sam_segmentor import SAMSegmentor
        model = SAMSegmentor(
            sam_checkpoint=config.SAM_CHECKPOINT,
            model_type=config.SAM_MODEL_TYPE,
            freeze_encoder=True,
            use_text_prompt=True, text_embed_dim=TEXT_EMBED_DIM,
        )
    else:
        raise ValueError(f"Unknown model: {model_key}")

    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"  Loaded: {ckpt_path}")
    else:
        print(f"  WARNING: checkpoint not found — {ckpt_path}  (random weights)")

    model.to(device).eval()
    return model


def build_text_components(device: torch.device):
    tok = SimpleTokenizer(vocab_size=TEXT_VOCAB_SIZE)
    tok.build_vocab(ALL_PROMPTS)
    enc = TextEncoder(vocab_size=TEXT_VOCAB_SIZE, embed_dim=TEXT_EMBED_DIM)
    enc.to(device).eval()
    return tok, enc


# ══════════════════════════════════════════════════════
#  Single-image inference → binary mask
# ══════════════════════════════════════════════════════

@torch.no_grad()
def infer_mask(
    model, image_tensor: torch.Tensor,
    token_ids: torch.Tensor, text_encoder,
    orig_h: int, orig_w: int,
    threshold: float = INFERENCE_THRESHOLD,
) -> np.ndarray:
    """Returns uint8 mask [H, W] with values {0, 1}."""
    text_embeds = text_encoder(token_ids.unsqueeze(0))
    logits      = model(image_tensor.unsqueeze(0), text_embeds)["out"]
    logits_up   = F.interpolate(logits, (orig_h, orig_w), mode="bilinear", align_corners=False)
    probs       = torch.sigmoid(logits_up).squeeze().cpu().numpy()
    return (probs >= threshold).astype(np.uint8)


# ══════════════════════════════════════════════════════
#  Metric helpers
# ══════════════════════════════════════════════════════

def iou_dice(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    eps  = 1e-7
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    tp   = (pred & gt).sum()
    fp   = (pred & ~gt).sum()
    fn   = (~pred & gt).sum()
    iou  = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    return float(iou), float(dice)


# ══════════════════════════════════════════════════════
#  Evaluate one model on one dataset test split
# ══════════════════════════════════════════════════════

def evaluate_dataset(
    model,
    dataset_name: str,
    prompt: str,
    tokenizer: SimpleTokenizer,
    text_encoder,
    device: torch.device,
    save_samples: int = 4,
    sample_dir: str | None = None,
) -> dict:
    """
    Evaluate model on the test split of one dataset.

    Returns
    -------
    {
      "prompt"       : str,
      "dataset"      : str,
      "n_images"     : int,
      "mean_iou"     : float,
      "mean_dice"    : float,
      "std_iou"      : float,   # consistency metric
      "per_image"    : [{file, iou, dice}]
    }
    """
    # ── Locate test annotation file ───────────────────────────────────
    split_json = os.path.join(SPLIT_OUTPUT_DIR, dataset_name, f"{dataset_name}_test.json")
    root       = DATASET_ROOTS.get(dataset_name, "")
    img_dir    = None
    for c in ("train", "valid", "images"):
        d = os.path.join(root, c)
        if os.path.isdir(d):
            img_dir = d
            break

    if not os.path.isfile(split_json):
        print(f"  [SKIP] Test split not found: {split_json}  (run --mode split first)")
        return {}
    if img_dir is None:
        print(f"  [SKIP] Image directory not found for dataset: {dataset_name}")
        return {}

    ds = build_dataset(split_json, img_dir,
                       transform=get_val_transforms(),
                       tokenizer=tokenizer, train=False)

    # Encode the evaluation prompt once
    ids = tokenizer.encode(prompt)
    ids = (ids + [0] * 16)[:16]
    token_ids = torch.tensor(ids, dtype=torch.long, device=device)

    tf   = get_val_transforms()
    ious, dices = [], []
    per_image   = []

    os.makedirs(sample_dir or PREDICTION_DIR, exist_ok=True)
    saved = 0

    for idx in range(len(ds)):
        image_tensor, gt_mask, _, _ = ds[idx]
        image_tensor = image_tensor.to(device)
        gt_np        = gt_mask.numpy().astype(np.uint8)

        # Original size from the dataset's image info
        img_info = ds.images[ds.image_ids[idx]]
        orig_h, orig_w = img_info["height"], img_info["width"]

        pred = infer_mask(model, image_tensor, token_ids, text_encoder, orig_h, orig_w)

        # Resize GT to original size for fair comparison
        gt_orig = np.array(
            Image.fromarray(gt_np * 255).resize((orig_w, orig_h), Image.NEAREST)
        ) // 255

        iou, dice = iou_dice(pred, gt_orig)
        ious.append(iou)
        dices.append(dice)

        fname = img_info.get("file_name", f"{idx}.jpg")
        per_image.append({"file": fname, "iou": round(iou, 4), "dice": round(dice, 4)})

        # Save sample overlay
        if saved < save_samples and sample_dir:
            _save_sample(
                img_dir, fname, gt_orig, pred,
                f"{dataset_name}_{idx:03d}",
                prompt, sample_dir,
            )
            saved += 1

    return {
        "prompt"    : prompt,
        "dataset"   : dataset_name,
        "n_images"  : len(ious),
        "mean_iou"  : round(float(np.mean(ious)),  4),
        "mean_dice" : round(float(np.mean(dices)), 4),
        "std_iou"   : round(float(np.std(ious)),   4),
        "per_image" : per_image,
    }


def _save_sample(img_dir, fname, gt, pred, stem, prompt, out_dir):
    """Save a 3-panel overlay: image | GT | prediction."""
    try:
        from utils.visualization import save_comparison
        img_path = os.path.join(img_dir, fname)
        if not os.path.isfile(img_path):
            return
        image = np.array(Image.open(img_path).convert("RGB"))
        gt_r  = np.array(Image.fromarray(gt   * 255).resize(
            (image.shape[1], image.shape[0]), Image.NEAREST)) // 255
        pr_r  = np.array(Image.fromarray(pred * 255).resize(
            (image.shape[1], image.shape[0]), Image.NEAREST)) // 255
        save_path = os.path.join(out_dir, f"{stem}__sample.png")
        save_comparison(image, gt_r, pr_r, save_path, title=prompt)
    except Exception:
        pass


# ══════════════════════════════════════════════════════
#  Full evaluation loop
# ══════════════════════════════════════════════════════

def run_evaluation(
    model_keys: list[str],
    device: torch.device,
    seed: int = 42,
    sample_dir: str | None = None,
) -> dict:
    set_seed(seed)
    tokenizer, text_encoder = build_text_components(device)

    all_results = {}   # model → dataset → result dict

    for model_key in model_keys:
        print(f"\n{'─'*56}")
        print(f"  Model: {model_key}")
        print(f"{'─'*56}")
        try:
            model = load_model(model_key, device)
        except Exception as e:
            print(f"  ERROR loading model: {e}")
            continue

        model_results = {}
        for ds_name, prompt in DATASET_CANONICAL_PROMPT.items():
            print(f"  Evaluating on [{ds_name}]  prompt: '{prompt}'")
            result = evaluate_dataset(
                model, ds_name, prompt, tokenizer, text_encoder, device,
                sample_dir=sample_dir or os.path.join(LOG_DIR, "samples", model_key),
            )
            if result:
                model_results[ds_name] = result
                print(
                    f"    mIoU={result['mean_iou']:.4f}  "
                    f"Dice={result['mean_dice']:.4f}  "
                    f"IoU_std={result['std_iou']:.4f}  "
                    f"(n={result['n_images']})"
                )

        all_results[model_key] = model_results

    return all_results


# ══════════════════════════════════════════════════════
#  Pretty table printer
# ══════════════════════════════════════════════════════

def print_results_table(all_results: dict):
    print("\n" + "═" * 78)
    print("  EVALUATION RESULTS")
    print("═" * 78)

    header = f"{'Model':12s} │ {'Dataset':8s} │ {'Prompt':24s} │ {'mIoU':6s} │ {'Dice':6s} │ {'Std↓':6s}"
    print(header)
    print("─" * 78)

    for model_key, ds_results in all_results.items():
        for ds_name, r in ds_results.items():
            print(
                f"{model_key:12s} │ {ds_name:8s} │ {r['prompt']:24s} │ "
                f"{r['mean_iou']:.4f} │ {r['mean_dice']:.4f} │ {r['std_iou']:.4f}"
            )
        print("─" * 78)

    # Summary: mean across both datasets per model
    print("\n  Per-model average (across both datasets):")
    print(f"  {'Model':12s}  mIoU    Dice    Consistency(IoU_std↓)")
    for model_key, ds_results in all_results.items():
        if not ds_results:
            continue
        ious  = [r["mean_iou"]  for r in ds_results.values()]
        dices = [r["mean_dice"] for r in ds_results.values()]
        stds  = [r["std_iou"]   for r in ds_results.values()]
        print(
            f"  {model_key:12s}  {np.mean(ious):.4f}  "
            f"{np.mean(dices):.4f}  {np.mean(stds):.4f}"
        )
    print("═" * 78)


# ══════════════════════════════════════════════════════
#  Save results JSON
# ══════════════════════════════════════════════════════

def save_results(all_results: dict, seed: int):
    path = os.path.join(LOG_DIR, "eval_results.json")
    payload = {"seed": seed, "results": all_results}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Results saved → {path}")
    return path


# ══════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Evaluate trained segmentation models")
    p.add_argument("--model", default="all",
                   choices=["all", "pretrained", "custom", "active", "sam"],
                   help="Which model(s) to evaluate")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--samples", default=None, help="Directory to save visual samples")
    args = p.parse_args()

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}  |  Seed: {args.seed}")

    model_keys = (
        ["pretrained", "custom", "active", "sam"]
        if args.model == "all" else [args.model]
    )

    all_results = run_evaluation(model_keys, device, seed=args.seed,
                                  sample_dir=args.samples)
    print_results_table(all_results)
    save_results(all_results, args.seed)


if __name__ == "__main__":
    main()
