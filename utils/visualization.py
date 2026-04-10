"""
utils/visualization.py
Visualization helpers:
  • Overlay predicted mask on image
  • Side-by-side comparison (image | gt | pred)
  • Training curve plots
  • Model comparison bar chart
"""

import os
import re
import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    _MPL = True
except ImportError:
    _MPL = False


# ══════════════════════════════════════════════════════
#  Mask overlay
# ══════════════════════════════════════════════════════

def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple = (255, 0, 0),
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Blend a binary mask onto an RGB image.

    Args:
        image : [H, W, 3] uint8
        mask  : [H, W]    binary (0 or 1)
        color : RGB tuple for foreground
        alpha : blend factor
    Returns:
        [H, W, 3] uint8 blended image
    """
    overlay = image.copy().astype(np.float32)
    fg      = (mask > 0)
    for c, col in enumerate(color):
        overlay[fg, c] = (1 - alpha) * overlay[fg, c] + alpha * col
    return overlay.clip(0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════
#  Side-by-side comparison
# ══════════════════════════════════════════════════════

def save_comparison(
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    save_path: str,
    title: str = "",
):
    """Save a 1×3 panel: image | gt overlay | pred overlay."""
    if not _MPL:
        _save_comparison_pil(image, gt_mask, pred_mask, save_path)
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    if title:
        fig.suptitle(title, fontsize=13)

    axes[0].imshow(image)
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(overlay_mask(image, gt_mask, color=(0, 200, 0)))
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(overlay_mask(image, pred_mask, color=(255, 50, 50)))
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _save_comparison_pil(image, gt, pred, save_path):
    """Fallback PIL comparison panel."""
    H, W   = image.shape[:2]
    canvas = Image.new("RGB", (W * 3, H))
    canvas.paste(Image.fromarray(image), (0, 0))
    canvas.paste(Image.fromarray(overlay_mask(image, gt, (0, 200, 0))), (W, 0))
    canvas.paste(Image.fromarray(overlay_mask(image, pred, (255, 50, 50))), (W * 2, 0))
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    canvas.save(save_path)


# ══════════════════════════════════════════════════════
#  Training curve
# ══════════════════════════════════════════════════════

def plot_training_curve(log_path: str, save_path: str | None = None):
    """Read training log JSON and plot loss + per-prompt IoU curves."""
    if not _MPL:
        print("[visualization] matplotlib not available – skipping plot.")
        return

    with open(log_path) as f:
        history = json.load(f)

    epochs     = [r["epoch"]          for r in history]
    train_loss = [r["train"]["loss"]   for r in history]
    miou       = [r["val"].get("miou", r["val"]["overall"]["iou"])  for r in history]
    mdice      = [r["val"].get("mdice", r["val"]["overall"]["dice"]) for r in history]
    crack_iou  = [r["val"].get("crack",  {}).get("iou",  None) for r in history]
    taping_iou = [r["val"].get("taping", {}).get("iou",  None) for r in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_loss, label="Train Loss", color="steelblue", linewidth=2)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss"); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, miou,  label="mIoU",       color="darkorange", linewidth=2)
    ax2.plot(epochs, mdice, label="mDice",       color="seagreen",   linewidth=2, linestyle="--")
    if any(v is not None for v in crack_iou):
        ax2.plot(epochs, [v or 0 for v in crack_iou],  label="crack IoU",  color="steelblue",  linestyle=":")
    if any(v is not None for v in taping_iou):
        ax2.plot(epochs, [v or 0 for v in taping_iou], label="taping IoU", color="tomato",      linestyle=":")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Score"); ax2.set_ylim(0, 1)
    ax2.set_title("Validation Metrics (per prompt)"); ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = save_path or log_path.replace(".json", "_curve.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualization] Curve saved: {out}")


# ══════════════════════════════════════════════════════
#  Model comparison bar chart
# ══════════════════════════════════════════════════════

def plot_model_comparison(
    metrics_by_model: dict[str, dict],
    save_path: str,
    metric_keys: list[str] = ("iou", "dice"),
):
    """
    Bar chart comparing multiple models.

    Args:
        metrics_by_model : {"pretrained": {"iou": .85, "dice": .91}, ...}
        save_path        : output PNG path
        metric_keys      : which metrics to include in the chart
    """
    if not _MPL:
        print("[visualization] matplotlib not available – skipping comparison plot.")
        _print_comparison_table(metrics_by_model)
        return

    models  = list(metrics_by_model.keys())
    n_m     = len(metric_keys)
    x       = np.arange(len(models))
    width   = 0.8 / n_m

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 2.5), 5))
    colors  = ["steelblue", "darkorange", "seagreen", "tomato"]

    for i, key in enumerate(metric_keys):
        vals = [metrics_by_model[m].get(key, 0) for m in models]
        bars = ax.bar(x + i * width - (n_m - 1) * width / 2, vals, width, label=key.upper(), color=colors[i % len(colors)])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualization] Comparison chart saved: {save_path}")


def save_val_panels(
    model: "torch.nn.Module",
    loader: "DataLoader",
    device: "torch.device",
    save_dir: str,
    model_tag: str = "model",
    n_panels: int = 4,
    text_encoder=None,
    threshold: float = 0.5,
    mean: list = None,
    std: list = None,
):
    """
    Save panels: Original | Ground Truth | Prediction.
    Guarantees at least n_panels // 2 from each prompt bucket (crack / taping)
    so both classes are always represented in the visuals.

    Filenames: <model_tag>_panel_<bucket>_<n>__<prompt>.png
    """
    import torch
    from training.metrics import prompt_to_bucket

    if mean is None:
        from config import MEAN, STD, INFERENCE_THRESHOLD
        mean, std = MEAN, STD
        threshold = INFERENCE_THRESHOLD

    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    if text_encoder is not None:
        text_encoder.eval()

    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t  = torch.tensor(std).view(3, 1, 1)

    # Collect up to n_panels // 2 from each bucket
    per_bucket   = max(1, n_panels // 2)
    bucket_count: dict[str, int] = {}
    total_saved  = 0

    for images, masks, token_ids, prompts in loader:
        if total_saved >= n_panels:
            break

        images    = images.to(device)
        masks     = masks.to(device)
        token_ids = token_ids.to(device)

        with torch.no_grad():
            text_embeds = text_encoder(token_ids) if text_encoder is not None else None
            logits      = model(images, text_embeds)["out"]
            pred_masks  = (torch.sigmoid(logits.squeeze(1)) >= threshold).float()

        for i in range(images.shape[0]):
            if total_saved >= n_panels:
                break

            prompt = prompts[i] if isinstance(prompts, (list, tuple)) else list(prompts)[i]
            bucket = prompt_to_bucket(prompt)

            # Skip if we already have enough from this bucket
            if bucket_count.get(bucket, 0) >= per_bucket:
                continue

            # De-normalise image
            img_t  = images[i].cpu() * std_t + mean_t
            img_np = (img_t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            gt_np   = masks[i].cpu().numpy().astype(np.uint8)
            pred_np = pred_masks[i].cpu().numpy().astype(np.uint8)

            n_in_bucket = bucket_count.get(bucket, 0) + 1
            safe_prompt = re.sub(r'[^\w]', '_', prompt)[:30]
            fname = f"{model_tag}_{bucket}_{n_in_bucket:02d}__{safe_prompt}.png"

            save_comparison(
                img_np, gt_np, pred_np,
                save_path=os.path.join(save_dir, fname),
                title=f"{model_tag} | {prompt}",
            )

            bucket_count[bucket] = n_in_bucket
            total_saved += 1

    summary = "  ".join(f"{b}:{c}" for b, c in sorted(bucket_count.items()))
    print(f"[visualization] {total_saved} panels saved [{summary}] → {save_dir}")



def _print_comparison_table(metrics_by_model: dict):
    print("\nModel Comparison")
    print("-" * 50)
    for model, m in metrics_by_model.items():
        scores = " | ".join(f"{k}: {v:.4f}" for k, v in m.items())
        print(f"  {model:20s} → {scores}")