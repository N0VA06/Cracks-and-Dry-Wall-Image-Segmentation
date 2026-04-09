"""
training/metrics.py
Binary segmentation metrics with per-prompt tracking.

Two levels of granularity:
  • SegmentationMetrics  – overall IoU / Dice / accuracy accumulator
  • PromptAwareMetrics   – per-bucket (crack | taping) + overall mIoU / mDice
"""

import torch
import numpy as np


# ══════════════════════════════════════════════════════
#  Prompt → bucket mapping
# ══════════════════════════════════════════════════════

PROMPT_BUCKETS = {
    "crack": ["crack"],
    "taping": ["taping", "tape", "seam", "joint", "drywall"],
}

def prompt_to_bucket(prompt_text: str) -> str:
    """Map any prompt string to its canonical bucket key."""
    p = prompt_text.lower()
    for bucket, keywords in PROMPT_BUCKETS.items():
        if any(k in p for k in keywords):
            return bucket
    return "other"


# ══════════════════════════════════════════════════════
#  Single-prompt accumulator
# ══════════════════════════════════════════════════════

class SegmentationMetrics:
    """Pixel-level binary metrics accumulator for one prompt bucket."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.tn = 0.0
        self.n_samples = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        logits  : [B, 1, H, W]
        targets : [B, H, W]  float 0/1
        """
        preds = (torch.sigmoid(logits.squeeze(1)) >= self.threshold).float()
        tgts  = targets.float()
        self.tp += (preds * tgts).sum().item()
        self.fp += (preds * (1 - tgts)).sum().item()
        self.fn += ((1 - preds) * tgts).sum().item()
        self.tn += ((1 - preds) * (1 - tgts)).sum().item()
        self.n_samples += tgts.shape[0]

    def compute(self) -> dict[str, float]:
        eps  = 1e-7
        iou  = self.tp / (self.tp + self.fp + self.fn + eps)
        dice = 2 * self.tp / (2 * self.tp + self.fp + self.fn + eps)
        acc  = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn + eps)
        prec = self.tp / (self.tp + self.fp + eps)
        rec  = self.tp / (self.tp + self.fn + eps)
        return {
            "iou"      : round(iou,  4),
            "dice"     : round(dice, 4),
            "acc"      : round(acc,  4),
            "precision": round(prec, 4),
            "recall"   : round(rec,  4),
            "n_samples": int(self.n_samples),
        }

    def is_empty(self) -> bool:
        return self.n_samples == 0


# ══════════════════════════════════════════════════════
#  Multi-prompt accumulator
# ══════════════════════════════════════════════════════

class PromptAwareMetrics:
    """
    Tracks IoU / Dice per prompt bucket AND overall mIoU / mDice.

    Usage in validation loop:
        metrics = PromptAwareMetrics()
        for images, masks, token_ids, prompts in loader:
            ...
            metrics.update(logits, masks, prompts)
        result = metrics.compute()
        # result["crack"]["iou"], result["taping"]["dice"], result["miou"]
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.overall   = SegmentationMetrics(threshold)
        self.buckets: dict[str, SegmentationMetrics] = {}

    def reset(self):
        self.overall.reset()
        for m in self.buckets.values():
            m.reset()

    @torch.no_grad()
    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        prompts: list[str],
    ):
        """
        logits  : [B, 1, H, W]
        targets : [B, H, W] float
        prompts : list of B prompt strings
        """
        # Overall metrics
        self.overall.update(logits, targets)

        # Per-bucket metrics — iterate over unique buckets in this batch
        bucket_indices: dict[str, list[int]] = {}
        for i, p in enumerate(prompts):
            b = prompt_to_bucket(p)
            bucket_indices.setdefault(b, []).append(i)

        for bucket, idxs in bucket_indices.items():
            if bucket not in self.buckets:
                self.buckets[bucket] = SegmentationMetrics(self.threshold)
            idx_t = torch.tensor(idxs, device=logits.device)
            self.buckets[bucket].update(
                logits[idx_t],
                targets[idx_t],
            )

    def compute(self) -> dict:
        """
        Returns:
            {
              "overall": {iou, dice, acc, precision, recall, n_samples},
              "crack":   {iou, dice, ...},   # only if crack samples seen
              "taping":  {iou, dice, ...},   # only if taping samples seen
              "miou":    float,              # mean IoU across non-empty buckets
              "mdice":   float,              # mean Dice across non-empty buckets
            }
        """
        result = {"overall": self.overall.compute()}

        per_bucket_ious  = []
        per_bucket_dices = []

        for bucket, m in sorted(self.buckets.items()):
            if not m.is_empty():
                stats = m.compute()
                result[bucket] = stats
                per_bucket_ious.append(stats["iou"])
                per_bucket_dices.append(stats["dice"])

        # mIoU / mDice = mean across all non-empty prompt buckets
        if per_bucket_ious:
            result["miou"]  = round(float(np.mean(per_bucket_ious)),  4)
            result["mdice"] = round(float(np.mean(per_bucket_dices)), 4)
        else:
            # Fall back to overall if no per-bucket data
            result["miou"]  = result["overall"]["iou"]
            result["mdice"] = result["overall"]["dice"]

        return result


# ══════════════════════════════════════════════════════
#  Logging helper
# ══════════════════════════════════════════════════════

def format_metrics_log(metrics: dict) -> str:
    """
    Produce a human-readable multi-line metrics summary.

    Example:
        mIoU 0.7234  mDice 0.8312  (overall IoU 0.7100)
          crack  → IoU 0.7821  Dice 0.8634  [n=120]
          taping → IoU 0.6647  Dice 0.7990  [n=80]
    """
    lines = []
    lines.append(
        f"  mIoU {metrics.get('miou', 0):.4f}  "
        f"mDice {metrics.get('mdice', 0):.4f}  "
        f"(overall IoU {metrics['overall']['iou']:.4f})"
    )
    for bucket in ("crack", "taping", "other"):
        if bucket in metrics:
            m = metrics[bucket]
            lines.append(
                f"    {bucket:<7s}→ IoU {m['iou']:.4f}  "
                f"Dice {m['dice']:.4f}  "
                f"[n={m['n_samples']}]"
            )
    return "\n".join(lines)
