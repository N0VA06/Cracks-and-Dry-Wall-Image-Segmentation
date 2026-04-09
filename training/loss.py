"""
training/loss.py
Combined BCE + Dice loss for binary segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice Loss for binary segmentation (operates on probabilities)."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : [B, 1, H, W] raw logits
            targets : [B, H, W]   binary float masks (0 or 1)
        """
        probs   = torch.sigmoid(logits).squeeze(1)        # [B, H, W]
        targets = targets.float()

        intersection = (probs * targets).sum(dim=(1, 2))
        union        = probs.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class SegmentationLoss(nn.Module):
    """
    Weighted combination of BCEWithLogitsLoss + DiceLoss.

    Args:
        bce_weight  : weight for BCE term
        dice_weight : weight for Dice term
        pos_weight  : optional class-imbalance weight for BCE (scalar tensor)
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        pos_weight: float | None = None,
    ):
        super().__init__()
        self.bce_w  = bce_weight
        self.dice_w = dice_weight

        pw = torch.tensor([pos_weight]) if pos_weight else None
        self.bce  = nn.BCEWithLogitsLoss(pos_weight=pw)
        self.dice = DiceLoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            logits  : [B, 1, H, W]
            targets : [B, H, W]  float

        Returns:
            total_loss, {"bce": ..., "dice": ...}
        """
        bce_loss  = self.bce(logits.squeeze(1), targets)
        dice_loss = self.dice(logits, targets)
        total     = self.bce_w * bce_loss + self.dice_w * dice_loss

        return total, {"bce": bce_loss.item(), "dice": dice_loss.item()}
