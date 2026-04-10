"""
training/loss.py
BCE (with pos_weight) + Dice loss for binary segmentation.

pos_weight is critical for thin-feature datasets:
  tape / cracks occupy ~1-5% of pixels → BCE trivially minimised by
  predicting all-background → IoU stays 0 even as loss falls.
  pos_weight=15 penalises missing a foreground pixel 15× vs background,
  counteracting the ~1:15–1:50 pixel imbalance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import BCE_WEIGHT, DICE_WEIGHT, BCE_POS_WEIGHT


class DiceLoss(nn.Module):
    """Soft Dice Loss — operates on sigmoid probabilities, not hard predictions."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : [B, 1, H, W]
        targets : [B, H, W]  float 0/1
        """
        probs  = torch.sigmoid(logits).squeeze(1)   # [B, H, W]
        tgts   = targets.float()
        inter  = (probs * tgts).sum(dim=(1, 2))
        union  = probs.sum(dim=(1, 2)) + tgts.sum(dim=(1, 2))
        dice   = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class SegmentationLoss(nn.Module):
    """
    BCE (weighted) + Dice loss.

    pos_weight is stored as a module buffer so it follows model.to(device) /
    model.cuda() automatically — avoids device mismatch errors.

    Args:
        bce_weight  : scalar multiplier for BCE term
        dice_weight : scalar multiplier for Dice term
        pos_weight  : foreground pixel weight in BCE (config default: 15.0)
    """

    def __init__(
        self,
        bce_weight:  float = BCE_WEIGHT,
        dice_weight: float = DICE_WEIGHT,
        pos_weight:  float = BCE_POS_WEIGHT,
    ):
        super().__init__()
        self.bce_w  = bce_weight
        self.dice_w = dice_weight
        # Register buffer: moves to GPU with the criterion automatically
        self.register_buffer("pw", torch.tensor([pos_weight], dtype=torch.float32))
        self.dice = DiceLoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        logits  : [B, 1, H, W]
        targets : [B, H, W]  float 0/1

        Returns (total_loss, {"bce": float, "dice": float})
        """
        # pos_weight on same device as logits (buffer ensures this)
        bce_loss  = F.binary_cross_entropy_with_logits(
            logits.squeeze(1),
            targets.float(),
            pos_weight=self.pw,
        )
        dice_loss = self.dice(logits, targets)
        total     = self.bce_w * bce_loss + self.dice_w * dice_loss

        return total, {"bce": bce_loss.item(), "dice": dice_loss.item()}