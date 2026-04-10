"""
training/validate.py
Validation loop with per-prompt (crack | taping) IoU and Dice tracking.
"""

import torch
from torch.utils.data import DataLoader

from training.loss import SegmentationLoss
from training.metrics import PromptAwareMetrics


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: SegmentationLoss,
    device: torch.device,
    use_text_prompt: bool = True,
    text_encoder=None,
) -> dict:
    """
    Run one validation epoch.

    Returns:
        {
          "loss"   : float,
          "overall": {iou, dice, acc, precision, recall, n_samples},
          "crack"  : {iou, dice, ...},   (if crack images present)
          "taping" : {iou, dice, ...},   (if taping images present)
          "miou"   : float,              mean IoU across prompt buckets
          "mdice"  : float,              mean Dice across prompt buckets
        }
    """
    model.eval()
    if text_encoder is not None:
        text_encoder.eval()

    metrics    = PromptAwareMetrics()
    total_loss = 0.0
    n_batches  = 0

    for images, masks, token_ids, prompts in loader:
        images    = images.to(device, non_blocking=True)
        masks     = masks.to(device, non_blocking=True)
        token_ids = token_ids.to(device, non_blocking=True)

        # prompts is a tuple of strings from the DataLoader collate
        prompt_list = list(prompts)

        text_embeds = None
        if use_text_prompt and text_encoder is not None:
            text_embeds = text_encoder(token_ids)

        output = model(images, text_embeds)
        logits = output["out"]

        loss, _ = criterion(logits, masks)
        total_loss += loss.item()
        n_batches  += 1

        metrics.update(logits, masks, prompt_list)

    result         = metrics.compute()
    result["loss"] = round(total_loss / max(n_batches, 1), 4)
    return result
