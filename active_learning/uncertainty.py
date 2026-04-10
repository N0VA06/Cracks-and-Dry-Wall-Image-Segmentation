"""
active_learning/uncertainty.py
Entropy-based uncertainty estimation via MC-Dropout.

For each unlabelled image, we:
  1. Enable dropout at inference time
  2. Run N stochastic forward passes
  3. Average the sigmoid probabilities → mean prediction
  4. Compute pixel-wise entropy → scalar uncertainty score per image
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader


@torch.no_grad()
def mc_dropout_uncertainty(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_passes: int = 10,
    use_text_prompt: bool = False,
    text_encoder=None,
) -> np.ndarray:
    """
    Estimate uncertainty for every sample in `loader`.

    Returns:
        uncertainty_scores : np.ndarray of shape [N], one score per sample.
                             Higher = more uncertain.
    """
    model.eval()
    # Turn dropout layers back to train mode (MC-Dropout trick)
    if hasattr(model, "enable_dropout"):
        model.enable_dropout()
    else:
        _enable_dropout(model)

    all_scores = []

    for images, _, token_ids, _ in loader:
        images    = images.to(device)
        token_ids = token_ids.to(device)
        B         = images.shape[0]

        text_embeds = None
        if use_text_prompt and text_encoder is not None:
            with torch.enable_grad():
                pass
            text_embeds = text_encoder(token_ids)

        # Stochastic forward passes
        probs_stack = []   # [n_passes, B, H, W]
        for _ in range(n_passes):
            out    = model(images, text_embeds)
            probs  = torch.sigmoid(out["out"].squeeze(1))   # [B, H, W]
            probs_stack.append(probs.cpu())

        probs_stack = torch.stack(probs_stack, dim=0)   # [T, B, H, W]
        mean_probs  = probs_stack.mean(dim=0)            # [B, H, W]

        # Pixel-wise entropy: -p log p - (1-p) log(1-p)
        eps     = 1e-8
        entropy = (
            -mean_probs * torch.log(mean_probs + eps)
            - (1 - mean_probs) * torch.log(1 - mean_probs + eps)
        )   # [B, H, W]

        # Scalar score = mean entropy over spatial dims
        scores = entropy.mean(dim=(1, 2)).numpy()   # [B]
        all_scores.append(scores)

    return np.concatenate(all_scores, axis=0)


def _enable_dropout(model: torch.nn.Module):
    """Put all Dropout / Dropout2d layers in train mode."""
    import torch.nn as nn
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()
