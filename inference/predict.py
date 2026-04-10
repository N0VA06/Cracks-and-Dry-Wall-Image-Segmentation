"""
inference/predict.py

Text-conditioned binary segmentation inference.

Spec compliance:
  • Output PNG: single-channel, values {0, 255}
  • Spatial size: EXACTLY equal to source image (H × W)
  • Filename: <image_id>__<sanitised_prompt>.png
  • Prompt drives which class the model segments
"""

import os
import re
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path

from config import (
    IMAGE_SIZE, MEAN, STD,
    INFERENCE_THRESHOLD, PREDICTION_DIR,
    ALL_PROMPTS, TEXT_VOCAB_SIZE, TEXT_EMBED_DIM,
)
from models.text_conditioning import SimpleTokenizer, TextEncoder


# ══════════════════════════════════════════════════════
#  Image pre-processing (keeps original size)
# ══════════════════════════════════════════════════════

def load_and_preprocess(image_path: str) -> tuple[torch.Tensor, tuple[int, int]]:
    """
    Load image → normalised [1, 3, H_model, W_model] tensor.
    Returns (tensor, (H_original, W_original)).
    """
    img  = Image.open(image_path).convert("RGB")
    orig_h, orig_w = img.height, img.width

    # Resize to model input size
    img_resized = img.resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.BILINEAR)
    arr = np.array(img_resized).astype(np.float32) / 255.0

    mean = np.array(MEAN, dtype=np.float32)
    std  = np.array(STD,  dtype=np.float32)
    arr  = (arr - mean) / std

    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)   # [1,3,H,W]
    return tensor, (orig_h, orig_w)


# ══════════════════════════════════════════════════════
#  Mask post-processing
# ══════════════════════════════════════════════════════

def logits_to_mask(
    logits: torch.Tensor,
    orig_h: int,
    orig_w: int,
    threshold: float = INFERENCE_THRESHOLD,
) -> np.ndarray:
    """
    logits [1, 1, H_model, W_model]
    → uint8 numpy [H_original, W_original]   values {0, 255}

    Two-step approach to guarantee exact original spatial size:
      1. Bilinear upsample to (orig_h, orig_w)
      2. Threshold → {0, 1} → scale to {0, 255}
    """
    # Upsample to original spatial dimensions
    logits_up = F.interpolate(
        logits,
        size=(orig_h, orig_w),
        mode="bilinear",
        align_corners=False,
    )
    probs  = torch.sigmoid(logits_up).squeeze().cpu().numpy()   # [H, W]  ∈ [0,1]
    binary = (probs >= threshold).astype(np.uint8) * 255        # {0, 255}

    assert binary.shape == (orig_h, orig_w), (
        f"Mask shape mismatch: expected ({orig_h},{orig_w}), got {binary.shape}"
    )
    return binary


# ══════════════════════════════════════════════════════
#  Filename helper
# ══════════════════════════════════════════════════════

def make_output_filename(image_path: str, prompt: str) -> str:
    """
    Build filename: <image_id>__<sanitised_prompt>.png
    Sanitises prompt: lowercase, spaces → underscores, non-word chars stripped.
    """
    image_id    = Path(image_path).stem
    safe_prompt = re.sub(r"[^\w]", "_", prompt.lower()).strip("_")
    safe_prompt = re.sub(r"_+", "_", safe_prompt)[:64]
    return f"{image_id}__{safe_prompt}.png"


# ══════════════════════════════════════════════════════
#  Build default tokenizer + encoder for inference
# ══════════════════════════════════════════════════════

def build_inference_text_components(device: torch.device):
    tokenizer    = SimpleTokenizer(vocab_size=TEXT_VOCAB_SIZE)
    tokenizer.build_vocab(ALL_PROMPTS)
    text_encoder = TextEncoder(vocab_size=TEXT_VOCAB_SIZE, embed_dim=TEXT_EMBED_DIM)
    text_encoder.to(device).eval()
    return tokenizer, text_encoder


# ══════════════════════════════════════════════════════
#  Main predict function
# ══════════════════════════════════════════════════════

def predict(
    model: torch.nn.Module,
    image_path: str,
    prompt: str,
    device: torch.device,
    checkpoint_path: str | None = None,
    tokenizer: SimpleTokenizer | None = None,
    text_encoder=None,
    output_dir: str = PREDICTION_DIR,
    threshold: float = INFERENCE_THRESHOLD,
) -> str:
    """
    Run inference on a single image and save the binary mask PNG.

    Args:
        model          : segmentation model (any of the 4 variants)
        image_path     : path to input image
        prompt         : text prompt e.g. "segment crack"
        device         : torch.device
        checkpoint_path: .pth checkpoint (optional, loaded if given)
        tokenizer      : pre-built SimpleTokenizer
        text_encoder   : TextEncoder module
        output_dir     : where to save the PNG
        threshold      : sigmoid threshold for binary decision

    Returns:
        path to saved PNG
    """
    # ── Load checkpoint ───────────────────────────────────────────────
    if checkpoint_path and os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"[Predict] Loaded checkpoint: {checkpoint_path}")

    model.to(device).eval()

    if text_encoder is not None:
        text_encoder.to(device).eval()

    # ── Build tokenizer if not provided ──────────────────────────────
    if tokenizer is None or text_encoder is None:
        tokenizer, text_encoder = build_inference_text_components(device)

    # ── Preprocess image ─────────────────────────────────────────────
    image_tensor, (orig_h, orig_w) = load_and_preprocess(image_path)
    image_tensor = image_tensor.to(device)

    # ── Encode prompt ─────────────────────────────────────────────────
    ids = tokenizer.encode(prompt)
    ids = (ids + [0] * 16)[: 16]
    token_ids   = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        text_embeds = text_encoder(token_ids)
        output      = model(image_tensor, text_embeds)
        logits      = output["out"]

    # ── Post-process → {0, 255} at original resolution ───────────────
    mask = logits_to_mask(logits, orig_h, orig_w, threshold)

    # ── Save ──────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    filename  = make_output_filename(image_path, prompt)
    save_path = os.path.join(output_dir, filename)

    Image.fromarray(mask, mode="L").save(save_path)
    print(f"[Predict] Saved: {save_path}  (size: {orig_w}×{orig_h}, values: {{0,255}})")
    return save_path


# ══════════════════════════════════════════════════════
#  Batch inference
# ══════════════════════════════════════════════════════

def batch_predict(
    model: torch.nn.Module,
    image_paths: list[str],
    prompt: str,
    device: torch.device,
    **kwargs,
) -> list[str]:
    """Run predict() over a list of images with the same prompt."""
    return [predict(model, p, prompt, device, **kwargs) for p in image_paths]
