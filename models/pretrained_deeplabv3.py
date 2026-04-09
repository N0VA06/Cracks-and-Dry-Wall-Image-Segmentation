"""
models/pretrained_deeplabv3.py
Fine-tuned torchvision DeepLabV3 (ResNet-50/101) for binary segmentation.
"""

import torch
import torch.nn as nn
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    deeplabv3_resnet101,
    DeepLabV3_ResNet50_Weights,
    DeepLabV3_ResNet101_Weights,
)

from models.text_conditioning import TextConditionedDecoder


class PretrainedDeepLabV3(nn.Module):
    """
    Wraps torchvision DeepLabV3 for binary (1-class) segmentation.

    Args:
        backbone    : "resnet50" | "resnet101"
        freeze_backbone : freeze all backbone params
        use_text_prompt : inject FiLM text conditioning in the decoder
        text_embed_dim  : dim of text embedding (used if use_text_prompt=True)
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        freeze_backbone: bool = False,
        use_text_prompt: bool = False,
        text_embed_dim: int = 128,
    ):
        super().__init__()
        self.use_text_prompt = use_text_prompt

        # ── Load pretrained model ─────────────────────────────────────
        if backbone == "resnet50":
            weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
            self.model = deeplabv3_resnet50(weights=weights)
        elif backbone == "resnet101":
            weights = DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
            self.model = deeplabv3_resnet101(weights=weights)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # ── Replace classifier head for binary output ─────────────────
        in_channels = self.model.classifier[4].in_channels   # 256
        self.model.classifier[4] = nn.Conv2d(in_channels, 1, kernel_size=1)

        # Replace aux classifier too (used during training)
        if self.model.aux_classifier is not None:
            aux_in = self.model.aux_classifier[4].in_channels
            self.model.aux_classifier[4] = nn.Conv2d(aux_in, 1, kernel_size=1)

        # ── Optional backbone freeze ───────────────────────────────────
        if freeze_backbone:
            for name, param in self.model.backbone.named_parameters():
                param.requires_grad = False

        # ── Optional text conditioning ─────────────────────────────────
        if use_text_prompt:
            self.text_decoder = TextConditionedDecoder(
                feature_channels=256, text_embed_dim=text_embed_dim
            )

    # ------------------------------------------------------------------
    def forward(
        self,
        images: torch.Tensor,
        text_embeds: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Returns dict with keys:
            "out"  – main logits  [B, 1, H, W]
            "aux"  – aux logits   [B, 1, H, W] (only when aux_classifier present)
        """
        result = self.model(images)
        out = result["out"]  # [B, 1, H, W]

        if self.use_text_prompt and text_embeds is not None:
            out = self.text_decoder(out, text_embeds)

        result["out"] = out
        return result

    # ------------------------------------------------------------------
    def get_trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]
