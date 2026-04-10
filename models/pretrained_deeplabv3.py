"""
models/pretrained_deeplabv3.py
Fine-tuned torchvision DeepLabV3 (ResNet-50/101) for binary segmentation.

FiLM text conditioning is injected into the 256-channel ASPP decoder
features, immediately before the final 1×1 classification head.

torchvision DeepLabHead layout (indices):
  [0] ASPP
  [1] Conv2d(256, 256, 3, padding=1, bias=False)
  [2] BatchNorm2d(256)
  [3] ReLU()
  [4] Conv2d(256, num_classes, 1)   ← replaced with 256→1

We run [0..3] → FiLM (on 256-ch) → [4] manually, bypassing the model's
black-box forward so the conditioning sees the correct feature dimension.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        backbone        : "resnet50" | "resnet101"
        freeze_backbone : freeze all backbone params
        use_text_prompt : inject FiLM text conditioning on 256-ch ASPP features
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

        # ── Load pretrained model ──────────────────────────────────────
        if backbone == "resnet50":
            weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
            self.model = deeplabv3_resnet50(weights=weights)
        elif backbone == "resnet101":
            weights = DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
            self.model = deeplabv3_resnet101(weights=weights)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # ── Replace classifier head for binary output ──────────────────
        # DeepLabHead[-1] is Conv2d(256, num_classes, 1);
        # we replace it with a binary head (256 → 1).
        in_channels = self.model.classifier[-1].in_channels   # 256
        self.model.classifier[-1] = nn.Conv2d(in_channels, 1, kernel_size=1)

        # Replace aux classifier head (FCNHead[-1]) for binary output.
        self._has_aux = self.model.aux_classifier is not None
        if self._has_aux:
            aux_in = self.model.aux_classifier[-1].in_channels
            self.model.aux_classifier[-1] = nn.Conv2d(aux_in, 1, kernel_size=1)

        # ── Optional backbone freeze ───────────────────────────────────
        if freeze_backbone:
            for param in self.model.backbone.parameters():
                param.requires_grad = False

        # ── FiLM text conditioning on 256-ch ASPP features ────────────
        # Placed here (not on the 1-ch logit) so the channel dim matches.
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
            "aux"  – aux logits   [B, 1, H, W]  (training only, when aux head present)

        Forward flow:
            backbone → ASPP body (cls[0:4]) → FiLM (256-ch) → cls[-1] → upsample
        """
        input_shape = images.shape[-2:]

        # ── Backbone ───────────────────────────────────────────────────
        # Returns OrderedDict with keys "out" and "aux".
        features = self.model.backbone(images)

        # ── Main decoder branch ────────────────────────────────────────
        x = features["out"]

        # Run all classifier layers except the final 1×1 head.
        # After this loop x is [B, 256, h, w] — the 256-ch ASPP output.
        cls_layers = list(self.model.classifier.children())
        for layer in cls_layers[:-1]:
            x = layer(x)

        # FiLM on the correct 256-ch features (not on 1-ch logits).
        if self.use_text_prompt and text_embeds is not None:
            x = self.text_decoder(x, text_embeds)   # [B, 256, h, w]

        # Final 1×1 classification head → [B, 1, h, w]
        x = cls_layers[-1](x)
        out = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)

        result = {"out": out}

        # ── Aux branch (training only) ─────────────────────────────────
        if self._has_aux and self.training:
            aux = features["aux"]
            aux_layers = list(self.model.aux_classifier.children())
            for layer in aux_layers[:-1]:
                aux = layer(aux)
            aux = aux_layers[-1](aux)
            aux = F.interpolate(aux, size=input_shape, mode="bilinear", align_corners=False)
            result["aux"] = aux

        return result

    # ------------------------------------------------------------------
    def get_trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]