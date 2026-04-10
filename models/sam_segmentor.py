"""
models/sam_segmentor.py

Meta's Segment Anything Model (SAM) adapted for binary segmentation.

Strategy:
  • SAM ViT image encoder is used as a frozen (or optionally unfrozen) backbone
    → produces [B, 256, 64, 64] dense embeddings
  • A lightweight ASPP-style BinaryDecoder is trained on top for pixel-level output
  • Optionally: SAM's own mask decoder is used in zero-shot / prompt mode
    for quick inference without any fine-tuning

Checkpoint download:
  python scripts/download_sam.py          # downloads vit_b (~375 MB)

Requirements:
  pip install git+https://github.com/facebookresearch/segment-anything.git
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.text_conditioning import TextConditionedDecoder


# ══════════════════════════════════════════════════════
#  Lightweight decoder on top of SAM image embeddings
# ══════════════════════════════════════════════════════

class ASPPConv(nn.Sequential):
    def __init__(self, in_ch, out_ch, dilation):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class BinaryDecoder(nn.Module):
    """
    ASPP + upsampling head that takes SAM's 256-channel image embeddings
    (spatial size H/16 of original, e.g. 64×64 for 1024 input) and
    produces a [B, 1, H, W] binary logit map at full input resolution.

    Args:
        in_channels  : SAM embedding channels (always 256)
        aspp_channels: internal ASPP width
        num_classes  : 1 for binary segmentation
    """

    def __init__(self, in_channels: int = 256, aspp_channels: int = 128, num_classes: int = 1):
        super().__init__()
        rates = [1, 6, 12, 18]

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, aspp_channels, 1 if r == 1 else 3,
                          padding=0 if r == 1 else r, dilation=1 if r == 1 else r, bias=False),
                nn.BatchNorm2d(aspp_channels), nn.ReLU(inplace=True),
            ) for r in rates
        ])
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, aspp_channels, 1, bias=False),
            nn.BatchNorm2d(aspp_channels), nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(aspp_channels * (len(rates) + 1), aspp_channels, 1, bias=False),
            nn.BatchNorm2d(aspp_channels), nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
        )
        self.head = nn.Sequential(
            nn.Conv2d(aspp_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1),
        )

    def forward(self, x: torch.Tensor, target_size: tuple) -> torch.Tensor:
        h, w = x.shape[-2:]
        feats = [b(x) for b in self.branches]
        feats.append(F.interpolate(self.gap(x), (h, w), mode="bilinear", align_corners=False))
        x = self.project(torch.cat(feats, dim=1))
        x = self.head(x)
        return F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)


# ══════════════════════════════════════════════════════
#  Main model
# ══════════════════════════════════════════════════════

class SAMSegmentor(nn.Module):
    """
    Fine-tunable SAM wrapper for binary semantic segmentation.

    Args:
        sam_checkpoint    : path to SAM .pth weights (e.g. sam_vit_b_01ec64.pth)
        model_type        : "vit_b" | "vit_l" | "vit_h"
        freeze_encoder    : if True, SAM image encoder weights are frozen (recommended)
        aspp_channels     : width of the custom binary decoder
        use_text_prompt   : enable FiLM text conditioning on decoder features
        text_embed_dim    : dim of text embedding
    """

    SAM_EMBED_DIM = 256   # fixed for all SAM variants

    def __init__(
        self,
        sam_checkpoint: str,
        model_type: str = "vit_b",
        freeze_encoder: bool = True,
        aspp_channels: int = 128,
        use_text_prompt: bool = False,
        text_embed_dim: int = 128,
    ):
        super().__init__()
        self.use_text_prompt = use_text_prompt

        # ── Load SAM ─────────────────────────────────────────────────
        try:
            from segment_anything import sam_model_registry
        except ImportError:
            raise ImportError(
                "segment-anything not installed.\n"
                "Run: pip install git+https://github.com/facebookresearch/segment-anything.git"
            )

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.image_encoder = sam.image_encoder     # ViT backbone
        self.image_size    = sam.image_encoder.img_size  # typically 1024

        # ── Freeze encoder ────────────────────────────────────────────
        if freeze_encoder:
            for p in self.image_encoder.parameters():
                p.requires_grad = False

        # ── Custom binary decoder ─────────────────────────────────────
        self.decoder = BinaryDecoder(
            in_channels=self.SAM_EMBED_DIM,
            aspp_channels=aspp_channels,
        )

        # ── Optional text conditioning on decoder input ───────────────
        if use_text_prompt:
            self.text_cond = TextConditionedDecoder(
                feature_channels=self.SAM_EMBED_DIM,
                text_embed_dim=text_embed_dim,
            )

        self._init_decoder()

    # ------------------------------------------------------------------
    def _init_decoder(self):
        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def _resize_for_sam(self, images: torch.Tensor) -> torch.Tensor:
        """
        SAM's ViT expects (1024, 1024) by default.
        We resize only if the input differs from self.image_size.
        """
        target = self.image_size
        if images.shape[-1] != target or images.shape[-2] != target:
            images = F.interpolate(images, (target, target), mode="bilinear", align_corners=False)
        return images

    # ------------------------------------------------------------------
    def forward(
        self,
        images: torch.Tensor,
        text_embeds: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            images      : [B, 3, H, W]  (normalised per SAM convention)
            text_embeds : [B, T] optional

        Returns:
            dict with "out" → logits [B, 1, H, W] at original input resolution
        """
        original_size = images.shape[-2:]

        # Resize to SAM's expected input
        sam_input = self._resize_for_sam(images)

        # Extract dense image embeddings: [B, 256, 64, 64]
        image_embeddings = self.image_encoder(sam_input)

        # Optional text conditioning before decoding
        if self.use_text_prompt and text_embeds is not None:
            image_embeddings = self.text_cond(image_embeddings, text_embeds)

        # Decode → [B, 1, H_orig, W_orig]
        logits = self.decoder(image_embeddings, target_size=original_size)

        return {"out": logits}

    # ------------------------------------------------------------------
    def get_trainable_params(self) -> list:
        """Return only decoder (+ optional text cond) params for optimizer."""
        params = list(self.decoder.parameters())
        if self.use_text_prompt and hasattr(self, "text_cond"):
            params += list(self.text_cond.parameters())
        return params

    # ------------------------------------------------------------------
    @torch.no_grad()
    def zero_shot_predict(
        self,
        image_np: "np.ndarray",
        sam_checkpoint: str,
        model_type: str = "vit_b",
        points_per_side: int = 32,
    ) -> "np.ndarray":
        """
        Zero-shot binary mask via SAM's automatic mask generator.
        Unions all generated masks → single foreground mask.

        Args:
            image_np : [H, W, 3] uint8 numpy image
        Returns:
            binary mask [H, W] uint8
        """
        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        except ImportError:
            raise ImportError("segment-anything not installed.")

        import numpy as np
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.eval()
        generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=points_per_side,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
        )
        masks = generator.generate(image_np)
        H, W  = image_np.shape[:2]
        union = np.zeros((H, W), dtype=np.uint8)
        for m in masks:
            union = np.maximum(union, m["segmentation"].astype(np.uint8))
        return union
