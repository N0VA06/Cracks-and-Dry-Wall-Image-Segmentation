"""
models/custom_deeplabv3.py
Full DeepLabV3 implementation from scratch:
  • ResNet-like backbone (configurable depth)
  • ASPP module with dilated convolutions
  • Simple decoder head
  • Optional MC-Dropout for active learning uncertainty
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.text_conditioning import TextConditionedDecoder


# ══════════════════════════════════════════════════════
#  Building blocks
# ══════════════════════════════════════════════════════

class ConvBnRelu(nn.Sequential):
    """Conv2d → BatchNorm → ReLU."""

    def __init__(
        self, in_ch, out_ch, kernel_size=3, stride=1,
        padding=1, dilation=1, bias=False
    ):
        super().__init__(
            nn.Conv2d(
                in_ch, out_ch, kernel_size,
                stride=stride, padding=padding,
                dilation=dilation, bias=bias
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class ResBlock(nn.Module):
    """Standard residual block with optional stride and dilation."""

    def __init__(self, in_ch, out_ch, stride=1, dilation=1):
        super().__init__()
        padding = dilation
        self.conv1 = ConvBnRelu(in_ch, out_ch, stride=stride, dilation=dilation, padding=padding)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        return self.relu(self.conv2(self.conv1(x)) + self.shortcut(x))


# ══════════════════════════════════════════════════════
#  Backbone
# ══════════════════════════════════════════════════════

class ResNetBackbone(nn.Module):
    """
    Simplified ResNet backbone.
    Output stride = 8 (last two stages use dilation instead of stride).
    Returns feature map at 1/8 input resolution.
    """

    def __init__(self, base_channels: int = 64):
        super().__init__()
        bc = base_channels

        # Stem: 3 → bc, downsample ×2
        self.stem = nn.Sequential(
            ConvBnRelu(3, bc, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(3, stride=2, padding=1),   # ×4 total
        )

        # Stage 1: bc → bc  (no stride, dilation=1)
        self.stage1 = self._make_stage(bc, bc, n_blocks=2, stride=1)

        # Stage 2: bc → bc*2  (stride=2 → ×8 total)
        self.stage2 = self._make_stage(bc, bc * 2, n_blocks=2, stride=2)

        # Stage 3: bc*2 → bc*4  (dilation=2 keeps spatial size)
        self.stage3 = self._make_stage(bc * 2, bc * 4, n_blocks=3, stride=1, dilation=2)

        # Stage 4: bc*4 → bc*8  (dilation=4)
        self.stage4 = self._make_stage(bc * 4, bc * 8, n_blocks=2, stride=1, dilation=4)

        self.out_channels = bc * 8   # expose for ASPP

    @staticmethod
    def _make_stage(in_ch, out_ch, n_blocks, stride=1, dilation=1):
        layers = [ResBlock(in_ch, out_ch, stride=stride, dilation=dilation)]
        for _ in range(1, n_blocks):
            layers.append(ResBlock(out_ch, out_ch, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)    # /4
        x = self.stage1(x)  # /4
        x = self.stage2(x)  # /8
        x = self.stage3(x)  # /8 (dilated)
        x = self.stage4(x)  # /8 (dilated)
        return x


# ══════════════════════════════════════════════════════
#  ASPP
# ══════════════════════════════════════════════════════

class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling.
    Rates: [1, 6, 12, 18] + global average pooling branch.
    """

    def __init__(self, in_channels: int, out_channels: int = 256):
        super().__init__()
        rates = [1, 6, 12, 18]

        self.branches = nn.ModuleList([
            self._aspp_branch(in_channels, out_channels, rate) for rate in rates
        ])

        # Global average pooling branch
        self.global_avg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Projection after concatenation
        self.project = ConvBnRelu(out_channels * (len(rates) + 1), out_channels, kernel_size=1, padding=0)
        self.dropout  = nn.Dropout2d(p=0.5)

    @staticmethod
    def _aspp_branch(in_ch, out_ch, rate):
        if rate == 1:
            return ConvBnRelu(in_ch, out_ch, kernel_size=1, padding=0)
        return ConvBnRelu(in_ch, out_ch, kernel_size=3, dilation=rate, padding=rate)

    def forward(self, x):
        h, w = x.shape[-2:]
        feats = [branch(x) for branch in self.branches]
        gap   = F.interpolate(self.global_avg(x), size=(h, w), mode="bilinear", align_corners=False)
        feats.append(gap)
        out = self.project(torch.cat(feats, dim=1))
        return self.dropout(out)


# ══════════════════════════════════════════════════════
#  Decoder
# ══════════════════════════════════════════════════════

class Decoder(nn.Module):
    """Upsample ASPP features back to input resolution and produce logits."""

    def __init__(self, aspp_channels: int = 256, num_classes: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBnRelu(aspp_channels, 128),
            ConvBnRelu(128, 64),
        )
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, features, input_size):
        x = self.conv(features)
        x = self.classifier(x)
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return x


# ══════════════════════════════════════════════════════
#  Full model
# ══════════════════════════════════════════════════════

class CustomDeepLabV3(nn.Module):
    """
    Custom DeepLabV3 trained from scratch.

    Args:
        base_channels   : backbone width multiplier (default 64)
        aspp_channels   : ASPP output channels (default 256)
        num_classes     : 1 for binary segmentation
        use_dropout     : enable MC-Dropout (for active learning)
        use_text_prompt : enable FiLM text conditioning
        text_embed_dim  : dim of text embedding
    """

    def __init__(
        self,
        base_channels: int = 64,
        aspp_channels: int = 256,
        num_classes: int = 1,
        use_dropout: bool = False,
        use_text_prompt: bool = False,
        text_embed_dim: int = 128,
    ):
        super().__init__()
        self.use_text_prompt = use_text_prompt
        self.use_dropout     = use_dropout

        self.backbone = ResNetBackbone(base_channels)
        self.aspp     = ASPP(self.backbone.out_channels, aspp_channels)
        self.decoder  = Decoder(aspp_channels, num_classes)

        if use_dropout:
            # Extra dropout layers for MC-Dropout inference
            self.mc_drop1 = nn.Dropout2d(p=0.3)
            self.mc_drop2 = nn.Dropout2d(p=0.3)

        if use_text_prompt:
            self.text_decoder = TextConditionedDecoder(
                feature_channels=aspp_channels, text_embed_dim=text_embed_dim
            )

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        images: torch.Tensor,
        text_embeds: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Returns dict with key "out" → logits [B, 1, H, W].
        """
        input_size = images.shape[-2:]

        feats = self.backbone(images)

        if self.use_dropout:
            feats = self.mc_drop1(feats)

        aspp_out = self.aspp(feats)

        if self.use_dropout:
            aspp_out = self.mc_drop2(aspp_out)

        if self.use_text_prompt and text_embeds is not None:
            aspp_out = self.text_decoder(aspp_out, text_embeds)

        logits = self.decoder(aspp_out, input_size)
        return {"out": logits}

    # ------------------------------------------------------------------
    def enable_dropout(self):
        """Switch all Dropout layers to train mode (for MC-Dropout at inference)."""
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.train()
