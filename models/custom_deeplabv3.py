"""
models/custom_deeplabv3.py
Full DeepLabV3 implementation from scratch:
  • ResNet-like backbone (configurable depth)
  • ASPP module with dilated convolutions
  • Simple decoder head
  • Optional MC-Dropout for active learning uncertainty

Fixes applied vs. original:
  1. FiLM text conditioning is now applied BEFORE mc_drop2.
     Previously FiLM ran after mc_drop2, so gamma (scale) was wasted on
     zeroed-out channels (γ·0 + β = β), breaking text conditioning and
     contaminating the uncertainty estimate with beta offsets.
     Correct order: ASPP → FiLM → mc_drop2 → Decoder.

  2. enable_dropout() now targets only mc_drop1 / mc_drop2.
     The original iterated all modules(), which also enabled ASPP's
     internal Dropout2d(p=0.5) — a training regulariser that must stay
     off during MC-Dropout inference.  A guard is also added so calling
     enable_dropout() on a model built with use_dropout=False raises a
     clear RuntimeError instead of silently enabling ASPP dropout.

  3. get_trainable_params() added for API consistency with
     PretrainedDeepLabV3 (train/main scripts may call it on any model).
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

        # Training regulariser — intentionally NOT a MC-Dropout layer.
        # enable_dropout() will NOT touch this.
        self.dropout = nn.Dropout2d(p=0.5)

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

    Forward flow (with all options enabled):
        backbone → mc_drop1 → ASPP → FiLM → mc_drop2 → Decoder

    FiLM is applied to the full 256-ch ASPP features (before mc_drop2),
    so text conditioning sees complete, unperturbed features and the
    MC-Dropout uncertainty is estimated on the conditioned output.
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
            # MC-Dropout layers — only these are activated by enable_dropout().
            # Placed around the FiLM step so uncertainty is estimated on
            # text-conditioned features entering the decoder.
            self.mc_drop1 = nn.Dropout2d(p=0.3)   # after backbone
            self.mc_drop2 = nn.Dropout2d(p=0.3)   # after FiLM, before decoder

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

        Data flow:
            images
              → backbone                       [B, bc*8, H/8, W/8]
              → mc_drop1 (if use_dropout)      [B, bc*8, H/8, W/8]
              → ASPP (incl. Dropout2d 0.5)     [B, 256,  H/8, W/8]
              → FiLM (if use_text_prompt)      [B, 256,  H/8, W/8]  ← on full features
              → mc_drop2 (if use_dropout)      [B, 256,  H/8, W/8]  ← uncertainty after FiLM
              → Decoder (256→128→64→1 + up)   [B, 1,    H,   W  ]
        """
        input_size = images.shape[-2:]

        # ── Backbone ───────────────────────────────────────────────────
        feats = self.backbone(images)                          # [B, bc*8, H/8, W/8]

        if self.use_dropout:
            feats = self.mc_drop1(feats)

        # ── ASPP ───────────────────────────────────────────────────────
        aspp_out = self.aspp(feats)                            # [B, 256, H/8, W/8]

        # ── FiLM text conditioning (BEFORE mc_drop2) ──────────────────
        # Applied to the complete 256-ch ASPP features so that gamma
        # (scale) modulation is meaningful on every channel.
        if self.use_text_prompt and text_embeds is not None:
            aspp_out = self.text_decoder(aspp_out, text_embeds)

        # ── MC-Dropout (AFTER FiLM) ────────────────────────────────────
        # Uncertainty is estimated on the text-conditioned features that
        # feed the decoder, not on raw features that then get FiLM-shifted.
        if self.use_dropout:
            aspp_out = self.mc_drop2(aspp_out)

        # ── Decoder ────────────────────────────────────────────────────
        logits = self.decoder(aspp_out, input_size)            # [B, 1, H, W]
        return {"out": logits}

    # ------------------------------------------------------------------
    def enable_dropout(self):
        """
        Switch only the MC-Dropout layers (mc_drop1, mc_drop2) to train
        mode for stochastic inference.

        Call after model.eval() to keep BatchNorm in eval mode while
        enabling uncertainty-producing dropout:
            model.eval()
            model.enable_dropout()

        Raises RuntimeError if the model was built with use_dropout=False,
        instead of silently enabling ASPP's regularisation dropout.
        """
        if not self.use_dropout:
            raise RuntimeError(
                "Cannot enable MC-Dropout: model was created with "
                "use_dropout=False.  Rebuild with use_dropout=True."
            )
        # Target ONLY the explicit MC-Dropout layers, not ASPP's p=0.5
        # regularisation dropout which must stay in eval mode.
        self.mc_drop1.train()
        self.mc_drop2.train()

    # ------------------------------------------------------------------
    def get_trainable_params(self):
        """Return trainable parameters (mirrors PretrainedDeepLabV3 API)."""
        return [p for p in self.parameters() if p.requires_grad]