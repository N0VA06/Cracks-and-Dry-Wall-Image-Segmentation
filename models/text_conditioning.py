"""
models/text_conditioning.py
Lightweight text encoder + FiLM-style feature modulation.

Usage:
  encoder = TextEncoder(vocab_size=1000, embed_dim=128)
  text_embed = encoder(token_ids)              # [B, text_embed_dim]

  decoder = TextConditionedDecoder(256, 128)
  modulated = decoder(visual_features, text_embed)   # [B, 256, H, W]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════
#  Simple word-level text encoder
# ══════════════════════════════════════════════════════

class TextEncoder(nn.Module):
    """
    Bag-of-words text encoder:
      token ids → mean-pooled embedding → MLP → [B, embed_dim]

    For production you could swap this for a CLIP text encoder.
    """

    def __init__(self, vocab_size: int = 1000, embed_dim: int = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.mlp   = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids : [B, T] long tensor (0 = pad)
        Returns:
            [B, embed_dim]
        """
        mask = (token_ids != 0).float().unsqueeze(-1)      # [B, T, 1]
        emb  = self.embed(token_ids) * mask                 # zero out pads
        pooled = emb.sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # mean
        return self.mlp(pooled)


# ══════════════════════════════════════════════════════
#  Simple tokenizer (rule-based, no external deps)
# ══════════════════════════════════════════════════════

class SimpleTokenizer:
    """
    Minimal whitespace tokenizer with a built vocabulary.
    Builds vocab from seen sentences; pads/truncates to max_len.
    """

    PAD, UNK = 0, 1

    def __init__(self, vocab_size: int = 1000, max_len: int = 16):
        self.vocab_size = vocab_size
        self.max_len    = max_len
        self.word2idx   = {"<pad>": 0, "<unk>": 1}

    def build_vocab(self, sentences: list[str]):
        from collections import Counter
        counter = Counter()
        for s in sentences:
            counter.update(s.lower().split())
        for word, _ in counter.most_common(self.vocab_size - 2):
            if word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)

    def encode(self, sentence: str) -> list[int]:
        tokens = sentence.lower().split()[: self.max_len]
        ids = [self.word2idx.get(w, self.UNK) for w in tokens]
        ids += [self.PAD] * (self.max_len - len(ids))
        return ids

    def batch_encode(self, sentences: list[str]) -> torch.Tensor:
        ids = [self.encode(s) for s in sentences]
        return torch.tensor(ids, dtype=torch.long)


# ══════════════════════════════════════════════════════
#  FiLM modulation
# ══════════════════════════════════════════════════════

class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation.
    Given text embedding t ∈ [B, T], predicts γ, β ∈ [B, C]
    and applies: out = γ * features + β  (channel-wise).
    """

    def __init__(self, text_embed_dim: int, feature_channels: int):
        super().__init__()
        self.gamma_proj = nn.Linear(text_embed_dim, feature_channels)
        self.beta_proj  = nn.Linear(text_embed_dim, feature_channels)

    def forward(
        self,
        features: torch.Tensor,    # [B, C, H, W]
        text_embed: torch.Tensor,  # [B, T]
    ) -> torch.Tensor:
        gamma = self.gamma_proj(text_embed)[:, :, None, None]  # [B,C,1,1]
        beta  = self.beta_proj(text_embed)[:, :, None, None]
        return gamma * features + beta


# ══════════════════════════════════════════════════════
#  TextConditionedDecoder  (used inside both model variants)
# ══════════════════════════════════════════════════════

class TextConditionedDecoder(nn.Module):
    """
    Applies FiLM to visual features using a text embedding,
    then returns modulated features (same shape as input).

    The calling model is responsible for the final conv projection.
    """

    def __init__(self, feature_channels: int, text_embed_dim: int):
        super().__init__()
        self.film = FiLM(text_embed_dim, feature_channels)
        self.norm = nn.GroupNorm(min(32, feature_channels), feature_channels)

    def forward(
        self,
        features: torch.Tensor,    # [B, C, H, W]
        text_embed: torch.Tensor,  # [B, T]
    ) -> torch.Tensor:
        return F.relu(self.norm(self.film(features, text_embed)), inplace=True)
