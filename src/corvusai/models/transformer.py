"""Minimal decoder-only Transformer with randomly initialized parameters."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 256
    context_length: int = 128
    width: int = 128
    heads: int = 4
    layers: int = 2

    def __post_init__(self) -> None:
        if min(self.vocab_size, self.context_length, self.width, self.heads, self.layers) <= 0:
            raise ValueError("Model dimensions must be positive")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.width)
        self.attn = nn.MultiheadAttention(config.width, config.heads, batch_first=True)
        self.norm2 = nn.LayerNorm(config.width)
        self.ff = nn.Sequential(
            nn.Linear(config.width, 4 * config.width),
            nn.GELU(),
            nn.Linear(4 * config.width, config.width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[1]
        mask = torch.ones(length, length, device=x.device, dtype=torch.bool).triu(1)
        normalized = self.norm1(x)
        attention, _ = self.attn(normalized, normalized, normalized, attn_mask=mask, need_weights=False)
        x = x + attention
        return x + self.ff(self.norm2(x))


class CorvusTransformer(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        super().__init__()
        self.config = config
        self.tokens = nn.Embedding(config.vocab_size, config.width)
        self.positions = nn.Embedding(config.context_length, config.width)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.layers)])
        self.norm = nn.LayerNorm(config.width)
        self.head = nn.Linear(config.width, config.vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.shape[1] == 0:
            raise ValueError("Expected nonempty token tensor of shape [batch, sequence]")
        if tokens.shape[1] > self.config.context_length:
            raise ValueError("Sequence exceeds context length")
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.tokens(tokens) + self.positions(positions)[None, :, :]
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))

    def loss(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] < 2:
            raise ValueError("At least two tokens are needed for next-token training")
        logits = self(tokens[:, :-1])
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1))
