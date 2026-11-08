"""Opt-in modern decoder architecture; legacy checkpoints remain the default."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModernConfig:
    vocab_size: int = 256
    context_length: int = 128
    width: int = 128
    heads: int = 4
    layers: int = 2
    kv_heads: int = 2
    rope_base: float = 10000.0

    def __post_init__(self) -> None:
        if min(self.vocab_size, self.context_length, self.width, self.heads,
               self.layers, self.kv_heads) <= 0:
            raise ValueError("Model dimensions must be positive")
        if self.width % self.heads or self.heads % self.kv_heads:
            raise ValueError("width must divide heads and heads must divide kv_heads")
        if (self.width // self.heads) % 2:
            raise ValueError("RoPE requires an even head dimension")
        if self.rope_base <= 0:
            raise ValueError("rope_base must be positive")


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


def _rotate(x: torch.Tensor, positions: torch.Tensor, base: float) -> torch.Tensor:
    dimension = x.shape[-1]
    inverse = base ** (-torch.arange(0, dimension, 2, device=x.device,
                                    dtype=torch.float32) / dimension)
    angles = positions.to(torch.float32)[:, None] * inverse[None, :]
    cosine = angles.cos().to(x.dtype)[None, None, :, :]
    sine = angles.sin().to(x.dtype)[None, None, :, :]
    even, odd = x[..., ::2], x[..., 1::2]
    return torch.stack((even * cosine - odd * sine,
                        even * sine + odd * cosine), dim=-1).flatten(-2)


class ModernBlock(nn.Module):
    def __init__(self, config: ModernConfig) -> None:
        super().__init__()
        self.heads = config.heads
        self.kv_heads = config.kv_heads
        self.head_dim = config.width // config.heads
        self.rope_base = config.rope_base
        self.norm1 = RMSNorm(config.width)
        self.q = nn.Linear(config.width, config.heads * self.head_dim, bias=False)
        self.k = nn.Linear(config.width, config.kv_heads * self.head_dim, bias=False)
        self.v = nn.Linear(config.width, config.kv_heads * self.head_dim, bias=False)
        self.o = nn.Linear(config.width, config.width, bias=False)
        self.norm2 = RMSNorm(config.width)
        self.gate = nn.Linear(config.width, 4 * config.width, bias=False)
        self.up = nn.Linear(config.width, 4 * config.width, bias=False)
        self.down = nn.Linear(4 * config.width, config.width, bias=False)

    def forward(
        self, x: torch.Tensor, *, positions: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch, length, _ = x.shape
        normalized = self.norm1(x)
        q = self.q(normalized).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(normalized).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v(normalized).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        q = _rotate(q, positions, self.rope_base)
        k = _rotate(k, positions, self.rope_base)
        if cache is not None:
            k = torch.cat((cache[0], k), dim=2)
            v = torch.cat((cache[1], v), dim=2)
        next_cache = (k, v)
        if self.kv_heads != self.heads:
            repeat = self.heads // self.kv_heads
            attention_k = k.repeat_interleave(repeat, dim=1)
            attention_v = v.repeat_interleave(repeat, dim=1)
        else:
            attention_k, attention_v = k, v
        if cache is None:
            attention = F.scaled_dot_product_attention(q, attention_k, attention_v,
                                                        is_causal=True)
        else:
            past = k.shape[2] - length
            mask = torch.arange(k.shape[2], device=x.device)[None, :] <= (
                past + torch.arange(length, device=x.device)[:, None]
            )
            attention = F.scaled_dot_product_attention(
                q, attention_k, attention_v, attn_mask=mask,
            )
        attention = attention.transpose(1, 2).contiguous().view(batch, length, -1)
        x = x + self.o(attention)
        y = self.norm2(x)
        return x + self.down(F.silu(self.gate(y)) * self.up(y)), next_cache


class ModernCorvusTransformer(nn.Module):
    def __init__(self, config: ModernConfig | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else ModernConfig()
        self.tokens = nn.Embedding(self.config.vocab_size, self.config.width)
        self.blocks = nn.ModuleList(
            [ModernBlock(self.config) for _ in range(self.config.layers)]
        )
        self.norm = RMSNorm(self.config.width)
        self.head = nn.Linear(self.config.width, self.config.vocab_size, bias=False)

    def forward(
        self, tokens: torch.Tensor, *,
        cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        if tokens.ndim != 2 or tokens.shape[1] < 1:
            raise ValueError("Expected nonempty [batch, sequence] tokens")
        if cache is not None and len(cache) != len(self.blocks):
            raise ValueError("Cache layer count mismatch")
        past = cache[0][0].shape[2] if cache is not None else 0
        if past + tokens.shape[1] > self.config.context_length:
            raise ValueError("Sequence exceeds context length")
        positions = torch.arange(past, past + tokens.shape[1], device=tokens.device)
        x = self.tokens(tokens)
        new_cache = []
        for index, block in enumerate(self.blocks):
            x, layer_cache = block(
                x, positions=positions, cache=cache[index] if cache is not None else None,
            )
            if use_cache:
                new_cache.append(layer_cache)
        logits = self.head(self.norm(x))
        return (logits, new_cache) if use_cache else logits

    def loss(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] < 2:
            raise ValueError("At least two tokens are needed for next-token training")
        logits = self(tokens[:, :-1])
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1))
