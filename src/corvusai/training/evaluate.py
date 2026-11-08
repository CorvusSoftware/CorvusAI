"""Held-out next-byte loss evaluation."""

from __future__ import annotations

import math

import torch

from corvusai.training.dataset import sample_batch


@torch.no_grad()
def evaluate(model: torch.nn.Module, data: torch.Tensor, *, batches: int, batch_size: int,
             context_length: int, device: torch.device) -> dict[str, float]:
    if batches < 1:
        raise ValueError("batches must be >= 1")
    was_training = model.training
    model.eval()
    try:
        total = 0.0
        for _ in range(batches):
            batch = sample_batch(data, batch_size, context_length, device)
            total += float(model.loss(batch))
        loss = total / batches
        return {"validation_loss": loss, "perplexity": math.exp(loss)}
    finally:
        model.train(was_training)
