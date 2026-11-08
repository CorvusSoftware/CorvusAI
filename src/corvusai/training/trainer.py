"""Reproducible tiny pretraining loop with atomic checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch

from corvusai.models.transformer import CorvusTransformer, ModelConfig
from corvusai.tokenization.byte_tokenizer import encode


def train(
    corpus: Path,
    output: Path,
    *,
    steps: int = 20,
    batch_size: int = 4,
    context_length: int = 64,
    seed: int = 42,
    resume: bool = False,
    device: str | None = None,
) -> list[float]:
    if steps < 1 or batch_size < 1 or context_length < 2:
        raise ValueError("steps and batch_size must be positive; context_length >= 2")
    random.seed(seed)
    torch.manual_seed(seed)
    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    raw = corpus.read_text(encoding="utf-8")
    tokens = torch.tensor(encode(raw), dtype=torch.long)
    if len(tokens) < context_length + 1:
        raise ValueError("Corpus must contain at least context_length + 1 UTF-8 bytes")
    config = ModelConfig(context_length=context_length)
    model = CorvusTransformer(config).to(target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    completed = 0
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "latest.pt"
    if resume:
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        # Checkpoints contain Python pickled metadata: load only trusted local files.
        state = torch.load(checkpoint, map_location=target, weights_only=True)
        if state["config"] != vars(config):
            raise ValueError("Checkpoint model configuration mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        completed = int(state["step"])
        torch.set_rng_state(state["cpu_rng"])
        if target.type == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(state["cuda_rng"], device=target)
    losses: list[float] = []
    model.train()
    for step in range(completed, steps):
        starts = torch.randint(0, len(tokens) - context_length, (batch_size,))
        batch = torch.stack([tokens[i : i + context_length + 1] for i in starts.tolist()])
        batch = batch.to(target)
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(batch)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at step {step + 1}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        value = float(loss.detach().cpu())
        losses.append(value)
        print(json.dumps({"step": step + 1, "loss": round(value, 5), "device": str(target)}))
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step + 1,
            "config": vars(config),
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(target) if target.type == "cuda" else None,
        }
        temporary = output / "latest.tmp"
        torch.save(state, temporary)
        os.replace(temporary, checkpoint)
    return losses


def main() -> None:
    parser = argparse.ArgumentParser(description="CorvusAI experimental pretraining")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/tiny"))
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    args = parser.parse_args()
    started = time.monotonic()
    train(args.corpus, args.output, steps=args.steps, batch_size=args.batch_size,
          context_length=args.context_length, seed=args.seed, resume=args.resume,
          device=args.device)
    print(f"Elapsed: {time.monotonic() - started:.1f}s")


if __name__ == "__main__":
    main()
