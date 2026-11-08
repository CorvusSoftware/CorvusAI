"""Train a tiny model on BPE IDs with a tokenizer-bound checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from corvusai.models.transformer import CorvusTransformer, ModelConfig
from corvusai.tokenization.bpe import ByteBPE
from corvusai.training.dataset import sample_batch


def train_bpe(
    corpus: Path,
    tokenizer_path: Path,
    output: Path,
    *,
    steps: int = 20,
    context_length: int = 64,
    batch_size: int = 4,
    seed: int = 42,
    device: str = "cpu",
) -> list[float]:
    if steps < 1 or context_length < 2 or batch_size < 1:
        raise ValueError("Invalid training dimensions")
    tokenizer_bytes = tokenizer_path.read_bytes()
    tokenizer = ByteBPE.load(tokenizer_path)
    corpus_bytes = corpus.read_bytes()
    ids = tokenizer.encode_bytes(corpus_bytes)
    if len(ids) <= context_length:
        raise ValueError("Corpus has too few BPE tokens for context length")
    torch.manual_seed(seed)
    target = torch.device(device)
    config = ModelConfig(vocab_size=tokenizer.vocab_size, context_length=context_length)
    model = CorvusTransformer(config).to(target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    data = torch.tensor(ids, dtype=torch.long)
    output.mkdir(parents=True, exist_ok=True)
    losses: list[float] = []
    for step in range(steps):
        batch = sample_batch(data, batch_size, context_length, target)
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(batch)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at step {step + 1}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        state = {
            "format": "corvus-bpe-checkpoint-v1",
            "config": vars(config),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step + 1,
            "tokenizer_sha256": hashlib.sha256(tokenizer_bytes).hexdigest(),
            "corpus_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
        }
        temporary = output / "latest.tmp"
        torch.save(state, temporary)
        os.replace(temporary, output / "latest.pt")
        print(json.dumps({"step": step + 1, "train_loss": losses[-1]}), flush=True)
    return losses


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CorvusAI with a local BPE tokenizer")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/bpe"))
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    train_bpe(args.corpus, args.tokenizer, args.output, steps=args.steps,
              context_length=args.context_length, batch_size=args.batch_size,
              seed=args.seed, device=args.device)


if __name__ == "__main__":
    main()
