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
from corvusai.training.dataset import corpus_digest, load_corpus, sample_batch
from corvusai.training.evaluate import evaluate


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
    validation_corpus: Path | None = None,
    eval_every: int = 10,
    mixed_precision: bool = False,
) -> list[float]:
    if steps < 1 or batch_size < 1 or context_length < 2:
        raise ValueError("steps and batch_size must be positive; context_length >= 2")
    if eval_every < 1:
        raise ValueError("eval_every must be >= 1")
    random.seed(seed)
    torch.manual_seed(seed)
    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokens = load_corpus(corpus, context_length)
    validation = load_corpus(validation_corpus, context_length) if validation_corpus else None
    if validation_corpus and corpus.resolve() == validation_corpus.resolve():
        raise ValueError("Training and validation corpus must be different files")
    use_amp = mixed_precision and target.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
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
        if state.get("corpus_sha256") not in (None, corpus_digest(corpus)):
            raise ValueError("Training corpus changed since checkpoint")
        if state.get("validation_sha256") != (corpus_digest(validation_corpus) if validation_corpus else None):
            raise ValueError("Validation corpus changed since checkpoint")
        if state["config"] != vars(config):
            raise ValueError("Checkpoint model configuration mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        completed = int(state["step"])
        if use_amp and state.get("scaler") is not None:
            scaler.load_state_dict(state["scaler"])
        torch.set_rng_state(state["cpu_rng"])
        if target.type == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(state["cuda_rng"], device=target)
    losses: list[float] = []
    model.train()
    for step in range(completed, steps):
        batch = sample_batch(tokens, batch_size, context_length, target)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=target.type, dtype=torch.float16, enabled=use_amp):
            loss = model.loss(batch)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at step {step + 1}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        value = float(loss.detach().cpu())
        losses.append(value)
        metrics = {"step": step + 1, "train_loss": round(value, 5), "device": str(target)}
        if validation is not None and ((step + 1) % eval_every == 0 or step + 1 == steps):
            metrics.update(evaluate(model, validation, batches=2, batch_size=batch_size,
                                    context_length=context_length, device=target))
        if target.type == "cuda":
            metrics["peak_vram_mib"] = round(torch.cuda.max_memory_allocated(target) / 2**20, 1)
        print(json.dumps(metrics), flush=True)
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step + 1,
            "config": vars(config),
            "corpus_sha256": corpus_digest(corpus),
            "validation_sha256": corpus_digest(validation_corpus) if validation_corpus else None,
            "scaler": scaler.state_dict() if use_amp else None,
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
    parser.add_argument("--validation-corpus", type=Path)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    args = parser.parse_args()
    started = time.monotonic()
    train(args.corpus, args.output, steps=args.steps, batch_size=args.batch_size,
          context_length=args.context_length, seed=args.seed, resume=args.resume,
          device=args.device, validation_corpus=args.validation_corpus,
          eval_every=args.eval_every, mixed_precision=args.mixed_precision)
    print(f"Elapsed: {time.monotonic() - started:.1f}s")


if __name__ == "__main__":
    main()
