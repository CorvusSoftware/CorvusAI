"""Reproducible multi-file BPE pretraining with resumable checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path

import torch

from corvusai.models.transformer import CorvusTransformer, ModelConfig
from corvusai.tokenization.bpe import ByteBPE


def _load_documents(corpus: Path, tokenizer: ByteBPE, context_length: int) -> tuple[list[torch.Tensor], str]:
    paths = sorted(corpus.rglob("*.py")) if corpus.is_dir() else [corpus]
    if not paths or any(not path.is_file() for path in paths):
        raise ValueError("Corpus must be a file or a directory containing Python files")
    digest = hashlib.sha256()
    documents: list[torch.Tensor] = []
    for path in paths:
        data = path.read_bytes()
        relative = path.relative_to(corpus) if corpus.is_dir() else Path(path.name)
        name = relative.as_posix().encode("utf-8")
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
        ids = tokenizer.encode_bytes(data)
        if len(ids) > context_length:
            documents.append(torch.tensor(ids, dtype=torch.long))
    if not documents:
        raise ValueError("Corpus has too few BPE tokens for context length")
    return documents, digest.hexdigest()


def _sample(documents: list[torch.Tensor], batch_size: int, context_length: int,
            target: torch.device) -> torch.Tensor:
    # Choose a valid window uniformly, without crossing document boundaries.
    counts = [len(doc) - context_length for doc in documents]
    cumulative = torch.tensor(counts).cumsum(0)
    draws = torch.randint(0, int(cumulative[-1]), (batch_size,))
    rows = []
    for draw in draws.tolist():
        doc_index = int(torch.searchsorted(cumulative, draw, right=True))
        previous = int(cumulative[doc_index - 1]) if doc_index else 0
        start = draw - previous
        rows.append(documents[doc_index][start:start + context_length + 1])
    return torch.stack(rows).to(target)


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
    resume: bool = False,
    mixed_precision: bool = False,
    gradient_accumulation: int = 1,
    warmup_steps: int = 0,
    decay_steps: int = 0,
) -> list[float]:
    if steps < 1 or context_length < 2 or batch_size < 1 or gradient_accumulation < 1:
        raise ValueError("Invalid training dimensions or gradient accumulation")
    if warmup_steps < 0 or decay_steps < warmup_steps:
        raise ValueError("Invalid learning rate schedule")
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    tokenizer_bytes = tokenizer_path.read_bytes()
    tokenizer = ByteBPE.load(tokenizer_path)
    documents, corpus_sha256 = _load_documents(corpus, tokenizer, context_length)
    tokenizer_sha256 = hashlib.sha256(tokenizer_bytes).hexdigest()
    random.seed(seed)
    torch.manual_seed(seed)
    if target.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    config = ModelConfig(vocab_size=tokenizer.vocab_size, context_length=context_length)
    model = CorvusTransformer(config).to(target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    def lr_multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        if not decay_steps:
            return 1.0
        progress = min(1.0, max(0.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps)))
        return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    use_amp = mixed_precision and target.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "latest.pt"
    completed = 0
    if resume:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        # Only resume from checkpoints produced locally and trusted by the operator.
        state = torch.load(checkpoint, map_location=target, weights_only=True)
        if state.get("format") != "corvus-bpe-checkpoint-v2":
            raise ValueError("Resume requires a V0.6 BPE checkpoint")
        expected = {
            "config": vars(config),
            "tokenizer_sha256": tokenizer_sha256,
            "corpus_sha256": corpus_sha256,
            "batch_size": batch_size,
            "gradient_accumulation": gradient_accumulation,
            "mixed_precision": use_amp,
            "warmup_steps": warmup_steps,
            "decay_steps": decay_steps,
        }
        if ("warmup_steps" not in state and "decay_steps" not in state
                and warmup_steps == 0 and decay_steps == 0):
            expected = {key: value for key, value in expected.items()
                        if key not in ("warmup_steps", "decay_steps")}
        if any(state.get(key) != value for key, value in expected.items()):
            raise ValueError("Checkpoint training configuration, corpus or tokenizer mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        if "scheduler" in state:
            scheduler.load_state_dict(state["scheduler"])
        elif warmup_steps or decay_steps:
            raise ValueError("Cannot resume scheduled training from a legacy checkpoint")
        else:
            scheduler.last_epoch = int(state["step"])
            scheduler._step_count = int(state["step"]) + 1
        completed = int(state["step"])
        torch.set_rng_state(state["cpu_rng"].cpu())
        if target.type == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(state["cuda_rng"].cpu(), device=target)
    losses: list[float] = []
    model.train()
    for step in range(completed, steps):
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0.0
        for _ in range(gradient_accumulation):
            batch = _sample(documents, batch_size, context_length, target)
            with torch.autocast(device_type=target.type, dtype=torch.float16, enabled=use_amp):
                loss = model.loss(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at step {step + 1}")
            accumulated += float(loss.detach().cpu()) / gradient_accumulation
            scaler.scale(loss / gradient_accumulation).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        losses.append(accumulated)
        state = {
            "format": "corvus-bpe-checkpoint-v2",
            "config": vars(config),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step + 1,
            "tokenizer_sha256": tokenizer_sha256,
            "corpus_sha256": corpus_sha256,
            "batch_size": batch_size,
            "gradient_accumulation": gradient_accumulation,
            "mixed_precision": use_amp,
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(target) if target.type == "cuda" else None,
        }
        temporary = output / "latest.tmp"
        torch.save(state, temporary)
        os.replace(temporary, checkpoint)
        metrics = {"step": step + 1, "train_loss": accumulated, "device": str(target),
                   "learning_rate": optimizer.param_groups[0]["lr"]}
        if target.type == "cuda":
            metrics["peak_vram_mib"] = round(torch.cuda.max_memory_allocated(target) / 2**20, 1)
        print(json.dumps(metrics), flush=True)
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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--decay-steps", type=int, default=0)
    args = parser.parse_args()
    train_bpe(args.corpus, args.tokenizer, args.output, steps=args.steps,
              context_length=args.context_length, batch_size=args.batch_size,
              seed=args.seed, device=args.device, resume=args.resume,
              mixed_precision=args.mixed_precision,
              gradient_accumulation=args.gradient_accumulation,
              warmup_steps=args.warmup_steps, decay_steps=args.decay_steps)


if __name__ == "__main__":
    main()
