"""Measure local model inference throughput; no training or external calls."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from corvusai.inference.brain import BrainSession


def profile(checkpoint: Path, *, tokenizer: Path | None = None,
            device: str = "cpu", prompt: str = "def add(a, b):",
            tokens: int = 32) -> dict[str, object]:
    if tokens < 1:
        raise ValueError("tokens must be positive")
    start = time.perf_counter()
    session = BrainSession(checkpoint, tokenizer_path=tokenizer, device=device)
    load_seconds = time.perf_counter() - start
    if device == "cuda":
        import torch
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    session.complete(prompt, max_new_tokens=tokens)
    if device == "cuda":
        import torch
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    result: dict[str, object] = {
        "device": device, "requested_new_tokens": tokens,
        "load_seconds": load_seconds, "generation_seconds": seconds,
        "requested_tokens_per_second": tokens / seconds,
        "note": "Requested token throughput, not a quality benchmark.",
    }
    if device == "cuda":
        result["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile trusted CorvusBrain inference")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--tokens", type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(profile(args.checkpoint, tokenizer=args.tokenizer,
                             device=args.device, tokens=args.tokens), indent=2))


if __name__ == "__main__":
    main()
