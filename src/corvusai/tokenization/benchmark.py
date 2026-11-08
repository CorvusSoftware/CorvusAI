"""Measure tokenizer roundtrip and compression, not model coding ability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from corvusai.tokenization.bpe import ByteBPE


def benchmark(tokenizer: ByteBPE, documents: list[bytes]) -> dict[str, float | int]:
    total_bytes = sum(map(len, documents))
    if total_bytes == 0:
        raise ValueError("Documents must contain data")
    total_tokens = 0
    for document in documents:
        ids = tokenizer.encode_bytes(document)
        if tokenizer.decode_bytes(ids) != document:
            raise AssertionError("Tokenizer roundtrip failed")
        total_tokens += len(ids)
    return {"bytes": total_bytes, "tokens": total_tokens,
            "bytes_per_token": total_bytes / total_tokens,
            "reduction_vs_bytes_percent": 100 * (1 - total_tokens / total_bytes)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a CorvusAI tokenizer")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.input.rglob("*.py"))
    if not paths:
        parser.error("No Python files found")
    print(json.dumps(benchmark(ByteBPE.load(args.tokenizer),
                               [path.read_bytes() for path in paths]), indent=2))


if __name__ == "__main__":
    main()
