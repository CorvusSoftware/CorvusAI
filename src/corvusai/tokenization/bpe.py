"""Deterministic byte-pair encoding learned from scratch.

The first 256 token IDs represent individual bytes; learned IDs are appended.
This implementation prioritizes transparency over training speed.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


class ByteBPE:
    def __init__(self, merges: list[tuple[int, int]] | None = None) -> None:
        self.merges = list(merges or [])
        self.pieces: list[bytes] = [bytes([i]) for i in range(256)]
        for left, right in self.merges:
            if not (0 <= left < len(self.pieces) and 0 <= right < len(self.pieces)):
                raise ValueError("Invalid BPE merge reference")
            self.pieces.append(self.pieces[left] + self.pieces[right])
        self.ranks = {pair: 256 + i for i, pair in enumerate(self.merges)}

    @property
    def vocab_size(self) -> int:
        return len(self.pieces)

    def encode_bytes(self, data: bytes) -> list[int]:
        tokens = list(data)
        while len(tokens) > 1:
            candidates = (
                (self.ranks[(a, b)], i)
                for i, (a, b) in enumerate(zip(tokens, tokens[1:]))
                if (a, b) in self.ranks
            )
            match = min(candidates, default=None)
            if match is None:
                break
            rank, index = match
            tokens[index : index + 2] = [rank]
        return tokens

    def encode(self, text: str) -> list[int]:
        return self.encode_bytes(text.encode("utf-8"))

    def decode_bytes(self, tokens: list[int]) -> bytes:
        return b"".join(self.pieces[token] for token in tokens)

    def decode(self, tokens: list[int]) -> str:
        return self.decode_bytes(tokens).decode("utf-8")

    @classmethod
    def train(cls, documents: list[bytes], merges: int = 256) -> "ByteBPE":
        if merges < 0:
            raise ValueError("merges must be nonnegative")
        if not documents or not any(documents):
            raise ValueError("At least one nonempty document is required")
        sequences = [list(data) for data in documents]
        learned: list[tuple[int, int]] = []
        for _ in range(merges):
            counts: Counter[tuple[int, int]] = Counter()
            for sequence in sequences:
                counts.update(zip(sequence, sequence[1:]))
            if not counts:
                break
            # Stable tie-breaking independent of hash iteration order.
            pair, frequency = min(counts.items(), key=lambda item: (-item[1], item[0]))
            if frequency < 2:
                break
            new_id = 256 + len(learned)
            learned.append(pair)
            for i, sequence in enumerate(sequences):
                result: list[int] = []
                cursor = 0
                while cursor < len(sequence):
                    if cursor + 1 < len(sequence) and (sequence[cursor], sequence[cursor + 1]) == pair:
                        result.append(new_id)
                        cursor += 2
                    else:
                        result.append(sequence[cursor])
                        cursor += 1
                sequences[i] = result
        return cls(learned)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"format": "corvus-byte-bpe-v1", "merges": self.merges},
                                   indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ByteBPE":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != "corvus-byte-bpe-v1":
            raise ValueError("Unsupported tokenizer format")
        return cls([tuple(pair) for pair in payload["merges"]])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CorvusAI byte-level BPE from scratch")
    parser.add_argument("--input", type=Path, required=True, help="Directory containing .py files")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--merges", type=int, default=256)
    args = parser.parse_args()
    paths = sorted(args.input.rglob("*.py"))
    if not paths:
        parser.error("No Python files found")
    model = ByteBPE.train([path.read_bytes() for path in paths], merges=args.merges)
    model.save(args.output)
    print(json.dumps({"documents": len(paths), "vocab_size": model.vocab_size,
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
