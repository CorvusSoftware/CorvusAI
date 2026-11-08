"""Deterministic file-level corpus splits to avoid overlapping training windows."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch


def load_corpus(path: Path, context_length: int) -> torch.Tensor:
    if not path.is_file():
        raise ValueError(f"Corpus is not a file: {path}")
    data = path.read_bytes()
    if len(data) < context_length + 1:
        raise ValueError(f"Corpus too short: {path} (need {context_length + 1} bytes)")
    return torch.tensor(list(data), dtype=torch.long)


def split_files(directory: Path, validation_fraction: float = 0.2) -> tuple[list[Path], list[Path]]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")
    paths = sorted(p for p in directory.rglob("*.py") if p.is_file())
    if len(paths) < 2:
        raise ValueError("At least two Python files are needed for file-level splitting")
    # Stable across platforms, Python versions and discovery order.
    paths.sort(key=lambda p: hashlib.sha256(p.relative_to(directory).as_posix().encode()).hexdigest())
    count = max(1, min(len(paths) - 1, round(len(paths) * validation_fraction)))
    return paths[count:], paths[:count]


def sample_batch(data: torch.Tensor, batch_size: int, context_length: int, device: torch.device) -> torch.Tensor:
    starts = torch.randint(0, len(data) - context_length, (batch_size,))
    return torch.stack([data[i : i + context_length + 1] for i in starts.tolist()]).to(device)


def corpus_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
