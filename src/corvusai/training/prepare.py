"""Prepare deterministic, exact-deduplicated train/validation manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def prepare(directory: Path, output: Path, validation_fraction: float = 0.2) -> dict:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")
    paths = sorted(p for p in directory.rglob("*.py") if p.is_file())
    seen: set[str] = set()
    records: list[dict[str, str | int]] = []
    for path in paths:
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest in seen or not content:
            continue
        seen.add(digest)
        records.append({"path": path.relative_to(directory).as_posix(),
                        "sha256": digest, "bytes": len(content)})
    if len(records) < 2:
        raise ValueError("At least two unique nonempty Python files are required")
    records.sort(key=lambda record: hashlib.sha256(str(record["path"]).encode()).hexdigest())
    validation_count = max(1, min(len(records) - 1, round(len(records) * validation_fraction)))
    manifest = {
        "format": "corvus-dataset-manifest-v1",
        "root": str(directory.resolve()),
        "train": records[validation_count:],
        "validation": records[:validation_count],
        "notes": "Exact-content deduplication only; review licenses and near-duplicates manually.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a CorvusAI dataset manifest")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()
    manifest = prepare(args.input, args.output, args.validation_fraction)
    print(json.dumps({"train_files": len(manifest["train"]),
                      "validation_files": len(manifest["validation"])}))


if __name__ == "__main__":
    main()
