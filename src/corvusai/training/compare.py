"""Compare completed BPE training runs from their local JSONL metrics."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def summarize(path: Path, *, warmup: int = 0) -> dict[str, object]:
    """Summarize a run without assuming training quality or identical hardware."""
    if warmup < 0:
        raise ValueError("warmup must be nonnegative")
    metrics_path = path / "metrics.jsonl" if path.is_dir() else path
    records = []
    for number, line in enumerate(metrics_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON at line {number}") from error
        if not isinstance(entry, dict):
            raise ValueError(f"Expected metrics object at line {number}")
        records.append(entry)
    if not records:
        raise ValueError(f"No metrics in {metrics_path}")
    steps = [record.get("step") for record in records]
    if any(type(step) is not int or step < 1 for step in steps):
        raise ValueError("Metrics steps must be positive integers")
    if steps != sorted(set(steps)):
        raise ValueError("Metrics steps must be strictly increasing")
    measured = records[warmup:]
    if not measured:
        raise ValueError("No measured steps after warmup")
    for record in measured:
        for key in ("train_loss", "step_seconds", "train_tokens_per_second"):
            value = record.get(key)
            if type(value) not in (int, float) or not (float("-inf") < value < float("inf")):
                raise ValueError(f"Invalid {key} at step {record['step']}")
        if record["step_seconds"] <= 0 or record["train_tokens_per_second"] <= 0:
            raise ValueError("Timing and throughput must be positive")
    result: dict[str, object] = {
        "run": str(metrics_path.parent),
        "steps_recorded": len(records),
        "steps_measured": len(measured),
        "last_step": steps[-1],
        "last_train_loss": records[-1].get("train_loss"),
        "mean_train_loss": statistics.mean(row["train_loss"] for row in measured),
        "median_step_seconds": statistics.median(row["step_seconds"] for row in measured),
        "median_train_tokens_per_second": statistics.median(
            row["train_tokens_per_second"] for row in measured
        ),
    }
    if all("peak_vram_mib" in row for row in measured):
        result["max_peak_vram_mib"] = max(row["peak_vram_mib"] for row in measured)
    if all("model_parameters" in row for row in measured):
        result["model_parameters"] = measured[-1]["model_parameters"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize local BPE training runs")
    parser.add_argument("runs", type=Path, nargs="+", help="Run folders or metrics.jsonl files")
    parser.add_argument("--warmup", type=int, default=0,
                        help="Skip this many initial recorded steps in each run")
    args = parser.parse_args()
    print(json.dumps([summarize(run, warmup=args.warmup) for run in args.runs], indent=2))


if __name__ == "__main__":
    main()
