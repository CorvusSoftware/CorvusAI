"""Compare completed BPE training runs from their local JSONL metrics."""

from __future__ import annotations

import argparse
import json
import math
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
            raise TypeError(f"Expected metrics object at line {number}")
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
    validation_records = [record for record in records if "validation_loss" in record]
    for record in validation_records:
        loss = record["validation_loss"]
        if type(loss) not in (int, float) or not math.isfinite(loss) or loss < 0:
            raise ValueError(f"Invalid validation_loss at step {record['step']}")
        perplexity = record.get("validation_perplexity")
        if (perplexity is not None and
                (type(perplexity) not in (int, float)
                 or not math.isfinite(perplexity) or perplexity < 1)):
            raise ValueError(f"Invalid validation_perplexity at step {record['step']}")
    if validation_records:
        best = min(validation_records, key=lambda record: record["validation_loss"])
        latest = validation_records[-1]
        result["validation_evaluations"] = len(validation_records)
        result["best_validation_step"] = best["step"]
        result["best_validation_loss"] = best["validation_loss"]
        result["last_validation_step"] = latest["step"]
        result["last_validation_loss"] = latest["validation_loss"]
        if "validation_perplexity" in best:
            result["best_validation_perplexity"] = best["validation_perplexity"]
    if all("peak_vram_mib" in row for row in measured):
        result["max_peak_vram_mib"] = max(row["peak_vram_mib"] for row in measured)
    if all("model_parameters" in row for row in measured):
        result["model_parameters"] = measured[-1]["model_parameters"]
    for key in ("device", "architecture", "context_length", "batch_size",
                "gradient_accumulation", "mixed_precision", "corpus_sha256",
                "tokenizer_sha256", "validation_sha256"):
        values = [record.get(key) for record in records]
        if all(value is not None and value == values[0] for value in values):
            result[key] = values[0]
    return result


def compatibility_warnings(summaries: list[dict[str, object]]) -> list[str]:
    """Report known confounders; unknown metadata cannot establish comparability."""
    if len(summaries) < 2:
        return []
    warnings = []
    for key in ("device", "context_length", "batch_size",
                "gradient_accumulation", "mixed_precision", "corpus_sha256",
                "tokenizer_sha256", "validation_sha256"):
        values = [summary.get(key) for summary in summaries]
        if any(value is None for value in values):
            warnings.append(f"Cannot verify comparable {key}: missing metadata")
        elif len(set(values)) != 1:
            warnings.append(f"Runs differ in {key}")
    warnings.append("Hardware model and software environment are not verified")
    return warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize local BPE training runs")
    parser.add_argument("runs", type=Path, nargs="+", help="Run folders or metrics.jsonl files")
    parser.add_argument("--warmup", type=int, default=0,
                        help="Skip this many initial recorded steps in each run")
    args = parser.parse_args()
    summaries = [summarize(run, warmup=args.warmup) for run in args.runs]
    print(json.dumps({"runs": summaries, "warnings": compatibility_warnings(summaries)},
                     indent=2))


if __name__ == "__main__":
    main()
