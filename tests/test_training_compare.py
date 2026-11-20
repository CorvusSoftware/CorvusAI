"""Regression coverage for comparing saved training telemetry."""

import json
from pathlib import Path

import pytest

from corvusai.training.compare import summarize


def _write_metrics(path: Path, records: list[dict]) -> None:
    path.mkdir()
    (path / "metrics.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8",
    )


def test_compare_summary_and_warmup(tmp_path: Path):
    run = tmp_path / "modern"
    _write_metrics(run, [
        {"step": 1, "train_loss": 3.0, "step_seconds": 4.0,
         "train_tokens_per_second": 2.0, "model_parameters": 120},
        {"step": 2, "train_loss": 2.0, "step_seconds": 2.0,
         "train_tokens_per_second": 4.0, "model_parameters": 120},
        {"step": 3, "train_loss": 1.0, "step_seconds": 1.0,
         "train_tokens_per_second": 8.0, "model_parameters": 120},
    ])
    result = summarize(run, warmup=1)
    assert result["steps_recorded"] == 3
    assert result["steps_measured"] == 2
    assert result["last_step"] == 3
    assert result["last_train_loss"] == 1.0
    assert result["mean_train_loss"] == pytest.approx(1.5)
    assert result["median_train_tokens_per_second"] == pytest.approx(6.0)
    assert result["model_parameters"] == 120


@pytest.mark.parametrize("records", [
    [],
    [{"step": 1, "train_loss": 1, "step_seconds": 0,
      "train_tokens_per_second": 1}],
    [{"step": 2, "train_loss": 1, "step_seconds": 1,
      "train_tokens_per_second": 1},
     {"step": 1, "train_loss": 1, "step_seconds": 1,
      "train_tokens_per_second": 1}],
    [{"step": 1, "train_loss": 1, "step_seconds": 1,
      "train_tokens_per_second": float("nan")}],
])
def test_compare_rejects_bad_metrics(tmp_path: Path, records: list[dict]):
    run = tmp_path / "invalid"
    _write_metrics(run, records)
    with pytest.raises(ValueError):
        summarize(run)


def test_compare_rejects_excessive_warmup(tmp_path: Path):
    run = tmp_path / "run"
    _write_metrics(run, [{"step": 1, "train_loss": 1,
                          "step_seconds": 1, "train_tokens_per_second": 1}])
    with pytest.raises(ValueError, match="warmup"):
        summarize(run, warmup=1)
