"""Regression coverage for comparing saved training telemetry."""

import json
from pathlib import Path

import pytest

from corvusai.training.compare import compatibility_warnings, summarize


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


def test_compare_warns_on_incompatible_runs():
    baseline = {
        "device": "cpu", "context_length": 8, "batch_size": 1,
        "gradient_accumulation": 1, "mixed_precision": False,
        "corpus_sha256": "corpus-a", "tokenizer_sha256": "tokenizer-a",
    }
    changed = {**baseline, "context_length": 16, "corpus_sha256": "corpus-b"}
    warnings = compatibility_warnings([baseline, changed])
    assert "Runs differ in context_length" in warnings
    assert "Runs differ in corpus_sha256" in warnings
    assert any("Hardware model" in warning for warning in warnings)


def test_compare_warns_on_missing_metadata():
    warnings = compatibility_warnings([{"device": "cpu"}, {"device": "cpu"}])
    assert any("missing metadata" in warning for warning in warnings)
    assert compatibility_warnings([{"device": "cpu"}]) == []


def test_compare_preserves_consistent_metadata(tmp_path: Path):
    run = tmp_path / "metadata"
    base = {
        "train_loss": 1.0, "step_seconds": 1.0,
        "train_tokens_per_second": 8.0, "device": "cpu",
        "architecture": "modern", "context_length": 8,
        "corpus_sha256": "abc", "tokenizer_sha256": "def",
    }
    _write_metrics(run, [{**base, "step": 1}, {**base, "step": 2}])
    result = summarize(run)
    assert result["architecture"] == "modern"
    assert result["corpus_sha256"] == "abc"
    assert result["context_length"] == 8


def test_compare_rejects_non_object_record(tmp_path: Path):
    run = tmp_path / "non-object"
    run.mkdir()
    (run / "metrics.jsonl").write_text("[1, 2, 3]\n", encoding="utf-8")
    with pytest.raises(TypeError, match="Expected metrics object"):
        summarize(run)
