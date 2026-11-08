"""Regression tests for safe, reproducible code evaluation."""

import json
from pathlib import Path

import pytest

from corvusai.evaluation import code


def _benchmark(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / "benchmark.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records),
                    encoding="utf-8")
    return path


def test_code_evaluation_metrics_and_reproducibility(tmp_path: Path, monkeypatch):
    benchmark = _benchmark(tmp_path, [
        {"id": "a", "prompt": "x = ", "reference": "1\n"},
        {"id": "b", "prompt": "def f():\n    ", "reference": "return 2\n"},
        {"id": "c", "prompt": "y = ", "reference": "3\n"},
    ])
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"dummy checkpoint")
    calls = []
    def fake_generate(_checkpoint, prompt, **kwargs):
        calls.append((prompt, kwargs["seed"]))
        return prompt + {"x = ": "1\n", "def f():\n    ": "return 2\n",
                         "y = ": "?"}[prompt]
    monkeypatch.setattr(code, "generate", fake_generate)
    output = tmp_path / "report.json"
    report = code.evaluate_code(benchmark, checkpoint, output, max_new_tokens=12, seed=9)
    assert calls == [("x = ", 9), ("def f():\n    ", 10), ("y = ", 11)]
    assert report["syntax_valid_rate"] == pytest.approx(2 / 3)
    assert report["exact_match_rate"] == pytest.approx(2 / 3)
    assert report["reference_tasks"] == 3
    assert report == code.evaluate_code(benchmark, checkpoint, output,
                                        max_new_tokens=12, seed=9)
    assert json.loads(output.read_text()) == report


def test_optional_references(tmp_path: Path, monkeypatch):
    benchmark = _benchmark(tmp_path, [{"id": "a", "prompt": "x = "}])
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(code, "generate", lambda *_args, **_kwargs: "x = 1")
    report = code.evaluate_code(benchmark, checkpoint, tmp_path / "out.json")
    assert report["exact_match_rate"] is None
    assert report["reference_tasks"] == 0


@pytest.mark.parametrize("records", [
    [],
    [{"id": "a", "prompt": ""}],
    [{"id": "a", "prompt": "x"}, {"id": "a", "prompt": "y"}],
    [{"id": "a", "prompt": "x", "reference": 5}],
])
def test_invalid_benchmarks(tmp_path: Path, records: list[dict]):
    benchmark = _benchmark(tmp_path, records)
    with pytest.raises((ValueError, TypeError)):
        code.evaluate_code(benchmark, tmp_path / "checkpoint.pt", tmp_path / "out.json")


def test_rejects_non_prefix_generation(tmp_path: Path, monkeypatch):
    benchmark = _benchmark(tmp_path, [{"id": "a", "prompt": "x = "}])
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(code, "generate", lambda *_args, **_kwargs: "wrong")
    with pytest.raises(ValueError, match="prompt"):
        code.evaluate_code(benchmark, checkpoint, tmp_path / "out.json")
