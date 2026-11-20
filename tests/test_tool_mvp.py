"""Regression tests for the experimental CorvusAI tool entry points."""
import hashlib
import json

import pytest

from corvusai.agent import explore
from corvusai.training.audit import audit


def test_agent_search(tmp_path):
    (tmp_path / "sample.py").write_text("def answer():\n    return 42\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("return 42", encoding="utf-8")
    matches = explore(tmp_path, "return")
    assert matches == [{"path": "sample.py", "line": 2, "text": "    return 42"}]
    assert explore(tmp_path, "missing") == []


def test_agent_rejects_invalid_query(tmp_path):
    with pytest.raises(ValueError):
        explore(tmp_path, "")
    with pytest.raises(ValueError):
        explore(tmp_path, "test", limit=0)


def test_manifest_audit(tmp_path):
    (tmp_path / "a.py").write_text("a = 1", encoding="utf-8")
    (tmp_path / "b.py").write_text("b = 2", encoding="utf-8")
    records = [{"path": name, "sha256": hashlib.sha256(
        (tmp_path / name).read_bytes()).hexdigest()} for name in ("a.py", "b.py")]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"root": str(tmp_path), "train": [records[0]],
                                    "validation": [records[1]]}), encoding="utf-8")
    assert audit(manifest)["passed"]
    manifest.write_text(json.dumps({"root": str(tmp_path), "train": [records[0]],
                                    "validation": [records[0]]}), encoding="utf-8")
    assert not audit(manifest)["passed"]
