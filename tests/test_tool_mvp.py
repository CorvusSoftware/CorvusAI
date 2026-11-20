"""Regression tests for the experimental CorvusAI tool entry points."""
import hashlib
import json

import pytest

from corvusai.agent import explore, preview_patch
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


def test_manifest_audit_detects_modified_and_missing_sources(tmp_path):
    (tmp_path / "source.py").write_text("original", encoding="utf-8")
    digest = hashlib.sha256(b"original").hexdigest()
    manifest = tmp_path / "manifest.json"
    record = {"path": "source.py", "sha256": digest}
    manifest.write_text(json.dumps({"root": str(tmp_path), "train": [record],
                                    "validation": []}), encoding="utf-8")
    assert audit(manifest)["passed"]
    (tmp_path / "source.py").write_text("changed", encoding="utf-8")
    assert audit(manifest)["changed_files"] == ["source.py"]
    (tmp_path / "source.py").unlink()
    assert audit(manifest)["missing_files"] == ["source.py"]


def test_manifest_audit_rejects_traversal_and_symlink(tmp_path):
    (tmp_path / "source.py").write_text("x", encoding="utf-8")
    (tmp_path / "alias.py").symlink_to(tmp_path / "source.py")
    manifest = tmp_path / "manifest.json"
    records = [{"path": "../outside.py", "sha256": "invalid"},
               {"path": "alias.py", "sha256": "invalid"}]
    manifest.write_text(json.dumps({"root": str(tmp_path), "train": records,
                                    "validation": []}), encoding="utf-8")
    report = audit(manifest)
    assert not report["passed"]
    assert report["unsafe_paths"] == ["../outside.py", "alias.py"]


def test_patch_preview_does_not_write(tmp_path):
    source = tmp_path / "sample.py"
    source.write_text("x = 1\n", encoding="utf-8")
    diff = preview_patch(tmp_path, "sample.py", "x = 2\n")
    assert "-x = 1" in diff
    assert "+x = 2" in diff
    assert source.read_text(encoding="utf-8") == "x = 1\n"


def test_patch_preview_rejects_traversal(tmp_path):
    with pytest.raises(ValueError):
        preview_patch(tmp_path, "../outside.py", "bad")
