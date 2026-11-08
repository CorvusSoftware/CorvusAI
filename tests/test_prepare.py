from pathlib import Path

import pytest

from corvusai.training.prepare import prepare


def test_prepare_deduplicates_and_splits(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.py").write_text("a = 1\n")
    (source / "b.py").write_text("a = 1\n")
    (source / "c.py").write_text("c = 3\n")
    (source / "d.py").write_text("d = 4\n")
    output = tmp_path / "manifest.json"
    manifest = prepare(source, output)
    assert output.exists()
    assert len(manifest["train"]) + len(manifest["validation"]) == 3
    assert set(x["sha256"] for x in manifest["train"]).isdisjoint(
        x["sha256"] for x in manifest["validation"]
    )


def test_prepare_rejects_single_unique_file(tmp_path: Path):
    (tmp_path / "one.py").write_text("x = 1")
    with pytest.raises(ValueError, match="unique"):
        prepare(tmp_path, tmp_path / "out.json")
