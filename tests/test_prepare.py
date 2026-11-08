import json
from pathlib import Path

import pytest

from corvusai.training.prepare import prepare


def _licenses(root: Path, names: list[str]) -> Path:
    path = root.parent / "licenses.json"
    path.write_text(json.dumps({name: "MIT" for name in names}), encoding="utf-8")
    return path


def test_prepare_deduplicates_and_splits(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.py").write_text("a = 1\n")
    (source / "b.py").write_text("a = 1\n")
    (source / "c.py").write_text("c = 3\n")
    (source / "d.py").write_text("d = 4\n")
    output = tmp_path / "manifest.json"
    licenses = _licenses(source, ["a.py", "b.py", "c.py", "d.py"])
    manifest = prepare(source, output, licenses=licenses)
    assert output.exists()
    assert manifest["format"] == "corvus-dataset-manifest-v2"
    assert len(manifest["train"]) + len(manifest["validation"]) == 3
    assert {x["sha256"] for x in manifest["train"]}.isdisjoint(
        x["sha256"] for x in manifest["validation"]
    )
    assert all(x["license"] == "MIT" for x in manifest["train"] + manifest["validation"])


def test_prepare_rejects_single_unique_file(tmp_path: Path):
    (tmp_path / "one.py").write_text("x = 1")
    licenses = _licenses(tmp_path, ["one.py"])
    with pytest.raises(ValueError, match="unique"):
        prepare(tmp_path, tmp_path / "out.json", licenses=licenses)


def test_filters_secrets_licenses_and_normalized_duplicates(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    files = {
        "a.py": "answer = 1\n",
        "b.py": "# comment\nanswer=1\n",
        "c.py": "other = 2\n",
        "d.py": "password = 'super-secret-value'\n",
        "e.py": "private = 5\n",
        "f.py": "more = 6\n",
    }
    for name, value in files.items():
        (source / name).write_text(value)
    licenses = _licenses(source, ["a.py", "b.py", "c.py", "d.py", "f.py"])
    manifest = prepare(source, tmp_path / "manifest.json", licenses=licenses)
    included = {record["path"] for record in manifest["train"] + manifest["validation"]}
    assert "d.py" not in included
    assert "e.py" not in included
    assert len(included.intersection({"a.py", "b.py"})) == 1
    assert manifest["rejected_counts"]["potential_secret"] == 1
    assert manifest["rejected_counts"]["license_missing_or_not_allowlisted"] == 1


def test_reproducible_manifest(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    for name in ["a.py", "b.py", "c.py"]:
        (source / name).write_text(f"value_{name[0]} = 1\n")
    licenses = _licenses(source, ["a.py", "b.py", "c.py"])
    first = prepare(source, tmp_path / "one.json", licenses=licenses)
    second = prepare(source, tmp_path / "two.json", licenses=licenses)
    assert first == second
    assert (tmp_path / "one.json").read_bytes() == (tmp_path / "two.json").read_bytes()


def test_requires_license_declarations(tmp_path: Path):
    with pytest.raises(ValueError, match="licenses"):
        prepare(tmp_path, tmp_path / "out.json")
