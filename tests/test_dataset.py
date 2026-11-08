from pathlib import Path

import pytest

from corvusai.training.dataset import corpus_digest, load_corpus, split_files


def test_file_split_is_disjoint_and_stable(tmp_path: Path):
    for name in ("a.py", "b.py", "c.py", "d.py"):
        (tmp_path / name).write_text("x = 1\n" * 10)
    first = split_files(tmp_path)
    second = split_files(tmp_path)
    assert first == second
    assert set(first[0]).isdisjoint(first[1])
    assert len(first[0]) + len(first[1]) == 4


def test_short_corpus_rejected(tmp_path: Path):
    path = tmp_path / "short.py"
    path.write_bytes(b"x")
    with pytest.raises(ValueError, match="too short"):
        load_corpus(path, 8)


def test_digest_changes_with_content(tmp_path: Path):
    path = tmp_path / "x.py"
    path.write_bytes(b"abc")
    before = corpus_digest(path)
    path.write_bytes(b"abd")
    assert before != corpus_digest(path)
