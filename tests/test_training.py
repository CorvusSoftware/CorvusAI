from pathlib import Path

import pytest
import torch

from corvusai.training.trainer import train


def test_train_and_resume(tmp_path: Path):
    corpus = tmp_path / "sample.py"
    corpus.write_text("def add(a, b):\n    return a + b\n" * 6, encoding="utf-8")
    output = tmp_path / "checkpoints"
    losses = train(corpus, output, steps=1, batch_size=1, context_length=8, device="cpu")
    assert len(losses) == 1
    assert (output / "latest.pt").exists()
    more = train(corpus, output, steps=2, batch_size=1, context_length=8,
                 device="cpu", resume=True)
    assert len(more) == 1
    state = torch.load(output / "latest.pt", weights_only=True)
    assert state["step"] == 2


def test_rejects_short_corpus(tmp_path: Path):
    corpus = tmp_path / "tiny.py"
    corpus.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="Corpus"):
        train(corpus, tmp_path / "out", context_length=8, device="cpu")
