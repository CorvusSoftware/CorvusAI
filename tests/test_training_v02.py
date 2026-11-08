from pathlib import Path

import pytest

from corvusai.training.trainer import train


def test_validation_and_resume_dataset_integrity(tmp_path: Path, capsys):
    corpus = tmp_path / "train.py"
    validation = tmp_path / "validation.py"
    corpus.write_text("def f(x):\n    return x + 1\n" * 6, encoding="utf-8")
    validation.write_text("def g(x):\n    return x * 2\n" * 6, encoding="utf-8")
    output = tmp_path / "out"
    train(corpus, output, steps=1, batch_size=1, context_length=8, device="cpu",
          validation_corpus=validation, eval_every=1)
    captured = capsys.readouterr().out
    assert '"validation_loss"' in captured
    assert '"perplexity"' in captured
    train(corpus, output, steps=2, batch_size=1, context_length=8, device="cpu",
          validation_corpus=validation, resume=True)
    corpus.write_text("changed content" * 6, encoding="utf-8")
    with pytest.raises(ValueError, match="corpus changed"):
        train(corpus, output, steps=3, batch_size=1, context_length=8, device="cpu",
              validation_corpus=validation, resume=True)


def test_rejects_same_train_and_validation_file(tmp_path: Path):
    corpus = tmp_path / "train.py"
    corpus.write_text("x = 1\n" * 20, encoding="utf-8")
    with pytest.raises(ValueError, match="different files"):
        train(corpus, tmp_path / "out", steps=1, context_length=8, device="cpu",
              validation_corpus=corpus)
