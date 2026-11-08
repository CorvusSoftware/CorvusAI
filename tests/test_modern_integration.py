"""End-to-end modern BPE training and cached inference smoke tests."""

from pathlib import Path

import pytest
import torch

from corvusai.inference.brain import BrainSession
from corvusai.tokenization.bpe import ByteBPE
from corvusai.training.bpe_trainer import train_bpe


def test_modern_training_resume_and_cached_inference(tmp_path: Path):
    corpus = tmp_path / "example.py"
    corpus.write_bytes(b"def add(a, b):\n    return a + b\n" * 12)
    tokenizer_path = tmp_path / "tokenizer.json"
    ByteBPE.train([corpus.read_bytes()], merges=8).save(tokenizer_path)
    output = tmp_path / "modern"
    kwargs = {"context_length": 8, "batch_size": 1, "seed": 11,
              "architecture": "modern", "width": 16, "heads": 2,
              "layers": 1, "kv_heads": 1}
    train_bpe(corpus, tokenizer_path, output, steps=1, **kwargs)
    checkpoint = output / "latest.pt"
    state = torch.load(checkpoint, weights_only=True)
    assert state["format"] == "corvus-modern-bpe-checkpoint-v1"
    train_bpe(corpus, tokenizer_path, output, steps=2, resume=True, **kwargs)
    session = BrainSession(checkpoint, tokenizer_path=tokenizer_path)
    assert session.modern
    first = session.complete("def ", max_new_tokens=12, seed=4)
    assert first.startswith("def ")
    assert first == session.complete("def ", max_new_tokens=12, seed=4)


def test_architecture_mismatch_rejected(tmp_path: Path):
    corpus = tmp_path / "example.py"
    corpus.write_bytes(b"print('hello')\n" * 12)
    tokenizer_path = tmp_path / "tokenizer.json"
    ByteBPE.train([corpus.read_bytes()], merges=4).save(tokenizer_path)
    output = tmp_path / "modern"
    kwargs = {"context_length": 8, "batch_size": 1, "width": 16,
              "heads": 2, "layers": 1, "kv_heads": 1}
    train_bpe(corpus, tokenizer_path, output, steps=1, architecture="modern", **kwargs)
    with pytest.raises(ValueError, match="checkpoint"):
        train_bpe(corpus, tokenizer_path, output, steps=2, resume=True,
                  architecture="legacy", context_length=8, batch_size=1)
