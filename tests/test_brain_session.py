"""Tests for the V1.0 persistent local inference shell."""

from pathlib import Path

import pytest
import torch

from corvusai.inference.brain import BrainSession
from corvusai.models.transformer import CorvusTransformer, ModelConfig
from corvusai.tokenization.bpe import ByteBPE


def _checkpoint(tmp_path: Path, *, bpe: bool = False) -> tuple[Path, Path | None]:
    tokenizer_path = None
    vocab_size = 256
    state = {}
    if bpe:
        tokenizer_path = tmp_path / "tokenizer.json"
        tokenizer = ByteBPE([(97, 98)])
        tokenizer.save(tokenizer_path)
        vocab_size = tokenizer.vocab_size
        import hashlib
        state["format"] = "corvus-bpe-checkpoint-v2"
        state["tokenizer_sha256"] = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    config = ModelConfig(vocab_size=vocab_size, context_length=8, width=16, heads=2, layers=1)
    state["config"] = vars(config)
    state["model"] = CorvusTransformer(config).state_dict()
    checkpoint = tmp_path / "latest.pt"
    torch.save(state, checkpoint)
    return checkpoint, tokenizer_path


def test_session_deterministic_and_reusable(tmp_path: Path):
    checkpoint, _ = _checkpoint(tmp_path)
    session = BrainSession(checkpoint)
    first = session.complete("x = ", max_new_tokens=3, seed=12)
    assert first.startswith("x = ")
    assert first == session.complete("x = ", max_new_tokens=3, seed=12)
    assert session.complete("x = ", max_new_tokens=0) == "x = "
    with pytest.raises(ValueError, match="Prompt"):
        session.complete("")
    with pytest.raises(ValueError, match="parameters"):
        session.complete("x", temperature=0)


def test_bpe_checkpoint_validation(tmp_path: Path):
    checkpoint, tokenizer = _checkpoint(tmp_path, bpe=True)
    assert tokenizer is not None
    with pytest.raises(ValueError, match="tokenizer"):
        BrainSession(checkpoint)
    session = BrainSession(checkpoint, tokenizer_path=tokenizer)
    assert session.complete("ab", max_new_tokens=0) == "ab"
    tokenizer.write_text(tokenizer.read_text() + " ")
    with pytest.raises(ValueError, match="SHA-256"):
        BrainSession(checkpoint, tokenizer_path=tokenizer)
