from pathlib import Path

import pytest
import torch

from corvusai.inference.generate import generate
from corvusai.tokenization.bpe import ByteBPE
from corvusai.training.bpe_trainer import train_bpe


def test_bpe_checkpoint_has_tokenizer_fingerprint(tmp_path: Path):
    corpus = tmp_path / "corpus.py"
    corpus.write_bytes(b"def foo():\n    return 42\n" * 16)
    tokenizer_path = tmp_path / "tokenizer.json"
    ByteBPE.train([corpus.read_bytes()], merges=16).save(tokenizer_path)
    losses = train_bpe(corpus, tokenizer_path, tmp_path / "output",
                       steps=1, context_length=8, batch_size=1)
    assert len(losses) == 1
    state = torch.load(tmp_path / "output" / "latest.pt", weights_only=True)
    assert state["format"] == "corvus-bpe-checkpoint-v1"
    assert state["config"]["vocab_size"] == ByteBPE.load(tokenizer_path).vocab_size
    assert len(state["tokenizer_sha256"]) == 64
    generated = generate(tmp_path / "output" / "latest.pt", "def ",
                         max_new_tokens=3, tokenizer_path=tokenizer_path, device="cpu")
    assert generated.startswith("def ")
    other = tmp_path / "other.json"
    ByteBPE.train([b"another corpus" * 5], merges=8).save(other)
    with pytest.raises(ValueError, match="SHA-256"):
        generate(tmp_path / "output" / "latest.pt", "def ",
                 tokenizer_path=other, device="cpu")


def test_rejects_short_corpus(tmp_path: Path):
    corpus = tmp_path / "short.py"
    corpus.write_bytes(b"a")
    tokenizer_path = tmp_path / "tokenizer.json"
    ByteBPE.train([b"aaaa"], merges=1).save(tokenizer_path)
    with pytest.raises(ValueError, match="too few"):
        train_bpe(corpus, tokenizer_path, tmp_path / "output", context_length=8)
