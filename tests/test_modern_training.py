"""Small CPU regression tests for the modern CorvusBrain training path."""

import json
from pathlib import Path

import pytest
import torch

from corvusai.inference.brain import BrainSession
from corvusai.models.modern import ModernConfig, ModernCorvusTransformer
from corvusai.tokenization.bpe import ByteBPE
from corvusai.training.bpe_trainer import train_bpe


def test_modern_cached_logits_match_full_forward():
    torch.manual_seed(7)
    config = ModernConfig(vocab_size=32, context_length=12, width=16,
                          heads=4, kv_heads=2, layers=2)
    model = ModernCorvusTransformer(config).eval()
    tokens = torch.randint(0, config.vocab_size, (1, 7))
    with torch.inference_mode():
        full = model(tokens)
        first, cache = model(tokens[:, :4], use_cache=True)
        second, cache = model(tokens[:, 4:], cache=cache, use_cache=True)
    assert torch.allclose(full[:, :4], first, atol=1e-5, rtol=1e-5)
    assert torch.allclose(full[:, 4:], second, atol=1e-5, rtol=1e-5)
    assert cache[0][0].shape[2] == tokens.shape[1]


def test_modern_train_resume_and_generate(tmp_path: Path, capsys):
    corpus = tmp_path / "corpus.py"
    corpus.write_bytes(b"def add(a, b):\n    return a + b\n" * 8)
    tokenizer_path = tmp_path / "tokenizer.json"
    ByteBPE.train([corpus.read_bytes()], merges=8).save(tokenizer_path)
    options = dict(context_length=8, batch_size=1, architecture="modern",
                   width=16, heads=4, kv_heads=2, layers=1, seed=23)
    output = tmp_path / "run"
    losses = train_bpe(corpus, tokenizer_path, output, steps=1, **options)
    assert len(losses) == 1
    metrics = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert metrics["step_seconds"] > 0
    assert metrics["train_tokens_per_second"] > 0
    assert metrics["model_parameters"] > 0
    state = torch.load(output / "latest.pt", weights_only=True)
    assert state["format"] == "corvus-modern-bpe-checkpoint-v1"
    assert state["step"] == 1
    resumed = train_bpe(corpus, tokenizer_path, output, steps=2, resume=True,
                        **options)
    assert len(resumed) == 1
    session = BrainSession(output / "latest.pt", tokenizer_path=tokenizer_path)
    assert session.complete("def ", max_new_tokens=2).startswith("def ")
    with pytest.raises(ValueError, match="mismatch"):
        train_bpe(corpus, tokenizer_path, output, steps=3, resume=True,
                  **{**options, "width": 32})
