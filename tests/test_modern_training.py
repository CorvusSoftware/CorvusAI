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
    options = {"context_length": 8, "batch_size": 1, "architecture": "modern",
               "width": 16, "heads": 4, "kv_heads": 2, "layers": 1, "seed": 23}
    output = tmp_path / "run"
    losses = train_bpe(corpus, tokenizer_path, output, steps=1, **options)
    assert len(losses) == 1
    metrics = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert metrics["step_seconds"] > 0
    assert metrics["train_tokens_per_second"] > 0
    assert metrics["model_parameters"] > 0
    metrics_path = output / "metrics.jsonl"
    assert json.loads(metrics_path.read_text(encoding="utf-8").strip())["step"] == 1
    state = torch.load(output / "latest.pt", weights_only=True)
    assert state["format"] == "corvus-modern-bpe-checkpoint-v1"
    assert state["step"] == 1
    resumed = train_bpe(corpus, tokenizer_path, output, steps=2, resume=True,
                        **options)
    assert len(resumed) == 1
    recorded = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    assert [entry["step"] for entry in recorded] == [1, 2]
    session = BrainSession(output / "latest.pt", tokenizer_path=tokenizer_path)
    assert session.complete("def ", max_new_tokens=2).startswith("def ")
    with pytest.raises(ValueError, match="mismatch"):
        train_bpe(corpus, tokenizer_path, output, steps=3, resume=True,
                  **{**options, "width": 32})


def test_modern_loss_backpropagates():
    torch.manual_seed(5)
    model = ModernCorvusTransformer(ModernConfig(
        vocab_size=32, context_length=8, width=16, heads=4, kv_heads=2, layers=1,
    ))
    tokens = torch.randint(0, 32, (2, 6))
    loss = model.loss(tokens)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.tokens.weight.grad is not None
    assert torch.isfinite(model.tokens.weight.grad).all()


def test_modern_rejects_excessive_context():
    model = ModernCorvusTransformer(ModernConfig(
        vocab_size=32, context_length=4, width=16, heads=4, kv_heads=2, layers=1,
    ))
    with pytest.raises(ValueError, match="context length"):
        model(torch.randint(0, 32, (1, 5)))


def test_modern_rejects_invalid_cache_shapes():
    model = ModernCorvusTransformer(ModernConfig(
        vocab_size=32, context_length=8, width=16, heads=4, kv_heads=2, layers=2,
    ))
    tokens = torch.randint(0, 32, (1, 2))
    with torch.inference_mode():
        _, cache = model(tokens, use_cache=True)
    with pytest.raises(ValueError, match="Cache layer count"):
        model(tokens[:, :1], cache=cache[:1])
    bad_cache = list(cache)
    bad_cache[1] = (cache[1][0][:, :, :1, :], cache[1][1])
    with pytest.raises(ValueError, match="Invalid cache shape"):
        model(tokens[:, :1], cache=bad_cache)
    bad_cache = list(cache)
    bad_cache[0] = (cache[0][0].repeat(2, 1, 1, 1), cache[0][1])
    with pytest.raises(ValueError, match="Invalid cache shape"):
        model(tokens[:, :1], cache=bad_cache)


def test_modern_heldout_validation_and_resume(tmp_path: Path):
    corpus = tmp_path / "train.py"
    validation = tmp_path / "validation.py"
    corpus.write_bytes(b"def square(x):\n    return x * x\n" * 6)
    validation.write_bytes(b"def cube(y):\n    return y * y * y\n" * 6)
    tokenizer_path = tmp_path / "tokenizer.json"
    ByteBPE.train([corpus.read_bytes()], merges=8).save(tokenizer_path)
    output = tmp_path / "validation-run"
    options = {"context_length": 8, "batch_size": 1, "architecture": "modern",
               "width": 16, "heads": 4, "kv_heads": 2, "layers": 1,
               "validation_corpus": validation, "eval_every": 2}
    train_bpe(corpus, tokenizer_path, output, steps=2, **options)
    records = [json.loads(line) for line in
               (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert "validation_loss" not in records[0]
    assert records[1]["validation_loss"] > 0
    assert records[1]["validation_perplexity"] > 1
    assert records[1]["validation_tokens"] > 0
    train_bpe(corpus, tokenizer_path, output, steps=3, resume=True, **options)
    state = torch.load(output / "latest.pt", weights_only=True)
    assert state["eval_every"] == 2
    with pytest.raises(ValueError, match="mismatch"):
        train_bpe(corpus, tokenizer_path, output, steps=4, resume=True,
                  **{**options, "eval_every": 1})
    with pytest.raises(ValueError, match="mismatch"):
        train_bpe(corpus, tokenizer_path, output, steps=4, resume=True,
                  **{**options, "validation_corpus": None, "eval_every": 0})
    with pytest.raises(ValueError, match="differ"):
        train_bpe(corpus, tokenizer_path, tmp_path / "invalid", steps=1,
                  **{**options, "validation_corpus": corpus})
