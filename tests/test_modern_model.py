"""Modern decoder forward and cached attention regression tests."""

import pytest
import torch

from corvusai.models.modern import ModernConfig, ModernCorvusTransformer


def test_cached_logits_match_full_forward():
    torch.manual_seed(4)
    config = ModernConfig(vocab_size=32, context_length=16, width=32,
                          heads=4, kv_heads=2, layers=2)
    model = ModernCorvusTransformer(config).eval()
    tokens = torch.randint(0, 32, (2, 9))
    with torch.inference_mode():
        full = model(tokens)
        first, cache = model(tokens[:, :5], use_cache=True)
        second, cache = model(tokens[:, 5:7], cache=cache, use_cache=True)
        third, cache = model(tokens[:, 7:], cache=cache, use_cache=True)
    assert torch.allclose(full, torch.cat((first, second, third), dim=1),
                          atol=2e-5, rtol=2e-5)
    assert len(cache) == 2
    assert cache[0][0].shape == (2, 2, 9, 8)


def test_modern_model_backward():
    model = ModernCorvusTransformer(
        ModernConfig(vocab_size=24, context_length=8, width=32,
                     heads=4, kv_heads=1, layers=1)
    )
    loss = model.loss(torch.randint(0, 24, (2, 7)))
    assert torch.isfinite(loss)
    loss.backward()
    assert model.blocks[0].q.weight.grad is not None


@pytest.mark.parametrize("kwargs", [
    {"width": 30}, {"kv_heads": 3}, {"width": 12},
    {"context_length": 0}, {"rope_base": 0},
])
def test_invalid_configs(kwargs):
    with pytest.raises(ValueError):
        ModernConfig(**kwargs)


def test_cache_rejects_context_overflow():
    model = ModernCorvusTransformer(
        ModernConfig(vocab_size=16, context_length=4, width=16,
                     heads=2, kv_heads=1, layers=1)
    )
    _, cache = model(torch.ones(1, 4, dtype=torch.long), use_cache=True)
    with pytest.raises(ValueError, match="context"):
        model(torch.ones(1, 1, dtype=torch.long), cache=cache)
