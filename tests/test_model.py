import pytest
import torch

from corvusai.models.transformer import CorvusTransformer, ModelConfig


def test_forward_and_backward():
    model = CorvusTransformer(ModelConfig(vocab_size=256, context_length=16, width=32, heads=4, layers=2))
    tokens = torch.randint(0, 256, (2, 10))
    logits = model(tokens)
    assert logits.shape == (2, 10, 256)
    loss = model.loss(tokens)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.tokens.weight.grad is not None


def test_causal_mask():
    model = CorvusTransformer(ModelConfig(context_length=16, width=32, heads=4, layers=1))
    model.eval()
    a = torch.tensor([[1, 2, 3, 4]])
    b = torch.tensor([[1, 2, 9, 8]])
    with torch.no_grad():
        first = model(a)[:, :2]
        second = model(b)[:, :2]
    torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-6)


def test_rejects_long_context():
    model = CorvusTransformer(ModelConfig(context_length=4, width=32, heads=4, layers=1))
    with pytest.raises(ValueError):
        model(torch.ones((1, 5), dtype=torch.long))
