from pathlib import Path

import pytest
import torch

from corvusai.inference.generate import generate
from corvusai.models.transformer import CorvusTransformer, ModelConfig


def test_generate_is_deterministic(tmp_path: Path):
    model = CorvusTransformer(ModelConfig(context_length=8, width=32, heads=4, layers=1))
    checkpoint = tmp_path / "tiny.pt"
    torch.save({"config": vars(model.config), "model": model.state_dict()}, checkpoint)
    a = generate(checkpoint, "def ", max_new_tokens=4, seed=7, device="cpu")
    b = generate(checkpoint, "def ", max_new_tokens=4, seed=7, device="cpu")
    assert a == b
    assert a.startswith("def ")


def test_rejects_empty_prompt(tmp_path: Path):
    model = CorvusTransformer(ModelConfig(context_length=8, width=32, heads=4, layers=1))
    checkpoint = tmp_path / "tiny.pt"
    torch.save({"config": vars(model.config), "model": model.state_dict()}, checkpoint)
    with pytest.raises(ValueError, match="Prompt"):
        generate(checkpoint, "", device="cpu")
