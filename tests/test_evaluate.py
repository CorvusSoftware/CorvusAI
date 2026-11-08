import torch

from corvusai.models.transformer import CorvusTransformer, ModelConfig
from corvusai.training.evaluate import evaluate


def test_evaluation_restores_training_mode():
    model = CorvusTransformer(ModelConfig(context_length=8, width=32, heads=4, layers=1))
    model.train()
    data = torch.randint(0, 256, (80,))
    result = evaluate(model, data, batches=2, batch_size=1, context_length=8,
                      device=torch.device("cpu"))
    assert model.training
    assert result["validation_loss"] > 0
    assert result["perplexity"] > 1
