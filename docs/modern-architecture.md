# CorvusBrain V1.1 — Modern architecture prototype

This release introduces an **opt-in** modern decoder in
`corvusai.models.modern`, without modifying the existing
`CorvusTransformer` or the checkpoint format used by V1.0.

## Architecture

- **RoPE** rotary position embeddings on attention queries and keys.
- **RMSNorm** pre-normalization.
- **Grouped-query attention (GQA)** with configurable query and KV heads.
- **PyTorch SDPA** attention kernel selection (hardware-dependent).
- **SwiGLU** feed-forward network.
- **KV cache** for incremental decoding within the configured context window.

## Quick smoke test

```python
import torch
from corvusai.models.modern import ModernConfig, ModernCorvusTransformer

model = ModernCorvusTransformer(
    ModernConfig(vocab_size=256, context_length=128, width=128,
                 heads=4, kv_heads=2, layers=2)
).eval()
tokens = torch.tensor([[100, 101, 102]])
with torch.inference_mode():
    logits, cache = model(tokens, use_cache=True)
    next_logits, cache = model(torch.tensor([[103]]), cache=cache, use_cache=True)
print(logits.shape, next_logits.shape)
```

**Compatibility:** Existing V1.0 checkpoints remain loadable through
the legacy model and CLI. This new architecture has different parameter
names and shapes; old model weights cannot be loaded into it directly.

**Scope:** The modern decoder and cache have unit tests, but the V1.0
training and interactive inference CLIs still use the legacy architecture.
Training checkpoints and a CLI backed by this modern model require a
separate integration step. No GPU throughput, VRAM, or accuracy improvement
is claimed without benchmarks. The cache has no sliding-window eviction:
it raises when the total sequence exceeds the configured context.
