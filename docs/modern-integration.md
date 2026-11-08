# CorvusBrain V1.2 — Modern model training and inference

This integration connects the opt-in V1.1 RoPE/RMSNorm/GQA decoder to
the existing BPE training and interactive inference workflows.

## Train a new modern checkpoint

```powershell
python -m corvusai.training.bpe_trainer `
  --corpus data `
  --tokenizer checkpoints/tokenizer.json `
  --output checkpoints/modern `
  --architecture modern `
  --width 128 --heads 4 --kv-heads 2 --layers 2 `
  --context-length 64 --steps 1000 `
  --batch-size 2 --gradient-accumulation 8 `
  --device cuda --mixed-precision `
  --warmup-steps 50 --decay-steps 1000
```

Resume with the same settings and `--resume`, increasing `--steps`.
Modern checkpoints have format `corvus-modern-bpe-checkpoint-v1`.
Legacy training remains the default and its checkpoint format is unchanged.

## Run local inference with a KV cache

```powershell
python -m corvusai.inference.brain `
  --checkpoint checkpoints/modern/latest.pt `
  --tokenizer checkpoints/tokenizer.json `
  --device cuda `
  --max-new-tokens 80
```

The shell automatically detects modern checkpoints and uses incremental
KV caching. Once the context window is full, it rebuilds the cache from
the latest context tokens. This avoids exceeding the configured context,
but rebuilding adds computation.

## Compatibility and limitations

Existing legacy checkpoints continue to load through the legacy model.
Modern model weights cannot be converted from old checkpoint weights.
Only trusted local checkpoints should be loaded; generated code is not
executed. No GPU speedup, VRAM reduction, or code quality improvement
is claimed without benchmarks. The modern training path requires a new
training run from random initialization.
