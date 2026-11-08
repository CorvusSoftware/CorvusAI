# CorvusBrain V0.5 — BPE pretraining and inference

Train the tokenizer using only data you are permitted to use:

```powershell
.\.venv\Scripts\python.exe -m corvusai.tokenization.bpe --input data --output checkpoints/tokenizer.json --merges 64
.\.venv\Scripts\python.exe -m corvusai.training.bpe_trainer --corpus data/sample.py --tokenizer checkpoints/tokenizer.json --output checkpoints/bpe --steps 20 --context-length 32 --device cpu
.\.venv\Scripts\python.exe -m corvusai.inference.generate --checkpoint checkpoints/bpe/latest.pt --tokenizer checkpoints/tokenizer.json --prompt "def add(a, b):" --max-new-tokens 50 --device cpu
```

BPE checkpoints record the SHA-256 of the tokenizer JSON. Generation rejects
a tokenizer whose bytes do not match the training tokenizer. Older byte-level
checkpoints continue to work without a tokenizer argument.

This is a small CPU reference implementation; CUDA is supported by the training
CLI but has not been benchmarked on the RTX 4060. Resume, mixed precision, and
multi-file manifest loading are not yet implemented for the BPE trainer.

## V0.6 — multi-file BPE training

Pass a directory containing `.py` files to `--corpus`. Documents are
sorted by relative path; sampling windows never cross file boundaries.
Documents with fewer than `context_length + 1` tokens are skipped.

```powershell
python -m corvusai.training.bpe_trainer --corpus data --tokenizer checkpoints/tokenizer.json --output checkpoints/bpe --steps 100 --batch-size 2 --context-length 128 --gradient-accumulation 8 --device cuda --mixed-precision
```

Continue to a **total** of 200 optimizer steps:

```powershell
python -m corvusai.training.bpe_trainer --corpus data --tokenizer checkpoints/tokenizer.json --output checkpoints/bpe --steps 200 --batch-size 2 --context-length 128 --gradient-accumulation 8 --device cuda --mixed-precision --resume
```

V0.6 checkpoints (`corvus-bpe-checkpoint-v2`) store optimizer, AMP scaler,
CPU/CUDA RNG states, model configuration, corpus fingerprint, tokenizer
fingerprint, batch size, and accumulation count. Resume refuses incompatible
settings or changed data. Use checkpoints from trusted local sources only.

`--mixed-precision` enables CUDA FP16 autocast and gradient scaling. It is
disabled on CPU. Gradient accumulation divides each microbatch loss by the
number of microbatches before backward; `--steps` counts optimizer updates.
The entire tokenized corpus is currently held in CPU RAM. A future release
will add streaming datasets, validation splits, and larger-model optimizations.
