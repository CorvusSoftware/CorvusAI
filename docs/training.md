# Tiny model training

This is a smoke test, not a useful code model. It trains a randomly initialized
Transformer on UTF-8 bytes from a local file and writes a checkpoint after each step.

## Windows / RTX 4060

Install a CUDA-enabled PyTorch build using the current instructions at
https://pytorch.org/get-started/locally/ then install the project:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m corvusai.training.trainer --corpus data/sample.py --steps 20 --device cuda
```

To continue to step 40:

```powershell
.\.venv\Scripts\python.exe -m corvusai.training.trainer --corpus data/sample.py --steps 40 --device cuda --resume
```

The first run is intentionally small and uses full-precision AdamW. The actual
GPU memory and speed must be measured on your own machine. Checkpoints are
saved under `checkpoints/tiny` and excluded from Git.

Do not train on unlicensed source code. For real pretraining, add a validated
dataset pipeline and independent held-out evaluation.

## V0.2: held-out validation and CUDA memory tracking

Use **different files** for training and validation. The validation corpus must not
be a copy of the training corpus. For real experiments, split by project/repository
before preprocessing to avoid near-duplicate leakage.

```powershell
.\.venv\Scripts\python.exe -m corvusai.training.trainer --corpus data/sample.py --validation-corpus data/validation.py --steps 50 --eval-every 10 --mixed-precision --device cuda
```

Provide your own `data/validation.py` before running this example. Metrics include
training loss, validation loss, perplexity and CUDA peak allocated VRAM. The
`--mixed-precision` option uses CUDA FP16 autocast and gradient scaling; it
is ignored on CPU. Resuming checks corpus SHA-256 hashes, so changed datasets
cannot silently reuse a checkpoint. Do not compare perplexities from different
tokenizers as if they were directly equivalent.

`split_files(directory)` provides a deterministic, disjoint file-level split
for dataset preparation. A full deduplication and licensing pipeline is still
needed before training on large third-party corpora.


## Modern architecture verification and throughput

The BPE trainer supports an opt-in modern Transformer using RoPE, RMSNorm, grouped-query attention, SwiGLU and PyTorch scaled dot-product attention:

```bash
python -m corvusai.training.bpe_trainer --corpus data/sample.py --tokenizer checkpoints/tokenizer.json --output checkpoints/modern --architecture modern --context-length 64 --width 128 --heads 4 --kv-heads 2 --layers 2 --steps 20 --device cpu
```

The `legacy` architecture remains the default for compatibility. Checkpoints are architecture-specific; resume using the same configuration and tokenizer. The modern path has CPU regression tests covering cached attention versus full forward, checkpoint resume and local generation.

Each training step now logs `step_seconds`, `train_tokens_per_second`, and `model_parameters`. Token throughput counts `batch_size * context_length * gradient_accumulation` training tokens per optimizer step; timing excludes checkpoint serialization, and synchronizes CUDA before reading elapsed time. Compare only runs with comparable hardware, batch settings and context lengths. These metrics do not establish model quality or optimized GPU kernel selection.

Verify with `ruff check .` and `python -m pytest -q` before treating changes as validated. GPU throughput and peak memory must be measured on the target GPU, not inferred from CPU tests.


## Persistent training metrics

Each BPE training run writes one JSON object per completed optimizer step to
`<output>/metrics.jsonl`, alongside `latest.pt`. The file contains the same
metrics emitted to standard output, including loss, learning rate, elapsed
training-step time, tokens per second and parameter count. Fresh runs reset
the metrics file. On `--resume`, existing records after the checkpoint's
committed step are discarded before new records are appended, so an
interrupted run cannot retain measurements for steps that were not restored.

This is experimental telemetry, not a benchmark result. Measure GPU
performance on the intended hardware and confirm the CI before relying on it.
