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


## Comparing training runs

Use the comparison command on local BPE run folders, or pass explicit
`metrics.jsonl` paths:

```bash
python -m corvusai.training.compare checkpoints/legacy checkpoints/modern --warmup 2
```

The command prints a JSON array containing per-run summary statistics.
`--warmup 2` excludes the first two *recorded* steps from mean/median
calculations, while retaining the final recorded loss and step count.
Missing, malformed, nonfinite or nonpositive timing/throughput metrics
are rejected. This tool does not run training, infer speedups or claim
the runs are directly comparable. For meaningful throughput comparisons,
keep the device, corpus, tokenizer, batch size, context length and
measurement conditions consistent. Training loss alone is not a
held-out quality metric.

## Comparison compatibility checks

New BPE telemetry records the architecture, context length, batch size,
gradient accumulation, mixed precision and SHA-256 fingerprints of the
corpus and tokenizer. The comparison CLI emits a JSON object with
`runs` and `warnings`. For multiple runs it flags different or
missing settings that prevent a fair speed comparison. It also warns
that the hardware model and software environment are not verified.
Older `metrics.jsonl` files remain readable, but their missing
metadata is explicitly reported rather than silently assumed equal.


## Held-out evaluation for BPE training

Use a separate, independently held-out Python corpus to measure next-token
validation loss and perplexity without updating model weights:

```bash
python -m corvusai.training.bpe_trainer --corpus data/train.py --validation-corpus data/heldout.py --tokenizer checkpoints/tokenizer.json --architecture modern --output checkpoints/modern --context-length 64 --steps 20 --eval-every 5
```

`--eval-every 5` records `validation_loss`, `validation_perplexity`,
`validation_tokens` and `validation_sha256` on every fifth optimizer step.
When a validation corpus is provided without `--eval-every`, evaluation runs
after every step. Validation is performed in evaluation/inference mode and
does not update model weights. Checkpoint resume requires the same validation
corpus fingerprint; existing runs without validation remain supported.

The validation corpus must be distinct from training data. The fingerprint
check catches identical input paths/content under the same corpus name, but
does not detect near-duplicates or leakage across files. Validation
perplexity is tokenizer-dependent, and does not by itself establish code
generation quality. The validation pass adds computation after each
checkpoint; `step_seconds` excludes that evaluation.

## Validation cadence on resume

For new checkpoints with a held-out corpus, the effective validation
frequency (`eval_every`, where `0` means every step) is saved in
`latest.pt`. Resume rejects a changed frequency, preventing gaps or
unexpected changes in the validation history. Earlier checkpoints without
this field can be resumed with the default every-step frequency only.
A training corpus and validation corpus cannot resolve to the same path,
and matching corpus fingerprints are also rejected. These checks do not
guarantee the absence of near-duplicate data leakage.

## Exact duplicate protection

When `--validation-corpus` is provided, the BPE trainer hashes the
bytes of every input source file in both corpora and rejects any exact
cross-split match, even when filenames or directory structures differ.
This also catches a shared file within otherwise distinct directories.
The check runs before optimizer steps and leaves training-only runs
unchanged. It is not a near-duplicate detector: copied functions,
reformatted code and similar files still require a separate leakage
audit. Prepare validation splits by repository/project where possible.

## Best held-out validation results

The training comparison CLI now reports `best_validation_step`,
`best_validation_loss`, `best_validation_perplexity` (when recorded),
`last_validation_step`, `last_validation_loss`, and
`validation_evaluations` for runs with held-out measurements.
The best step minimizes validation loss across recorded evaluation
steps, independently of the `--warmup` exclusion used for throughput
statistics. It is not necessarily the final checkpoint: the BPE trainer
still writes only `latest.pt` and does not preserve a separate
best-validation checkpoint. Compare validation losses only for
compatible tokenizers and held-out datasets; the CLI warns when the
validation dataset fingerprint is missing or differs.

## Best-validation checkpoint

With `--validation-corpus`, training now writes `best.pt` whenever
held-out loss strictly improves. `latest.pt` continues to be written
at every optimizer step for resuming training. Both checkpoints contain
the complete model and optimizer state, and the best loss is persisted
in the latest checkpoint so resumed runs keep the previous best score.
`best.pt` is not created when validation is disabled, and evaluation
only happens on the configured `--eval-every` cadence. An existing
`best.pt` should be kept alongside `latest.pt` when transferring a
run between machines. A fresh run in an existing output directory
should use a new folder to avoid mixing artifacts from separate runs.
