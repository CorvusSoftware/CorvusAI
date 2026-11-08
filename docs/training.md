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
