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
