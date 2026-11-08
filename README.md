<div align="center">

<a href="https://github.com/CorvusSoftware">
  <img src="https://github.com/CorvusSoftware.png" alt="CorvusSoftware — logo de l'organisation GitHub" width="112" height="112">
</a>

# CorvusAI

**An open-source foundation for independent code intelligence.**

*CorvusBrain · CorvusCode*

[![CI](https://img.shields.io/github/actions/workflow/status/CorvusSoftware/CorvusAI/ci.yml?branch=main&style=plastic&label=CI&logo=github&logoColor=white)](https://github.com/CorvusSoftware/CorvusAI/actions)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white&style=plastic)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?logo=pytorch&logoColor=white&style=plastic)](https://pytorch.org/)
[![Licence AGPL-3.0](https://img.shields.io/badge/Licence-AGPL--3.0-663399?style=plastic&logo=gnu&logoColor=white)](LICENSE)

[Getting started](#getting-started) · [Architecture](docs/architecture.md) · [Training](docs/training.md) · [Contributing](CONTRIBUTING.md)

</div>

---

## Why CorvusAI?

**CorvusAI** is a research project exploring a language model built **from random initialization**, without relying on pretrained model weights or a hosted inference API.

The project has two long-term goals:

- **CorvusBrain** — the language model, tokenizer, training pipeline, and inference runtime.
- **CorvusCode** — a future coding assistant powered by CorvusBrain.

The repository is experimental. Its small reference models and sample datasets are intended for learning, reproducible experiments, and incremental development—not reliable code generation or production use.

## Getting started

**Requirements:** Python 3.11 or newer and a compatible PyTorch installation. For NVIDIA GPU acceleration, install a CUDA-enabled PyTorch build using the [official PyTorch instructions](https://pytorch.org/get-started/locally/).

```powershell
git clone https://github.com/CorvusSoftware/CorvusAI.git
cd CorvusAI
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
```

A minimal CPU training experiment:

```powershell
.\.venv\Scripts\python.exe -m corvusai.training.trainer --corpus data/sample.py --output checkpoints/tiny --steps 20 --device cpu
```

To generate text from a local checkpoint:

```powershell
.\.venv\Scripts\python.exe -m corvusai.inference.generate --checkpoint checkpoints/tiny/latest.pt --prompt "def add(a, b):" --device cpu
```

GPU users can choose `--device cuda` after installing and verifying CUDA-enabled PyTorch. Generated text is untrusted: review it and never execute it without appropriate isolation.

## Interactive local CorvusBrain (V1.0)

Once you have trained a trusted local BPE checkpoint, start the persistent
inference shell:

```powershell
.\.venv\Scripts\python.exe -m corvusai.inference.brain --checkpoint checkpoints/bpe/latest.pt --tokenizer checkpoints/tokenizer.json --device cpu
```

See [V1.0 quickstart](docs/v1.0.md) for CUDA and one-shot mode, or
[V1.2 integration](docs/modern-integration.md) to train and run the modern
RoPE/GQA architecture with KV caching.
This is a research interface, not a pretrained or production-ready assistant.

## Documentation

The README is deliberately **version-neutral**. Detailed features, experiments, and evolving commands belong in the documentation:

| Topic | Documentation |
| --- | --- |
| Design and roadmap | [Architecture](docs/architecture.md) |
| Training and checkpoints | [Training guide](docs/training.md) |
| Tokenization and data preparation | [Tokenizer guide](docs/tokenizer.md) |
| Local text generation | [Inference guide](docs/inference.md) |
| BPE training and generation | [BPE integration](docs/bpe-training.md) |
| Modern Transformer architecture | [V1.1 architecture](docs/modern-architecture.md) |
| Modern BPE training and KV-cache inference | [V1.2 integration](docs/modern-integration.md) |
| Dataset preparation and licensing | [Dataset guide](docs/dataset.md) |
| Offline code evaluation | [Code evaluation](docs/code-evaluation.md) |

## Community and security

Contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request, follow the [Code of Conduct](CODE_OF_CONDUCT.md), and report vulnerabilities according to [SECURITY.md](SECURITY.md).

## License

CorvusAI is licensed under the **GNU Affero General Public License v3.0 only** ([AGPL-3.0-only](LICENSE)). Training data, third-party dependencies, and model artifacts may have their own license obligations; verify them independently.

---

<div align="center">
  <sub>Built openly by <a href="https://github.com/CorvusSoftware">CorvusSoftware</a> · Research-first · No pretrained model weights required</sub>
</div>
