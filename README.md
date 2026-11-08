# CorvusAI

Open-source research into **CorvusCode**, an autonomous coding assistant, and
**CorvusBrain**, its neural model trained from random initialization.

> Experimental: this is a tiny Transformer baseline and training smoke test,
> not a production coding assistant.

## Quick start (Windows / RTX 4060)

Install Python 3.11 and a CUDA-enabled PyTorch build following
[PyTorch's official installer](https://pytorch.org/get-started/locally/).
Then, from the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m corvusai.training.trainer --corpus data/sample.py --steps 20 --device cuda
```

Resume with `--steps 40 --resume` to train to step 40.

## Project documentation

- [Architecture and roadmap](docs/architecture.md)
- [Training guide](docs/training.md)
- [V0.3 tokenizer and dataset preparation](docs/tokenizer.md)
- [V0.4 local inference](docs/inference.md)
- [Contributor guide](CONTRIBUTING.md)
- [Code of conduct](CODE_OF_CONDUCT.md)
- [Security policy](SECURITY.md)

## License

Licensed under the **GNU Affero General Public License v3.0 only**
([AGPL-3.0-only](LICENSE)). Contributions are accepted under the same license.
Third-party training datasets and model artifacts may have separate licenses;
review those before use.
