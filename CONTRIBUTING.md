# Contributing to CorvusAI

Thanks for helping build reproducible coding-model research.

## Start here

1. Open an issue describing the change before substantial architectural work.
2. Fork the repository and create a focused feature branch.
3. Install Python 3.11+ and the project with `python -m pip install -e ".[dev]"`.
4. Run `ruff check .` and `python -m pytest -q`.
5. Open a pull request using the template and include reproducible evidence.

## Engineering standards

- Keep model architecture, training, inference and agent tools separated.
- Add tests for bug fixes and behavior changes.
- Document random seeds, hardware, dataset licenses and evaluation splits for experiments.
- Do not commit datasets, trained weights, secrets or personally identifying information.
- Never execute generated code outside an isolated environment.
- Avoid unrelated refactors and benchmark claims without measurements.

## Licensing

By submitting a contribution, you agree that it is licensed under GNU AGPL-3.0-only.
You must have the right to submit all code and data you contribute. No contributor
license agreement or copyright assignment is currently required.

## Code of conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Report sensitive security issues
privately using the process in [SECURITY.md](SECURITY.md).
