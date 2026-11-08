# Agent instructions — CorvusAI

These instructions apply to AI coding agents working anywhere in this repository.

## Mission

CorvusAI is an open-source, from-scratch research project for **CorvusBrain**
(the neural language model) and **CorvusCode** (the future coding assistant).
Preserve the project's independence from pretrained model weights and hosted
model APIs. Do not present experimental models as production-ready.

## Repository navigation

- `src/corvusai/models/` — decoder-only Transformer and model configuration.
- `src/corvusai/tokenization/` — byte and BPE tokenization.
- `src/corvusai/training/` — datasets, evaluation, and training entry points.
- `src/corvusai/inference/` — local checkpoint-based text generation.
- `tests/` — regression and smoke tests.
- `docs/` — architecture, training, tokenizer, and inference documentation.

Inspect the current files before changing interfaces or relying on this map.

## Engineering rules

1. Make the smallest change that satisfies the request; avoid unrelated refactors.
2. Preserve compatibility with existing checkpoints and CLIs unless a migration
   is explicitly documented and tested.
3. Keep experiments reproducible: explicit seeds, deterministic data handling
   where practical, and documented configuration.
4. Treat downloaded checkpoints, generated code, and training datasets as
   untrusted. Do not execute generated code or load arbitrary pickle payloads.
5. Do not introduce pretrained weights, undisclosed telemetry, hosted inference
   dependencies, or data without verified usage rights.
6. Never commit credentials, personal data, datasets of uncertain provenance,
   model checkpoints, or large generated artifacts.
7. Keep README.md version-neutral; put milestone-specific usage and status in
   `docs/`.
8. Preserve the repository's **AGPL-3.0-only** licensing and review dependency
   and dataset licenses.

## Validation

For Python changes, run the project checks where the environment allows:

```bash
python -m pip install -e '.[dev]'
ruff check .
python -m pytest -q
```

For GPU-specific changes, describe the hardware and CUDA environment actually
tested. If a command was not run, report it as unverified; do not claim CI passed
without checking its result.

## Pull requests and communication

- Prefer a focused feature branch and a pull request targeting `main`.
- Explain the problem, implementation, test evidence, and remaining limitations.
- Do not merge a pull request without explicit authorization.
- Report failures with their observed error messages and root cause when known.

## Agent entry points

This file is the **single source of truth** for repository-wide agent rules.
Claude Code should load it through [CLAUDE.md](CLAUDE.md). Other agents should
read this file directly.
