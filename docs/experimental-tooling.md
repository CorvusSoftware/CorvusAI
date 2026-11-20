# Experimental V1.3–V1.6 foundations

These modules are incremental prototypes, **not** completed milestones or a production-ready coding agent. They require Python 3.11+ and the project's development dependencies.

## V1.3 — Dataset manifest audit

```bash
python -m corvusai.training.audit --manifest data/manifest.json
```

Reports missing files, duplicate paths and identical SHA-256 values across train/validation splits. Exit status is nonzero on a failed audit. It does **not** prove license validity, detect all secrets, or identify semantic near-duplicates. The existing `prepare` command remains the source of manifests.

## V1.4 — Local inference profiling

```bash
python -m corvusai.training.profile --checkpoint checkpoints/tiny/latest.pt --device cpu --tokens 32
```

Measures model loading time and requested-token generation throughput. With `--device cuda`, reports peak CUDA allocated memory. Results depend on hardware and checkpoint and are not claims about model quality or training efficiency.

## V1.5 — CorvusAgent read-only exploration

```bash
python -m corvusai.agent --repo . --query "def train"
```

Searches bounded Python source files without executing code or modifying the repository. This is not yet a patching, planning or sandboxed execution agent.

## V1.6 — CorvusCode experimental CLI

```bash
python -m corvusai.code search "def train" --repo .
python -m corvusai.code complete "def add(a, b):" --checkpoint checkpoints/tiny/latest.pt
```

The `complete` command requires a **trusted, locally trained** checkpoint. BPE checkpoints also require `--tokenizer`. No pretrained weights or hosted API are used. Outputs are untrusted and must not be executed without isolation.

## Verification

```bash
ruff check .
python -m pytest -q
```

The commands above must be run in a configured development environment. Adding the modules and tests to GitHub does not imply they have passed CI.

## Remaining milestones

- Functional code benchmarks in a secure sandbox, with pass@k and held-out datasets.
- Audited dataset provenance, repository-level splitting, stronger deduplication and licensing checks.
- Measured training throughput, memory and optimization on target hardware.
- Agent planning, permissioned patches, sandboxed tests, rollback and a usable CorvusCode workflow.
