# CorvusBrain V0.9 — Python code evaluation

V0.9 adds a reproducible **offline, non-executing** benchmark for locally
trained CorvusBrain checkpoints. It does not claim HumanEval, pass@k,
functional correctness, or production code quality.

## Benchmark format

Create a UTF-8 JSONL file with unique task IDs, nonempty Python prompts,
and optional **exact** expected completions:

```jsonl
{"id":"assignment","prompt":"answer = ","reference":"42\n"}
{"id":"function","prompt":"def square(x):\n    ","reference":"return x * x\n"}
```

Use a held-out benchmark. Never train on evaluation prompts or references.

## Run

```powershell
python -m corvusai.evaluation.code --benchmark benchmarks/python.jsonl --checkpoint checkpoints/bpe/latest.pt --tokenizer checkpoints/tokenizer.json --output reports/v0.9.json --max-new-tokens 80 --seed 42 --device cpu
```

For byte-level checkpoints, omit `--tokenizer`. The evaluation uses the
existing generation path, checks checkpoint/tokenizer compatibility, and
writes a JSON report containing hashes, parameters, per-task completions,
and aggregate metrics.

**Metrics:** `syntax_valid_rate` measures whether Python can parse the
entire prompt plus completion; `exact_match_rate` measures literal
completion equality only for tasks with references. Neither measures
semantic correctness or whether tests would pass.

**Security:** generated code is never executed or imported. Do not run
untrusted generated code without isolation. Review reports before sharing:
completions may contain sensitive or inappropriate content.

**Reproducibility:** fixed task order, per-task seeds and file SHA-256
fingerprints. CPU/GPU numerical differences and changes in PyTorch versions
can still affect sampling.
