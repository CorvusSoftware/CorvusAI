# CorvusBrain V0.4 — local inference

Train a small byte-level model first, then generate text:

```powershell
.\.venv\Scripts\python.exe -m corvusai.training.trainer --corpus data/sample.py --steps 20 --device cpu
.\.venv\Scripts\python.exe -m corvusai.inference.generate --checkpoint checkpoints/tiny/latest.pt --prompt "def add(a, b):" --max-new-tokens 80 --device cpu
```

Use `--device cuda` on a CUDA-enabled PyTorch installation. `--seed`
controls sampling and `--temperature` controls randomness. Generation
uses a sliding context window and byte-level tokens. UTF-8 output may contain
replacement characters when a sampled byte sequence is not valid UTF-8.

This tiny model trained on a toy corpus is **not** expected to write
correct programs. Do not execute generated code without review and sandboxing.

The BPE tokenizer developed in V0.3 is not yet wired to training or inference.
That integration requires a new checkpoint/tokenizer compatibility contract.
