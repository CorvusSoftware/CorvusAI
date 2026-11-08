# CorvusBrain V0.5 — BPE pretraining and inference

Train the tokenizer using only data you are permitted to use:

```powershell
.\.venv\Scripts\python.exe -m corvusai.tokenization.bpe --input data --output checkpoints/tokenizer.json --merges 64
.\.venv\Scripts\python.exe -m corvusai.training.bpe_trainer --corpus data/sample.py --tokenizer checkpoints/tokenizer.json --output checkpoints/bpe --steps 20 --context-length 32 --device cpu
.\.venv\Scripts\python.exe -m corvusai.inference.generate --checkpoint checkpoints/bpe/latest.pt --tokenizer checkpoints/tokenizer.json --prompt "def add(a, b):" --max-new-tokens 50 --device cpu
```

BPE checkpoints record the SHA-256 of the tokenizer JSON. Generation rejects
a tokenizer whose bytes do not match the training tokenizer. Older byte-level
checkpoints continue to work without a tokenizer argument.

This is a small CPU reference implementation; CUDA is supported by the training
CLI but has not been benchmarked on the RTX 4060. Resume, mixed precision, and
multi-file manifest loading are not yet implemented for the BPE trainer.
