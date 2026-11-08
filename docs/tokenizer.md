# CorvusBrain V0.3: from-scratch byte BPE

The tokenizer starts with 256 byte tokens and learns deterministic pair merges
from local Python files. It uses no pretrained vocabulary or external model.

```powershell
.\.venv\Scripts\python.exe -m corvusai.training.prepare --input data --output checkpoints/dataset-manifest.json
.\.venv\Scripts\python.exe -m corvusai.tokenization.bpe --input data --output checkpoints/tokenizer.json --merges 256
.\.venv\Scripts\python.exe -m corvusai.tokenization.benchmark --tokenizer checkpoints/tokenizer.json --input data
```

Dataset preparation requires at least two unique nonempty `.py` files.
The manifest records paths and SHA-256 hashes; it removes **exact** duplicates
only. This is not a licensing audit or near-duplicate detection. Keep third-party
data out of Git until its usage rights have been reviewed.

**Important:** the current Transformer training loop still uses byte tokens
(vocabulary 256). The V0.3 BPE model is a separately trained and benchmarked
component; integrating its vocabulary into pretraining and checkpoint metadata
is a later milestone. The benchmark measures token compression and roundtrip
only, **not** functional code generation quality.

The simple reference implementation is intentionally not optimized for
multi-gigabyte corpora.
