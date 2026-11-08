# CorvusBrain V0.7 — Curated code datasets

This tool builds a **deterministic, filtered train/validation manifest**
from local Python files. It does not download or redistribute code.

## Review licenses first

Create a JSON file mapping **every approved relative file path** to its
manually reviewed SPDX license identifier:

```json
{
  "project_a/module.py": "MIT",
  "project_b/util.py": "Apache-2.0"
}
```

Only explicitly listed paths with allowlisted identifiers are admitted.
**A declared identifier is not proof of permission.** Verify the actual
license, attribution requirements, provenance, and suitability for training.
The policy is intentionally conservative and excludes unreviewed files.

## Build a manifest

```powershell
python -m corvusai.training.prepare --input data --licenses data-licenses.json --output checkpoints/dataset-manifest.json --validation-fraction 0.2
```

The JSON contains file paths relative to the source root, SHA-256 hashes,
byte counts, declared license identifiers, disjoint train/validation lists,
and aggregate rejection counts. Splits are deterministic by hashed file path.
Exact duplicates and simple token-equivalent near-duplicates are removed
**before** splitting.

Files are excluded for missing/disallowed license declarations, size bounds,
non-UTF-8 or binary data, and selected high-confidence secret patterns.
The manifest intentionally does not include rejected file contents.

## Limitations and security

- Secret filtering is heuristic, **not** a substitute for dedicated scanning
  and human review. Do not upload raw unreviewed corpora to GitHub.
- Near-duplicate detection ignores comments and whitespace, not semantic
  similarity; related files may still leak between train and validation.
- No repository-level grouping, legal compliance certification, or license
  compatibility guarantee is provided.
- V0.7 produces a manifest for auditing. The V0.6 BPE trainer still accepts
  a file or directory and **does not consume this manifest**. Only feed it a
  separately curated training directory, never an unfiltered root containing
  validation files. Manifest-driven training and held-out evaluation are
  future integration work.
