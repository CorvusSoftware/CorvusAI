# CorvusAI architecture

## Product boundaries

- **CorvusBrain**: from-scratch model research and training; no pretrained weights.
- **CorvusAgent** (future): controlled repository exploration, edits, test execution and verification.
- **CorvusCode**: coding assistant product built from these components.

## V0.1 scope

A small, CPU/GPU-compatible causal Transformer, reversible byte tokenizer and automated tests. The initial weights are random. No meaningful coding ability is claimed.

## Research roadmap

1. Reproducible pretraining data pipeline with license review, train/validation split and deduplication.
2. Training CLI with checkpoint resume, logging and CUDA memory measurements on RTX 4060 8 GB.
3. Benchmarks for syntax and held-out functional tests.
4. Experimental SSM/hybrid architectures evaluated against the Transformer baseline.
5. Sandboxed agent with explicit approval for destructive actions.

Model checkpoints and raw datasets are excluded from Git. Never execute generated code outside a sandbox.
