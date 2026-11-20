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


## Modern KV-cache validation

The modern Transformer validates the number of cache layers, the key/value tensor shapes and their device before applying attention. A valid cache layer has shape `[batch, kv_heads, cached_tokens, head_dim]`, where `head_dim = width / heads`. All layers must share the same cached-token length. Malformed caches raise `ValueError` rather than being passed to PyTorch attention operations. CPU regression tests cover valid cached-versus-full logits and malformed cache shapes.

This hardening does not increase the model's context window or guarantee a particular GPU attention kernel. The legacy checkpoint format and the opt-in `--architecture modern` selection remain unchanged.
