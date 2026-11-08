"""Autoregressive byte-level generation from a trusted local checkpoint."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from corvusai.models.transformer import CorvusTransformer, ModelConfig
from corvusai.tokenization.bpe import ByteBPE


def generate(
    checkpoint: Path,
    prompt: str,
    *,
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    seed: int = 42,
    device: str | None = None,
    tokenizer_path: Path | None = None,
) -> str:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    # Only load checkpoints from trusted sources.
    state = torch.load(checkpoint, map_location=target, weights_only=True)
    config = ModelConfig(**state["config"])
    tokenizer = None
    if state.get("format") == "corvus-bpe-checkpoint-v1":
        if tokenizer_path is None:
            raise ValueError("BPE checkpoint requires --tokenizer")
        digest = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
        if digest != state["tokenizer_sha256"]:
            raise ValueError("Tokenizer SHA-256 does not match checkpoint")
        tokenizer = ByteBPE.load(tokenizer_path)
        if tokenizer.vocab_size != config.vocab_size:
            raise ValueError("Tokenizer vocabulary does not match model")
    elif config.vocab_size != 256:
        raise ValueError("Legacy checkpoints require byte vocabulary of size 256")
    model = CorvusTransformer(config).to(target)
    model.load_state_dict(state["model"])
    model.eval()
    generator = torch.Generator(device=target).manual_seed(seed)
    sequence = tokenizer.encode(prompt) if tokenizer else list(prompt.encode("utf-8"))
    if not sequence:
        raise ValueError("Prompt must not be empty")
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            context = torch.tensor([sequence[-config.context_length:]], dtype=torch.long,
                                   device=target)
            logits = model(context)[0, -1] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = int(torch.multinomial(probs, 1, generator=generator))
            sequence.append(next_token)
    result = tokenizer.decode_bytes(sequence) if tokenizer else bytes(sequence)
    return result.decode("utf-8", errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a CorvusAI checkpoint")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/tiny/latest.pt"))
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    args = parser.parse_args()
    print(generate(args.checkpoint, args.prompt, max_new_tokens=args.max_new_tokens,
                   temperature=args.temperature, seed=args.seed, device=args.device,
                   tokenizer_path=args.tokenizer))


if __name__ == "__main__":
    main()
