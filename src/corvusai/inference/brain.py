"""Interactive local CorvusBrain shell with one-time checkpoint loading."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from corvusai.models.modern import ModernConfig, ModernCorvusTransformer
from corvusai.models.transformer import CorvusTransformer, ModelConfig
from corvusai.tokenization.bpe import ByteBPE


class BrainSession:
    """Keep a trusted local checkpoint loaded across multiple prompts."""

    def __init__(
        self,
        checkpoint: Path,
        *,
        tokenizer_path: Path | None = None,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        state = torch.load(checkpoint, map_location=self.device, weights_only=True)
        self.modern = state.get("format") == "corvus-modern-bpe-checkpoint-v1"
        self.config = (ModernConfig(**state["config"]) if self.modern
                       else ModelConfig(**state["config"]))
        self.tokenizer = None
        if state.get("format") in ("corvus-bpe-checkpoint-v1", "corvus-bpe-checkpoint-v2",
                                    "corvus-modern-bpe-checkpoint-v1"):
            if tokenizer_path is None:
                raise ValueError("BPE checkpoint requires --tokenizer")
            digest = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
            if digest != state["tokenizer_sha256"]:
                raise ValueError("Tokenizer SHA-256 does not match checkpoint")
            self.tokenizer = ByteBPE.load(tokenizer_path)
            if self.tokenizer.vocab_size != self.config.vocab_size:
                raise ValueError("Tokenizer vocabulary does not match checkpoint")
        elif self.config.vocab_size != 256:
            raise ValueError("Legacy checkpoint must use the 256-byte vocabulary")
        self.model = (ModernCorvusTransformer(self.config) if self.modern
                      else CorvusTransformer(self.config)).to(self.device)
        self.model.load_state_dict(state["model"])
        self.model.eval()

    def complete(
        self, prompt: str, *, max_new_tokens: int = 80,
        temperature: float = 0.8, seed: int = 42,
    ) -> str:
        if not prompt:
            raise ValueError("Prompt must not be empty")
        if max_new_tokens < 0 or temperature <= 0:
            raise ValueError("Invalid generation parameters")
        sequence = (self.tokenizer.encode(prompt) if self.tokenizer
                    else list(prompt.encode("utf-8")))
        generator = torch.Generator(device=self.device).manual_seed(seed)
        with torch.inference_mode():
            cache = None
            for _ in range(max_new_tokens):
                if self.modern:
                    # Rebuild the cache when the context window is full.
                    # RoPE positions are relative to the active context.
                    if cache is None or len(sequence) > self.config.context_length:
                        context = sequence[-self.config.context_length:]
                        logits, cache = self.model(
                            torch.tensor([context], dtype=torch.long, device=self.device),
                            use_cache=True,
                        )
                    else:
                        logits, cache = self.model(
                            torch.tensor([[sequence[-1]]], dtype=torch.long,
                                         device=self.device),
                            cache=cache, use_cache=True,
                        )
                    last_logits = logits[0, -1]
                else:
                    context = torch.tensor(
                        [sequence[-self.config.context_length:]],
                        dtype=torch.long, device=self.device,
                    )
                    last_logits = self.model(context)[0, -1]
                next_token = int(torch.multinomial(
                    torch.softmax(last_logits / temperature, dim=-1),
                    1, generator=generator,
                ))
                sequence.append(next_token)
                if self.modern and cache is not None:
                    # Cache covers all tokens except the newly sampled one.
                    # Next iteration can append only while there is space.
                    if cache[0][0].shape[2] >= self.config.context_length:
                        cache = None
        data = (self.tokenizer.decode_bytes(sequence) if self.tokenizer
                else bytes(sequence))
        return data.decode("utf-8", errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser(description="CorvusBrain interactive local inference")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--prompt", help="One-shot prompt; omit for interactive mode")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_new_tokens < 0 or args.temperature <= 0:
        parser.error("max-new-tokens must be nonnegative and temperature must be positive")
    session = BrainSession(args.checkpoint, tokenizer_path=args.tokenizer, device=args.device)
    if args.prompt is not None:
        print(session.complete(args.prompt, max_new_tokens=args.max_new_tokens,
                               temperature=args.temperature, seed=args.seed))
        return
    print("CorvusBrain local shell. Enter a Python prompt, or /quit to exit.")
    index = 0
    while True:
        try:
            prompt = input("corvus> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.strip() in ("/quit", "/exit"):
            break
        if not prompt:
            continue
        try:
            print(session.complete(prompt, max_new_tokens=args.max_new_tokens,
                                   temperature=args.temperature, seed=args.seed + index))
            index += 1
        except (ValueError, RuntimeError) as error:
            print(f"Generation error: {error}")


if __name__ == "__main__":
    main()
