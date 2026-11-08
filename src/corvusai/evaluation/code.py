"""Offline, non-executing evaluation of Python code completions."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

from corvusai.inference.generate import generate


def _syntax_valid(source: str) -> bool:
    try:
        ast.parse(source)
    except (SyntaxError, ValueError):
        return False
    return True


def evaluate_code(
    benchmark: Path,
    checkpoint: Path,
    output: Path,
    *,
    tokenizer_path: Path | None = None,
    max_new_tokens: int = 80,
    seed: int = 42,
    temperature: float = 0.8,
    device: str = "cpu",
) -> dict:
    """Evaluate trusted local checkpoints without executing generated code.

    Benchmark JSONL records: {"id": "unique", "prompt": "...",
    "reference": "optional exact expected completion"}. Exact match is
    reported only for examples with references.
    """
    if max_new_tokens < 0 or temperature <= 0:
        raise ValueError("Invalid generation parameters")
    tasks = []
    seen: set[str] = set()
    for line_number, line in enumerate(benchmark.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise TypeError(f"Benchmark line {line_number} must be an object")
        identifier = record.get("id")
        prompt = record.get("prompt")
        reference = record.get("reference")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError(f"Invalid or duplicate task id on line {line_number}")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"Invalid prompt on line {line_number}")
        if reference is not None and not isinstance(reference, str):
            raise TypeError(f"Reference must be a string on line {line_number}")
        seen.add(identifier)
        tasks.append((identifier, prompt, reference))
    if not tasks:
        raise ValueError("Benchmark has no tasks")

    results = []
    for index, (identifier, prompt, reference) in enumerate(tasks):
        full_text = generate(
            checkpoint, prompt, max_new_tokens=max_new_tokens,
            temperature=temperature, seed=seed + index, device=device,
            tokenizer_path=tokenizer_path,
        )
        if not full_text.startswith(prompt):
            raise ValueError("Generator output does not preserve the prompt")
        completion = full_text[len(prompt):]
        results.append({
            "id": identifier,
            "syntax_valid": _syntax_valid(full_text),
            "exact_match": completion == reference if reference is not None else None,
            "completion": completion,
        })
    matched = [result["exact_match"] for result in results
               if result["exact_match"] is not None]
    report = {
        "format": "corvus-code-evaluation-v1",
        "benchmark_sha256": hashlib.sha256(benchmark.read_bytes()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "tokenizer_sha256": (hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
                             if tokenizer_path is not None else None),
        "parameters": {"max_new_tokens": max_new_tokens, "temperature": temperature,
                       "seed": seed, "device": device},
        "tasks": len(results),
        "syntax_valid_rate": sum(result["syntax_valid"] for result in results) / len(results),
        "exact_match_rate": sum(matched) / len(matched) if matched else None,
        "reference_tasks": len(matched),
        "results": results,
        "notes": "Syntax validity and exact-match are proxy metrics, not functional correctness. Generated code is never executed.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Python completions without executing them")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    report = evaluate_code(
        args.benchmark, args.checkpoint, args.output, tokenizer_path=args.tokenizer,
        max_new_tokens=args.max_new_tokens, seed=args.seed,
        temperature=args.temperature, device=args.device,
    )
    print(json.dumps({key: report[key] for key in
                      ("tasks", "syntax_valid_rate", "exact_match_rate", "reference_tasks")}))


if __name__ == "__main__":
    main()
