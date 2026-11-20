"""Experimental CorvusCode CLI: inspect a repo or query a trusted local model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from corvusai.agent import explore


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimental local CorvusCode CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    search = sub.add_parser("search", help="Read-only search in Python repository files")
    search.add_argument("query")
    search.add_argument("--repo", type=Path, default=Path("."))
    search.add_argument("--limit", type=int, default=20)
    complete = sub.add_parser("complete", help="Generate with a trusted local checkpoint")
    complete.add_argument("prompt")
    complete.add_argument("--checkpoint", type=Path, required=True)
    complete.add_argument("--tokenizer", type=Path)
    complete.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    complete.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args()
    if args.command == "search":
        print(json.dumps(explore(args.repo, args.query, limit=args.limit), indent=2))
    else:
        from corvusai.inference.brain import BrainSession

        session = BrainSession(args.checkpoint, tokenizer_path=args.tokenizer,
                               device=args.device)
        print(session.complete(args.prompt, max_new_tokens=args.max_new_tokens))


if __name__ == "__main__":
    main()
