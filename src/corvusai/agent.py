"""Read-only repository exploration for the experimental CorvusAgent MVP.

No generated code is executed and no repository files are modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def explore(root: Path, query: str, *, limit: int = 20) -> list[dict[str, object]]:
    if not query.strip() or limit < 1:
        raise ValueError("query must be nonempty and limit positive")
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("root must be a directory")
    results: list[dict[str, object]] = []
    ignored = {".git", ".venv", "venv", "node_modules", "__pycache__", "checkpoints"}
    for path in sorted(root.rglob("*.py")):
        if any(part in ignored for part in path.relative_to(root).parts):
            continue
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 1_000_000:
            continue
        if not path.resolve().is_relative_to(root):
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            continue
        for number, line in enumerate(lines, 1):
            if query.casefold() in line.casefold():
                results.append({"path": path.relative_to(root).as_posix(),
                                "line": number, "text": line[:300]})
                if len(results) >= limit:
                    return results
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only CorvusAgent repository search")
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--query", required=True)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(explore(args.repo, args.query, limit=args.limit), indent=2))


if __name__ == "__main__":
    main()
