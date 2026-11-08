"""Prepare an auditable, deterministic, filtered Python code dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tokenize
from io import BytesIO
from pathlib import Path

# Conservative high-confidence patterns. This is not a complete secret scanner.
_SECRET_PATTERNS = (
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    re.compile(rb"(?i)(?:api[_-]?key|secret[_-]?key|access[_-]?token|password)\s*[:=]\s*['\"][^'\"]{8,}['\"]"),
    re.compile(rb"gh[pousr]_[A-Za-z0-9_]{20,}"),
    re.compile(rb"AKIA[0-9A-Z]{16}"),
)
_ALLOWED_LICENSES = frozenset({"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause",
                               "ISC", "CC0-1.0", "Unlicense", "AGPL-3.0-only"})


def _normalized_digest(data: bytes) -> str:
    """Token fingerprint ignoring whitespace and comments, preserving code tokens."""
    try:
        tokens = tokenize.tokenize(BytesIO(data).readline)
        parts = [(token.type, token.string) for token in tokens
                 if token.type not in (tokenize.ENCODING, tokenize.COMMENT,
                                       tokenize.NL, tokenize.NEWLINE,
                                       tokenize.INDENT, tokenize.DEDENT,
                                       tokenize.ENDMARKER)]
        return hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()
    except (SyntaxError, tokenize.TokenError, UnicodeError, IndentationError):
        return hashlib.sha256(data).hexdigest()


def prepare(
    directory: Path,
    output: Path,
    validation_fraction: float = 0.2,
    *,
    licenses: Path | None = None,
    max_bytes: int = 1_000_000,
    min_bytes: int = 1,
) -> dict:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if min_bytes < 1 or max_bytes < min_bytes:
        raise ValueError("Invalid file size limits")
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")
    if licenses is None:
        raise ValueError("An explicit --licenses JSON file is required for V0.7 preparation")
    declarations = json.loads(licenses.read_text(encoding="utf-8"))
    if not isinstance(declarations, dict):
        raise ValueError("License declarations must map relative paths to SPDX identifiers")
    paths = sorted(p for p in directory.rglob("*.py") if p.is_file() and not p.is_symlink())
    seen_exact: set[str] = set()
    seen_normalized: set[str] = set()
    records: list[dict[str, str | int]] = []
    rejected: dict[str, int] = {}
    def reject(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1

    for path in paths:
        relative = path.relative_to(directory).as_posix()
        license_id = declarations.get(relative)
        if license_id not in _ALLOWED_LICENSES:
            reject("license_missing_or_not_allowlisted")
            continue
        size = path.stat().st_size
        if not min_bytes <= size <= max_bytes:
            reject("file_size")
            continue
        content = path.read_bytes()
        if b"\x00" in content:
            reject("binary")
            continue
        try:
            content.decode("utf-8")
        except UnicodeDecodeError:
            reject("non_utf8")
            continue
        if any(pattern.search(content) for pattern in _SECRET_PATTERNS):
            reject("potential_secret")
            continue
        digest = hashlib.sha256(content).hexdigest()
        normalized = _normalized_digest(content)
        if digest in seen_exact or normalized in seen_normalized:
            reject("duplicate_or_near_duplicate")
            continue
        seen_exact.add(digest)
        seen_normalized.add(normalized)
        records.append({"path": relative, "sha256": digest,
                        "bytes": len(content), "license": license_id})
    if len(records) < 2:
        raise ValueError("At least two accepted unique Python files are required")
    records.sort(key=lambda record: hashlib.sha256(str(record["path"]).encode()).hexdigest())
    validation_count = max(1, min(len(records) - 1, round(len(records) * validation_fraction)))
    manifest = {
        "format": "corvus-dataset-manifest-v2",
        "root": str(directory.resolve()),
        "train": records[validation_count:],
        "validation": records[:validation_count],
        "rejected_counts": rejected,
        "policy": {"min_bytes": min_bytes, "max_bytes": max_bytes,
                   "licenses_allowlist": sorted(_ALLOWED_LICENSES)},
        "notes": "SPDX IDs are user declarations, not verified licenses. Secret and near-duplicate filtering is heuristic.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a filtered CorvusAI Python dataset")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--licenses", type=Path, required=True,
                        help="JSON mapping relative Python paths to reviewed SPDX IDs")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--min-bytes", type=int, default=1)
    parser.add_argument("--max-bytes", type=int, default=1_000_000)
    args = parser.parse_args()
    manifest = prepare(args.input, args.output, args.validation_fraction,
                       licenses=args.licenses, min_bytes=args.min_bytes,
                       max_bytes=args.max_bytes)
    print(json.dumps({"train_files": len(manifest["train"]),
                      "validation_files": len(manifest["validation"]),
                      "rejected_counts": manifest["rejected_counts"]}))


if __name__ == "__main__":
    main()
