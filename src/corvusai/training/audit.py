"""Audit dataset manifests for leakage, source integrity and path safety."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def audit(manifest_path: Path) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = Path(manifest["root"]).resolve(strict=True)
    train = manifest["train"]
    validation = manifest["validation"]
    if not isinstance(train, list) or not isinstance(validation, list):
        raise TypeError("train and validation must be lists")

    def field(records: list[dict], key: str) -> set[str]:
        return {str(record[key]) for record in records}

    path_overlap = sorted(field(train, "path") & field(validation, "path"))
    hash_overlap = sorted(field(train, "sha256") & field(validation, "sha256"))
    missing: list[str] = []
    changed: list[str] = []
    unsafe: list[str] = []
    duplicate_paths: list[str] = []
    seen: set[str] = set()
    for record in train + validation:
        name = str(record["path"])
        relative = PurePosixPath(name)
        if (relative.is_absolute() or ".." in relative.parts or
                "\\" in name or not name or name == "."):
            unsafe.append(name)
            continue
        if name in seen:
            duplicate_paths.append(name)
        seen.add(name)
        candidate = root / name
        if candidate.is_symlink() or not candidate.resolve().is_relative_to(root):
            unsafe.append(name)
            continue
        if not candidate.is_file():
            missing.append(name)
            continue
        actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if actual != record["sha256"]:
            changed.append(name)
    return {
        "train_files": len(train), "validation_files": len(validation),
        "path_overlap": path_overlap, "sha256_overlap": hash_overlap,
        "duplicate_paths": sorted(set(duplicate_paths)),
        "missing_files": sorted(set(missing)),
        "changed_files": sorted(set(changed)),
        "unsafe_paths": sorted(set(unsafe)),
        "passed": not any((path_overlap, hash_overlap, duplicate_paths,
                           missing, changed, unsafe)),
        "note": "Verifies file hashes and split overlap; does not certify licenses, "
                "secret removal or semantic deduplication.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit CorvusAI dataset manifest")
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.manifest)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
