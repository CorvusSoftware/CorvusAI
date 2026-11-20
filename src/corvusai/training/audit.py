"""Audit a prepared dataset manifest for overlap and missing source files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def audit(manifest_path: Path) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = Path(manifest["root"]).resolve()
    train = manifest["train"]
    validation = manifest["validation"]
    def field(records: list[dict], key: str) -> set[str]:
        return {str(record[key]) for record in records}
    path_overlap = sorted(field(train, "path") & field(validation, "path"))
    hash_overlap = sorted(field(train, "sha256") & field(validation, "sha256"))
    missing = sorted(str(record["path"]) for record in train + validation
                     if not (root / str(record["path"])).is_file()
                     or not (root / str(record["path"])).resolve().is_relative_to(root))
    return {"train_files": len(train), "validation_files": len(validation),
            "path_overlap": path_overlap, "sha256_overlap": hash_overlap,
            "missing_files": missing,
            "passed": not (path_overlap or hash_overlap or missing),
            "note": "Exact SHA-256 overlap only; near-duplicate and license review remain necessary."}


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
