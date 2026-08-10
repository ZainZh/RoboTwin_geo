#!/usr/bin/env python3
"""Create or verify an immutable checksum inventory for Hammer Stage42 results."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_INPUTS = (
    "outputs/paper_revision/hammer_stage42_independent_local4090_seed20260805_e4000_split424242_20260806",
    "outputs/paper_revision/hammer_stage42_independent_remote6000ada_seed20260806_e4000_split424242_20260806",
    "outputs/paper_revision/hammer_stage42_independent_local4090_seed20260807_e4000_split424242_20260806",
    "outputs/paper_revision/hammer_stage42_independent_test_evaluation_e1750_v1",
    "outputs/paper_revision/hammer_stage42_independent_test_summary_e1750_v1",
    "outputs/paper_revision/hammer_stage42_e1750_15run_manifest_v1",
    "corl_version0/HAMMER_STAGE42_FINAL_RESULTS_20260807_zh.md",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(repo: Path, *args: str) -> str | None:
    result = subprocess.run(
        ("git", *args), cwd=repo, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def enumerate_files(repo: Path, inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for name in inputs:
        path = repo / name
        if not path.exists():
            raise FileNotFoundError(f"required result artifact is missing: {name}")
        if path.is_file():
            files.append(path)
        else:
            files.extend(
                item for item in path.rglob("*") if item.is_file() and not item.name.endswith(".lock")
            )
    return sorted(set(files), key=lambda item: item.relative_to(repo).as_posix())


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def create(repo: Path, output: Path, inputs: list[str]) -> None:
    files = enumerate_files(repo, inputs)
    records = [
        {
            "path": path.relative_to(repo).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in files
    ]
    payload = {
        "schema": "hammer_stage42_frozen_results_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(repo),
        "git_head": git_value(repo, "rev-parse", "HEAD"),
        "git_branch": git_value(repo, "branch", "--show-current"),
        "protocol": {
            "training_runs": 15,
            "training_seeds": [20260805, 20260806, 20260807],
            "target_epoch": 1750,
            "split_seed": 424242,
            "held_out_evaluations": 30,
            "paper_claim_eligible": True,
        },
        "inputs": inputs,
        "file_count": len(records),
        "total_size_bytes": sum(record["size_bytes"] for record in records),
        "files": records,
    }
    atomic_json(output, payload)
    print(
        f"frozen {payload['file_count']} files, {payload['total_size_bytes']} bytes -> {output}"
    )


def verify(repo: Path, manifest: Path) -> None:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    errors: list[str] = []
    for record in payload["files"]:
        path = repo / record["path"]
        if not path.is_file():
            errors.append(f"missing: {record['path']}")
            continue
        size = path.stat().st_size
        if size != record["size_bytes"]:
            errors.append(f"size mismatch: {record['path']}")
            continue
        actual = sha256(path)
        if actual != record["sha256"]:
            errors.append(f"sha256 mismatch: {record['path']}")
    if errors:
        raise SystemExit("verification failed:\n" + "\n".join(errors))
    print(f"verified {len(payload['files'])} files against {manifest}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/paper_revision/hammer_stage42_frozen_results_20260807_v1/preservation_manifest.json"
        ),
    )
    parser.add_argument("--input", action="append", dest="inputs")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output if args.output.is_absolute() else repo / args.output
    if args.verify:
        verify(repo, output)
    else:
        create(repo, output, args.inputs or list(DEFAULT_INPUTS))


if __name__ == "__main__":
    main()
