#!/usr/bin/env python3
"""Preserve and atomically vacate an interrupted semantic-field run."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Iterable

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def active_training_pids(run_name: str, proc_root: Path = Path("/proc")) -> list[int]:
    matches = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            tokens = [
                item.decode("utf-8", errors="replace")
                for item in (entry / "cmdline").read_bytes().split(b"\0")
                if item
            ]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if not any("train_semantic_field_loss_ablation_" in item for item in tokens):
            continue
        for index, token in enumerate(tokens[:-1]):
            if token == "--run-name" and tokens[index + 1] == run_name:
                matches.append(int(entry.name))
                break
    return sorted(matches)


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def inspect_checkpoint(run_dir: Path) -> dict[str, object]:
    config_path = run_dir / "config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    checkpoint = torch.load(
        run_dir / "last.pt", map_location="cpu", weights_only=True
    )
    if checkpoint.get("format") != "semantic_field_weights_only_v1":
        raise RuntimeError("last.pt is not the expected weights-only format")
    target_epoch = int(config["epochs"])
    epoch = int(checkpoint.get("epoch", -1))
    if epoch < 0 or epoch >= target_epoch:
        raise RuntimeError(
            f"interrupted checkpoint epoch must satisfy 0 <= {epoch} < {target_epoch}"
        )
    checkpoint_args = checkpoint.get("args")
    if not isinstance(checkpoint_args, dict):
        raise RuntimeError("last.pt is missing primitive args")
    for key in ("run_name", "seed", "epochs"):
        if checkpoint_args.get(key) != config.get(key):
            raise RuntimeError(
                f"checkpoint/config mismatch for {key}: "
                f"{checkpoint_args.get(key)!r} != {config.get(key)!r}"
            )
    return {
        "checkpoint_epoch": epoch,
        "global_step": int(checkpoint.get("global_step", -1)),
        "target_epoch": target_epoch,
        "config": config,
    }


def archive_run(
    output_root: Path,
    run_name: str,
    *,
    timestamp: str | None = None,
    proc_root: Path = Path("/proc"),
) -> Path:
    output_root = output_root.resolve()
    run_dir = output_root / run_name
    log_path = output_root / "logs" / f"{run_name}.log"
    inner_completion = run_dir / "completion.json"
    outer_completion = output_root / "completed" / f"{run_name}.complete"
    outer_manifest = output_root / "manifests" / f"{run_name}.sha256"
    lock_path = output_root / f".{run_name}.lock"

    if not run_dir.is_dir():
        raise RuntimeError(f"missing interrupted run directory: {run_dir}")
    if not log_path.is_file():
        raise RuntimeError(f"missing interrupted log: {log_path}")
    for forbidden in (inner_completion, outer_completion, outer_manifest):
        if forbidden.exists():
            raise RuntimeError(f"refusing to archive completed/finalized run: {forbidden}")
    required = ("config.json", "last.pt", "best.pt", "best_sem.pt")
    for name in required:
        if not (run_dir / name).is_file():
            raise RuntimeError(f"interrupted run is missing {run_dir / name}")

    active_before = active_training_pids(run_name, proc_root)
    if active_before:
        raise RuntimeError(f"active training PIDs for {run_name}: {active_before}")

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(
                lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
            )
        except BlockingIOError as error:
            raise RuntimeError(f"run lock is held: {lock_path}") from error

        active_after = active_training_pids(run_name, proc_root)
        if active_after:
            raise RuntimeError(
                f"training appeared while acquiring lock for {run_name}: {active_after}"
            )
        metadata = inspect_checkpoint(run_dir)
        file_hashes = {
            str(path.relative_to(run_dir)): sha256(path)
            for path in sorted(run_dir.rglob("*"))
            if path.is_file()
        }
        log_hash = sha256(log_path)

        archive_root = output_root / "interrupted_archive"
        archive_root.mkdir(parents=True, exist_ok=True)
        if timestamp is None:
            timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        if not re_safe_timestamp(timestamp):
            raise RuntimeError(f"unsafe archive timestamp: {timestamp!r}")
        archive_name = f"{run_name}.{timestamp}"
        destination = archive_root / archive_name
        staging = archive_root / f".{archive_name}.tmp-{os.getpid()}"
        if destination.exists() or staging.exists():
            raise RuntimeError(f"archive destination already exists: {destination}")
        staging.mkdir()

        manifest = {
            "format": "interrupted_semantic_run_archive_v1",
            "created_utc": timestamp,
            "source_output_root": str(output_root),
            "run_name": run_name,
            "reason": "stopped throughput profile; training_resume_supported=false",
            "checkpoint_epoch": metadata["checkpoint_epoch"],
            "global_step": metadata["global_step"],
            "target_epoch": metadata["target_epoch"],
            "config_sha256": file_hashes["config.json"],
            "run_files_sha256": file_hashes,
            "log_sha256": log_hash,
            "config": metadata["config"],
        }
        manifest_path = staging / "archive_manifest.json"
        with manifest_path.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())

        moved_run = staging / "run"
        moved_log = staging / "log.log"
        try:
            os.replace(run_dir, moved_run)
            os.replace(log_path, moved_log)
            fsync_directory(staging)
            os.replace(staging, destination)
            fsync_directory(archive_root)
        except Exception:
            # Roll back both user artifacts when possible; never delete them.
            if moved_log.exists() and not log_path.exists():
                os.replace(moved_log, log_path)
            if moved_run.exists() and not run_dir.exists():
                os.replace(moved_run, run_dir)
            raise

        for relative, expected in file_hashes.items():
            actual = sha256(destination / "run" / relative)
            if actual != expected:
                raise RuntimeError(f"post-archive SHA mismatch: {relative}")
        if sha256(destination / "log.log") != log_hash:
            raise RuntimeError("post-archive log SHA mismatch")
        return destination


def re_safe_timestamp(value: str) -> bool:
    return bool(value) and all(
        character.isdigit() or character in {"T", "Z"} for character in value
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    parser.add_argument("run_name")
    parser.add_argument("--timestamp")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination = archive_run(
        args.output_root, args.run_name, timestamp=args.timestamp
    )
    print(destination)


if __name__ == "__main__":
    main()
