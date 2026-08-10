#!/usr/bin/env python3
"""Validate and atomically manifest one formal RoboTwin policy run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from safe_dp3_checkpoint import (
    validate_policy_checkpoint_metadata,
    validate_weights_only_checkpoint,
)


HYDRA_FILES = ("config.yaml", "overrides.yaml", "hydra.yaml")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"required artifact does not exist: {resolved}")
    display_path = str(resolved)
    if relative_to is not None:
        try:
            display_path = str(resolved.relative_to(relative_to.expanduser().resolve()))
        except ValueError:
            pass
    return {
        "path": display_path,
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite immutable manifest: {destination}")
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-epoch", type=int, required=True)
    parser.add_argument("--expected-seed", type=int, required=True)
    parser.add_argument("--expected-task-name", required=True)
    parser.add_argument("--expected-zarr-path", required=True)
    parser.add_argument("--expected-precision", choices=("fp32", "bf16"), required=True)
    parser.add_argument("--expected-allow-tf32", choices=("true", "false"), required=True)
    parser.add_argument("--expected-obs-key", required=True)
    parser.add_argument("--expected-feature-dim", type=int, required=True)
    parser.add_argument("--hydra-dir", type=Path, required=True)
    parser.add_argument("--code-path", action="append", type=Path, default=[])
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--role", choices=("local", "remote", "local_receipt"), required=True)
    return parser.parse_args()


def build_expected_shape_meta(obs_key: str, feature_dim: int) -> tuple[dict, str | None]:
    normalized_key = str(obs_key).strip()
    shape_meta = {
        "obs": {
            "point_cloud": {"shape": [1024, 6], "type": "point_cloud"},
            "agent_pos": {"shape": [14], "type": "low_dim"},
        },
        "action": {"shape": [14]},
    }
    if normalized_key.lower() in {"", "none"}:
        if int(feature_dim) != 0:
            raise ValueError("Vanilla/no-extra-observation manifest requires feature_dim=0")
        return shape_meta, None
    if int(feature_dim) < 3:
        raise ValueError("point-cloud auxiliary observation requires feature_dim >= 3")
    shape_meta["obs"][normalized_key] = {
        "shape": [128, int(feature_dim)],
        "type": "point_cloud",
    }
    return shape_meta, normalized_key


def main() -> int:
    args = parse_args()
    expected_metadata = {
        "epoch": args.expected_epoch,
        "seed": args.expected_seed,
        "task_name": args.expected_task_name,
        "zarr_path": args.expected_zarr_path,
        "precision": args.expected_precision,
        "allow_tf32": args.expected_allow_tf32 == "true",
    }
    expected_shape_meta, expected_obs_key = build_expected_shape_meta(
        args.expected_obs_key, args.expected_feature_dim
    )
    payload = validate_weights_only_checkpoint(
        args.weights,
        expected_metadata=expected_metadata,
        mmap=True,
    )
    validate_policy_checkpoint_metadata(
        payload,
        expected_task_name=args.expected_task_name,
        expected_use_ema=True,
        expected_shape_meta=expected_shape_meta,
    )
    actual_metadata = {
        key: payload["metadata"].get(key)
        for key in (*expected_metadata, "use_ema", "shape_meta")
    }
    del payload

    run_dir = args.weights.expanduser().resolve().parent
    checkpoint = artifact_record(args.weights, relative_to=run_dir)
    hydra = {
        name: artifact_record(args.hydra_dir / name, relative_to=run_dir)
        for name in HYDRA_FILES
    }
    code = {
        path.name: artifact_record(path)
        for path in args.code_path
    }

    source_record = None
    if args.source_manifest is not None:
        source_path = args.source_manifest.expanduser().resolve()
        with source_path.open("r", encoding="utf-8") as handle:
            source = json.load(handle)
        if source.get("schema") != "robotwin_sim_policy_run_completion_v2":
            raise ValueError("source manifest schema mismatch")
        if source.get("status") != "complete" or source.get("role") != "remote":
            raise ValueError("source manifest status/role mismatch")
        source_checkpoint = source.get("checkpoint", {})
        for key in ("size_bytes", "sha256"):
            if source_checkpoint.get(key) != checkpoint[key]:
                raise ValueError(
                    f"source manifest checkpoint {key} mismatch: "
                    f"expected {source_checkpoint.get(key)!r}, got {checkpoint[key]!r}"
                )
        if source_checkpoint.get("path") != checkpoint["path"]:
            raise ValueError("source manifest checkpoint path mismatch")
        if source.get("expected_metadata") != expected_metadata:
            raise ValueError("source manifest expected_metadata mismatch")
        expected_policy = {
            "obs_key": expected_obs_key,
            "feature_dim": args.expected_feature_dim,
            "shape_meta": expected_shape_meta,
            "use_ema": True,
        }
        if source.get("expected_policy") != expected_policy:
            raise ValueError("source manifest expected_policy mismatch")
        if source.get("actual_metadata") != actual_metadata:
            raise ValueError("source manifest actual_metadata mismatch")
        source_hydra = source.get("hydra", {})
        for name, record in hydra.items():
            if name not in source_hydra:
                raise ValueError(f"source manifest is missing Hydra artifact {name}")
            for key in ("size_bytes", "sha256"):
                if source_hydra[name].get(key) != record[key]:
                    raise ValueError(f"source manifest Hydra {name} {key} mismatch")
        source_code = source.get("code", {})
        for name, record in code.items():
            if name not in source_code or source_code[name].get("sha256") != record["sha256"]:
                raise ValueError(f"source manifest code SHA mismatch for {name}")
        source_record = artifact_record(source_path)

    manifest = {
        "schema": "robotwin_sim_policy_run_completion_v2",
        "status": "complete",
        "role": args.role,
        "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": checkpoint,
        "expected_metadata": expected_metadata,
        "expected_policy": {
            "obs_key": expected_obs_key,
            "feature_dim": args.expected_feature_dim,
            "shape_meta": expected_shape_meta,
            "use_ema": True,
        },
        "actual_metadata": actual_metadata,
        "hydra": hydra,
        "code": code,
        "source_manifest": source_record,
    }
    write_json_atomic(args.manifest, manifest)
    print(json.dumps({"status": "complete", "manifest": str(args.manifest), "sha256": checkpoint["sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
