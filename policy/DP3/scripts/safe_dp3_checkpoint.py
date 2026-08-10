"""Safe, weights-only DP3 checkpoint helpers.

The legacy workspace checkpoint contains arbitrary pickle/dill objects.  Paper
experiments should export this simpler tensor-and-primitive payload so it can be
loaded with ``torch.load(..., weights_only=True)``.
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping

import torch


FORMAT_NAME = "dp3_weights_only_v1"
SUPPORTED_PRECISIONS = {"fp32", "bf16"}


def _state_dict_to_cpu(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
        for key, value in state_dict.items()
    }


def save_weights_only_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    ema_model: torch.nn.Module | None,
    metadata: Mapping[str, Any],
) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    payload = {
        "format": FORMAT_NAME,
        "metadata": dict(metadata),
        "model": _state_dict_to_cpu(model.state_dict()),
        "ema_model": None if ema_model is None else _state_dict_to_cpu(ema_model.state_dict()),
    }
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        del payload
        # Re-open with the restricted loader before publishing. mmap avoids a
        # second full in-memory copy for multi-gigabyte DP3 checkpoints.
        validated = load_weights_only_checkpoint(temporary, mmap=True)
        del validated
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_weights_only_checkpoint(path: str | Path, *, mmap: bool = True) -> dict[str, Any]:
    payload = torch.load(
        Path(path).expanduser().resolve(),
        map_location="cpu",
        mmap=mmap,
        weights_only=True,
    )
    if not isinstance(payload, dict) or payload.get("format") != FORMAT_NAME:
        raise ValueError(f"Not a supported safe DP3 checkpoint: {path}")
    if not isinstance(payload.get("metadata"), dict):
        raise ValueError(f"Safe DP3 checkpoint has no metadata mapping: {path}")
    if not isinstance(payload.get("model"), dict):
        raise ValueError(f"Safe DP3 checkpoint has no model state_dict: {path}")
    if payload.get("ema_model") is not None and not isinstance(payload.get("ema_model"), dict):
        raise ValueError(f"Safe DP3 checkpoint has an invalid ema_model state_dict: {path}")
    return payload


def validate_weights_only_checkpoint(
    path: str | Path,
    *,
    expected_metadata: Mapping[str, Any] | None = None,
    mmap: bool = True,
) -> dict[str, Any]:
    payload = load_weights_only_checkpoint(path, mmap=mmap)
    metadata = payload["metadata"]
    for key, expected_value in dict(expected_metadata or {}).items():
        actual_value = metadata.get(key)
        if actual_value != expected_value:
            raise ValueError(
                f"Safe DP3 checkpoint metadata mismatch for {key!r}: "
                f"expected {expected_value!r}, got {actual_value!r}"
            )
    return payload


def validate_policy_checkpoint_metadata(
    payload: Mapping[str, Any],
    *,
    expected_task_name: str,
    expected_use_ema: bool,
    expected_shape_meta: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate metadata that determines a deploy-time DP3 architecture."""
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Safe DP3 checkpoint has no metadata mapping")
    required = {"task_name", "use_ema", "shape_meta"}
    missing = sorted(required.difference(metadata))
    if missing:
        raise ValueError(f"Safe DP3 checkpoint metadata is missing {missing}")
    if not isinstance(metadata["use_ema"], bool):
        raise ValueError("Safe DP3 checkpoint metadata 'use_ema' must be a bool")

    expected = {
        "task_name": str(expected_task_name),
        "use_ema": bool(expected_use_ema),
        "shape_meta": dict(expected_shape_meta),
    }
    for key, expected_value in expected.items():
        actual_value = metadata[key]
        if actual_value != expected_value:
            raise ValueError(
                f"Safe DP3 checkpoint policy metadata mismatch for {key!r}: "
                f"expected {expected_value!r}, got {actual_value!r}"
            )
    if metadata["use_ema"] and payload.get("ema_model") is None:
        raise ValueError(
            "Safe DP3 checkpoint metadata requests EMA policy but ema_model is absent"
        )
    return metadata


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a safe DP3 weights-only checkpoint.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--expected-epoch", type=int)
    parser.add_argument("--expected-seed", type=int)
    parser.add_argument("--expected-task-name")
    parser.add_argument("--expected-zarr-path")
    parser.add_argument("--expected-precision", choices=sorted(SUPPORTED_PRECISIONS))
    parser.add_argument("--expected-allow-tf32", choices=("true", "false"))
    args = parser.parse_args(argv)
    expected_metadata = {
        key: value
        for key, value in {
            "epoch": args.expected_epoch,
            "seed": args.expected_seed,
            "task_name": args.expected_task_name,
            "zarr_path": args.expected_zarr_path,
            "precision": args.expected_precision,
            "allow_tf32": (
                None if args.expected_allow_tf32 is None else args.expected_allow_tf32 == "true"
            ),
        }.items()
        if value is not None
    }
    try:
        validate_weights_only_checkpoint(args.path, expected_metadata=expected_metadata)
    except Exception as exc:
        print(f"invalid safe DP3 checkpoint: {exc}", file=sys.stderr)
        return 1
    print(f"valid safe DP3 checkpoint: {args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
