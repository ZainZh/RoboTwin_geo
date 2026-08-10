#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from build_semantic_policy_ablation_zarr import _copy_source_zarr, _infer_sidecar
from semantic_field_eval_utils import sha256_file
from utonia_projection_utils import (
    PROJECTION_FORMAT,
    build_frozen_orthogonal_projection,
    load_projection_artifact,
    project_utonia_cloud,
    save_projection_artifact,
)


def _replace_array(root, *, key: str, matrix: np.ndarray, batch_frames: int) -> dict:
    source = root[f"data/{key}"]
    if source.ndim != 3 or source.shape[-1] != 3 + matrix.shape[0]:
        raise ValueError(
            f"{key} must be [T,Q,{3 + matrix.shape[0]}], got {source.shape}"
        )
    source_shape = tuple(int(value) for value in source.shape)
    output_shape = (*source_shape[:-1], 3 + int(matrix.shape[1]))
    output_chunks = (*tuple(int(value) for value in source.chunks[:-1]), output_shape[-1])
    temporary_key = f"{key}__utonia128_tmp"
    if temporary_key in root["data"]:
        raise KeyError(f"temporary key already exists: {temporary_key}")
    output = root["data"].create_dataset(
        temporary_key,
        shape=output_shape,
        chunks=output_chunks,
        dtype=np.float32,
        compressor=source.compressor,
        overwrite=False,
    )
    for start in range(0, source_shape[0], int(batch_frames)):
        end = min(source_shape[0], start + int(batch_frames))
        values = np.asarray(source[start:end], dtype=np.float32)
        projected = project_utonia_cloud(values, matrix)
        if not np.array_equal(values[..., :3], projected[..., :3]):
            raise RuntimeError(f"query XYZ changed in frames {start}:{end}")
        output[start:end] = projected
        print(f"projected {end} / {source_shape[0]} frames", flush=True)
    del root["data"][key]
    root["data"].move(temporary_key, key)
    return {
        "key": key,
        "source_shape": list(source_shape),
        "output_shape": list(output_shape),
        "xyz_bitwise_identical": True,
    }


def _verify_invariants(source_root, output_root) -> dict:
    result = {}
    for path in ("data/action", "data/state", "data/point_cloud", "meta/episode_ends"):
        result[path] = bool(
            np.array_equal(np.asarray(source_root[path][:]), np.asarray(output_root[path][:]))
        )
    if not all(result.values()):
        raise RuntimeError(f"matched non-representation arrays changed: {result}")
    return result


def build(args: argparse.Namespace) -> dict:
    import zarr

    source = args.source_zarr.expanduser().resolve()
    destination = args.output_zarr.expanduser().resolve()
    projection_path = args.projection_path.expanduser().resolve()
    output_meta = (
        args.output_meta.expanduser().resolve()
        if args.output_meta is not None
        else _infer_sidecar(destination)
    )
    if not source.is_dir():
        raise FileNotFoundError(f"source Utonia zarr does not exist: {source}")
    for path in (destination, projection_path, output_meta):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {path}")
    if args.batch_frames <= 0:
        raise ValueError("batch_frames must be positive")

    source_root = zarr.open(str(source), mode="r")
    source_array = source_root[f"data/{args.key}"]
    if source_array.ndim != 3 or source_array.shape[-1] <= 3:
        raise ValueError(f"invalid Utonia source shape: {source_array.shape}")
    input_dim = int(source_array.shape[-1]) - 3
    matrix = build_frozen_orthogonal_projection(
        input_dim, int(args.output_dim), seed=int(args.seed)
    )
    save_projection_artifact(projection_path, matrix, seed=int(args.seed))
    projection = load_projection_artifact(
        projection_path,
        expected_input_dim=input_dim,
        expected_output_dim=int(args.output_dim),
    )

    try:
        _copy_source_zarr(source, destination)
        output_root = zarr.open(str(destination), mode="a")
        transform = _replace_array(
            output_root,
            key=str(args.key),
            matrix=projection["matrix"],
            batch_frames=int(args.batch_frames),
        )
        invariants = _verify_invariants(source_root, output_root)
        source_meta_path = args.source_meta or _infer_sidecar(source)
        source_meta_sha = (
            sha256_file(source_meta_path.expanduser().resolve())
            if source_meta_path.is_file()
            else None
        )
        report = {
            "format": "robotwin_matched_utonia128_policy_zarr_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_zarr": str(source),
            "output_zarr": str(destination),
            "source_meta": str(source_meta_path),
            "source_meta_sha256": source_meta_sha,
            "projection": {
                "format": PROJECTION_FORMAT,
                "path": str(projection_path),
                "file_sha256": sha256_file(projection_path),
                "matrix_sha256": projection["matrix_sha256"],
                "seed": int(projection["seed"]),
                "input_dim": input_dim,
                "output_dim": int(args.output_dim),
                "norm_preserving_scale": float(np.sqrt(input_dim / int(args.output_dim))),
                "fit_data": "none_frozen_random_orthogonal",
                "test_data_used": False,
            },
            "transform": transform,
            "invariants": invariants,
            "formal_comparison_eligible": True,
        }
        output_meta.parent.mkdir(parents=True, exist_ok=True)
        temporary_meta = output_meta.with_name(f".{output_meta.name}.tmp")
        temporary_meta.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary_meta.replace(output_meta)
        return report
    except Exception:
        if destination.exists():
            shutil.rmtree(destination)
        if projection_path.exists():
            projection_path.unlink()
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive a matched 128D frozen-projection Utonia policy zarr."
    )
    parser.add_argument("--source-zarr", type=Path, required=True)
    parser.add_argument("--output-zarr", type=Path, required=True)
    parser.add_argument("--projection-path", type=Path, required=True)
    parser.add_argument("--source-meta", type=Path, default=None)
    parser.add_argument("--output-meta", type=Path, default=None)
    parser.add_argument("--key", default="utonia_point_cloud_A")
    parser.add_argument("--output-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--batch-frames", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    report = build(parse_args())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
