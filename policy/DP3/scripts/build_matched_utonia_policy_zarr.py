#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from build_semantic_policy_ablation_zarr import _copy_source_zarr, _infer_sidecar
from data_path_utils import raw_task_data_dir
from matched_utonia_feature_utils import (
    compute_utonia_features_at_queries,
    nearest_context_support,
)
from object_pointcloud_utils import (
    extract_placeholder_point_cloud,
    load_hdf5,
    load_scene_info,
    parse_target_extents,
    valid_xyz_centroid,
)
from semantic_field_eval_utils import sha256_file
from utonia_feature_utils import load_utonia_model


def _validate_source(root, *, source_key: str) -> tuple[int, int]:
    if "data" not in root or "meta" not in root:
        raise KeyError("source zarr must contain data and meta groups")
    required = ("action", "state", "point_cloud", source_key)
    missing = [key for key in required if key not in root["data"]]
    if missing:
        raise KeyError(f"source zarr is missing arrays: {missing}")
    source = root[f"data/{source_key}"]
    if source.ndim != 3 or source.shape[-1] < 3:
        raise ValueError(f"source query cloud must be [T,Q,C>=3], got {source.shape}")
    frame_count = int(source.shape[0])
    query_count = int(source.shape[1])
    for key in ("action", "state", "point_cloud"):
        if int(root[f"data/{key}"].shape[0]) != frame_count:
            raise ValueError(f"frame count mismatch for data/{key}")
    episode_ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
    if len(episode_ends) == 0 or int(episode_ends[-1]) != frame_count:
        raise ValueError("episode_ends do not terminate at the source frame count")
    return frame_count, query_count


def _checkpoint_manifest(checkpoint: str) -> dict:
    candidate = Path(checkpoint).expanduser()
    if checkpoint == "auto":
        candidate = Path.home() / ".cache" / "utonia" / "ckpt" / "utonia.pth"
    manifest = {"requested": str(checkpoint)}
    if candidate.is_file():
        resolved = candidate.resolve()
        manifest.update({"resolved": str(resolved), "sha256": sha256_file(resolved)})
    return manifest


def _source_meta(path: Path, explicit: Path | None) -> tuple[Path, dict]:
    meta_path = explicit.expanduser().resolve() if explicit else _infer_sidecar(path)
    if not meta_path.is_file():
        raise FileNotFoundError(f"source sidecar metadata is unavailable: {meta_path}")
    return meta_path, json.loads(meta_path.read_text(encoding="utf-8"))


def _resolve_raw_dir(args: argparse.Namespace, source_meta: dict) -> Path:
    if args.raw_data_dir is not None:
        raw_dir = args.raw_data_dir.expanduser().resolve()
    else:
        raw_dir = raw_task_data_dir(
            str(source_meta["task_name"]), str(source_meta["task_config"])
        )
    required = (raw_dir / "scene_info.json", raw_dir / "data" / "episode0.hdf5")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "raw support mode requires the original demonstrations; missing "
            + ", ".join(missing)
        )
    return raw_dir


def _create_output_array(root, *, source_key: str, output_key: str, feature_dim: int):
    data = root["data"]
    if output_key in data:
        raise KeyError(f"output data key already exists: {output_key}")
    source = data[source_key]
    output_shape = (int(source.shape[0]), int(source.shape[1]), 3 + int(feature_dim))
    source_chunks = tuple(int(value) for value in source.chunks)
    output_chunks = (source_chunks[0], source_chunks[1], 3 + int(feature_dim))
    temporary_key = f"{output_key}__building"
    if temporary_key in data:
        raise KeyError(f"temporary output key already exists: {temporary_key}")
    output = data.create_dataset(
        temporary_key,
        shape=output_shape,
        chunks=output_chunks,
        dtype=np.float32,
        compressor=source.compressor,
        overwrite=False,
    )
    return temporary_key, output


def _write_features(
    *,
    output,
    frame_index: int,
    artifacts: dict,
    support: np.ndarray,
    queries: np.ndarray,
    args: argparse.Namespace,
) -> None:
    features = compute_utonia_features_at_queries(
        artifacts,
        support,
        queries,
        placeholder=str(args.placeholder),
        color_mode=str(args.color_mode),
        normal_mode=str(args.normal_mode),
    )
    if not np.array_equal(features[:, :3], queries):
        raise RuntimeError(f"query xyz changed at frame {frame_index}")
    if not np.isfinite(features).all():
        raise RuntimeError(f"non-finite Utonia features at frame {frame_index}")
    output[frame_index] = features


def _build_from_saved_support(
    *,
    root,
    output,
    artifacts: dict,
    args: argparse.Namespace,
    process_frames: int,
) -> None:
    query_array = root[f"data/{args.source_key}"]
    context_array = root["data/point_cloud"]
    for frame_index in range(process_frames):
        queries = np.asarray(query_array[frame_index, :, :3], dtype=np.float32)
        if args.support_source == "query":
            support = np.zeros((len(queries), 6), dtype=np.float32)
            support[:, :3] = queries
        else:
            support = nearest_context_support(
                np.asarray(context_array[frame_index], dtype=np.float32),
                queries,
                support_num_points=int(args.context_support_points),
            )
        _write_features(
            output=output,
            frame_index=frame_index,
            artifacts=artifacts,
            support=support,
            queries=queries,
            args=args,
        )
        if (frame_index + 1) % int(args.progress_every) == 0:
            print(f"processed {frame_index + 1} / {process_frames} frames", flush=True)


def _build_from_raw_support(
    *,
    root,
    output,
    artifacts: dict,
    args: argparse.Namespace,
    process_frames: int,
    raw_dir: Path,
) -> None:
    query_array = root[f"data/{args.source_key}"]
    episode_ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
    scene_info = load_scene_info(str(raw_dir / "scene_info.json"))
    global_frame = 0
    previous_end = 0
    for episode_index, episode_end in enumerate(episode_ends.tolist()):
        episode_frames = int(episode_end) - int(previous_end)
        previous_end = int(episode_end)
        episode_path = raw_dir / "data" / f"episode{episode_index}.hdf5"
        if not episode_path.is_file():
            raise FileNotFoundError(f"raw episode is unavailable: {episode_path}")
        episode = load_hdf5(str(episode_path))
        if int(episode["vector"].shape[0]) - 1 != episode_frames:
            raise ValueError(
                f"episode {episode_index} frame mismatch: raw="
                f"{int(episode['vector'].shape[0]) - 1}, zarr={episode_frames}"
            )
        previous_centroid = None
        target_extents, _ = parse_target_extents(
            scene_info, episode_index, str(args.placeholder)
        )
        for local_frame in range(episode_frames):
            if global_frame >= process_frames:
                return
            support, _ = extract_placeholder_point_cloud(
                episode,
                frame_idx=local_frame,
                placeholder=str(args.placeholder),
                target_num_points=int(args.raw_support_points),
                target_extents=target_extents,
                prev_centroid=previous_centroid,
                cluster_eps=float(args.cluster_eps),
                min_cluster_points=int(args.min_cluster_points),
                table_quantile=float(args.table_quantile),
                table_margin=float(args.table_margin),
            )
            centroid = valid_xyz_centroid(support)
            if centroid is not None:
                previous_centroid = centroid
            queries = np.asarray(query_array[global_frame, :, :3], dtype=np.float32)
            _write_features(
                output=output,
                frame_index=global_frame,
                artifacts=artifacts,
                support=support,
                queries=queries,
                args=args,
            )
            global_frame += 1
            if global_frame % int(args.progress_every) == 0:
                print(f"processed {global_frame} / {process_frames} frames", flush=True)


def _verify_copied_arrays(source_root, output_root) -> dict[str, bool]:
    invariants = {}
    for path in ("data/action", "data/state", "data/point_cloud", "meta/episode_ends"):
        invariants[path] = bool(
            np.array_equal(np.asarray(source_root[path][:]), np.asarray(output_root[path][:]))
        )
    if not all(invariants.values()):
        raise RuntimeError(f"non-representation source arrays changed: {invariants}")
    return invariants


def build_matched_utonia_zarr(args: argparse.Namespace) -> dict:
    import zarr

    np.random.seed(int(args.seed))
    source = args.source_zarr.expanduser().resolve()
    destination = args.output_zarr.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"source zarr does not exist: {source}")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite output path: {destination}")
    source_meta_path, source_meta = _source_meta(source, args.source_meta)
    raw_dir = (
        _resolve_raw_dir(args, source_meta)
        if args.support_source == "raw"
        else None
    )

    source_root = zarr.open(str(source), mode="r")
    frame_count, query_count = _validate_source(
        source_root, source_key=str(args.source_key)
    )
    process_frames = frame_count
    if int(args.max_frames) > 0:
        process_frames = min(frame_count, int(args.max_frames))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    artifacts = load_utonia_model(
        device=device,
        checkpoint=str(args.utonia_checkpoint),
        repo_id=str(args.utonia_repo_id),
        upcast_levels=int(args.utonia_upcast_levels),
        grid_size=float(args.grid_size),
        coord_scale=float(args.coord_scale),
        center_shift_z=not bool(args.disable_z_shift),
    )
    feature_dim = int(artifacts["feature_dim"])

    _copy_source_zarr(source, destination)
    output_root = zarr.open(str(destination), mode="a")
    temporary_key, output = _create_output_array(
        output_root,
        source_key=str(args.source_key),
        output_key=str(args.output_key),
        feature_dim=feature_dim,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    if args.support_source == "raw":
        _build_from_raw_support(
            root=output_root,
            output=output,
            artifacts=artifacts,
            args=args,
            process_frames=process_frames,
            raw_dir=raw_dir,
        )
    else:
        _build_from_saved_support(
            root=output_root,
            output=output,
            artifacts=artifacts,
            args=args,
            process_frames=process_frames,
        )
    elapsed = time.perf_counter() - started

    source_queries = output_root[f"data/{args.source_key}"]
    written_queries = output[..., :3]
    if not np.array_equal(
        np.asarray(source_queries[:process_frames, :, :3]),
        np.asarray(written_queries[:process_frames]),
    ):
        raise RuntimeError("written Utonia query xyz are not bitwise-identical")

    del output_root["data"][str(args.source_key)]
    output_root["data"].move(temporary_key, str(args.output_key))
    invariants = _verify_copied_arrays(source_root, output_root)
    complete = process_frames == frame_count
    formal_eligible = bool(complete and args.support_source == "raw")
    warning = None
    if args.support_source != "raw":
        warning = (
            "Engineering smoke only: independent raw object support was unavailable; "
            "do not report this dataset as a matched paper baseline."
        )
    elif not complete:
        warning = "Partial dataset generated with --max-frames; do not use for training."

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_zarr": str(source),
        "source_meta": str(source_meta_path),
        "output_zarr": str(destination),
        "source_key": str(args.source_key),
        "output_key": str(args.output_key),
        "placeholder": str(args.placeholder),
        "source_shape": list(source_root[f"data/{args.source_key}"].shape),
        "output_shape": [frame_count, query_count, 3 + feature_dim],
        "frames_written": process_frames,
        "complete": complete,
        "formal_comparison_eligible": formal_eligible,
        "warning": warning,
        "support_source": str(args.support_source),
        "raw_data_dir": str(raw_dir) if raw_dir is not None else None,
        "raw_support_points": int(args.raw_support_points),
        "context_support_points": int(args.context_support_points),
        "color_mode": str(args.color_mode),
        "normal_mode": str(args.normal_mode),
        "query_xyz_bitwise_identical": True,
        "copied_array_invariants": invariants,
        "utonia_feature_dim": feature_dim,
        "utonia_checkpoint": _checkpoint_manifest(str(args.utonia_checkpoint)),
        "utonia_repo_id": str(args.utonia_repo_id),
        "utonia_upcast_levels": int(args.utonia_upcast_levels),
        "seed": int(args.seed),
        "elapsed_seconds": elapsed,
        "peak_cuda_mib": (
            float(torch.cuda.max_memory_allocated(device) / (1024**2))
            if device.type == "cuda"
            else 0.0
        ),
    }
    output_meta = args.output_meta or _infer_sidecar(destination)
    output_meta = output_meta.expanduser().resolve()
    output_meta.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a frozen-Utonia policy zarr at the bitwise-identical semantic query xyz."
        )
    )
    parser.add_argument("--source-zarr", type=Path, required=True)
    parser.add_argument("--output-zarr", type=Path, required=True)
    parser.add_argument("--source-meta", type=Path)
    parser.add_argument("--output-meta", type=Path)
    parser.add_argument("--source-key", default="semantic_point_cloud_A")
    parser.add_argument("--output-key", default="utonia_point_cloud_A")
    parser.add_argument("--placeholder", default="{A}")
    parser.add_argument(
        "--support-source",
        choices=("raw", "context_nearest", "query"),
        default="raw",
    )
    parser.add_argument("--raw-data-dir", type=Path)
    parser.add_argument("--raw-support-points", type=int, default=1024)
    parser.add_argument("--context-support-points", type=int, default=512)
    parser.add_argument("--color-mode", choices=("debug_placeholder", "stored_scaled", "stored"), default="debug_placeholder")
    parser.add_argument("--normal-mode", choices=("fallback", "estimated"), default="fallback")
    parser.add_argument("--utonia-checkpoint", default="auto")
    parser.add_argument("--utonia-repo-id", default="Pointcept/Utonia")
    parser.add_argument("--utonia-upcast-levels", type=int, default=0)
    parser.add_argument("--grid-size", type=float, default=0.01)
    parser.add_argument("--coord-scale", type=float, default=1.0)
    parser.add_argument("--disable-z-shift", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--cluster-eps", type=float, default=0.04)
    parser.add_argument("--min-cluster-points", type=int, default=24)
    parser.add_argument("--table-quantile", type=float, default=0.08)
    parser.add_argument("--table-margin", type=float, default=0.01)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    build_matched_utonia_zarr(args)


if __name__ == "__main__":
    main()
