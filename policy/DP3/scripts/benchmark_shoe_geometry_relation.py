#!/usr/bin/env python3
"""Pure-geometry benchmark for the ID-free observation relation estimator."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from data_path_utils import raw_task_data_dir
from geometry_relation_estimator import create_estimator_from_spec
from object_pointcloud_utils import (
    extract_placeholder_point_cloud,
    load_hdf5,
    load_scene_info,
    parse_target_extents,
)
from train_ndf_goal_regressor import _supervision_for_frame


def _parse_ids(value: str) -> set[int]:
    return {int(item.strip()) for item in str(value).split(",") if item.strip()}


def _episode_index(path: Path) -> int:
    match = re.fullmatch(r"episode(\d+)\.hdf5", path.name)
    if match is None:
        raise ValueError(f"unexpected episode filename {path.name!r}")
    return int(match.group(1))


def _select_query_frame(episode: dict, mode: str) -> int:
    task_state = episode.get("task_state", {})
    phase = task_state.get("relation_phase") if isinstance(task_state, dict) else None
    if phase is not None:
        candidates = np.flatnonzero(np.asarray(phase).reshape(-1) > 0.5)
    else:
        length = int(np.asarray(episode["object_pointcloud"]["{A}"]).shape[0])
        candidates = np.arange(length, dtype=np.int64)
    if len(candidates) == 0:
        raise ValueError("episode has no query frame")
    if mode == "first_active":
        return int(candidates[0])
    if mode == "middle_active":
        return int(candidates[len(candidates) // 2])
    if mode == "last_active":
        return int(candidates[-1])
    raise ValueError(f"unsupported query frame mode {mode!r}")


def _rotation_error_deg(predicted: np.ndarray, target: np.ndarray) -> float:
    delta = predicted[:3, :3] @ target[:3, :3].T
    cosine = float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _summary(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0}
    translation = np.asarray(
        [row["translation_error_m"] for row in rows], dtype=np.float64
    )
    rotation = np.asarray(
        [row["rotation_error_deg"] for row in rows], dtype=np.float64
    )
    runtime = np.asarray([row["runtime_ms"] for row in rows], dtype=np.float64)
    success = np.asarray([row["geometry_success"] for row in rows], dtype=bool)
    flipped = np.asarray([row["flipped"] for row in rows], dtype=bool)
    return {
        "count": len(rows),
        "success_rate": float(success.mean()),
        "flip_rate": float(flipped.mean()),
        "translation_error_m": {
            "median": float(np.median(translation)),
            "p90": float(np.percentile(translation, 90)),
            "mean": float(np.mean(translation)),
        },
        "rotation_error_deg": {
            "median": float(np.median(rotation)),
            "p90": float(np.percentile(rotation, 90)),
            "mean": float(np.mean(rotation)),
        },
        "runtime_ms": {
            "median": float(np.median(runtime)),
            "p90": float(np.percentile(runtime, 90)),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark point-cloud-derived T_A_from_B. Object IDs and oracle "
            "labels are evaluator-only and are never passed to the estimator."
        )
    )
    parser.add_argument("task_name")
    parser.add_argument("task_config")
    parser.add_argument("expert_data_num", type=int)
    parser.add_argument("--estimator_spec", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--include_shoe_ids", default="")
    parser.add_argument("--exclude_shoe_ids", default="")
    parser.add_argument("--max_episodes", type=int, default=0)
    parser.add_argument("--target_num_points", type=int, default=1024)
    parser.add_argument(
        "--query_frame",
        choices=("first_active", "middle_active", "last_active"),
        default="first_active",
    )
    parser.add_argument("--success_translation_m", type=float, default=0.05)
    parser.add_argument("--success_rotation_deg", type=float, default=20.0)
    parser.add_argument("--allow_legacy_asset_supervision", action="store_true")
    parser.add_argument("--device", default="")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    include_ids = _parse_ids(args.include_shoe_ids)
    exclude_ids = _parse_ids(args.exclude_shoe_ids)
    if include_ids & exclude_ids:
        raise ValueError("include_shoe_ids and exclude_shoe_ids overlap")

    estimator = create_estimator_from_spec(
        args.estimator_spec,
        device_override=str(args.device) if args.device else None,
    )
    load_dir = raw_task_data_dir(args.task_name, args.task_config)
    scene_info = load_scene_info(str(load_dir / "scene_info.json"))
    episode_paths = sorted(
        (load_dir / "data").glob("episode*.hdf5"), key=_episode_index
    )
    episode_paths = [
        path for path in episode_paths if _episode_index(path) < int(args.expert_data_num)
    ]

    rows = []
    for episode_path in episode_paths:
        episode_idx = _episode_index(episode_path)
        episode = load_hdf5(str(episode_path))
        frame_idx = _select_query_frame(episode, str(args.query_frame))

        np.random.seed(200003 + episode_idx)
        pointclouds = {}
        modes = {}
        for placeholder in ("{A}", "{B}"):
            target_extents, _ = parse_target_extents(
                scene_info, episode_idx, placeholder
            )
            pointcloud, metadata = extract_placeholder_point_cloud(
                episode,
                frame_idx=frame_idx,
                placeholder=placeholder,
                target_num_points=int(args.target_num_points),
                target_extents=target_extents,
            )
            pointclouds[placeholder] = pointcloud
            modes[placeholder] = str(metadata.get("mode", "unknown"))

        start = time.perf_counter()
        prediction = estimator.estimate_goal(
            pointclouds["{A}"],
            pointclouds["{B}"],
        )
        runtime_ms = (time.perf_counter() - start) * 1000.0

        # Evaluator-only metadata is deliberately read after estimator inference.
        oracle_goal, shoe_id, supervision_source = _supervision_for_frame(
            episode,
            frame_index=frame_idx,
            episode_index=episode_idx,
            allow_legacy_asset_supervision=bool(args.allow_legacy_asset_supervision),
            scene_info=scene_info,
        )
        if include_ids and shoe_id not in include_ids:
            continue
        if shoe_id in exclude_ids:
            continue

        predicted_goal = prediction.goal_t_a_from_b
        translation_error = float(
            np.linalg.norm(predicted_goal[:3, 3] - oracle_goal[:3, 3])
        )
        rotation_error = _rotation_error_deg(predicted_goal, oracle_goal)
        success = (
            translation_error <= float(args.success_translation_m)
            and rotation_error <= float(args.success_rotation_deg)
        )
        rows.append(
            {
                "episode": episode_idx,
                "frame": frame_idx,
                "shoe_id_evaluator_only": shoe_id,
                "supervision_source_evaluator_only": supervision_source,
                "pointcloud_mode": modes,
                "translation_error_m": translation_error,
                "rotation_error_deg": rotation_error,
                "flipped": bool(
                    rotation_error >= 90.0 or prediction.flip_probability >= 0.5
                ),
                "geometry_success": bool(success),
                "solver_energy": float(prediction.solver_energy),
                "confidence": float(prediction.confidence),
                "flip_probability": float(prediction.flip_probability),
                "runtime_ms": float(runtime_ms),
                "diagnostics": prediction.diagnostics or {},
            }
        )
        print(
            f"episode={episode_idx} shoe={shoe_id} "
            f"t={translation_error:.4f}m r={rotation_error:.2f}deg "
            f"success={int(success)}"
        )
        if int(args.max_episodes) > 0 and len(rows) >= int(args.max_episodes):
            break

    if not rows:
        raise RuntimeError("the requested query split produced no benchmark episodes")
    by_shoe = {}
    for shoe_id in sorted({int(row["shoe_id_evaluator_only"]) for row in rows}):
        by_shoe[str(shoe_id)] = _summary(
            [row for row in rows if int(row["shoe_id_evaluator_only"]) == shoe_id]
        )
    result = {
        "schema_version": 1,
        "task_name": args.task_name,
        "task_config": args.task_config,
        "estimator_spec": str(Path(args.estimator_spec).resolve()),
        "query_frame": args.query_frame,
        "include_shoe_ids": sorted(include_ids),
        "exclude_shoe_ids": sorted(exclude_ids),
        "thresholds": {
            "translation_m": float(args.success_translation_m),
            "rotation_deg": float(args.success_rotation_deg),
        },
        "summary": _summary(rows),
        "by_shoe_evaluator_only": by_shoe,
        "episodes": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"wrote benchmark: {output}")


if __name__ == "__main__":
    main()
