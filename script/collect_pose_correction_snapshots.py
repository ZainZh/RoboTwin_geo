#!/usr/bin/env python3
"""Collect policy-independent semantic snapshots for exact paired evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from script.collect_geometry_flow_recovery import build_environment_args
from script.collect_pose_correction import (
    discard_episode,
    parse_levels,
    perturbation_seed_for_scene,
    prepare_perturbed_episode,
)
from script.evaluate_pose_correction_closedloop import (
    _capture_task_state,
    _snapshot_to_jsonable,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--task-name", default="place_shoe_geometry_marker")
    result.add_argument(
        "--task-config", default="demo_clean_3d_object_pc_geometry_marker_unseen05"
    )
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--episodes", type=int, default=24)
    result.add_argument("--seed-start", type=int, default=319)
    result.add_argument("--seed-stop", type=int, default=10000)
    result.add_argument("--allowed-shoes", default="0,5")
    result.add_argument("--max-per-shoe", type=int, default=12)
    result.add_argument("--seed", type=int, default=20260805)
    result.add_argument("--levels", default="1:3:5:20")
    result.add_argument("--success-translation-cm", type=float, default=4.0)
    result.add_argument("--success-rotation-deg", type=float, default=15.0)
    result.add_argument("--maximum-initial-translation-cm", type=float, default=8.0)
    result.add_argument("--maximum-initial-rotation-deg", type=float, default=30.0)
    return result


def _write_manifest(
    path: Path,
    args: argparse.Namespace,
    allowed: set[int],
    records: list[dict],
    attempts: list[dict],
    *,
    status: str,
) -> None:
    payload = {
        "schema_version": 2,
        "task": "policy_independent_pose_correction_semantic_snapshots",
        "task_name": str(args.task_name),
        "task_config": str(args.task_config),
        "allowed_shoes": sorted(allowed),
        "levels": str(args.levels),
        "requested_episodes": int(args.episodes),
        "collected_episodes": int(len(records)),
        "success_threshold": {
            "translation_cm": float(args.success_translation_cm),
            "rotation_deg": float(args.success_rotation_deg),
        },
        "initial_error_ceiling": {
            "translation_cm": float(args.maximum_initial_translation_cm),
            "rotation_deg": float(args.maximum_initial_rotation_deg),
        },
        "admission_contract": (
            "setup, held-out identity, motion-planner preparation, and initial "
            "pose-error threshold only; no learned policy is loaded or executed"
        ),
        "perturbation_seed_rule": "collection_seed_plus_scene_seed",
        "setup_is_test": False,
        "status": str(status),
        "attempts": attempts,
        "records": records,
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    from envs.utils.create_actor import UnStableError
    from script.collect_data import class_decorator

    args = parser().parse_args()
    if int(args.episodes) < 1 or int(args.max_per_shoe) < 1:
        raise ValueError("episodes and max-per-shoe must be positive")
    if float(args.success_translation_cm) <= 0.0:
        raise ValueError("success-translation-cm must be positive")
    if not 0.0 < float(args.success_rotation_deg) <= 180.0:
        raise ValueError("success-rotation-deg must lie in (0, 180]")
    if float(args.maximum_initial_translation_cm) <= float(args.success_translation_cm):
        raise ValueError("maximum initial translation must exceed success threshold")
    if not float(args.success_rotation_deg) < float(args.maximum_initial_rotation_deg) <= 180.0:
        raise ValueError("maximum initial rotation must exceed success threshold")
    levels = parse_levels(str(args.levels))
    allowed = {int(item) for item in str(args.allowed_shoes).split(",") if item}
    if not allowed:
        raise ValueError("allowed-shoes cannot be empty")
    if int(args.episodes) > int(args.max_per_shoe) * len(allowed):
        raise ValueError("episodes exceeds balanced per-shoe capacity")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refuse to overwrite snapshot manifest: {args.output}")

    environment = build_environment_args(args.task_name, args.task_config, args.output.parent)
    environment["data_type"].update(
        {"third_view": False, "pointcloud": False, "qpos": False}
    )
    task = class_decorator(args.task_name)
    records: list[dict] = []
    attempts: list[dict] = []
    shoe_counts: dict[int, int] = {}
    for scene_seed in range(int(args.seed_start), int(args.seed_stop)):
        if len(records) >= int(args.episodes):
            break
        episode = len(records)
        try:
            task.setup_demo(
                now_ep_num=episode,
                seed=scene_seed,
                is_test=False,
                **environment,
            )
        except UnStableError as error:
            attempts.append(
                {
                    "scene_seed": int(scene_seed),
                    "admitted": False,
                    "preparation_error": f"{type(error).__name__}: {error}",
                }
            )
            continue
        shoe_id = int(task.shoe_id)
        if shoe_id not in allowed or shoe_counts.get(shoe_id, 0) >= int(args.max_per_shoe):
            task.close_env()
            continue
        level_index = shoe_counts.get(shoe_id, 0) % len(levels)
        perturbation_seed = perturbation_seed_for_scene(args.seed, scene_seed)
        try:
            arm, _final_pose, perturbation, initial = prepare_perturbed_episode(
                task,
                levels[level_index],
                np.random.default_rng(perturbation_seed),
            )
            initial_success = bool(
                initial["translation_m"] * 100.0 <= float(args.success_translation_cm)
                and initial["rotation_deg"] <= float(args.success_rotation_deg)
            )
            if initial_success:
                raise RuntimeError("perturbed state already satisfies success threshold")
            if (
                initial["translation_m"] * 100.0
                > float(args.maximum_initial_translation_cm)
                or initial["rotation_deg"]
                > float(args.maximum_initial_rotation_deg)
            ):
                raise RuntimeError(
                    "perturbed state exceeds preregistered initial-error ceiling: "
                    f"{initial['translation_m'] * 100.0:.3f} cm / "
                    f"{initial['rotation_deg']:.3f} deg"
                )
            snapshot = _snapshot_to_jsonable(_capture_task_state(task))
            record = {
                "episode": int(episode),
                "scene_seed": int(scene_seed),
                "shoe_id": int(shoe_id),
                "arm": str(arm),
                "level": int(level_index),
                "perturbation_seed": int(perturbation_seed),
                "commanded_perturbation": perturbation,
                "initial": initial,
                "initial_success": False,
                "semantic_snapshot": snapshot,
                "admitted": True,
            }
            # Validate the exact serialization contract before accepting it.
            json.dumps(record, sort_keys=True)
        except (AssertionError, RuntimeError, TypeError, ValueError) as error:
            failure = {
                "episode": int(episode),
                "scene_seed": int(scene_seed),
                "shoe_id": int(shoe_id),
                "level": int(level_index),
                "perturbation_seed": int(perturbation_seed),
                "admitted": False,
                "preparation_error": f"{type(error).__name__}: {error}",
            }
            attempts.append(failure)
            discard_episode(task)
            _write_manifest(
                args.output, args, allowed, records, attempts, status="in_progress"
            )
            print(json.dumps(failure), flush=True)
            continue

        attempts.append({key: value for key, value in record.items() if key != "semantic_snapshot"})
        records.append(record)
        shoe_counts[shoe_id] = shoe_counts.get(shoe_id, 0) + 1
        task.close_env(clear_cache=False)
        _write_manifest(
            args.output, args, allowed, records, attempts, status="in_progress"
        )
        print(
            json.dumps(
                {
                    "episode": int(episode),
                    "scene_seed": int(scene_seed),
                    "shoe_id": int(shoe_id),
                    "level": int(level_index),
                    "initial": initial,
                    "admitted": True,
                    "shoe_counts": shoe_counts,
                }
            ),
            flush=True,
        )

    status = "complete" if len(records) == int(args.episodes) else "incomplete"
    _write_manifest(args.output, args, allowed, records, attempts, status=status)
    if status != "complete":
        raise RuntimeError(f"collected {len(records)}, requested {args.episodes}")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "records": len(records),
                "shoe_counts": shoe_counts,
                "attempts": len(attempts),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    from test_render import Sapien_TEST

    Sapien_TEST()
    main()
