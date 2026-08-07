#!/usr/bin/env python3
"""Collect single-stage, gripper-closed SE(3) pose-correction episodes.

The scripted expert is used only to generate demonstrations.  Each episode
grasp-preservingly perturbs an object already held near its placement target,
then records exactly one return motion to the final pre-release pose.  There is
no pre-place/release/settle stage ambiguity in the resulting action labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

DEFAULT_LEVELS = (
    ((0.01, 0.03), (5.0, 20.0)),
    ((0.03, 0.06), (20.0, 45.0)),
    ((0.05, 0.08), (45.0, 75.0)),
    ((0.002, 0.01), (1.0, 5.0)),
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--task-name", default="place_shoe_geometry_marker")
    result.add_argument(
        "--task-config", default="demo_clean_3d_object_pc_geometry_marker_remote50"
    )
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--episodes", type=int, default=12)
    result.add_argument("--seed-start", type=int, default=0)
    result.add_argument("--seed-stop", type=int, default=10000)
    result.add_argument("--allowed-shoes", default="2,3,4,7,8,9")
    result.add_argument("--max-per-shoe", type=int, default=4)
    result.add_argument("--seed", type=int, default=20260803)
    result.add_argument(
        "--levels",
        default="1:3:5:20,3:6:20:45,5:8:45:75,0.2:1:1:5",
        help="Comma-separated min_cm:max_cm:min_deg:max_deg levels.",
    )
    result.add_argument("--full-recording", action="store_true")
    result.add_argument(
        "--require-initial-failure",
        action="store_true",
        help=(
            "Admit only episodes whose perturbed initial pose is outside the "
            "declared success threshold. This prevents already-solved states "
            "from inflating closed-loop recovery success."
        ),
    )
    result.add_argument("--success-translation-cm", type=float, default=4.0)
    result.add_argument("--success-rotation-deg", type=float, default=15.0)
    return result


def parse_levels(value: str) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    levels = []
    for raw in value.split(","):
        fields = [float(item) for item in raw.split(":")]
        if len(fields) != 4:
            raise ValueError(f"invalid perturbation level {raw!r}")
        min_cm, max_cm, min_deg, max_deg = fields
        if not 0 <= min_cm <= max_cm or not 0 <= min_deg <= max_deg <= 180:
            raise ValueError(f"invalid perturbation bounds {raw!r}")
        levels.append(((min_cm / 100.0, max_cm / 100.0), (min_deg, max_deg)))
    if not levels:
        raise ValueError("at least one perturbation level is required")
    return tuple(levels)


def pose7_rotation(pose: np.ndarray) -> Rotation:
    value = np.asarray(pose, dtype=np.float64).reshape(7)
    return Rotation.from_quat(value[[4, 5, 6, 3]])


def pose7(position: np.ndarray, rotation: Rotation) -> np.ndarray:
    x, y, z, w = rotation.as_quat()
    return np.asarray([*position, w, x, y, z], dtype=np.float64)


def perturbation_seed_for_scene(collection_seed: int, scene_seed: int) -> int:
    """Return a replayable perturbation seed independent of admission order."""

    return int(collection_seed) + int(scene_seed)


def perturb_pose(
    final_pose: np.ndarray,
    pre_pose: np.ndarray,
    translation_range: tuple[float, float],
    rotation_range_deg: tuple[float, float],
    generator: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float]]:
    final = np.asarray(final_pose, dtype=np.float64).reshape(7)
    pre = np.asarray(pre_pose, dtype=np.float64).reshape(7)
    outward = pre[:3] - final[:3]
    outward /= max(float(np.linalg.norm(outward)), 1e-12)
    lateral = generator.normal(size=3)
    lateral -= outward * float(np.dot(lateral, outward))
    lateral /= max(float(np.linalg.norm(lateral)), 1e-12)
    direction = outward + generator.uniform(-0.45, 0.45) * lateral
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    translation_m = generator.uniform(*translation_range)
    translation = direction * translation_m
    axis = generator.normal(size=3)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    rotation_deg = generator.uniform(*rotation_range_deg)
    delta = Rotation.from_rotvec(axis * np.deg2rad(rotation_deg))
    perturbed = pose7(final[:3] + translation, delta * pose7_rotation(final))
    return perturbed, {
        "translation_m": float(translation_m),
        "rotation_deg": float(rotation_deg),
    }


def alignment(task) -> dict[str, float]:
    from envs.placement_metrics import functional_pose_alignment_errors

    shoe = task.shoe.get_functional_point(0, "pose")
    target = task.target_block.get_functional_point(0, "pose")
    errors = functional_pose_alignment_errors(shoe.p, shoe.q, target.p, target.q)
    return {
        "translation_m": float(errors["translation_error_norm_m"]),
        "rotation_deg": float(errors["rotation_error_deg"]),
    }


def discard_episode(task) -> None:
    try:
        task.close_env(clear_cache=False)
    finally:
        if hasattr(task, "folder_path"):
            task.remove_data_cache()


def write_manifest(
    output: Path,
    args: argparse.Namespace,
    allowed: set[int],
    records: list[dict],
    attempts: list[dict],
    *,
    status: str,
) -> dict:
    manifest = {
        "schema_version": 1,
        "task": "single_stage_gripper_closed_pose_correction",
        "task_name": args.task_name,
        "task_config": args.task_config,
        "allowed_shoes": sorted(allowed),
        "levels": args.levels,
        "requested_episodes": int(args.episodes),
        "collected_episodes": len(records),
        "require_initial_failure": bool(args.require_initial_failure),
        "success_threshold": {
            "translation_cm": float(args.success_translation_cm),
            "rotation_deg": float(args.success_rotation_deg),
        },
        "perturbation_seed_rule": "collection_seed_plus_scene_seed",
        "setup_is_test": False,
        "status": str(status),
        "attempts": attempts,
        "records": records,
    }
    path = output / "pose_correction_manifest.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    return manifest


def prepare_perturbed_episode(task, level, generator):
    """Prepare one grasp-preserving perturbation before recording starts."""
    from envs.utils import ArmTag

    arm = ArmTag(str(task.prepare_policy_placement_phase()))
    target_pose = task.target_block.get_functional_point(0)
    pre_pose = np.asarray(
        task.get_place_pose(
            task.shoe,
            arm,
            target_pose,
            functional_point_id=0,
            pre_dis=0.12,
            constrain="align",
        ),
        dtype=np.float64,
    )
    final_pose = np.asarray(
        task.get_place_pose(
            task.shoe,
            arm,
            target_pose,
            functional_point_id=0,
            pre_dis=0.02,
            constrain="align",
        ),
        dtype=np.float64,
    )
    task.move(task.move_to_pose(arm, pre_pose))
    task.move(task.move_to_pose(arm, final_pose))
    if not task.plan_success:
        raise RuntimeError("failed to reach the nominal correction endpoint")
    perturbed, perturbation = perturb_pose(final_pose, pre_pose, *level, generator)
    task.move(task.move_to_pose(arm, perturbed))
    if not task.plan_success:
        raise RuntimeError("failed to reach the sampled perturbation")
    return arm, final_pose, perturbation, alignment(task)


def main() -> None:
    from envs.utils.create_actor import UnStableError
    from script.collect_data import class_decorator
    from script.collect_geometry_flow_recovery import build_environment_args

    args = parser().parse_args()
    if float(args.success_translation_cm) <= 0.0:
        raise ValueError("success-translation-cm must be positive")
    if not 0.0 < float(args.success_rotation_deg) <= 180.0:
        raise ValueError("success-rotation-deg must lie in (0,180]")
    args.output.mkdir(parents=True, exist_ok=True)
    levels = parse_levels(args.levels)
    allowed = {int(item) for item in args.allowed_shoes.split(",") if item}
    environment = build_environment_args(args.task_name, args.task_config, args.output)
    if not args.full_recording:
        environment["data_type"].update(
            {"third_view": False, "pointcloud": False, "qpos": False}
        )
    task = class_decorator(args.task_name)
    records = []
    attempts = []
    shoe_counts: dict[int, int] = {}
    for scene_seed in range(int(args.seed_start), int(args.seed_stop)):
        if len(records) >= int(args.episodes):
            break
        episode = len(records)
        try:
            task.setup_demo(now_ep_num=episode, seed=scene_seed, **environment)
        except UnStableError:
            continue
        shoe_id = int(task.shoe_id)
        if shoe_id not in allowed or shoe_counts.get(shoe_id, 0) >= int(args.max_per_shoe):
            task.close_env()
            continue
        # Each shoe identity receives every perturbation level before repeats,
        # rather than letting scene-seed frequency bias the difficulty mix.
        level_index = shoe_counts.get(shoe_id, 0) % len(levels)
        perturbation_seed = perturbation_seed_for_scene(args.seed, scene_seed)
        generator = np.random.default_rng(perturbation_seed)
        try:
            arm, final_pose, perturbation, before = prepare_perturbed_episode(
                task, levels[level_index], generator
            )
        except (AssertionError, RuntimeError, ValueError) as error:
            failure = {
                "episode": episode,
                "scene_seed": int(scene_seed),
                "shoe_id": shoe_id,
                "level": int(level_index),
                "perturbation_seed": int(perturbation_seed),
                "admitted": False,
                "preparation_error": f"{type(error).__name__}: {error}",
            }
            attempts.append(failure)
            discard_episode(task)
            write_manifest(
                args.output, args, allowed, records, attempts, status="in_progress"
            )
            print(json.dumps(failure), flush=True)
            continue

        task.save_data = True
        task.FRAME_IDX = 0
        task.eval_success = False
        task.move(task.move_to_pose(arm, final_pose))
        after = alignment(task)
        record = {
            "episode": episode,
            "scene_seed": int(scene_seed),
            "shoe_id": shoe_id,
            "arm": str(arm),
            "level": int(level_index),
            "perturbation_seed": int(perturbation_seed),
            "commanded_perturbation": perturbation,
            "before": before,
            "after": after,
            "recorded_frames": int(task.FRAME_IDX),
            "plan_success": bool(task.plan_success),
        }
        initial_success = bool(
            before["translation_m"] * 100.0
            <= float(args.success_translation_cm)
            and before["rotation_deg"] <= float(args.success_rotation_deg)
        )
        record["initial_success"] = initial_success
        attempts.append(record)
        admitted = bool(
            task.plan_success
            and task.FRAME_IDX >= 2
            and after["translation_m"] * 100.0
            <= float(args.success_translation_cm)
            and after["rotation_deg"] <= float(args.success_rotation_deg)
            and (not args.require_initial_failure or not initial_success)
        )
        record["admitted"] = admitted
        if not admitted:
            discard_episode(task)
            write_manifest(
                args.output, args, allowed, records, attempts, status="in_progress"
            )
            print(json.dumps(record), flush=True)
            continue
        task.close_env(clear_cache=False)
        task.merge_pkl_to_hdf5_video()
        task.remove_data_cache()
        records.append(record)
        shoe_counts[shoe_id] = shoe_counts.get(shoe_id, 0) + 1
        write_manifest(
            args.output, args, allowed, records, attempts, status="in_progress"
        )
        print(json.dumps(record), flush=True)

    manifest = write_manifest(
        args.output,
        args,
        allowed,
        records,
        attempts,
        status="complete" if len(records) >= int(args.episodes) else "incomplete",
    )
    if len(records) < int(args.episodes):
        raise RuntimeError(
            f"collected {len(records)} pose corrections, requested {args.episodes}"
        )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    from test_render import Sapien_TEST

    Sapien_TEST()
    main()
