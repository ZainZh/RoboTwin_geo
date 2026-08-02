#!/usr/bin/env python3
"""Collect expert recovery trajectories from GeometryFlow policy states."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import yaml

from envs._GLOBAL_CONFIGS import CONFIGS_PATH
from envs.utils import ArmTag
from envs.utils.create_actor import UnStableError
from policy.GeometryFlow import deploy_policy
from script.collect_data import class_decorator, get_embodiment_config


def build_environment_args(task_name: str, task_config: str, output: Path) -> dict:
    with open(f"task_config/{task_config}.yml", "r", encoding="utf-8") as handle:
        args = yaml.load(handle.read(), Loader=yaml.FullLoader)
    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as handle:
        embodiments = yaml.load(handle.read(), Loader=yaml.FullLoader)
    embodiment = args["embodiment"]
    if len(embodiment) != 1:
        raise ValueError("recovery collector currently expects one shared dual-arm embodiment")
    robot_file = embodiments[embodiment[0]]["file_path"]
    args.update(
        {
            "task_name": task_name,
            "task_config": task_config,
            "embodiment_name": str(embodiment[0]),
            "left_robot_file": robot_file,
            "right_robot_file": robot_file,
            "left_embodiment_config": get_embodiment_config(robot_file),
            "right_embodiment_config": get_embodiment_config(robot_file),
            "dual_arm_embodied": True,
            "save_path": str(output.resolve()),
            "need_plan": True,
            "save_data": False,
            "eval_mode": True,
            "render_freq": 0,
            "eval_video_log": False,
        }
    )
    return args


def load_seed_list(path: Path) -> list[int]:
    values = [int(value) for value in path.read_text(encoding="utf-8").split()]
    if not values:
        raise ValueError(f"seed file is empty: {path}")
    return values


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--task-name", default="place_shoe_geometry_marker")
    result.add_argument(
        "--task-config", default="demo_clean_3d_object_pc_geometry_marker"
    )
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--library", type=Path, required=True)
    result.add_argument(
        "--representation",
        choices=("pca", "grasp_relation_geometry"),
        default="pca",
    )
    result.add_argument(
        "--oracle-task-dataset",
        type=Path,
        default=Path("outputs/geometry_flow/task_flow_oracle_50_stride1_goal_labels.npz"),
    )
    result.add_argument("--grasp-fit-threshold-m2", type=float, default=2.5e-4)
    result.add_argument("--seed-file", type=Path)
    result.add_argument("--seed-start", type=int, default=0)
    result.add_argument("--seed-stop", type=int, default=1000)
    result.add_argument("--allowed-shoes", default="2,3,4,7,8,9")
    result.add_argument("--train-shoes", default="2,3,4,7,8,9")
    result.add_argument("--episodes", type=int, default=6)
    result.add_argument(
        "--max-per-shoe",
        type=int,
        default=1,
        help="Maximum committed recoveries per shoe identity.",
    )
    result.add_argument("--rollout-calls", type=int, default=2)
    result.add_argument("--execute-steps", type=int, default=6)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--max-translation-delta-m", type=float, default=0.08)
    result.add_argument("--max-rotation-delta-rad", type=float, default=0.5)
    result.add_argument(
        "--full-recording",
        action="store_true",
        help=(
            "Retain generic scene point clouds, qpos, and observer RGB.  The "
            "default recovery dataset records only the head RGB required by "
            "the HDF5 merger, A/B object clouds, end poses, and task labels."
        ),
    )
    return result


def main() -> None:
    args_cli = parser().parse_args()
    args_cli.output.mkdir(parents=True, exist_ok=True)
    allowed = {int(value) for value in args_cli.allowed_shoes.split(",") if value}
    seeds = (
        load_seed_list(args_cli.seed_file)
        if args_cli.seed_file is not None
        else list(range(int(args_cli.seed_start), int(args_cli.seed_stop)))
    )
    if not seeds:
        raise ValueError("the requested recovery seed range is empty")
    environment_args = build_environment_args(
        args_cli.task_name, args_cli.task_config, args_cli.output
    )
    if not args_cli.full_recording:
        environment_args["data_type"].update(
            {
                "third_view": False,
                "pointcloud": False,
                "qpos": False,
            }
        )
    model = deploy_policy.get_model(
        {
            "geometry_flow_checkpoint": str(args_cli.checkpoint),
            "geometry_flow_library": str(args_cli.library),
            "geometry_flow_train_shoes": args_cli.train_shoes,
            "geometry_flow_representation": args_cli.representation,
            "geometry_flow_oracle_task_dataset": str(args_cli.oracle_task_dataset),
            "geometry_flow_grasp_fit_threshold_m2": float(
                args_cli.grasp_fit_threshold_m2
            ),
            "geometry_flow_device": args_cli.device,
            "geometry_flow_execute_steps": int(args_cli.execute_steps),
            "geometry_flow_route_to_closed_gripper": True,
            "geometry_flow_hold_active_gripper_closed": True,
            "max_eef_translation_delta_m": float(
                args_cli.max_translation_delta_m
            ),
            "max_eef_rotation_delta_rad": float(args_cli.max_rotation_delta_rad),
        }
    )
    task = class_decorator(args_cli.task_name)
    records = []
    attempts = []
    unstable_seeds = []
    shoe_counts: dict[int, int] = {}
    for seed in seeds:
        if len(records) >= int(args_cli.episodes):
            break
        episode = len(records)
        try:
            task.setup_demo(now_ep_num=episode, seed=int(seed), **environment_args)
        except UnStableError:
            unstable_seeds.append(int(seed))
            try:
                task.close_env(clear_cache=False)
            except Exception:
                pass
            print(json.dumps({"seed": int(seed), "unstable": True}), flush=True)
            continue
        shoe_id = int(task.shoe_id)
        if shoe_id not in allowed or shoe_counts.get(shoe_id, 0) >= int(
            args_cli.max_per_shoe
        ):
            task.close_env()
            continue
        arm_tag = task.prepare_policy_placement_phase()
        deploy_policy.reset_model(model)
        for _ in range(int(args_cli.rollout_calls)):
            if task.eval_success or task.take_action_cnt >= task.step_lim:
                break
            deploy_policy.eval(task, model, task.get_obs())
        if task.eval_success:
            task.close_env()
            continue
        if model.predicted_flow is None:
            task.close_env()
            raise RuntimeError("policy rollout did not construct a task flow")

        task.save_data = True
        task.FRAME_IDX = 0
        task.eval_success = False
        recovered = task.recover_policy_placement_phase(ArmTag(str(arm_tag)))
        record = {
            "episode": episode,
            "seed": int(seed),
            "shoe_id": shoe_id,
            "arm": str(arm_tag),
            "rollout_calls": int(args_cli.rollout_calls),
            "policy_actions": int(model.executed_actions),
            "recovered": bool(recovered and task.plan_success),
            "source_episode": model.flow_diagnostics.get("source_episode"),
            "source_shoe": model.flow_diagnostics.get("source_shoe"),
            "representation_cost": model.flow_diagnostics.get("representation_cost"),
        }
        attempts.append(record)
        if not record["recovered"]:
            task.close_env(clear_cache=False)
            if hasattr(task, "folder_path"):
                task.remove_data_cache()
            print(json.dumps(record), flush=True)
            continue

        reference_dir = args_cli.output / "reference_flow"
        reference_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            reference_dir / f"episode{episode}.npz",
            predicted_flow=model.predicted_flow,
            initial_anchors=model.initial_anchors,
        )
        task.close_env(clear_cache=False)
        task.merge_pkl_to_hdf5_video()
        task.remove_data_cache()
        records.append(record)
        shoe_counts[shoe_id] = shoe_counts.get(shoe_id, 0) + 1
        print(json.dumps(record), flush=True)

    manifest = {
        "schema_version": 1,
        "checkpoint": str(args_cli.checkpoint.resolve()),
        "library": str(args_cli.library.resolve()),
        "representation": str(args_cli.representation),
        "oracle_task_dataset": str(args_cli.oracle_task_dataset.resolve()),
        "grasp_fit_threshold_m2": float(args_cli.grasp_fit_threshold_m2),
        "max_translation_delta_m": float(args_cli.max_translation_delta_m),
        "max_rotation_delta_rad": float(args_cli.max_rotation_delta_rad),
        "allowed_shoes": sorted(allowed),
        "requested_episodes": int(args_cli.episodes),
        "max_per_shoe": int(args_cli.max_per_shoe),
        "full_recording": bool(args_cli.full_recording),
        "attempted_recoveries": len(attempts),
        "unstable_seed_count": len(unstable_seeds),
        "unstable_seeds": unstable_seeds,
        "recovery_admission_rate": (
            float(len(records) / len(attempts)) if attempts else None
        ),
        "collected_episodes": len(records),
        "attempts": attempts,
        "records": records,
    }
    (args_cli.output / "recovery_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if len(records) < int(args_cli.episodes):
        raise RuntimeError(
            f"collected {len(records)} recoveries, requested {args_cli.episodes}"
        )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    from test_render import Sapien_TEST

    Sapien_TEST()
    main()
