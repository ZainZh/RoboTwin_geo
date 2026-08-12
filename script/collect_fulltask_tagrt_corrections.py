#!/usr/bin/env python3
"""Collect expert continuations from full-task TAGRT policy states.

The learned policy is executed from the task start to an on-policy handoff
state. A simulator expert then provides only the remaining dense-control
supervision. Privileged state is used solely for collection admission; deployed
policies never use that signal or invoke the expert.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path

import yaml

from envs._GLOBAL_CONFIGS import CONFIGS_PATH
from envs.utils import ArmTag
from envs.utils.create_actor import UnStableError
from policy.GeometryFlow import deploy_fulltask_tagrt_v2
from script.collect_data import class_decorator, get_embodiment_config


def artifact_root() -> Path:
    return Path(os.environ.get("TAGRT_ARTIFACT_ROOT", "/shared2/sz/TAGRT-v1"))


def parser() -> argparse.ArgumentParser:
    root = artifact_root()
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--task-name", default="place_shoe_geometry_marker")
    result.add_argument(
        "--task-config",
        default="demo_clean_3d_object_pc_geometry_marker_densecontrol_gate2_train30",
    )
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--frame-checkpoints", nargs="+", type=Path, required=True)
    result.add_argument("--camera-metadata", type=Path, required=True)
    result.add_argument("--ensemble-metadata", type=Path, required=True)
    result.add_argument("--seed-file", type=Path, required=True)
    result.add_argument("--source-episode-start", type=int, required=True)
    result.add_argument("--episodes", type=int, default=6)
    result.add_argument("--execute-steps", type=int, default=15)
    result.add_argument("--max-policy-chunks", type=int, default=80)
    result.add_argument(
        "--handoff-after-chunks",
        type=int,
        default=None,
        help=(
            "Optional collection-only fallback: hand off before executing this "
            "policy chunk even if no close is predicted. This supports iterative "
            "on-policy correction after the rollout distribution changes."
        ),
    )
    result.add_argument("--post-lift-chunks", type=int, default=0)
    result.add_argument(
        "--handoff-mode",
        choices=("preclose", "postlift"),
        default="preclose",
        help=(
            "preclose hands off before the first predicted left-gripper close "
            "and records a full expert correction; postlift is diagnostic only."
        ),
    )
    result.add_argument("--minimum-lift-m", type=float, default=0.03)
    result.add_argument("--device", default="cuda:0")
    result.add_argument(
        "--output",
        type=Path,
        default=root / "simulator_data/place_shoe_geometry_marker/fulltask_tagrt_correction_pilot",
    )
    return result


def load_seeds(path: Path, count: int) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("accepted_seeds", payload) if isinstance(payload, dict) else payload
    if not isinstance(values, list) or len(values) < int(count):
        raise ValueError(f"seed file must provide at least {count} seeds")
    return [int(value) for value in values[: int(count)]]


def environment_args(task_name: str, task_config: str, output: Path) -> dict:
    with open(f"task_config/{task_config}.yml", "r", encoding="utf-8") as handle:
        args = yaml.load(handle.read(), Loader=yaml.FullLoader)
    with open(
        os.path.join(CONFIGS_PATH, "_embodiment_config.yml"),
        "r",
        encoding="utf-8",
    ) as handle:
        embodiments = yaml.load(handle.read(), Loader=yaml.FullLoader)
    embodiment = args["embodiment"]
    if len(embodiment) != 1:
        raise ValueError("correction collection expects a shared dual-arm embodiment")
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
    # Fast policy induction: retain exactly the observation fields consumed by
    # FullTaskTAGRTV2Runtime.  RGB is enabled again for committed correction
    # frames because the standard HDF5 converter requires a head image.
    args["data_type"].update(
        {
            "rgb": False,
            "third_view": False,
            "depth": False,
            "pointcloud": False,
            "object_pointcloud": True,
            "endpose": True,
            "qpos": False,
            "mesh_segmentation": False,
            "actor_segmentation": False,
        }
    )
    return args


def left_object_is_held(task, minimum_lift_m: float) -> bool:
    lift = float(task.shoe.get_pose().p[2]) - float(task.initial_shoe_z)
    return bool(task.is_left_gripper_close() and lift >= float(minimum_lift_m))


def main() -> None:
    args_cli = parser().parse_args()
    if args_cli.handoff_after_chunks is not None and args_cli.handoff_after_chunks < 0:
        raise ValueError("--handoff-after-chunks must be non-negative")
    args_cli.output.mkdir(parents=True, exist_ok=True)
    seeds = load_seeds(args_cli.seed_file, args_cli.episodes)
    env = environment_args(args_cli.task_name, args_cli.task_config, args_cli.output)
    model = deploy_fulltask_tagrt_v2.get_model(
        {
            "fulltask_tagrt_checkpoint": str(args_cli.checkpoint),
            "fulltask_tagrt_frame_checkpoints": [
                str(path) for path in args_cli.frame_checkpoints
            ],
            "fulltask_tagrt_camera_metadata": str(args_cli.camera_metadata),
            "fulltask_tagrt_ensemble_metadata": str(args_cli.ensemble_metadata),
            "fulltask_tagrt_device": args_cli.device,
            "fulltask_tagrt_continuous_gripper": True,
            "fulltask_tagrt_execute_steps": int(args_cli.execute_steps),
            "fulltask_tagrt_max_policy_steps": int(
                args_cli.max_policy_chunks * args_cli.execute_steps
            ),
        }
    )
    task = class_decorator(args_cli.task_name)
    records = []
    attempts = []
    committed_scene_info = {}
    for offset, seed in enumerate(seeds):
        source_episode = int(args_cli.source_episode_start) + offset
        try:
            task.setup_demo(
                now_ep_num=source_episode,
                seed=int(seed),
                is_test=True,
                **env,
            )
        except UnStableError:
            attempts.append(
                {"source_episode": source_episode, "seed": int(seed), "unstable": True}
            )
            task.close_env(clear_cache=False)
            continue
        # The schedule already used source_episode.  Committed correction files
        # are independently indexed from zero for downstream archive builders.
        correction_episode = len(records)
        task.ep_num = correction_episode
        deploy_fulltask_tagrt_v2.reset_model(model)
        first_lift_chunk = None
        first_predicted_close_chunk = None
        held = False
        handoff_ready = False
        handoff_reason = None
        for policy_chunk in range(int(args_cli.max_policy_chunks)):
            observation = task.get_obs()
            actions = model.runtime.predict(observation)
            dense_actions = model.runtime.action_representation == "dense_joint26"
            left_gripper_channel = 12 if dense_actions else 6
            imminent_left_close = bool(
                (actions[: model.execute_steps, left_gripper_channel] < 0.5).any()
            )
            if args_cli.handoff_mode == "preclose" and imminent_left_close:
                first_predicted_close_chunk = policy_chunk
                handoff_ready = True
                handoff_reason = "predicted_left_close"
                break
            if (
                args_cli.handoff_after_chunks is not None
                and policy_chunk >= int(args_cli.handoff_after_chunks)
            ):
                handoff_ready = True
                handoff_reason = "collection_chunk_budget"
                break
            deploy_fulltask_tagrt_v2.execute_predicted_actions(
                task, model, observation, actions
            )
            held = left_object_is_held(task, args_cli.minimum_lift_m)
            if held and first_lift_chunk is None:
                first_lift_chunk = policy_chunk
            if (
                args_cli.handoff_mode == "postlift"
                and
                held
                and first_lift_chunk is not None
                and policy_chunk - first_lift_chunk >= int(args_cli.post_lift_chunks)
            ):
                handoff_ready = True
                handoff_reason = "held_after_lift"
                break
            if task.eval_success or task.take_action_cnt >= task.step_lim:
                break

        attempt = {
            "episode": correction_episode,
            "source_episode": source_episode,
            "seed": int(seed),
            "shoe_id": int(task.shoe_id),
            "policy_chunks": int(model.action_chunks),
            "policy_controls": int(model.executed_actions),
            "first_lift_chunk": (
                None if first_lift_chunk is None else int(first_lift_chunk)
            ),
            "first_predicted_close_chunk": (
                None
                if first_predicted_close_chunk is None
                else int(first_predicted_close_chunk)
            ),
            "held_at_handoff": bool(held),
            "handoff_mode": str(args_cli.handoff_mode),
            "handoff_reason": handoff_reason,
        }
        if not handoff_ready:
            attempt["recovered"] = False
            attempts.append(attempt)
            task.close_env(clear_cache=False)
            print(json.dumps(attempt), flush=True)
            continue

        # Save only the expert continuation.  Resetting the dense trace makes
        # frame zero map to command -1 and keeps policy actions out of labels.
        task.data_type.update(
            {
                "rgb": True,
                "third_view": False,
                "depth": False,
                "pointcloud": False,
                "object_pointcloud": True,
                "endpose": True,
                "qpos": False,
            }
        )
        task.save_data = True
        task.FRAME_IDX = 0
        task.reset_dense_control_trace()
        task.eval_success = False
        task._take_picture()
        recovered = (
            bool(task.play_once() and task.plan_success and task.check_success())
            if args_cli.handoff_mode == "preclose"
            else task.recover_policy_placement_phase(ArmTag("left"))
        )
        attempt["recovered"] = bool(recovered and task.plan_success)
        attempt["expert_plan_success"] = bool(task.plan_success)
        attempt["correction_frames"] = int(task.FRAME_IDX)
        attempt["correction_controls"] = int(len(task.dense_control_position))
        attempts.append(deepcopy(attempt))
        task.close_env(clear_cache=False)
        if not attempt["recovered"]:
            task.remove_data_cache()
            print(json.dumps(attempt), flush=True)
            continue
        task.merge_pkl_to_hdf5_video()
        task.remove_data_cache()
        records.append(attempt)
        committed_scene_info[f"episode_{correction_episode}"] = deepcopy(task.info)
        print(json.dumps(attempt), flush=True)

    manifest = {
        "schema_version": 1,
        "protocol": "fulltask_tagrt_policy_induced_expert_continuation_v1",
        "checkpoint": str(args_cli.checkpoint.resolve()),
        "frame_checkpoints": [str(path.resolve()) for path in args_cli.frame_checkpoints],
        "camera_metadata": str(args_cli.camera_metadata.resolve()),
        "ensemble_metadata": str(args_cli.ensemble_metadata.resolve()),
        "task_config": args_cli.task_config,
        "source_episode_start": int(args_cli.source_episode_start),
        "seeds": seeds,
        "execute_steps": int(args_cli.execute_steps),
        "handoff_after_chunks": args_cli.handoff_after_chunks,
        "post_lift_chunks": int(args_cli.post_lift_chunks),
        "handoff_mode": str(args_cli.handoff_mode),
        "minimum_lift_m": float(args_cli.minimum_lift_m),
        "requested_episodes": int(args_cli.episodes),
        "collected_episodes": int(len(records)),
        "attempts": attempts,
        "records": records,
        "deployment_change": False,
        "privileged_state_role": "collection admission only",
    }
    (args_cli.output / "correction_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args_cli.output / "seed.txt").write_text(
        " ".join(str(record["seed"]) for record in records) + "\n",
        encoding="utf-8",
    )
    (args_cli.output / "scene_info.json").write_text(
        json.dumps(committed_scene_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if len(records) != int(args_cli.episodes):
        raise RuntimeError(
            f"collected {len(records)} corrections, requested {args_cli.episodes}"
        )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    from test_render import Sapien_TEST

    Sapien_TEST()
    main()
