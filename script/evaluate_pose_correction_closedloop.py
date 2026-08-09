#!/usr/bin/env python3
"""Paired camera-only closed-loop evaluation from held-out pose perturbations.

This evaluator starts after the shoe has been grasped and moved near its final
placement pose.  It applies the same deterministic, grasp-preserving SE(3)
perturbation to every compared checkpoint, then executes a fixed number of
receding-horizon learned-policy actions while keeping the gripper closed.

Simulator poses are used only for post-action metrics.  Policy inputs remain
the segmented A/B camera clouds and robot proprioception; the relative SE(3)
condition is produced online by the frozen NDF ensemble and camera marker.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from policy.GeometryFlow import deploy_interaction_flow
from script.collect_geometry_flow_recovery import build_environment_args
from script.collect_pose_correction import (
    alignment,
    parse_levels,
    prepare_perturbed_episode,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument(
        "--checkpoint",
        nargs=2,
        action="append",
        metavar=("NAME", "PATH"),
        required=True,
    )
    result.add_argument(
        "--dense-frame-checkpoint", type=Path, action="append", default=[]
    )
    result.add_argument("--camera-metadata", type=Path)
    result.add_argument("--ensemble-metadata", type=Path)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--device", default="cuda:0")
    result.add_argument(
        "--dense-frame-mode",
        choices=("clean", "inverse", "zero", "oracle"),
        default="clean",
        help=(
            "Online geometry intervention. 'zero' keeps the policy and all "
            "camera/proprioceptive inputs fixed but removes the relative "
            "functional-frame token and its confidence."
        ),
    )
    result.add_argument(
        "--allow-privileged-oracle",
        action="store_true",
        help=(
            "Enable simulator-state relative geometry only for an explicitly "
            "labeled diagnostic ceiling. Never use this for camera-only results."
        ),
    )
    result.add_argument(
        "--variant-dense-frame-mode",
        nargs=2,
        action="append",
        default=[],
        metavar=("NAME", "MODE"),
        help=(
            "Optional per-checkpoint clean/inverse/zero intervention. This "
            "permits causal variants of one checkpoint in the same exact "
            "snapshot evaluation."
        ),
    )
    result.add_argument("--episodes", default="")
    result.add_argument(
        "--levels",
        default="",
        help="Optional comma-separated manifest levels; empty evaluates all levels.",
    )
    result.add_argument("--perturbation-seed", type=int, default=20260804)
    result.add_argument("--policy-calls", type=int, default=6)
    result.add_argument("--execute-steps", type=int, default=1)
    result.add_argument(
        "--continue-after-success",
        action="store_true",
        help=(
            "Execute the same fixed number of policy calls for every paired "
            "variant; still record the first threshold-crossing call."
        ),
    )
    result.add_argument(
        "--target-latch-frames",
        type=int,
        default=3,
        help=(
            "Number of early B observations fused for the stationary target "
            "frame. Training used three frames."
        ),
    )
    result.add_argument("--success-translation-cm", type=float, default=4.0)
    result.add_argument("--success-rotation-deg", type=float, default=15.0)
    result.add_argument("--max-eef-translation-delta-m", type=float, default=0.08)
    result.add_argument("--max-eef-rotation-delta-rad", type=float, default=0.5)
    result.add_argument(
        "--maximum-manifest-initial-translation-difference-cm",
        type=float,
        default=0.25,
    )
    result.add_argument(
        "--maximum-manifest-initial-rotation-difference-deg",
        type=float,
        default=1.0,
    )
    result.add_argument(
        "--resummarize-existing",
        action="store_true",
        help="Recompute summaries in an existing output without rerunning simulation.",
    )
    result.add_argument(
        "--resume-existing",
        action="store_true",
        help=(
            "Resume an interrupted in-progress output, skipping already recorded "
            "(variant, episode) pairs after validating the frozen protocol."
        ),
    )
    result.add_argument(
        "--paired-snapshot",
        action="store_true",
        help=(
            "Prepare each perturbation once and restore an in-process SAPIEN "
            "snapshot for every checkpoint. This removes method-to-method "
            "motion-planner replay drift."
        ),
    )
    result.add_argument(
        "--snapshot-settle-steps",
        type=int,
        default=5,
        help=(
            "Identical no-action physics steps after every snapshot restore, "
            "used to rebuild contact solver state before measuring the paired "
            "initial condition."
        ),
    )
    return result


def _integer_set(value: str) -> set[int] | None:
    stripped = str(value).strip()
    if not stripped:
        return None
    return {int(item) for item in stripped.split(",") if item.strip()}


def _grasp_relation(task, arm: str) -> dict[str, list[float]]:
    """Return object functional pose in the active EEF frame for diagnostics."""

    eef = np.asarray(task.get_arm_pose(str(arm)), dtype=np.float64).reshape(7)
    functional = task.shoe.get_functional_point(0, "pose")
    object_position = np.asarray(functional.p, dtype=np.float64).reshape(3)
    object_rotation = Rotation.from_quat(
        np.asarray(functional.q, dtype=np.float64)[[1, 2, 3, 0]]
    )
    eef_rotation = Rotation.from_quat(eef[[4, 5, 6, 3]])
    relative_rotation = eef_rotation.inv() * object_rotation
    relative_position = eef_rotation.inv().apply(object_position - eef[:3])
    quaternion_xyzw = relative_rotation.as_quat()
    return {
        "position_xyz_m": relative_position.tolist(),
        "quaternion_wxyz": quaternion_xyzw[[3, 0, 1, 2]].tolist(),
    }


def _grasp_relation_drift(
    initial: dict[str, list[float]], current: dict[str, list[float]]
) -> dict[str, float]:
    initial_position = np.asarray(initial["position_xyz_m"], dtype=np.float64)
    current_position = np.asarray(current["position_xyz_m"], dtype=np.float64)
    initial_quaternion = np.asarray(initial["quaternion_wxyz"], dtype=np.float64)
    current_quaternion = np.asarray(current["quaternion_wxyz"], dtype=np.float64)
    initial_rotation = Rotation.from_quat(initial_quaternion[[1, 2, 3, 0]])
    current_rotation = Rotation.from_quat(current_quaternion[[1, 2, 3, 0]])
    return {
        "translation_cm": float(
            np.linalg.norm(current_position - initial_position) * 100.0
        ),
        "rotation_deg": float(
            np.rad2deg((initial_rotation.inv() * current_rotation).magnitude())
        ),
    }


def _capture_task_state(task) -> dict:
    """Capture the post-perturbation simulator state for paired replay."""

    actor_poses = {}
    for name, actor in (
        ("shoe", task.shoe.actor),
        ("target", task.target_block.actor),
    ):
        pose = actor.get_pose()
        actor_poses[name] = {
            "position": np.asarray(pose.p, dtype=np.float64).copy(),
            "quaternion": np.asarray(pose.q, dtype=np.float64).copy(),
        }
    shoe_dynamic_components = []
    for component_index, component in enumerate(task.shoe.actor.get_components()):
        if component.__class__.__name__ != "PhysxRigidDynamicComponent":
            continue
        shoe_dynamic_components.append(
            {
                "component_index": int(component_index),
                "linear_velocity": np.asarray(
                    component.get_linear_velocity(), dtype=np.float64
                ).copy(),
                "angular_velocity": np.asarray(
                    component.get_angular_velocity(), dtype=np.float64
                ).copy(),
                "sleeping": bool(component.is_sleeping),
            }
        )
    articulations = {}
    for name, articulation in (
        ("left", task.robot.left_entity),
        ("right", task.robot.right_entity),
    ):
        root_pose = articulation.get_root_pose()
        joints = []
        for joint in articulation.get_joints():
            joints.append(
                (
                    np.asarray(joint.get_drive_target(), dtype=np.float64).copy(),
                    np.asarray(
                        joint.get_drive_velocity_target(), dtype=np.float64
                    ).copy(),
                )
            )
        articulations[name] = {
                "root_position": np.asarray(root_pose.p, dtype=np.float64).copy(),
                "root_quaternion": np.asarray(root_pose.q, dtype=np.float64).copy(),
                "root_linear_velocity": np.asarray(
                    articulation.get_root_linear_velocity(), dtype=np.float64
                ).copy(),
                "root_angular_velocity": np.asarray(
                    articulation.get_root_angular_velocity(), dtype=np.float64
                ).copy(),
                "qpos": np.asarray(articulation.get_qpos(), dtype=np.float64).copy(),
                "qvel": np.asarray(articulation.get_qvel(), dtype=np.float64).copy(),
                "qacc": np.asarray(articulation.get_qacc(), dtype=np.float64).copy(),
                "qf": np.asarray(articulation.get_qf(), dtype=np.float64).copy(),
                "joints": joints,
            }
    task_attributes = {}
    for name in (
        "FRAME_IDX",
        "eval_success",
        "left_cnt",
        "plan_success",
        "right_cnt",
        "shoe_arm_name",
        "stage_success_tag",
        "step_lim",
        "take_action_cnt",
        "_best_combined_alignment_score",
        "_best_combined_rotation_error_deg",
        "_best_combined_translation_error_norm_m",
        "_first_pose_alignment_step",
        "_first_raw_success_step",
        "_max_success_hold_count",
        "_min_rotation_error_deg",
        "_min_translation_error_norm_m",
        "_pose_alignment_step_count",
        "_raw_success_step_count",
        "_success_hold_count",
    ):
        if hasattr(task, name):
            task_attributes[name] = deepcopy(getattr(task, name))
    robot_attributes = {}
    for name in ("left_gripper_val", "right_gripper_val"):
        if hasattr(task.robot, name):
            robot_attributes[name] = deepcopy(getattr(task.robot, name))
    return {
        "actor_poses": actor_poses,
        "shoe_dynamic_components": shoe_dynamic_components,
        "articulations": articulations,
        "task_attributes": task_attributes,
        "robot_attributes": robot_attributes,
    }


def _snapshot_to_jsonable(value):
    """Convert a semantic simulator snapshot to portable JSON primitives."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _snapshot_to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_snapshot_to_jsonable(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"snapshot contains unsupported value {type(value).__name__}")


def _restore_task_state(task, state: dict) -> None:
    """Restore a state captured by :func:`_capture_task_state` in-place."""

    import sapien

    for name, actor in (
        ("shoe", task.shoe.actor),
        ("target", task.target_block.actor),
    ):
        pose = state["actor_poses"][name]
        actor.set_pose(sapien.Pose(pose["position"], pose["quaternion"]))
    for name, articulation in (
        ("left", task.robot.left_entity),
        ("right", task.robot.right_entity),
    ):
        item = state["articulations"][name]
        articulation.set_root_pose(
            sapien.Pose(item["root_position"], item["root_quaternion"])
        )
        articulation.set_qpos(item["qpos"])
        articulation.set_qvel(item["qvel"])
        articulation.set_qacc(item["qacc"])
        articulation.set_qf(item["qf"])
        articulation.set_root_linear_velocity(item["root_linear_velocity"])
        articulation.set_root_angular_velocity(item["root_angular_velocity"])
        for joint, (target, velocity_target) in zip(
            articulation.get_joints(), item["joints"], strict=True
        ):
            target = np.asarray(target, dtype=np.float64)
            velocity_target = np.asarray(velocity_target, dtype=np.float64)
            if target.size:
                joint.set_drive_target(
                    float(target.item()) if target.size == 1 else target
                )
            if velocity_target.size:
                joint.set_drive_velocity_target(
                    float(velocity_target.item())
                    if velocity_target.size == 1
                    else velocity_target
                )
    shoe_components = task.shoe.actor.get_components()
    for item in state["shoe_dynamic_components"]:
        component = shoe_components[
            int(item["component_index"])
        ]
        component.set_linear_velocity(item["linear_velocity"])
        component.set_angular_velocity(item["angular_velocity"])
        if item["sleeping"]:
            component.put_to_sleep()
        else:
            component.wake_up()
    for name, value in state["task_attributes"].items():
        setattr(task, name, deepcopy(value))
    for name, value in state["robot_attributes"].items():
        setattr(task.robot, name, deepcopy(value))
    task.left_js = None
    task.right_js = None
    task.scene.update_render()


def _selected_records(manifest: dict, episodes: set[int] | None, levels: set[int] | None):
    records = []
    for raw in manifest.get("records", []):
        record = dict(raw)
        if episodes is not None and int(record["episode"]) not in episodes:
            continue
        if levels is not None and int(record["level"]) not in levels:
            continue
        records.append(record)
    if not records:
        raise ValueError("manifest selection contains no episodes")
    return records


def _model_arguments(
    args: argparse.Namespace,
    checkpoint: Path,
    *,
    dense_frame_mode: str | None = None,
) -> dict:
    return {
        "interaction_flow_checkpoint": str(checkpoint.resolve()),
        "interaction_flow_device": str(args.device),
        "interaction_flow_observation_points": 128,
        "interaction_flow_execute_steps": int(args.execute_steps),
        "interaction_flow_max_policy_steps": int(args.policy_calls)
        * int(args.execute_steps),
        "interaction_flow_temporal_ensemble_decay": 0.0,
        "interaction_flow_route_to_closed_gripper": True,
        "interaction_flow_hold_active_gripper_closed": True,
        "max_eef_translation_delta_m": float(args.max_eef_translation_delta_m),
        "max_eef_rotation_delta_rad": float(args.max_eef_rotation_delta_rad),
        "interaction_flow_dense_frame_checkpoints": [
            str(path.resolve()) for path in args.dense_frame_checkpoint
        ],
        "interaction_flow_dense_camera_metadata": (
            "" if args.camera_metadata is None else str(args.camera_metadata.resolve())
        ),
        "interaction_flow_dense_ensemble_metadata": (
            ""
            if args.ensemble_metadata is None
            else str(args.ensemble_metadata.resolve())
        ),
        "interaction_flow_dense_frame_mode": str(
            args.dense_frame_mode if dense_frame_mode is None else dense_frame_mode
        ),
        "interaction_flow_allow_privileged_oracle": bool(
            args.allow_privileged_oracle
        ),
    }


def _checkpoint_uses_dense_camera_frame(checkpoint: Path) -> bool:
    """Return whether a policy checkpoint requires the online NDF frame.

    A factorial ablation mixes plain point/relation policies with the dense
    functional-frame policy in one exact paired simulator snapshot.  Plain
    policies must receive ``dense_frame_mode=off`` while dense policies retain
    the requested clean/zero intervention.
    """

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return bool(
        payload.get("condition")
        in deploy_interaction_flow.InteractionFlowRuntime.FUNCTIONAL_FRAME_CONDITIONS
        and payload.get("functional_frame_encoding")
        in {
            "camera_ndf_current_marker_goal_se3_columns_v1",
            "dataset_relative_se3_columns_v1",
        }
    )


def _write_output(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _variant_summary(records: list[dict], args: argparse.Namespace) -> dict:
    completed = [record for record in records if record.get("status") == "complete"]
    if not completed:
        return {
            "attempted_episodes": int(len(records)),
            "episodes": 0,
            "errors": int(len(records)),
        }
    initial_translation = np.asarray(
        [record["initial"]["translation_m"] for record in completed], dtype=np.float64
    )
    initial_rotation = np.asarray(
        [record["initial"]["rotation_deg"] for record in completed], dtype=np.float64
    )
    final_translation = np.asarray(
        [record["final"]["translation_m"] for record in completed], dtype=np.float64
    )
    final_rotation = np.asarray(
        [record["final"]["rotation_deg"] for record in completed], dtype=np.float64
    )
    initial_success = (
        initial_translation * 100.0 <= float(args.success_translation_cm)
    ) & (initial_rotation <= float(args.success_rotation_deg))
    success = (
        final_translation * 100.0 <= float(args.success_translation_cm)
    ) & (final_rotation <= float(args.success_rotation_deg))
    eligible = ~initial_success
    recovered = success & eligible
    return {
        "attempted_episodes": int(len(records)),
        "episodes": int(len(completed)),
        "errors": int(len(records) - len(completed)),
        "initial_successes": int(initial_success.sum()),
        "recovery_eligible_episodes": int(eligible.sum()),
        "successes": int(success.sum()),
        "success_rate": float(success.mean()),
        "recovered_from_initial_failure": int(recovered.sum()),
        "recovery_success_rate": (
            float(recovered.sum() / eligible.sum()) if eligible.any() else None
        ),
        "initial_translation_cm_mean": float(initial_translation.mean() * 100.0),
        "final_translation_cm_mean": float(final_translation.mean() * 100.0),
        "translation_reduction_cm_mean": float(
            (initial_translation - final_translation).mean() * 100.0
        ),
        "initial_rotation_deg_mean": float(initial_rotation.mean()),
        "final_rotation_deg_mean": float(final_rotation.mean()),
        "rotation_reduction_deg_mean": float(
            (initial_rotation - final_rotation).mean()
        ),
    }


def _pose_threshold_met(errors: dict[str, float], args: argparse.Namespace) -> bool:
    return bool(
        float(errors["translation_m"]) * 100.0
        <= float(args.success_translation_cm)
        and float(errors["rotation_deg"]) <= float(args.success_rotation_deg)
    )


def _paired_summary(
    results: dict[str, list[dict]],
    *,
    maximum_initial_translation_difference_cm: float = 0.25,
    maximum_initial_rotation_difference_deg: float = 1.0,
) -> dict:
    names = list(results)
    if len(names) < 2:
        return {}
    baseline = {
        int(record["episode"]): record
        for record in results[names[0]]
        if record.get("status") == "complete"
    }
    output = {}
    for name in names[1:]:
        candidate = {
            int(record["episode"]): record
            for record in results[name]
            if record.get("status") == "complete"
        }
        shared_complete = sorted(set(baseline) & set(candidate))
        matched = []
        excluded = []
        for index in shared_complete:
            translation_difference_cm = abs(
                float(baseline[index]["initial"]["translation_m"])
                - float(candidate[index]["initial"]["translation_m"])
            ) * 100.0
            rotation_difference_deg = abs(
                float(baseline[index]["initial"]["rotation_deg"])
                - float(candidate[index]["initial"]["rotation_deg"])
            )
            audit = {
                "episode": int(index),
                "initial_translation_difference_cm": float(
                    translation_difference_cm
                ),
                "initial_rotation_difference_deg": float(rotation_difference_deg),
            }
            if (
                translation_difference_cm
                <= float(maximum_initial_translation_difference_cm)
                and rotation_difference_deg
                <= float(maximum_initial_rotation_difference_deg)
            ):
                matched.append(index)
            else:
                excluded.append(audit)
        if not matched:
            output[f"{name}_vs_{names[0]}"] = {
                "episodes": [],
                "shared_complete_episodes": shared_complete,
                "excluded_initial_mismatch": excluded,
            }
            continue
        translation = np.asarray(
            [
                baseline[index]["final"]["translation_m"]
                - candidate[index]["final"]["translation_m"]
                for index in matched
            ],
            dtype=np.float64,
        )
        rotation = np.asarray(
            [
                baseline[index]["final"]["rotation_deg"]
                - candidate[index]["final"]["rotation_deg"]
                for index in matched
            ],
            dtype=np.float64,
        )
        output[f"{name}_vs_{names[0]}"] = {
            "episodes": matched,
            "shared_complete_episodes": shared_complete,
            "excluded_initial_mismatch": excluded,
            "maximum_initial_translation_difference_cm": float(
                maximum_initial_translation_difference_cm
            ),
            "maximum_initial_rotation_difference_deg": float(
                maximum_initial_rotation_difference_deg
            ),
            "final_translation_improvement_cm_mean": float(
                translation.mean() * 100.0
            ),
            "final_translation_better_fraction": float((translation > 0.0).mean()),
            "final_rotation_improvement_deg_mean": float(rotation.mean()),
            "final_rotation_better_fraction": float((rotation > 0.0).mean()),
        }
    return output


def _evaluate_current_snapshot(
    task,
    model,
    *,
    early_observations: list[dict],
    arm: str,
    initial: dict[str, float],
    args: argparse.Namespace,
) -> dict:
    """Run one policy from the task's current, already-restored state."""

    deploy_interaction_flow.reset_model(model)
    target_latch = deploy_interaction_flow.latch_dense_target_observations(
        model, early_observations
    )
    trajectory = [dict(initial)]
    first_success_call = 0 if _pose_threshold_met(initial, args) else None
    initial_grasp_relation = _grasp_relation(task, str(arm))
    grasp_relation_trajectory = [initial_grasp_relation]
    grasp_slip_trajectory = [{"translation_cm": 0.0, "rotation_deg": 0.0}]
    policy_metrics_trajectory = []
    for call_index in range(int(args.policy_calls)):
        if first_success_call is not None and not args.continue_after_success:
            break
        observation = task.get_obs()
        deploy_interaction_flow.eval(task, model, observation)
        current_alignment = alignment(task)
        trajectory.append(current_alignment)
        current_grasp_relation = _grasp_relation(task, str(arm))
        grasp_relation_trajectory.append(current_grasp_relation)
        grasp_slip_trajectory.append(
            _grasp_relation_drift(initial_grasp_relation, current_grasp_relation)
        )
        policy_metrics_trajectory.append(dict(model.get_evaluation_metrics()))
        if (
            first_success_call is None
            and _pose_threshold_met(current_alignment, args)
        ):
            first_success_call = int(call_index) + 1
            if not args.continue_after_success:
                break
    return {
        "status": "complete",
        "target_latch": target_latch,
        "final": trajectory[-1],
        "trajectory": trajectory,
        "requested_policy_calls": int(args.policy_calls),
        "executed_policy_calls": int(len(policy_metrics_trajectory)),
        "initial_success": bool(first_success_call == 0),
        "first_success_call": first_success_call,
        "grasp_relation_trajectory": grasp_relation_trajectory,
        "grasp_slip_trajectory": grasp_slip_trajectory,
        "policy_metrics_trajectory": policy_metrics_trajectory,
        "policy_metrics": model.get_evaluation_metrics(),
    }


def _commit_result(
    *,
    output: dict,
    output_path: Path,
    name: str,
    record: dict,
    episode: int,
    args: argparse.Namespace,
) -> None:
    output["results"][name].append(record)
    output["summaries"] = {
        variant: _variant_summary(values, args)
        for variant, values in output["results"].items()
    }
    output["paired"] = _paired_summary(output["results"])
    _write_output(output_path, output)
    progress = {
        "variant": str(name),
        "episode": int(episode),
        "status": str(record["status"]),
    }
    if record["status"] == "complete":
        progress.update(
            {
                "initial": record["initial"],
                "final": record["final"],
                "first_success_call": record["first_success_call"],
                "executed_policy_calls": record["executed_policy_calls"],
                "final_grasp_slip": record["grasp_slip_trajectory"][-1],
            }
        )
    else:
        progress["error"] = record.get("error")
    print(json.dumps(progress), flush=True)


def _validate_resume_output(
    *,
    existing: dict,
    expected: dict,
    records: list[dict],
) -> dict[str, set[int]]:
    """Validate an interrupted output and return completed episodes per variant."""

    invariant_keys = (
        "schema_version",
        "protocol",
        "manifest",
        "policy_calls",
        "execute_steps",
        "continue_after_success",
        "target_latch_frames",
        "snapshot_settle_steps",
        "dense_frame_mode",
        "variant_dense_frame_modes",
        "perturbation_seed",
        "success_threshold",
        "manifest_replay_tolerance",
        "policy_inputs",
        "simulator_pose_usage",
        "setup_is_test",
    )
    for key in invariant_keys:
        if existing.get(key) != expected.get(key):
            raise ValueError(
                f"resume output protocol mismatch for {key!r}: "
                f"{existing.get(key)!r} != {expected.get(key)!r}"
            )
    if set(existing.get("results", {})) != set(expected["results"]):
        raise ValueError(
            "resume output variants do not match requested checkpoints: "
            f"{sorted(existing.get('results', {}))!r} != "
            f"{sorted(expected['results'])!r}"
        )

    selected = {int(record["episode"]): record for record in records}
    completed: dict[str, set[int]] = {}
    for name, values in existing["results"].items():
        episodes: set[int] = set()
        for value in values:
            episode = int(value["episode"])
            if episode in episodes:
                raise ValueError(
                    f"resume output has duplicate {name!r} episode {episode}"
                )
            if episode not in selected:
                raise ValueError(
                    f"resume output has {name!r} episode {episode} outside selection"
                )
            source = selected[episode]
            for key in ("scene_seed", "shoe_id", "level"):
                if int(value[key]) != int(source[key]):
                    raise ValueError(
                        f"resume output {name!r} episode {episode} mismatches "
                        f"manifest {key}: {value[key]!r} != {source[key]!r}"
                    )
            episodes.add(episode)
        completed[str(name)] = episodes
    return completed


def _run_paired_snapshot_evaluation(
    *,
    class_decorator,
    manifest: dict,
    environment: dict,
    models: dict,
    records: list[dict],
    levels,
    output: dict,
    args: argparse.Namespace,
    completed: dict[str, set[int]] | None = None,
) -> None:
    """Evaluate variants from one portable state in independent fresh scenes."""

    output["paired_state_replay"] = "fresh_scene_portable_sapien_snapshot_v2"
    preparation_task = class_decorator(str(manifest["task_name"]))
    variant_tasks = {
        name: class_decorator(str(manifest["task_name"])) for name in models
    }
    for source in records:
        episode = int(source["episode"])
        pending_names = [
            name
            for name in models
            if episode not in (completed or {}).get(str(name), set())
        ]
        if not pending_names:
            continue
        scene_seed = int(source["scene_seed"])
        level_index = int(source["level"])
        common = {
            "episode": episode,
            "scene_seed": scene_seed,
            "shoe_id": int(source["shoe_id"]),
            "level": level_index,
        }
        episode_environment = deepcopy(environment)
        if str(manifest["task_name"]) == "place_handled_mug_geometry_marker":
            episode_environment["container_geometry"] = {
                "allowed_container_ids": [int(source["shoe_id"])]
            }
        task = preparation_task
        preparation_error = None
        try:
            task.setup_demo(
                now_ep_num=episode,
                seed=scene_seed,
                is_test=bool(manifest.get("setup_is_test", True)),
                **episode_environment,
            )
            # Task observations expose the active-arm identity.  Restore this
            # semantic state before even the optional early target latch.
            task.shoe_arm_name = str(source["arm"])
            early_observations = [
                task.get_obs() for _ in range(int(args.target_latch_frames))
            ]
            if int(task.shoe_id) != int(source["shoe_id"]):
                raise RuntimeError(
                    f"shoe identity drifted: {task.shoe_id} != {source['shoe_id']}"
                )
            if "semantic_snapshot" in source:
                # The benchmark collector admits a state before any policy is
                # evaluated. Reuse that post-perturbation state directly and
                # avoid a second motion-planner replay during evaluation.
                snapshot = deepcopy(source["semantic_snapshot"])
                arm = str(source["arm"])
                perturbation = dict(source["commanded_perturbation"])
                initial = dict(source["initial"])
                common["perturbation_seed"] = int(source["perturbation_seed"])
                common["perturbation_replay_source"] = "stored_semantic_snapshot"
                common["manifest_initial_translation_difference_cm"] = 0.0
                common["manifest_initial_rotation_difference_deg"] = 0.0
            else:
                perturbation_seed = int(
                    source.get(
                        "perturbation_seed",
                        int(args.perturbation_seed) + episode,
                    )
                )
                common["perturbation_seed"] = perturbation_seed
                common["perturbation_replay_source"] = (
                    "manifest"
                    if "perturbation_seed" in source
                    else "legacy_evaluator_seed_plus_episode"
                )
                generator = np.random.default_rng(perturbation_seed)
                arm, _final_pose, perturbation, initial = prepare_perturbed_episode(
                    task, levels[level_index], generator
                )
                if "before" in source:
                    manifest_initial = source["before"]
                    translation_difference_cm = abs(
                        float(initial["translation_m"])
                        - float(manifest_initial["translation_m"])
                    ) * 100.0
                    rotation_difference_deg = abs(
                        float(initial["rotation_deg"])
                        - float(manifest_initial["rotation_deg"])
                    )
                    common["manifest_initial_translation_difference_cm"] = float(
                        translation_difference_cm
                    )
                    common["manifest_initial_rotation_difference_deg"] = float(
                        rotation_difference_deg
                    )
                    if (
                        translation_difference_cm
                        > float(args.maximum_manifest_initial_translation_difference_cm)
                        or rotation_difference_deg
                        > float(args.maximum_manifest_initial_rotation_difference_deg)
                    ):
                        raise RuntimeError(
                            "replayed initial state exceeds manifest tolerance: "
                            f"{translation_difference_cm:.4f} cm / "
                            f"{rotation_difference_deg:.4f} deg"
                        )
                snapshot = _capture_task_state(task)
            task.eval_success = False
            common.update(
                {
                    "arm": str(arm),
                    "commanded_perturbation": perturbation,
                    "initial": dict(initial),
                }
            )
        except Exception as error:
            preparation_error = f"{type(error).__name__}: {error}"
        try:
            task.close_env(clear_cache=False)
        except Exception:
            pass

        for _variant_index, name in enumerate(pending_names):
            model = models[name]
            record = deepcopy(common)
            variant_task = None
            if preparation_error is not None:
                record.update({"status": "error", "error": preparation_error})
            else:
                try:
                    # Every variant receives a fresh physics scene (and hence
                    # an identical empty contact-solver cache) before the
                    # portable post-perturbation state is restored.
                    variant_task = variant_tasks[name]
                    variant_task.setup_demo(
                        now_ep_num=episode,
                        seed=scene_seed,
                        is_test=bool(manifest.get("setup_is_test", True)),
                        **episode_environment,
                    )
                    _restore_task_state(variant_task, snapshot)
                    # The active arm is semantic episode state rather than a
                    # SAPIEN actor property.  Older portable manifests did not
                    # serialize it, so restore it explicitly from the admitted
                    # record before the first observation.
                    variant_task.shoe_arm_name = str(arm)
                    for _ in range(int(args.snapshot_settle_steps)):
                        variant_task.scene.step()
                    variant_task.scene.update_render()
                    replay_initial = alignment(variant_task)
                    record["prepared_initial"] = dict(initial)
                    record["initial"] = dict(replay_initial)
                    record["snapshot_settle_drift"] = {
                        "translation_cm": float(
                            abs(
                                replay_initial["translation_m"]
                                - initial["translation_m"]
                            )
                            * 100.0
                        ),
                        "rotation_deg": float(
                            abs(
                                replay_initial["rotation_deg"]
                                - initial["rotation_deg"]
                            )
                        ),
                    }
                    record.update(
                        _evaluate_current_snapshot(
                            variant_task,
                            model,
                            early_observations=early_observations,
                            arm=str(arm),
                            initial=replay_initial,
                            args=args,
                        )
                    )
                except Exception as error:
                    record.update(
                        {
                            "status": "error",
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                finally:
                    if variant_task is not None:
                        try:
                            variant_task.close_env(clear_cache=False)
                        except Exception:
                            pass
            _commit_result(
                output=output,
                output_path=args.output,
                name=str(name),
                record=record,
                episode=episode,
                args=args,
            )


def main() -> None:
    from script.collect_data import class_decorator

    args = parser().parse_args()
    if args.resummarize_existing and args.resume_existing:
        raise ValueError(
            "--resummarize-existing and --resume-existing are mutually exclusive"
        )
    if (
        int(args.policy_calls) < 1
        or int(args.execute_steps) < 1
        or int(args.target_latch_frames) < 1
        or int(args.snapshot_settle_steps) < 0
    ):
        raise ValueError(
            "policy-calls, execute-steps, and target-latch-frames must be "
            "positive; snapshot-settle-steps must be nonnegative"
        )
    if args.resummarize_existing:
        if not args.output.is_file():
            raise FileNotFoundError(args.output)
        output = json.loads(args.output.read_text(encoding="utf-8"))
        output["summaries"] = {
            variant: _variant_summary(values, args)
            for variant, values in output["results"].items()
        }
        output["paired"] = _paired_summary(output["results"])
        _write_output(args.output, output)
        print(
            json.dumps(
                {"output": str(args.output), "paired": output["paired"]}
            ),
            flush=True,
        )
        return
    if args.output.exists() and not args.resume_existing:
        raise FileExistsError(f"refuse to overwrite closed-loop result: {args.output}")
    if args.resume_existing and not args.output.is_file():
        raise FileNotFoundError(
            f"--resume-existing requires an interrupted output: {args.output}"
        )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = _selected_records(
        manifest, _integer_set(args.episodes), _integer_set(args.levels)
    )
    levels = parse_levels(str(manifest["levels"]))
    dense_requested = any(
        _checkpoint_uses_dense_camera_frame(Path(checkpoint))
        for _name, checkpoint in args.checkpoint
    )
    if dense_requested and (
        not args.dense_frame_checkpoint
        or args.camera_metadata is None
        or args.ensemble_metadata is None
    ):
        raise ValueError(
            "dense functional-frame checkpoints require frame checkpoints, "
            "camera metadata, and ensemble metadata"
        )
    optional_dense_paths = [
        path
        for path in (args.camera_metadata, args.ensemble_metadata)
        if path is not None
    ]
    for path in (
        args.manifest,
        *(Path(value[1]) for value in args.checkpoint),
        *optional_dense_paths,
        *args.dense_frame_checkpoint,
    ):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    environment = build_environment_args(
        str(manifest["task_name"]), str(manifest["task_config"]), args.output.parent
    )
    if str(manifest["task_name"]) == "place_container_plate":
        allowed_objects = [int(value) for value in manifest["allowed_shoes"]]
        environment["container_geometry"] = {
            "allowed_categories": ["021_cup"],
            "allowed_container_ids": allowed_objects,
        }
    elif str(manifest["task_name"]) == "place_handled_mug_geometry_marker":
        environment["container_geometry"] = {
            "allowed_container_ids": [
                int(value) for value in manifest["allowed_shoes"]
            ]
        }
    environment["data_type"].update(
        {"third_view": False, "pointcloud": False, "qpos": False}
    )
    checkpoint_names = [str(name) for name, _checkpoint in args.checkpoint]
    if len(set(checkpoint_names)) != len(checkpoint_names):
        raise ValueError("checkpoint variant names must be unique")
    variant_dense_modes = {
        str(name): (
            str(args.dense_frame_mode)
            if _checkpoint_uses_dense_camera_frame(Path(checkpoint))
            else "off"
        )
        for name, checkpoint in args.checkpoint
    }
    seen_mode_overrides = set()
    for raw_name, raw_mode in args.variant_dense_frame_mode:
        name, mode = str(raw_name), str(raw_mode)
        if name not in variant_dense_modes:
            raise ValueError(f"dense-frame override names unknown checkpoint {name!r}")
        if name in seen_mode_overrides:
            raise ValueError(f"duplicate dense-frame override for {name!r}")
        if mode not in {"off", "clean", "inverse", "zero", "oracle"}:
            raise ValueError(
                "dense-frame override must be off, clean, inverse, zero, or oracle, "
                f"got {mode!r}"
            )
        variant_dense_modes[name] = mode
        seen_mode_overrides.add(name)
    if "oracle" in variant_dense_modes.values() and not args.allow_privileged_oracle:
        raise PermissionError(
            "oracle dense-frame mode requires --allow-privileged-oracle"
        )
    models = {
        str(name): deploy_interaction_flow.get_model(
            _model_arguments(
                args,
                Path(checkpoint),
                dense_frame_mode=variant_dense_modes[str(name)],
            )
        )
        for name, checkpoint in args.checkpoint
    }
    expected_output = {
        "schema_version": 1,
        "protocol": "paired_camera_only_gripper_closed_pose_correction_v1",
        "manifest": str(args.manifest.resolve()),
        "policy_calls": int(args.policy_calls),
        "execute_steps": int(args.execute_steps),
        "continue_after_success": bool(args.continue_after_success),
        "target_latch_frames": int(args.target_latch_frames),
        "snapshot_settle_steps": int(args.snapshot_settle_steps),
        "dense_frame_mode": str(args.dense_frame_mode),
        "variant_dense_frame_modes": variant_dense_modes,
        "perturbation_seed": int(args.perturbation_seed),
        "success_threshold": {
            "translation_cm": float(args.success_translation_cm),
            "rotation_deg": float(args.success_rotation_deg),
        },
        "manifest_replay_tolerance": {
            "translation_cm": float(
                args.maximum_manifest_initial_translation_difference_cm
            ),
            "rotation_deg": float(
                args.maximum_manifest_initial_rotation_difference_deg
            ),
        },
        "policy_inputs": "camera A/B point clouds plus proprioception",
        "simulator_pose_usage": (
            "metrics plus explicitly privileged oracle diagnostic"
            if "oracle" in variant_dense_modes.values()
            else "metrics only"
        ),
        "setup_is_test": bool(manifest.get("setup_is_test", True)),
        "checkpoint_sources": {
            str(name): str(Path(checkpoint).resolve())
            for name, checkpoint in args.checkpoint
        },
        "dense_frame_checkpoint_sources": [
            str(path.resolve()) for path in args.dense_frame_checkpoint
        ],
        "camera_metadata_source": (
            None if args.camera_metadata is None else str(args.camera_metadata.resolve())
        ),
        "ensemble_metadata_source": (
            None
            if args.ensemble_metadata is None
            else str(args.ensemble_metadata.resolve())
        ),
        "results": {name: [] for name in models},
        "status": "in_progress",
    }
    completed: dict[str, set[int]] = {name: set() for name in models}
    if args.resume_existing:
        output = json.loads(args.output.read_text(encoding="utf-8"))
        completed = _validate_resume_output(
            existing=output,
            expected=expected_output,
            records=records,
        )
        provenance_keys = (
            "checkpoint_sources",
            "dense_frame_checkpoint_sources",
            "camera_metadata_source",
            "ensemble_metadata_source",
        )
        for key in provenance_keys:
            if key in output and output[key] != expected_output[key]:
                raise ValueError(
                    f"resume output provenance mismatch for {key!r}: "
                    f"{output[key]!r} != {expected_output[key]!r}"
                )
            output[key] = expected_output[key]
        output["status"] = "in_progress"
    else:
        output = expected_output
    _write_output(args.output, output)

    if args.paired_snapshot:
        _run_paired_snapshot_evaluation(
            class_decorator=class_decorator,
            manifest=manifest,
            environment=environment,
            models=models,
            records=records,
            levels=levels,
            output=output,
            args=args,
            completed=completed,
        )
        output["status"] = "complete"
        output["summaries"] = {
            variant: _variant_summary(values, args)
            for variant, values in output["results"].items()
        }
        output["paired"] = _paired_summary(output["results"])
        _write_output(args.output, output)
        print(
            json.dumps({"output": str(args.output), **output["summaries"]}),
            flush=True,
        )
        return

    for name, model in models.items():
        task = class_decorator(str(manifest["task_name"]))
        for source in records:
            episode = int(source["episode"])
            if episode in completed.get(str(name), set()):
                continue
            scene_seed = int(source["scene_seed"])
            level_index = int(source["level"])
            record = {
                "episode": episode,
                "scene_seed": scene_seed,
                "shoe_id": int(source["shoe_id"]),
                "level": level_index,
            }
            episode_environment = deepcopy(environment)
            if str(manifest["task_name"]) == "place_handled_mug_geometry_marker":
                episode_environment["container_geometry"] = {
                    "allowed_container_ids": [int(source["shoe_id"])]
                }
            try:
                task.setup_demo(
                    now_ep_num=episode,
                    seed=scene_seed,
                    is_test=bool(manifest.get("setup_is_test", True)),
                    **episode_environment,
                )
                # Match training: cache the stationary target from the first
                # three task observations, before the nominal demonstration can
                # occlude its marker near the correction endpoint.
                deploy_interaction_flow.reset_model(model)
                target_latch = deploy_interaction_flow.latch_dense_target_observations(
                    model,
                    [
                        task.get_obs()
                        for _ in range(int(args.target_latch_frames))
                    ],
                )
                record["target_latch"] = target_latch
                perturbation_seed = int(
                    source.get(
                        "perturbation_seed",
                        int(args.perturbation_seed) + episode,
                    )
                )
                record["perturbation_seed"] = perturbation_seed
                record["perturbation_replay_source"] = (
                    "manifest"
                    if "perturbation_seed" in source
                    else "legacy_evaluator_seed_plus_episode"
                )
                generator = np.random.default_rng(perturbation_seed)
                arm, _final_pose, perturbation, initial = prepare_perturbed_episode(
                    task, levels[level_index], generator
                )
                if "before" in source:
                    manifest_initial = source["before"]
                    translation_difference_cm = abs(
                        float(initial["translation_m"])
                        - float(manifest_initial["translation_m"])
                    ) * 100.0
                    rotation_difference_deg = abs(
                        float(initial["rotation_deg"])
                        - float(manifest_initial["rotation_deg"])
                    )
                    record["manifest_initial_translation_difference_cm"] = float(
                        translation_difference_cm
                    )
                    record["manifest_initial_rotation_difference_deg"] = float(
                        rotation_difference_deg
                    )
                    if (
                        translation_difference_cm
                        > float(
                            args.maximum_manifest_initial_translation_difference_cm
                        )
                        or rotation_difference_deg
                        > float(
                            args.maximum_manifest_initial_rotation_difference_deg
                        )
                    ):
                        raise RuntimeError(
                            "replayed initial state exceeds manifest tolerance: "
                            f"{translation_difference_cm:.4f} cm / "
                            f"{rotation_difference_deg:.4f} deg"
                        )
                if int(task.shoe_id) != int(source["shoe_id"]):
                    raise RuntimeError(
                        f"shoe identity drifted: {task.shoe_id} != {source['shoe_id']}"
                    )
                task.eval_success = False
                trajectory = [dict(initial)]
                first_success_call = 0 if _pose_threshold_met(initial, args) else None
                initial_grasp_relation = _grasp_relation(task, str(arm))
                grasp_relation_trajectory = [initial_grasp_relation]
                grasp_slip_trajectory = [
                    {"translation_cm": 0.0, "rotation_deg": 0.0}
                ]
                policy_metrics_trajectory = []
                for call_index in range(int(args.policy_calls)):
                    if (
                        first_success_call is not None
                        and not args.continue_after_success
                    ):
                        break
                    observation = task.get_obs()
                    deploy_interaction_flow.eval(task, model, observation)
                    current_alignment = alignment(task)
                    trajectory.append(current_alignment)
                    current_grasp_relation = _grasp_relation(task, str(arm))
                    grasp_relation_trajectory.append(current_grasp_relation)
                    grasp_slip_trajectory.append(
                        _grasp_relation_drift(
                            initial_grasp_relation, current_grasp_relation
                        )
                    )
                    policy_metrics_trajectory.append(
                        dict(model.get_evaluation_metrics())
                    )
                    if (
                        first_success_call is None
                        and _pose_threshold_met(current_alignment, args)
                    ):
                        first_success_call = int(call_index) + 1
                        if not args.continue_after_success:
                            break
                record.update(
                    {
                        "status": "complete",
                        "arm": str(arm),
                        "commanded_perturbation": perturbation,
                        "initial": trajectory[0],
                        "final": trajectory[-1],
                        "trajectory": trajectory,
                        "requested_policy_calls": int(args.policy_calls),
                        "executed_policy_calls": int(len(policy_metrics_trajectory)),
                        "initial_success": bool(first_success_call == 0),
                        "first_success_call": first_success_call,
                        "grasp_relation_trajectory": grasp_relation_trajectory,
                        "grasp_slip_trajectory": grasp_slip_trajectory,
                        "policy_metrics_trajectory": policy_metrics_trajectory,
                        "policy_metrics": model.get_evaluation_metrics(),
                    }
                )
            except Exception as error:
                record.update(
                    {
                        "status": "error",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            finally:
                try:
                    task.close_env(clear_cache=False)
                except Exception:
                    pass
            output["results"][name].append(record)
            output["summaries"] = {
                variant: _variant_summary(values, args)
                for variant, values in output["results"].items()
            }
            output["paired"] = _paired_summary(output["results"])
            _write_output(args.output, output)
            progress = {
                "variant": str(name),
                "episode": int(episode),
                "status": str(record["status"]),
            }
            if record["status"] == "complete":
                progress.update(
                    {
                        "initial": record["initial"],
                        "final": record["final"],
                        "first_success_call": record["first_success_call"],
                        "executed_policy_calls": record["executed_policy_calls"],
                        "final_grasp_slip": record["grasp_slip_trajectory"][-1],
                    }
                )
            else:
                progress["error"] = record.get("error")
            print(json.dumps(progress), flush=True)

    output["status"] = "complete"
    output["summaries"] = {
        variant: _variant_summary(values, args)
        for variant, values in output["results"].items()
    }
    output["paired"] = _paired_summary(output["results"])
    _write_output(args.output, output)
    print(json.dumps({"output": str(args.output), **output["summaries"]}), flush=True)


if __name__ == "__main__":
    if "--resummarize-existing" not in sys.argv:
        from test_render import Sapien_TEST

        Sapien_TEST()
    main()
