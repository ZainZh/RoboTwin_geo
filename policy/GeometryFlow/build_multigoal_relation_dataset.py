#!/usr/bin/env python3
"""Build physically consistent multi-goal pairs for relation-identifiable BC.

Each source row keeps its current object cloud, robot state, and NDF-estimated
current functional frame.  It is paired with several real camera target clouds
from the same object-disjoint split.  Label-only simulator poses construct a
smooth rigid-grasp expert trajectory to the paired goal.  They are never
policy inputs.

This differs from token shuffling: B, the camera goal frame, the two-level NDF
relation, the action chunk, and the auxiliary object flow all describe the
same counterfactual task.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .build_task_flow_dataset import eef_delta_action14
from .canonicalize_task_flow_dataset import matrix_to_pose7, pose7_to_matrix
from .functional_action_frame import frame9_rotation, pose9_rotation
from .augment_handled_mug_ndf_relation import rotation_error_deg


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--train-ids", nargs="+", type=int, required=True)
    result.add_argument("--validation-ids", nargs="+", type=int, required=True)
    result.add_argument("--test-ids", nargs="+", type=int, required=True)
    result.add_argument("--goals-per-source", type=int, default=4)
    result.add_argument("--flow-steps", type=int, default=16)
    result.add_argument(
        "--relation-only",
        action="store_true",
        help="Use only post-grasp relation-phase source rows.",
    )
    result.add_argument(
        "--max-sources-per-episode",
        type=int,
        default=0,
        help=(
            "Uniformly subsample at most this many source rows per episode; "
            "zero keeps every eligible row."
        ),
    )
    result.add_argument(
        "--maximum-relation-translation-cm",
        type=float,
        help="Keep source and donor goals within this short-horizon distance.",
    )
    result.add_argument(
        "--maximum-relation-rotation-deg",
        type=float,
        help="Keep source and donor goals within this short-horizon angle.",
    )
    result.add_argument("--overwrite", action="store_true")
    return result


def frame9_from_matrix(transform: np.ndarray) -> np.ndarray:
    value = np.asarray(transform, dtype=np.float64)
    return np.concatenate(
        (value[:3, 3], value[:3, 0], value[:3, 1])
    ).astype(np.float32)


def pose9_from_matrix(transform: np.ndarray) -> np.ndarray:
    """Encode historical object pose9 (row-major R[:, :2])."""

    value = np.asarray(transform, dtype=np.float64)
    return np.concatenate(
        (value[:3, 3], value[:3, :2].reshape(6))
    ).astype(np.float32)


def matrix_from_pose9(pose9: np.ndarray) -> np.ndarray:
    value = np.asarray(pose9, dtype=np.float64)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = pose9_rotation(value)
    result[:3, 3] = value[:3]
    return result


def matrix_from_frame9(frame9: np.ndarray) -> np.ndarray:
    value = np.asarray(frame9, dtype=np.float64)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = frame9_rotation(value)
    result[:3, 3] = value[:3]
    return result


def interpolate_pose(
    source: np.ndarray, target: np.ndarray, fraction: float
) -> np.ndarray:
    """Interpolate one proper SE(3) pose without quaternion sign ambiguity."""

    start = np.asarray(source, dtype=np.float64)
    finish = np.asarray(target, dtype=np.float64)
    fraction = float(fraction)
    relative_vector = Rotation.from_matrix(
        start[:3, :3].T @ finish[:3, :3]
    ).as_rotvec()
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = start[:3, :3] @ Rotation.from_rotvec(
        fraction * relative_vector
    ).as_matrix()
    result[:3, 3] = (
        (1.0 - fraction) * start[:3, 3] + fraction * finish[:3, 3]
    )
    return result


def transform_xyz(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    value = np.asarray(points, dtype=np.float64)
    return (value @ transform[:3, :3].T + transform[:3, 3]).astype(np.float32)


def predicted_current_frame(
    source_goal_frame9: np.ndarray, source_relative_frame9: np.ndarray
) -> np.ndarray:
    """Recover the camera/NDF current frame from its stored goal relation."""

    goal = matrix_from_frame9(source_goal_frame9)
    relative = matrix_from_frame9(source_relative_frame9)
    current = np.eye(4, dtype=np.float64)
    current[:3, :3] = relative[:3, :3].T @ goal[:3, :3]
    current[:3, 3] = goal[:3, 3] - relative[:3, 3]
    return current


def relative_frame9(goal: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Match the repository's world-expressed goal-current relation contract."""

    rotation = goal[:3, :3] @ current[:3, :3].T
    return np.concatenate(
        (
            goal[:3, 3] - current[:3, 3],
            rotation[:, 0],
            rotation[:, 1],
        )
    ).astype(np.float32)


def rigid_grasp_labels(
    *,
    current_object: np.ndarray,
    goal_object: np.ndarray,
    current_eef_pose7: np.ndarray,
    current_anchors: np.ndarray,
    active_arm_right: bool,
    grippers: tuple[float, float],
    action_horizon: int,
    flow_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return actions, future EEF poses, object anchor flow, and pose flow."""

    if int(action_horizon) < 1 or int(flow_steps) < 2:
        raise ValueError("action_horizon must be positive and flow_steps >= 2")
    current_eef = np.stack(
        [pose7_to_matrix(pose) for pose in np.asarray(current_eef_pose7)]
    )
    active = 1 if bool(active_arm_right) else 0

    def state_at(fraction: float) -> tuple[np.ndarray, np.ndarray]:
        object_pose = interpolate_pose(current_object, goal_object, fraction)
        object_delta = object_pose @ np.linalg.inv(current_object)
        eef = current_eef.copy()
        eef[active] = object_delta @ current_eef[active]
        return object_pose, eef

    eef_trajectory = [current_eef]
    for step in range(1, int(action_horizon) + 1):
        _object_pose, eef = state_at(step / float(action_horizon))
        eef_trajectory.append(eef)
    actions = []
    future = []
    for step in range(int(action_horizon)):
        before = np.stack([matrix_to_pose7(value) for value in eef_trajectory[step]])
        after = np.stack(
            [matrix_to_pose7(value) for value in eef_trajectory[step + 1]]
        )
        actions.append(
            eef_delta_action14(
                before[0], grippers[0], before[1], grippers[1],
                after[0], grippers[0], after[1], grippers[1],
            )
        )
        future.append(after)

    anchor_flow = []
    pose_flow = []
    for fraction in np.linspace(0.0, 1.0, int(flow_steps)):
        object_pose, _eef = state_at(float(fraction))
        object_delta = object_pose @ np.linalg.inv(current_object)
        anchor_flow.append(transform_xyz(object_delta, current_anchors))
        pose_flow.append(pose9_from_matrix(object_pose))
    return (
        np.asarray(actions, dtype=np.float32),
        np.asarray(future, dtype=np.float32),
        np.stack(anchor_flow, axis=1).astype(np.float32),
        np.asarray(pose_flow, dtype=np.float32),
    )


def _split_lookup(
    train_ids: list[int], validation_ids: list[int], test_ids: list[int]
) -> dict[int, str]:
    groups = {
        "train": tuple(int(v) for v in train_ids),
        "validation": tuple(int(v) for v in validation_ids),
        "test": tuple(int(v) for v in test_ids),
    }
    if any(not values for values in groups.values()):
        raise ValueError("every object split must be nonempty")
    flat = [value for values in groups.values() for value in values]
    if len(flat) != len(set(flat)):
        raise ValueError("object splits must be disjoint and duplicate-free")
    return {value: name for name, values in groups.items() for value in values}


def select_goal_rows(
    payload: dict[str, np.ndarray],
    source_index: int,
    candidate_rows: np.ndarray,
    count: int,
    maximum_translation_cm: float | None = None,
    maximum_rotation_deg: float | None = None,
) -> list[int]:
    """Select the original goal plus maximally separated real camera goals."""

    if int(count) < 1:
        raise ValueError("goals_per_source must be positive")
    source_episode = int(payload["episode_index"][source_index])
    original = next(
        int(row)
        for row in candidate_rows
        if int(payload["episode_index"][row]) == source_episode
    )
    current = matrix_from_pose9(payload["current_object_pose9"][source_index])
    scored = []
    for row in candidate_rows:
        row = int(row)
        goal = matrix_from_frame9(payload["goal_frame9"][row])
        translation_m = float(np.linalg.norm(goal[:3, 3] - current[:3, 3]))
        rotation_rad = float(Rotation.from_matrix(
            goal[:3, :3] @ current[:3, :3].T
        ).magnitude())
        if maximum_translation_cm is not None and (
            translation_m * 100.0 > float(maximum_translation_cm)
        ):
            continue
        if maximum_rotation_deg is not None and (
            np.rad2deg(rotation_rad) > float(maximum_rotation_deg)
        ):
            continue
        if row == original:
            continue
        score = translation_m / 0.03 + rotation_rad / np.deg2rad(30.0)
        scored.append((float(score), row))
    scored.sort(key=lambda value: (-value[0], value[1]))
    selected = [original] + [row for _score, row in scored[: int(count) - 1]]
    if len(selected) != int(count):
        raise ValueError(
            f"requested {count} goals but split provides only {len(selected)}"
        )
    return selected


def select_source_rows(
    payload: dict[str, np.ndarray],
    *,
    relation_only: bool,
    max_sources_per_episode: int,
    maximum_translation_cm: float | None = None,
    maximum_rotation_deg: float | None = None,
) -> np.ndarray:
    """Select balanced source observations without crossing episode bounds."""

    maximum = int(max_sources_per_episode)
    if maximum < 0:
        raise ValueError("max_sources_per_episode must be nonnegative")
    rows = np.arange(len(payload["shoe_id"]), dtype=np.int64)
    if bool(relation_only):
        if "relation_phase" not in payload:
            raise KeyError("relation-only selection requires relation_phase")
        rows = rows[np.asarray(payload["relation_phase"])[rows] >= 0.5]
    if maximum_translation_cm is not None:
        if float(maximum_translation_cm) <= 0.0:
            raise ValueError("maximum relation translation must be positive")
        if "goal_translation_error_xyz_m" not in payload:
            raise KeyError("translation-limited selection requires goal labels")
        translation_cm = np.linalg.norm(
            np.asarray(payload["goal_translation_error_xyz_m"])[rows], axis=-1
        ) * 100.0
        rows = rows[translation_cm <= float(maximum_translation_cm)]
    if maximum_rotation_deg is not None:
        if float(maximum_rotation_deg) <= 0.0:
            raise ValueError("maximum relation rotation must be positive")
        if "goal_rotation_error_rotvec" not in payload:
            raise KeyError("rotation-limited selection requires goal labels")
        rotation_deg = np.linalg.norm(
            np.asarray(payload["goal_rotation_error_rotvec"])[rows], axis=-1
        ) * (180.0 / np.pi)
        rows = rows[rotation_deg <= float(maximum_rotation_deg)]
    if not len(rows):
        raise ValueError("source selection produced no eligible rows")
    if maximum == 0:
        return rows
    selected = []
    episode_index = np.asarray(payload["episode_index"], dtype=np.int64)
    for episode in np.unique(episode_index[rows]):
        episode_rows = rows[episode_index[rows] == int(episode)]
        if len(episode_rows) > maximum:
            positions = np.linspace(
                0, len(episode_rows) - 1, maximum, dtype=np.int64
            )
            episode_rows = episode_rows[positions]
        selected.append(episode_rows)
    return np.concatenate(selected).astype(np.int64, copy=False)


def build_multigoal_payload(
    payload: dict[str, np.ndarray],
    *,
    train_ids: list[int],
    validation_ids: list[int],
    test_ids: list[int],
    goals_per_source: int,
    flow_steps: int,
    source_indices: np.ndarray | None = None,
    maximum_relation_translation_cm: float | None = None,
    maximum_relation_rotation_deg: float | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    required = {
        "points_a", "points_b", "state", "action", "current_anchors",
        "current_object_pose9", "current_eef_pose7", "future_eef_pose7",
        "active_arm_right", "episode_index", "shoe_id", "frame_index",
        "relation_phase", "goal_frame9", "target_frame9",
        "source_frame6_columns", "goal_translation_error_xyz_m",
        "goal_rotation_error_rotvec",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise KeyError(f"dataset is missing required fields: {missing}")
    split_for_id = _split_lookup(train_ids, validation_ids, test_ids)
    unknown = sorted(set(np.asarray(payload["shoe_id"], dtype=int)) - set(split_for_id))
    if unknown:
        raise ValueError(f"dataset contains object IDs outside split: {unknown}")
    if source_indices is None:
        selected_sources = np.arange(len(payload["shoe_id"]), dtype=np.int64)
    else:
        selected_sources = np.asarray(source_indices, dtype=np.int64).reshape(-1)
        if not len(selected_sources):
            raise ValueError("source_indices must be nonempty")
        if np.any(selected_sources < 0) or np.any(
            selected_sources >= len(payload["shoe_id"])
        ):
            raise IndexError("source_indices contains an out-of-range row")
        if len(np.unique(selected_sources)) != len(selected_sources):
            raise ValueError("source_indices must be duplicate-free")

    episode_rows = {}
    for episode in np.unique(payload["episode_index"]):
        rows = np.flatnonzero(payload["episode_index"] == int(episode))
        episode_rows[int(episode)] = int(rows[0])
    split_goal_rows = {}
    for split_name in ("train", "validation", "test"):
        rows = [
            row for row in episode_rows.values()
            if split_for_id[int(payload["shoe_id"][row])] == split_name
        ]
        split_goal_rows[split_name] = np.asarray(sorted(rows), dtype=np.int64)

    sample_keys = [
        key for key, value in payload.items()
        if np.asarray(value).ndim >= 1 and len(value) == len(payload["shoe_id"])
        and key not in {"episode_flow", "episode_pose", "episode_shoe_id"}
    ]
    values: dict[str, list[np.ndarray | float | int]] = {
        key: [] for key in sample_keys
    }
    episode_flow = []
    episode_pose = []
    episode_shoe_id = []
    donor_episode = []
    source_sample = []
    action_horizon = int(payload["action"].shape[1])

    for source_value in selected_sources:
        source = int(source_value)
        split_name = split_for_id[int(payload["shoe_id"][source])]
        donors = select_goal_rows(
            payload,
            source,
            split_goal_rows[split_name],
            int(goals_per_source),
            maximum_translation_cm=maximum_relation_translation_cm,
            maximum_rotation_deg=maximum_relation_rotation_deg,
        )
        true_current = matrix_from_pose9(payload["current_object_pose9"][source])
        predicted_current = predicted_current_frame(
            payload["goal_frame9"][source], payload["target_frame9"][source]
        )
        for donor in donors:
            goal = matrix_from_frame9(payload["goal_frame9"][donor])
            actions, future, flow, poses = rigid_grasp_labels(
                current_object=true_current,
                goal_object=goal,
                current_eef_pose7=payload["current_eef_pose7"][source],
                current_anchors=payload["current_anchors"][source],
                active_arm_right=bool(payload["active_arm_right"][source] >= 0.5),
                grippers=(
                    float(payload["state"][source, 9]),
                    float(payload["state"][source, 19]),
                ),
                action_horizon=action_horizon,
                flow_steps=int(flow_steps),
            )
            output_index = len(episode_flow)
            for key in sample_keys:
                donor_keys = {"points_b", "goal_frame9", "marker_frame9_camera"}
                value = payload[key][donor if key in donor_keys else source]
                values[key].append(np.asarray(value).copy())
            values["action"][-1] = actions
            values["future_eef_pose7"][-1] = future
            values["episode_index"][-1] = np.asarray(output_index, dtype=np.int64)
            if "episode_id" in values:
                values["episode_id"][-1] = np.asarray(output_index, dtype=np.int64)
            values["frame_index"][-1] = np.asarray(0, dtype=np.int64)
            values["relation_phase"][-1] = np.asarray(1.0, dtype=np.float32)
            if "goal_aligned" in values:
                values["goal_aligned"][-1] = np.asarray(0.0, dtype=np.float32)
            if "steps_to_release" in values:
                values["steps_to_release"][-1] = np.asarray(
                    float(action_horizon), dtype=np.float32
                )
            if "stop_phase" in values:
                values["stop_phase"][-1] = np.asarray(0.0, dtype=np.float32)

            true_relation = relative_frame9(goal, true_current)
            predicted_relation = relative_frame9(goal, predicted_current)
            values["goal_translation_error_xyz_m"][-1] = true_relation[:3]
            values["goal_rotation_error_rotvec"][-1] = Rotation.from_matrix(
                frame9_rotation(true_relation)
            ).as_rotvec().astype(np.float32)
            values["target_frame9"][-1] = predicted_relation
            if "target_frame_confidence" in values:
                source_confidence = float(
                    payload.get("source_frame_confidence", np.ones(len(payload["shoe_id"])))[source]
                )
                values["target_frame_confidence"][-1] = np.asarray(
                    source_confidence, dtype=np.float32
                )
            if "camera_relative_frame_error_rotation_deg_label_only" in values:
                error = rotation_error_deg(
                    frame9_rotation(predicted_relation)[None],
                    frame9_rotation(true_relation)[None],
                )[0]
                values["camera_relative_frame_error_rotation_deg_label_only"][-1] = np.asarray(
                    error, dtype=np.float32
                )
            episode_flow.append(flow)
            episode_pose.append(poses)
            episode_shoe_id.append(int(payload["shoe_id"][source]))
            donor_episode.append(int(payload["episode_index"][donor]))
            source_sample.append(int(source))

    output = {
        key: np.asarray(items, dtype=np.asarray(payload[key]).dtype)
        for key, items in values.items()
    }
    output["episode_flow"] = np.asarray(episode_flow, dtype=np.float32)
    output["episode_pose"] = np.asarray(episode_pose, dtype=np.float32)
    output["episode_shoe_id"] = np.asarray(episode_shoe_id, dtype=np.int64)
    output["multigoal_source_sample"] = np.asarray(source_sample, dtype=np.int64)
    output["multigoal_donor_episode"] = np.asarray(donor_episode, dtype=np.int64)
    metadata = {
        "schema_version": 1,
        "method": "real-camera-target pairing with label-only rigid-grasp expert",
        "samples": int(len(episode_flow)),
        "source_samples": int(len(selected_sources)),
        "source_dataset_samples": int(len(payload["shoe_id"])),
        "goals_per_source": int(goals_per_source),
        "flow_steps": int(flow_steps),
        "maximum_relation_translation_cm": (
            None
            if maximum_relation_translation_cm is None
            else float(maximum_relation_translation_cm)
        ),
        "maximum_relation_rotation_deg": (
            None
            if maximum_relation_rotation_deg is None
            else float(maximum_relation_rotation_deg)
        ),
        "object_split": {
            "train": [int(v) for v in train_ids],
            "validation": [int(v) for v in validation_ids],
            "test": [int(v) for v in test_ids],
        },
        "simulator_pose_usage": "training labels and diagnostics only",
        "policy_inputs": "camera A/B clouds, NDF/camera relation, proprioception",
    }
    return output, metadata


def main() -> None:
    args = parser().parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    with np.load(args.dataset, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    source_indices = select_source_rows(
        payload,
        relation_only=bool(args.relation_only),
        max_sources_per_episode=int(args.max_sources_per_episode),
        maximum_translation_cm=args.maximum_relation_translation_cm,
        maximum_rotation_deg=args.maximum_relation_rotation_deg,
    )
    output, metadata = build_multigoal_payload(
        payload,
        train_ids=args.train_ids,
        validation_ids=args.validation_ids,
        test_ids=args.test_ids,
        goals_per_source=args.goals_per_source,
        flow_steps=args.flow_steps,
        source_indices=source_indices,
        maximum_relation_translation_cm=args.maximum_relation_translation_cm,
        maximum_relation_rotation_deg=args.maximum_relation_rotation_deg,
    )
    metadata["source_selection"] = {
        "relation_only": bool(args.relation_only),
        "max_sources_per_episode": int(args.max_sources_per_episode),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)
    source_metadata = args.dataset.with_suffix(".json")
    if source_metadata.is_file():
        metadata["source_metadata"] = json.loads(
            source_metadata.read_text(encoding="utf-8")
        )
    metadata["source_dataset"] = str(args.dataset.resolve())
    metadata["output"] = str(args.output.resolve())
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
