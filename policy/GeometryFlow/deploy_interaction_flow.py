"""RoboTwin deployment adapter for camera-only interaction flow tokens."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from policy.DP3.scripts.eef_delta_action_utils import decode_eef_delta_action16

from .build_grasp_dataset import valid_xyz
from .deploy_policy import (
    _bool,
    _endpose_state,
    hold_active_gripper_closed,
    _object_clouds,
    route_action_to_closed_gripper,
    temporal_ensemble_action,
)
from .interaction_flow_runtime import InteractionFlowRuntime
from .online_hanging_mug_camera_rotation import (
    OnlineHangingMugCameraRotationProvider,
)
from .online_ndf_functional_frame import relative_frame9_metric


def _path_list(value) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    raise TypeError("dense NDF checkpoints must be a list or comma-separated string")


def _pose7_wxyz_matrix(pose7: np.ndarray) -> np.ndarray:
    value = np.asarray(pose7, dtype=np.float64).reshape(7)
    quaternion_wxyz = value[3:]
    quaternion_wxyz /= max(float(np.linalg.norm(quaternion_wxyz)), 1e-12)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_quat(
        quaternion_wxyz[[1, 2, 3, 0]]
    ).as_matrix()
    transform[:3, 3] = value[:3]
    return transform


def _privileged_oracle_relative_frame9(observation: dict) -> np.ndarray:
    """Simulator-only label ceiling; never called by clean/zero deployment.

    This mirrors ``object_goal_error_labels`` from dataset construction:
    desired ``world_from_A`` is
    ``world_from_B @ inv(goal_T_A_from_B_oracle)``.  Using the two functional
    points directly can give the wrong root-position translation label.
    """

    task_state = observation.get("task_state")
    required = (
        "object_pose_A",
        "object_pose_B",
        "goal_T_A_from_B_oracle",
    )
    if not isinstance(task_state, dict):
        raise ValueError("dense oracle ceiling requires observation task_state")
    missing = [key for key in required if key not in task_state]
    if missing:
        raise ValueError(f"dense oracle task_state lacks {missing}")
    current = _pose7_wxyz_matrix(task_state["object_pose_A"])
    world_from_b = _pose7_wxyz_matrix(task_state["object_pose_B"])
    goal_t_a_from_b = np.asarray(
        task_state["goal_T_A_from_B_oracle"], dtype=np.float64
    ).reshape(4, 4)
    goal = world_from_b @ np.linalg.inv(goal_t_a_from_b)
    return relative_frame9_metric(
        current[:3, 3],
        current[:3, :3],
        goal[:3, 3],
        goal[:3, :3],
    )


def get_model(usr_args):
    checkpoint = Path(
        str(usr_args.get("interaction_flow_checkpoint", ""))
    ).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"interaction-flow checkpoint not found: {checkpoint}"
        )
    priority_override = usr_args.get("interaction_flow_target_color_priority")
    dense_checkpoints = _path_list(
        usr_args.get("interaction_flow_dense_frame_checkpoints")
    )
    dense_frame_mode = str(
        usr_args.get("interaction_flow_dense_frame_mode", "off")
    )
    dense_camera_metadata = usr_args.get(
        "interaction_flow_dense_camera_metadata"
    )
    dense_ensemble_metadata = usr_args.get(
        "interaction_flow_dense_ensemble_metadata"
    )
    dense_provider = None
    if dense_frame_mode != "off" and dense_camera_metadata not in {None, ""}:
        calibration_path = Path(str(dense_camera_metadata)).expanduser()
        if calibration_path.suffix == ".npz":
            dense_provider = OnlineHangingMugCameraRotationProvider.from_artifacts(
                frame_checkpoints=dense_checkpoints,
                calibration=calibration_path,
                ensemble_metadata=Path(str(dense_ensemble_metadata)).expanduser(),
                device=str(usr_args.get("interaction_flow_device", "cuda:0")),
            )
    model = SimpleNamespace(
        action_policy=InteractionFlowRuntime(
            checkpoint,
            device=str(usr_args.get("interaction_flow_device", "cuda:0")),
            observation_points=int(
                usr_args.get("interaction_flow_observation_points", 64)
            ),
            target_color_priority=(
                None if priority_override is None else _bool(priority_override)
            ),
            ndf_checkpoint=usr_args.get("interaction_flow_ndf_checkpoint"),
            source_axis_mode=str(
                usr_args.get("interaction_flow_source_axis_mode", "clean")
            ),
            target_axis_mode=str(
                usr_args.get("interaction_flow_target_axis_mode", "clean")
            ),
            dense_frame_checkpoints=dense_checkpoints,
            dense_camera_metadata=dense_camera_metadata,
            dense_ensemble_metadata=dense_ensemble_metadata,
            dense_frame_mode=dense_frame_mode,
            allow_privileged_oracle=_bool(
                usr_args.get(
                    "interaction_flow_allow_privileged_oracle", False
                )
            ),
            dense_frame_provider=dense_provider,
        ),
        checkpoint=str(checkpoint),
        execute_steps=max(
            1, int(usr_args.get("interaction_flow_execute_steps", 1))
        ),
        max_policy_steps=max(
            0, int(usr_args.get("interaction_flow_max_policy_steps", 240))
        ),
        temporal_ensemble_decay=max(
            0.0,
            float(usr_args.get("interaction_flow_temporal_ensemble_decay", 0.0)),
        ),
        route_to_closed_gripper=_bool(
            usr_args.get("interaction_flow_route_to_closed_gripper", True)
        ),
        hold_active_gripper_closed=_bool(
            usr_args.get("interaction_flow_hold_active_gripper_closed", False)
        ),
        max_eef_translation_delta_m=float(
            usr_args.get("max_eef_translation_delta_m", 0.08)
        ),
        max_eef_rotation_delta_rad=float(
            usr_args.get("max_eef_rotation_delta_rad", 0.5)
        ),
        pending_action_chunks=[],
        action_chunks=0,
        executed_actions=0,
        routed_action_count=0,
        routed_arm=None,
        temporal_ensemble_candidates=0,
        predicted_translation_norm_sum=0.0,
        predicted_translation_norm_max=0.0,
        predicted_rotation_norm_sum=0.0,
        predicted_rotation_norm_max=0.0,
        cached_operated_point_cloud=None,
        cached_target_point_cloud=None,
    )

    def metrics():
        count = max(int(model.executed_actions), 1)
        return {
            "interaction_flow_checkpoint": model.checkpoint,
            "interaction_flow_condition": model.action_policy.condition,
            "interaction_flow_policy_frame": model.action_policy.policy_frame,
            "interaction_flow_source_axis_mode": (
                model.action_policy.source_axis_mode
            ),
            "interaction_flow_target_axis_mode": (
                model.action_policy.target_axis_mode
            ),
            "interaction_flow_ndf_checkpoint": (
                None
                if model.action_policy.ndf_checkpoint is None
                else str(model.action_policy.ndf_checkpoint)
            ),
            "interaction_flow_dense_frame_mode": (
                model.action_policy.dense_frame_mode
            ),
            "interaction_flow_dense_frame_checkpoints": dense_checkpoints,
            "interaction_flow_dense_camera_metadata": (
                None
                if usr_args.get("interaction_flow_dense_camera_metadata")
                in {None, ""}
                else str(usr_args.get("interaction_flow_dense_camera_metadata"))
            ),
            "interaction_flow_dense_ensemble_metadata": (
                None
                if usr_args.get("interaction_flow_dense_ensemble_metadata")
                in {None, ""}
                else str(usr_args.get("interaction_flow_dense_ensemble_metadata"))
            ),
            "interaction_flow_privileged_oracle_enabled": bool(
                model.action_policy.allow_privileged_oracle
            ),
            "interaction_flow_action_chunks": int(model.action_chunks),
            "interaction_flow_executed_actions": int(model.executed_actions),
            "interaction_flow_execute_steps": int(model.execute_steps),
            "interaction_flow_routed_action_count": int(model.routed_action_count),
            "interaction_flow_routed_arm": model.routed_arm,
            "interaction_flow_hold_active_gripper_closed": bool(
                model.hold_active_gripper_closed
            ),
            "interaction_flow_mean_temporal_ensemble_candidates": (
                float(model.temporal_ensemble_candidates) / count
            ),
            "interaction_flow_mean_translation_delta_m": (
                float(model.predicted_translation_norm_sum) / count
            ),
            "interaction_flow_max_translation_delta_m": float(
                model.predicted_translation_norm_max
            ),
            "interaction_flow_mean_rotation_delta_deg": float(
                np.rad2deg(model.predicted_rotation_norm_sum / count)
            ),
            "interaction_flow_max_rotation_delta_deg": float(
                np.rad2deg(model.predicted_rotation_norm_max)
            ),
            **{
                f"interaction_flow_{key}": value
                for key, value in model.action_policy.last_diagnostic.items()
            },
        }

    model.get_evaluation_metrics = metrics
    return model


def eval(TASK_ENV, model, observation):
    operated, target = _object_clouds(observation)
    if len(valid_xyz(operated)) > 0:
        model.cached_operated_point_cloud = operated.copy()
    elif getattr(model, "cached_operated_point_cloud", None) is not None:
        operated = model.cached_operated_point_cloud
    if len(valid_xyz(target)) > 0:
        model.cached_target_point_cloud = target.copy()
    elif getattr(model, "cached_target_point_cloud", None) is None:
        raise ValueError("target point cloud is empty before any valid observation")
    target = model.cached_target_point_cloud
    oracle_frame = None
    if model.action_policy.dense_frame_mode == "oracle":
        if not model.action_policy.allow_privileged_oracle:
            raise PermissionError("privileged oracle ceiling is not enabled")
        oracle_frame = _privileged_oracle_relative_frame9(observation)
    actions = model.action_policy.predict(
        operated_point_cloud=operated,
        target_point_cloud=target,
        eef_state20=_endpose_state(observation),
        oracle_relative_frame9_metric=oracle_frame,
    )
    model.action_chunks += 1
    chunk_start = int(model.executed_actions)
    model.pending_action_chunks.append((chunk_start, actions.copy()))
    endpose = {
        "left_endpose": np.asarray(
            observation["endpose"]["left_endpose"], dtype=np.float32
        ).copy(),
        "left_gripper": float(observation["endpose"]["left_gripper"]),
        "right_endpose": np.asarray(
            observation["endpose"]["right_endpose"], dtype=np.float32
        ).copy(),
        "right_gripper": float(observation["endpose"]["right_gripper"]),
    }

    for raw_action in actions[: model.execute_steps]:
        if model.temporal_ensemble_decay > 0.0:
            action, candidate_count = temporal_ensemble_action(
                model.pending_action_chunks,
                model.executed_actions,
                model.temporal_ensemble_decay,
            )
        else:
            action = raw_action
            candidate_count = 1
        model.temporal_ensemble_candidates += int(candidate_count)
        if model.route_to_closed_gripper:
            action, routed_arm = route_action_to_closed_gripper(
                action, endpose, model.routed_arm
            )
            if routed_arm is not None:
                model.routed_arm = routed_arm
                model.routed_action_count += 1
        if model.hold_active_gripper_closed:
            action = hold_active_gripper_closed(action, model.routed_arm)

        # The demonstrations use continuous gripper commands in [0,1].  Keep
        # regression overshoot out of the simulator without discretizing the
        # learned release trajectory.
        action = np.asarray(action, dtype=np.float32).copy()
        action[6] = np.clip(action[6], 0.0, 1.0)
        action[13] = np.clip(action[13], 0.0, 1.0)

        translation_norm = max(
            float(np.linalg.norm(action[0:3])),
            float(np.linalg.norm(action[7:10])),
        )
        rotation_norm = max(
            float(np.linalg.norm(action[3:6])),
            float(np.linalg.norm(action[10:13])),
        )
        model.predicted_translation_norm_sum += translation_norm
        model.predicted_translation_norm_max = max(
            model.predicted_translation_norm_max, translation_norm
        )
        model.predicted_rotation_norm_sum += rotation_norm
        model.predicted_rotation_norm_max = max(
            model.predicted_rotation_norm_max, rotation_norm
        )
        decoded = decode_eef_delta_action16(
            action,
            endpose["left_endpose"],
            endpose["right_endpose"],
            max_translation_delta_m=model.max_eef_translation_delta_m,
            max_rotation_delta_rad=model.max_eef_rotation_delta_rad,
        )
        model.executed_actions += 1
        TASK_ENV.take_action(decoded, action_type="ee")
        # A chunk contains sequential EEF deltas.  Reuse the commanded target
        # as the next proprioceptive reference and acquire a new camera frame
        # only after the chunk.  The environment already checks task success
        # inside take_action(), so this does not delay success detection.
        endpose = {
            "left_endpose": decoded[:7].copy(),
            "left_gripper": float(decoded[7]),
            "right_endpose": decoded[8:15].copy(),
            "right_gripper": float(decoded[15]),
        }
        if (
            model.max_policy_steps > 0
            and model.executed_actions >= model.max_policy_steps
        ):
            TASK_ENV.step_lim = TASK_ENV.take_action_cnt
            break
        if TASK_ENV.eval_success or TASK_ENV.take_action_cnt >= TASK_ENV.step_lim:
            break

    current_index = int(model.executed_actions)
    model.pending_action_chunks = [
        (start, chunk)
        for start, chunk in model.pending_action_chunks
        if current_index - int(start) < len(chunk)
    ]


def reset_model(model):
    model.pending_action_chunks = []
    model.action_chunks = 0
    model.executed_actions = 0
    model.routed_action_count = 0
    model.routed_arm = None
    model.temporal_ensemble_candidates = 0
    model.predicted_translation_norm_sum = 0.0
    model.predicted_translation_norm_max = 0.0
    model.predicted_rotation_norm_sum = 0.0
    model.predicted_rotation_norm_max = 0.0
    model.cached_operated_point_cloud = None
    model.cached_target_point_cloud = None
    model.action_policy.reset()


def latch_dense_target_observations(model, observations) -> dict:
    """Fuse early target observations exactly as camera-frame training does.

    The camera augmentation used the first three B observations as one marker
    fit, whereas the original online runtime latched only the first B frame.
    Keeping this small adapter at deployment makes that train/test contract
    explicit without changing the per-step B tokens consumed by the policy.
    """

    provider = getattr(model.action_policy, "dense_frame_provider", None)
    if provider is None:
        return {
            "available": False,
            "requested_frames": int(len(observations)),
            "valid_frames": 0,
            "fused_points": 0,
        }
    target_clouds = []
    for observation in observations:
        _operated, target = _object_clouds(observation)
        if len(valid_xyz(target)):
            target_clouds.append(np.asarray(target, dtype=np.float32))
    if not target_clouds:
        raise ValueError("target latch received no valid B observations")
    fused = np.concatenate(target_clouds, axis=0)
    provider.latch_target(fused)
    # Preserve the last ordinary observation for the per-point policy tokens;
    # only the marker frame estimator consumes the fused cloud.
    model.cached_target_point_cloud = target_clouds[-1].copy()
    return {
        "available": True,
        "requested_frames": int(len(observations)),
        "valid_frames": int(len(target_clouds)),
        "fused_points": int(len(fused)),
    }
