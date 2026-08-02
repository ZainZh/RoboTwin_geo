"""RoboTwin deployment adapter for the promoted camera-flow action route."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[2]
DP3_SCRIPTS = REPO_ROOT / "policy" / "DP3" / "scripts"
if str(DP3_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(DP3_SCRIPTS))

from eef_delta_action_utils import (  # noqa: E402
    decode_eef_delta_action16,
    eef_state20,
)

from .build_dp3_flow_dataset import camera_reanchored_anchors
from .build_grasp_dataset import valid_xyz
from .eef_frame_action import action_eef_to_world
from .functional_frame_flow import (
    FunctionalPcaFlowEstimator,
    endpoint_correct_flow as oracle_endpoint_correct_flow,
    rotation_endpoint_correct_flow as oracle_rotation_endpoint_correct_flow,
)
from .online_flow import OnlinePcaFlowTransporter
from .ndf_adapter import NdfPointwiseAdapter
from .runtime_action_policy import load_flow_action_runtime
from .train_task_flow_benchmark import geometric_flow_progress


def _int_tuple(value) -> tuple[int, ...]:
    if isinstance(value, str):
        value = [part.strip() for part in value.split(",") if part.strip()]
    return tuple(int(item) for item in value)


def _bool(value) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
        raise ValueError(f"invalid boolean value {value!r}")
    return bool(value)


def _pose_matrix(pose) -> np.ndarray:
    """Convert a SAPIEN pose to a world-frame homogeneous transform."""
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_quat(
        [pose.q[1], pose.q[2], pose.q[3], pose.q[0]]
    ).as_matrix()
    result[:3, 3] = np.asarray(pose.p, dtype=np.float64)
    return result


def temporal_ensemble_action(
    chunks: list[tuple[int, np.ndarray]],
    action_index: int,
    decay: float,
) -> tuple[np.ndarray, int]:
    """Average overlapping chunk predictions for one absolute action index.

    More recent predictions receive greater weight.  This is deliberately a
    deployment-only operation: it leaves the learned geometry representation
    and policy checkpoint unchanged, making it a clean sequential-stability
    ablation.
    """
    candidates = []
    weights = []
    for start, chunk in chunks:
        offset = int(action_index) - int(start)
        if 0 <= offset < len(chunk):
            candidates.append(np.asarray(chunk[offset], dtype=np.float64))
            weights.append(np.exp(-float(decay) * float(offset)))
    if not candidates:
        raise RuntimeError(f"no action chunk covers action index {action_index}")
    value = np.average(np.stack(candidates), axis=0, weights=np.asarray(weights))
    return value.astype(np.float32), len(candidates)


def route_action_to_closed_gripper(
    action: np.ndarray,
    endpose: dict,
    latched_arm: str | None = None,
) -> tuple[np.ndarray, str | None]:
    """Keep motion on the grasping arm and latch it once it is observable.

    Placement rollouts can contain erroneous gripper predictions.  Re-deciding
    the active arm at every step lets such an error silently switch control to
    the other arm, so a previously observed active arm takes precedence.
    """
    value = np.asarray(action, dtype=np.float32).copy()
    left = float(endpose["left_gripper"])
    right = float(endpose["right_gripper"])
    active_arm = latched_arm
    if active_arm is None:
        left_closed = left < 0.5
        right_closed = right < 0.5
        if left_closed == right_closed:
            return value, None
        active_arm = "left" if left_closed else "right"
    if active_arm == "left":
        value[7:13] = 0.0
        value[13] = right
        return value, "left"
    if active_arm == "right":
        value[:6] = 0.0
        value[6] = left
        return value, "right"
    raise ValueError(f"invalid latched active arm {active_arm!r}")


def hold_active_gripper_closed(action: np.ndarray, active_arm: str | None) -> np.ndarray:
    """Keep a known grasp closed during placement-only recovery collection."""
    value = np.asarray(action, dtype=np.float32).copy()
    if active_arm is None:
        return value
    if active_arm == "left":
        value[6] = 0.0
    elif active_arm == "right":
        value[13] = 0.0
    else:
        raise ValueError(f"invalid active arm {active_arm!r}")
    return value


def gate_active_gripper_by_flow_progress(
    action: np.ndarray,
    active_arm: str | None,
    progress: float,
    release_threshold: float,
) -> tuple[np.ndarray, str | None]:
    """Use camera-derived task progress as a high-precision release gate.

    A negative threshold disables the gate.  With the gate enabled, the grasp
    remains closed until the observed object anchors reach the late part of the
    transported task flow, then release is forced.  This prevents a learned
    gripper logit from switching the manipulation phase prematurely while
    keeping the phase decision independent of simulator task state.
    """
    value = np.asarray(action, dtype=np.float32).copy()
    if float(release_threshold) < 0.0 or active_arm is None:
        return value, None
    gripper_index = 6 if active_arm == "left" else 13 if active_arm == "right" else None
    if gripper_index is None:
        raise ValueError(f"invalid active arm {active_arm!r}")
    released = float(progress) >= float(release_threshold)
    value[gripper_index] = 1.0 if released else 0.0
    return value, "released" if released else "closed"


def get_model(usr_args):
    checkpoint = Path(str(usr_args.get("geometry_flow_checkpoint", ""))).expanduser()
    library = Path(str(usr_args.get("geometry_flow_library", ""))).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"geometry-flow action checkpoint not found: {checkpoint}")
    if not library.is_file():
        raise FileNotFoundError(f"geometry-flow library not found: {library}")
    device = str(usr_args.get("geometry_flow_device", "cuda:0"))
    representation = str(usr_args.get("geometry_flow_representation", "pca"))
    if representation not in {
        "pca",
        "ndf",
        "oracle_endpoint",
        "oracle_rotation_endpoint",
        "grasp_relation_geometry",
    }:
        raise ValueError(f"unsupported geometry-flow representation {representation!r}")
    action_decoder = str(
        usr_args.get("geometry_flow_action_decoder", "learned_delta")
    )
    if action_decoder not in {"learned_delta", "relation_planner"}:
        raise ValueError(f"unsupported geometry-flow action decoder {action_decoder!r}")
    if action_decoder == "relation_planner" and representation != "grasp_relation_geometry":
        raise ValueError(
            "relation_planner requires geometry_flow_representation="
            "grasp_relation_geometry"
        )
    descriptor_encoder = None
    if representation == "ndf":
        descriptor_encoder = NdfPointwiseAdapter(
            usr_args.get("geometry_flow_ndf_checkpoint", "/shared2/sz/model/ndf/shoe.pth"),
            device=device,
        )
    model = SimpleNamespace(
        action_policy=load_flow_action_runtime(checkpoint, device=device),
        flow_transporter=OnlinePcaFlowTransporter(
            library,
            train_shoe_ids=_int_tuple(
                usr_args.get("geometry_flow_train_shoes", "2,3,4,7,8,9")
            ),
            representation=(
                representation if representation in {"pca", "ndf"} else "pca"
            ),
            descriptor_encoder=descriptor_encoder,
        ),
        functional_flow_estimator=(
            FunctionalPcaFlowEstimator(
                library,
                Path(
                    str(
                        usr_args.get(
                            "geometry_flow_oracle_task_dataset",
                            "outputs/geometry_flow/task_flow_oracle_50.npz",
                        )
                    )
                ).expanduser(),
                train_shoe_ids=_int_tuple(
                    usr_args.get("geometry_flow_train_shoes", "2,3,4,7,8,9")
                ),
            )
            if representation == "grasp_relation_geometry"
            else None
        ),
        flow_representation=representation,
        action_decoder=action_decoder,
        max_eef_translation_delta_m=float(
            usr_args.get("max_eef_translation_delta_m", 0.08)
        ),
        max_eef_rotation_delta_rad=float(
            usr_args.get("max_eef_rotation_delta_rad", 0.5)
        ),
        execute_steps=max(1, int(usr_args.get("geometry_flow_execute_steps", 1))),
        max_policy_steps=max(
            0, int(usr_args.get("geometry_flow_max_policy_steps", 0))
        ),
        temporal_ensemble_decay=max(
            0.0, float(usr_args.get("geometry_flow_temporal_ensemble_decay", 0.0))
        ),
        route_to_closed_gripper=_bool(
            usr_args.get("geometry_flow_route_to_closed_gripper", False)
        ),
        hold_active_gripper_closed=_bool(
            usr_args.get("geometry_flow_hold_active_gripper_closed", False)
        ),
        gripper_release_progress=float(
            usr_args.get("geometry_flow_gripper_release_progress", -1.0)
        ),
        grasp_fit_threshold_m2=float(
            usr_args.get("geometry_flow_grasp_fit_threshold_m2", 2.5e-4)
        ),
        gripper_symmetry_hypotheses=_bool(
            usr_args.get("geometry_flow_gripper_symmetry_hypotheses", False)
        ),
        gripper_symmetry_min_cost_improvement_m2=max(
            0.0,
            float(
                usr_args.get(
                    "geometry_flow_gripper_symmetry_min_cost_improvement_m2",
                    0.0,
                )
            ),
        ),
        relation_pre_clearance_m=float(
            usr_args.get("geometry_flow_relation_pre_clearance_m", 0.10)
        ),
        relation_final_clearance_m=float(
            usr_args.get("geometry_flow_relation_final_clearance_m", 0.012)
        ),
        relation_settle_physics_steps=max(
            1, int(usr_args.get("geometry_flow_relation_settle_physics_steps", 50))
        ),
        relation_closed_dwell_physics_steps=max(
            0,
            int(
                usr_args.get(
                    "geometry_flow_relation_closed_dwell_physics_steps", 0
                )
            ),
        ),
        relation_release_steps=max(
            1, int(usr_args.get("geometry_flow_relation_release_steps", 1))
        ),
        relation_post_release_retract_m=max(
            0.0,
            float(
                usr_args.get(
                    "geometry_flow_relation_post_release_retract_m", 0.0
                )
            ),
        ),
        relation_plan_stage=0,
        relation_target_frame=None,
        relation_source_episode=None,
        relation_symmetry_variant=0,
        pending_action_chunks=[],
        initial_operated_xyz=None,
        predicted_flow=None,
        initial_anchors=None,
        flow_diagnostics={},
        action_chunks=0,
        temporal_ensemble_candidates=0,
        routed_action_count=0,
        routed_arm=None,
        executed_actions=0,
        predicted_translation_norm_sum=0.0,
        predicted_translation_norm_max=0.0,
        predicted_rotation_norm_sum=0.0,
        predicted_rotation_norm_max=0.0,
        flow_progress=0.0,
        max_flow_progress=0.0,
        progress_gate_closed_actions=0,
        progress_gate_released_actions=0,
        learned_phase=None,
        learned_stop_probabilities=None,
        learned_steps_to_release=None,
        learned_phase_counts={"continue": 0, "settle": 0, "release": 0},
    )
    model.get_evaluation_metrics = lambda: {
        "geometry_flow_source_episode": model.flow_diagnostics.get("source_episode"),
        "geometry_flow_representation": model.flow_representation,
        "geometry_flow_action_decoder": model.action_decoder,
        "geometry_flow_source_shoe": model.flow_diagnostics.get("source_shoe"),
        "geometry_flow_representation_cost": model.flow_diagnostics.get(
            "representation_cost"
        ),
        "geometry_flow_selection_margin": model.flow_diagnostics.get(
            "selection_margin"
        ),
        "geometry_flow_grasp_relation_accepted": model.flow_diagnostics.get(
            "grasp_relation_accepted"
        ),
        "geometry_flow_grasp_fit_threshold_m2": float(
            model.grasp_fit_threshold_m2
        ),
        "geometry_flow_gripper_symmetry_hypotheses": bool(
            model.gripper_symmetry_hypotheses
        ),
        "geometry_flow_gripper_symmetry_min_cost_improvement_m2": float(
            model.gripper_symmetry_min_cost_improvement_m2
        ),
        "geometry_flow_gripper_symmetry_variant": int(
            model.relation_symmetry_variant
        ),
        "geometry_flow_relation_plan_stage": int(model.relation_plan_stage),
        "geometry_flow_relation_pre_clearance_m": float(
            model.relation_pre_clearance_m
        ),
        "geometry_flow_relation_final_clearance_m": float(
            model.relation_final_clearance_m
        ),
        "geometry_flow_relation_settle_physics_steps": int(
            model.relation_settle_physics_steps
        ),
        "geometry_flow_relation_closed_dwell_physics_steps": int(
            model.relation_closed_dwell_physics_steps
        ),
        "geometry_flow_relation_release_steps": int(model.relation_release_steps),
        "geometry_flow_relation_post_release_retract_m": float(
            model.relation_post_release_retract_m
        ),
        "geometry_flow_action_chunks": int(model.action_chunks),
        "geometry_flow_action_frame": model.action_policy.action_frame,
        "geometry_flow_executed_actions": int(model.executed_actions),
        "geometry_flow_execute_steps": int(model.execute_steps),
        "geometry_flow_max_policy_steps": int(model.max_policy_steps),
        "geometry_flow_temporal_ensemble_decay": float(model.temporal_ensemble_decay),
        "geometry_flow_mean_temporal_ensemble_candidates": (
            float(model.temporal_ensemble_candidates / model.executed_actions)
            if model.executed_actions
            else None
        ),
        "geometry_flow_route_to_closed_gripper": bool(model.route_to_closed_gripper),
        "geometry_flow_hold_active_gripper_closed": bool(
            model.hold_active_gripper_closed
        ),
        "geometry_flow_gripper_release_progress": float(
            model.gripper_release_progress
        ),
        "geometry_flow_final_progress": float(model.flow_progress),
        "geometry_flow_max_progress": float(model.max_flow_progress),
        "geometry_flow_progress_gate_closed_actions": int(
            model.progress_gate_closed_actions
        ),
        "geometry_flow_progress_gate_released_actions": int(
            model.progress_gate_released_actions
        ),
        "geometry_flow_learned_phase": model.learned_phase,
        "geometry_flow_learned_stop_probabilities": model.learned_stop_probabilities,
        "geometry_flow_learned_steps_to_release": model.learned_steps_to_release,
        "geometry_flow_learned_phase_counts": dict(model.learned_phase_counts),
        "geometry_flow_routed_action_count": int(model.routed_action_count),
        "geometry_flow_routed_arm": model.routed_arm,
        "geometry_flow_mean_predicted_translation_delta_m": (
            float(model.predicted_translation_norm_sum / model.executed_actions)
            if model.executed_actions
            else None
        ),
        "geometry_flow_max_predicted_translation_delta_m": float(
            model.predicted_translation_norm_max
        ),
        "geometry_flow_mean_predicted_rotation_delta_deg": (
            float(
                np.rad2deg(model.predicted_rotation_norm_sum / model.executed_actions)
            )
            if model.executed_actions
            else None
        ),
        "geometry_flow_max_predicted_rotation_delta_deg": float(
            np.rad2deg(model.predicted_rotation_norm_max)
        ),
    }
    return model


def _object_clouds(observation: dict) -> tuple[np.ndarray, np.ndarray]:
    clouds = observation.get("object_pointcloud", {})
    missing = [key for key in ("{A}", "{B}") if key not in clouds]
    if missing:
        raise RuntimeError(f"geometry-flow observation is missing {missing}")
    return np.asarray(clouds["{A}"], dtype=np.float32), np.asarray(
        clouds["{B}"], dtype=np.float32
    )


def _endpose_state(observation: dict) -> np.ndarray:
    endpose = observation.get("endpose", {})
    required = ("left_endpose", "left_gripper", "right_endpose", "right_gripper")
    missing = [key for key in required if key not in endpose]
    if missing:
        raise RuntimeError(f"geometry-flow observation is missing endpose {missing}")
    return eef_state20(
        endpose["left_endpose"],
        endpose["left_gripper"],
        endpose["right_endpose"],
        endpose["right_gripper"],
    )


def _proprio_observation(TASK_ENV) -> dict:
    """Refresh EEF state between actions without rendering unused cameras."""
    return {
        "endpose": {
            "left_endpose": TASK_ENV.get_arm_pose("left"),
            "left_gripper": TASK_ENV.robot.get_left_gripper_val(),
            "right_endpose": TASK_ENV.get_arm_pose("right"),
            "right_gripper": TASK_ENV.robot.get_right_gripper_val(),
        }
    }


def _execute_relation_plan(TASK_ENV, model, observation: dict) -> None:
    """Execute a bounded EEF contact plan induced by the selected grasp relation.

    This is an action-decoder diagnostic, not an oracle controller: its only
    task target is the camera-derived frame stored at flow construction time.
    The default stages are approach, contact-clearance placement, and release.
    Optional closed-gripper dwell, gradual release, and post-release retraction
    expose contact timing as a controlled ablation without requesting another
    camera frame or simulator object pose.
    """
    if model.relation_target_frame is None or model.relation_source_episode is None:
        raise RuntimeError("relation plan requested before geometry initialization")
    endpose = observation["endpose"]
    if model.routed_arm not in {"left", "right"}:
        raise RuntimeError("relation plan has no active grasping arm")
    stage = int(model.relation_plan_stage)
    dwell_physics_steps = max(
        0, int(getattr(model, "relation_closed_dwell_physics_steps", 0))
    )
    release_steps = max(1, int(getattr(model, "relation_release_steps", 1)))
    retract_m = max(
        0.0, float(getattr(model, "relation_post_release_retract_m", 0.0))
    )
    dwell_stage = 2 if dwell_physics_steps > 0 else None
    release_start = 2 + int(dwell_stage is not None)
    release_stop = release_start + release_steps
    retract_stage = release_stop if retract_m > 0.0 else None
    settle_start = release_stop + int(retract_stage is not None)
    if stage >= settle_start:
        # Replanning the same EEF target after release injects small contact
        # disturbances.  Let physics settle, matching the expert's admission
        # check, so stability is not confounded with target-pose accuracy.
        model.executed_actions += 1
        model.routed_action_count += 1
        model.progress_gate_released_actions += 1
        TASK_ENV.take_action_cnt += 1
        for _ in range(int(model.relation_settle_physics_steps)):
            TASK_ENV.scene.step()
            TASK_ENV._update_render()
            if TASK_ENV.check_success():
                TASK_ENV.eval_success = True
                TASK_ENV.get_obs()
                break
        return
    if dwell_stage is not None and stage == dwell_stage:
        # Preserve the commanded pose and closed gripper while the placement
        # velocity dissipates.  This is proprioceptive/open-loop timing, not a
        # privileged contact query.  check_success is called only to preserve
        # the task's normal per-physics-step metric accounting; it cannot pass
        # while the active gripper remains closed.
        model.executed_actions += 1
        model.routed_action_count += 1
        model.progress_gate_closed_actions += 1
        TASK_ENV.take_action_cnt += 1
        for _ in range(dwell_physics_steps):
            TASK_ENV.scene.step()
            TASK_ENV._update_render()
            TASK_ENV.check_success()
        model.relation_plan_stage += 1
        return
    clearance = (
        model.relation_pre_clearance_m
        if stage == 0
        else (
            retract_m
            if retract_stage is not None and stage == retract_stage
            else model.relation_final_clearance_m
        )
    )
    active_pose = model.functional_flow_estimator.desired_eef_pose7(
        model.relation_target_frame,
        source_episode=int(model.relation_source_episode),
        clearance_m=float(clearance),
        gripper_symmetry_variant=int(model.relation_symmetry_variant),
    )
    left_pose = np.asarray(endpose["left_endpose"], dtype=np.float32).copy()
    right_pose = np.asarray(endpose["right_endpose"], dtype=np.float32).copy()
    left_gripper = float(endpose["left_gripper"])
    right_gripper = float(endpose["right_gripper"])
    if release_start <= stage < release_stop:
        release_fraction = float(stage - release_start + 1) / float(release_steps)
    elif retract_stage is not None and stage == retract_stage:
        release_fraction = 1.0
    else:
        release_fraction = 0.0
    released = release_fraction >= 1.0
    if model.routed_arm == "left":
        left_pose = active_pose
        left_gripper = release_fraction
    else:
        right_pose = active_pose
        right_gripper = release_fraction
    action = np.concatenate(
        (left_pose, [left_gripper], right_pose, [right_gripper])
    ).astype(np.float32)
    model.executed_actions += 1
    model.routed_action_count += 1
    if released:
        model.progress_gate_released_actions += 1
    else:
        model.progress_gate_closed_actions += 1
    TASK_ENV.take_action(action, action_type="ee")
    model.relation_plan_stage = min(stage + 1, settle_start)


def eval(TASK_ENV, model, observation):
    # A grasp relation is selected once from the camera.  After initialization
    # the planner transports it with EEF proprioception, so requesting another
    # segmented cloud is both unnecessary and a source of post-release noise.
    if model.action_decoder == "relation_planner" and model.predicted_flow is not None:
        _execute_relation_plan(TASK_ENV, model, observation)
        if (
            model.max_policy_steps > 0
            and model.executed_actions >= model.max_policy_steps
        ):
            TASK_ENV.step_lim = TASK_ENV.take_action_cnt
        return
    operated, target = _object_clouds(observation)
    operated_xyz = valid_xyz(operated)
    if model.predicted_flow is None:
        result = model.flow_transporter.predict(operated, target)
        model.predicted_flow = result.flow
        source_episode = int(result.source_episode)
        source_shoe = int(result.source_shoe)
        representation_cost = float(result.representation_cost)
        selection_margin = None
        grasp_relation_accepted = None
        if model.flow_representation in {
            "oracle_endpoint",
            "oracle_rotation_endpoint",
        }:
            shoe_pose = _pose_matrix(TASK_ENV.shoe.get_functional_point(0, "pose"))
            target_pose = _pose_matrix(
                TASK_ENV.target_block.get_functional_point(0, "pose")
            )
            correction_function = (
                oracle_endpoint_correct_flow
                if model.flow_representation == "oracle_endpoint"
                else oracle_rotation_endpoint_correct_flow
            )
            model.predicted_flow = correction_function(
                result.initial_anchors,
                model.predicted_flow,
                target_pose @ np.linalg.inv(shoe_pose),
            )
        elif model.flow_representation == "grasp_relation_geometry":
            endpose = observation["endpose"]
            active_right = float(endpose["right_gripper"]) < float(
                endpose["left_gripper"]
            )
            active_pose = (
                endpose["right_endpose"] if active_right else endpose["left_endpose"]
            )
            grasp_symmetry_variant = 0
            if model.gripper_symmetry_hypotheses:
                (
                    grasp_flow,
                    grasp_source_episode,
                    grasp_cost,
                    selection_margin,
                    grasp_symmetry_variant,
                ) = model.functional_flow_estimator.predict_symmetry_selected_grasp_flow(
                    operated,
                    active_pose,
                    result.target_frame,
                    initial_anchors=result.initial_anchors,
                    base_flow=result.flow,
                    min_cost_improvement_m2=(
                        model.gripper_symmetry_min_cost_improvement_m2
                    ),
                )
            else:
                (
                    grasp_flow,
                    grasp_source_episode,
                    grasp_cost,
                    selection_margin,
                ) = model.functional_flow_estimator.predict_geometry_selected_grasp_flow(
                    operated,
                    active_pose,
                    result.target_frame,
                    initial_anchors=result.initial_anchors,
                    base_flow=result.flow,
                )
            grasp_relation_accepted = bool(
                grasp_cost <= model.grasp_fit_threshold_m2
            )
            if grasp_relation_accepted:
                model.predicted_flow = grasp_flow
                source_episode = int(grasp_source_episode)
                source_shoe = -1
                model.relation_symmetry_variant = int(grasp_symmetry_variant)
            else:
                model.relation_symmetry_variant = 0
            representation_cost = float(grasp_cost)
            if model.action_decoder == "relation_planner":
                model.routed_arm = "right" if active_right else "left"
                model.relation_target_frame = np.asarray(
                    result.target_frame, dtype=np.float64
                ).copy()
                model.relation_source_episode = int(source_episode)
        model.initial_anchors = result.initial_anchors
        model.initial_operated_xyz = operated_xyz.copy()
        model.flow_diagnostics = {
            "source_episode": int(source_episode),
            "source_shoe": int(source_shoe),
            "representation_cost": float(representation_cost),
            "selection_margin": (
                float(selection_margin) if selection_margin is not None else None
            ),
            "grasp_relation_accepted": grasp_relation_accepted,
            "gripper_symmetry_variant": int(model.relation_symmetry_variant),
        }
    current_anchors = camera_reanchored_anchors(
        model.initial_operated_xyz,
        operated_xyz,
        model.initial_anchors,
    ).astype(np.float32)
    model.flow_progress = geometric_flow_progress(model.predicted_flow, current_anchors)
    model.max_flow_progress = max(model.max_flow_progress, model.flow_progress)
    if model.action_decoder == "relation_planner":
        _execute_relation_plan(TASK_ENV, model, observation)
        if (
            model.max_policy_steps > 0
            and model.executed_actions >= model.max_policy_steps
        ):
            TASK_ENV.step_lim = TASK_ENV.take_action_cnt
        return
    prediction_args = {
        "operated_point_cloud": operated,
        "target_point_cloud": target,
        "eef_state20": _endpose_state(observation),
        "current_anchors": current_anchors,
        "predicted_episode_flow": model.predicted_flow,
    }
    if model.action_policy.is_recovery_aware:
        prediction_args["active_arm"] = model.routed_arm
    actions = model.action_policy.predict(**prediction_args)
    if model.action_policy.is_recovery_aware:
        model.learned_phase = model.action_policy.last_stop_phase
        model.learned_stop_probabilities = (
            model.action_policy.last_stop_probabilities.tolist()
        )
        model.learned_steps_to_release = model.action_policy.last_steps_to_release
        model.learned_phase_counts[model.learned_phase] += 1
    model.action_chunks += 1
    chunk_start = int(model.executed_actions)
    model.pending_action_chunks.append((chunk_start, actions.copy()))
    for local_step, raw_action in enumerate(actions[: model.execute_steps]):
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
        endpose = observation["endpose"]
        if model.action_policy.action_frame == "eef":
            action = action_eef_to_world(
                action,
                np.stack((endpose["left_endpose"], endpose["right_endpose"])),
            )
        if model.route_to_closed_gripper:
            action, routed_arm = route_action_to_closed_gripper(
                action, endpose, model.routed_arm
            )
            if routed_arm is not None:
                model.routed_action_count += 1
                model.routed_arm = routed_arm
        action, gate_state = gate_active_gripper_by_flow_progress(
            action,
            model.routed_arm,
            model.flow_progress,
            model.gripper_release_progress,
        )
        if gate_state == "closed":
            model.progress_gate_closed_actions += 1
        elif gate_state == "released":
            model.progress_gate_released_actions += 1
        if model.hold_active_gripper_closed:
            action = hold_active_gripper_closed(action, model.routed_arm)
        translation_norm = max(
            float(np.linalg.norm(action[0:3])),
            float(np.linalg.norm(action[7:10])),
        )
        rotation_norm = max(
            float(np.linalg.norm(action[3:6])),
            float(np.linalg.norm(action[10:13])),
        )
        model.executed_actions += 1
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
        TASK_ENV.take_action(decoded, action_type="ee")
        # The remainder of this already-predicted chunk only needs current EEF
        # poses to compose delta actions.  The outer evaluation loop obtains a
        # fresh camera observation before the next chunk/replanning call.
        observation = _proprio_observation(TASK_ENV)
        if (
            model.max_policy_steps > 0
            and model.executed_actions >= model.max_policy_steps
        ):
            # Diagnostic rollouts that have already missed the manipulation
            # window can be capped without changing the task's success test.
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
    model.initial_operated_xyz = None
    model.predicted_flow = None
    model.initial_anchors = None
    model.flow_diagnostics = {}
    model.action_chunks = 0
    model.pending_action_chunks = []
    model.temporal_ensemble_candidates = 0
    model.routed_action_count = 0
    model.routed_arm = None
    model.executed_actions = 0
    model.predicted_translation_norm_sum = 0.0
    model.predicted_translation_norm_max = 0.0
    model.predicted_rotation_norm_sum = 0.0
    model.predicted_rotation_norm_max = 0.0
    model.flow_progress = 0.0
    model.max_flow_progress = 0.0
    model.progress_gate_closed_actions = 0
    model.progress_gate_released_actions = 0
    model.learned_phase = None
    model.learned_stop_probabilities = None
    model.learned_steps_to_release = None
    model.learned_phase_counts = {"continue": 0, "settle": 0, "release": 0}
    model.relation_plan_stage = 0
    model.relation_target_frame = None
    model.relation_source_episode = None
    model.relation_symmetry_variant = 0
