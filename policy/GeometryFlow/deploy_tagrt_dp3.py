"""Full-task RoboTwin deployment adapter for a TAGRT-conditioned DP3 policy."""

from __future__ import annotations

from collections import deque
from pathlib import Path
import sys
from types import SimpleNamespace

import dill
import numpy as np
from scipy.spatial.transform import Rotation
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
DP3_ROOT = REPO_ROOT / "policy" / "DP3" / "3D-Diffusion-Policy"
DP3_SCRIPTS = REPO_ROOT / "policy" / "DP3" / "scripts"
for path in (DP3_ROOT, DP3_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from diffusion_policy_3d.common.pytorch_util import dict_apply  # noqa: E402
from diffusion_policy_3d.dataset.tagrt_dataset import (  # noqa: E402
    TaskAlignedGeometryDataset,
    task_aligned_anchor_flow_tokens,
    task_aligned_local_coordinates,
)
from train_dp3 import TrainDP3Workspace  # noqa: E402
from eef_delta_action_utils import decode_eef_delta_action16  # noqa: E402

from .build_task_flow_dataset import deterministic_sample
from .deploy_policy import _endpose_state, _object_clouds
from .online_ndf_functional_frame import OnlineNdfFunctionalFrameProvider


def _path_list(value) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item) for item in value]


def _goal_frame9(transform: np.ndarray) -> np.ndarray:
    value = np.asarray(transform, dtype=np.float32).reshape(4, 4)
    return np.concatenate((value[:3, 3], value[:3, 0], value[:3, 1])).astype(
        np.float32
    )


def _frame9_rotation_angle_deg(frame9: np.ndarray) -> float:
    return float(np.rad2deg(np.linalg.norm(_frame9_rotation_rotvec(frame9))))


def _frame9_rotation_rotvec(frame9: np.ndarray) -> np.ndarray:
    """Return the signed world-frame rotation vector encoded by a 9D frame."""

    value = np.asarray(frame9, dtype=np.float64).reshape(9)
    first = value[3:6]
    first /= max(float(np.linalg.norm(first)), 1.0e-8)
    second = value[6:9] - first * float(np.dot(first, value[6:9]))
    second /= max(float(np.linalg.norm(second)), 1.0e-8)
    rotation = np.column_stack((first, second, np.cross(first, second)))
    return Rotation.from_matrix(rotation).as_rotvec().astype(np.float64)


def _selected_task_pose_metrics(task_env) -> dict[str, object] | None:
    """Read privileged simulator metrics for logging after an action only.

    This helper is deliberately called by the evaluator, never by the runtime
    observation path.  It lets failure analysis separate perception error from
    policy-response error without leaking simulator object pose into actions.
    """

    metrics_fn = getattr(task_env, "get_evaluation_metrics", None)
    if not callable(metrics_fn):
        return None
    metrics = metrics_fn()
    if not isinstance(metrics, dict):
        return None
    keys = (
        "translation_error_xyz_m",
        "translation_error_norm_m",
        "rotation_error_deg",
        "gripper_open",
        "shoe_ramp_contact",
    )
    return {key: metrics.get(key) for key in keys if key in metrics}


def _apply_rotation_recovery_cooldown(
    prediction: dict[str, torch.Tensor],
    cooldown_remaining: int,
    cooldown_chunks: int,
) -> tuple[int, np.ndarray | None, bool]:
    """Debounce repeated recovery chunks while preserving nominal gripper output."""

    remaining = max(0, int(cooldown_remaining))
    configured = max(0, int(cooldown_chunks))
    active = prediction.get(
        "recovery_action_head_active",
        prediction.get("rotation_action_head_active"),
    )
    if active is None:
        return max(remaining - 1, 0), None, False
    requested = active[0].detach().cpu().numpy().astype(bool)
    suppressed = bool(np.any(requested)) and remaining > 0
    if suppressed:
        nominal = prediction.get("action_before_rotation_recovery")
        if nominal is None:
            raise KeyError(
                "recovery cooldown requires action_before_rotation_recovery"
            )
        prediction["action"] = nominal
        for key in (
            "rotation_action_head_active",
            "translation_recovery_head_active",
            "recovery_action_head_active",
        ):
            if key in prediction:
                prediction[key] = torch.zeros_like(
                    prediction[key], dtype=torch.bool
                )
    if remaining > 0:
        remaining -= 1
    elif bool(np.any(requested)):
        remaining = configured
    return remaining, requested, suppressed


def _apply_rotation_recovery_persistence(
    prediction: dict[str, torch.Tensor],
    requested_streak: int,
    min_consecutive_chunks: int,
) -> tuple[int, np.ndarray | None, bool]:
    """Require a persistent learned recovery request before changing motion."""

    active = prediction.get(
        "recovery_action_head_active",
        prediction.get("rotation_action_head_active"),
    )
    if active is None:
        return 0, None, False
    requested = active[0].detach().cpu().numpy().astype(bool)
    streak = int(requested_streak) + 1 if bool(np.any(requested)) else 0
    required = max(1, int(min_consecutive_chunks))
    suppressed = bool(np.any(requested)) and streak < required
    if suppressed:
        nominal = prediction.get("action_before_rotation_recovery")
        if nominal is None:
            raise KeyError(
                "recovery persistence requires action_before_rotation_recovery"
            )
        prediction["action"] = nominal
        for key in (
            "rotation_action_head_active",
            "translation_recovery_head_active",
            "recovery_action_head_active",
        ):
            if key in prediction:
                prediction[key] = torch.zeros_like(
                    prediction[key], dtype=torch.bool
                )
    return streak, requested, suppressed


class TagrtDP3Runtime:
    """Load one DP3 checkpoint and reproduce its camera-only TAGRT contract."""

    def __init__(
        self,
        checkpoint: str | Path,
        dataset: str | Path,
        *,
        frame_provider: OnlineNdfFunctionalFrameProvider,
        device: str = "cuda:0",
        condition: str = "correct",
        binary_gripper_threshold: float | None = None,
        rotation_gate_threshold: float | None = None,
        rotation_support_min_deg: float | None = None,
        rotation_support_max_deg: float | None = None,
        translation_support_min_m: float | None = None,
        translation_support_max_m: float | None = None,
        translation_rotation_max_deg: float | None = None,
        rotation_cooldown_chunks: int = 0,
        translation_cooldown_chunks: int | None = None,
        rotation_min_consecutive_chunks: int = 1,
        translation_min_consecutive_chunks: int | None = None,
    ) -> None:
        if condition not in {"correct", "zero"}:
            raise ValueError("online condition must be correct or zero")
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.dataset_path = Path(dataset).expanduser().resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        payload = torch.load(
            self.checkpoint.open("rb"),
            pickle_module=dill,
            map_location="cpu",
        )
        self.cfg = payload["cfg"]
        workspace = TrainDP3Workspace(
            self.cfg, output_dir=str(self.checkpoint.parent)
        )
        workspace.load_payload(payload, exclude_keys=("optimizer",))
        self.policy = (
            workspace.ema_model
            if workspace.ema_model is not None
            else workspace.model
        )
        self.device = torch.device(device)
        self.policy.to(self.device).eval()
        if binary_gripper_threshold is not None:
            threshold = float(binary_gripper_threshold)
            if not 0.0 < threshold < 1.0:
                raise ValueError("binary gripper threshold must be in (0, 1)")
            self.policy.binary_gripper_threshold = threshold
        if rotation_gate_threshold is not None:
            threshold = float(rotation_gate_threshold)
            if not 0.0 <= threshold <= 1.0:
                raise ValueError("rotation gate threshold must be in [0, 1]")
            self.policy.rotation_action_gate_threshold = threshold
        support_min = float(
            getattr(self.policy, "rotation_action_support_min_deg", 0.0)
            if rotation_support_min_deg is None
            else rotation_support_min_deg
        )
        support_max = float(
            getattr(self.policy, "rotation_action_support_max_deg", 180.0)
            if rotation_support_max_deg is None
            else rotation_support_max_deg
        )
        if not 0.0 <= support_min <= support_max <= 180.0:
            raise ValueError(
                "rotation support must satisfy 0 <= min <= max <= 180"
            )
        self.policy.rotation_action_support_min_deg = support_min
        self.policy.rotation_action_support_max_deg = support_max
        translation_support_min = float(
            getattr(self.policy, "translation_recovery_support_min_m", 0.0)
            if translation_support_min_m is None
            else translation_support_min_m
        )
        translation_support_max = float(
            getattr(self.policy, "translation_recovery_support_max_m", 0.0)
            if translation_support_max_m is None
            else translation_support_max_m
        )
        translation_rotation_max = float(
            getattr(self.policy, "translation_recovery_rotation_max_deg", 180.0)
            if translation_rotation_max_deg is None
            else translation_rotation_max_deg
        )
        if not 0.0 <= translation_support_min <= translation_support_max:
            raise ValueError(
                "translation support must satisfy 0 <= min <= max"
            )
        if not 0.0 <= translation_rotation_max <= 180.0:
            raise ValueError(
                "translation recovery rotation maximum must be in [0, 180]"
            )
        self.policy.translation_recovery_support_min_m = translation_support_min
        self.policy.translation_recovery_support_max_m = translation_support_max
        self.policy.translation_recovery_rotation_max_deg = translation_rotation_max
        geometry_dataset = TaskAlignedGeometryDataset(
            str(self.dataset_path),
            horizon=int(self.cfg.horizon),
            n_obs_steps=int(self.cfg.n_obs_steps),
            n_action_steps=int(self.cfg.n_action_steps),
            train_object_ids=list(self.cfg.task.dataset.train_object_ids),
            val_object_ids=list(self.cfg.task.dataset.val_object_ids),
            test_object_ids=list(self.cfg.task.dataset.test_object_ids),
            split="train",
            condition_mode="correct",
            use_color=bool(self.cfg.task.dataset.get("use_color", False)),
            geometry_scale_m=self.cfg.task.dataset.get("geometry_scale_m", None),
        )
        self.geometry_scale_m = float(geometry_dataset.geometry_scale_m)
        self.policy.translation_recovery_geometry_scale_m = self.geometry_scale_m
        self.observation_points = int(
            self.cfg.task.shape_meta.obs.point_cloud.shape[0]
        )
        self.n_obs_steps = int(self.cfg.n_obs_steps)
        self.frame_provider = frame_provider
        self.condition = str(condition)
        self.history: deque[dict[str, np.ndarray]] = deque(
            maxlen=self.n_obs_steps
        )
        self.observation_index = 0
        self.cached_target: np.ndarray | None = None
        self.cached_operated: np.ndarray | None = None
        self.last_diagnostic: dict = {}
        self.last_gripper_closed_probability: np.ndarray | None = None
        self.last_gripper_retention_active: np.ndarray | None = None
        self.last_gripper_retention_normalized_state: np.ndarray | None = None
        self.last_gripper_retention_geometry: np.ndarray | None = None
        self.last_rotation_action_gate_probability: float | None = None
        self.last_rotation_action_head_active: np.ndarray | None = None
        self.last_translation_recovery_head_active: np.ndarray | None = None
        self.last_recovery_action_head_active: np.ndarray | None = None
        self.last_agent_state: np.ndarray | None = None
        self.binary_gripper_threshold = float(
            getattr(self.policy, "binary_gripper_threshold", 0.5)
        )
        self.rotation_gate_threshold = float(
            getattr(self.policy, "rotation_action_gate_threshold", 0.5)
        )
        self.rotation_support_min_deg = support_min
        self.rotation_support_max_deg = support_max
        self.translation_support_min_m = translation_support_min
        self.translation_support_max_m = translation_support_max
        self.translation_rotation_max_deg = translation_rotation_max
        self.rotation_cooldown_chunks = max(0, int(rotation_cooldown_chunks))
        self.rotation_cooldown_remaining = 0
        self.translation_cooldown_chunks = max(
            0,
            int(
                self.rotation_cooldown_chunks
                if translation_cooldown_chunks is None
                else translation_cooldown_chunks
            ),
        )
        self.translation_cooldown_remaining = 0
        self.rotation_min_consecutive_chunks = max(
            1, int(rotation_min_consecutive_chunks)
        )
        self.translation_min_consecutive_chunks = max(
            1,
            int(
                self.rotation_min_consecutive_chunks
                if translation_min_consecutive_chunks is None
                else translation_min_consecutive_chunks
            ),
        )
        self.rotation_requested_streak = 0
        self.last_rotation_action_head_requested: np.ndarray | None = None
        self.last_rotation_action_suppressed_by_persistence = False
        self.last_rotation_action_suppressed_by_cooldown = False

    def reset(self) -> None:
        self.history.clear()
        self.observation_index = 0
        self.cached_target = None
        self.cached_operated = None
        self.last_diagnostic = {}
        self.last_gripper_closed_probability = None
        self.last_gripper_retention_active = None
        self.last_gripper_retention_normalized_state = None
        self.last_gripper_retention_geometry = None
        self.last_rotation_action_gate_probability = None
        self.last_rotation_action_head_active = None
        self.last_translation_recovery_head_active = None
        self.last_recovery_action_head_active = None
        self.last_rotation_action_head_requested = None
        self.last_rotation_action_suppressed_by_persistence = False
        self.last_rotation_action_suppressed_by_cooldown = False
        self.rotation_cooldown_remaining = 0
        self.translation_cooldown_remaining = 0
        self.rotation_requested_streak = 0
        self.last_agent_state = None
        self.frame_provider.reset()

    def latch_target(self, point_cloud: np.ndarray) -> None:
        self.frame_provider.latch_target(point_cloud)
        self.cached_target = np.asarray(point_cloud, dtype=np.float32).copy()

    def _observation(self, observation: dict) -> dict[str, np.ndarray]:
        operated, target = _object_clouds(observation)
        if len(operated):
            self.cached_operated = operated.copy()
        elif self.cached_operated is None:
            raise ValueError("operated object A is not visible")
        if self.cached_target is None:
            if not len(target):
                raise ValueError("target object B is not visible at task start")
            self.cached_target = target.copy()
        operated = self.cached_operated
        target = self.cached_target
        sampled_a = deterministic_sample(
            operated,
            self.observation_points,
            seed=self.observation_index * 17 + 1,
            point_channels=3,
        )
        sampled_b = deterministic_sample(
            target,
            self.observation_points,
            seed=2,
            point_channels=3,
        )
        estimate = self.frame_provider.estimate(
            current_point_cloud=operated,
            target_point_cloud=target,
            mode="zero" if self.condition == "zero" else "clean",
        )
        agent_state = _endpose_state(observation).astype(np.float32)
        confidence = float(estimate.combined_confidence)
        if confidence > 0.0 and estimate.goal_transform is not None:
            goal = _goal_frame9(estimate.goal_transform)
            source_local, target_local = task_aligned_local_coordinates(
                sampled_a,
                sampled_b,
                goal,
                estimate.frame9_metric,
                scale_m=self.geometry_scale_m,
            )
            local = np.concatenate(
                (
                    np.concatenate(
                        (source_local, -np.ones((len(source_local), 1), np.float32)),
                        axis=-1,
                    ),
                    np.concatenate(
                        (target_local, np.ones((len(target_local), 1), np.float32)),
                        axis=-1,
                    ),
                ),
                axis=0,
            )
            global_relation = estimate.frame9_metric.astype(np.float32).copy()
            global_relation[:3] /= self.geometry_scale_m
            anchor_flow = task_aligned_anchor_flow_tokens(
                sampled_a,
                agent_state,
                goal,
                estimate.frame9_metric,
                scale_m=self.geometry_scale_m,
            )
        else:
            local = np.zeros((2 * self.observation_points, 4), dtype=np.float32)
            global_relation = np.zeros(9, dtype=np.float32)
            anchor_flow = np.zeros(
                (self.observation_points, 9), dtype=np.float32
            )
            confidence = 0.0
        rotation_rotvec = _frame9_rotation_rotvec(estimate.frame9_metric)
        rotation_norm = float(np.linalg.norm(rotation_rotvec))
        self.last_diagnostic = {
            "condition": self.condition,
            "confidence": confidence,
            "source_confidence": float(estimate.source_confidence),
            "target_confidence": float(estimate.target_confidence),
            "source_disagreement_deg": estimate.source_disagreement_deg,
            "source_input_points": estimate.source_input_points,
            "source_encoded_points": estimate.source_encoded_points,
            "estimated_frame9_metric": estimate.frame9_metric.astype(
                float
            ).tolist(),
            "estimated_translation_xyz_m": estimate.frame9_metric[:3].tolist(),
            "estimated_translation_norm_m": float(
                np.linalg.norm(estimate.frame9_metric[:3])
            ),
            "estimated_rotation_deg": float(np.rad2deg(rotation_norm)),
            "estimated_rotation_rotvec_rad": rotation_rotvec.astype(float).tolist(),
            "estimated_rotation_axis": (
                np.zeros(3, dtype=np.float64)
                if rotation_norm <= 1.0e-12
                else rotation_rotvec / rotation_norm
            ).astype(float).tolist(),
            "target_latched": bool(estimate.target_latched),
        }
        self.observation_index += 1
        self.last_agent_state = agent_state.copy()
        result = {
            "point_cloud": sampled_a,
            "point_cloud_B": sampled_b,
            "agent_pos": agent_state,
            "tagrt_local": local.astype(np.float32),
            "tagrt_global": global_relation.astype(np.float32),
            "tagrt_confidence": np.asarray([confidence], dtype=np.float32),
        }
        if (
            getattr(
                self.policy,
                "rotation_action_translation_anchor_flow_key",
                None,
            )
            is not None
        ):
            result["tagrt_anchor_flow"] = anchor_flow.astype(np.float32)
        return result

    @torch.no_grad()
    def predict(self, observation: dict) -> np.ndarray:
        encoded = self._observation(observation)
        if not self.history:
            for _ in range(self.n_obs_steps):
                self.history.append({key: value.copy() for key, value in encoded.items()})
        else:
            self.history.append(encoded)
        stacked = {
            key: np.stack([item[key] for item in self.history], axis=0)[None]
            for key in encoded
        }
        tensor = dict_apply(
            stacked,
            lambda value: torch.from_numpy(value).to(self.device, non_blocking=True),
        )
        prediction = self.policy.predict_action(tensor)
        persistence_required = self.rotation_min_consecutive_chunks
        translation_request = prediction.get("translation_recovery_head_active")
        rotation_request = prediction.get("rotation_action_head_active")
        if (
            translation_request is not None
            and bool(translation_request.any())
            and (rotation_request is None or not bool(rotation_request.any()))
        ):
            persistence_required = self.translation_min_consecutive_chunks
        (
            self.rotation_requested_streak,
            self.last_rotation_action_head_requested,
            self.last_rotation_action_suppressed_by_persistence,
        ) = _apply_rotation_recovery_persistence(
            prediction,
            self.rotation_requested_streak,
            persistence_required,
        )
        translation_request = prediction.get("translation_recovery_head_active")
        rotation_request = prediction.get("rotation_action_head_active")
        translation_only = (
            translation_request is not None
            and bool(translation_request.any())
            and (rotation_request is None or not bool(rotation_request.any()))
        )
        if translation_only:
            self.rotation_cooldown_remaining = max(
                self.rotation_cooldown_remaining - 1, 0
            )
            (
                self.translation_cooldown_remaining,
                _cooldown_requested,
                self.last_rotation_action_suppressed_by_cooldown,
            ) = _apply_rotation_recovery_cooldown(
                prediction,
                self.translation_cooldown_remaining,
                self.translation_cooldown_chunks,
            )
        else:
            self.translation_cooldown_remaining = max(
                self.translation_cooldown_remaining - 1, 0
            )
            (
                self.rotation_cooldown_remaining,
                _cooldown_requested,
                self.last_rotation_action_suppressed_by_cooldown,
            ) = _apply_rotation_recovery_cooldown(
                prediction,
                self.rotation_cooldown_remaining,
                self.rotation_cooldown_chunks,
            )
        probability = prediction.get("binary_gripper_closed_probability")
        self.last_gripper_closed_probability = (
            None
            if probability is None
            else probability[0].detach().cpu().numpy().astype(np.float64)
        )
        retention_active = prediction.get("binary_gripper_retention_active")
        retention_state = prediction.get(
            "binary_gripper_retention_normalized_state"
        )
        retention_geometry = prediction.get("binary_gripper_retention_geometry")
        rotation_gate_probability = prediction.get(
            "rotation_action_gate_probability"
        )
        rotation_head_active = prediction.get("rotation_action_head_active")
        translation_head_active = prediction.get(
            "translation_recovery_head_active"
        )
        recovery_head_active = prediction.get("recovery_action_head_active")
        self.last_gripper_retention_active = (
            None
            if retention_active is None
            else retention_active[0].detach().cpu().numpy().astype(bool)
        )
        self.last_gripper_retention_normalized_state = (
            None
            if retention_state is None
            else retention_state[0].detach().cpu().numpy().astype(np.float64)
        )
        self.last_gripper_retention_geometry = (
            None
            if retention_geometry is None
            else retention_geometry[0].detach().cpu().numpy().astype(np.float64)
        )
        self.last_rotation_action_gate_probability = (
            None
            if rotation_gate_probability is None
            else float(rotation_gate_probability[0].detach().cpu())
        )
        self.last_rotation_action_head_active = (
            None
            if rotation_head_active is None
            else rotation_head_active[0].detach().cpu().numpy().astype(bool)
        )
        self.last_translation_recovery_head_active = (
            None
            if translation_head_active is None
            else translation_head_active[0].detach().cpu().numpy().astype(bool)
        )
        self.last_recovery_action_head_active = (
            None
            if recovery_head_active is None
            else recovery_head_active[0].detach().cpu().numpy().astype(bool)
        )
        return prediction["action"][0].cpu().numpy()


def get_model(usr_args):
    device = str(usr_args.get("tagrt_dp3_device", "cuda:0"))
    frame_checkpoints = _path_list(
        usr_args.get("tagrt_dp3_frame_checkpoints")
    )
    provider = OnlineNdfFunctionalFrameProvider.from_metadata(
        ndf_checkpoints=frame_checkpoints,
        camera_metadata=usr_args["tagrt_dp3_camera_metadata"],
        ensemble_metadata=usr_args["tagrt_dp3_ensemble_metadata"],
        device=device,
    )
    runtime = TagrtDP3Runtime(
        usr_args["tagrt_dp3_checkpoint"],
        usr_args["tagrt_dp3_dataset"],
        frame_provider=provider,
        device=device,
        condition=str(usr_args.get("tagrt_dp3_condition", "correct")),
        binary_gripper_threshold=(
            None
            if usr_args.get("tagrt_dp3_binary_gripper_threshold") is None
            else float(usr_args["tagrt_dp3_binary_gripper_threshold"])
        ),
        rotation_gate_threshold=(
            None
            if usr_args.get("tagrt_dp3_rotation_gate_threshold") is None
            else float(usr_args["tagrt_dp3_rotation_gate_threshold"])
        ),
        rotation_support_min_deg=(
            None
            if usr_args.get("tagrt_dp3_rotation_support_min_deg") is None
            else float(usr_args["tagrt_dp3_rotation_support_min_deg"])
        ),
        rotation_support_max_deg=(
            None
            if usr_args.get("tagrt_dp3_rotation_support_max_deg") is None
            else float(usr_args["tagrt_dp3_rotation_support_max_deg"])
        ),
        translation_support_min_m=(
            None
            if usr_args.get("tagrt_dp3_translation_support_min_m") is None
            else float(usr_args["tagrt_dp3_translation_support_min_m"])
        ),
        translation_support_max_m=(
            None
            if usr_args.get("tagrt_dp3_translation_support_max_m") is None
            else float(usr_args["tagrt_dp3_translation_support_max_m"])
        ),
        translation_rotation_max_deg=(
            None
            if usr_args.get("tagrt_dp3_translation_rotation_max_deg") is None
            else float(usr_args["tagrt_dp3_translation_rotation_max_deg"])
        ),
        rotation_cooldown_chunks=int(
            usr_args.get("tagrt_dp3_rotation_cooldown_chunks", 0)
        ),
        translation_cooldown_chunks=(
            None
            if usr_args.get("tagrt_dp3_translation_cooldown_chunks") is None
            else int(usr_args["tagrt_dp3_translation_cooldown_chunks"])
        ),
        rotation_min_consecutive_chunks=int(
            usr_args.get("tagrt_dp3_rotation_min_consecutive_chunks", 1)
        ),
        translation_min_consecutive_chunks=(
            None
            if usr_args.get("tagrt_dp3_translation_min_consecutive_chunks") is None
            else int(usr_args["tagrt_dp3_translation_min_consecutive_chunks"])
        ),
    )
    model = SimpleNamespace(
        runtime=runtime,
        execute_steps=max(1, int(usr_args.get("tagrt_dp3_execute_steps", 1))),
        max_policy_steps=max(0, int(usr_args.get("tagrt_dp3_max_policy_steps", 300))),
        max_eef_translation_delta_m=float(
            usr_args.get("max_eef_translation_delta_m", 0.08)
        ),
        max_eef_rotation_delta_rad=float(
            usr_args.get("max_eef_rotation_delta_rad", 0.5)
        ),
        action_chunks=0,
        executed_actions=0,
        confidence_sum=0.0,
        valid_geometry_chunks=0,
        predicted_translation_norm_sum=0.0,
        predicted_translation_norm_max=0.0,
        predicted_rotation_norm_sum=0.0,
        predicted_rotation_norm_max=0.0,
        predicted_gripper_sum=np.zeros(2, dtype=np.float64),
        predicted_gripper_closed=np.zeros(2, dtype=np.int64),
        predicted_gripper_closed_probability_sum=np.zeros(2, dtype=np.float64),
        predicted_gripper_closed_probability_min=np.full(2, np.inf, dtype=np.float64),
        predicted_gripper_closed_probability_max=np.full(2, -np.inf, dtype=np.float64),
        predicted_gripper_closed_probability_count=0,
        retention_geometry_at_min_left_probability=None,
        estimated_translation_norm_min=np.inf,
        estimated_rotation_deg_min=np.inf,
        first_active_release_step=None,
        first_active_release_geometry_translation_m=None,
        first_active_release_geometry_rotation_deg=None,
        first_active_release_geometry_confidence=None,
        action_trace=[],
        rotation_gate_probability_sum=0.0,
        rotation_gate_probability_max=0.0,
        rotation_gate_probability_count=0,
        rotation_head_active_chunks=0,
    )

    def metrics():
        count = max(model.action_chunks, 1)
        return {
            "tagrt_dp3_checkpoint": str(runtime.checkpoint),
            "tagrt_dp3_condition": runtime.condition,
            "tagrt_dp3_binary_gripper_threshold": (
                runtime.binary_gripper_threshold
            ),
            "tagrt_dp3_rotation_gate_threshold": (
                runtime.rotation_gate_threshold
            ),
            "tagrt_dp3_rotation_support_min_deg": (
                runtime.rotation_support_min_deg
            ),
            "tagrt_dp3_rotation_support_max_deg": (
                runtime.rotation_support_max_deg
            ),
            "tagrt_dp3_translation_support_min_m": (
                runtime.translation_support_min_m
            ),
            "tagrt_dp3_translation_support_max_m": (
                runtime.translation_support_max_m
            ),
            "tagrt_dp3_translation_rotation_max_deg": (
                runtime.translation_rotation_max_deg
            ),
            "tagrt_dp3_rotation_cooldown_chunks": (
                runtime.rotation_cooldown_chunks
            ),
            "tagrt_dp3_translation_cooldown_chunks": (
                runtime.translation_cooldown_chunks
            ),
            "tagrt_dp3_rotation_min_consecutive_chunks": (
                runtime.rotation_min_consecutive_chunks
            ),
            "tagrt_dp3_translation_min_consecutive_chunks": (
                runtime.translation_min_consecutive_chunks
            ),
            "tagrt_dp3_action_chunks": int(model.action_chunks),
            "tagrt_dp3_executed_actions": int(model.executed_actions),
            "tagrt_dp3_mean_confidence": float(model.confidence_sum / count),
            "tagrt_dp3_valid_geometry_fraction": float(
                model.valid_geometry_chunks / count
            ),
            "tagrt_dp3_mean_predicted_translation_delta_m": float(
                model.predicted_translation_norm_sum
                / max(2 * model.executed_actions, 1)
            ),
            "tagrt_dp3_max_predicted_translation_delta_m": float(
                model.predicted_translation_norm_max
            ),
            "tagrt_dp3_mean_predicted_rotation_delta_deg": float(
                np.rad2deg(
                    model.predicted_rotation_norm_sum
                    / max(2 * model.executed_actions, 1)
                )
            ),
            "tagrt_dp3_max_predicted_rotation_delta_deg": float(
                np.rad2deg(model.predicted_rotation_norm_max)
            ),
            "tagrt_dp3_mean_left_gripper_command": float(
                model.predicted_gripper_sum[0] / max(model.executed_actions, 1)
            ),
            "tagrt_dp3_mean_right_gripper_command": float(
                model.predicted_gripper_sum[1] / max(model.executed_actions, 1)
            ),
            "tagrt_dp3_left_gripper_closed_fraction": float(
                model.predicted_gripper_closed[0]
                / max(model.executed_actions, 1)
            ),
            "tagrt_dp3_right_gripper_closed_fraction": float(
                model.predicted_gripper_closed[1]
                / max(model.executed_actions, 1)
            ),
            "tagrt_dp3_min_estimated_translation_norm_m": (
                None
                if not np.isfinite(model.estimated_translation_norm_min)
                else float(model.estimated_translation_norm_min)
            ),
            "tagrt_dp3_min_estimated_rotation_deg": (
                None
                if not np.isfinite(model.estimated_rotation_deg_min)
                else float(model.estimated_rotation_deg_min)
            ),
            "tagrt_dp3_mean_gripper_closed_probability": (
                model.predicted_gripper_closed_probability_sum
                / max(model.predicted_gripper_closed_probability_count, 1)
            ).astype(float).tolist(),
            "tagrt_dp3_min_gripper_closed_probability": np.where(
                np.isfinite(model.predicted_gripper_closed_probability_min),
                model.predicted_gripper_closed_probability_min,
                np.nan,
            ).astype(float).tolist(),
            "tagrt_dp3_max_gripper_closed_probability": np.where(
                np.isfinite(model.predicted_gripper_closed_probability_max),
                model.predicted_gripper_closed_probability_max,
                np.nan,
            ).astype(float).tolist(),
            "tagrt_dp3_last_gripper_retention_active": (
                None
                if runtime.last_gripper_retention_active is None
                else runtime.last_gripper_retention_active.tolist()
            ),
            "tagrt_dp3_last_gripper_retention_normalized_state": (
                None
                if runtime.last_gripper_retention_normalized_state is None
                else runtime.last_gripper_retention_normalized_state.astype(
                    float
                ).tolist()
            ),
            "tagrt_dp3_last_gripper_retention_geometry": (
                None
                if runtime.last_gripper_retention_geometry is None
                else runtime.last_gripper_retention_geometry.astype(float).tolist()
            ),
            "tagrt_dp3_retention_geometry_at_min_left_probability": (
                None
                if model.retention_geometry_at_min_left_probability is None
                else model.retention_geometry_at_min_left_probability.astype(
                    float
                ).tolist()
            ),
            "tagrt_dp3_first_active_release_step": model.first_active_release_step,
            "tagrt_dp3_first_active_release_geometry_translation_m": (
                model.first_active_release_geometry_translation_m
            ),
            "tagrt_dp3_first_active_release_geometry_rotation_deg": (
                model.first_active_release_geometry_rotation_deg
            ),
            "tagrt_dp3_first_active_release_geometry_confidence": (
                model.first_active_release_geometry_confidence
            ),
            "tagrt_dp3_action_trace": list(model.action_trace),
            "tagrt_dp3_mean_rotation_gate_probability": (
                None
                if model.rotation_gate_probability_count == 0
                else float(
                    model.rotation_gate_probability_sum
                    / model.rotation_gate_probability_count
                )
            ),
            "tagrt_dp3_max_rotation_gate_probability": (
                None
                if model.rotation_gate_probability_count == 0
                else float(model.rotation_gate_probability_max)
            ),
            "tagrt_dp3_rotation_head_active_chunks": int(
                model.rotation_head_active_chunks
            ),
            **{
                f"tagrt_dp3_{key}": value
                for key, value in runtime.last_diagnostic.items()
            },
        }

    model.get_evaluation_metrics = metrics
    return model


def eval(TASK_ENV, model, observation):
    actions = model.runtime.predict(observation)
    chunk_trace = {
        "chunk_index": int(model.action_chunks),
        "executed_action_start": int(model.executed_actions),
        "camera_geometry": dict(model.runtime.last_diagnostic),
        "agent_state20": (
            None
            if model.runtime.last_agent_state is None
            else model.runtime.last_agent_state.astype(float).tolist()
        ),
        "predicted_action14": np.asarray(actions, dtype=np.float64).tolist(),
        "gripper_closed_probability": (
            None
            if model.runtime.last_gripper_closed_probability is None
            else model.runtime.last_gripper_closed_probability.astype(float).tolist()
        ),
        "gripper_retention_active": (
            None
            if model.runtime.last_gripper_retention_active is None
            else model.runtime.last_gripper_retention_active.tolist()
        ),
        "gripper_retention_normalized_state": (
            None
            if model.runtime.last_gripper_retention_normalized_state is None
            else model.runtime.last_gripper_retention_normalized_state.astype(
                float
            ).tolist()
        ),
        "rotation_action_gate_probability": (
            model.runtime.last_rotation_action_gate_probability
        ),
        "rotation_action_head_active": (
            None
            if model.runtime.last_rotation_action_head_active is None
            else model.runtime.last_rotation_action_head_active.tolist()
        ),
        "translation_recovery_head_active": (
            None
            if model.runtime.last_translation_recovery_head_active is None
            else model.runtime.last_translation_recovery_head_active.tolist()
        ),
        "recovery_action_head_active": (
            None
            if model.runtime.last_recovery_action_head_active is None
            else model.runtime.last_recovery_action_head_active.tolist()
        ),
        "rotation_action_head_requested": (
            None
            if model.runtime.last_rotation_action_head_requested is None
            else model.runtime.last_rotation_action_head_requested.tolist()
        ),
        "rotation_action_suppressed_by_cooldown": bool(
            model.runtime.last_rotation_action_suppressed_by_cooldown
        ),
        "rotation_action_suppressed_by_persistence": bool(
            model.runtime.last_rotation_action_suppressed_by_persistence
        ),
        "rotation_requested_streak": int(
            model.runtime.rotation_requested_streak
        ),
        "rotation_cooldown_remaining": int(
            model.runtime.rotation_cooldown_remaining
        ),
        "translation_cooldown_remaining": int(
            model.runtime.translation_cooldown_remaining
        ),
        "privileged_task_metrics_before_chunk": _selected_task_pose_metrics(
            TASK_ENV
        ),
        "executed": [],
    }
    model.action_trace.append(chunk_trace)
    model.action_chunks += 1
    model.confidence_sum += float(
        model.runtime.last_diagnostic.get("confidence", 0.0)
    )
    model.valid_geometry_chunks += int(
        float(model.runtime.last_diagnostic.get("confidence", 0.0)) > 0.0
    )
    if model.runtime.last_rotation_action_gate_probability is not None:
        gate_probability = float(
            model.runtime.last_rotation_action_gate_probability
        )
        model.rotation_gate_probability_sum += gate_probability
        model.rotation_gate_probability_max = max(
            model.rotation_gate_probability_max, gate_probability
        )
        model.rotation_gate_probability_count += 1
    if (
        model.runtime.last_rotation_action_head_active is not None
        and bool(np.any(model.runtime.last_rotation_action_head_active))
    ):
        model.rotation_head_active_chunks += 1
    estimated_translation = model.runtime.last_diagnostic.get(
        "estimated_translation_norm_m"
    )
    estimated_rotation = model.runtime.last_diagnostic.get("estimated_rotation_deg")
    if estimated_translation is not None:
        model.estimated_translation_norm_min = min(
            model.estimated_translation_norm_min, float(estimated_translation)
        )
    if estimated_rotation is not None:
        model.estimated_rotation_deg_min = min(
            model.estimated_rotation_deg_min, float(estimated_rotation)
        )
    endpose = dict(observation["endpose"])
    probabilities = model.runtime.last_gripper_closed_probability
    for action_index, raw_action in enumerate(actions[: model.execute_steps]):
        action = np.asarray(raw_action, dtype=np.float32).copy()
        action[6] = np.clip(action[6], 0.0, 1.0)
        action[13] = np.clip(action[13], 0.0, 1.0)
        gripper_now = np.asarray(
            [endpose["left_gripper"], endpose["right_gripper"]],
            dtype=np.float32,
        )
        active_release = (gripper_now < 0.5) & (action[[6, 13]] >= 0.5)
        if model.first_active_release_step is None and bool(np.any(active_release)):
            diagnostic = model.runtime.last_diagnostic
            model.first_active_release_step = int(model.executed_actions)
            model.first_active_release_geometry_translation_m = diagnostic.get(
                "estimated_translation_norm_m"
            )
            model.first_active_release_geometry_rotation_deg = diagnostic.get(
                "estimated_rotation_deg"
            )
            model.first_active_release_geometry_confidence = diagnostic.get(
                "confidence"
            )
        translation_norms = (
            float(np.linalg.norm(action[:3])),
            float(np.linalg.norm(action[7:10])),
        )
        rotation_norms = (
            float(np.linalg.norm(action[3:6])),
            float(np.linalg.norm(action[10:13])),
        )
        model.predicted_translation_norm_sum += sum(translation_norms)
        model.predicted_translation_norm_max = max(
            model.predicted_translation_norm_max, *translation_norms
        )
        model.predicted_rotation_norm_sum += sum(rotation_norms)
        model.predicted_rotation_norm_max = max(
            model.predicted_rotation_norm_max, *rotation_norms
        )
        model.predicted_gripper_sum += action[[6, 13]]
        model.predicted_gripper_closed += (action[[6, 13]] < 0.5).astype(
            np.int64
        )
        if probabilities is not None:
            probability = np.asarray(probabilities[action_index], dtype=np.float64)
            if probability[0] < model.predicted_gripper_closed_probability_min[0]:
                geometry = model.runtime.last_gripper_retention_geometry
                model.retention_geometry_at_min_left_probability = (
                    None if geometry is None else geometry[-1].copy()
                )
            model.predicted_gripper_closed_probability_sum += probability
            model.predicted_gripper_closed_probability_min = np.minimum(
                model.predicted_gripper_closed_probability_min, probability
            )
            model.predicted_gripper_closed_probability_max = np.maximum(
                model.predicted_gripper_closed_probability_max, probability
            )
            model.predicted_gripper_closed_probability_count += 1
        decoded = decode_eef_delta_action16(
            action,
            endpose["left_endpose"],
            endpose["right_endpose"],
            max_translation_delta_m=model.max_eef_translation_delta_m,
            max_rotation_delta_rad=model.max_eef_rotation_delta_rad,
        )
        TASK_ENV.take_action(decoded, action_type="ee")
        geometry_rotvec = np.asarray(
            model.runtime.last_diagnostic.get(
                "estimated_rotation_rotvec_rad", [0.0, 0.0, 0.0]
            ),
            dtype=np.float64,
        )
        geometry_rotation_norm = float(np.linalg.norm(geometry_rotvec))
        action_alignment = []
        for rotation_slice in (slice(3, 6), slice(10, 13)):
            action_rotvec = np.asarray(action[rotation_slice], dtype=np.float64)
            denominator = geometry_rotation_norm * float(np.linalg.norm(action_rotvec))
            action_alignment.append(
                None
                if denominator <= 1.0e-12
                else float(np.dot(geometry_rotvec, action_rotvec) / denominator)
            )
        chunk_trace["executed"].append(
            {
                "action_index": int(action_index),
                "global_action_index": int(model.executed_actions),
                "gripper_state_before": gripper_now.astype(float).tolist(),
                "action14": action.astype(float).tolist(),
                "geometry_rotation_action_cosine_left_right": action_alignment,
                "privileged_task_metrics_after_action": _selected_task_pose_metrics(
                    TASK_ENV
                ),
            }
        )
        model.executed_actions += 1
        # Each stored action is a delta from the immediately preceding EEF
        # pose. A multi-step DP3 chunk must therefore be decoded cumulatively
        # against the actually reached pose.
        endpose["left_endpose"] = TASK_ENV.get_arm_pose("left")
        endpose["right_endpose"] = TASK_ENV.get_arm_pose("right")
        if (
            model.max_policy_steps > 0
            and model.executed_actions >= model.max_policy_steps
        ):
            TASK_ENV.step_lim = TASK_ENV.take_action_cnt
            break
        if TASK_ENV.eval_success or TASK_ENV.take_action_cnt >= TASK_ENV.step_lim:
            break


def reset_model(model):
    model.runtime.reset()
    model.action_chunks = 0
    model.executed_actions = 0
    model.confidence_sum = 0.0
    model.valid_geometry_chunks = 0
    model.predicted_translation_norm_sum = 0.0
    model.predicted_translation_norm_max = 0.0
    model.predicted_rotation_norm_sum = 0.0
    model.predicted_rotation_norm_max = 0.0
    model.predicted_gripper_sum.fill(0.0)
    model.predicted_gripper_closed.fill(0)
    model.predicted_gripper_closed_probability_sum.fill(0.0)
    model.predicted_gripper_closed_probability_min.fill(np.inf)
    model.predicted_gripper_closed_probability_max.fill(-np.inf)
    model.predicted_gripper_closed_probability_count = 0
    model.retention_geometry_at_min_left_probability = None
    model.estimated_translation_norm_min = np.inf
    model.estimated_rotation_deg_min = np.inf
    model.first_active_release_step = None
    model.first_active_release_geometry_translation_m = None
    model.first_active_release_geometry_rotation_deg = None
    model.first_active_release_geometry_confidence = None
    model.rotation_gate_probability_sum = 0.0
    model.rotation_gate_probability_max = 0.0
    model.rotation_gate_probability_count = 0
    model.rotation_head_active_chunks = 0
    model.action_trace.clear()


def latch_target_observations(model, observations) -> dict:
    target_clouds = []
    for observation in observations:
        _operated, target = _object_clouds(observation)
        if len(target):
            target_clouds.append(target)
    if not target_clouds:
        raise ValueError("target latch received no valid B observations")
    fused = np.concatenate(target_clouds, axis=0)
    model.runtime.latch_target(fused)
    return {
        "requested_frames": int(len(observations)),
        "valid_frames": int(len(target_clouds)),
        "fused_points": int(len(fused)),
    }
