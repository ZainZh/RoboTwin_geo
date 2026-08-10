"""Full-task RoboTwin deployment adapter for a TAGRT-conditioned DP3 policy."""

from __future__ import annotations

from collections import deque
from pathlib import Path
import sys
from types import SimpleNamespace

import dill
import numpy as np
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
    value = np.asarray(frame9, dtype=np.float64).reshape(9)
    first = value[3:6]
    first /= max(float(np.linalg.norm(first)), 1.0e-8)
    second = value[6:9] - first * float(np.dot(first, value[6:9]))
    second /= max(float(np.linalg.norm(second)), 1.0e-8)
    rotation = np.column_stack((first, second, np.cross(first, second)))
    cosine = np.clip((float(np.trace(rotation)) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))


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
        self.binary_gripper_threshold = float(
            getattr(self.policy, "binary_gripper_threshold", 0.5)
        )

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
        else:
            local = np.zeros((2 * self.observation_points, 4), dtype=np.float32)
            global_relation = np.zeros(9, dtype=np.float32)
            confidence = 0.0
        self.last_diagnostic = {
            "condition": self.condition,
            "confidence": confidence,
            "source_confidence": float(estimate.source_confidence),
            "target_confidence": float(estimate.target_confidence),
            "source_disagreement_deg": estimate.source_disagreement_deg,
            "source_input_points": estimate.source_input_points,
            "source_encoded_points": estimate.source_encoded_points,
            "estimated_translation_xyz_m": estimate.frame9_metric[:3].tolist(),
            "estimated_translation_norm_m": float(
                np.linalg.norm(estimate.frame9_metric[:3])
            ),
            "estimated_rotation_deg": _frame9_rotation_angle_deg(
                estimate.frame9_metric
            ),
            "target_latched": bool(estimate.target_latched),
        }
        self.observation_index += 1
        return {
            "point_cloud": sampled_a,
            "point_cloud_B": sampled_b,
            "agent_pos": _endpose_state(observation).astype(np.float32),
            "tagrt_local": local.astype(np.float32),
            "tagrt_global": global_relation.astype(np.float32),
            "tagrt_confidence": np.asarray([confidence], dtype=np.float32),
        }

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
    )

    def metrics():
        count = max(model.action_chunks, 1)
        return {
            "tagrt_dp3_checkpoint": str(runtime.checkpoint),
            "tagrt_dp3_condition": runtime.condition,
            "tagrt_dp3_binary_gripper_threshold": (
                runtime.binary_gripper_threshold
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
            **{
                f"tagrt_dp3_{key}": value
                for key, value in runtime.last_diagnostic.items()
            },
        }

    model.get_evaluation_metrics = metrics
    return model


def eval(TASK_ENV, model, observation):
    actions = model.runtime.predict(observation)
    model.action_chunks += 1
    model.confidence_sum += float(
        model.runtime.last_diagnostic.get("confidence", 0.0)
    )
    model.valid_geometry_chunks += int(
        float(model.runtime.last_diagnostic.get("confidence", 0.0)) > 0.0
    )
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
