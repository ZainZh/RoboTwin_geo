"""Inference wrapper for interaction-centric geometric flow-token policies."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch

from .build_grasp_dataset import deterministic_farthest_points, valid_xyz
from .functional_action_frame import (
    action14_from_target_frame,
    frame9_rotation,
    relative_frame9_to_target_frame,
    transform_eef_state20_to_target_frame,
    transform_point_features_to_target_frame,
)
from .interaction_flow_tokens import attention_entropy
from .hand_object_frame import (
    HAND_OBJECT_TRANSLATION_SCALE_M,
    active_hand_object_frame11,
)
from .ndf_adapter import NdfPointwiseAdapter
from .online_ndf_functional_frame import OnlineNdfFunctionalFrameProvider
from .target_frame import estimate_geometry_marker_frame
from .train_interaction_flow_tokens import build_model
from .train_task_flow_benchmark import Normalization, normalize_point_features


class InteractionFlowRuntime:
    """Load a camera-only checkpoint and predict a six-step EEF-delta chunk.

    The predicted flow is an auxiliary output, not a policy input.  At runtime
    the model consumes only segmented A/B point clouds and the 20-D EEF state.
    """

    SUPPORTED_CONDITIONS = {
        "point_tokens",
        "interaction_tokens",
        "interaction_flow",
        "interaction_flow_conditioned",
        "interaction_functional",
        "interaction_functional_flow",
        "interaction_functional_direct",
        "interaction_functional_direct_flow",
        "interaction_functional_dual_frame_flow",
        "interaction_functional_correction_adapter_flow",
        "interaction_functional_equivariant_adapter_flow",
        "interaction_functional_local_equivariant_adapter_flow",
        "interaction_functional_hybrid_equivariant_adapter_flow",
        "interaction_functional_biframe_adapter_flow",
        "interaction_functional_gated_biframe_adapter_flow",
        "interaction_functional_grasp_adapter_flow",
        "interaction_functional_gated",
        "interaction_functional_gated_flow",
    }
    FUNCTIONAL_FRAME_CONDITIONS = {
        "interaction_functional",
        "interaction_functional_flow",
        "interaction_functional_direct",
        "interaction_functional_direct_flow",
        "interaction_functional_dual_frame_flow",
        "interaction_functional_correction_adapter_flow",
        "interaction_functional_equivariant_adapter_flow",
        "interaction_functional_local_equivariant_adapter_flow",
        "interaction_functional_hybrid_equivariant_adapter_flow",
        "interaction_functional_biframe_adapter_flow",
        "interaction_functional_gated_biframe_adapter_flow",
        "interaction_functional_grasp_adapter_flow",
        "interaction_functional_gated",
        "interaction_functional_gated_flow",
    }

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "cpu",
        observation_points: int = 64,
        target_color_priority: bool | None = None,
        ndf_checkpoint: str | Path | None = None,
        source_axis_mode: str = "clean",
        target_axis_mode: str = "clean",
        dense_frame_checkpoints: list[str | Path] | tuple[str | Path, ...] | None = None,
        dense_camera_metadata: str | Path | None = None,
        dense_ensemble_metadata: str | Path | None = None,
        dense_frame_mode: str = "off",
        allow_privileged_oracle: bool = False,
        dense_frame_provider: object | None = None,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.device = torch.device(device)
        payload = torch.load(
            self.checkpoint, map_location=self.device, weights_only=False
        )
        self.condition = str(payload["condition"])
        if self.condition not in self.SUPPORTED_CONDITIONS:
            raise ValueError(
                f"unsupported interaction-flow condition {self.condition!r}"
            )
        self.horizon = int(payload["horizon"])
        self.flow_steps = int(payload["flow_steps"])
        self.num_anchors = int(payload["num_anchors"])
        self.feature_dim = int(payload["feature_dim"])
        self.layers = int(payload["layers"])
        self.point_channels = int(payload.get("point_channels", 3))
        self.target_color_adapter = bool(
            payload.get("target_color_adapter", False)
        )
        self.source_geometry_adapter = bool(
            payload.get("source_geometry_adapter", False)
        )
        self.source_axis_relation_adapter = bool(
            payload.get("source_axis_relation_adapter", False)
        )
        self.target_axis_relation_adapter = bool(
            payload.get("target_axis_relation_adapter", False)
        )
        self.hand_object_frame_token = bool(
            payload.get("hand_object_frame_token", False)
        )
        expected_recovery_action_frame = (
            "hybrid_source_rotation"
            if self.condition
            == "interaction_functional_hybrid_equivariant_adapter_flow"
            else "source_functional"
            if self.condition in {
                "interaction_functional_equivariant_adapter_flow",
                "interaction_functional_local_equivariant_adapter_flow",
            }
            else "world"
        )
        self.recovery_action_frame = str(
            payload.get(
                "recovery_action_frame", expected_recovery_action_frame
            )
        )
        if self.recovery_action_frame != expected_recovery_action_frame:
            raise ValueError(
                "checkpoint recovery_action_frame is inconsistent with its "
                f"condition: {self.recovery_action_frame!r}"
            )
        expected_local_frame_token = (
            self.condition
            in {
                "interaction_functional_local_equivariant_adapter_flow",
                "interaction_functional_hybrid_equivariant_adapter_flow",
                "interaction_functional_biframe_adapter_flow",
                "interaction_functional_gated_biframe_adapter_flow",
            }
        )
        self.recovery_local_frame_token = bool(
            payload.get(
                "recovery_local_frame_token", expected_local_frame_token
            )
        )
        if self.recovery_local_frame_token != expected_local_frame_token:
            raise ValueError(
                "checkpoint recovery_local_frame_token is inconsistent with "
                "its condition"
            )
        self.hand_object_translation_scale_m = float(
            payload.get(
                "hand_object_translation_scale_m",
                HAND_OBJECT_TRANSLATION_SCALE_M,
            )
        )
        self.functional_frame_encoding = str(
            payload.get("functional_frame_encoding", "target_frame9_columns_v1")
        )
        self.functional_frame_translation_mode = str(
            payload.get("functional_frame_translation_mode", "absolute")
        )
        marker_camera_relative = bool(
            self.condition in self.FUNCTIONAL_FRAME_CONDITIONS
            and self.functional_frame_encoding
            == "camera_ndf_current_marker_goal_se3_columns_v1"
        )
        supplied_camera_relative = bool(
            dense_frame_provider is not None
            and self.condition in self.FUNCTIONAL_FRAME_CONDITIONS
            and self.functional_frame_encoding
            in {
                "camera_ndf_current_marker_goal_se3_columns_v1",
                "dataset_relative_se3_columns_v1",
            }
        )
        self.dense_camera_relative = bool(
            marker_camera_relative or supplied_camera_relative
        )
        if (
            (
                self.recovery_action_frame != "world"
                or self.recovery_local_frame_token
            )
            and not self.dense_camera_relative
        ):
            raise ValueError(
                "source-functional recovery requires the deployable camera/NDF "
                "current-object frame"
            )
        if dense_frame_mode not in {"off", "clean", "inverse", "zero", "oracle"}:
            raise ValueError(
                "dense_frame_mode must be off, clean, inverse, zero, or oracle"
            )
        self.dense_frame_mode = str(dense_frame_mode)
        self.allow_privileged_oracle = bool(allow_privileged_oracle)
        if self.dense_camera_relative and self.dense_frame_mode == "off":
            raise ValueError(
                "dense camera-relative checkpoint requires an explicit "
                "dense_frame_mode (clean, inverse, zero, or oracle)"
            )
        if not self.dense_camera_relative and self.dense_frame_mode != "off":
            raise ValueError(
                "dense_frame_mode requires a deployable camera-relative "
                "checkpoint/provider pair"
            )
        if self.dense_camera_relative:
            if int(observation_points) != 128:
                raise ValueError(
                    "dense camera-relative NDF policy requires exactly 128 "
                    "policy/FPS points"
                )
            if self.functional_frame_translation_mode != "delta":
                raise ValueError(
                    "dense camera-relative checkpoint must use delta translation"
                )
            if self.dense_frame_mode == "oracle" and not self.allow_privileged_oracle:
                raise PermissionError(
                    "dense oracle mode requires allow_privileged_oracle=True"
                )
        self.dense_frame_provider = dense_frame_provider
        if marker_camera_relative and self.dense_frame_provider is None:
            checkpoints = list(dense_frame_checkpoints or ())
            if len(checkpoints) < 2:
                raise ValueError(
                    "dense clean/inverse/zero/oracle protocol requires at least two "
                    "NDF functional-frame checkpoints"
                )
            if dense_camera_metadata in {None, ""}:
                raise ValueError("dense camera metadata path is required")
            if dense_ensemble_metadata in {None, ""}:
                raise ValueError("dense ensemble metadata path is required")
            self.dense_frame_provider = OnlineNdfFunctionalFrameProvider.from_metadata(
                ndf_checkpoints=checkpoints,
                camera_metadata=Path(dense_camera_metadata),
                ensemble_metadata=Path(dense_ensemble_metadata),
                device=str(self.device),
                allow_privileged_oracle=self.allow_privileged_oracle,
            )
        if source_axis_mode not in {"clean", "zero", "orthogonal", "sign_flip"}:
            raise ValueError(
                "source_axis_mode must be clean, zero, orthogonal, or sign_flip"
            )
        self.source_axis_mode = str(source_axis_mode)
        if target_axis_mode not in {"clean", "zero", "orthogonal", "sign_flip"}:
            raise ValueError(
                "target_axis_mode must be clean, zero, orthogonal, or sign_flip"
            )
        self.target_axis_mode = str(target_axis_mode)
        marker_local = payload.get("target_axis_marker_local3")
        self.target_axis_marker_local3 = None
        if self.target_axis_relation_adapter:
            if marker_local is None:
                raise ValueError(
                    "target-axis checkpoint lacks target_axis_marker_local3"
                )
            self.target_axis_marker_local3 = np.asarray(
                marker_local, dtype=np.float32
            ).reshape(3)
            self.target_axis_marker_local3 /= max(
                float(np.linalg.norm(self.target_axis_marker_local3)), 1e-6
            )
        self.ndf_checkpoint = (
            None
            if ndf_checkpoint in {None, ""}
            else Path(ndf_checkpoint).expanduser().resolve()
        )
        self.ndf_adapter = None
        if self.source_axis_relation_adapter and self.source_axis_mode != "zero":
            if self.ndf_checkpoint is None or not self.ndf_checkpoint.is_file():
                raise FileNotFoundError(
                    f"source-axis policy requires an NDF checkpoint: {self.ndf_checkpoint}"
                )
            self.ndf_adapter = NdfPointwiseAdapter(
                self.ndf_checkpoint,
                device=str(self.device),
                include_vector_direction=True,
            )
        self.functional_frame_token = (
            self.condition in self.FUNCTIONAL_FRAME_CONDITIONS
        )
        self.dual_functional_frame_token = (
            self.condition == "interaction_functional_dual_frame_flow"
        )
        checkpoint_priority = bool(payload.get("target_color_priority", False))
        self.target_color_priority = (
            checkpoint_priority
            if target_color_priority is None
            else bool(target_color_priority)
        )
        if self.target_color_priority and self.point_channels < 6:
            raise ValueError(
                "target color priority requires an XYZRGB checkpoint"
            )
        self.flow_parameterization = str(
            payload.get("flow_parameterization", "free")
        )
        self.predict_rotation = bool(payload.get("predict_rotation", False))
        self.action_frame = str(payload.get("action_frame", "world"))
        if self.action_frame not in {"world", "target"}:
            raise ValueError(
                f"unsupported interaction-flow action frame {self.action_frame!r}"
            )
        self.policy_frame = str(payload.get("policy_frame", "world"))
        if self.policy_frame not in {
            "world",
            "goal_action",
            "goal_state_action",
            "goal_geometry",
            "goal",
        }:
            raise ValueError(
                f"unsupported interaction-flow policy frame {self.policy_frame!r}"
            )
        if self.policy_frame != "world":
            if self.action_frame != "world":
                raise ValueError(
                    "goal policy frame is incompatible with an additional action-frame transform"
                )
            if not self.dense_camera_relative:
                raise ValueError(
                    "goal policy frame requires the deployable dense camera-relative geometry"
                )
        if (
            self.recovery_action_frame != "world"
            or self.recovery_local_frame_token
        ) and (
            self.policy_frame != "world" or self.action_frame != "world"
        ):
            raise ValueError(
                "source-functional recovery requires world policy/action frames"
            )
        self.normalization = Normalization(**payload["normalization"])
        self.model = build_model(
            self.condition,
            horizon=self.horizon,
            flow_steps=self.flow_steps,
            num_anchors=self.num_anchors,
            feature_dim=self.feature_dim,
            layers=self.layers,
            flow_parameterization=self.flow_parameterization,
            xyz_std=np.asarray(self.normalization.xyz_std),
            action_std=np.asarray(self.normalization.action_std),
            predict_rotation=self.predict_rotation,
            point_channels=self.point_channels,
            target_color_adapter=self.target_color_adapter,
            source_geometry_adapter=self.source_geometry_adapter,
            source_axis_relation_adapter=self.source_axis_relation_adapter,
            target_axis_relation_adapter=self.target_axis_relation_adapter,
            hand_object_frame_token=self.hand_object_frame_token,
        ).to(self.device)
        incompatible = self.model.load_state_dict(payload["model"], strict=False)
        allowed_missing = (
            {"xyz_std", "action_std"}
            if self.condition != "point_tokens"
            else set()
        )
        if set(incompatible.missing_keys) - allowed_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "checkpoint/model mismatch: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        self.model.eval()
        self.observation_points = int(observation_points)
        self.last_diagnostic: dict[str, float | None] = {}

    @staticmethod
    def _orthogonal_axis(axis: np.ndarray) -> np.ndarray:
        """Return a deterministic unit vector perpendicular to ``axis``."""
        value = np.asarray(axis, dtype=np.float32).reshape(3)
        norm = float(np.linalg.norm(value))
        if norm <= 1e-6:
            return np.zeros(3, dtype=np.float32)
        value = value / norm
        basis = np.eye(3, dtype=np.float32)[int(np.argmin(np.abs(value)))]
        result = np.cross(value, basis)
        return (result / max(float(np.linalg.norm(result)), 1e-6)).astype(
            np.float32
        )

    def _source_axis(self, points_xyz: np.ndarray) -> tuple[np.ndarray, float]:
        if not self.source_axis_relation_adapter or self.source_axis_mode == "zero":
            return np.zeros(3, dtype=np.float32), 0.0
        if self.ndf_adapter is None:
            raise RuntimeError("source-axis policy has no NDF adapter")
        start = time.perf_counter()
        features = self.ndf_adapter.encode(points_xyz, points_xyz)
        axis = np.asarray(features[:, -3:], dtype=np.float32).mean(axis=0)
        axis /= max(float(np.linalg.norm(axis)), 1e-6)
        if self.source_axis_mode == "orthogonal":
            axis = self._orthogonal_axis(axis)
        elif self.source_axis_mode == "sign_flip":
            axis = -axis
        latency_ms = (time.perf_counter() - start) * 1000.0
        return axis.astype(np.float32), float(latency_ms)

    @staticmethod
    def _farthest_indices(points: np.ndarray, count: int) -> np.ndarray:
        sampled_xyz = deterministic_farthest_points(points[:, :3], count)
        return np.square(
            sampled_xyz[:, None, :] - points[None, :, :3]
        ).sum(axis=-1).argmin(axis=1)

    def _points(
        self, point_cloud: np.ndarray, *, target: bool = False
    ) -> np.ndarray:
        raw = np.asarray(point_cloud, dtype=np.float32)
        keep = np.all(np.isfinite(raw[:, : self.point_channels]), axis=1)
        points = raw[keep, : self.point_channels]
        if len(points) == 0:
            raise ValueError("interaction-flow policy received an empty point cloud")
        priority = np.empty((0,), dtype=np.int64)
        if target and self.target_color_priority:
            color = points[:, 3:6]
            priority = np.flatnonzero(
                color[:, 0] + color[:, 1] - 2.0 * color[:, 2] > 0.65
            )
            if len(priority) > self.observation_points:
                priority = priority[
                    self._farthest_indices(
                        points[priority], self.observation_points
                    )
                ]
        remaining = self.observation_points - len(priority)
        if remaining > 0:
            background = np.setdiff1d(
                np.arange(len(points), dtype=np.int64),
                priority,
                assume_unique=False,
            )
            if len(background) == 0:
                background = priority
            selected_background = background[
                self._farthest_indices(points[background], remaining)
            ]
            indices = np.concatenate((priority, selected_background))
        else:
            indices = priority
        if self.point_channels == 3:
            return points[indices, :3]
        return points[indices]

    @torch.no_grad()
    def predict(
        self,
        *,
        operated_point_cloud: np.ndarray,
        target_point_cloud: np.ndarray,
        eef_state20: np.ndarray,
        oracle_relative_frame9_metric: np.ndarray | None = None,
    ) -> np.ndarray:
        norm = self.normalization
        points_a = self._points(operated_point_cloud)
        points_b = self._points(target_point_cloud, target=True)
        state = np.asarray(eef_state20, dtype=np.float32).reshape(20)
        extra_batch: dict[str, torch.Tensor] = {}
        source_axis = np.zeros(3, dtype=np.float32)
        ndf_latency_ms = 0.0
        if self.source_axis_relation_adapter:
            source_axis, ndf_latency_ms = self._source_axis(points_a[:, :3])
            extra_batch["source_axis3"] = torch.from_numpy(source_axis)[None].to(
                self.device
            )
        frame9 = None
        frame9_metric = None
        goal_frame9_metric = None
        dense_estimate = None
        dense_latency_ms = 0.0
        target_axis = np.zeros(3, dtype=np.float32)
        hand_object_frame = None
        if (
            self.functional_frame_token
            or self.action_frame == "target"
            or self.target_axis_relation_adapter
        ):
            if self.dense_camera_relative:
                if self.dense_frame_provider is None:
                    raise RuntimeError("dense checkpoint has no online frame provider")
                start = time.perf_counter()
                estimate_kwargs = {
                    "current_point_cloud": points_a[:, :3],
                    # Marker fitting needs the untouched XYZRGB observation.
                    "target_point_cloud": np.asarray(
                        target_point_cloud, dtype=np.float32
                    ),
                }
                goal_estimate = None
                if self.policy_frame != "world":
                    # The goal coordinate system is observable from B even for
                    # zero/oracle relation-token interventions.  Keep that basis
                    # fixed while changing only the relative geometry condition.
                    goal_estimate = self.dense_frame_provider.estimate(
                        **estimate_kwargs,
                        mode="clean",
                    )
                    dense_estimate = (
                        goal_estimate
                        if self.dense_frame_mode == "clean"
                        else self.dense_frame_provider.estimate(
                            **estimate_kwargs,
                            mode=self.dense_frame_mode,
                            oracle_relative_frame9_metric=(
                                oracle_relative_frame9_metric
                                if self.dense_frame_mode == "oracle"
                                else None
                            ),
                        )
                    )
                    if goal_estimate.goal_transform is None:
                        raise RuntimeError(
                            "dense clean estimator did not return a goal transform"
                        )
                    goal_transform = goal_estimate.goal_transform
                    goal_frame9_metric = np.concatenate(
                        (
                            goal_transform[:3, 3],
                            goal_transform[:3, 0],
                            goal_transform[:3, 1],
                        )
                    ).astype(np.float32)
                else:
                    dense_estimate = self.dense_frame_provider.estimate(
                        **estimate_kwargs,
                        mode=self.dense_frame_mode,
                        oracle_relative_frame9_metric=(
                            oracle_relative_frame9_metric
                            if self.dense_frame_mode == "oracle"
                            else None
                        ),
                    )
                dense_latency_ms = (time.perf_counter() - start) * 1000.0
                frame9_metric = dense_estimate.frame9_metric.copy()
                if self.policy_frame in {"goal", "goal_geometry"}:
                    frame9_metric = relative_frame9_to_target_frame(
                        frame9_metric, goal_frame9_metric
                    )
                frame9 = frame9_metric.copy()
                frame9[:3] /= np.asarray(norm.xyz_std, dtype=np.float32)
                extra_batch["target_frame9"] = torch.from_numpy(frame9)[None].to(
                    self.device
                )
                extra_batch["target_frame_confidence"] = torch.tensor(
                    [dense_estimate.combined_confidence],
                    dtype=torch.float32,
                    device=self.device,
                )
                if (
                    self.recovery_action_frame != "world"
                    or self.recovery_local_frame_token
                ):
                    source_frame_estimate = dense_estimate
                    if source_frame_estimate.source_rotation is None:
                        source_frame_estimate = goal_estimate
                    if (
                        source_frame_estimate is None
                        or source_frame_estimate.source_rotation is None
                    ):
                        source_frame_estimate = self.dense_frame_provider.estimate(
                            **estimate_kwargs,
                            mode="clean",
                        )
                    source_rotation = source_frame_estimate.source_rotation
                    if source_rotation is None:
                        raise RuntimeError(
                            "source-functional recovery did not estimate an "
                            "NDF current-object rotation"
                        )
                    source_frame6 = np.concatenate(
                        (source_rotation[:, 0], source_rotation[:, 1])
                    ).astype(np.float32)
                    extra_batch["source_frame6_columns"] = torch.from_numpy(
                        source_frame6
                    )[None].to(self.device)
                    if self.recovery_local_frame_token:
                        source_frame9 = np.concatenate(
                            (
                                np.zeros(3, dtype=np.float32),
                                source_frame6,
                            )
                        )
                        source_local_frame9 = relative_frame9_to_target_frame(
                            frame9_metric, source_frame9
                        )
                        source_local_frame9[:3] /= float(
                            np.asarray(norm.xyz_std, dtype=np.float32).mean()
                        )
                        extra_batch["target_frame9_source_local"] = (
                            torch.from_numpy(source_local_frame9)[None].to(
                                self.device
                            )
                        )
                if self.dual_functional_frame_token:
                    if self.dense_frame_mode == "zero":
                        local_frame9 = np.zeros(9, dtype=np.float32)
                    else:
                        if dense_estimate.goal_transform is None:
                            raise RuntimeError(
                                "dual-frame clean/oracle runtime requires an "
                                "observable goal transform"
                            )
                        goal_transform = dense_estimate.goal_transform
                        local_goal_frame9 = np.concatenate(
                            (
                                goal_transform[:3, 3],
                                goal_transform[:3, 0],
                                goal_transform[:3, 1],
                            )
                        ).astype(np.float32)
                        local_frame9 = relative_frame9_to_target_frame(
                            dense_estimate.frame9_metric,
                            local_goal_frame9,
                        )
                        local_frame9[:3] /= np.asarray(
                            norm.xyz_std, dtype=np.float32
                        )
                    extra_batch["target_frame9_local"] = torch.from_numpy(
                        local_frame9
                    )[None].to(self.device)
            else:
                if oracle_relative_frame9_metric is not None:
                    raise ValueError(
                        "oracle frame is only accepted by the explicit dense "
                        "privileged-ceiling branch"
                    )
                target_frame, _ = estimate_geometry_marker_frame(
                    np.asarray(target_point_cloud, dtype=np.float32)
                )
                frame9 = np.concatenate(
                    (
                        (target_frame[:3, 3] - norm.xyz_mean) / norm.xyz_std,
                        target_frame[:3, 0],
                        target_frame[:3, 1],
                    )
                ).astype(np.float32)
                if self.functional_frame_token:
                    extra_batch["target_frame9"] = torch.from_numpy(frame9)[None].to(
                        self.device
                    )
                if self.target_axis_relation_adapter:
                    target_axis = (
                        target_frame[:3, :3] @ self.target_axis_marker_local3
                    ).astype(np.float32)
                    target_axis /= max(float(np.linalg.norm(target_axis)), 1e-6)
                    if self.target_axis_mode == "zero":
                        target_axis.fill(0.0)
                    elif self.target_axis_mode == "sign_flip":
                        target_axis *= -1.0
                    elif self.target_axis_mode == "orthogonal":
                        basis = np.eye(3, dtype=np.float32)[
                            int(np.argmin(np.abs(target_axis)))
                        ]
                        target_axis = np.cross(target_axis, basis)
                        target_axis /= max(float(np.linalg.norm(target_axis)), 1e-6)
                    extra_batch["target_axis3"] = torch.from_numpy(target_axis)[None].to(
                        self.device
                    )
        if self.hand_object_frame_token:
            if dense_estimate is None or dense_estimate.source_rotation is None:
                raise RuntimeError(
                    "hand-object token requires a clean camera/NDF source frame"
                )
            if dense_estimate.goal_transform is None:
                raise RuntimeError(
                    "hand-object token requires the camera marker goal transform"
                )
            current_position = (
                dense_estimate.goal_transform[:3, 3]
                - dense_estimate.frame9_metric[:3]
            )
            hand_object_frame = active_hand_object_frame11(
                current_position,
                dense_estimate.source_rotation,
                state,
                translation_scale_m=self.hand_object_translation_scale_m,
            )
            extra_batch["hand_object_frame11"] = torch.from_numpy(
                hand_object_frame
            )[None].to(self.device)
            extra_batch["hand_object_frame_confidence"] = torch.tensor(
                [dense_estimate.source_confidence],
                dtype=torch.float32,
                device=self.device,
            )
        if self.policy_frame in {"goal", "goal_geometry"}:
            if goal_frame9_metric is None:
                raise RuntimeError("goal policy frame was not estimated")
            points_a = transform_point_features_to_target_frame(
                points_a, goal_frame9_metric
            )
            points_b = transform_point_features_to_target_frame(
                points_b, goal_frame9_metric
            )
        if self.policy_frame == "goal":
            state = transform_eef_state20_to_target_frame(
                state, goal_frame9_metric
            )
        elif self.policy_frame == "goal_state_action":
            if goal_frame9_metric is None:
                raise RuntimeError("goal policy frame was not estimated")
            state = transform_eef_state20_to_target_frame(
                state, goal_frame9_metric
            )
        batch = {
            "points_a": torch.from_numpy(
                normalize_point_features(points_a, norm)
            )[None].to(self.device),
            "points_b": torch.from_numpy(
                normalize_point_features(points_b, norm)
            )[None].to(self.device),
            "state": torch.from_numpy(
                ((state - norm.state_mean) / norm.state_std).astype(np.float32)
            )[None].to(self.device),
            **extra_batch,
        }
        output = self.model(batch)
        normalized_action = output.action.cpu().numpy()[0]
        action = (
            normalized_action * norm.action_std[None]
            + norm.action_mean[None]
        ).astype(np.float32)
        if self.policy_frame in {"goal_action", "goal_state_action", "goal"}:
            action = action14_from_target_frame(
                action, goal_frame9_metric
            ).astype(np.float32)
        elif self.action_frame == "target":
            action = action14_from_target_frame(action, frame9).astype(
                np.float32
            )
        if not np.all(np.isfinite(action)):
            raise RuntimeError("interaction-flow policy returned non-finite actions")

        diagnostic: dict[str, float | None] = {
            "attention_entropy_a": None,
            "attention_entropy_b": None,
            "anchor_pair_distance_cm": None,
            "predicted_endpoint_motion_cm": None,
            "sampled_target_marker_points": None,
            "ndf_axis_x": float(source_axis[0]),
            "ndf_axis_y": float(source_axis[1]),
            "ndf_axis_z": float(source_axis[2]),
            "ndf_axis_norm": float(np.linalg.norm(source_axis)),
            "ndf_latency_ms": float(ndf_latency_ms),
            "target_axis_x": float(target_axis[0]),
            "target_axis_y": float(target_axis[1]),
            "target_axis_z": float(target_axis[2]),
            "target_axis_norm": float(np.linalg.norm(target_axis)),
            "dense_frame_mode": (
                self.dense_frame_mode if self.dense_camera_relative else None
            ),
            "dense_combined_confidence": (
                None
                if dense_estimate is None
                else float(dense_estimate.combined_confidence)
            ),
            "dense_source_confidence": (
                None
                if dense_estimate is None
                else float(dense_estimate.source_confidence)
            ),
            "dense_target_confidence": (
                None
                if dense_estimate is None
                else float(dense_estimate.target_confidence)
            ),
            "dense_source_disagreement_deg": (
                None
                if dense_estimate is None
                else dense_estimate.source_disagreement_deg
            ),
            "dense_target_latched": (
                None
                if dense_estimate is None
                else float(dense_estimate.target_latched)
            ),
            "dense_marker_points": (
                None if dense_estimate is None else dense_estimate.marker_points
            ),
            "dense_marker_fit_score_m2": (
                None
                if dense_estimate is None
                else dense_estimate.marker_fit_score_m2
            ),
            "dense_frame_latency_ms": (
                float(dense_latency_ms) if dense_estimate is not None else None
            ),
            "dense_relative_translation_x_cm": (
                None
                if dense_estimate is None
                else float(dense_estimate.frame9_metric[0] * 100.0)
            ),
            "dense_relative_translation_y_cm": (
                None
                if dense_estimate is None
                else float(dense_estimate.frame9_metric[1] * 100.0)
            ),
            "dense_relative_translation_z_cm": (
                None
                if dense_estimate is None
                else float(dense_estimate.frame9_metric[2] * 100.0)
            ),
            "dense_relative_translation_norm_cm": (
                None
                if dense_estimate is None
                else float(
                    np.linalg.norm(dense_estimate.frame9_metric[:3]) * 100.0
                )
            ),
            "dense_relative_rotation_deg": (
                None
                if dense_estimate is None
                else float(
                    np.rad2deg(
                        np.arccos(
                            np.clip(
                                (
                                    np.trace(
                                        frame9_rotation(
                                            dense_estimate.frame9_metric
                                        )
                                    )
                                    - 1.0
                                )
                                * 0.5,
                                -1.0,
                                1.0,
                            )
                        )
                    )
                )
            ),
            "hand_object_active_left": (
                None if hand_object_frame is None else float(hand_object_frame[-2])
            ),
            "hand_object_active_right": (
                None if hand_object_frame is None else float(hand_object_frame[-1])
            ),
            "hand_object_translation_norm_cm": (
                None
                if hand_object_frame is None
                else float(
                    np.linalg.norm(hand_object_frame[:3])
                    * self.hand_object_translation_scale_m
                    * 100.0
                )
            ),
        }
        if self.target_color_priority:
            color = points_b[:, 3:6]
            diagnostic["sampled_target_marker_points"] = float(
                np.sum(color[:, 0] + color[:, 1] - 2.0 * color[:, 2] > 0.65)
            )
        if output.attention_a is not None:
            diagnostic["attention_entropy_a"] = float(
                attention_entropy(output.attention_a).item()
            )
            diagnostic["attention_entropy_b"] = float(
                attention_entropy(output.attention_b).item()
            )
        if output.anchor_a is not None and output.anchor_a.shape[1] > 1:
            scale = torch.as_tensor(
                norm.xyz_std,
                dtype=output.anchor_a.dtype,
                device=output.anchor_a.device,
            )
            distance = torch.cdist(output.anchor_a * scale, output.anchor_a * scale)
            mask = ~torch.eye(
                distance.shape[-1], dtype=torch.bool, device=distance.device
            )[None]
            diagnostic["anchor_pair_distance_cm"] = float(
                distance[mask.expand_as(distance)].mean().item() * 100.0
            )
        if output.predicted_flow is not None:
            scale = torch.as_tensor(
                norm.xyz_std,
                dtype=output.predicted_flow.dtype,
                device=output.predicted_flow.device,
            )
            endpoint_motion = (
                output.predicted_flow[:, :, -1]
                - output.predicted_flow[:, :, 0]
            ) * scale
            diagnostic["predicted_endpoint_motion_cm"] = float(
                torch.linalg.vector_norm(endpoint_motion, dim=-1).mean().item()
                * 100.0
            )
        self.last_diagnostic = diagnostic
        return action

    def reset(self) -> None:
        """Reset episode-local diagnostics and the early target-frame latch."""

        self.last_diagnostic = {}
        if self.dense_frame_provider is not None:
            self.dense_frame_provider.reset()
