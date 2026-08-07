"""Online camera-only rotation tokens for the hanging-mug policy.

This is the deployable counterpart of
``augment_hanging_mug_with_camera_rotation.py``.  It estimates the current mug
frame with a frozen equivariant-frame ensemble, registers the observed rack to
one frozen training reference, and returns
``[0, 0, 0, (R_goal R_current.T)[:, 0:2]]``.  Simulator poses are never read.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from .augment_hanging_mug_with_camera_rotation import (
    PROPER_AXIS_SIGNS,
    trimmed_icp,
)
from .ensemble_ndf_functional_frames import (
    maximum_pairwise_disagreement,
    project_rotation_mean,
)
from .ndf_adapter import NdfFunctionalFrameAdapter
from .online_ndf_functional_frame import OnlineFunctionalFrameEstimate


def _valid_xyz(points: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(points, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] < 3:
        raise ValueError(f"{name} must have shape [N,>=3], got {value.shape}")
    value = value[np.all(np.isfinite(value[:, :3]), axis=1), :3]
    if len(value) < 4:
        raise ValueError(f"{name} must contain at least four finite XYZ points")
    return value


def _rotation(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(3, 3)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(result.T @ result, np.eye(3), atol=2e-3):
        raise ValueError(f"{name} is not orthonormal")
    if not np.isclose(np.linalg.det(result), 1.0, atol=2e-3):
        raise ValueError(f"{name} is not a proper rotation")
    return result.astype(np.float32)


class OnlineHangingMugCameraRotationProvider:
    """Produce a zero-translation relative rotation token from live A/B clouds."""

    MODES = {"clean", "inverse", "zero"}

    def __init__(
        self,
        *,
        reference_target_cloud: np.ndarray,
        reference_goal_rotation: np.ndarray,
        source_confidence_threshold_deg: float,
        registration_confidence_threshold_m: float,
        frame_checkpoints: Sequence[str | Path] | None = None,
        frame_encoders: Sequence[object] | None = None,
        device: str = "cpu",
        registration_points: int = 512,
        icp_iterations: int = 30,
        icp_trim_fraction: float = 0.8,
        temporal_maximum_step_deg: float = 30.0,
        temporal_minimum_margin_deg: float = 30.0,
    ) -> None:
        if (frame_checkpoints is None) == (frame_encoders is None):
            raise ValueError(
                "provide exactly one of frame_checkpoints or frame_encoders"
            )
        self.encoders = (
            [
                NdfFunctionalFrameAdapter(path, device=device)
                for path in frame_checkpoints or ()
            ]
            if frame_encoders is None
            else list(frame_encoders)
        )
        if len(self.encoders) < 2:
            raise ValueError("camera rotation confidence requires at least two models")
        self.reference_target_cloud = _valid_xyz(
            reference_target_cloud, "reference_target_cloud"
        )
        self.reference_goal_rotation = _rotation(
            reference_goal_rotation, "reference_goal_rotation"
        )
        self.source_confidence_threshold_deg = float(
            source_confidence_threshold_deg
        )
        self.registration_confidence_threshold_m = float(
            registration_confidence_threshold_m
        )
        self.registration_points = int(registration_points)
        self.icp_iterations = int(icp_iterations)
        self.icp_trim_fraction = float(icp_trim_fraction)
        self.temporal_maximum_step_deg = float(temporal_maximum_step_deg)
        self.temporal_minimum_margin_deg = float(temporal_minimum_margin_deg)
        if self.source_confidence_threshold_deg < 0.0:
            raise ValueError("source confidence threshold must be nonnegative")
        if self.registration_confidence_threshold_m < 0.0:
            raise ValueError("registration confidence threshold must be nonnegative")
        if self.registration_points < 4 or self.icp_iterations < 1:
            raise ValueError("registration_points and icp_iterations are invalid")
        if not 0.5 <= self.icp_trim_fraction <= 1.0:
            raise ValueError("icp_trim_fraction must lie in [0.5,1]")
        if self.temporal_maximum_step_deg <= 0.0:
            raise ValueError("temporal maximum step must be positive")
        if self.temporal_minimum_margin_deg < 0.0:
            raise ValueError("temporal minimum margin must be nonnegative")
        self._goal_rotation: np.ndarray | None = None
        self._target_cloud: np.ndarray | None = None
        self._registration_cost_m: float | None = None
        self._target_confidence: float | None = None
        self._previous_source_rotation: np.ndarray | None = None

    @classmethod
    def from_artifacts(
        cls,
        *,
        frame_checkpoints: Sequence[str | Path],
        calibration: str | Path,
        ensemble_metadata: str | Path,
        device: str = "cpu",
    ) -> "OnlineHangingMugCameraRotationProvider":
        calibration_path = Path(calibration).expanduser()
        ensemble_path = Path(ensemble_metadata).expanduser()
        with np.load(calibration_path, allow_pickle=False) as archive:
            values = {key: np.asarray(archive[key]) for key in archive.files}
        ensemble = json.loads(ensemble_path.read_text(encoding="utf-8"))
        required = {
            "reference_target_cloud",
            "reference_goal_rotation",
            "registration_confidence_threshold_m",
            "registration_points",
            "icp_iterations",
            "icp_trim_fraction",
            "temporal_maximum_step_deg",
            "temporal_minimum_margin_deg",
        }
        missing = sorted(required - set(values))
        if missing:
            raise KeyError(f"camera calibration {calibration_path} lacks {missing}")
        if "confidence_threshold_deg" not in ensemble:
            raise KeyError(
                f"ensemble metadata {ensemble_path} lacks confidence_threshold_deg"
            )
        selected = ensemble.get("selected_seeds")
        if isinstance(selected, list) and len(frame_checkpoints) != len(selected):
            raise ValueError(
                f"ensemble expects {len(selected)} checkpoints, got {len(frame_checkpoints)}"
            )
        return cls(
            frame_checkpoints=frame_checkpoints,
            reference_target_cloud=values["reference_target_cloud"],
            reference_goal_rotation=values["reference_goal_rotation"],
            source_confidence_threshold_deg=float(
                ensemble["confidence_threshold_deg"]
            ),
            registration_confidence_threshold_m=float(
                values["registration_confidence_threshold_m"]
            ),
            registration_points=int(values["registration_points"]),
            icp_iterations=int(values["icp_iterations"]),
            icp_trim_fraction=float(values["icp_trim_fraction"]),
            temporal_maximum_step_deg=float(
                values["temporal_maximum_step_deg"]
            ),
            temporal_minimum_margin_deg=float(
                values["temporal_minimum_margin_deg"]
            ),
            device=device,
        )

    @property
    def target_latched(self) -> bool:
        return self._goal_rotation is not None

    @property
    def cached_target_cloud(self) -> np.ndarray | None:
        return None if self._target_cloud is None else self._target_cloud.copy()

    def reset(self) -> None:
        self._goal_rotation = None
        self._target_cloud = None
        self._registration_cost_m = None
        self._target_confidence = None
        self._previous_source_rotation = None

    @staticmethod
    def _sample(points: np.ndarray, count: int) -> np.ndarray:
        if len(points) <= int(count):
            return points.copy()
        indices = np.linspace(0, len(points) - 1, int(count), dtype=np.int64)
        return points[indices]

    def latch_target(self, target_point_cloud: np.ndarray) -> None:
        if self.target_latched:
            return
        target = _valid_xyz(target_point_cloud, "target_point_cloud")
        reference = self._sample(
            self.reference_target_cloud, self.registration_points
        )
        sampled_target = self._sample(target, self.registration_points)
        rack_rotation, _translation, cost = trimmed_icp(
            reference,
            sampled_target,
            iterations=self.icp_iterations,
            trim_fraction=self.icp_trim_fraction,
        )
        self._target_cloud = target.copy()
        self._goal_rotation = _rotation(
            rack_rotation @ self.reference_goal_rotation,
            "registered_goal_rotation",
        )
        self._registration_cost_m = float(cost)
        self._target_confidence = float(
            cost <= self.registration_confidence_threshold_m
        )

    def _stabilize(self, raw_rotation: np.ndarray) -> tuple[np.ndarray, float]:
        raw = _rotation(raw_rotation, "raw_source_rotation")
        if self._previous_source_rotation is None:
            selected = raw
            temporal_confidence = 0.0
        else:
            candidates = raw[None] @ PROPER_AXIS_SIGNS
            distance = np.rad2deg(
                Rotation.from_matrix(
                    candidates @ self._previous_source_rotation.T
                ).magnitude()
            )
            order = np.argsort(distance)
            selected = candidates[int(order[0])]
            margin = float(distance[int(order[1])] - distance[int(order[0])])
            temporal_confidence = float(
                distance[int(order[0])] <= self.temporal_maximum_step_deg
                and margin >= self.temporal_minimum_margin_deg
            )
        self._previous_source_rotation = np.asarray(
            selected, dtype=np.float32
        ).copy()
        return self._previous_source_rotation.copy(), temporal_confidence

    def _clean_estimate(
        self,
        current_point_cloud: np.ndarray,
        target_point_cloud: np.ndarray,
    ) -> OnlineFunctionalFrameEstimate:
        current = _valid_xyz(current_point_cloud, "current_point_cloud")
        self.latch_target(target_point_cloud)
        frames = np.stack(
            [encoder.encode_frame(current) for encoder in self.encoders], axis=0
        ).astype(np.float32)
        if frames.shape != (len(self.encoders), 3, 3):
            raise ValueError(f"frame encoders returned invalid shape {frames.shape}")
        raw_rotation = project_rotation_mean(frames[:, None])[0]
        disagreement = float(
            maximum_pairwise_disagreement(frames[:, None])[0]
        )
        source_rotation, temporal_confidence = self._stabilize(raw_rotation)
        ensemble_confidence = float(
            disagreement <= self.source_confidence_threshold_deg
        )
        source_confidence = max(ensemble_confidence, temporal_confidence)
        target_confidence = float(self._target_confidence)
        goal_rotation = np.asarray(self._goal_rotation, dtype=np.float32)
        relative_rotation = goal_rotation @ source_rotation.T
        frame9 = np.concatenate(
            (
                np.zeros(3, dtype=np.float32),
                relative_rotation[:, 0],
                relative_rotation[:, 1],
            )
        ).astype(np.float32)
        goal_transform = np.eye(4, dtype=np.float32)
        goal_transform[:3, :3] = goal_rotation
        return OnlineFunctionalFrameEstimate(
            frame9_metric=frame9,
            combined_confidence=float(source_confidence * target_confidence),
            source_confidence=float(source_confidence),
            target_confidence=float(target_confidence),
            source_disagreement_deg=disagreement,
            source_rotation=source_rotation,
            goal_transform=goal_transform,
            mode="clean",
            target_latched=True,
            marker_points=int(len(self._target_cloud)),
            marker_fit_score_m2=float(self._registration_cost_m),
        )

    def estimate(
        self,
        *,
        current_point_cloud: np.ndarray,
        target_point_cloud: np.ndarray,
        mode: str = "clean",
        oracle_relative_frame9_metric: np.ndarray | None = None,
    ) -> OnlineFunctionalFrameEstimate:
        if oracle_relative_frame9_metric is not None:
            raise ValueError("camera rotation provider does not consume oracle poses")
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {sorted(self.MODES)}, got {mode!r}")
        if mode == "zero":
            return OnlineFunctionalFrameEstimate(
                frame9_metric=np.zeros(9, dtype=np.float32),
                combined_confidence=0.0,
                source_confidence=0.0,
                target_confidence=0.0,
                source_disagreement_deg=None,
                source_rotation=None,
                goal_transform=None,
                mode="zero",
                target_latched=self.target_latched,
                marker_points=None,
                marker_fit_score_m2=None,
            )
        clean = self._clean_estimate(current_point_cloud, target_point_cloud)
        if mode == "clean":
            return clean
        rotation = np.column_stack(
            (
                clean.frame9_metric[3:6],
                clean.frame9_metric[6:9],
                np.cross(clean.frame9_metric[3:6], clean.frame9_metric[6:9]),
            )
        )
        inverse = rotation.T
        frame9 = np.concatenate(
            (np.zeros(3, dtype=np.float32), inverse[:, 0], inverse[:, 1])
        ).astype(np.float32)
        return replace(clean, frame9_metric=frame9, mode="inverse")
