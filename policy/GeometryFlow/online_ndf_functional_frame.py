"""Online camera-only relative functional frames for dense NDF policies.

This module deliberately does not depend on the interaction-policy runtime.
It implements the camera geometry contract used by
``augment_task_flow_with_camera_ndf_frame.py``:

``current A cloud -> NDF SO(3) ensemble``
``early-latched B cloud -> marker goal frame``
``[goal_position-current_position, (R_goal R_current^T)[:, :2]]``

The returned confidence is the product of NDF ensemble agreement and marker
frame reliability.  A privileged oracle mode exists only as an explicitly
enabled evaluation ceiling; it is disabled by default.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np

from .ensemble_ndf_functional_frames import (
    maximum_pairwise_disagreement,
    project_rotation_mean,
    rotation_medoid,
    stabilize_rotation_against_previous,
)
from .ndf_adapter import NdfFunctionalFrameAdapter
from .target_frame import estimate_geometry_marker_frame, yellow_marker_mask


class FunctionalFrameEncoder(Protocol):
    def encode_frame(self, support_points_world: np.ndarray) -> np.ndarray: ...


MarkerFrameEstimator = Callable[[np.ndarray], tuple[np.ndarray, dict]]


@dataclass(frozen=True)
class OnlineFunctionalFrameEstimate:
    """One policy-ready relative frame and observable diagnostics.

    ``frame9_metric`` stores translation in metres.  The policy runtime must
    divide its first three values by the checkpoint ``xyz_std`` exactly as the
    training ``InteractionFlowDataset`` does for ``dataset_relative`` frames.
    """

    frame9_metric: np.ndarray
    combined_confidence: float
    source_confidence: float
    target_confidence: float
    source_disagreement_deg: float | None
    source_rotation: np.ndarray | None
    goal_transform: np.ndarray | None
    mode: str
    target_latched: bool
    marker_points: int | None
    marker_fit_score_m2: float | None
    source_temporal_ambiguous: bool = False
    source_temporal_corrected: bool = False
    source_temporal_jump_deg: float | None = None

    def policy_frame9(self, xyz_std: np.ndarray | Sequence[float]) -> np.ndarray:
        """Return the tensor value consumed by a trained policy.

        Rotation columns stay dimensionless.  Only metric relative translation
        is normalized, matching ``InteractionFlowDataset`` in
        ``functional_frame_translation_mode='delta'``.
        """

        scale = _as_vector(xyz_std, "xyz_std")
        if np.any(scale <= 0.0):
            raise ValueError("xyz_std must be strictly positive")
        result = self.frame9_metric.copy()
        result[:3] /= scale
        return result


def _as_vector(value: np.ndarray | Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32).reshape(-1)
    if result.shape != (3,):
        raise ValueError(f"{name} must contain three values, got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _as_rotation(value: np.ndarray | Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.size != 9:
        raise ValueError(f"{name} must contain nine values, got {result.shape}")
    result = result.reshape(3, 3)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(result.T @ result, np.eye(3), atol=2e-3):
        raise ValueError(f"{name} is not approximately orthonormal")
    if float(np.linalg.det(result)) <= 0.0:
        raise ValueError(f"{name} must be a right-handed rotation")
    return result


def _valid_xyz(point_cloud: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(point_cloud, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] < 3:
        raise ValueError(f"{name} must have shape [N,>=3], got {value.shape}")
    xyz = value[:, :3]
    xyz = xyz[np.all(np.isfinite(xyz), axis=1)]
    if len(xyz) < 3:
        raise ValueError(f"{name} has fewer than three finite XYZ points")
    return xyz


def relative_frame9_metric(
    current_position: np.ndarray,
    current_rotation: np.ndarray,
    goal_position: np.ndarray,
    goal_rotation: np.ndarray,
) -> np.ndarray:
    """Build the exact dataset-relative ``[delta-p, R6D columns]`` token."""

    current_position = _as_vector(current_position, "current_position")
    goal_position = _as_vector(goal_position, "goal_position")
    current_rotation = _as_rotation(current_rotation, "current_rotation")
    goal_rotation = _as_rotation(goal_rotation, "goal_rotation")
    relative_rotation = goal_rotation @ current_rotation.T
    return np.concatenate(
        (
            goal_position - current_position,
            relative_rotation[:, 0],
            relative_rotation[:, 1],
        )
    ).astype(np.float32)


class OnlineNdfFunctionalFrameProvider:
    """Produce confidence-gated relative frames from live A/B camera clouds."""

    MODES = {"clean", "inverse", "zero", "oracle"}

    def __init__(
        self,
        *,
        current_origin_offset_local3: np.ndarray | Sequence[float],
        goal_position_offset_marker_local3: np.ndarray | Sequence[float],
        goal_rotation_offset_marker_local9: np.ndarray | Sequence[float],
        confidence_threshold_deg: float,
        aggregation: str = "projected_mean",
        ndf_checkpoints: Sequence[str | Path] | None = None,
        frame_encoders: Sequence[FunctionalFrameEncoder] | None = None,
        device: str = "cpu",
        marker_frame_estimator: MarkerFrameEstimator = estimate_geometry_marker_frame,
        marker_min_points: int = 20,
        marker_max_fit_score_m2: float = 2.0e-5,
        allow_privileged_oracle: bool = False,
        temporal_stabilization: bool = False,
        temporal_jump_trigger_deg: float = 120.0,
        temporal_alternative_accept_deg: float = 45.0,
    ) -> None:
        if (ndf_checkpoints is None) == (frame_encoders is None):
            raise ValueError(
                "provide exactly one of ndf_checkpoints or frame_encoders"
            )
        encoders = (
            [
                NdfFunctionalFrameAdapter(path, device=device)
                for path in ndf_checkpoints or ()
            ]
            if frame_encoders is None
            else list(frame_encoders)
        )
        if len(encoders) < 2:
            raise ValueError(
                "online epistemic confidence requires at least two NDF models"
            )
        threshold = float(confidence_threshold_deg)
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError("confidence_threshold_deg must be finite and nonnegative")
        if int(marker_min_points) < 1:
            raise ValueError("marker_min_points must be positive")
        if float(marker_max_fit_score_m2) < 0.0:
            raise ValueError("marker_max_fit_score_m2 must be nonnegative")
        if aggregation not in {"projected_mean", "medoid"}:
            raise ValueError("aggregation must be projected_mean or medoid")
        self.encoders = encoders
        self.current_origin_offset_local3 = _as_vector(
            current_origin_offset_local3, "current_origin_offset_local3"
        )
        self.goal_position_offset_marker_local3 = _as_vector(
            goal_position_offset_marker_local3,
            "goal_position_offset_marker_local3",
        )
        self.goal_rotation_offset_marker_local9 = _as_rotation(
            goal_rotation_offset_marker_local9,
            "goal_rotation_offset_marker_local9",
        )
        self.confidence_threshold_deg = threshold
        self.aggregation = str(aggregation)
        self.marker_frame_estimator = marker_frame_estimator
        self.marker_min_points = int(marker_min_points)
        self.marker_max_fit_score_m2 = float(marker_max_fit_score_m2)
        self.allow_privileged_oracle = bool(allow_privileged_oracle)
        self.temporal_stabilization = bool(temporal_stabilization)
        self.temporal_jump_trigger_deg = float(temporal_jump_trigger_deg)
        self.temporal_alternative_accept_deg = float(
            temporal_alternative_accept_deg
        )
        if self.temporal_stabilization:
            stabilize_rotation_against_previous(
                np.eye(3, dtype=np.float32),
                np.eye(3, dtype=np.float32),
                jump_trigger_deg=self.temporal_jump_trigger_deg,
                alternative_accept_deg=self.temporal_alternative_accept_deg,
            )
        self._cached_target_cloud: np.ndarray | None = None
        self._cached_marker_frame: np.ndarray | None = None
        self._cached_marker_diagnostic: dict | None = None
        self._cached_target_confidence: float | None = None
        self._previous_source_rotation: np.ndarray | None = None

    @classmethod
    def from_metadata(
        cls,
        *,
        ndf_checkpoints: Sequence[str | Path],
        camera_metadata: str | Path,
        ensemble_metadata: str | Path,
        device: str = "cpu",
        allow_privileged_oracle: bool = False,
    ) -> "OnlineNdfFunctionalFrameProvider":
        camera_path = Path(camera_metadata).expanduser()
        ensemble_path = Path(ensemble_metadata).expanduser()
        camera = json.loads(camera_path.read_text(encoding="utf-8"))
        ensemble = json.loads(ensemble_path.read_text(encoding="utf-8"))
        required = (
            "current_origin_offset_local3",
            "goal_position_offset_marker_local3",
            "goal_rotation_offset_marker_local9",
        )
        missing = [key for key in required if key not in camera]
        if missing:
            raise KeyError(
                f"camera metadata {camera_path} lacks calibration: {missing}"
            )
        if "confidence_threshold_deg" not in ensemble:
            raise KeyError(
                f"ensemble metadata {ensemble_path} lacks confidence_threshold_deg"
            )
        camera_fold = camera.get("fold")
        ensemble_fold = ensemble.get("fold")
        if (
            camera_fold is not None
            and ensemble_fold is not None
            and int(camera_fold) != int(ensemble_fold)
        ):
            raise ValueError(
                f"camera/ensemble fold mismatch: {camera_fold} != {ensemble_fold}"
            )
        if camera.get("deployable") is False:
            raise ValueError(f"camera metadata {camera_path} is marked non-deployable")
        encoding = str(camera.get("functional_frame_encoding", ""))
        if encoding and encoding != "camera_ndf_current_marker_goal_se3_columns_v1":
            raise ValueError(
                f"unsupported online functional-frame encoding {encoding!r}"
            )
        declared_member_counts = []
        if ensemble.get("selection_top_k") is not None:
            declared_member_counts.append(
                ("selection_top_k", int(ensemble["selection_top_k"]))
            )
        for key in (
            "frame_predictions",
            "full_frame_predictions",
            "selected_seeds",
            "selected_result_jsons",
        ):
            value = ensemble.get(key)
            if isinstance(value, list):
                declared_member_counts.append((key, len(value)))
        distinct_counts = {count for _key, count in declared_member_counts}
        if len(distinct_counts) > 1:
            raise ValueError(
                f"ensemble metadata {ensemble_path} has inconsistent member "
                f"counts: {dict(declared_member_counts)}"
            )
        checkpoint_count = len(ndf_checkpoints)
        if distinct_counts and checkpoint_count != next(iter(distinct_counts)):
            raise ValueError(
                f"ensemble metadata expects {next(iter(distinct_counts))} NDF "
                f"checkpoints, received {checkpoint_count}"
            )
        return cls(
            ndf_checkpoints=ndf_checkpoints,
            current_origin_offset_local3=camera["current_origin_offset_local3"],
            goal_position_offset_marker_local3=camera[
                "goal_position_offset_marker_local3"
            ],
            goal_rotation_offset_marker_local9=camera[
                "goal_rotation_offset_marker_local9"
            ],
            confidence_threshold_deg=ensemble["confidence_threshold_deg"],
            aggregation=str(ensemble.get("aggregation", "projected_mean")),
            device=device,
            allow_privileged_oracle=allow_privileged_oracle,
            temporal_stabilization=bool(
                ensemble.get("temporal_stabilization", False)
            ),
            temporal_jump_trigger_deg=float(
                ensemble.get("temporal_jump_trigger_deg", 120.0)
            ),
            temporal_alternative_accept_deg=float(
                ensemble.get("temporal_alternative_accept_deg", 45.0)
            ),
        )

    @property
    def target_latched(self) -> bool:
        return self._cached_marker_frame is not None

    @property
    def cached_target_cloud(self) -> np.ndarray | None:
        if self._cached_target_cloud is None:
            return None
        return self._cached_target_cloud.copy()

    def reset(self) -> None:
        self._cached_target_cloud = None
        self._cached_marker_frame = None
        self._cached_marker_diagnostic = None
        self._cached_target_confidence = None
        self._previous_source_rotation = None

    def latch_target(self, target_point_cloud: np.ndarray) -> None:
        """Latch one early B observation (or a caller-concatenated early stack)."""

        self._latch_target(target_point_cloud)

    def _latch_target(self, target_point_cloud: np.ndarray) -> None:
        if self.target_latched:
            return
        target = np.asarray(target_point_cloud, dtype=np.float32)
        _valid_xyz(target, "target_point_cloud")
        self._cached_target_cloud = target.copy()
        try:
            frame, diagnostic = self.marker_frame_estimator(target)
            frame = np.asarray(frame, dtype=np.float32).reshape(4, 4)
            diagnostic = dict(diagnostic)
            marker_points = int(diagnostic.get("marker_points", 0))
            fit_score = diagnostic.get("fit_score_m2")
            fit_score_value = None if fit_score is None else float(fit_score)
            reliable = bool(
                marker_points >= self.marker_min_points
                and fit_score_value is not None
                and np.isfinite(fit_score_value)
                and fit_score_value <= self.marker_max_fit_score_m2
            )
        except (ValueError, np.linalg.LinAlgError):
            # The feature is masked by confidence=0.  Retaining a finite frame
            # lets the policy execute its exact no-geometry path safely.
            xyz = _valid_xyz(target, "target_point_cloud")
            frame = np.eye(4, dtype=np.float32)
            frame[:3, 3] = xyz.mean(axis=0)
            diagnostic = {
                "marker_points": int(
                    yellow_marker_mask(target).sum()
                    if target.shape[1] >= 6
                    else 0
                ),
                "fit_score_m2": None,
                "estimation_failed": True,
            }
            reliable = False
        if not np.all(np.isfinite(frame)):
            raise ValueError("marker estimator returned a non-finite frame")
        _as_rotation(frame[:3, :3], "marker_frame_rotation")
        self._cached_marker_frame = frame
        self._cached_marker_diagnostic = diagnostic
        self._cached_target_confidence = float(reliable)

    def _clean_estimate(
        self,
        current_point_cloud: np.ndarray,
        target_point_cloud: np.ndarray,
    ) -> OnlineFunctionalFrameEstimate:
        current_xyz = _valid_xyz(current_point_cloud, "current_point_cloud")
        self._latch_target(target_point_cloud)
        rotations = np.stack(
            [encoder.encode_frame(current_xyz) for encoder in self.encoders], axis=0
        ).astype(np.float32)
        if rotations.shape != (len(self.encoders), 3, 3):
            raise ValueError(
                "NDF encoders must each return one [3,3] frame, got "
                f"{rotations.shape}"
            )
        source_rotation = (
            project_rotation_mean(rotations[:, None])[0]
            if self.aggregation == "projected_mean"
            else rotation_medoid(rotations[:, None])[0]
        )
        disagreement = float(
            maximum_pairwise_disagreement(rotations[:, None])[0]
        )
        temporal_ambiguous = False
        temporal_corrected = False
        temporal_jump = None
        if self.temporal_stabilization:
            source_rotation, temporal_ambiguous, temporal_corrected, temporal_jump = (
                stabilize_rotation_against_previous(
                    source_rotation,
                    self._previous_source_rotation,
                    jump_trigger_deg=self.temporal_jump_trigger_deg,
                    alternative_accept_deg=self.temporal_alternative_accept_deg,
                )
            )
        epistemic_reliable = bool(disagreement <= self.confidence_threshold_deg)
        source_confidence = float(epistemic_reliable and not temporal_ambiguous)
        if self.temporal_stabilization and (
            self._previous_source_rotation is None
            or source_confidence > 0.0
            or temporal_corrected
        ):
            self._previous_source_rotation = source_rotation.copy()
        target_confidence = float(self._cached_target_confidence)
        marker_frame = np.asarray(self._cached_marker_frame, dtype=np.float32)
        current_position = (
            current_xyz.mean(axis=0)
            + source_rotation @ self.current_origin_offset_local3
        )
        goal_rotation = (
            marker_frame[:3, :3] @ self.goal_rotation_offset_marker_local9
        )
        goal_position = (
            marker_frame[:3, 3]
            + marker_frame[:3, :3]
            @ self.goal_position_offset_marker_local3
        )
        frame9 = relative_frame9_metric(
            current_position,
            source_rotation,
            goal_position,
            goal_rotation,
        )
        diagnostic = self._cached_marker_diagnostic or {}
        fit_score = diagnostic.get("fit_score_m2")
        return OnlineFunctionalFrameEstimate(
            frame9_metric=frame9,
            combined_confidence=float(source_confidence * target_confidence),
            source_confidence=source_confidence,
            target_confidence=target_confidence,
            source_disagreement_deg=disagreement,
            source_rotation=source_rotation.copy(),
            goal_transform=np.block(
                [
                    [goal_rotation, goal_position[:, None]],
                    [np.zeros((1, 3), dtype=np.float32), np.ones((1, 1), dtype=np.float32)],
                ]
            ).astype(np.float32),
            mode="clean",
            target_latched=True,
            marker_points=(
                None
                if "marker_points" not in diagnostic
                else int(diagnostic["marker_points"])
            ),
            marker_fit_score_m2=(
                None if fit_score is None else float(fit_score)
            ),
            source_temporal_ambiguous=bool(temporal_ambiguous),
            source_temporal_corrected=bool(temporal_corrected),
            source_temporal_jump_deg=(
                None if temporal_jump is None else float(temporal_jump)
            ),
        )

    def estimate(
        self,
        *,
        current_point_cloud: np.ndarray,
        target_point_cloud: np.ndarray,
        mode: str = "clean",
        oracle_relative_frame9_metric: np.ndarray | None = None,
    ) -> OnlineFunctionalFrameEstimate:
        """Estimate one frame; privileged oracle use must be explicitly enabled."""

        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {sorted(self.MODES)}, got {mode!r}")
        if mode in {"clean", "inverse"}:
            if oracle_relative_frame9_metric is not None:
                raise ValueError(f"{mode} mode does not accept an oracle frame")
            clean = self._clean_estimate(current_point_cloud, target_point_cloud)
            if mode == "clean":
                return clean
            relative_rotation = np.column_stack(
                (
                    clean.frame9_metric[3:6],
                    clean.frame9_metric[6:9],
                    np.cross(
                        clean.frame9_metric[3:6], clean.frame9_metric[6:9]
                    ),
                )
            )
            inverse_rotation = relative_rotation.T
            inverse_frame = np.concatenate(
                (
                    -clean.frame9_metric[:3],
                    inverse_rotation[:, 0],
                    inverse_rotation[:, 1],
                )
            ).astype(np.float32)
            return replace(clean, frame9_metric=inverse_frame, mode="inverse")
        if mode == "zero":
            if oracle_relative_frame9_metric is not None:
                raise ValueError("zero mode does not accept an oracle frame")
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
        if not self.allow_privileged_oracle:
            raise PermissionError(
                "oracle mode is a privileged evaluation ceiling and is disabled"
            )
        if oracle_relative_frame9_metric is None:
            raise ValueError("oracle mode requires oracle_relative_frame9_metric")
        frame = np.asarray(oracle_relative_frame9_metric, dtype=np.float32).reshape(9)
        if not np.all(np.isfinite(frame)):
            raise ValueError("oracle frame contains non-finite values")
        # Validate the R6D columns by reconstructing their orthonormal frame.
        first = frame[3:6]
        second = frame[6:9]
        first = first / max(float(np.linalg.norm(first)), 1e-8)
        second = second - first * float(np.dot(first, second))
        second = second / max(float(np.linalg.norm(second)), 1e-8)
        _as_rotation(
            np.column_stack((first, second, np.cross(first, second))),
            "oracle_relative_rotation",
        )
        return OnlineFunctionalFrameEstimate(
            frame9_metric=frame.copy(),
            combined_confidence=1.0,
            source_confidence=1.0,
            target_confidence=1.0,
            source_disagreement_deg=0.0,
            source_rotation=None,
            goal_transform=None,
            mode="oracle",
            target_latched=self.target_latched,
            marker_points=None,
            marker_fit_score_m2=None,
        )
