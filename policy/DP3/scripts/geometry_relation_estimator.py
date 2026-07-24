#!/usr/bin/env python3
"""Observation-derived object relation estimator.

The estimator API accepts only the two observed point clouds.  Simulator
poses remain outside this module and are used later solely to convert the
predicted actor-local goal relation into the DP3 correction token.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ndf_goal_regressor import (
    FrozenNdfFeatureConfig,
    FrozenNdfGeometryEncoder,
    GoalRelationRegressor,
    target_to_goal_matrix,
)


OBSERVATION_RELATION_ROUTES = ("ndf_observation_goal",)


@dataclass(frozen=True)
class GeometryRelationPrediction:
    goal_t_a_from_b: np.ndarray
    solver_energy: float = 0.0
    confidence: float = 1.0
    flip_probability: float = 0.0
    diagnostics: dict | None = None

    def __post_init__(self) -> None:
        transform = np.asarray(self.goal_t_a_from_b, dtype=np.float64)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError("goal_t_a_from_b must be a finite 4x4 matrix")
        if not np.isfinite(float(self.solver_energy)):
            raise ValueError("solver_energy must be finite")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be within [0, 1]")
        if not 0.0 <= float(self.flip_probability) <= 1.0:
            raise ValueError("flip_probability must be within [0, 1]")


class GeometryRelationEstimator(ABC):
    """ID-free relation estimator interface."""

    @abstractmethod
    def estimate_goal(
        self,
        object_pointcloud_a: np.ndarray,
        object_pointcloud_b: np.ndarray,
    ) -> GeometryRelationPrediction:
        raise NotImplementedError


def _resolve_path(value: str, *, relative_to: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


class NdfGoalRegressionEstimator(GeometryRelationEstimator):
    """Frozen NDF invariant features followed by a learned SE(3) head."""

    def __init__(
        self,
        *,
        regressor_checkpoint: str,
        ndf_checkpoint: str,
        device: str = "cuda:0",
    ) -> None:
        import torch

        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            requested = torch.device("cpu")
        self.device = requested

        try:
            checkpoint = torch.load(
                regressor_checkpoint, map_location=self.device, weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(regressor_checkpoint, map_location=self.device)
        if not isinstance(checkpoint, dict):
            raise ValueError("geometry regressor checkpoint must be a dictionary")
        feature_config = FrozenNdfFeatureConfig(
            latent_dim=int(checkpoint.get("latent_dim", 256)),
            target_num_points=int(checkpoint.get("target_num_points", 1024)),
        )
        self.encoder = FrozenNdfGeometryEncoder(
            checkpoint=ndf_checkpoint,
            device=self.device,
            config=feature_config,
        )
        hidden_dims = tuple(int(value) for value in checkpoint.get("hidden_dims", (256, 128)))
        dropout = float(checkpoint.get("dropout", 0.05))
        self.regressor = GoalRelationRegressor(
            input_dim=int(checkpoint.get("input_dim", feature_config.output_dim)),
            hidden_dims=hidden_dims,
            dropout=dropout,
        ).to(self.device)
        state = checkpoint.get("model_state_dict")
        if not isinstance(state, dict):
            raise ValueError("checkpoint is missing model_state_dict")
        self.regressor.load_state_dict(state, strict=True)
        self.regressor.eval()
        self.feature_mean = torch.as_tensor(
            checkpoint.get("feature_mean"), dtype=torch.float32, device=self.device
        ).reshape(-1)
        self.feature_std = torch.as_tensor(
            checkpoint.get("feature_std"), dtype=torch.float32, device=self.device
        ).reshape(-1)
        if (
            self.feature_mean.numel() != feature_config.output_dim
            or self.feature_std.numel() != feature_config.output_dim
        ):
            raise ValueError("checkpoint feature normalization has the wrong width")
        self.feature_std = self.feature_std.clamp_min(1e-5)
        self.validation_metrics = dict(checkpoint.get("validation_metrics", {}))

    def estimate_goal(
        self,
        object_pointcloud_a: np.ndarray,
        object_pointcloud_b: np.ndarray,
    ) -> GeometryRelationPrediction:
        import torch

        feature = self.encoder.encode(object_pointcloud_a, object_pointcloud_b)
        feature = (feature - self.feature_mean) / self.feature_std
        with torch.no_grad():
            output = self.regressor(feature.unsqueeze(0))[0].detach().cpu().numpy()
        goal = target_to_goal_matrix(output)
        diagnostics = {
            "feature_source": "ndf_vector_norm_plus_invariant_geometry",
            "regressor_validation": self.validation_metrics,
        }
        return GeometryRelationPrediction(
            goal_t_a_from_b=goal,
            # This regression route has no iterative solver.  Keep energy zero
            # and confidence one so the DP3 token is not silently attenuated.
            solver_energy=0.0,
            confidence=1.0,
            flip_probability=0.0,
            diagnostics=diagnostics,
        )


def load_estimator_spec(path: str | Path) -> tuple[dict, Path]:
    spec_path = Path(path).expanduser().resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise ValueError("geometry estimator spec must contain a JSON object")
    return spec, spec_path


def create_estimator_from_spec(
    path: str | Path,
    *,
    device_override: str | None = None,
) -> GeometryRelationEstimator:
    spec, spec_path = load_estimator_spec(path)
    estimator_type = str(spec.get("type", ""))
    if estimator_type != "ndf_goal_regressor":
        raise ValueError(
            "unsupported geometry estimator type "
            f"{estimator_type!r}; expected 'ndf_goal_regressor'"
        )
    regressor_value = spec.get("regressor_checkpoint")
    ndf_value = spec.get("ndf_checkpoint")
    if not regressor_value or not ndf_value:
        raise ValueError(
            "NDF goal-regressor spec requires regressor_checkpoint and ndf_checkpoint"
        )
    return NdfGoalRegressionEstimator(
        regressor_checkpoint=str(
            _resolve_path(str(regressor_value), relative_to=spec_path.parent)
        ),
        ndf_checkpoint=str(_resolve_path(str(ndf_value), relative_to=spec_path.parent)),
        device=str(device_override or spec.get("device", "cuda:0")),
    )


__all__ = (
    "GeometryRelationEstimator",
    "GeometryRelationPrediction",
    "NdfGoalRegressionEstimator",
    "OBSERVATION_RELATION_ROUTES",
    "create_estimator_from_spec",
    "load_estimator_spec",
)
