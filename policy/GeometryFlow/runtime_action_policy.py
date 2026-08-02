"""Runtime wrapper for the promoted flow-conditioned Cartesian action policy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .build_grasp_dataset import deterministic_farthest_points, valid_xyz
from .train_task_flow_benchmark import (
    Normalization,
    build_action_predictor,
    geometric_flow_progress,
    remaining_flow_from_current,
    rigid_flow_se3_tokens,
)


class FlowActionRuntime:
    """Load a benchmark checkpoint and predict a six-step 14D EEF-delta chunk.

    Flow construction is intentionally outside this class: callers may use PCA,
    UTONIA, NDF, or a fused estimator as long as it returns the same rigid
    ``[anchors, task_steps, XYZ]`` interface.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "cpu",
        observation_points: int = 64,
    ) -> None:
        self.device = torch.device(device)
        payload = torch.load(Path(checkpoint), map_location=self.device, weights_only=False)
        if payload["condition"] not in {"flow", "zero_flow"}:
            raise ValueError("FlowActionRuntime requires a flow-conditioned checkpoint")
        self.architecture = str(payload.get("architecture", "mlp"))
        self.explicit_flow_progress = bool(payload.get("explicit_flow_progress", False))
        self.model = build_action_predictor(
            payload["condition"],
            int(payload["flow_steps"]),
            int(payload["horizon"]),
            self.architecture,
            self.explicit_flow_progress,
        ).to(self.device)
        self.model.load_state_dict(payload["model"])
        self.model.eval()
        self.normalization = Normalization(**payload["normalization"])
        self.observation_points = int(observation_points)
        self.flow_steps = int(payload["flow_steps"])
        self.horizon = int(payload["horizon"])
        self.flow_context_mode = str(payload.get("flow_context_mode", "full"))
        self.action_frame = str(payload.get("action_frame", "world"))
        if self.action_frame not in {"world", "eef"}:
            raise ValueError(f"unsupported checkpoint action frame {self.action_frame!r}")

    def _points(self, point_cloud: np.ndarray) -> np.ndarray:
        xyz = valid_xyz(point_cloud)
        return deterministic_farthest_points(xyz, self.observation_points)

    @torch.no_grad()
    def predict(
        self,
        *,
        operated_point_cloud: np.ndarray,
        target_point_cloud: np.ndarray,
        eef_state20: np.ndarray,
        current_anchors: np.ndarray,
        predicted_episode_flow: np.ndarray,
    ) -> np.ndarray:
        norm = self.normalization
        points_a = self._points(operated_point_cloud)
        points_b = self._points(target_point_cloud)
        state = np.asarray(eef_state20, dtype=np.float32).reshape(20)
        anchors = np.asarray(current_anchors, dtype=np.float32)
        flow = np.asarray(predicted_episode_flow, dtype=np.float32)
        if flow.ndim != 3 or flow.shape[1:] != (self.flow_steps, 3):
            raise ValueError(
                f"predicted flow must be [anchors,{self.flow_steps},3], got {flow.shape}"
            )
        if anchors.shape != (flow.shape[0], 3):
            raise ValueError(
                f"current anchors {anchors.shape} do not match flow {flow.shape}"
            )
        if self.flow_context_mode == "remaining":
            flow = remaining_flow_from_current(flow, anchors)
        progress = geometric_flow_progress(flow, anchors)
        flow_token = np.concatenate(
            (
                (anchors - norm.xyz_mean) / norm.xyz_std,
                ((flow - anchors[:, None, :]) / norm.xyz_std).reshape(len(anchors), -1),
            ),
            axis=-1,
        ).astype(np.float32)
        batch = {
            "points_a": torch.from_numpy(
                ((points_a - norm.xyz_mean) / norm.xyz_std).astype(np.float32)
            )[None].to(self.device),
            "points_b": torch.from_numpy(
                ((points_b - norm.xyz_mean) / norm.xyz_std).astype(np.float32)
            )[None].to(self.device),
            "state": torch.from_numpy(
                ((state - norm.state_mean) / norm.state_std).astype(np.float32)
            )[None].to(self.device),
            "flow": torch.from_numpy(flow_token)[None].to(self.device),
            "flow_se3": torch.from_numpy(
                rigid_flow_se3_tokens(flow, anchors, norm.xyz_std)
            )[None].to(self.device),
            "flow_progress": torch.tensor(
                [[progress]], dtype=torch.float32, device=self.device
            ),
        }
        normalized_action = self.model(batch).cpu().numpy()[0]
        action = normalized_action * norm.action_std[None] + norm.action_mean[None]
        if not np.all(np.isfinite(action)):
            raise RuntimeError("flow-conditioned action policy returned non-finite actions")
        return action.astype(np.float32)
