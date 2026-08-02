#!/usr/bin/env python3
"""Fast object-disjoint inverse-dynamics test for oracle task flow.

This is deliberately smaller than DP3.  It is a screening experiment: if an
oracle flow cannot improve held-out-object action prediction here, investing
in a learned flow generator is unlikely to be worthwhile yet.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .eef_frame_action import payload_with_eef_frame_actions


FOLDS = (
    {"train": (2, 3, 4, 7, 8, 9), "validation": (1, 6), "test": (0, 5)},
    {"train": (0, 3, 4, 5, 8, 9), "validation": (2, 7), "test": (1, 6)},
    {"train": (0, 1, 4, 5, 6, 9), "validation": (3, 8), "test": (2, 7)},
    {"train": (0, 1, 2, 5, 6, 7), "validation": (4, 9), "test": (3, 8)},
    {"train": (1, 2, 3, 6, 7, 8), "validation": (0, 5), "test": (4, 9)},
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class Normalization:
    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray
    xyz_mean: np.ndarray
    xyz_std: np.ndarray
    point_feature_mean: np.ndarray | None = None
    point_feature_std: np.ndarray | None = None


def safe_std(value: np.ndarray, axis) -> np.ndarray:
    result = np.asarray(value.std(axis=axis), dtype=np.float32)
    return np.maximum(result, 1e-4)


def fit_normalization(payload: dict[str, np.ndarray], indices: np.ndarray) -> Normalization:
    state = payload["state"][indices]
    action = payload["action"][indices].reshape(-1, payload["action"].shape[-1])
    xyz = np.concatenate(
        (
            payload["points_a"][indices, :, :3].reshape(-1, 3),
            payload["points_b"][indices, :, :3].reshape(-1, 3),
        ),
        axis=0,
    )
    feature_mean = None
    feature_std = None
    if payload["points_a"].shape[-1] > 3:
        feature = np.concatenate(
            (
                payload["points_a"][indices, :, 3:].reshape(
                    -1, payload["points_a"].shape[-1] - 3
                ),
                payload["points_b"][indices, :, 3:].reshape(
                    -1, payload["points_b"].shape[-1] - 3
                ),
            ),
            axis=0,
        )
        feature_mean = feature.mean(axis=0).astype(np.float32)
        feature_std = safe_std(feature, axis=0)
    return Normalization(
        state_mean=state.mean(axis=0).astype(np.float32),
        state_std=safe_std(state, axis=0),
        action_mean=action.mean(axis=0).astype(np.float32),
        action_std=safe_std(action, axis=0),
        xyz_mean=xyz.mean(axis=0).astype(np.float32),
        xyz_std=safe_std(xyz, axis=0),
        point_feature_mean=feature_mean,
        point_feature_std=feature_std,
    )


def rigid_flow_se3_tokens(
    flow: np.ndarray, current_anchors: np.ndarray, xyz_std: np.ndarray
) -> np.ndarray:
    """Factor point flow into camera-computable centroid motion and SO(3).

    Each token contains normalized centroid displacement followed by the first
    two rotation-matrix columns.  This is derived only from the supplied flow
    and current anchors; no simulator object pose is used.
    """
    target = np.asarray(flow, dtype=np.float64)
    source = np.asarray(current_anchors, dtype=np.float64)
    if target.ndim != 3 or target.shape[0] != len(source) or target.shape[-1] != 3:
        raise ValueError(
            f"flow/current anchors must be [anchors,steps,3]/[anchors,3], got "
            f"{target.shape}/{source.shape}"
        )
    source_center = source.mean(axis=0)
    source_zero = source - source_center
    tokens = []
    for step in range(target.shape[1]):
        target_center = target[:, step].mean(axis=0)
        target_zero = target[:, step] - target_center
        left, _, right = np.linalg.svd(source_zero.T @ target_zero)
        rotation = right.T @ left.T
        if np.linalg.det(rotation) < 0.0:
            right[-1] *= -1.0
            rotation = right.T @ left.T
        tokens.append(
            np.concatenate(
                (
                    (target_center - source_center) / np.asarray(xyz_std),
                    rotation[:, :2].reshape(6),
                )
            )
        )
    return np.asarray(tokens, dtype=np.float32)


def normalize_point_features(
    points: np.ndarray, normalization: Normalization
) -> np.ndarray:
    value = np.asarray(points, dtype=np.float32)
    xyz = (value[..., :3] - normalization.xyz_mean) / normalization.xyz_std
    if value.shape[-1] == 3:
        return xyz.astype(np.float32)
    if (
        normalization.point_feature_mean is None
        or normalization.point_feature_std is None
    ):
        raise ValueError("point-feature normalization is missing for colored points")
    feature = (
        value[..., 3:] - normalization.point_feature_mean
    ) / normalization.point_feature_std
    return np.concatenate((xyz, feature), axis=-1).astype(np.float32)


class TaskFlowDataset(Dataset):
    def __init__(
        self,
        payload: dict[str, np.ndarray],
        indices: np.ndarray,
        normalization: Normalization,
        flow_noise_translation_std_m: float = 0.0,
        flow_noise_rotation_std_deg: float = 0.0,
        flow_context_mode: str = "full",
    ) -> None:
        self.payload = payload
        self.indices = np.asarray(indices, dtype=np.int64)
        self.norm = normalization
        self.flow_noise_translation_std_m = float(flow_noise_translation_std_m)
        self.flow_noise_rotation_std_deg = float(flow_noise_rotation_std_deg)
        if flow_context_mode not in {"full", "remaining"}:
            raise ValueError(f"unknown flow context mode {flow_context_mode!r}")
        self.flow_context_mode = str(flow_context_mode)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = int(self.indices[item])
        episode = int(self.payload["episode_index"][index])
        current_anchors = self.payload["current_anchors"][index]
        flow = self.payload["episode_flow"][episode]
        if self.flow_noise_translation_std_m > 0.0 or self.flow_noise_rotation_std_deg > 0.0:
            flow = noisy_rigid_flow(
                flow,
                translation_std_m=self.flow_noise_translation_std_m,
                rotation_std_deg=self.flow_noise_rotation_std_deg,
            )
        progress = geometric_flow_progress(flow, current_anchors)
        if self.flow_context_mode == "remaining":
            flow = remaining_flow_from_current(flow, current_anchors)
        flow_track = np.concatenate(
            (
                (current_anchors - self.norm.xyz_mean) / self.norm.xyz_std,
                ((flow - current_anchors[:, None, :]) / self.norm.xyz_std).reshape(
                    len(current_anchors), -1
                ),
            ),
            axis=-1,
        ).astype(np.float32)
        pose_trajectory = self.payload["episode_pose"][episode]
        current_pose = self.payload["current_object_pose9"][index]
        pose_condition = np.concatenate(
            (
                (pose_trajectory[:, :3] - current_pose[None, :3]) / self.norm.xyz_std,
                pose_trajectory[:, 3:],
            ),
            axis=-1,
        ).astype(np.float32)
        result = {
            "points_a": torch.from_numpy(
                normalize_point_features(self.payload["points_a"][index], self.norm)
            ),
            "points_b": torch.from_numpy(
                normalize_point_features(self.payload["points_b"][index], self.norm)
            ),
            "state": torch.from_numpy(
                ((self.payload["state"][index] - self.norm.state_mean) / self.norm.state_std).astype(
                    np.float32
                )
            ),
            "action": torch.from_numpy(
                ((self.payload["action"][index] - self.norm.action_mean) / self.norm.action_std).astype(
                    np.float32
                )
            ),
            "flow": torch.from_numpy(flow_track),
            "flow_se3": torch.from_numpy(
                rigid_flow_se3_tokens(flow, current_anchors, self.norm.xyz_std)
            ),
            "flow_progress": torch.tensor([progress], dtype=torch.float32),
            "pose": torch.from_numpy(pose_condition),
            "index": torch.tensor(index, dtype=torch.long),
            "active_arm_right": torch.tensor(
                self.payload["active_arm_right"][index], dtype=torch.float32
            ),
            "relation_phase": torch.tensor(
                self.payload["relation_phase"][index], dtype=torch.float32
            ),
            "is_recovery_sample": torch.tensor(
                self.payload.get(
                    "is_recovery_sample",
                    self.payload.get(
                        "is_recovery_augmented",
                        np.zeros(len(self.payload["state"]), dtype=np.float32),
                    ),
                )[index],
                dtype=torch.float32,
            ),
        }
        return result


def noisy_rigid_flow(
    flow: np.ndarray,
    *,
    translation_std_m: float,
    rotation_std_deg: float,
    generator: np.random.Generator | None = None,
) -> np.ndarray:
    """Add a smooth, rigidity-preserving endpoint error while fixing step zero."""
    value = np.asarray(flow, dtype=np.float64)
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(f"flow must be [anchors, steps, 3], got {value.shape}")
    normal = generator.normal if generator is not None else np.random.normal
    translation = normal(size=3) * float(translation_std_m)
    rotation_vector = normal(size=3)
    norm = np.linalg.norm(rotation_vector)
    if norm > 1e-12:
        rotation_vector *= (
            np.deg2rad(float(rotation_std_deg)) * float(normal()) / norm
        )
    else:
        rotation_vector[:] = 0.0
    fractions = np.linspace(0.0, 1.0, value.shape[1])
    output = np.empty_like(value)
    for step, fraction in enumerate(fractions):
        points = value[:, step]
        center = points.mean(axis=0)
        rotation = Rotation.from_rotvec(rotation_vector * fraction).as_matrix()
        output[:, step] = (
            (points - center) @ rotation.T + center + translation * fraction
        )
    output[:, 0] = value[:, 0]
    return output.astype(np.float32)


def _rigid_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_zero = source - source.mean(axis=0)
    target_zero = target - target.mean(axis=0)
    left, _, right = np.linalg.svd(source_zero.T @ target_zero)
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right[-1] *= -1.0
        rotation = right.T @ left.T
    return rotation


def remaining_flow_from_current(flow: np.ndarray, current_anchors: np.ndarray) -> np.ndarray:
    """Re-express the future suffix of a rigid flow from the current anchors."""
    value = np.asarray(flow, dtype=np.float64)
    current = np.asarray(current_anchors, dtype=np.float64)
    if value.ndim != 3 or value.shape[0] != len(current) or value.shape[-1] != 3:
        raise ValueError(
            f"flow/current anchor mismatch: flow={value.shape}, current={current.shape}"
        )
    distance = np.linalg.norm(value - current[:, None, :], axis=-1).mean(axis=0)
    start = int(np.argmin(distance))
    query = np.linspace(float(start), float(value.shape[1] - 1), value.shape[1])
    lower = np.floor(query).astype(np.int64)
    upper = np.minimum(lower + 1, value.shape[1] - 1)
    weight = query - lower
    future = (
        value[:, lower] * (1.0 - weight)[None, :, None]
        + value[:, upper] * weight[None, :, None]
    )
    reference = future[:, 0]
    reference_center = reference.mean(axis=0)
    current_center = current.mean(axis=0)
    current_local = current - current_center
    output = []
    for step in range(future.shape[1]):
        rotation = _rigid_rotation(reference, future[:, step])
        center_delta = future[:, step].mean(axis=0) - reference_center
        output.append(current_local @ rotation.T + current_center + center_delta)
    result = np.stack(output, axis=1)
    result[:, 0] = current
    return result.astype(np.float32)


def geometric_flow_progress(flow: np.ndarray, current_anchors: np.ndarray) -> float:
    """Estimate normalized task phase by nearest rigid-flow anchor set."""
    value = np.asarray(flow, dtype=np.float64)
    current = np.asarray(current_anchors, dtype=np.float64)
    if value.ndim != 3 or value.shape[0] != len(current) or value.shape[-1] != 3:
        raise ValueError(
            f"flow/current anchor mismatch: flow={value.shape}, current={current.shape}"
        )
    distance = np.linalg.norm(value - current[:, None, :], axis=-1).mean(axis=0)
    index = int(np.argmin(distance))
    return float(index / max(value.shape[1] - 1, 1))


def load_predicted_episode_flow(path: Path, episode_count: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        indices = archive["episode_indices"].astype(np.int64)
        predicted = archive["predicted_flow"].astype(np.float32)
    if set(indices.tolist()) != set(range(int(episode_count))):
        raise ValueError("test flow prediction must contain every episode exactly once")
    result = np.empty_like(predicted)
    result[indices] = predicted
    return result


class PointSetEncoder(nn.Module):
    def __init__(self, output_dim: int = 128) -> None:
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 128),
            nn.SiLU(),
        )
        self.output = nn.Sequential(nn.Linear(256, output_dim), nn.LayerNorm(output_dim), nn.SiLU())

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        features = self.point_mlp(points)
        pooled = torch.cat((features.mean(dim=1), features.max(dim=1).values), dim=-1)
        return self.output(pooled)


class FlowEncoder(nn.Module):
    def __init__(self, track_dim: int, output_dim: int = 128) -> None:
        super().__init__()
        self.track_mlp = nn.Sequential(
            nn.Linear(track_dim, 192),
            nn.LayerNorm(192),
            nn.SiLU(),
            nn.Linear(192, 128),
            nn.SiLU(),
        )
        self.output = nn.Sequential(nn.Linear(256, output_dim), nn.LayerNorm(output_dim), nn.SiLU())

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        features = self.track_mlp(flow)
        pooled = torch.cat((features.mean(dim=1), features.max(dim=1).values), dim=-1)
        return self.output(pooled)


class ActionPredictor(nn.Module):
    def __init__(
        self,
        condition: str,
        flow_steps: int,
        horizon: int,
        explicit_flow_progress: bool = False,
    ) -> None:
        super().__init__()
        if condition not in {"base", "pose", "flow", "zero_flow"}:
            raise ValueError(f"unknown condition {condition}")
        self.condition = condition
        self.explicit_flow_progress = bool(explicit_flow_progress)
        self.point_encoder = PointSetEncoder(128)
        self.state_encoder = nn.Sequential(
            nn.Linear(20 + int(self.explicit_flow_progress), 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
        )
        if condition in {"flow", "zero_flow"}:
            self.condition_encoder = FlowEncoder(3 + flow_steps * 3, 128)
        elif condition == "pose":
            self.condition_encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flow_steps * 9, 256),
                nn.LayerNorm(256),
                nn.SiLU(),
                nn.Linear(256, 128),
                nn.SiLU(),
            )
        else:
            self.condition_encoder = None
        self.head = nn.Sequential(
            nn.Linear(128 * 4, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, horizon * 14),
        )
        self.horizon = int(horizon)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        a = self.point_encoder(batch["points_a"])
        b = self.point_encoder(batch["points_b"])
        state_input = batch["state"]
        if self.explicit_flow_progress:
            progress = batch["flow_progress"]
            if self.condition == "zero_flow":
                progress = torch.zeros_like(progress)
            state_input = torch.cat((state_input, progress), dim=-1)
        state = self.state_encoder(state_input)
        if self.condition_encoder is None:
            condition = torch.zeros_like(state)
        else:
            condition_input = batch["pose"] if self.condition == "pose" else batch["flow"]
            if self.condition == "zero_flow":
                condition_input = torch.zeros_like(condition_input)
            condition = self.condition_encoder(condition_input)
        return self.head(torch.cat((a, b, state, condition), dim=-1)).reshape(-1, self.horizon, 14)


class FlowTokenTransformer(nn.Module):
    """Decode Cartesian action chunks while preserving object/flow tokens."""

    def __init__(
        self,
        condition: str,
        flow_steps: int,
        horizon: int,
        hidden_dim: int = 128,
        explicit_flow_progress: bool = False,
    ) -> None:
        super().__init__()
        if condition not in {"flow", "zero_flow"}:
            raise ValueError("FlowTokenTransformer supports flow and zero_flow controls")
        self.condition = str(condition)
        self.horizon = int(horizon)
        self.explicit_flow_progress = bool(explicit_flow_progress)
        self.point_token = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.flow_token = nn.Sequential(
            nn.Linear(3 + int(flow_steps) * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.state_token = nn.Sequential(
            nn.Linear(20 + int(self.explicit_flow_progress), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.type_embedding = nn.Parameter(torch.empty(4, hidden_dim))
        nn.init.normal_(self.type_embedding, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=3, norm=nn.LayerNorm(hidden_dim)
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=2, norm=nn.LayerNorm(hidden_dim)
        )
        self.action_query = nn.Parameter(torch.empty(self.horizon, hidden_dim))
        nn.init.normal_(self.action_query, std=0.02)
        self.action_head = nn.Linear(hidden_dim, 14)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        state_input = batch["state"]
        if self.explicit_flow_progress:
            progress = batch["flow_progress"]
            if self.condition == "zero_flow":
                progress = torch.zeros_like(progress)
            state_input = torch.cat((state_input, progress), dim=-1)
        state = self.state_token(state_input)[:, None] + self.type_embedding[0]
        points_a = self.point_token(batch["points_a"]) + self.type_embedding[1]
        points_b = self.point_token(batch["points_b"]) + self.type_embedding[2]
        flow_input = batch["flow"]
        if self.condition == "zero_flow":
            flow_input = torch.zeros_like(flow_input)
        flow = self.flow_token(flow_input) + self.type_embedding[3]
        memory = self.encoder(torch.cat((state, points_a, points_b, flow), dim=1))
        query = self.action_query[None].expand(len(memory), -1, -1)
        return self.action_head(self.decoder(query, memory))


class FlowBottleneckTransformer(nn.Module):
    """Predict actions through a strict object-flow bottleneck.

    Unlike ``FlowTokenTransformer``, this model has no raw point-cloud action
    path.  Visual geometry must first be expressed as tracked rigid-flow tokens;
    proprioception is retained because converting object motion into an EEF
    command depends on the current robot configuration and grasping arm.
    ``zero_flow`` has the identical parameterization and is the matched
    geometry-blind control.
    """

    def __init__(
        self,
        condition: str,
        flow_steps: int,
        horizon: int,
        hidden_dim: int = 128,
        explicit_flow_progress: bool = False,
        include_se3_tokens: bool = False,
    ) -> None:
        super().__init__()
        if condition not in {"flow", "zero_flow"}:
            raise ValueError(
                "FlowBottleneckTransformer supports flow and zero_flow controls"
            )
        self.condition = str(condition)
        self.horizon = int(horizon)
        self.explicit_flow_progress = bool(explicit_flow_progress)
        self.include_se3_tokens = bool(include_se3_tokens)
        self.state_token = nn.Sequential(
            nn.Linear(20 + int(self.explicit_flow_progress), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.flow_token = nn.Sequential(
            nn.Linear(3 + int(flow_steps) * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.se3_token = (
            nn.Sequential(
                nn.Linear(9, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            if self.include_se3_tokens
            else None
        )
        self.type_embedding = nn.Parameter(
            torch.empty(3 if self.include_se3_tokens else 2, hidden_dim)
        )
        nn.init.normal_(self.type_embedding, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=3, norm=nn.LayerNorm(hidden_dim)
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=2, norm=nn.LayerNorm(hidden_dim)
        )
        self.action_query = nn.Parameter(torch.empty(self.horizon, hidden_dim))
        nn.init.normal_(self.action_query, std=0.02)
        self.action_head = nn.Linear(hidden_dim, 14)

    def encode_memory(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode proprioception and flow without exposing raw point clouds."""
        state_input = batch["state"]
        flow_input = batch["flow"]
        if self.condition == "zero_flow":
            flow_input = torch.zeros_like(flow_input)
        if self.explicit_flow_progress:
            progress = batch["flow_progress"]
            if self.condition == "zero_flow":
                progress = torch.zeros_like(progress)
            state_input = torch.cat((state_input, progress), dim=-1)
        state = self.state_token(state_input)[:, None] + self.type_embedding[0]
        flow = self.flow_token(flow_input) + self.type_embedding[1]
        memory_tokens = [state, flow]
        if self.se3_token is not None:
            se3_input = batch["flow_se3"]
            if self.condition == "zero_flow":
                se3_input = torch.zeros_like(se3_input)
            memory_tokens.append(self.se3_token(se3_input) + self.type_embedding[2])
        return self.encoder(torch.cat(memory_tokens, dim=1))

    def decode_action_tokens(
        self, memory: torch.Tensor
    ) -> torch.Tensor:
        query = self.action_query[None].expand(len(memory), -1, -1)
        return self.decoder(query, memory)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        memory = self.encode_memory(batch)
        return self.action_head(self.decode_action_tokens(memory))


def build_action_predictor(
    condition: str,
    flow_steps: int,
    horizon: int,
    architecture: str = "mlp",
    explicit_flow_progress: bool = False,
) -> nn.Module:
    if architecture == "mlp":
        return ActionPredictor(
            condition, flow_steps, horizon, explicit_flow_progress
        )
    if architecture == "flow_transformer":
        return FlowTokenTransformer(
            condition,
            flow_steps,
            horizon,
            explicit_flow_progress=explicit_flow_progress,
        )
    if architecture == "flow_bottleneck":
        return FlowBottleneckTransformer(
            condition,
            flow_steps,
            horizon,
            explicit_flow_progress=explicit_flow_progress,
        )
    if architecture == "flow_se3_bottleneck":
        return FlowBottleneckTransformer(
            condition,
            flow_steps,
            horizon,
            explicit_flow_progress=explicit_flow_progress,
            include_se3_tokens=True,
        )
    raise ValueError(f"unknown action predictor architecture {architecture!r}")


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def validation_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        prediction = model(batch)
        loss = (prediction - batch["action"]).square().mean(dim=(1, 2))
        total += float(loss.sum().item())
        count += int(len(loss))
    return total / max(count, 1)


def weighted_action_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    active_arm_right: torch.Tensor,
    relation_phase: torch.Tensor,
    *,
    active_arm_weight: float,
    relation_weight: float,
    rotation_channel_weight: float = 1.0,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    square = torch.square(prediction - target)
    batch = square.shape[0]
    channel_weight = torch.ones((batch, 14), dtype=square.dtype, device=square.device)
    active_right = active_arm_right >= 0.5
    channel_weight[:, :7] = torch.where(
        active_right[:, None],
        torch.ones_like(channel_weight[:, :7]),
        torch.full_like(channel_weight[:, :7], float(active_arm_weight)),
    )
    channel_weight[:, 7:] = torch.where(
        active_right[:, None],
        torch.full_like(channel_weight[:, 7:], float(active_arm_weight)),
        torch.ones_like(channel_weight[:, 7:]),
    )
    channel_weight[:, 3:6] *= float(rotation_channel_weight)
    channel_weight[:, 10:13] *= float(rotation_channel_weight)
    per_sample = (square * channel_weight[:, None, :]).sum(dim=(1, 2))
    per_sample = per_sample / (square.shape[1] * channel_weight.sum(dim=1))
    phase_weight = torch.where(
        relation_phase >= 0.5,
        torch.full_like(relation_phase, float(relation_weight)),
        torch.ones_like(relation_phase),
    )
    if sample_weight is not None:
        phase_weight = phase_weight * sample_weight.reshape(-1).to(
            dtype=phase_weight.dtype, device=phase_weight.device
        )
    return (per_sample * phase_weight).sum() / phase_weight.sum().clamp_min(1e-8)


def active_arm_routing_loss(
    prediction: torch.Tensor, active_arm_right: torch.Tensor
) -> torch.Tensor:
    """Supervise which arm owns the predicted Cartesian motion chunk."""
    left_motion = torch.linalg.vector_norm(prediction[:, :, :3], dim=-1).sum(dim=1)
    right_motion = torch.linalg.vector_norm(prediction[:, :, 7:10], dim=-1).sum(dim=1)
    logits = torch.stack((left_motion, right_motion), dim=-1)
    return nn.functional.cross_entropy(
        logits, (active_arm_right >= 0.5).to(dtype=torch.long)
    )


def geodesic_rotvec_error_deg(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    shape = prediction.shape[:-1]
    pred_rotation = Rotation.from_rotvec(prediction.reshape(-1, 3))
    target_rotation = Rotation.from_rotvec(target.reshape(-1, 3))
    error = pred_rotation * target_rotation.inv()
    return np.rad2deg(error.magnitude()).reshape(shape)


def integrate_active_endpoint(
    actions: np.ndarray,
    initial_pose: np.ndarray,
    active_arm_right: np.ndarray,
    action_frame_eef: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    batch, horizon = actions.shape[:2]
    positions = np.empty((batch, horizon, 3), dtype=np.float64)
    rotations = np.empty((batch, horizon, 4), dtype=np.float64)
    for sample in range(batch):
        arm = 1 if active_arm_right[sample] >= 0.5 else 0
        offset = 7 if arm else 0
        position = initial_pose[sample, arm, :3].astype(np.float64).copy()
        rotation = Rotation.from_quat(
            [
                initial_pose[sample, arm, 4],
                initial_pose[sample, arm, 5],
                initial_pose[sample, arm, 6],
                initial_pose[sample, arm, 3],
            ]
        )
        for step in range(horizon):
            delta_translation = actions[sample, step, offset : offset + 3]
            delta_rotation = Rotation.from_rotvec(
                actions[sample, step, offset + 3 : offset + 6]
            )
            if action_frame_eef:
                position += rotation.apply(delta_translation)
                rotation = rotation * delta_rotation
            else:
                position += delta_translation
                rotation = delta_rotation * rotation
            positions[sample, step] = position
            rotations[sample, step] = rotation.as_quat()
    return positions, rotations


def summarize_errors(
    payload: dict[str, np.ndarray], indices: np.ndarray, prediction: np.ndarray
) -> dict[str, float]:
    target = payload["action"][indices]
    active_right = payload["active_arm_right"][indices]
    offsets = np.where(active_right >= 0.5, 7, 0).astype(np.int64)
    expanded_offsets = np.broadcast_to(offsets[:, None], (len(indices), target.shape[1]))
    translation_indices = expanded_offsets[..., None] + np.arange(3)
    rotation_indices = expanded_offsets[..., None] + 3 + np.arange(3)
    pred_translation = np.take_along_axis(prediction, translation_indices, axis=2)
    target_translation = np.take_along_axis(target, translation_indices, axis=2)
    pred_rotation = np.take_along_axis(prediction, rotation_indices, axis=2)
    target_rotation = np.take_along_axis(target, rotation_indices, axis=2)
    translation_error = np.linalg.norm(pred_translation - target_translation, axis=-1)
    rotation_error = geodesic_rotvec_error_deg(pred_rotation, target_rotation)

    initial_pose = payload["current_eef_pose7"][indices]
    action_frame_eef = bool(
        int(np.asarray(payload.get("action_frame_eef", 0)).reshape(-1)[0])
    )
    pred_position, pred_quaternion = integrate_active_endpoint(
        prediction,
        initial_pose,
        active_right,
        action_frame_eef=action_frame_eef,
    )
    future = payload["future_eef_pose7"][indices]
    target_active_pose = np.stack(
        [future[sample, :, int(active_right[sample] >= 0.5)] for sample in range(len(indices))]
    )
    endpoint_translation = np.linalg.norm(pred_position - target_active_pose[..., :3], axis=-1)
    target_quaternion_xyzw = target_active_pose[..., [4, 5, 6, 3]]
    endpoint_rotation = (
        Rotation.from_quat(pred_quaternion.reshape(-1, 4))
        * Rotation.from_quat(target_quaternion_xyzw.reshape(-1, 4)).inv()
    ).magnitude().reshape(len(indices), target.shape[1])

    grip_correct = []
    for sample, offset in enumerate(offsets):
        grip_correct.append(
            (prediction[sample, :, offset + 6] >= 0.5)
            == (target[sample, :, offset + 6] >= 0.5)
        )
    grip_correct_value = np.asarray(grip_correct)
    left_motion = np.linalg.norm(prediction[:, :, :3], axis=-1).sum(axis=1)
    right_motion = np.linalg.norm(prediction[:, :, 7:10], axis=-1).sum(axis=1)
    selected_right = right_motion > left_motion
    return {
        "samples": int(len(indices)),
        "active_delta_translation_l2_mm": float(translation_error.mean() * 1000.0),
        "active_delta_rotation_geodesic_deg": float(rotation_error.mean()),
        "active_gripper_accuracy": float(grip_correct_value.mean()),
        "active_arm_motion_selection_accuracy": float(
            np.mean(selected_right == (active_right >= 0.5))
        ),
        "active_endpoint_translation_l2_cm": float(endpoint_translation[:, -1].mean() * 100.0),
        "active_endpoint_rotation_geodesic_deg": float(
            np.rad2deg(endpoint_rotation[:, -1]).mean()
        ),
    }


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    normalization: Normalization,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_indices = []
    all_predictions = []
    for batch in loader:
        batch = move_batch(batch, device)
        normalized = model(batch).cpu().numpy()
        value = normalized * normalization.action_std[None, None, :] + normalization.action_mean[
            None, None, :
        ]
        all_predictions.append(value)
        all_indices.append(batch["index"].cpu().numpy())
    return np.concatenate(all_indices), np.concatenate(all_predictions)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--conditions", nargs="+", default=["base", "pose", "flow"])
    result.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--epochs", type=int, default=120)
    result.add_argument("--patience", type=int, default=18)
    result.add_argument("--batch-size", type=int, default=256)
    result.add_argument("--learning-rate", type=float, default=1e-3)
    result.add_argument("--weight-decay", type=float, default=1e-5)
    result.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    result.add_argument("--num-workers", type=int, default=2)
    result.add_argument("--test-flow-prediction", type=Path)
    result.add_argument("--flow-noise-translation-std-cm", type=float, default=0.0)
    result.add_argument("--flow-noise-rotation-std-deg", type=float, default=0.0)
    result.add_argument("--active-arm-loss-weight", type=float, default=1.0)
    result.add_argument(
        "--arm-routing-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary loss weight for selecting the demonstrated moving arm.",
    )
    result.add_argument("--relation-loss-weight", type=float, default=1.0)
    result.add_argument(
        "--sample-phase",
        choices=("all", "pre_relation", "relation"),
        default="all",
        help="Optionally train and evaluate only one recorded task phase.",
    )
    result.add_argument(
        "--flow-context-mode",
        choices=("full", "remaining"),
        default="full",
    )
    result.add_argument(
        "--architecture",
        choices=(
            "mlp",
            "flow_transformer",
            "flow_bottleneck",
            "flow_se3_bottleneck",
        ),
        default="mlp",
    )
    result.add_argument("--explicit-flow-progress", action="store_true")
    result.add_argument(
        "--action-frame",
        choices=("world", "eef"),
        default="world",
        help="Express Cartesian translation/rotvec labels in world or instantaneous EEF frames.",
    )
    result.add_argument(
        "--evaluate-recovery-samples",
        action="store_true",
        help="Evaluate synthetic recovery rows as well as nominal test rows.",
    )
    result.add_argument(
        "--recovery-target-fraction",
        type=float,
        default=None,
        help=(
            "Reweight recovery rows so their total training-loss mass equals "
            "this fraction; e.g. 0.3 implements a nominal/recovery 70:30 mix."
        ),
    )
    return result


def main() -> None:
    args = parser().parse_args()
    if args.recovery_target_fraction is not None and not (
        0.0 < float(args.recovery_target_fraction) < 1.0
    ):
        raise ValueError("--recovery-target-fraction must lie in (0,1)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.dataset, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    if args.action_frame == "eef":
        payload = payload_with_eef_frame_actions(payload)
    test_payload = payload
    if args.test_flow_prediction is not None:
        test_payload = dict(payload)
        test_payload["episode_flow"] = load_predicted_episode_flow(
            args.test_flow_prediction, len(payload["episode_flow"])
        )
    shoe_ids = payload["shoe_id"]
    flow_steps = int(payload["episode_flow"].shape[2])
    horizon = int(payload["action"].shape[1])
    device = torch.device(args.device)
    all_results = []

    for fold_index in args.folds:
        fold = FOLDS[int(fold_index)]
        masks = {
            split: np.flatnonzero(np.isin(shoe_ids, ids))
            for split, ids in fold.items()
        }
        recovery_mask = np.asarray(
            payload.get(
                "is_recovery_sample",
                payload.get("is_recovery_augmented", np.zeros(len(shoe_ids))),
            ),
            dtype=np.float32,
        )
        # Recovery rows are a training augmentation.  Validation is always
        # nominal; testing includes them only for an explicit robustness audit.
        masks["validation"] = masks["validation"][
            recovery_mask[masks["validation"]] < 0.5
        ]
        if not args.evaluate_recovery_samples:
            masks["test"] = masks["test"][recovery_mask[masks["test"]] < 0.5]
        if args.sample_phase != "all":
            phase_value = 1.0 if args.sample_phase == "relation" else 0.0
            masks = {
                split: indices[
                    np.isclose(payload["relation_phase"][indices], phase_value)
                ]
                for split, indices in masks.items()
            }
            if any(len(indices) == 0 for indices in masks.values()):
                raise RuntimeError(
                    f"sample phase {args.sample_phase!r} emptied an object split"
                )
        nominal_train = masks["train"][recovery_mask[masks["train"]] < 0.5]
        recovery_train = masks["train"][recovery_mask[masks["train"]] >= 0.5]
        recovery_loss_weight = 1.0
        if args.recovery_target_fraction is not None:
            if len(recovery_train) == 0 or len(nominal_train) == 0:
                raise RuntimeError(
                    "recovery reweighting requires both nominal and recovery training rows"
                )
            target_fraction = float(args.recovery_target_fraction)
            recovery_loss_weight = (
                target_fraction
                * float(len(nominal_train))
                / ((1.0 - target_fraction) * float(len(recovery_train)))
            )
        normalization = fit_normalization(
            payload, nominal_train if len(nominal_train) else masks["train"]
        )
        datasets = {
            "train": TaskFlowDataset(
                payload,
                masks["train"],
                normalization,
                flow_noise_translation_std_m=float(args.flow_noise_translation_std_cm) / 100.0,
                flow_noise_rotation_std_deg=float(args.flow_noise_rotation_std_deg),
                flow_context_mode=str(args.flow_context_mode),
            ),
            "validation": TaskFlowDataset(
                payload,
                masks["validation"],
                normalization,
                flow_context_mode=str(args.flow_context_mode),
            ),
            "test": TaskFlowDataset(
                test_payload,
                masks["test"],
                normalization,
                flow_context_mode=str(args.flow_context_mode),
            ),
        }
        loaders = {
            "train": DataLoader(
                datasets["train"],
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
                persistent_workers=args.num_workers > 0,
            ),
            "validation": DataLoader(
                datasets["validation"],
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
                persistent_workers=args.num_workers > 0,
            ),
            "test": DataLoader(
                datasets["test"],
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
                persistent_workers=args.num_workers > 0,
            ),
        }

        for condition in args.conditions:
            run_seed = int(args.seed) + int(fold_index) * 100
            seed_everything(run_seed)
            model = build_action_predictor(
                condition,
                flow_steps,
                horizon,
                str(args.architecture),
                bool(args.explicit_flow_progress),
            ).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
            )
            best_state = None
            best_validation = float("inf")
            best_epoch = -1
            stale = 0
            history = []
            for epoch in range(int(args.epochs)):
                model.train()
                total_loss = 0.0
                total_count = 0
                for batch in loaders["train"]:
                    batch = move_batch(batch, device)
                    optimizer.zero_grad(set_to_none=True)
                    prediction_value = model(batch)
                    loss = weighted_action_loss(
                        prediction_value,
                        batch["action"],
                        batch["active_arm_right"],
                        batch["relation_phase"],
                        active_arm_weight=float(args.active_arm_loss_weight),
                        relation_weight=float(args.relation_loss_weight),
                        sample_weight=torch.where(
                            batch["is_recovery_sample"] >= 0.5,
                            torch.full_like(
                                batch["is_recovery_sample"],
                                float(recovery_loss_weight),
                            ),
                            torch.ones_like(batch["is_recovery_sample"]),
                        ),
                    )
                    routing_loss = active_arm_routing_loss(
                        prediction_value, batch["active_arm_right"]
                    )
                    loss = loss + float(args.arm_routing_loss_weight) * routing_loss
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    total_loss += float(loss.item()) * len(batch["action"])
                    total_count += int(len(batch["action"]))
                val = validation_loss(model, loaders["validation"], device)
                train = total_loss / max(total_count, 1)
                history.append({"epoch": epoch, "train": train, "validation": val})
                if val < best_validation - 1e-5:
                    best_validation = val
                    best_epoch = epoch
                    best_state = copy.deepcopy(model.state_dict())
                    stale = 0
                else:
                    stale += 1
                if epoch % 10 == 0 or stale == 0:
                    print(
                        json.dumps(
                            {
                                "fold": fold_index,
                                "condition": condition,
                                "epoch": epoch,
                                "train": train,
                                "validation": val,
                                "best_validation": best_validation,
                            }
                        ),
                        flush=True,
                    )
                if stale >= int(args.patience):
                    break
            if best_state is None:
                raise RuntimeError("training did not produce a checkpoint")
            model.load_state_dict(best_state)
            test_indices, test_prediction = predict(
                model, loaders["test"], device, normalization
            )
            order = np.argsort(test_indices)
            test_indices = test_indices[order]
            test_prediction = test_prediction[order]
            summaries = {
                "all": summarize_errors(payload, test_indices, test_prediction)
            }
            recovery_key = (
                "is_recovery_sample"
                if "is_recovery_sample" in payload
                else "is_recovery_augmented"
                if "is_recovery_augmented" in payload
                else None
            )
            if recovery_key is not None:
                for recovery_name, recovery_value in (("nominal", 0.0), ("recovery", 1.0)):
                    keep = np.isclose(
                        payload[recovery_key][test_indices], recovery_value
                    )
                    if np.any(keep):
                        summaries[recovery_name] = summarize_errors(
                            payload, test_indices[keep], test_prediction[keep]
                        )
            for phase_name, phase_value in (("pre_relation", 0.0), ("relation", 1.0)):
                keep = np.isclose(payload["relation_phase"][test_indices], phase_value)
                if np.any(keep):
                    summaries[phase_name] = summarize_errors(
                        payload, test_indices[keep], test_prediction[keep]
                    )
            per_shoe = {}
            for shoe_id in sorted(np.unique(payload["shoe_id"][test_indices]).tolist()):
                keep = payload["shoe_id"][test_indices] == shoe_id
                per_shoe[str(int(shoe_id))] = summarize_errors(
                    payload, test_indices[keep], test_prediction[keep]
                )
            result = {
                "condition": condition,
                "fold": int(fold_index),
                "seed": int(run_seed),
                "split": {key: list(value) for key, value in fold.items()},
                "best_epoch": int(best_epoch),
                "best_validation_normalized_mse": float(best_validation),
                "summary": summaries,
                "per_shoe": per_shoe,
                "history": history,
                "test_flow_prediction": (
                    str(args.test_flow_prediction.resolve())
                    if args.test_flow_prediction is not None
                    else None
                ),
                "flow_noise_translation_std_cm": float(args.flow_noise_translation_std_cm),
                "flow_noise_rotation_std_deg": float(args.flow_noise_rotation_std_deg),
                "active_arm_loss_weight": float(args.active_arm_loss_weight),
                "arm_routing_loss_weight": float(args.arm_routing_loss_weight),
                "relation_loss_weight": float(args.relation_loss_weight),
                "sample_phase": str(args.sample_phase),
                "flow_context_mode": str(args.flow_context_mode),
                "architecture": str(args.architecture),
                "explicit_flow_progress": bool(args.explicit_flow_progress),
                "action_frame": str(args.action_frame),
                "evaluate_recovery_samples": bool(args.evaluate_recovery_samples),
                "recovery_target_fraction": (
                    float(args.recovery_target_fraction)
                    if args.recovery_target_fraction is not None
                    else None
                ),
                "recovery_loss_weight": float(recovery_loss_weight),
                "normalization_uses_nominal_training_only": bool(len(nominal_train)),
            }
            result_path = args.output_dir / f"fold{fold_index}_{condition}_seed{args.seed}.json"
            result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            torch.save(
                {
                    "model": model.state_dict(),
                    "condition": condition,
                    "flow_steps": flow_steps,
                    "horizon": horizon,
                    "normalization": normalization.__dict__,
                    "flow_noise_translation_std_cm": float(args.flow_noise_translation_std_cm),
                    "flow_noise_rotation_std_deg": float(args.flow_noise_rotation_std_deg),
                    "active_arm_loss_weight": float(args.active_arm_loss_weight),
                    "arm_routing_loss_weight": float(args.arm_routing_loss_weight),
                    "relation_loss_weight": float(args.relation_loss_weight),
                    "flow_context_mode": str(args.flow_context_mode),
                    "architecture": str(args.architecture),
                    "explicit_flow_progress": bool(args.explicit_flow_progress),
                    "action_frame": str(args.action_frame),
                    "trained_with_recovery_samples": bool(
                        np.any(recovery_mask[masks["train"]] >= 0.5)
                    ),
                    "recovery_target_fraction": (
                        float(args.recovery_target_fraction)
                        if args.recovery_target_fraction is not None
                        else None
                    ),
                    "recovery_loss_weight": float(recovery_loss_weight),
                },
                args.output_dir / f"fold{fold_index}_{condition}_seed{args.seed}.pt",
            )
            all_results.append(result)
            print(json.dumps({key: result[key] for key in ("condition", "fold", "summary")}), flush=True)

    summary_path = args.output_dir / f"summary_seed{args.seed}.json"
    summary_path.write_text(json.dumps(all_results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(summary_path), "runs": len(all_results)}, indent=2))


if __name__ == "__main__":
    main()
