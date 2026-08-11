"""Relation-preserving temporal policy for full RoboTwin manipulation tasks.

The model deliberately keeps scene, object, local relation and global SE(3)
observations as separate memory tokens.  Continuous Cartesian actions are
decoded either by direct regression or conditional flow matching; gripper
commands use a separate binary head in both cases.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .interaction_flow_tokens import CrossAttentionBlock


def continuous_time_embedding(time: torch.Tensor, feature_dim: int) -> torch.Tensor:
    """Sinusoidal embedding for flow time in [0, 1]."""
    if time.ndim != 1:
        raise ValueError(f"time must be [B], got {tuple(time.shape)}")
    half = feature_dim // 2
    frequency = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=time.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    phase = 1_000.0 * time.float()[:, None] * frequency[None]
    result = torch.cat((phase.sin(), phase.cos()), dim=-1)
    if feature_dim % 2:
        result = nn.functional.pad(result, (0, 1))
    return result


class TokenProjector(nn.Module):
    def __init__(self, input_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


class ActionFlowDecoder(nn.Module):
    """Action queries repeatedly cross-attend the complete temporal memory."""

    def __init__(
        self,
        *,
        horizon: int,
        feature_dim: int,
        heads: int,
        layers: int,
        dropout: float,
        decoder_type: str,
        motion_dim: int = 12,
    ) -> None:
        super().__init__()
        if decoder_type not in {"regression", "flow"}:
            raise ValueError("decoder_type must be regression or flow")
        self.horizon = int(horizon)
        self.motion_dim = int(motion_dim)
        self.decoder_type = str(decoder_type)
        self.action_projection = nn.Linear(self.motion_dim, feature_dim)
        self.action_position = nn.Parameter(torch.empty(horizon, feature_dim))
        nn.init.normal_(self.action_position, std=0.02)
        self.time_projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.SiLU(),
            nn.Linear(feature_dim * 2, feature_dim),
        )
        self.self_attention = nn.ModuleList()
        self.memory_attention = nn.ModuleList()
        for _ in range(int(layers)):
            self.self_attention.append(
                nn.TransformerEncoderLayer(
                    d_model=feature_dim,
                    nhead=heads,
                    dim_feedforward=feature_dim * 4,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
            )
            self.memory_attention.append(
                CrossAttentionBlock(feature_dim, heads, dropout)
            )
        self.output_norm = nn.LayerNorm(feature_dim)
        self.motion_head = nn.Linear(feature_dim, self.motion_dim)
        self.gripper_head = nn.Linear(feature_dim, 2)

    def forward(
        self,
        noisy_motion: torch.Tensor,
        time: torch.Tensor,
        scene: torch.Tensor,
        memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected = (self.horizon, self.motion_dim)
        if noisy_motion.shape[1:] != expected:
            raise ValueError(
                "noisy_motion must be "
                f"[B,{self.horizon},{self.motion_dim}], got {tuple(noisy_motion.shape)}"
            )
        time_token = self.time_projection(
            continuous_time_embedding(time, scene.shape[-1])
        )
        query = (
            self.action_projection(noisy_motion)
            + self.action_position[None]
            + scene[:, None]
            + time_token[:, None]
        )
        for self_block, memory_block in zip(
            self.self_attention, self.memory_attention
        ):
            query = self_block(query)
            query = memory_block(query, memory)
        query = self.output_norm(query)
        return self.motion_head(query), self.gripper_head(query)


@dataclass
class FullTaskPolicyOutput:
    motion: torch.Tensor
    gripper_logits: torch.Tensor


class FullTaskTAGRTPolicy(nn.Module):
    """Three-frame relation/temporal Transformer with a shared action decoder."""

    ROLE_SCENE = 0
    ROLE_SOURCE = 1
    ROLE_TARGET = 2
    ROLE_LOCAL = 3
    ROLE_GLOBAL = 4
    ROLE_STATE = 5

    def __init__(
        self,
        *,
        history: int = 3,
        horizon: int = 12,
        point_dim: int = 6,
        local_dim: int = 12,
        global_dim: int = 10,
        state_dim: int = 20,
        feature_dim: int = 192,
        heads: int = 6,
        memory_layers: int = 4,
        decoder_layers: int = 4,
        dropout: float = 0.05,
        decoder_type: str = "flow",
        motion_dim: int = 12,
    ) -> None:
        super().__init__()
        self.history = int(history)
        self.horizon = int(horizon)
        self.decoder_type = str(decoder_type)
        self.motion_dim = int(motion_dim)
        self.point_projection = TokenProjector(point_dim, feature_dim)
        self.local_projection = TokenProjector(local_dim, feature_dim)
        self.global_projection = TokenProjector(global_dim, feature_dim)
        self.state_projection = TokenProjector(state_dim, feature_dim)
        self.role_embedding = nn.Parameter(torch.empty(6, feature_dim))
        self.history_embedding = nn.Parameter(torch.empty(history, feature_dim))
        nn.init.normal_(self.role_embedding, std=0.02)
        nn.init.normal_(self.history_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.memory_encoder = nn.TransformerEncoder(layer, memory_layers)
        self.scene_norm = nn.LayerNorm(feature_dim)
        self.decoder = ActionFlowDecoder(
            horizon=horizon,
            feature_dim=feature_dim,
            heads=heads,
            layers=decoder_layers,
            dropout=dropout,
            decoder_type=decoder_type,
            motion_dim=self.motion_dim,
        )

    def _role_time(
        self, value: torch.Tensor, role: int, time_index: int
    ) -> torch.Tensor:
        return (
            value
            + self.role_embedding[int(role)][None, None]
            + self.history_embedding[int(time_index)][None, None]
        )

    def encode_memory(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        required = ("scene", "points_a", "points_b", "local", "global", "state")
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(f"missing full-task inputs: {missing}")
        tokens = []
        for time_index in range(self.history):
            tokens.extend(
                (
                    self._role_time(
                        self.point_projection(batch["scene"][:, time_index]),
                        self.ROLE_SCENE,
                        time_index,
                    ),
                    self._role_time(
                        self.point_projection(batch["points_a"][:, time_index]),
                        self.ROLE_SOURCE,
                        time_index,
                    ),
                    self._role_time(
                        self.point_projection(batch["points_b"][:, time_index]),
                        self.ROLE_TARGET,
                        time_index,
                    ),
                    self._role_time(
                        self.local_projection(batch["local"][:, time_index]),
                        self.ROLE_LOCAL,
                        time_index,
                    ),
                    self._role_time(
                        self.global_projection(
                            batch["global"][:, time_index, None]
                        ),
                        self.ROLE_GLOBAL,
                        time_index,
                    ),
                    self._role_time(
                        self.state_projection(batch["state"][:, time_index, None]),
                        self.ROLE_STATE,
                        time_index,
                    ),
                )
            )
        memory = self.memory_encoder(torch.cat(tokens, dim=1))
        # The newest state token is always the final token in the constructed memory.
        scene = self.scene_norm(memory[:, -1])
        return scene, memory

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        noisy_motion: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
    ) -> FullTaskPolicyOutput:
        scene, memory = self.encode_memory(batch)
        batch_size = len(scene)
        if noisy_motion is None:
            noisy_motion = torch.zeros(
                batch_size,
                self.horizon,
                self.motion_dim,
                dtype=scene.dtype,
                device=scene.device,
            )
        if time is None:
            time = torch.ones(batch_size, dtype=scene.dtype, device=scene.device)
        motion, gripper = self.decoder(noisy_motion, time, scene, memory)
        return FullTaskPolicyOutput(motion=motion, gripper_logits=gripper)

    @torch.no_grad()
    def sample(
        self,
        batch: dict[str, torch.Tensor],
        *,
        integration_steps: int = 16,
        initial_noise: torch.Tensor | None = None,
    ) -> FullTaskPolicyOutput:
        scene, memory = self.encode_memory(batch)
        if self.decoder_type == "regression":
            zero = torch.zeros(
                len(scene),
                self.horizon,
                self.motion_dim,
                device=scene.device,
                dtype=scene.dtype,
            )
            motion, gripper = self.decoder(
                zero, torch.ones(len(scene), device=scene.device), scene, memory
            )
            return FullTaskPolicyOutput(motion=motion, gripper_logits=gripper)
        motion = (
            torch.randn(
                len(scene),
                self.horizon,
                self.motion_dim,
                device=scene.device,
                dtype=scene.dtype,
            )
            if initial_noise is None
            else initial_noise.clone()
        )
        step_size = 1.0 / int(integration_steps)
        gripper = None
        for step in range(int(integration_steps)):
            time = torch.full(
                (len(scene),),
                (step + 0.5) * step_size,
                device=scene.device,
                dtype=scene.dtype,
            )
            velocity, gripper = self.decoder(motion, time, scene, memory)
            motion = motion + step_size * velocity
        assert gripper is not None
        return FullTaskPolicyOutput(motion=motion, gripper_logits=gripper)


def split_action14(action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if action.shape[-1] != 14:
        raise ValueError(f"action must end in 14 channels, got {tuple(action.shape)}")
    motion = torch.cat((action[..., :6], action[..., 7:13]), dim=-1)
    gripper = torch.stack((action[..., 6], action[..., 13]), dim=-1)
    return motion, gripper


def merge_action14(motion: torch.Tensor, gripper: torch.Tensor) -> torch.Tensor:
    if motion.shape[:-1] != gripper.shape[:-1] or motion.shape[-1] != 12 or gripper.shape[-1] != 2:
        raise ValueError("motion/gripper shapes are incompatible")
    return torch.cat(
        (
            motion[..., :6],
            gripper[..., :1],
            motion[..., 6:],
            gripper[..., 1:],
        ),
        dim=-1,
    )


def split_action26(action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split dense joint actions into 24 continuous + 2 gripper channels.

    Layout: left position6, left velocity6, left gripper, right position6,
    right velocity6, right gripper.
    """
    if action.shape[-1] != 26:
        raise ValueError(f"action must end in 26 channels, got {tuple(action.shape)}")
    motion = torch.cat((action[..., :12], action[..., 13:25]), dim=-1)
    gripper = torch.stack((action[..., 12], action[..., 25]), dim=-1)
    return motion, gripper


def merge_action26(motion: torch.Tensor, gripper: torch.Tensor) -> torch.Tensor:
    if (
        motion.shape[:-1] != gripper.shape[:-1]
        or motion.shape[-1] != 24
        or gripper.shape[-1] != 2
    ):
        raise ValueError("dense motion/gripper shapes are incompatible")
    return torch.cat(
        (
            motion[..., :12],
            gripper[..., :1],
            motion[..., 12:],
            gripper[..., 1:],
        ),
        dim=-1,
    )


def split_policy_action(action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if action.shape[-1] == 14:
        return split_action14(action)
    if action.shape[-1] == 26:
        return split_action26(action)
    raise ValueError(f"unsupported policy action width {action.shape[-1]}")
