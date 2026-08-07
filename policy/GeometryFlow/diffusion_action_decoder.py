"""Small diffusion transformer for geometry-conditioned action chunks.

This module is intentionally self contained.  It uses a cosine DDPM training
schedule and deterministic DDIM sampling, but does not depend on the large RDT
or pi0 stacks that are also vendored in this repository.  Geometry stays as a
sequence of memory tokens and is read by cross-attention at every denoising
layer.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .interaction_flow_tokens import CrossAttentionBlock


def _cosine_alpha_cumprod(steps: int, offset: float = 0.008) -> torch.Tensor:
    if steps < 2:
        raise ValueError("diffusion steps must be at least two")
    position = torch.linspace(0.0, float(steps), steps + 1, dtype=torch.float64)
    angle = ((position / float(steps)) + offset) / (1.0 + offset)
    cumulative = torch.cos(angle * math.pi * 0.5).square()
    cumulative = cumulative / cumulative[0]
    beta = (1.0 - cumulative[1:] / cumulative[:-1]).clamp(1e-5, 0.999)
    return torch.cumprod(1.0 - beta, dim=0).float()


def sinusoidal_timestep_embedding(
    timestep: torch.Tensor, feature_dim: int
) -> torch.Tensor:
    """Return a standard sin/cos embedding for integer diffusion steps."""
    half = feature_dim // 2
    if half == 0:
        raise ValueError("feature_dim must be at least two")
    frequency = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=timestep.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    phase = timestep.float()[:, None] * frequency[None]
    embedding = torch.cat((phase.sin(), phase.cos()), dim=-1)
    if feature_dim % 2:
        embedding = nn.functional.pad(embedding, (0, 1))
    return embedding


def stateless_normal_like(
    reference: torch.Tensor,
    sample_index: torch.Tensor,
    *,
    salt: float = 0.0,
) -> torch.Tensor:
    """Deterministic approximately Gaussian noise keyed by dataset index.

    A stateless source makes validation and test sampling independent of batch
    size and loader order.  Box-Muller is used instead of a global RNG so paired
    geometry/control runs receive exactly the same initial action noise.
    """
    if reference.ndim != 3:
        raise ValueError("reference must be [B,H,D]")
    key = sample_index.to(device=reference.device, dtype=torch.float32).reshape(-1, 1, 1)
    if len(key) != len(reference):
        raise ValueError("sample_index batch does not match reference")
    horizon = torch.arange(
        reference.shape[1], device=reference.device, dtype=torch.float32
    )[None, :, None]
    channel = torch.arange(
        reference.shape[2], device=reference.device, dtype=torch.float32
    )[None, None, :]
    first_phase = key * 12.9898 + horizon * 78.233 + channel * 37.719 + salt
    second_phase = key * 93.989 + horizon * 67.345 + channel * 11.135 + salt * 1.7
    uniform_one = torch.frac(torch.sin(first_phase) * 43_758.5453).abs()
    uniform_two = torch.frac(torch.sin(second_phase) * 24_634.6345).abs()
    uniform_one = uniform_one.clamp(1e-5, 1.0 - 1e-5)
    noise = torch.sqrt(-2.0 * torch.log(uniform_one)) * torch.cos(
        2.0 * math.pi * uniform_two
    )
    return noise.to(dtype=reference.dtype)


class DiffusionActionDecoder(nn.Module):
    """Denoise an action chunk while cross-attending to geometry tokens."""

    def __init__(
        self,
        *,
        horizon: int,
        feature_dim: int,
        heads: int,
        layers: int,
        dropout: float,
        action_dim: int = 14,
        diffusion_steps: int = 100,
        inference_steps: int = 10,
        prediction_type: str = "sample",
    ) -> None:
        super().__init__()
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if inference_steps <= 0:
            raise ValueError("inference_steps must be positive")
        if prediction_type not in {"sample", "epsilon"}:
            raise ValueError("prediction_type must be 'sample' or 'epsilon'")
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.diffusion_steps = int(diffusion_steps)
        self.inference_steps = int(inference_steps)
        self.prediction_type = str(prediction_type)
        self.action_projection = nn.Linear(self.action_dim, feature_dim)
        self.action_position = nn.Parameter(torch.empty(self.horizon, feature_dim))
        nn.init.normal_(self.action_position, std=0.02)
        self.time_projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.SiLU(),
            nn.Linear(feature_dim * 2, feature_dim),
        )
        self.self_attention = nn.ModuleList()
        self.geometry_attention = nn.ModuleList()
        for _ in range(layers):
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
            self.geometry_attention.append(
                CrossAttentionBlock(feature_dim, heads, dropout)
            )
        self.output_norm = nn.LayerNorm(feature_dim)
        self.prediction_head = nn.Linear(feature_dim, self.action_dim)
        # A zero prediction is a low-information initialization for either
        # normalized clean actions or epsilon and avoids a random shortcut.
        nn.init.zeros_(self.prediction_head.weight)
        nn.init.zeros_(self.prediction_head.bias)
        self.register_buffer(
            "alpha_cumprod", _cosine_alpha_cumprod(self.diffusion_steps)
        )

    def predict_noise(
        self,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
        scene: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_action.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError(
                "noisy_action must be "
                f"[B,{self.horizon},{self.action_dim}], got {tuple(noisy_action.shape)}"
            )
        time = self.time_projection(
            sinusoidal_timestep_embedding(timestep, scene.shape[-1])
        )
        token = (
            self.action_projection(noisy_action)
            + self.action_position[None]
            + scene[:, None, :]
            + time[:, None, :]
        )
        for self_block, geometry_block in zip(
            self.self_attention, self.geometry_attention
        ):
            token = self_block(token)
            token = geometry_block(token, memory)
        return self.prediction_head(self.output_norm(token))

    def add_noise(
        self,
        clean_action: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        alpha = self.alpha_cumprod[timestep].to(clean_action.dtype).reshape(-1, 1, 1)
        return alpha.sqrt() * clean_action + (1.0 - alpha).sqrt() * noise

    def training_inputs(
        self,
        clean_action: torch.Tensor,
        *,
        sample_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create random train inputs or fixed per-sample validation inputs."""
        if sample_index is None:
            noise = torch.randn_like(clean_action)
            timestep = torch.randint(
                0,
                self.diffusion_steps,
                (len(clean_action),),
                device=clean_action.device,
            )
        else:
            noise = stateless_normal_like(clean_action, sample_index, salt=17.0)
            timestep = (
                sample_index.to(device=clean_action.device, dtype=torch.long) * 37
                + 17
            ) % self.diffusion_steps
        target = clean_action if self.prediction_type == "sample" else noise
        return self.add_noise(clean_action, noise, timestep), timestep, target

    @torch.no_grad()
    def sample(
        self,
        scene: torch.Tensor,
        memory: torch.Tensor,
        *,
        initial_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if initial_noise is None:
            action = torch.randn(
                len(scene),
                self.horizon,
                self.action_dim,
                dtype=scene.dtype,
                device=scene.device,
            )
        else:
            action = initial_noise
        schedule = torch.linspace(
            self.diffusion_steps - 1,
            0,
            min(self.inference_steps, self.diffusion_steps),
            device=scene.device,
        ).round().long().unique_consecutive()
        for position, value in enumerate(schedule):
            timestep = value.expand(len(scene))
            prediction = self.predict_noise(action, timestep, scene, memory)
            alpha = self.alpha_cumprod[value].to(action.dtype)
            if self.prediction_type == "sample":
                clean = prediction
                noise = (
                    action - torch.sqrt(alpha) * clean
                ) / torch.sqrt(1.0 - alpha).clamp_min(1e-6)
            else:
                noise = prediction
                clean = (
                    action - torch.sqrt(1.0 - alpha) * noise
                ) / torch.sqrt(alpha).clamp_min(1e-6)
            clean = clean.clamp(-5.0, 5.0)
            if position + 1 == len(schedule):
                action = clean
            else:
                previous = schedule[position + 1]
                previous_alpha = self.alpha_cumprod[previous].to(action.dtype)
                action = (
                    torch.sqrt(previous_alpha) * clean
                    + torch.sqrt(1.0 - previous_alpha) * noise
                )
        return action
