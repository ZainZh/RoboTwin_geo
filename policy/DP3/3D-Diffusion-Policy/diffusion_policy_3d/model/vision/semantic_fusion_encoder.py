from __future__ import annotations

import copy
import math
from typing import Dict

import torch
import torch.nn as nn

from diffusion_policy_3d.model.vision.pointnet_extractor import (
    DP3Encoder,
    PointNetEncoderXYZRGB,
)


SUPPORTED_SEMANTIC_FUSION_MODES = {
    "residual_pointnet",
    "state_cross_attention",
    "part_soft_pool",
}


class SemanticFusionDP3Encoder(nn.Module):
    """Add semantic point sets to an unchanged object-centric DP3 encoder.

    The base encoder is constructed first from only the primary point cloud and
    low-dimensional observations. Semantic modules are constructed afterwards,
    so with the same torch seed the base weights match Strict Vanilla exactly.
    Every semantic path is multiplied by a tanh gate initialized to zero by
    default. Consequently, the initial output is bitwise identical to the base
    encoder and keeps the same feature dimension.
    """

    def __init__(
        self,
        observation_space: Dict,
        img_crop_shape=None,
        out_channel: int = 256,
        state_mlp_size=(64, 64),
        state_mlp_activation_fn=nn.ReLU,
        pointcloud_encoder_cfg=None,
        use_pc_color: bool = False,
        pointnet_type: str = "pointnet",
        pointcloud_fusion_mode: str = "residual_pointnet",
        semantic_gate_init: float = 0.0,
        semantic_gate_trainable: bool = True,
        semantic_gate_scale: float = 1.0,
        semantic_attention_heads: int = 4,
        semantic_attention_dropout: float = 0.0,
        part_pool_temperature: float = 1.0,
    ):
        super().__init__()
        mode = str(pointcloud_fusion_mode)
        if mode not in SUPPORTED_SEMANTIC_FUSION_MODES:
            raise ValueError(
                f"Unsupported pointcloud_fusion_mode {mode!r}; expected one of "
                f"{sorted(SUPPORTED_SEMANTIC_FUSION_MODES)}"
            )
        if pointnet_type != "pointnet":
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")
        if float(part_pool_temperature) <= 0.0:
            raise ValueError("part_pool_temperature must be positive")

        self.primary_point_cloud_key = "point_cloud"
        self.imagination_key = "imagin_robot"
        self.rgb_image_key = "image"
        self.out_channel = int(out_channel)
        self.pointcloud_fusion_mode = mode
        self.part_pool_temperature = float(part_pool_temperature)
        self.semantic_gate_scale = float(semantic_gate_scale)
        if not math.isfinite(self.semantic_gate_scale):
            raise ValueError("semantic_gate_scale must be finite")

        point_cloud_keys = []
        low_dim_keys = []
        for key, shape in observation_space.items():
            if key in {self.imagination_key, self.rgb_image_key}:
                continue
            rank = len(tuple(shape))
            if rank == 2:
                point_cloud_keys.append(key)
            elif rank == 1:
                low_dim_keys.append(key)
        if self.primary_point_cloud_key not in point_cloud_keys:
            raise ValueError(
                "SemanticFusionDP3Encoder requires the primary 'point_cloud' key"
            )
        self.semantic_point_cloud_keys = [
            key for key in point_cloud_keys if key != self.primary_point_cloud_key
        ]
        if not self.semantic_point_cloud_keys:
            raise ValueError(
                "SemanticFusionDP3Encoder requires at least one secondary point cloud"
            )

        base_keys = [self.primary_point_cloud_key, *low_dim_keys]
        if self.imagination_key in observation_space:
            base_keys.append(self.imagination_key)
        base_space = {key: observation_space[key] for key in base_keys}
        self.base_observation_keys = tuple(base_space)

        # Construct this before every semantic module. With a shared torch seed,
        # its parameters therefore match a Strict Vanilla DP3Encoder exactly.
        self.base_encoder = DP3Encoder(
            observation_space=base_space,
            img_crop_shape=img_crop_shape,
            out_channel=self.out_channel,
            state_mlp_size=state_mlp_size,
            state_mlp_activation_fn=state_mlp_activation_fn,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            use_pc_color=use_pc_color,
            pointnet_type=pointnet_type,
        )
        self.n_output_channels = int(self.base_encoder.output_shape())
        self.state_feature_dim = self.n_output_channels - self.out_channel
        if self.state_feature_dim < 0:
            raise RuntimeError("Base encoder output is smaller than point feature size")

        self.point_cloud_keys = [
            self.primary_point_cloud_key,
            *self.semantic_point_cloud_keys,
        ]
        self.point_cloud_shapes = {
            key: observation_space[key] for key in self.point_cloud_keys
        }
        self.point_cloud_channel_map = dict(
            self.base_encoder.point_cloud_channel_map
        )
        semantic_rng_state = torch.random.get_rng_state()
        try:
            for key in self.semantic_point_cloud_keys:
                self.point_cloud_channel_map[key] = int(
                    tuple(observation_space[key])[-1]
                )
    
            gate_value = torch.full(
                (self.out_channel,),
                float(semantic_gate_init),
                dtype=torch.float32,
            )
            self.semantic_gates = nn.ParameterDict(
                {
                    key: nn.Parameter(
                        gate_value.clone(),
                        requires_grad=bool(semantic_gate_trainable),
                    )
                    for key in self.semantic_point_cloud_keys
                }
            )
    
            self.semantic_extractors = nn.ModuleDict()
            self.semantic_input_projections = nn.ModuleDict()
            self.semantic_attentions = nn.ModuleDict()
            self.semantic_query_projections = nn.ModuleDict()
            self.semantic_output_mlps = nn.ModuleDict()
            self.semantic_task_queries = nn.ParameterDict()
            self.part_pool_mlps = nn.ModuleDict()
    
            for key in self.semantic_point_cloud_keys:
                channels = self.point_cloud_channel_map[key]
                if mode == "residual_pointnet":
                    cfg = copy.deepcopy(pointcloud_encoder_cfg)
                    cfg.in_channels = channels
                    self.semantic_extractors[key] = PointNetEncoderXYZRGB(**cfg)
                elif mode == "state_cross_attention":
                    if self.out_channel % int(semantic_attention_heads) != 0:
                        raise ValueError(
                            "out_channel must be divisible by semantic_attention_heads"
                        )
                    self.semantic_input_projections[key] = nn.Sequential(
                        nn.Linear(channels, self.out_channel),
                        nn.LayerNorm(self.out_channel),
                        nn.GELU(),
                        nn.Linear(self.out_channel, self.out_channel),
                    )
                    self.semantic_attentions[key] = nn.MultiheadAttention(
                        embed_dim=self.out_channel,
                        num_heads=int(semantic_attention_heads),
                        dropout=float(semantic_attention_dropout),
                        batch_first=True,
                    )
                    if self.state_feature_dim > 0:
                        self.semantic_query_projections[key] = nn.Linear(
                            self.state_feature_dim,
                            self.out_channel,
                        )
                    self.semantic_output_mlps[key] = nn.Sequential(
                        nn.LayerNorm(self.out_channel),
                        nn.Linear(self.out_channel, self.out_channel * 2),
                        nn.GELU(),
                        nn.Linear(self.out_channel * 2, self.out_channel),
                    )
                    self.semantic_task_queries[key] = nn.Parameter(
                        torch.zeros(self.out_channel)
                    )
                else:
                    if channels <= 3:
                        raise ValueError(
                            "part_soft_pool requires XYZ plus at least one part score"
                        )
                    num_parts = channels - 3
                    descriptor_dim = num_parts * 7
                    self.part_pool_mlps[key] = nn.Sequential(
                        nn.Linear(descriptor_dim, self.out_channel),
                        nn.LayerNorm(self.out_channel),
                        nn.GELU(),
                        nn.Linear(self.out_channel, self.out_channel),
                    )

        finally:
            torch.random.set_rng_state(semantic_rng_state)

    def _semantic_update(
        self,
        key: str,
        points: torch.Tensor,
        main_feature: torch.Tensor,
        state_feature: torch.Tensor,
    ) -> torch.Tensor:
        if self.pointcloud_fusion_mode == "residual_pointnet":
            return self.semantic_extractors[key](points)

        if self.pointcloud_fusion_mode == "state_cross_attention":
            tokens = self.semantic_input_projections[key](points)
            query = main_feature + self.semantic_task_queries[key].unsqueeze(0)
            if self.state_feature_dim > 0:
                query = query + self.semantic_query_projections[key](state_feature)
            attended, _ = self.semantic_attentions[key](
                query.unsqueeze(1),
                tokens,
                tokens,
                need_weights=False,
            )
            return self.semantic_output_mlps[key](attended.squeeze(1))

        xyz = points[..., :3]
        scores = points[..., 3:]
        weights = torch.softmax(
            scores / self.part_pool_temperature,
            dim=-1,
        )
        mass_sum = weights.sum(dim=1).clamp_min(1e-6)
        centroids = torch.einsum("bnk,bnd->bkd", weights, xyz)
        centroids = centroids / mass_sum.unsqueeze(-1)
        centered = xyz.unsqueeze(2) - centroids.unsqueeze(1)
        variances = torch.einsum(
            "bnk,bnkd->bkd",
            weights,
            centered.square(),
        )
        variances = variances / mass_sum.unsqueeze(-1)
        mean_mass = weights.mean(dim=1).unsqueeze(-1)
        descriptor = torch.cat(
            [centroids, variances, mean_mass],
            dim=-1,
        ).flatten(start_dim=1)
        return self.part_pool_mlps[key](descriptor)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        base_observations = {
            key: observations[key] for key in self.base_observation_keys
        }
        base_feature = self.base_encoder(base_observations)
        main_feature = base_feature[..., : self.out_channel]
        state_feature = base_feature[..., self.out_channel :]

        fused_feature = main_feature
        for key in self.semantic_point_cloud_keys:
            if self.semantic_gate_scale == 0.0:
                continue
            points = observations[key]
            if points.ndim != 3:
                raise ValueError(
                    f"Observation {key!r} must have shape [B,N,C], got {points.shape}"
                )
            expected_channels = self.point_cloud_channel_map[key]
            if points.shape[-1] < expected_channels:
                raise ValueError(
                    f"Observation {key!r} has {points.shape[-1]} channels; "
                    f"expected at least {expected_channels}"
                )
            if points.shape[-1] > expected_channels:
                points = points[..., :expected_channels]
            update = self._semantic_update(
                key,
                points,
                main_feature,
                state_feature,
            )
            gate = (
                self.semantic_gate_scale
                * torch.tanh(self.semantic_gates[key])
            ).unsqueeze(0)
            fused_feature = fused_feature + gate * update

        if self.state_feature_dim == 0:
            return fused_feature
        return torch.cat([fused_feature, state_feature], dim=-1)

    def output_shape(self) -> int:
        return self.n_output_channels
