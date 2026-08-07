"""Interaction-centric geometric flow tokens for camera-only action prediction.

The policy consumes only the current segmented A/B point clouds and robot
proprioception.  Learned task queries select an unordered set of interaction
anchors on both objects.  Relation tokens built from these anchors condition
every action-decoder layer.  During training, an optional auxiliary head
predicts the future trajectory of the A anchors as an unordered point set.

Simulator object poses may be used to build the auxiliary target, but neither
the target nor an estimated pose is an inference input.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class InteractionFlowOutput:
    action: torch.Tensor
    recovery_action_residual: torch.Tensor | None = None
    anchor_a: torch.Tensor | None = None
    anchor_b: torch.Tensor | None = None
    predicted_flow: torch.Tensor | None = None
    predicted_rotation: torch.Tensor | None = None
    attention_a: torch.Tensor | None = None
    attention_b: torch.Tensor | None = None


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention followed by a residual feed-forward block."""

    def __init__(self, feature_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(feature_dim)
        self.memory_norm = nn.LayerNorm(feature_dim)
        self.attention = nn.MultiheadAttention(
            feature_dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.feed_forward_norm = nn.LayerNorm(feature_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim),
        )

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        normalized_query = self.query_norm(query)
        normalized_memory = self.memory_norm(memory)
        attended, _ = self.attention(
            normalized_query,
            normalized_memory,
            normalized_memory,
            need_weights=False,
        )
        query = query + attended
        return query + self.feed_forward(self.feed_forward_norm(query))


class PointTokenEncoder(nn.Module):
    """Encode normalized XYZ while retaining a token for every observed point."""

    def __init__(self, feature_dim: int, input_dim: int = 3) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(int(input_dim), feature_dim),
            nn.LayerNorm(feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.SiLU(),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 3 or points.shape[-1] != self.network[0].in_features:
            raise ValueError(
                f"points must be [B,N,{self.network[0].in_features}], "
                f"got {tuple(points.shape)}"
            )
        return self.network(points)


class SourceGeometryAdapter(nn.Module):
    """Inject pointwise source-object descriptors without corrupting XYZ tokens.

    The scalar gate starts at zero, so enabling this adapter is initially an
    exact no-op.  This gives a learned policy a safe fallback when a frozen
    geometry encoder is uninformative while still allowing action/flow losses
    to turn the descriptor stream on.
    """

    def __init__(self, descriptor_dim: int, feature_dim: int) -> None:
        super().__init__()
        if int(descriptor_dim) <= 0:
            raise ValueError("descriptor_dim must be positive")
        self.projection = nn.Sequential(
            # Bias-free endpoints make an all-zero descriptor an exact zero
            # feature.  Consequently the capacity-matched zero control cannot
            # turn this branch into an unrelated learned constant residual.
            nn.Linear(int(descriptor_dim), feature_dim, bias=False),
            nn.LayerNorm(feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim, bias=False),
        )
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, descriptor: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.gate) * self.projection(descriptor)


class SourceAxisRelationAdapter(nn.Module):
    """Couple a camera-frame functional axis to every A/B relation token."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(7, feature_dim, bias=False),
            nn.LayerNorm(feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim, bias=False),
        )
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        axis: torch.Tensor,
        delta_normalized: torch.Tensor,
        xyz_std: torch.Tensor,
    ) -> torch.Tensor:
        if axis.ndim != 2 or axis.shape[-1] != 3:
            raise ValueError(f"source axis must be [B,3], got {tuple(axis.shape)}")
        magnitude = torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
        unit_axis = axis / magnitude.clamp_min(1e-6)
        # Preserve an exact all-zero capacity control after normalization.
        unit_axis = torch.where(magnitude > 1e-6, unit_axis, torch.zeros_like(axis))
        expanded_axis = unit_axis[:, None, :].expand_as(delta_normalized)
        delta_metric = delta_normalized * xyz_std.to(
            dtype=delta_normalized.dtype, device=delta_normalized.device
        )
        axial_delta = (delta_metric * expanded_axis).sum(dim=-1, keepdim=True)
        cross_delta = torch.cross(expanded_axis, delta_metric, dim=-1)
        relation = torch.cat((expanded_axis, axial_delta, cross_delta), dim=-1)
        return torch.tanh(self.gate) * self.projection(relation)


class TargetAxisRelationAdapter(nn.Module):
    """Add the desired target axis and its explicit relation to source/delta.

    This branch complements :class:`SourceAxisRelationAdapter`.  It deliberately
    contains no source-only term, so an all-zero target axis is an exact no-op
    and provides a capacity-matched source-axis-only control.
    """

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(11, feature_dim, bias=False),
            nn.LayerNorm(feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim, bias=False),
        )
        self.gate = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _unit(vector: torch.Tensor) -> torch.Tensor:
        magnitude = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
        unit = vector / magnitude.clamp_min(1e-6)
        return torch.where(magnitude > 1e-6, unit, torch.zeros_like(vector))

    def forward(
        self,
        source_axis: torch.Tensor,
        target_axis: torch.Tensor,
        delta_normalized: torch.Tensor,
        xyz_std: torch.Tensor,
    ) -> torch.Tensor:
        for name, value in (("source", source_axis), ("target", target_axis)):
            if value.ndim != 2 or value.shape[-1] != 3:
                raise ValueError(f"{name} axis must be [B,3], got {tuple(value.shape)}")
        source = self._unit(source_axis)
        target = self._unit(target_axis)
        target_expanded = target[:, None, :].expand_as(delta_normalized)
        delta_metric = delta_normalized * xyz_std.to(
            dtype=delta_normalized.dtype, device=delta_normalized.device
        )
        source_target_dot = (source * target).sum(dim=-1, keepdim=True)
        source_target_cross = torch.cross(source, target, dim=-1)
        target_axial_delta = (delta_metric * target_expanded).sum(
            dim=-1, keepdim=True
        )
        target_cross_delta = torch.cross(target_expanded, delta_metric, dim=-1)
        relation = torch.cat(
            (
                target_expanded,
                source_target_dot[:, None, :].expand(-1, delta_normalized.shape[1], -1),
                source_target_cross[:, None, :].expand(-1, delta_normalized.shape[1], -1),
                target_axial_delta,
                target_cross_delta,
            ),
            dim=-1,
        )
        # Every term is zero when target_axis is the zero control.
        return torch.tanh(self.gate) * self.projection(relation)


class ActionQueryDecoder(nn.Module):
    """Let each action-horizon query repeatedly read geometric memory tokens."""

    def __init__(
        self,
        *,
        horizon: int,
        feature_dim: int,
        heads: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.action_queries = nn.Parameter(torch.empty(horizon, feature_dim))
        nn.init.normal_(self.action_queries, std=0.02)
        self.blocks = nn.ModuleList(
            [
                CrossAttentionBlock(feature_dim, heads, dropout)
                for _ in range(layers)
            ]
        )
        self.output_norm = nn.LayerNorm(feature_dim)
        self.action_head = nn.Linear(feature_dim, 14)

    def forward(self, scene: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        query = scene[:, None, :] + self.action_queries[None, :, :]
        for block in self.blocks:
            query = block(query, memory)
        return self.action_head(self.output_norm(query))


class PointTokenActionPolicy(nn.Module):
    """Control: action queries attend to all A/B point tokens without anchors."""

    def __init__(
        self,
        *,
        horizon: int,
        feature_dim: int = 128,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.05,
        point_channels: int = 3,
        target_color_adapter: bool = False,
        source_geometry_adapter: bool = False,
    ) -> None:
        super().__init__()
        self.target_color_adapter = bool(target_color_adapter)
        self.source_geometry_adapter = bool(source_geometry_adapter)
        if self.target_color_adapter and self.source_geometry_adapter:
            raise ValueError(
                "target_color_adapter and source_geometry_adapter are mutually exclusive"
            )
        encoder_input = (
            3
            if self.target_color_adapter or self.source_geometry_adapter
            else int(point_channels)
        )
        self.point_encoder = PointTokenEncoder(feature_dim, input_dim=encoder_input)
        self.source_geometry_projection = None
        if self.source_geometry_adapter:
            if int(point_channels) <= 3:
                raise ValueError("source geometry adapter requires point_channels > 3")
            self.source_geometry_projection = SourceGeometryAdapter(
                int(point_channels) - 3, feature_dim
            )
        self.target_color_projection = None
        if self.target_color_adapter:
            if int(point_channels) <= 3:
                raise ValueError("target color adapter requires point_channels > 3")
            self.target_color_projection = nn.Linear(
                int(point_channels) - 3, feature_dim
            )
            nn.init.zeros_(self.target_color_projection.weight)
            nn.init.zeros_(self.target_color_projection.bias)
        self.state_encoder = nn.Sequential(
            nn.Linear(20, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.type_embedding = nn.Parameter(torch.empty(2, feature_dim))
        nn.init.normal_(self.type_embedding, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.memory_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
            norm=nn.LayerNorm(feature_dim),
        )
        self.decoder = ActionQueryDecoder(
            horizon=horizon,
            feature_dim=feature_dim,
            heads=heads,
            layers=layers,
            dropout=dropout,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> InteractionFlowOutput:
        points_a = batch["points_a"]
        points_b = batch["points_b"]
        if self.target_color_projection is None and self.source_geometry_projection is None:
            point_a = self.point_encoder(points_a)
            point_b = self.point_encoder(points_b)
        else:
            point_a = self.point_encoder(points_a[..., :3])
            point_b = self.point_encoder(points_b[..., :3])
            if self.target_color_projection is not None:
                point_b = point_b + self.target_color_projection(points_b[..., 3:])
            if self.source_geometry_projection is not None:
                point_a = point_a + self.source_geometry_projection(points_a[..., 3:])
        point_a = point_a + self.type_embedding[0]
        point_b = point_b + self.type_embedding[1]
        memory = self.memory_encoder(torch.cat((point_a, point_b), dim=1))
        scene = self.state_encoder(batch["state"])
        return InteractionFlowOutput(action=self.decoder(scene, memory))


class InteractionFlowTokenPolicy(nn.Module):
    """Camera-only policy with learned relational anchors and future-flow loss."""

    def __init__(
        self,
        *,
        horizon: int,
        flow_steps: int,
        num_anchors: int = 16,
        feature_dim: int = 128,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.05,
        predict_flow: bool = True,
        predict_rotation: bool = False,
        flow_parameterization: str = "free",
        xyz_std: tuple[float, float, float] | list[float] | None = None,
        action_std: tuple[float, ...] | list[float] | None = None,
        attention_temperature: float = 0.25,
        point_channels: int = 3,
        target_color_adapter: bool = False,
        source_geometry_adapter: bool = False,
        source_axis_relation_adapter: bool = False,
        target_axis_relation_adapter: bool = False,
        condition_action_on_flow: bool = False,
        functional_frame_token: bool = False,
        functional_frame_in_scene: bool = False,
        functional_frame_gated_scene: bool = False,
        functional_frame_as_memory: bool = True,
        dual_functional_frame_token: bool = False,
        functional_relative_flow: bool = False,
        recovery_action_adapter: bool = False,
        recovery_action_frame: str = "world",
        recovery_local_frame_token: bool = False,
        recovery_local_frame_gated_fusion: bool = False,
        hand_object_frame_token: bool = False,
        action_decoder_type: str = "regression",
        diffusion_steps: int = 100,
        diffusion_inference_steps: int = 10,
        diffusion_prediction_type: str = "sample",
    ) -> None:
        super().__init__()
        if num_anchors <= 0:
            raise ValueError("num_anchors must be positive")
        if flow_steps < 2:
            raise ValueError("flow_steps must be at least two")
        self.num_anchors = int(num_anchors)
        self.flow_steps = int(flow_steps)
        self.predict_flow = bool(predict_flow)
        self.predict_rotation = bool(predict_rotation)
        self.condition_action_on_flow = bool(condition_action_on_flow)
        self.functional_frame_token = bool(functional_frame_token)
        self.functional_frame_in_scene = bool(functional_frame_in_scene)
        self.functional_frame_gated_scene = bool(functional_frame_gated_scene)
        self.functional_frame_as_memory = bool(functional_frame_as_memory)
        self.dual_functional_frame_token = bool(dual_functional_frame_token)
        self.functional_relative_flow = bool(functional_relative_flow)
        self.recovery_action_adapter_enabled = bool(recovery_action_adapter)
        if recovery_action_frame not in {
            "world",
            "source_functional",
            "hybrid_source_rotation",
        }:
            raise ValueError(
                "recovery_action_frame must be world, source_functional, or "
                "hybrid_source_rotation"
            )
        self.recovery_action_frame = str(recovery_action_frame)
        self.recovery_local_frame_token = bool(recovery_local_frame_token)
        self.recovery_local_frame_gated_fusion = bool(
            recovery_local_frame_gated_fusion
        )
        self.hand_object_frame_token = bool(hand_object_frame_token)
        if action_decoder_type not in {"regression", "diffusion"}:
            raise ValueError(
                "action_decoder_type must be 'regression' or 'diffusion'"
            )
        self.action_decoder_type = str(action_decoder_type)
        if self.functional_frame_in_scene and not self.functional_frame_token:
            raise ValueError(
                "functional_frame_in_scene requires functional_frame_token"
            )
        if self.functional_frame_gated_scene and not self.functional_frame_in_scene:
            raise ValueError(
                "functional_frame_gated_scene requires functional_frame_in_scene"
            )
        if self.functional_relative_flow and not self.functional_frame_token:
            raise ValueError(
                "functional_relative_flow requires functional_frame_token"
            )
        if self.dual_functional_frame_token and not self.functional_frame_token:
            raise ValueError(
                "dual_functional_frame_token requires functional_frame_token"
            )
        if self.recovery_action_adapter_enabled and not self.functional_frame_token:
            raise ValueError(
                "recovery_action_adapter requires functional_frame_token"
            )
        if (
            self.recovery_action_frame != "world"
            and not self.recovery_action_adapter_enabled
        ):
            raise ValueError(
                "a non-world recovery_action_frame requires recovery_action_adapter"
            )
        if (
            self.recovery_local_frame_token
            and not self.recovery_action_adapter_enabled
        ):
            raise ValueError(
                "recovery_local_frame_token requires recovery_action_adapter"
            )
        if (
            self.recovery_local_frame_gated_fusion
            and not self.recovery_local_frame_token
        ):
            raise ValueError(
                "recovery_local_frame_gated_fusion requires a local frame token"
            )
        if (
            self.recovery_local_frame_gated_fusion
            and not self.functional_frame_as_memory
        ):
            raise ValueError(
                "gated local-frame fusion requires the world frame memory slot"
            )
        if self.hand_object_frame_token and not self.recovery_action_adapter_enabled:
            raise ValueError(
                "hand_object_frame_token requires recovery_action_adapter"
            )
        self.functional_frame_gate = (
            nn.Parameter(torch.zeros(()))
            if self.functional_frame_gated_scene
            else None
        )
        if self.condition_action_on_flow and not self.predict_flow:
            raise ValueError("flow-conditioned actions require predict_flow=True")
        if flow_parameterization not in {"free", "rigid_se3"}:
            raise ValueError(
                "flow_parameterization must be 'free' or 'rigid_se3'"
            )
        self.flow_parameterization = str(flow_parameterization)
        self.attention_temperature = float(attention_temperature)
        self.register_buffer(
            "xyz_std",
            torch.as_tensor(
                xyz_std if xyz_std is not None else (1.0, 1.0, 1.0),
                dtype=torch.float32,
            ).reshape(1, 1, 3),
        )
        self.register_buffer(
            "action_std",
            torch.as_tensor(
                action_std if action_std is not None else (1.0,) * 14,
                dtype=torch.float32,
            ).reshape(14),
        )
        if torch.any(self.action_std <= 0.0):
            raise ValueError("action_std must be strictly positive")
        self.target_color_adapter = bool(target_color_adapter)
        self.source_geometry_adapter = bool(source_geometry_adapter)
        self.source_axis_relation_adapter = bool(source_axis_relation_adapter)
        self.target_axis_relation_adapter = bool(target_axis_relation_adapter)
        if self.target_axis_relation_adapter and not self.source_axis_relation_adapter:
            raise ValueError(
                "target_axis_relation_adapter requires source_axis_relation_adapter"
            )
        if self.target_color_adapter and self.source_geometry_adapter:
            raise ValueError(
                "target_color_adapter and source_geometry_adapter are mutually exclusive"
            )
        self.xyz_only_point_encoder = bool(
            self.target_color_adapter
            or self.source_geometry_adapter
            or self.source_axis_relation_adapter
            or self.target_axis_relation_adapter
        )
        encoder_input = 3 if self.xyz_only_point_encoder else int(point_channels)
        self.point_encoder = PointTokenEncoder(feature_dim, input_dim=encoder_input)
        self.source_geometry_projection = None
        if self.source_geometry_adapter:
            if int(point_channels) <= 3:
                raise ValueError("source geometry adapter requires point_channels > 3")
            self.source_geometry_projection = SourceGeometryAdapter(
                int(point_channels) - 3, feature_dim
            )
        self.source_axis_relation_projection = (
            SourceAxisRelationAdapter(feature_dim)
            if self.source_axis_relation_adapter
            else None
        )
        self.target_axis_relation_projection = (
            TargetAxisRelationAdapter(feature_dim)
            if self.target_axis_relation_adapter
            else None
        )
        self.target_color_projection = None
        if self.target_color_adapter:
            if int(point_channels) <= 3:
                raise ValueError("target color adapter requires point_channels > 3")
            self.target_color_projection = nn.Linear(
                int(point_channels) - 3, feature_dim
            )
            nn.init.zeros_(self.target_color_projection.weight)
            nn.init.zeros_(self.target_color_projection.bias)
        self.state_encoder = nn.Sequential(
            nn.Linear(20, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.global_scene = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.SiLU(),
        )
        self.anchor_queries = nn.Parameter(
            torch.empty(self.num_anchors, feature_dim)
        )
        nn.init.normal_(self.anchor_queries, std=0.02)
        self.query_condition = nn.Linear(feature_dim, feature_dim)
        self.query_norm = nn.LayerNorm(feature_dim)
        self.key_norm_a = nn.LayerNorm(feature_dim)
        self.key_norm_b = nn.LayerNorm(feature_dim)
        self.key_a = nn.Linear(feature_dim, feature_dim, bias=False)
        self.key_b = nn.Linear(feature_dim, feature_dim, bias=False)
        # Coordinates are explicit: the model cannot hide the relative geometry
        # exclusively inside an opaque pooled feature.
        relation_input_dim = feature_dim * 3 + 9
        self.relation_projection = nn.Sequential(
            nn.Linear(relation_input_dim, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.SiLU(),
            nn.Linear(feature_dim * 2, feature_dim),
        )
        relation_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.relation_encoder = nn.TransformerEncoder(
            relation_layer,
            num_layers=layers,
            norm=nn.LayerNorm(feature_dim),
        )
        if self.action_decoder_type == "regression":
            self.decoder = ActionQueryDecoder(
                horizon=horizon,
                feature_dim=feature_dim,
                heads=heads,
                layers=layers,
                dropout=dropout,
            )
        else:
            # Local import avoids a module cycle: the diffusion decoder reuses
            # CrossAttentionBlock defined above.
            from .diffusion_action_decoder import DiffusionActionDecoder

            self.decoder = DiffusionActionDecoder(
                horizon=horizon,
                feature_dim=feature_dim,
                heads=heads,
                layers=layers,
                dropout=dropout,
                diffusion_steps=diffusion_steps,
                inference_steps=diffusion_inference_steps,
                prediction_type=diffusion_prediction_type,
            )
        if self.recovery_action_adapter_enabled and self.action_decoder_type != "regression":
            raise ValueError("recovery_action_adapter requires regression actions")
        self.recovery_action_adapter = None
        if self.recovery_action_adapter_enabled:
            self.recovery_action_adapter = ActionQueryDecoder(
                horizon=horizon,
                feature_dim=feature_dim,
                heads=heads,
                layers=layers,
                dropout=dropout,
            )
            nn.init.zeros_(self.recovery_action_adapter.action_head.weight)
            nn.init.zeros_(self.recovery_action_adapter.action_head.bias)
        self.recovery_local_frame_projection = None
        self.recovery_local_frame_gate = None
        if self.recovery_local_frame_token:
            self.recovery_local_frame_projection = nn.Sequential(
                nn.Linear(9, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.SiLU(),
                nn.Linear(feature_dim, feature_dim),
            )
            if self.recovery_local_frame_gated_fusion:
                self.recovery_local_frame_gate = nn.Parameter(torch.zeros(()))
        self.hand_object_frame_projection = None
        if self.hand_object_frame_token:
            self.hand_object_frame_projection = nn.Sequential(
                nn.Linear(11, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.SiLU(),
                nn.Linear(feature_dim, feature_dim),
            )
            # Checkpoint initialization is an exact functional no-op before
            # the interaction relation is trained.
            nn.init.zeros_(self.hand_object_frame_projection[-1].weight)
            nn.init.zeros_(self.hand_object_frame_projection[-1].bias)
        self.flow_head = (
            nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, feature_dim * 2),
                nn.SiLU(),
                nn.Linear(feature_dim * 2, (self.flow_steps - 1) * 3),
            )
            if self.predict_flow and self.flow_parameterization == "free"
            else None
        )
        # Initialize every shared supervised head before optional conditioning
        # adapters. This keeps paired baseline/geometry runs parameter-matched,
        # including the SO(3) gradients that flow back into the shared encoder.
        self.rotation_head = None
        if self.predict_rotation:
            self.rotation_head = nn.Sequential(
                nn.LayerNorm(feature_dim * 2),
                nn.Linear(feature_dim * 2, feature_dim * 2),
                nn.SiLU(),
                nn.Linear(feature_dim * 2, 6),
            )
        self.flow_condition_projection = None
        if self.condition_action_on_flow:
            self.flow_condition_projection = nn.Sequential(
                nn.LayerNorm(self.flow_steps * 3),
                nn.Linear(self.flow_steps * 3, feature_dim),
                nn.SiLU(),
                nn.Linear(feature_dim, feature_dim),
            )
            # Begin exactly at the auxiliary-only policy and let action loss
            # learn how much of the predicted trajectory should enter control.
            nn.init.zeros_(self.flow_condition_projection[-1].weight)
            nn.init.zeros_(self.flow_condition_projection[-1].bias)
        self.functional_frame_projection = None
        if self.functional_frame_token:
            self.functional_frame_projection = nn.Sequential(
                nn.Linear(9, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.SiLU(),
                nn.Linear(feature_dim, feature_dim),
            )
        self.dual_functional_frame_projection = None
        self.dual_functional_frame_gate = None
        if self.dual_functional_frame_token:
            self.dual_functional_frame_projection = nn.Sequential(
                nn.Linear(9, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.SiLU(),
                nn.Linear(feature_dim, feature_dim),
            )
            # Start exactly from the validated world-frame policy, then learn
            # whether the task-aligned local relation adds predictive value.
            self.dual_functional_frame_gate = nn.Parameter(torch.zeros(()))
        self.functional_relative_projection = None
        self.functional_relative_gate = None
        if self.functional_relative_flow:
            self.functional_relative_projection = nn.Sequential(
                nn.Linear(9, feature_dim * 2),
                nn.LayerNorm(feature_dim * 2),
                nn.SiLU(),
                nn.Linear(feature_dim * 2, feature_dim),
            )
            self.functional_relative_gate = nn.Parameter(torch.zeros(()))
        self.flow_query = None
        self.flow_cross_attention = None
        self.se3_flow_head = None
        if self.predict_flow and self.flow_parameterization == "rigid_se3":
            self.flow_query = nn.Parameter(torch.empty(1, feature_dim))
            nn.init.normal_(self.flow_query, std=0.02)
            self.flow_cross_attention = CrossAttentionBlock(
                feature_dim, heads, dropout
            )
            self.se3_flow_head = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, feature_dim * 2),
                nn.SiLU(),
                nn.Linear(feature_dim * 2, (self.flow_steps - 1) * 9),
            )
            final = self.se3_flow_head[-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
            with torch.no_grad():
                identity_sixd = torch.tensor(
                    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
                )
                for step in range(self.flow_steps - 1):
                    start = step * 9 + 3
                    final.bias[start : start + 6].copy_(identity_sixd)

    @staticmethod
    def _rotation_sixd(value: torch.Tensor) -> torch.Tensor:
        first = nn.functional.normalize(value[..., :3], dim=-1, eps=1e-6)
        second_raw = value[..., 3:6]
        second = second_raw - (first * second_raw).sum(-1, keepdim=True) * first
        second = nn.functional.normalize(second, dim=-1, eps=1e-6)
        third = torch.cross(first, second, dim=-1)
        return torch.stack((first, second, third), dim=-1)

    def _rigid_flow(
        self,
        anchor_a: torch.Tensor,
        relation: torch.Tensor,
        scene: torch.Tensor,
    ) -> torch.Tensor:
        if self.flow_query is None or self.flow_cross_attention is None:
            raise RuntimeError("rigid flow modules are disabled")
        query = scene[:, None, :] + self.flow_query[None, :, :]
        context = self.flow_cross_attention(query, relation)[:, 0]
        parameters = self.se3_flow_head(context).reshape(
            len(relation), self.flow_steps - 1, 9
        )
        translation_normalized = parameters[..., :3]
        rotation = self._rotation_sixd(parameters[..., 3:])
        scale = self.xyz_std.to(dtype=anchor_a.dtype, device=anchor_a.device)
        anchor_metric = anchor_a * scale
        center_metric = anchor_metric.mean(dim=1, keepdim=True)
        local_metric = anchor_metric - center_metric
        rotated_metric = torch.einsum(
            "btij,bkj->bkti", rotation, local_metric
        )
        translation_metric = translation_normalized * scale[:, :, :]
        future_metric = (
            rotated_metric
            + center_metric[:, :, None, :]
            + translation_metric[:, None, :, :]
        )
        future_normalized = future_metric / scale[:, :, None, :]
        return torch.cat(
            (anchor_a[:, :, None, :], future_normalized), dim=2
        )

    def _select_anchors(
        self,
        points: torch.Tensor,
        point_features: torch.Tensor,
        queries: torch.Tensor,
        key_projection: nn.Module,
        key_norm: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        keys = key_norm(key_projection(point_features))
        normalized_queries = self.query_norm(queries)
        logits = torch.einsum("bkd,bnd->bkn", normalized_queries, keys)
        logits = logits / (
            keys.shape[-1] ** 0.5 * max(self.attention_temperature, 1e-4)
        )
        # Slot-style competition: points first choose among interaction slots,
        # then each slot normalizes the mass assigned to its surface region.
        # Independent point-wise softmaxes otherwise converge to the same object
        # centroid and destroy the intended local geometry.
        attention = torch.softmax(logits, dim=1).clamp_min(1e-8)
        attention = attention / attention.sum(dim=-1, keepdim=True)
        anchors = torch.einsum("bkn,bnd->bkd", attention, points)
        context = torch.einsum("bkn,bnd->bkd", attention, point_features)
        return anchors, context, attention

    def _recovery_action_gate(
        self, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Continuous large-error gate; it is not a task-stage prediction."""

        frame = batch["target_frame9"]
        scale = self.xyz_std.reshape(1, 3).to(
            dtype=frame.dtype, device=frame.device
        )
        translation_m = torch.linalg.vector_norm(frame[:, :3] * scale, dim=-1)
        rotation = self._rotation_sixd(frame[:, 3:9])
        cosine = ((rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5)
        rotation_rad = torch.acos(cosine.clamp(-1.0, 1.0))
        translation_gate = ((translation_m - 0.03) / 0.03).clamp(0.0, 1.0)
        rotation_gate = (
            (rotation_rad - torch.deg2rad(frame.new_tensor(15.0)))
            / torch.deg2rad(frame.new_tensor(15.0))
        ).clamp(0.0, 1.0)
        gate = torch.maximum(translation_gate, rotation_gate)
        if "target_frame_confidence" in batch:
            gate = gate * batch["target_frame_confidence"].to(gate.dtype)
        return gate[:, None, None]

    def _recovery_residual_in_world_frame(
        self,
        residual: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Map a functional-frame recovery residual to normalized world action.

        The decoder predicts translation and rotation-vector increments in the
        current operated-object NDF frame.  Each 3-vector uses one isotropic
        metric scale before rotation, which preserves SO(3) equivariance even
        though the nominal policy's world-action normalization is anisotropic.
        Gripper residuals remain dimensionless and are not rotated.
        """

        if self.recovery_action_frame == "world":
            return residual
        if "source_frame6_columns" not in batch:
            raise ValueError(
                "source-functional recovery requires "
                "batch['source_frame6_columns']"
            )
        frame6 = batch["source_frame6_columns"]
        if frame6.ndim != 2 or frame6.shape != (len(residual), 6):
            raise ValueError(
                "source_frame6_columns must have shape [batch,6], got "
                f"{tuple(frame6.shape)}"
            )
        rotation = self._rotation_sixd(frame6).to(
            dtype=residual.dtype, device=residual.device
        )
        action_std = self.action_std.to(
            dtype=residual.dtype, device=residual.device
        )
        result = residual.clone()
        vector_starts = (
            (3, 10)
            if self.recovery_action_frame == "hybrid_source_rotation"
            else (0, 3, 7, 10)
        )
        for start in vector_starts:
            world_std = action_std[start : start + 3]
            isotropic_metric_scale = world_std.mean()
            local_metric = residual[..., start : start + 3] * (
                isotropic_metric_scale
            )
            world_metric = torch.einsum(
                "bij,bhj->bhi", rotation, local_metric
            )
            result[..., start : start + 3] = world_metric / world_std
        return result

    def forward(self, batch: dict[str, torch.Tensor]) -> InteractionFlowOutput:
        points_a = batch["points_a"]
        points_b = batch["points_b"]
        xyz_a = points_a[..., :3]
        xyz_b = points_b[..., :3]
        if not self.xyz_only_point_encoder:
            features_a = self.point_encoder(points_a)
            features_b = self.point_encoder(points_b)
        else:
            features_a = self.point_encoder(xyz_a)
            features_b = self.point_encoder(xyz_b)
            if self.target_color_projection is not None:
                features_b = features_b + self.target_color_projection(
                    points_b[..., 3:]
                )
            if self.source_geometry_projection is not None:
                features_a = features_a + self.source_geometry_projection(
                    points_a[..., 3:]
                )
        robot = self.state_encoder(batch["state"])
        pooled_a = features_a.mean(dim=1)
        pooled_b = features_b.mean(dim=1)
        scene = self.global_scene(torch.cat((pooled_a, pooled_b, robot), dim=-1))
        frame_feature = None
        local_frame_feature = None
        if self.functional_frame_projection is not None:
            if "target_frame9" not in batch:
                raise ValueError(
                    "functional-frame policy requires batch['target_frame9']"
                )
            frame_feature = self.functional_frame_projection(
                batch["target_frame9"]
            )
            if "target_frame_confidence" in batch:
                frame_feature = frame_feature * batch[
                    "target_frame_confidence"
                ].to(frame_feature.dtype)[:, None]
            if self.functional_frame_in_scene:
                if self.functional_frame_gate is None:
                    scene = scene + frame_feature
                else:
                    scene = scene + torch.tanh(
                        self.functional_frame_gate
                    ) * frame_feature
        if self.dual_functional_frame_projection is not None:
            if "target_frame9_local" not in batch:
                raise ValueError(
                    "dual-frame policy requires batch['target_frame9_local']"
                )
            local_frame_feature = self.dual_functional_frame_projection(
                batch["target_frame9_local"]
            )
            if "target_frame_confidence" in batch:
                local_frame_feature = local_frame_feature * batch[
                    "target_frame_confidence"
                ].to(local_frame_feature.dtype)[:, None]
            local_frame_feature = torch.tanh(
                self.dual_functional_frame_gate
            ) * local_frame_feature
            scene = scene + local_frame_feature
        queries = self.anchor_queries[None, :, :] + self.query_condition(scene)[
            :, None, :
        ]
        anchor_a, context_a, attention_a = self._select_anchors(
            xyz_a, features_a, queries, self.key_a, self.key_norm_a
        )
        anchor_b, context_b, attention_b = self._select_anchors(
            xyz_b, features_b, queries, self.key_b, self.key_norm_b
        )
        delta = anchor_b - anchor_a
        relation = self.relation_projection(
            torch.cat(
                (
                    context_a,
                    context_b,
                    queries,
                    anchor_a,
                    anchor_b,
                    delta,
                ),
                dim=-1,
            )
        )
        if self.source_axis_relation_projection is not None:
            if "source_axis3" not in batch:
                raise ValueError(
                    "source-axis relation adapter requires batch['source_axis3']"
                )
            relation = relation + self.source_axis_relation_projection(
                batch["source_axis3"], delta, self.xyz_std
            )
        if self.target_axis_relation_projection is not None:
            if "target_axis3" not in batch:
                raise ValueError(
                    "target-axis relation adapter requires batch['target_axis3']"
                )
            relation = relation + self.target_axis_relation_projection(
                batch["source_axis3"],
                batch["target_axis3"],
                delta,
                self.xyz_std,
            )
        if self.functional_relative_projection is not None:
            relative_rotation = self._rotation_sixd(
                batch["target_frame9"][..., 3:9]
            )
            scale = self.xyz_std.to(
                dtype=anchor_a.dtype, device=anchor_a.device
            )
            anchor_metric = anchor_a * scale
            center_metric = anchor_metric.mean(dim=1, keepdim=True)
            local_metric = anchor_metric - center_metric
            desired_metric = torch.einsum(
                "bij,bkj->bki", relative_rotation, local_metric
            )
            desired_metric = (
                desired_metric
                + center_metric
                + batch["target_frame9"][:, None, :3] * scale
            )
            desired_anchor = desired_metric / scale
            relative_feature = self.functional_relative_projection(
                torch.cat(
                    (anchor_a, desired_anchor, desired_anchor - anchor_a),
                    dim=-1,
                )
            )
            relation = relation + torch.tanh(
                self.functional_relative_gate
            ) * relative_feature
        relation = self.relation_encoder(relation)
        predicted_flow = None
        if self.flow_head is not None:
            future_offset = self.flow_head(relation).reshape(
                len(relation), self.num_anchors, self.flow_steps - 1, 3
            )
            predicted_flow = torch.cat(
                (
                    anchor_a[:, :, None, :],
                    anchor_a[:, :, None, :] + future_offset,
                ),
                dim=2,
            )
        elif self.se3_flow_head is not None:
            predicted_flow = self._rigid_flow(anchor_a, relation, scene)
        action_memory = relation
        if self.flow_condition_projection is not None:
            if predicted_flow is None:
                raise RuntimeError("flow-conditioned decoder has no predicted flow")
            motion = predicted_flow - predicted_flow[:, :, :1]
            action_memory = action_memory + self.flow_condition_projection(
                motion.flatten(start_dim=2)
            )
        if (
            self.functional_frame_projection is not None
            and self.functional_frame_as_memory
        ):
            memory_frame_feature = frame_feature
            if local_frame_feature is not None:
                memory_frame_feature = memory_frame_feature + local_frame_feature
            action_memory = torch.cat(
                (action_memory, memory_frame_feature[:, None, :]), dim=1
            )
        if self.action_decoder_type == "regression":
            action = self.decoder(scene, action_memory)
        elif "diffusion_noisy_action" in batch:
            if "diffusion_timestep" not in batch:
                raise ValueError("diffusion_timestep is required with noisy action")
            action = self.decoder.predict_noise(
                batch["diffusion_noisy_action"],
                batch["diffusion_timestep"],
                scene,
                action_memory,
            )
        else:
            initial_noise = batch.get("diffusion_initial_noise")
            action = self.decoder.sample(
                scene, action_memory, initial_noise=initial_noise
            )
        recovery_action_residual = None
        if self.recovery_action_adapter is not None:
            recovery_memory = action_memory
            if self.recovery_local_frame_projection is not None:
                if "target_frame9_source_local" not in batch:
                    raise ValueError(
                        "local-equivariant recovery requires "
                        "batch['target_frame9_source_local']"
                    )
                recovery_local_frame_feature = (
                    self.recovery_local_frame_projection(
                        batch["target_frame9_source_local"]
                    )
                )
                if "target_frame_confidence" in batch:
                    recovery_local_frame_feature = (
                        recovery_local_frame_feature
                        * batch["target_frame_confidence"].to(
                            recovery_local_frame_feature.dtype
                        )[:, None]
                    )
                if self.recovery_local_frame_gate is None:
                    recovery_memory = torch.cat(
                        (
                            recovery_memory,
                            recovery_local_frame_feature[:, None, :],
                        ),
                        dim=1,
                    )
                else:
                    recovery_memory = recovery_memory.clone()
                    recovery_memory[:, -1, :] = (
                        recovery_memory[:, -1, :]
                        + torch.tanh(self.recovery_local_frame_gate)
                        * recovery_local_frame_feature
                    )
            if self.hand_object_frame_projection is not None:
                if "hand_object_frame11" not in batch:
                    raise ValueError(
                        "hand-object policy requires batch['hand_object_frame11']"
                    )
                hand_object_feature = self.hand_object_frame_projection(
                    batch["hand_object_frame11"]
                )
                if "hand_object_frame_confidence" in batch:
                    hand_object_feature = hand_object_feature * batch[
                        "hand_object_frame_confidence"
                    ].to(hand_object_feature.dtype)[:, None]
                # Fuse into the existing functional-frame memory slot.  This
                # keeps token count and attention normalization identical to
                # the initialized policy while the zero-start projection is
                # still inactive.
                recovery_memory = recovery_memory.clone()
                recovery_memory[:, -1, :] = (
                    recovery_memory[:, -1, :] + hand_object_feature
                )
            local_recovery_residual = self.recovery_action_adapter(
                scene, recovery_memory
            )
            recovery_action_residual = self._recovery_action_gate(
                batch
            ) * self._recovery_residual_in_world_frame(
                local_recovery_residual, batch
            )
            action = action + recovery_action_residual
        predicted_rotation = None
        if self.rotation_head is not None:
            rotation_sixd = self.rotation_head(
                torch.cat((scene, relation.mean(dim=1)), dim=-1)
            )
            predicted_rotation = self._rotation_sixd(rotation_sixd)
        return InteractionFlowOutput(
            action=action,
            recovery_action_residual=recovery_action_residual,
            anchor_a=anchor_a,
            anchor_b=anchor_b,
            predicted_flow=predicted_flow,
            predicted_rotation=predicted_rotation,
            attention_a=attention_a,
            attention_b=attention_b,
        )


def rotation_geodesic_loss(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Mean SO(3) geodesic loss in radians for rotation matrices."""
    if prediction.shape != target.shape or prediction.shape[-2:] != (3, 3):
        raise ValueError(
            f"rotation matrices must have matching [...,3,3] shapes, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    relative = prediction @ target.transpose(-1, -2)
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5)
    return torch.acos(cosine.clamp(-1.0 + 1e-6, 1.0 - 1e-6)).mean()


def unordered_flow_chamfer_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Symmetric Chamfer loss at each time without assuming anchor ordering."""
    if prediction.ndim != 4 or target.ndim != 4:
        raise ValueError("prediction and target must be [B,K,T,3]")
    if prediction.shape[0] != target.shape[0] or prediction.shape[2:] != target.shape[2:]:
        raise ValueError(
            f"flow batch/time mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}"
        )
    pred_by_time = prediction.permute(0, 2, 1, 3).reshape(-1, prediction.shape[1], 3)
    target_by_time = target.permute(0, 2, 1, 3).reshape(-1, target.shape[1], 3)
    distance = torch.cdist(pred_by_time, target_by_time).square()
    pred_to_target = distance.min(dim=2).values.mean()
    target_to_pred = distance.min(dim=1).values.mean()
    return 0.5 * (pred_to_target + target_to_pred)


def correspondence_flow_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    coverage_weight: float = 0.5,
) -> torch.Tensor:
    """Preserve each learned anchor's identity through the future trajectory.

    Learned anchors and simulator surface anchors need not share an ordering.
    We establish their correspondence once at the current frame, then supervise
    the complete trajectory with that fixed assignment.  This prevents a
    per-frame Chamfer loss from silently swapping toe/heel identities and
    discarding signed rotational motion.
    """
    if prediction.ndim != 4 or target.ndim != 4:
        raise ValueError("prediction and target must be [B,K,T,3]")
    if prediction.shape[0] != target.shape[0] or prediction.shape[2:] != target.shape[2:]:
        raise ValueError(
            f"flow batch/time mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}"
        )
    with torch.no_grad():
        current_distance = torch.cdist(
            prediction[:, :, 0], target[:, :, 0]
        )
        target_index = current_distance.argmin(dim=-1)
    gather_index = target_index[:, :, None, None].expand(
        -1, -1, target.shape[2], target.shape[3]
    )
    matched_target = torch.gather(target, dim=1, index=gather_index)
    time_weight = torch.linspace(
        0.5,
        1.5,
        prediction.shape[2],
        dtype=prediction.dtype,
        device=prediction.device,
    )
    trajectory = torch.square(prediction - matched_target).mean(dim=(1, 3))
    trajectory = (trajectory * time_weight[None]).mean()
    current_coverage = unordered_flow_chamfer_loss(
        prediction[:, :, :1], target[:, :, :1]
    )
    return trajectory + float(coverage_weight) * current_coverage


def motion_correspondence_flow_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    motion_scale: float = 10.0,
    coverage_weight: float = 0.1,
) -> torch.Tensor:
    """Supervise short-horizon anchor displacement instead of absolute shape.

    For a six-frame action chunk, geometric motion is much smaller than object
    coordinates.  An absolute point-set objective is therefore dominated by
    reconstructing the current surface.  This loss matches anchors at t=0 and
    compares their signed displacement over the action-aligned horizon.
    """
    if prediction.ndim != 4 or target.ndim != 4:
        raise ValueError("prediction and target must be [B,K,T,3]")
    if prediction.shape[0] != target.shape[0] or prediction.shape[2:] != target.shape[2:]:
        raise ValueError(
            f"flow batch/time mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}"
        )
    with torch.no_grad():
        target_index = torch.cdist(
            prediction[:, :, 0], target[:, :, 0]
        ).argmin(dim=-1)
    gather_index = target_index[:, :, None, None].expand(
        -1, -1, target.shape[2], target.shape[3]
    )
    matched_target = torch.gather(target, dim=1, index=gather_index)
    predicted_motion = prediction - prediction[:, :, :1]
    target_motion = matched_target - matched_target[:, :, :1]
    time_weight = torch.linspace(
        0.5,
        1.5,
        prediction.shape[2],
        dtype=prediction.dtype,
        device=prediction.device,
    )
    motion_error = torch.square(
        (predicted_motion - target_motion) * float(motion_scale)
    ).mean(dim=(1, 3))
    motion_error = (motion_error * time_weight[None]).mean()
    coverage = unordered_flow_chamfer_loss(
        prediction[:, :, :1], target[:, :, :1]
    )
    return motion_error + float(coverage_weight) * coverage


def rigid_flow_loss(prediction: torch.Tensor) -> torch.Tensor:
    """Penalize pairwise-distance drift across a predicted rigid-object flow."""
    if prediction.ndim != 4 or prediction.shape[-1] != 3:
        raise ValueError("prediction must be [B,K,T,3]")
    by_time = prediction.permute(0, 2, 1, 3)
    distances = torch.cdist(by_time, by_time)
    reference = distances[:, :1]
    return torch.square(distances - reference).mean()


def attention_entropy(attention: torch.Tensor) -> torch.Tensor:
    """Mean normalized point-selection entropy in [0,1]."""
    if attention.ndim != 3:
        raise ValueError("attention must be [B,K,N]")
    entropy = -(attention.clamp_min(1e-9) * attention.clamp_min(1e-9).log()).sum(-1)
    normalizer = torch.log(
        torch.as_tensor(attention.shape[-1], dtype=attention.dtype, device=attention.device)
    )
    return (entropy / normalizer.clamp_min(1e-9)).mean()
