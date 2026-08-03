from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .geometry import (
    GRIPPER_KEYPOINTS_METERS,
    keypoints_to_transform,
    normalize_object_points,
    rotation_6d_to_matrix,
    transform_keypoints_torch,
)


@dataclass
class GraspPrediction:
    transforms: torch.Tensor
    keypoints_world: torch.Tensor
    scores: torch.Tensor
    keypoints_normalized: torch.Tensor
    object_center: torch.Tensor
    object_scale: torch.Tensor
    attention: torch.Tensor | None = None
    anchors_normalized: torch.Tensor | None = None


class PointwiseObjectEncoder(nn.Module):
    """Shared encoder used by both direct and structured comparisons."""

    def __init__(
        self,
        *,
        additional_feature_dim: int = 0,
        channels: tuple[int, ...] = (64, 96, 128),
    ) -> None:
        super().__init__()
        if additional_feature_dim < 0:
            raise ValueError("additional_feature_dim must be non-negative")
        self.additional_feature_dim = int(additional_feature_dim)
        layers: list[nn.Module] = []
        input_dim = 3 + self.additional_feature_dim
        for output_dim in channels:
            layers.extend(
                [
                    nn.Linear(input_dim, output_dim),
                    nn.LayerNorm(output_dim),
                    nn.SiLU(),
                ]
            )
            input_dim = output_dim
        self.network = nn.Sequential(*layers)
        self.output_dim = int(channels[-1])

    def forward(
        self,
        points_normalized: torch.Tensor,
        additional_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if additional_features is None:
            if self.additional_feature_dim:
                raise ValueError("model expects additional per-point features")
            value = points_normalized
        else:
            if additional_features.shape[:-1] != points_normalized.shape[:-1]:
                raise ValueError("point and feature batch/point dimensions differ")
            if additional_features.shape[-1] != self.additional_feature_dim:
                raise ValueError(
                    f"expected {self.additional_feature_dim} feature channels, got "
                    f"{additional_features.shape[-1]}"
                )
            value = torch.cat((points_normalized, additional_features), dim=-1)
        point_features = self.network(value)
        pooled = torch.cat(
            (
                torch.amax(point_features, dim=1),
                torch.mean(point_features, dim=1),
            ),
            dim=-1,
        )
        return point_features, pooled


class _GraspModelBase(nn.Module):
    def __init__(
        self,
        *,
        num_candidates: int,
        additional_feature_dim: int,
        channels: tuple[int, ...],
    ) -> None:
        super().__init__()
        if num_candidates <= 0:
            raise ValueError("num_candidates must be positive")
        self.num_candidates = int(num_candidates)
        self.encoder = PointwiseObjectEncoder(
            additional_feature_dim=additional_feature_dim,
            channels=channels,
        )
        self.arm_embedding = nn.Embedding(2, 16)
        self.register_buffer(
            "canonical_keypoints",
            torch.as_tensor(GRIPPER_KEYPOINTS_METERS, dtype=torch.float32),
        )

    def _encode(
        self,
        points_world: torch.Tensor,
        active_arm_right: torch.Tensor,
        additional_features: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        points_normalized, center, scale = normalize_object_points(points_world)
        point_features, pooled = self.encoder(points_normalized, additional_features)
        arm_index = active_arm_right.reshape(-1).to(dtype=torch.long)
        arm_context = self.arm_embedding(arm_index)
        return points_normalized, point_features, pooled, arm_context, center, scale


class DirectGraspRegressor(_GraspModelBase):
    """Global-pooling SE(3) baseline with the same point encoder."""

    def __init__(
        self,
        *,
        num_candidates: int = 1,
        additional_feature_dim: int = 0,
        channels: tuple[int, ...] = (64, 96, 128),
    ) -> None:
        super().__init__(
            num_candidates=num_candidates,
            additional_feature_dim=additional_feature_dim,
            channels=channels,
        )
        input_dim = 2 * self.encoder.output_dim + 16
        self.head = nn.Sequential(
            nn.Linear(input_dim, 192),
            nn.LayerNorm(192),
            nn.SiLU(),
            nn.Linear(192, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, self.num_candidates * 10),
        )

    def forward(
        self,
        points_world: torch.Tensor,
        active_arm_right: torch.Tensor,
        additional_features: torch.Tensor | None = None,
    ) -> GraspPrediction:
        (
            _,
            _,
            pooled,
            arm_context,
            center,
            scale,
        ) = self._encode(points_world, active_arm_right, additional_features)
        output = self.head(torch.cat((pooled, arm_context), dim=-1))
        output = output.reshape(-1, self.num_candidates, 10)
        translation_normalized = output[..., :3]
        rotation = rotation_6d_to_matrix(output[..., 3:9])
        scores = output[..., 9]
        translation_world = translation_normalized * scale[:, None, :] + center[:, None, :]
        keypoints_world = transform_keypoints_torch(
            rotation, translation_world, self.canonical_keypoints
        )
        transforms = torch.zeros(
            (*rotation.shape[:-2], 4, 4),
            dtype=rotation.dtype,
            device=rotation.device,
        )
        transforms[..., :3, :3] = rotation
        transforms[..., :3, 3] = translation_world
        transforms[..., 3, 3] = 1.0
        keypoints_normalized = (
            keypoints_world - center[:, None, None, :]
        ) / scale[:, None, None, :]
        return GraspPrediction(
            transforms=transforms,
            keypoints_world=keypoints_world,
            scores=scores,
            keypoints_normalized=keypoints_normalized,
            object_center=center,
            object_scale=scale,
        )


class KeypointTransportGraspRegressor(_GraspModelBase):
    """Transport object features into explicit direction-aware gripper frames."""

    def __init__(
        self,
        *,
        num_candidates: int = 2,
        additional_feature_dim: int = 0,
        channels: tuple[int, ...] = (64, 96, 128),
        maximum_offset_normalized: float = 1.5,
        attention_temperature: float = 0.20,
    ) -> None:
        super().__init__(
            num_candidates=num_candidates,
            additional_feature_dim=additional_feature_dim,
            channels=channels,
        )
        self.num_keypoints = int(self.canonical_keypoints.shape[0])
        self.maximum_offset_normalized = float(maximum_offset_normalized)
        self.attention_temperature = float(attention_temperature)
        feature_dim = self.encoder.output_dim
        self.queries = nn.Parameter(
            torch.randn(self.num_candidates, self.num_keypoints, feature_dim) * 0.02
        )
        self.arm_to_query = nn.Linear(16, feature_dim)
        self.key_projection = nn.Linear(feature_dim, feature_dim, bias=False)
        offset_input_dim = feature_dim + 2 * feature_dim + feature_dim
        self.offset_head = nn.Sequential(
            nn.Linear(offset_input_dim, 192),
            nn.LayerNorm(192),
            nn.SiLU(),
            nn.Linear(192, 96),
            nn.LayerNorm(96),
            nn.SiLU(),
            nn.Linear(96, 3),
            nn.Tanh(),
        )
        self.score_head = nn.Sequential(
            nn.Linear(4 * feature_dim + 16, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        points_world: torch.Tensor,
        active_arm_right: torch.Tensor,
        additional_features: torch.Tensor | None = None,
    ) -> GraspPrediction:
        (
            points_normalized,
            point_features,
            pooled,
            arm_context,
            center,
            scale,
        ) = self._encode(points_world, active_arm_right, additional_features)
        batch_size = points_world.shape[0]
        queries = self.queries[None, ...] + self.arm_to_query(arm_context)[
            :, None, None, :
        ]
        keys = self.key_projection(point_features)
        logits = torch.einsum("bmkd,bnd->bmkn", queries, keys)
        logits = logits / (keys.shape[-1] ** 0.5 * self.attention_temperature)
        attention = torch.softmax(logits, dim=-1)
        anchors = torch.einsum("bmkn,bnd->bmkd", attention, points_normalized)
        point_context = torch.einsum("bmkn,bnd->bmkd", attention, point_features)
        pooled_expanded = pooled[:, None, None, :].expand(
            -1, self.num_candidates, self.num_keypoints, -1
        )
        offset_input = torch.cat(
            (point_context, pooled_expanded, queries), dim=-1
        )
        offsets = self.offset_head(offset_input) * self.maximum_offset_normalized
        keypoints_normalized = anchors + offsets
        keypoints_world = (
            keypoints_normalized * scale[:, None, None, :]
            + center[:, None, None, :]
        )
        transforms = keypoints_to_transform(keypoints_world, self.canonical_keypoints)
        candidate_context = point_context.mean(dim=2)
        candidate_query = queries.mean(dim=2)
        score_input = torch.cat(
            (
                pooled[:, None, :].expand(-1, self.num_candidates, -1),
                arm_context[:, None, :].expand(-1, self.num_candidates, -1),
                candidate_context,
                candidate_query,
            ),
            dim=-1,
        )
        scores = self.score_head(score_input).squeeze(-1)
        return GraspPrediction(
            transforms=transforms,
            keypoints_world=keypoints_world,
            scores=scores,
            keypoints_normalized=keypoints_normalized,
            object_center=center,
            object_scale=scale,
            attention=attention,
            anchors_normalized=anchors,
        )


class NdfDirectionalGraspRegressor(_GraspModelBase):
    """Use NDF's equivariant vector explicitly as the grasp longitudinal axis.

    Two candidates use opposite vector signs.  Gravity fixes the remaining
    roll convention, leaving the learned network to predict translation and
    rank the two physically distinct approach modes.
    """

    def __init__(
        self,
        *,
        num_candidates: int = 2,
        additional_feature_dim: int = 259,
        channels: tuple[int, ...] = (64, 96, 128),
    ) -> None:
        if num_candidates != 2:
            raise ValueError("ndf_directional requires exactly two sign candidates")
        if additional_feature_dim < 3:
            raise ValueError("ndf_directional requires a final 3D vector feature")
        self.ndf_input_feature_dim = int(additional_feature_dim)
        # NDF supplies the geometric axis explicitly below.  The translation
        # branch intentionally stays raw-XYZ-only so 256 scalar channels cannot
        # swamp 29 object-disjoint training examples.
        super().__init__(
            num_candidates=num_candidates,
            additional_feature_dim=0,
            channels=channels,
        )
        feature_dim = self.encoder.output_dim
        context_dim = 2 * feature_dim + 16
        self.direction_attention = nn.Linear(feature_dim, 1)
        self.translation_head = nn.Sequential(
            nn.Linear(context_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 3),
        )
        self.score_head = nn.Sequential(
            nn.Linear(context_dim + 3, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 2),
        )

    def _grasp_axis(
        self,
        points_normalized: torch.Tensor,
        point_features: torch.Tensor,
        additional_features: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return one directed, equivariant axis for each object cloud."""
        if additional_features is None or additional_features.shape[-1] < 3:
            raise ValueError("ndf_directional needs NDF features ending in a 3D vector")
        weights = torch.softmax(
            self.direction_attention(point_features).squeeze(-1), dim=-1
        )
        ndf_vector = additional_features[..., -3:]
        direction = torch.sum(weights[..., None] * ndf_vector, dim=1)
        return torch.nn.functional.normalize(direction, dim=-1, eps=1e-6)

    def forward(
        self,
        points_world: torch.Tensor,
        active_arm_right: torch.Tensor,
        additional_features: torch.Tensor | None = None,
    ) -> GraspPrediction:
        points_normalized, center, scale = normalize_object_points(points_world)
        point_features, pooled = self.encoder(points_normalized, None)
        arm_index = active_arm_right.reshape(-1).to(dtype=torch.long)
        arm_context = self.arm_embedding(arm_index)
        direction = self._grasp_axis(
            points_normalized, point_features, additional_features
        )
        z_axis = torch.stack((direction, -direction), dim=1)

        gravity = torch.zeros_like(z_axis)
        gravity[..., 2] = 1.0
        x_raw = gravity - torch.sum(gravity * z_axis, dim=-1, keepdim=True) * z_axis
        fallback = torch.zeros_like(z_axis)
        fallback[..., 0] = 1.0
        fallback = fallback - torch.sum(fallback * z_axis, dim=-1, keepdim=True) * z_axis
        use_fallback = torch.linalg.norm(x_raw, dim=-1, keepdim=True) < 0.1
        x_raw = torch.where(use_fallback, fallback, x_raw)
        # RoboTwin's shoe grasps consistently point the end-link longitudinal
        # axis away from gravity after projecting it into the grasp plane.
        x_axis = -torch.nn.functional.normalize(x_raw, dim=-1, eps=1e-6)
        y_axis = torch.cross(z_axis, x_axis, dim=-1)
        rotation = torch.stack((x_axis, y_axis, z_axis), dim=-1)

        context = torch.cat((pooled, arm_context), dim=-1)
        translation_normalized = self.translation_head(context)
        translation_world = translation_normalized * scale + center
        translation_candidates = translation_world[:, None, :].expand(-1, 2, -1)
        keypoints_world = transform_keypoints_torch(
            rotation, translation_candidates, self.canonical_keypoints
        )
        keypoints_normalized = (
            keypoints_world - center[:, None, None, :]
        ) / scale[:, None, None, :]
        transforms = torch.zeros(
            (len(points_world), 2, 4, 4),
            dtype=points_world.dtype,
            device=points_world.device,
        )
        transforms[..., :3, :3] = rotation
        transforms[..., :3, 3] = translation_candidates
        transforms[..., 3, 3] = 1.0
        scores = self.score_head(torch.cat((context, direction), dim=-1))
        return GraspPrediction(
            transforms=transforms,
            keypoints_world=keypoints_world,
            scores=scores,
            keypoints_normalized=keypoints_normalized,
            object_center=center,
            object_scale=scale,
        )


class PcaDirectionalGraspRegressor(NdfDirectionalGraspRegressor):
    """Architecture-matched raw-geometry control for NDF direction.

    The principal axis is computed from centered XYZ. Its otherwise arbitrary
    sign is fixed with the third moment, which is rotation invariant and uses
    shoe asymmetry (toe versus heel) without labels or simulator state. The
    model still emits both signs, exactly like ``ndf_directional``.
    """

    def __init__(
        self,
        *,
        num_candidates: int = 2,
        additional_feature_dim: int = 0,
        channels: tuple[int, ...] = (64, 96, 128),
    ) -> None:
        del additional_feature_dim
        # Construct the identical parameterized heads, including the unused
        # attention layer, so paired seeds have matching initialization/order.
        super().__init__(
            num_candidates=num_candidates,
            additional_feature_dim=3,
            channels=channels,
        )
        self.ndf_input_feature_dim = 0

    def _grasp_axis(
        self,
        points_normalized: torch.Tensor,
        point_features: torch.Tensor,
        additional_features: torch.Tensor | None,
    ) -> torch.Tensor:
        del point_features, additional_features
        centered = points_normalized - torch.mean(
            points_normalized, dim=1, keepdim=True
        )
        covariance = centered.transpose(-1, -2) @ centered
        covariance = covariance / max(int(centered.shape[1]), 1)
        _, eigenvectors = torch.linalg.eigh(covariance)
        direction = eigenvectors[..., -1]
        projection = torch.sum(centered * direction[:, None, :], dim=-1)
        third_moment = torch.mean(projection.pow(3), dim=-1, keepdim=True)
        sign = torch.where(third_moment < 0.0, -1.0, 1.0)
        direction = direction * sign
        return torch.nn.functional.normalize(direction, dim=-1, eps=1e-6)


MODEL_TYPES = {
    "direct": DirectGraspRegressor,
    "keypoint_transport": KeypointTransportGraspRegressor,
    "ndf_directional": NdfDirectionalGraspRegressor,
    "pca_directional": PcaDirectionalGraspRegressor,
}


def build_model(name: str, **kwargs) -> _GraspModelBase:
    try:
        model_type = MODEL_TYPES[str(name)]
    except KeyError as exc:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODEL_TYPES)}") from exc
    return model_type(**kwargs)
