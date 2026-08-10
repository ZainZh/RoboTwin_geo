from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from termcolor import cprint
import copy
import time
import pdb

# import pytorch3d.ops as torch3d_ops

from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy_3d.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.model_util import print_params
from diffusion_policy_3d.model.vision.pointnet_extractor import (
    DP3Encoder,
    PointNetEncoderXYZRGB,
)


def _se3_frame_invariants(relation: torch.Tensor) -> torch.Tensor:
    """Return translation norm and rotation geodesic angle from a frame9 token."""

    if relation.shape[-1] < 9:
        raise ValueError("SE(3) frame invariants require at least 9 values")
    translation_norm = torch.linalg.vector_norm(
        relation[..., :3], dim=-1, keepdim=True
    )
    first = F.normalize(relation[..., 3:6], dim=-1, eps=1.0e-6)
    second_raw = relation[..., 6:9]
    second = F.normalize(
        second_raw - first * torch.sum(first * second_raw, dim=-1, keepdim=True),
        dim=-1,
        eps=1.0e-6,
    )
    third = torch.cross(first, second, dim=-1)
    trace = first[..., 0] + second[..., 1] + third[..., 2]
    cosine = torch.clamp(
        (trace - 1.0) * 0.5, -1.0 + 1.0e-6, 1.0 - 1.0e-6
    )
    rotation_angle = torch.acos(cosine).unsqueeze(-1)
    return torch.cat((translation_norm, rotation_angle), dim=-1)


def _frame9_rotation_rotvec(relation: torch.Tensor) -> torch.Tensor:
    """Convert the two-axis frame9 rotation into a differentiable rotvec."""

    if relation.shape[-1] < 9:
        raise ValueError("frame9 rotation requires at least 9 values")
    first_raw = relation[..., 3:6]
    second_raw = relation[..., 6:9]
    first_norm = torch.linalg.vector_norm(first_raw, dim=-1, keepdim=True)
    first = F.normalize(first_raw, dim=-1, eps=1.0e-6)
    second_orthogonal = second_raw - first * torch.sum(
        first * second_raw, dim=-1, keepdim=True
    )
    second_norm = torch.linalg.vector_norm(
        second_orthogonal, dim=-1, keepdim=True
    )
    second = F.normalize(second_orthogonal, dim=-1, eps=1.0e-6)
    third = torch.cross(first, second, dim=-1)
    trace = first[..., 0] + second[..., 1] + third[..., 2]
    cosine = torch.clamp((trace - 1.0) * 0.5, -1.0 + 1.0e-6, 1.0)
    angle = torch.acos(cosine)
    skew = torch.stack(
        (
            second[..., 2] - third[..., 1],
            third[..., 0] - first[..., 2],
            first[..., 1] - second[..., 0],
        ),
        dim=-1,
    )
    sine = torch.sin(angle)
    coefficient = torch.where(
        torch.abs(sine) > 1.0e-4,
        angle / (2.0 * sine),
        torch.full_like(angle, 0.5),
    )
    rotvec = skew * coefficient.unsqueeze(-1)
    valid = (first_norm[..., 0] > 0.5) & (second_norm[..., 0] > 0.5)
    return torch.where(valid.unsqueeze(-1), rotvec, torch.zeros_like(rotvec))


class ZeroInitGeometryAdapter(nn.Module):
    """Map geometric point-flow tokens into a safe additive policy condition."""

    def __init__(
        self,
        in_channels: int,
        geometry_dim: int,
        output_dim: int,
        dropout: float = 0.0,
        force_zero: bool = False,
    ):
        super().__init__()
        self.encoder = PointNetEncoderXYZRGB(
            in_channels=int(in_channels),
            out_channels=int(geometry_dim),
            use_layernorm=True,
            final_norm="layernorm",
        )
        self.projection = nn.Linear(int(geometry_dim), int(output_dim))
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        self.dropout = float(dropout)
        self.force_zero = bool(force_zero)
        self.output_dim = int(output_dim)

    def forward(
        self,
        points: torch.Tensor,
        confidence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.force_zero:
            return torch.zeros(
                (points.shape[0], self.output_dim),
                dtype=points.dtype,
                device=points.device,
            )
        residual = self.projection(self.encoder(points))
        if confidence is not None:
            residual = residual * confidence.reshape(confidence.shape[0], -1).mean(
                dim=-1, keepdim=True
            ).clamp(0.0, 1.0)
        if self.training and self.dropout > 0.0:
            keep = (
                torch.rand(
                    (residual.shape[0], 1),
                    dtype=residual.dtype,
                    device=residual.device,
                )
                >= self.dropout
            )
            residual = residual * keep
        return residual


class ZeroInitGlobalGeometryAdapter(nn.Module):
    """Safe global SE(3)-relation route paired with the local point adapter."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.0,
        force_zero: bool = False,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.Mish(),
        )
        self.projection = nn.Linear(int(hidden_dim), int(output_dim))
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        self.dropout = float(dropout)
        self.force_zero = bool(force_zero)
        self.output_dim = int(output_dim)

    def forward(
        self,
        relation: torch.Tensor,
        confidence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.force_zero:
            return torch.zeros(
                (relation.shape[0], self.output_dim),
                dtype=relation.dtype,
                device=relation.device,
            )
        residual = self.projection(self.encoder(relation))
        if confidence is not None:
            residual = residual * confidence.reshape(confidence.shape[0], -1).mean(
                dim=-1, keepdim=True
            ).clamp(0.0, 1.0)
        if self.training and self.dropout > 0.0:
            keep = (
                torch.rand(
                    (residual.shape[0], 1),
                    dtype=residual.dtype,
                    device=residual.device,
                )
                >= self.dropout
            )
            residual = residual * keep
        return residual


class AnchorFlowTranslationHead(nn.Module):
    """Decode a six-step EEF chunk from explicit per-anchor rigid flow."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        global_dim: int,
        confidence_dim: int,
        output_dim: int,
    ):
        super().__init__()
        hidden = int(hidden_dim)
        self.point_mlp = nn.Sequential(
            nn.Linear(int(in_channels), hidden),
            nn.LayerNorm(hidden),
            nn.Mish(),
            nn.Linear(hidden, hidden),
            nn.Mish(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2 * hidden + int(global_dim) + int(confidence_dim), hidden),
            nn.LayerNorm(hidden),
            nn.Mish(),
            nn.Linear(hidden, int(output_dim)),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        global_relation: torch.Tensor,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.point_mlp(tokens)
        pooled = torch.cat((encoded.mean(dim=1), encoded.max(dim=1).values), dim=-1)
        return self.decoder(
            torch.cat((pooled, global_relation, confidence), dim=-1)
        )


class FlowConsistentAnchorTranslationHead(nn.Module):
    """Factor a chunk into a geometric endpoint and zero-sum timing residual."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        global_dim: int,
        confidence_dim: int,
        action_steps: int,
    ):
        super().__init__()
        self.action_steps = int(action_steps)
        input_dim = 4 * int(in_channels) + int(global_dim) + int(confidence_dim)
        hidden = int(hidden_dim)
        self.trunk = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.Mish(),
            nn.Linear(hidden, hidden),
            nn.Mish(),
        )
        self.endpoint = nn.Linear(hidden, 2 * 3)
        self.temporal_residual = nn.Linear(hidden, self.action_steps * 2 * 3)

    def forward(
        self,
        tokens: torch.Tensor,
        global_relation: torch.Tensor,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        raw_summary = torch.cat(
            (
                tokens.mean(dim=1),
                tokens.std(dim=1, unbiased=False),
                tokens.min(dim=1).values,
                tokens.max(dim=1).values,
                global_relation,
                confidence,
            ),
            dim=-1,
        )
        feature = self.trunk(raw_summary)
        endpoint = self.endpoint(feature).reshape(-1, 2, 3)
        residual = self.temporal_residual(feature).reshape(
            -1, self.action_steps, 2, 3
        )
        residual = residual - residual.mean(dim=1, keepdim=True)
        return residual + endpoint[:, None] / float(self.action_steps)


class FutureGeometryHead(nn.Module):
    """Predict a future object-flow endpoint from the shared policy context."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.Mish(),
            nn.Linear(int(hidden_dim), int(output_dim)),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.network(context)


class BinaryGripperHead(nn.Module):
    """Predict future open/closed commands from the shared policy context."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.Mish(),
            nn.Linear(int(hidden_dim), int(output_dim)),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.network(context)


class TaskAlignedRotationHead(nn.Module):
    """Predict active-arm rotvec chunks from shared and signed TAGRT context."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.Mish(),
            nn.Linear(int(hidden_dim), int(output_dim)),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.network(context)


def _recovery_action_channel_weights(
    is_recovery: torch.Tensor,
    active_arm_right: torch.Tensor,
    *,
    horizon: int,
    action_dim: int,
    rotation_weight: float,
) -> torch.Tensor:
    """Mask synthetic recovery supervision to the active-arm rotation axes."""

    if int(action_dim) != 14:
        raise ValueError("rotation-only recovery currently requires 14D bimanual EEF actions")
    recovery = is_recovery.reshape(-1) >= 0.5
    active_right = active_arm_right.reshape(-1) >= 0.5
    if recovery.shape != active_right.shape:
        raise ValueError("recovery and active-arm labels must have matching shapes")
    weights = torch.ones(
        (len(recovery), int(horizon), int(action_dim)),
        dtype=is_recovery.dtype,
        device=is_recovery.device,
    )
    if not bool(torch.any(recovery)):
        return weights
    weights[recovery] = 0.0
    left_recovery = recovery & ~active_right
    right_recovery = recovery & active_right
    weights[left_recovery, :, 3:6] = float(rotation_weight)
    weights[right_recovery, :, 10:13] = float(rotation_weight)
    return weights


def _replace_closed_arm_rotations(
    action: torch.Tensor,
    rotation_prediction: torch.Tensor,
    closed_arm_mask: torch.Tensor,
) -> torch.Tensor:
    """Replace only closed-arm rotvecs while preserving all other DP3 actions."""

    if action.ndim != 3 or int(action.shape[-1]) != 14:
        raise ValueError("rotation replacement requires [B, T, 14] actions")
    expected = (int(action.shape[0]), int(action.shape[1]), 2, 3)
    if tuple(rotation_prediction.shape) != expected:
        raise ValueError(
            "rotation prediction shape mismatch: "
            f"expected {expected}, got {tuple(rotation_prediction.shape)}"
        )
    if tuple(closed_arm_mask.shape) != (int(action.shape[0]), 2):
        raise ValueError("closed-arm mask must have shape [B, 2]")
    result = action.clone()
    left_mask = closed_arm_mask[:, None, 0, None]
    right_mask = closed_arm_mask[:, None, 1, None]
    result[..., 3:6] = torch.where(
        left_mask, rotation_prediction[..., 0, :], result[..., 3:6]
    )
    result[..., 10:13] = torch.where(
        right_mask, rotation_prediction[..., 1, :], result[..., 10:13]
    )
    return result


def _replace_closed_arm_translations(
    action: torch.Tensor,
    translation_prediction: torch.Tensor,
    closed_arm_mask: torch.Tensor,
) -> torch.Tensor:
    """Replace only gated active-arm translations and preserve other channels."""

    if action.ndim != 3 or int(action.shape[-1]) != 14:
        raise ValueError("translation replacement requires [B, T, 14] actions")
    expected = (int(action.shape[0]), int(action.shape[1]), 2, 3)
    if tuple(translation_prediction.shape) != expected:
        raise ValueError(
            "translation prediction shape mismatch: "
            f"expected {expected}, got {tuple(translation_prediction.shape)}"
        )
    if tuple(closed_arm_mask.shape) != (int(action.shape[0]), 2):
        raise ValueError("closed-arm mask must have shape [B, 2]")
    result = action.clone()
    result[..., :3] = torch.where(
        closed_arm_mask[:, None, 0, None],
        translation_prediction[..., 0, :],
        result[..., :3],
    )
    result[..., 7:10] = torch.where(
        closed_arm_mask[:, None, 1, None],
        translation_prediction[..., 1, :],
        result[..., 7:10],
    )
    return result


def _active_translation_action_loss(
    prediction: torch.Tensor,
    target_action: torch.Tensor,
    active_arm_right: torch.Tensor,
    valid_mask: torch.Tensor,
    endpoint_weight: float = 0.0,
) -> torch.Tensor:
    """Supervise only the gated active-arm Cartesian compensation."""

    if prediction.ndim != 4 or tuple(prediction.shape[-2:]) != (2, 3):
        raise ValueError("translation prediction must have shape [B, T, 2, 3]")
    if target_action.ndim != 3 or int(target_action.shape[-1]) != 14:
        raise ValueError("translation target requires [B, T, 14] actions")
    target_translation = torch.stack(
        (target_action[..., :3], target_action[..., 7:10]), dim=2
    )
    active_right = active_arm_right.reshape(-1).to(
        device=prediction.device, dtype=torch.long
    )
    batch_indices = torch.arange(prediction.shape[0], device=prediction.device)
    selected_prediction = prediction[batch_indices, :, active_right, :]
    selected_target = target_translation[batch_indices, :, active_right, :]
    per_sample = F.mse_loss(
        selected_prediction, selected_target, reduction="none"
    ).mean(dim=(1, 2))
    if float(endpoint_weight) < 0.0:
        raise ValueError("translation endpoint weight must be non-negative")
    if float(endpoint_weight) > 0.0:
        endpoint_loss = F.mse_loss(
            selected_prediction.sum(dim=1),
            selected_target.sum(dim=1),
            reduction="none",
        ).mean(dim=-1)
        per_sample = per_sample + float(endpoint_weight) * endpoint_loss
    valid = valid_mask.reshape(-1).to(
        device=prediction.device, dtype=prediction.dtype
    )
    if int(valid.shape[0]) != int(prediction.shape[0]):
        raise ValueError("translation validity labels must match batch size")
    return torch.sum(per_sample * valid) / torch.sum(valid).clamp_min(1.0)


def _active_rotation_action_loss(
    prediction: torch.Tensor,
    target_action: torch.Tensor,
    active_arm_right: torch.Tensor,
    is_recovery: torch.Tensor | None = None,
    recovery_weight: float = 1.0,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Supervise the task-active arm without leaking simulator pose at inference."""

    if prediction.ndim != 4 or tuple(prediction.shape[-2:]) != (2, 3):
        raise ValueError("rotation prediction must have shape [B, T, 2, 3]")
    if target_action.ndim != 3 or int(target_action.shape[-1]) != 14:
        raise ValueError("rotation target requires [B, T, 14] actions")
    if tuple(prediction.shape[:2]) != tuple(target_action.shape[:2]):
        raise ValueError("rotation prediction and action target horizons must match")
    active_right = active_arm_right.reshape(-1).to(
        device=prediction.device, dtype=torch.bool
    )
    if int(active_right.shape[0]) != int(prediction.shape[0]):
        raise ValueError("active-arm labels must match the batch size")
    target_rotation = torch.stack(
        (target_action[..., 3:6], target_action[..., 10:13]), dim=2
    )
    batch_indices = torch.arange(prediction.shape[0], device=prediction.device)
    arm_indices = active_right.to(dtype=torch.long)
    selected_prediction = prediction[batch_indices, :, arm_indices, :]
    selected_target = target_rotation[batch_indices, :, arm_indices, :]
    per_sample = F.mse_loss(
        selected_prediction, selected_target, reduction="none"
    ).mean(dim=(1, 2))
    sample_weight = torch.ones_like(per_sample)
    if valid_mask is not None:
        valid = valid_mask.reshape(-1).to(
            device=prediction.device, dtype=prediction.dtype
        )
        if int(valid.shape[0]) != int(prediction.shape[0]):
            raise ValueError("validity labels must match the batch size")
        sample_weight = sample_weight * valid.clamp(0.0, 1.0)
    if is_recovery is None:
        return torch.sum(per_sample * sample_weight) / torch.sum(
            sample_weight
        ).clamp_min(1.0)
    if float(recovery_weight) <= 0.0:
        raise ValueError("recovery_weight must be positive")
    recovery = is_recovery.reshape(-1).to(
        device=prediction.device, dtype=prediction.dtype
    )
    if int(recovery.shape[0]) != int(prediction.shape[0]):
        raise ValueError("recovery labels must match the batch size")
    sample_weight = sample_weight * (
        1.0 + (float(recovery_weight) - 1.0) * recovery.clamp(0.0, 1.0)
    )
    return torch.sum(per_sample * sample_weight) / torch.sum(
        sample_weight
    ).clamp_min(1.0)


class DP3(BasePolicy):

    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        horizon,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        obs_as_global_cond=True,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        condition_type="film",
        use_down_condition=True,
        use_mid_condition=True,
        use_up_condition=True,
        encoder_output_dim=256,
        crop_shape=None,
        use_pc_color=False,
        pointnet_type="pointnet",
        pointcloud_encoder_cfg=None,
        geometry_key=None,
        geometry_confidence_key=None,
        geometry_encoder_output_dim=64,
        geometry_dropout=0.0,
        geometry_force_zero=False,
        global_geometry_key=None,
        global_geometry_confidence_key=None,
        global_geometry_hidden_dim=64,
        global_geometry_dropout=0.0,
        global_geometry_force_zero=False,
        aux_geometry_key=None,
        aux_geometry_loss_weight=0.0,
        aux_geometry_hidden_dim=256,
        aux_geometry_target_mode="endpoint_displacement",
        binary_gripper_indices=(),
        binary_gripper_hidden_dim=128,
        binary_gripper_loss_weight=0.0,
        binary_gripper_positive_weight=3.0,
        binary_gripper_retention_positive_weight=None,
        binary_gripper_threshold=0.5,
        binary_gripper_geometry_key=None,
        binary_gripper_geometry_confidence_key=None,
        binary_gripper_retention_state_key=None,
        binary_gripper_retention_state_indices=(),
        binary_gripper_retention_closed_threshold=0.0,
        binary_gripper_retention_current_only=False,
        binary_gripper_retention_invariant=False,
        recovery_rotation_only_loss=False,
        recovery_rotation_loss_weight=4.0,
        rotation_action_head_hidden_dim=0,
        rotation_action_head_loss_weight=0.0,
        rotation_action_head_recovery_weight=1.0,
        rotation_action_head_axis_aligned=False,
        rotation_action_head_geometry_only=False,
        rotation_action_head_gated=False,
        rotation_action_gate_geometry_only=None,
        rotation_action_gate_loss_weight=0.0,
        rotation_action_gate_positive_weight=8.0,
        rotation_action_gate_threshold=0.5,
        rotation_action_support_min_deg=0.0,
        rotation_action_support_max_deg=180.0,
        translation_recovery_support_min_m=0.0,
        translation_recovery_support_max_m=0.0,
        translation_recovery_rotation_max_deg=180.0,
        translation_recovery_geometry_scale_m=1.0,
        rotation_action_translation_head_hidden_dim=0,
        rotation_action_translation_geometry_only=False,
        rotation_action_translation_current_only=False,
        rotation_action_translation_translation_only=False,
        rotation_action_translation_apply_on_high_rotation=False,
        rotation_action_translation_anchor_flow_key=None,
        rotation_action_translation_flow_consistent=False,
        rotation_action_translation_endpoint_loss_weight=0.0,
        rotation_action_translation_loss_weight=0.0,
        rotation_action_head_geometry_key=None,
        rotation_action_head_geometry_confidence_key=None,
        # parameters passed to step
        **kwargs,
    ):
        super().__init__()

        self.condition_type = condition_type

        # parse shape_meta
        action_shape = shape_meta["action"]["shape"]
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2:  # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")

        obs_shape_meta = shape_meta["obs"]
        obs_dict = dict_apply(obs_shape_meta, lambda x: x["shape"])

        self.geometry_key = geometry_key
        self.geometry_confidence_key = geometry_confidence_key
        self.global_geometry_key = global_geometry_key
        self.global_geometry_confidence_key = global_geometry_confidence_key
        self.aux_geometry_key = aux_geometry_key
        self.aux_geometry_loss_weight = float(aux_geometry_loss_weight)
        self.aux_geometry_target_mode = str(aux_geometry_target_mode)
        self.binary_gripper_indices = tuple(
            int(index) for index in (binary_gripper_indices or ())
        )
        self.binary_gripper_loss_weight = float(binary_gripper_loss_weight)
        self.binary_gripper_positive_weight = float(
            binary_gripper_positive_weight
        )
        self.binary_gripper_retention_positive_weight = float(
            binary_gripper_positive_weight
            if binary_gripper_retention_positive_weight is None
            else binary_gripper_retention_positive_weight
        )
        if self.binary_gripper_positive_weight <= 0.0:
            raise ValueError("binary_gripper_positive_weight must be positive")
        if self.binary_gripper_retention_positive_weight <= 0.0:
            raise ValueError(
                "binary_gripper_retention_positive_weight must be positive"
            )
        self.binary_gripper_threshold = float(binary_gripper_threshold)
        self.binary_gripper_geometry_key = binary_gripper_geometry_key
        self.binary_gripper_geometry_confidence_key = (
            binary_gripper_geometry_confidence_key
        )
        self.binary_gripper_retention_state_key = binary_gripper_retention_state_key
        self.binary_gripper_retention_state_indices = tuple(
            int(index) for index in (binary_gripper_retention_state_indices or ())
        )
        self.binary_gripper_retention_closed_threshold = float(
            binary_gripper_retention_closed_threshold
        )
        self.binary_gripper_retention_current_only = bool(
            binary_gripper_retention_current_only
        )
        self.binary_gripper_retention_invariant = bool(
            binary_gripper_retention_invariant
        )
        self.recovery_rotation_only_loss = bool(recovery_rotation_only_loss)
        self.recovery_rotation_loss_weight = float(
            recovery_rotation_loss_weight
        )
        if self.recovery_rotation_loss_weight <= 0.0:
            raise ValueError("recovery_rotation_loss_weight must be positive")
        self.rotation_action_head_hidden_dim = int(rotation_action_head_hidden_dim)
        self.rotation_action_head_loss_weight = float(rotation_action_head_loss_weight)
        self.rotation_action_head_recovery_weight = float(
            rotation_action_head_recovery_weight
        )
        self.rotation_action_head_axis_aligned = bool(
            rotation_action_head_axis_aligned
        )
        self.rotation_action_head_geometry_only = bool(
            rotation_action_head_geometry_only
        )
        self.rotation_action_head_gated = bool(rotation_action_head_gated)
        self.rotation_action_gate_geometry_only = bool(
            self.rotation_action_head_geometry_only
            if rotation_action_gate_geometry_only is None
            else rotation_action_gate_geometry_only
        )
        self.rotation_action_gate_loss_weight = float(
            rotation_action_gate_loss_weight
        )
        self.rotation_action_gate_positive_weight = float(
            rotation_action_gate_positive_weight
        )
        self.rotation_action_gate_threshold = float(rotation_action_gate_threshold)
        self.rotation_action_support_min_deg = float(
            rotation_action_support_min_deg
        )
        self.rotation_action_support_max_deg = float(
            rotation_action_support_max_deg
        )
        self.translation_recovery_support_min_m = float(
            translation_recovery_support_min_m
        )
        self.translation_recovery_support_max_m = float(
            translation_recovery_support_max_m
        )
        self.translation_recovery_rotation_max_deg = float(
            translation_recovery_rotation_max_deg
        )
        self.translation_recovery_geometry_scale_m = float(
            translation_recovery_geometry_scale_m
        )
        self.rotation_action_translation_head_hidden_dim = int(
            rotation_action_translation_head_hidden_dim
        )
        self.rotation_action_translation_geometry_only = bool(
            rotation_action_translation_geometry_only
        )
        self.rotation_action_translation_current_only = bool(
            rotation_action_translation_current_only
        )
        self.rotation_action_translation_translation_only = bool(
            rotation_action_translation_translation_only
        )
        self.rotation_action_translation_apply_on_high_rotation = bool(
            rotation_action_translation_apply_on_high_rotation
        )
        self.rotation_action_translation_anchor_flow_key = (
            rotation_action_translation_anchor_flow_key
        )
        self.rotation_action_translation_flow_consistent = bool(
            rotation_action_translation_flow_consistent
        )
        self.rotation_action_translation_endpoint_loss_weight = float(
            rotation_action_translation_endpoint_loss_weight
        )
        self.rotation_action_translation_loss_weight = float(
            rotation_action_translation_loss_weight
        )
        self.rotation_action_head_geometry_key = rotation_action_head_geometry_key
        self.rotation_action_head_geometry_confidence_key = (
            rotation_action_head_geometry_confidence_key
        )
        if self.rotation_action_head_loss_weight < 0.0:
            raise ValueError("rotation_action_head_loss_weight must be non-negative")
        if self.rotation_action_head_recovery_weight <= 0.0:
            raise ValueError("rotation_action_head_recovery_weight must be positive")
        if self.rotation_action_gate_loss_weight < 0.0:
            raise ValueError("rotation_action_gate_loss_weight must be non-negative")
        if self.rotation_action_gate_positive_weight <= 0.0:
            raise ValueError("rotation_action_gate_positive_weight must be positive")
        if not 0.0 <= self.rotation_action_gate_threshold <= 1.0:
            raise ValueError("rotation_action_gate_threshold must be in [0, 1]")
        if not (
            0.0
            <= self.rotation_action_support_min_deg
            <= self.rotation_action_support_max_deg
            <= 180.0
        ):
            raise ValueError(
                "rotation action support must satisfy 0 <= min <= max <= 180"
            )
        if not (
            0.0
            <= self.translation_recovery_support_min_m
            <= self.translation_recovery_support_max_m
        ):
            raise ValueError(
                "translation recovery support must satisfy 0 <= min <= max"
            )
        if not 0.0 <= self.translation_recovery_rotation_max_deg <= 180.0:
            raise ValueError(
                "translation recovery rotation maximum must be in [0, 180]"
            )
        if self.translation_recovery_geometry_scale_m <= 0.0:
            raise ValueError(
                "translation recovery geometry scale must be positive"
            )
        if self.rotation_action_translation_loss_weight < 0.0:
            raise ValueError(
                "rotation_action_translation_loss_weight must be non-negative"
            )
        if self.rotation_action_translation_endpoint_loss_weight < 0.0:
            raise ValueError(
                "rotation_action_translation_endpoint_loss_weight must be non-negative"
            )
        rotation_action_head_enabled = self.rotation_action_head_hidden_dim > 0
        if rotation_action_head_enabled:
            if not obs_as_global_cond:
                raise ValueError("rotation action head requires obs_as_global_cond")
            if not self.binary_gripper_retention_state_indices:
                raise ValueError(
                    "rotation action head requires closed-gripper routing state"
                )
            if self.rotation_action_head_geometry_key not in obs_shape_meta:
                raise KeyError("rotation action head geometry key is missing")
            if (
                self.rotation_action_head_geometry_confidence_key is not None
                and self.rotation_action_head_geometry_confidence_key
                not in obs_shape_meta
            ):
                raise KeyError("rotation action head confidence key is missing")
            if self.rotation_action_head_gated and not self.rotation_action_head_axis_aligned:
                raise ValueError("gated rotation action head requires axis alignment")
        elif self.rotation_action_head_gated:
            raise ValueError("rotation action gating requires a rotation action head")
        if self.rotation_action_translation_head_hidden_dim > 0 and not (
            rotation_action_head_enabled and self.rotation_action_head_gated
        ):
            raise ValueError(
                "translation compensation head requires a gated rotation action head"
            )
        if (
            self.rotation_action_translation_translation_only
            and not self.rotation_action_translation_geometry_only
        ):
            raise ValueError(
                "translation-only recovery context requires geometry-only decoding"
            )
        if len(set(self.binary_gripper_indices)) != len(
            self.binary_gripper_indices
        ):
            raise ValueError("binary_gripper_indices must be unique")
        if any(index < 0 or index >= action_dim for index in self.binary_gripper_indices):
            raise ValueError("binary_gripper_indices are outside the action shape")
        if self.binary_gripper_indices and not obs_as_global_cond:
            raise ValueError("binary gripper head currently requires obs_as_global_cond")
        if self.binary_gripper_retention_state_indices:
            if self.binary_gripper_geometry_key is None:
                raise ValueError(
                    "binary gripper retention requires binary_gripper_geometry_key"
                )
            if self.binary_gripper_retention_state_key not in obs_shape_meta:
                raise KeyError(
                    "binary gripper retention state key is missing from shape_meta: "
                    f"{self.binary_gripper_retention_state_key!r}"
                )
            if len(self.binary_gripper_retention_state_indices) != len(
                self.binary_gripper_indices
            ):
                raise ValueError(
                    "retention state indices must match the binary gripper outputs"
                )
            retention_state_size = int(
                math.prod(obs_shape_meta[self.binary_gripper_retention_state_key]["shape"])
            )
            if any(
                index < 0 or index >= retention_state_size
                for index in self.binary_gripper_retention_state_indices
            ):
                raise ValueError("binary gripper retention state index is out of range")
        binary_gripper_geometry_shape = None
        if self.binary_gripper_geometry_key is not None:
            if not self.binary_gripper_indices:
                raise ValueError(
                    "binary_gripper_geometry_key requires a binary gripper head"
                )
            if self.binary_gripper_geometry_key not in obs_shape_meta:
                raise KeyError(
                    "binary gripper geometry key is missing from shape_meta: "
                    f"{self.binary_gripper_geometry_key!r}"
                )
            binary_gripper_geometry_shape = tuple(
                obs_shape_meta[self.binary_gripper_geometry_key]["shape"]
            )
            if len(binary_gripper_geometry_shape) != 1 or int(
                binary_gripper_geometry_shape[0]
            ) < 3:
                raise ValueError(
                    "binary gripper geometry must be a vector with at least "
                    f"three translation values, got {binary_gripper_geometry_shape}"
                )
        binary_gripper_confidence_shape = None
        if self.binary_gripper_geometry_confidence_key is not None:
            if binary_gripper_geometry_shape is None:
                raise ValueError(
                    "binary gripper geometry confidence requires a geometry key"
                )
            if self.binary_gripper_geometry_confidence_key not in obs_shape_meta:
                raise KeyError(
                    "binary gripper confidence key is missing from shape_meta: "
                    f"{self.binary_gripper_geometry_confidence_key!r}"
                )
            binary_gripper_confidence_shape = tuple(
                obs_shape_meta[self.binary_gripper_geometry_confidence_key]["shape"]
            )
        if self.geometry_key is not None and self.aux_geometry_key is not None:
            raise ValueError(
                "geometry_key and aux_geometry_key are mutually exclusive: "
                "geometry cannot be both an inference condition and a label-only target"
            )
        geometry_shape = None
        if self.geometry_key is not None:
            if self.geometry_key not in obs_dict:
                raise KeyError(
                    f"geometry_key {self.geometry_key!r} is missing from shape_meta"
                )
            geometry_shape = tuple(obs_dict.pop(self.geometry_key))
            if len(geometry_shape) != 2:
                raise ValueError(
                    f"geometry observation must be [points, channels], got {geometry_shape}"
                )
        if self.geometry_confidence_key is not None:
            if self.geometry_confidence_key not in obs_dict:
                raise KeyError(
                    f"geometry_confidence_key {self.geometry_confidence_key!r} is missing from shape_meta"
                )
            obs_dict.pop(self.geometry_confidence_key)

        global_geometry_shape = None
        if self.global_geometry_key is not None:
            if self.global_geometry_key not in obs_dict:
                raise KeyError(
                    f"global_geometry_key {self.global_geometry_key!r} is missing from shape_meta"
                )
            global_geometry_shape = tuple(obs_dict.pop(self.global_geometry_key))
            if len(global_geometry_shape) != 1:
                raise ValueError(
                    "global geometry observation must be [features], got "
                    f"{global_geometry_shape}"
                )
        if self.global_geometry_confidence_key is not None:
            if self.global_geometry_confidence_key not in obs_shape_meta:
                raise KeyError(
                    "global_geometry_confidence_key "
                    f"{self.global_geometry_confidence_key!r} is missing from shape_meta"
                )
            if self.global_geometry_confidence_key != self.geometry_confidence_key:
                obs_dict.pop(self.global_geometry_confidence_key)

        anchor_flow_shape = None
        if self.rotation_action_translation_anchor_flow_key is not None:
            if self.rotation_action_translation_anchor_flow_key not in obs_dict:
                raise KeyError(
                    "translation anchor-flow key is missing from shape_meta: "
                    f"{self.rotation_action_translation_anchor_flow_key!r}"
                )
            anchor_flow_shape = tuple(
                obs_dict.pop(self.rotation_action_translation_anchor_flow_key)
            )
            if len(anchor_flow_shape) != 2:
                raise ValueError(
                    "translation anchor flow must be [points, channels], got "
                    f"{anchor_flow_shape}"
                )

        aux_geometry_shape = None
        if self.aux_geometry_key is not None:
            if self.aux_geometry_key not in obs_dict:
                raise KeyError(
                    f"aux_geometry_key {self.aux_geometry_key!r} is missing from shape_meta"
                )
            aux_geometry_shape = tuple(obs_dict.pop(self.aux_geometry_key))
            if self.aux_geometry_target_mode == "endpoint_displacement":
                if len(aux_geometry_shape) != 2 or aux_geometry_shape[-1] < 3:
                    raise ValueError(
                        "endpoint-displacement geometry must be "
                        f"[points, channels>=3], got {aux_geometry_shape}"
                    )
            elif self.aux_geometry_target_mode == "vector":
                if len(aux_geometry_shape) != 1:
                    raise ValueError(
                        f"vector auxiliary geometry must be [features], got {aux_geometry_shape}"
                    )
            else:
                raise ValueError(
                    "aux_geometry_target_mode must be 'endpoint_displacement' "
                    f"or 'vector', got {self.aux_geometry_target_mode!r}"
                )
            if self.aux_geometry_loss_weight < 0.0:
                raise ValueError("aux_geometry_loss_weight must be non-negative")

        obs_encoder = DP3Encoder(
            observation_space=obs_dict,
            img_crop_shape=crop_shape,
            out_channel=encoder_output_dim,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            use_pc_color=use_pc_color,
            pointnet_type=pointnet_type,
        )

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()
        geometry_adapter = None
        if geometry_shape is not None:
            geometry_adapter = ZeroInitGeometryAdapter(
                in_channels=int(geometry_shape[-1]),
                geometry_dim=int(geometry_encoder_output_dim),
                output_dim=int(obs_feature_dim),
                dropout=float(geometry_dropout),
                force_zero=bool(geometry_force_zero),
            )
            cprint(
                "[DP3] zero-init geometry adapter: "
                f"key={self.geometry_key}, shape={geometry_shape}, "
                f"output_dim={obs_feature_dim}, force_zero={geometry_force_zero}",
                "yellow",
            )
        global_geometry_adapter = None
        if global_geometry_shape is not None:
            global_geometry_adapter = ZeroInitGlobalGeometryAdapter(
                input_dim=int(global_geometry_shape[-1]),
                hidden_dim=int(global_geometry_hidden_dim),
                output_dim=int(obs_feature_dim),
                dropout=float(global_geometry_dropout),
                force_zero=bool(global_geometry_force_zero),
            )
            cprint(
                "[DP3] zero-init global geometry adapter: "
                f"key={self.global_geometry_key}, shape={global_geometry_shape}, "
                f"output_dim={obs_feature_dim}, force_zero={global_geometry_force_zero}",
                "yellow",
            )
        aux_geometry_head = None
        if aux_geometry_shape is not None:
            if self.aux_geometry_target_mode == "endpoint_displacement":
                aux_geometry_dim = int(aux_geometry_shape[0]) * 3
            else:
                aux_geometry_dim = int(aux_geometry_shape[0])
            aux_geometry_head = FutureGeometryHead(
                input_dim=int(obs_feature_dim) * int(n_obs_steps),
                hidden_dim=int(aux_geometry_hidden_dim),
                output_dim=aux_geometry_dim,
            )
            cprint(
                "[DP3] future-geometry auxiliary head: "
                f"key={self.aux_geometry_key}, target={self.aux_geometry_target_mode}, "
                f"output_dim={aux_geometry_dim}, weight={self.aux_geometry_loss_weight}",
                "yellow",
            )
        binary_gripper_head = None
        binary_gripper_retention_head = None
        if self.binary_gripper_indices:
            binary_gripper_input_dim = int(obs_feature_dim) * int(n_obs_steps)
            if binary_gripper_geometry_shape is not None:
                # Preserve the signed SE(3) token and add an invariant
                # translation magnitude for reliable release extrapolation.
                geometry_features = int(binary_gripper_geometry_shape[0]) + 1
                if binary_gripper_confidence_shape is not None:
                    geometry_features += int(math.prod(binary_gripper_confidence_shape))
                binary_gripper_input_dim += int(n_obs_steps) * geometry_features
            binary_gripper_head = BinaryGripperHead(
                input_dim=binary_gripper_input_dim,
                hidden_dim=int(binary_gripper_hidden_dim),
                output_dim=int(n_action_steps)
                * len(self.binary_gripper_indices),
            )
            if self.binary_gripper_retention_state_indices:
                retention_geometry_features = geometry_features
                if self.binary_gripper_retention_invariant:
                    if int(binary_gripper_geometry_shape[0]) < 9:
                        raise ValueError(
                            "invariant gripper retention requires a 9D SE(3) frame"
                        )
                    retention_geometry_features = 2
                    if binary_gripper_confidence_shape is not None:
                        retention_geometry_features += int(
                            math.prod(binary_gripper_confidence_shape)
                        )
                binary_gripper_retention_head = BinaryGripperHead(
                    input_dim=int(n_obs_steps) * retention_geometry_features,
                    hidden_dim=int(binary_gripper_hidden_dim),
                    output_dim=int(n_action_steps)
                    * len(self.binary_gripper_indices),
                )
            cprint(
                "[DP3] learned binary gripper head: "
                f"indices={self.binary_gripper_indices}, "
                f"steps={n_action_steps}, weight={self.binary_gripper_loss_weight}, "
                f"geometry={self.binary_gripper_geometry_key}, "
                f"retention={bool(self.binary_gripper_retention_state_indices)}, "
                f"retention_invariant={self.binary_gripper_retention_invariant}",
                "yellow",
            )
        rotation_action_head = None
        rotation_action_gate = None
        rotation_action_translation_head = None
        if rotation_action_head_enabled:
            rotation_geometry_shape = tuple(
                obs_shape_meta[self.rotation_action_head_geometry_key]["shape"]
            )
            if len(rotation_geometry_shape) != 1:
                raise ValueError("rotation action head geometry must be a vector")
            if self.rotation_action_head_geometry_only:
                # Translation plus signed rotvec, optionally with confidence.
                rotation_step_dim = 6
                if self.rotation_action_head_geometry_confidence_key is not None:
                    rotation_step_dim += int(
                        math.prod(
                            obs_shape_meta[
                                self.rotation_action_head_geometry_confidence_key
                            ]["shape"]
                        )
                    )
                rotation_context_dim = int(n_obs_steps) * rotation_step_dim
            else:
                rotation_context_dim = int(obs_feature_dim) * int(n_obs_steps)
                rotation_context_dim += int(n_obs_steps) * int(
                    math.prod(rotation_geometry_shape)
                )
                if self.rotation_action_head_geometry_confidence_key is not None:
                    rotation_context_dim += int(n_obs_steps) * int(
                        math.prod(
                            obs_shape_meta[
                                self.rotation_action_head_geometry_confidence_key
                            ]["shape"]
                        )
                    )
            rotation_action_head = TaskAlignedRotationHead(
                input_dim=rotation_context_dim,
                hidden_dim=self.rotation_action_head_hidden_dim,
                output_dim=int(n_action_steps)
                * (2 if self.rotation_action_head_axis_aligned else 6),
            )
            if self.rotation_action_head_gated:
                gate_context_dim = rotation_context_dim
                if (
                    self.rotation_action_gate_geometry_only
                    != self.rotation_action_head_geometry_only
                ):
                    if self.rotation_action_gate_geometry_only:
                        gate_step_dim = 6
                        if (
                            self.rotation_action_head_geometry_confidence_key
                            is not None
                        ):
                            gate_step_dim += int(
                                math.prod(
                                    obs_shape_meta[
                                        self.rotation_action_head_geometry_confidence_key
                                    ]["shape"]
                                )
                            )
                        gate_context_dim = int(n_obs_steps) * gate_step_dim
                    else:
                        gate_context_dim = int(obs_feature_dim) * int(n_obs_steps)
                        gate_context_dim += int(n_obs_steps) * int(
                            math.prod(rotation_geometry_shape)
                        )
                        if (
                            self.rotation_action_head_geometry_confidence_key
                            is not None
                        ):
                            gate_context_dim += int(n_obs_steps) * int(
                                math.prod(
                                    obs_shape_meta[
                                        self.rotation_action_head_geometry_confidence_key
                                    ]["shape"]
                                )
                            )
                rotation_action_gate = TaskAlignedRotationHead(
                    input_dim=gate_context_dim,
                    hidden_dim=min(self.rotation_action_head_hidden_dim, 64),
                    output_dim=1,
                )
            if self.rotation_action_translation_head_hidden_dim > 0:
                if anchor_flow_shape is not None:
                    confidence_dim = 0
                    if self.rotation_action_head_geometry_confidence_key is not None:
                        confidence_dim = int(
                            math.prod(
                                obs_shape_meta[
                                    self.rotation_action_head_geometry_confidence_key
                                ]["shape"]
                            )
                        )
                    if self.rotation_action_translation_flow_consistent:
                        rotation_action_translation_head = (
                            FlowConsistentAnchorTranslationHead(
                                in_channels=int(anchor_flow_shape[-1]),
                                hidden_dim=self.rotation_action_translation_head_hidden_dim,
                                global_dim=int(math.prod(rotation_geometry_shape)),
                                confidence_dim=confidence_dim,
                                action_steps=int(n_action_steps),
                            )
                        )
                    else:
                        rotation_action_translation_head = AnchorFlowTranslationHead(
                            in_channels=int(anchor_flow_shape[-1]),
                            hidden_dim=self.rotation_action_translation_head_hidden_dim,
                            global_dim=int(math.prod(rotation_geometry_shape)),
                            confidence_dim=confidence_dim,
                            output_dim=int(n_action_steps) * 6,
                        )
                elif self.rotation_action_translation_geometry_only:
                    translation_step_dim = (
                        3
                        if self.rotation_action_translation_translation_only
                        else 6
                    )
                    if self.rotation_action_head_geometry_confidence_key is not None:
                        translation_step_dim += int(
                            math.prod(
                                obs_shape_meta[
                                    self.rotation_action_head_geometry_confidence_key
                                ]["shape"]
                            )
                        )
                    translation_context_dim = translation_step_dim * (
                        1
                        if self.rotation_action_translation_current_only
                        else int(n_obs_steps)
                    )
                else:
                    translation_context_steps = (
                        1
                        if self.rotation_action_translation_current_only
                        else int(n_obs_steps)
                    )
                    translation_context_dim = int(obs_feature_dim) * translation_context_steps
                    translation_context_dim += translation_context_steps * int(
                        math.prod(rotation_geometry_shape)
                    )
                    if self.rotation_action_head_geometry_confidence_key is not None:
                        translation_context_dim += translation_context_steps * int(
                            math.prod(
                                obs_shape_meta[
                                    self.rotation_action_head_geometry_confidence_key
                                ]["shape"]
                            )
                        )
                if anchor_flow_shape is None:
                    rotation_action_translation_head = TaskAlignedRotationHead(
                        input_dim=translation_context_dim,
                        hidden_dim=self.rotation_action_translation_head_hidden_dim,
                        output_dim=int(n_action_steps) * 6,
                    )
            cprint(
                "[DP3] task-aligned rotation head: "
                f"geometry={self.rotation_action_head_geometry_key}, "
                f"steps={n_action_steps}, weight={self.rotation_action_head_loss_weight}, "
                f"axis_aligned={self.rotation_action_head_axis_aligned}, "
                f"geometry_only={self.rotation_action_head_geometry_only}, "
                f"gated={self.rotation_action_head_gated}, "
                f"gate_geometry_only={self.rotation_action_gate_geometry_only}",
                "yellow",
            )
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            if "cross_attention" in self.condition_type:
                global_cond_dim = obs_feature_dim
            else:
                global_cond_dim = obs_feature_dim * n_obs_steps

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(
            f"[DiffusionUnetHybridPointcloudPolicy] use_pc_color: {self.use_pc_color}",
            "yellow",
        )
        cprint(
            f"[DiffusionUnetHybridPointcloudPolicy] pointnet_type: {self.pointnet_type}",
            "yellow",
        )

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
        )

        self.obs_encoder = obs_encoder
        self.geometry_adapter = geometry_adapter
        self.global_geometry_adapter = global_geometry_adapter
        self.aux_geometry_head = aux_geometry_head
        self.binary_gripper_head = binary_gripper_head
        self.binary_gripper_retention_head = binary_gripper_retention_head
        self.rotation_action_head = rotation_action_head
        self.rotation_action_gate = rotation_action_gate
        self.rotation_action_translation_head = rotation_action_translation_head
        self.aux_geometry_shape = aux_geometry_shape
        self.model = model
        self.noise_scheduler = noise_scheduler

        self.noise_scheduler_pc = copy.deepcopy(noise_scheduler)
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )

        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        print_params(self)

    def _prepare_nobs(self, nobs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        point_cloud_channel_map = getattr(self.obs_encoder, "point_cloud_channel_map", {})
        for key, expected_channels in point_cloud_channel_map.items():
            if key not in nobs:
                continue
            if nobs[key].shape[-1] < expected_channels:
                raise RuntimeError(
                    f"Observation {key!r} has {nobs[key].shape[-1]} channels, expected at least {expected_channels}."
                )
            if nobs[key].shape[-1] > expected_channels:
                nobs[key] = nobs[key][..., :expected_channels]
        if self.geometry_adapter is not None:
            value = nobs[self.geometry_key]
            expected_channels = self.geometry_adapter.encoder.mlp[0].in_features
            if value.shape[-1] != expected_channels:
                raise RuntimeError(
                    f"Geometry observation {self.geometry_key!r} has "
                    f"{value.shape[-1]} channels, expected {expected_channels}."
                )
        return nobs

    def _encode_observations(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        features = self.obs_encoder(observations)
        if self.geometry_adapter is not None:
            confidence = None
            if self.geometry_confidence_key is not None:
                confidence = observations[self.geometry_confidence_key]
            features = features + self.geometry_adapter(
                observations[self.geometry_key], confidence=confidence
            )
        if self.global_geometry_adapter is not None:
            confidence = None
            if self.global_geometry_confidence_key is not None:
                confidence = observations[self.global_geometry_confidence_key]
            features = features + self.global_geometry_adapter(
                observations[self.global_geometry_key], confidence=confidence
            )
        return features

    def freeze_base_for_geometry(self) -> None:
        if self.geometry_adapter is None and self.global_geometry_adapter is None:
            raise RuntimeError("Cannot freeze for geometry: geometry adapters are disabled")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for adapter in (self.geometry_adapter, self.global_geometry_adapter):
            if adapter is not None:
                for parameter in adapter.parameters():
                    parameter.requires_grad_(True)

    def geometry_adapter_parameters(self):
        result = []
        for adapter in (self.geometry_adapter, self.global_geometry_adapter):
            if adapter is not None:
                result.extend(adapter.parameters())
        return list(result)

    def freeze_base_for_binary_gripper(self) -> None:
        if self.binary_gripper_head is None:
            raise RuntimeError("Cannot freeze for gripper: binary head is disabled")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.binary_gripper_head.parameters():
            parameter.requires_grad_(True)
        if self.binary_gripper_retention_head is not None:
            for parameter in self.binary_gripper_retention_head.parameters():
                parameter.requires_grad_(True)

    def freeze_binary_gripper_heads(self) -> None:
        """Keep learned grasp/retention decisions fixed during motion tuning."""

        if self.binary_gripper_head is None:
            raise RuntimeError("Cannot freeze gripper heads: binary head is disabled")
        for parameter in self.parameters():
            parameter.requires_grad_(True)
        for parameter in self.binary_gripper_head.parameters():
            parameter.requires_grad_(False)
        if self.binary_gripper_retention_head is not None:
            for parameter in self.binary_gripper_retention_head.parameters():
                parameter.requires_grad_(False)

    def freeze_base_for_rotation_action(self) -> None:
        if self.rotation_action_head is None:
            raise RuntimeError("Cannot freeze for rotation action: head is disabled")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.rotation_action_head.parameters():
            parameter.requires_grad_(True)
        if self.rotation_action_gate is not None:
            for parameter in self.rotation_action_gate.parameters():
                parameter.requires_grad_(True)
        if self.rotation_action_translation_head is not None:
            for parameter in self.rotation_action_translation_head.parameters():
                parameter.requires_grad_(True)

    def freeze_base_for_recovery_translation(self) -> None:
        if self.rotation_action_translation_head is None:
            raise RuntimeError("Cannot freeze for recovery translation: head is disabled")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.rotation_action_translation_head.parameters():
            parameter.requires_grad_(True)

    def freeze_base_for_rotation_gate(self) -> None:
        """Train recovery-state recognition without changing its action decoder."""

        if self.rotation_action_gate is None:
            raise RuntimeError("Cannot freeze for rotation gate: gate is disabled")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.rotation_action_gate.parameters():
            parameter.requires_grad_(True)

    def _rotation_action_context(
        self,
        encoded_features: torch.Tensor,
        observations: Dict[str, torch.Tensor],
        batch_size: int,
        *,
        geometry_only: bool | None = None,
        current_only: bool = False,
    ) -> torch.Tensor:
        geometry_only = (
            self.rotation_action_head_geometry_only
            if geometry_only is None
            else bool(geometry_only)
        )
        geometry_sequence = observations[self.rotation_action_head_geometry_key][
            :, : self.n_obs_steps
        ].reshape(int(batch_size), self.n_obs_steps, -1)
        if geometry_only:
            raw_geometry = self.normalizer[
                self.rotation_action_head_geometry_key
            ].unnormalize(geometry_sequence)
            if current_only:
                raw_geometry = raw_geometry[:, -1:]
            context = [
                raw_geometry[..., :3],
                _frame9_rotation_rotvec(raw_geometry),
            ]
        else:
            context = [
                encoded_features.reshape(int(batch_size), -1),
                geometry_sequence.reshape(int(batch_size), -1),
            ]
        if self.rotation_action_head_geometry_confidence_key is not None:
            confidence = observations[
                self.rotation_action_head_geometry_confidence_key
            ][:, : self.n_obs_steps].reshape(int(batch_size), -1)
            if geometry_only:
                confidence = self.normalizer[
                    self.rotation_action_head_geometry_confidence_key
                ].unnormalize(
                    confidence.reshape(int(batch_size), self.n_obs_steps, -1)
                )
                if current_only:
                    confidence = confidence[:, -1:]
                context.append(confidence)
            else:
                context.append(confidence)
        return torch.cat(
            [value.reshape(int(batch_size), -1) for value in context], dim=-1
        )

    def _rotation_action_gate_logits(
        self,
        encoded_features: torch.Tensor,
        observations: Dict[str, torch.Tensor],
        batch_size: int,
    ) -> torch.Tensor | None:
        if self.rotation_action_gate is None:
            return None
        return self.rotation_action_gate(
            self._rotation_action_context(
                encoded_features,
                observations,
                int(batch_size),
                geometry_only=self.rotation_action_gate_geometry_only,
            )
        ).reshape(int(batch_size))

    def _rotation_action_translation_prediction(
        self,
        encoded_features: torch.Tensor,
        observations: Dict[str, torch.Tensor],
        batch_size: int,
    ) -> torch.Tensor | None:
        if self.rotation_action_translation_head is None:
            return None
        if self.rotation_action_translation_anchor_flow_key is not None:
            tokens = observations[self.rotation_action_translation_anchor_flow_key][
                :, self.n_obs_steps - 1
            ]
            geometry = observations[self.rotation_action_head_geometry_key][
                :, self.n_obs_steps - 1
            ].reshape(int(batch_size), -1)
            if self.rotation_action_head_geometry_confidence_key is None:
                confidence = torch.empty(
                    (int(batch_size), 0),
                    dtype=tokens.dtype,
                    device=tokens.device,
                )
            else:
                confidence = observations[
                    self.rotation_action_head_geometry_confidence_key
                ][:, self.n_obs_steps - 1].reshape(int(batch_size), -1)
            prediction = self.rotation_action_translation_head(
                tokens, geometry, confidence
            )
            return prediction.reshape(int(batch_size), self.n_action_steps, 2, 3)
        if self.rotation_action_translation_geometry_only:
            if self.rotation_action_translation_translation_only:
                geometry = observations[self.rotation_action_head_geometry_key][
                    :, : self.n_obs_steps
                ].reshape(int(batch_size), self.n_obs_steps, -1)
                geometry = self.normalizer[
                    self.rotation_action_head_geometry_key
                ].unnormalize(geometry)
                if self.rotation_action_translation_current_only:
                    geometry = geometry[:, -1:]
                context_parts = [geometry[..., :3]]
                if self.rotation_action_head_geometry_confidence_key is not None:
                    confidence = observations[
                        self.rotation_action_head_geometry_confidence_key
                    ][:, : self.n_obs_steps].reshape(
                        int(batch_size), self.n_obs_steps, -1
                    )
                    confidence = self.normalizer[
                        self.rotation_action_head_geometry_confidence_key
                    ].unnormalize(confidence)
                    if self.rotation_action_translation_current_only:
                        confidence = confidence[:, -1:]
                    context_parts.append(confidence)
                context = torch.cat(
                    [part.reshape(int(batch_size), -1) for part in context_parts],
                    dim=-1,
                )
            else:
                context = self._rotation_action_context(
                    encoded_features,
                    observations,
                    int(batch_size),
                    geometry_only=True,
                    current_only=self.rotation_action_translation_current_only,
                )
            prediction = self.rotation_action_translation_head(context)
            return prediction.reshape(int(batch_size), self.n_action_steps, 2, 3)
        encoded_context = encoded_features.reshape(
            int(batch_size), self.n_obs_steps, -1
        )
        geometry = observations[self.rotation_action_head_geometry_key][
            :, : self.n_obs_steps
        ].reshape(int(batch_size), self.n_obs_steps, -1)
        if self.rotation_action_translation_current_only:
            encoded_context = encoded_context[:, -1:]
            geometry = geometry[:, -1:]
        context = [
            encoded_context.reshape(int(batch_size), -1),
            geometry.reshape(int(batch_size), -1),
        ]
        if self.rotation_action_head_geometry_confidence_key is not None:
            confidence = observations[
                self.rotation_action_head_geometry_confidence_key
            ][:, : self.n_obs_steps].reshape(
                int(batch_size), self.n_obs_steps, -1
            )
            if self.rotation_action_translation_current_only:
                confidence = confidence[:, -1:]
            context.append(confidence.reshape(int(batch_size), -1))
        prediction = self.rotation_action_translation_head(
            torch.cat(context, dim=-1)
        )
        return prediction.reshape(int(batch_size), self.n_action_steps, 2, 3)

    def _rotation_action_prediction(
        self,
        encoded_features: torch.Tensor,
        observations: Dict[str, torch.Tensor],
        batch_size: int,
    ) -> torch.Tensor | None:
        if self.rotation_action_head is None:
            return None
        prediction = self.rotation_action_head(
            self._rotation_action_context(
                encoded_features, observations, int(batch_size)
            )
        )
        if self.rotation_action_head_axis_aligned:
            magnitude = prediction.reshape(
                int(batch_size), self.n_action_steps, 2, 1
            )
            axis, _valid = self._rotation_action_axis(observations, batch_size)
            return magnitude * axis[:, None, None, :]
        return prediction.reshape(int(batch_size), self.n_action_steps, 2, 3)

    def _rotation_action_axis(
        self,
        observations: Dict[str, torch.Tensor],
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        relation = observations[self.rotation_action_head_geometry_key][
            :, self.n_obs_steps - 1
        ].reshape(int(batch_size), -1)
        relation = self.normalizer[
            self.rotation_action_head_geometry_key
        ].unnormalize(relation)
        rotvec = _frame9_rotation_rotvec(relation)
        magnitude = torch.linalg.vector_norm(rotvec, dim=-1)
        valid = magnitude > 1.0e-5
        if self.rotation_action_head_geometry_confidence_key is not None:
            confidence = observations[
                self.rotation_action_head_geometry_confidence_key
            ][:, self.n_obs_steps - 1].reshape(int(batch_size), -1)
            confidence = self.normalizer[
                self.rotation_action_head_geometry_confidence_key
            ].unnormalize(confidence)
            valid = valid & (confidence.mean(dim=-1) >= 0.5)
        axis = F.normalize(rotvec, dim=-1, eps=1.0e-6)
        return axis, valid

    def _recovery_geometry_valid(
        self,
        observations: Dict[str, torch.Tensor],
        batch_size: int,
    ) -> torch.Tensor:
        """Validate a camera-derived SE(3) relation without requiring rotation.

        Translation-only recovery remains meaningful when the remaining
        rotation is exactly zero, so it must not inherit the non-zero-axis
        requirement used by the rotation decoder.
        """

        relation = observations[self.rotation_action_head_geometry_key][
            :, self.n_obs_steps - 1
        ].reshape(int(batch_size), -1)
        relation = self.normalizer[
            self.rotation_action_head_geometry_key
        ].unnormalize(relation)
        valid = torch.isfinite(relation).all(dim=-1)
        if self.rotation_action_head_geometry_confidence_key is not None:
            confidence = observations[
                self.rotation_action_head_geometry_confidence_key
            ][:, self.n_obs_steps - 1].reshape(int(batch_size), -1)
            confidence = self.normalizer[
                self.rotation_action_head_geometry_confidence_key
            ].unnormalize(confidence)
            valid = valid & torch.isfinite(confidence).all(dim=-1)
            valid = valid & (confidence.mean(dim=-1) >= 0.5)
        return valid

    def _binary_gripper_geometry_context(
        self,
        observations: Dict[str, torch.Tensor],
        batch_size: int,
        current_only: bool = False,
        invariant: bool = False,
    ) -> torch.Tensor:
        if self.binary_gripper_geometry_key is None:
            raise RuntimeError("binary gripper geometry context is disabled")
        relation = observations[self.binary_gripper_geometry_key][
            :, : self.n_obs_steps
        ].reshape(int(batch_size), self.n_obs_steps, -1)
        if invariant:
            # ``observations`` have already passed through the policy's
            # component-wise normalizer. Norms and geodesic angles are not
            # invariant to that affine scaling, so restore the raw frame9 token
            # before computing physical SE(3) invariants.
            raw_relation = self.normalizer[
                self.binary_gripper_geometry_key
            ].unnormalize(relation)
            components = [_se3_frame_invariants(raw_relation)]
        else:
            translation_norm = torch.linalg.vector_norm(
                relation[..., :3], dim=-1, keepdim=True
            )
            components = [relation, translation_norm]
        if self.binary_gripper_geometry_confidence_key is not None:
            confidence = observations[
                self.binary_gripper_geometry_confidence_key
            ][:, : self.n_obs_steps].reshape(int(batch_size), self.n_obs_steps, -1)
            if invariant:
                confidence = self.normalizer[
                    self.binary_gripper_geometry_confidence_key
                ].unnormalize(confidence)
            if current_only:
                confidence = confidence.clone()
                confidence[:, :-1] = 0.0
            components.append(confidence)
        context = torch.cat(components, dim=-1)
        if current_only:
            context = context.clone()
            context[:, :-1] = 0.0
        return context.reshape(int(batch_size), -1)

    def _binary_gripper_context(
        self,
        observation_features: torch.Tensor,
        observations: Dict[str, torch.Tensor],
        batch_size: int,
    ) -> torch.Tensor:
        context = observation_features.reshape(int(batch_size), -1)
        if self.binary_gripper_geometry_key is None:
            return context
        return torch.cat(
            (context, self._binary_gripper_geometry_context(observations, batch_size)),
            dim=-1,
        )

    def _binary_gripper_logits(
        self,
        observation_features: torch.Tensor,
        observations: Dict[str, torch.Tensor],
        batch_size: int,
    ) -> torch.Tensor:
        logits = self.binary_gripper_head(
            self._binary_gripper_context(
                observation_features, observations, batch_size
            )
        ).reshape(int(batch_size), self.n_action_steps, -1)
        if self.binary_gripper_retention_head is None:
            return logits
        retention_logits = self.binary_gripper_retention_head(
            self._binary_gripper_geometry_context(
                observations,
                batch_size,
                current_only=self.binary_gripper_retention_current_only,
                invariant=self.binary_gripper_retention_invariant,
            )
        ).reshape(int(batch_size), self.n_action_steps, -1)
        current_closed, _current_gripper_state = self._binary_gripper_retention_mask(
            observations, batch_size
        )
        return torch.where(current_closed[:, None, :], retention_logits, logits)

    def _binary_gripper_retention_mask(
        self,
        observations: Dict[str, torch.Tensor],
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.binary_gripper_retention_head is None:
            raise RuntimeError("binary gripper retention is disabled")
        current_state = observations[self.binary_gripper_retention_state_key][
            :, self.n_obs_steps - 1
        ].reshape(int(batch_size), -1)
        current_gripper_state = current_state[
            :, list(self.binary_gripper_retention_state_indices)
        ]
        current_closed = (
            current_gripper_state < self.binary_gripper_retention_closed_threshold
        )
        return current_closed, current_gripper_state

    def _predict_aux_geometry(
        self, observation_features: torch.Tensor, batch_size: int
    ) -> torch.Tensor | None:
        if self.aux_geometry_head is None:
            return None
        context = observation_features.reshape(int(batch_size), -1)
        prediction = self.aux_geometry_head(context)
        if self.aux_geometry_target_mode == "endpoint_displacement":
            return prediction.reshape(
                int(batch_size), int(self.aux_geometry_shape[0]), 3
            )
        return prediction.reshape(int(batch_size), int(self.aux_geometry_shape[0]))

    def _unnormalize_aux_geometry(self, normalized: torch.Tensor) -> torch.Tensor:
        if self.aux_geometry_key is None:
            raise RuntimeError("auxiliary geometry prediction is disabled")
        params = self.normalizer.params_dict[self.aux_geometry_key]
        if self.aux_geometry_target_mode == "endpoint_displacement":
            scale = params["scale"][-3:]
            offset = params["offset"][-3:]
        else:
            scale = params["scale"]
            offset = params["offset"]
        return (normalized - offset) / scale

    def _aux_geometry_target(self, normalized_observations: Dict[str, torch.Tensor]):
        value = normalized_observations[self.aux_geometry_key][:, -1]
        if self.aux_geometry_target_mode == "endpoint_displacement":
            return value[..., -3:]
        return value

    # ========= inference  ============
    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        condition_data_pc=None,
        condition_mask_pc=None,
        local_cond=None,
        global_cond=None,
        generator=None,
        # keyword arguments to scheduler.step
        **kwargs,
    ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
        )

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            model_output = model(
                sample=trajectory,
                timestep=t,
                local_cond=local_cond,
                global_cond=global_cond,
            )

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output,
                t,
                trajectory,
            ).prev_sample

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        nobs = self._prepare_nobs(nobs)

        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        binary_gripper_logits = None
        rotation_action_prediction = None
        rotation_action_gate_logits = None
        rotation_action_translation_prediction = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self._encode_observations(this_nobs)
            aux_geometry_pred = self._predict_aux_geometry(nobs_features, B)
            if self.binary_gripper_head is not None:
                binary_gripper_logits = self._binary_gripper_logits(
                    nobs_features, nobs, B
                )
            rotation_action_prediction = self._rotation_action_prediction(
                nobs_features, nobs, B
            )
            rotation_action_gate_logits = self._rotation_action_gate_logits(
                nobs_features, nobs, B
            )
            rotation_action_translation_prediction = (
                self._rotation_action_translation_prediction(nobs_features, nobs, B)
            )
            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(B, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self._encode_observations(this_nobs)
            aux_geometry_pred = self._predict_aux_geometry(nobs_features, B)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da + Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs_features
            cond_mask[:, :To, Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs,
        )

        # unnormalize prediction
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]
        binary_gripper_probability = None
        if binary_gripper_logits is not None:
            binary_gripper_probability = torch.sigmoid(binary_gripper_logits)
            action = action.clone()
            command = torch.where(
                binary_gripper_probability >= self.binary_gripper_threshold,
                torch.zeros_like(binary_gripper_probability),
                torch.ones_like(binary_gripper_probability),
            )
            action[:, :, list(self.binary_gripper_indices)] = command
        # Keep the policy-generated motion available to a deployment-time
        # temporal debounce. This snapshot already includes learned binary
        # gripper commands but precedes the factorized recovery replacement.
        action_before_rotation_recovery = action.clone()
        rotation_action_active = None
        translation_recovery_active = None
        if rotation_action_prediction is not None:
            retention_active, _current_gripper_state = (
                self._binary_gripper_retention_mask(nobs, B)
            )
            _rotation_axis, rotation_geometry_valid = self._rotation_action_axis(
                nobs, B
            )
            relation = nobs[self.rotation_action_head_geometry_key][
                :, self.n_obs_steps - 1
            ].reshape(B, -1)
            relation = self.normalizer[
                self.rotation_action_head_geometry_key
            ].unnormalize(relation)
            rotation_geometry_deg = torch.rad2deg(
                torch.linalg.vector_norm(
                    _frame9_rotation_rotvec(relation), dim=-1
                )
            )
            translation_geometry_m = torch.linalg.vector_norm(
                relation[:, :3], dim=-1
            ) * self.translation_recovery_geometry_scale_m
            rotation_geometry_valid = rotation_geometry_valid & (
                rotation_geometry_deg >= self.rotation_action_support_min_deg
            ) & (
                rotation_geometry_deg <= self.rotation_action_support_max_deg
            )
            rotation_action_active = (
                retention_active & rotation_geometry_valid[:, None]
            )
            translation_recovery_enabled = (
                self.translation_recovery_support_max_m > 0.0
            )
            translation_geometry_valid = self._recovery_geometry_valid(nobs, B)
            translation_geometry_valid = translation_geometry_valid & (
                translation_geometry_m >= self.translation_recovery_support_min_m
            ) & (
                translation_geometry_m <= self.translation_recovery_support_max_m
            ) & (
                rotation_geometry_deg <= self.translation_recovery_rotation_max_deg
            )
            if not translation_recovery_enabled:
                translation_geometry_valid = torch.zeros_like(
                    translation_geometry_valid, dtype=torch.bool
                )
            translation_recovery_active = (
                retention_active & translation_geometry_valid[:, None]
            )
            if rotation_action_gate_logits is not None:
                gate_active = (
                    torch.sigmoid(rotation_action_gate_logits)
                    >= self.rotation_action_gate_threshold
                )
                rotation_action_active = rotation_action_active & gate_active[:, None]
                translation_recovery_active = (
                    translation_recovery_active & gate_active[:, None]
                )
            action = _replace_closed_arm_rotations(
                action, rotation_action_prediction, rotation_action_active
            )
            if rotation_action_translation_prediction is not None:
                translation_action_active = translation_recovery_active
                if self.rotation_action_translation_apply_on_high_rotation:
                    # Optional legacy behavior for heads explicitly trained to
                    # compensate translation during high-rotation recovery.
                    # Low-rotation stall heads keep this disabled so decoder
                    # duties remain orthogonal.
                    translation_action_active = (
                        translation_action_active | rotation_action_active
                    )
                action = _replace_closed_arm_translations(
                    action,
                    rotation_action_translation_prediction,
                    translation_action_active,
                )

        # get prediction
        result = {
            "action": action,
            "action_pred": action_pred,
            "action_before_rotation_recovery": action_before_rotation_recovery,
        }
        if aux_geometry_pred is not None:
            result["aux_geometry_pred_normalized"] = aux_geometry_pred
            result["aux_geometry_pred"] = self._unnormalize_aux_geometry(
                aux_geometry_pred
            )
        if binary_gripper_probability is not None:
            result["binary_gripper_closed_probability"] = (
                binary_gripper_probability
            )
            if self.binary_gripper_retention_head is not None:
                retention_active, retention_state = (
                    self._binary_gripper_retention_mask(nobs, B)
                )
                retention_geometry = self._binary_gripper_geometry_context(
                    nobs,
                    B,
                    current_only=self.binary_gripper_retention_current_only,
                    invariant=self.binary_gripper_retention_invariant,
                ).reshape(B, self.n_obs_steps, -1)
                result["binary_gripper_retention_active"] = retention_active
                result["binary_gripper_retention_normalized_state"] = (
                    retention_state
                )
                result["binary_gripper_retention_geometry"] = retention_geometry
        if rotation_action_prediction is not None:
            result["rotation_action_head_prediction"] = rotation_action_prediction
            result["rotation_action_head_active"] = rotation_action_active
            result["rotation_action_geometry_rotation_deg"] = (
                rotation_geometry_deg
            )
            result["rotation_action_geometry_translation_m"] = (
                translation_geometry_m
            )
            result["translation_recovery_head_active"] = (
                translation_recovery_active
            )
            result["recovery_action_head_active"] = (
                rotation_action_active | translation_recovery_active
            )
        if rotation_action_gate_logits is not None:
            result["rotation_action_gate_probability"] = torch.sigmoid(
                rotation_action_gate_logits
            )
        if rotation_action_translation_prediction is not None:
            result["rotation_action_translation_prediction"] = (
                rotation_action_translation_prediction
            )

        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input

        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])

        nobs = self._prepare_nobs(nobs)

        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        binary_gripper_logits = None
        rotation_action_prediction = None
        rotation_action_gate_logits = None
        rotation_action_translation_prediction = None
        trajectory = nactions
        cond_data = trajectory

        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self._encode_observations(this_nobs)
            aux_geometry_pred = self._predict_aux_geometry(
                nobs_features, batch_size
            )
            if self.binary_gripper_head is not None:
                binary_gripper_logits = self._binary_gripper_logits(
                    nobs_features, nobs, batch_size
                )
            rotation_action_prediction = self._rotation_action_prediction(
                nobs_features, nobs, batch_size
            )
            rotation_action_gate_logits = self._rotation_action_gate_logits(
                nobs_features, nobs, batch_size
            )
            rotation_action_translation_prediction = (
                self._rotation_action_translation_prediction(
                    nobs_features, nobs, batch_size
                )
            )

            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self._encode_observations(this_nobs)
            aux_geometry_pred = self._predict_aux_geometry(
                nobs_features[:, : self.n_obs_steps], batch_size
            )
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)

        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz, ),
            device=trajectory.device,
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # Predict the noise residual

        pred = self.model(
            sample=noisy_trajectory,
            timestep=timesteps,
            local_cond=local_cond,
            global_cond=global_cond,
        )

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        elif pred_type == "v_prediction":
            # https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # https://github.com/huggingface/diffusers/blob/v0.11.1-patch/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # sigma = self.noise_scheduler.sigmas[timesteps]
            # alpha_t, sigma_t = self.noise_scheduler._sigma_to_alpha_sigma_t(sigma)
            self.noise_scheduler.alpha_t = self.noise_scheduler.alpha_t.to(self.device)
            self.noise_scheduler.sigma_t = self.noise_scheduler.sigma_t.to(self.device)
            alpha_t, sigma_t = (
                self.noise_scheduler.alpha_t[timesteps],
                self.noise_scheduler.sigma_t[timesteps],
            )
            alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
            v_t = alpha_t * noise - sigma_t * trajectory
            target = v_t
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        action_loss = F.mse_loss(pred, target, reduction="none")
        action_weights = loss_mask.type(action_loss.dtype)
        if self.recovery_rotation_only_loss:
            if "is_recovery_augmented" not in batch or "active_arm_right" not in batch:
                raise KeyError(
                    "rotation-only recovery loss requires is_recovery_augmented "
                    "and active_arm_right batch labels"
                )
            recovery_weights = _recovery_action_channel_weights(
                batch["is_recovery_augmented"].to(
                    device=action_loss.device, dtype=action_loss.dtype
                ),
                batch["active_arm_right"].to(
                    device=action_loss.device, dtype=action_loss.dtype
                ),
                horizon=int(action_loss.shape[1]),
                action_dim=int(action_loss.shape[2]),
                rotation_weight=self.recovery_rotation_loss_weight,
            )
            action_weights = action_weights * recovery_weights
        action_loss = action_loss * action_weights
        action_loss = reduce(action_loss, "b ... -> b (...)", "mean")
        action_loss = action_loss.mean()

        loss_dict = {
            "bc_loss": action_loss.item(),
        }

        loss = action_loss
        if aux_geometry_pred is not None:
            # Match GAP's future-latent objective: predict the geometry at the
            # final frame in the sampled action horizon from only observed
            # frames. The flow value is label-only and never enters the policy
            # condition or inference path.
            aux_geometry_target = self._aux_geometry_target(nobs)
            geometry_loss = F.mse_loss(
                aux_geometry_pred, aux_geometry_target, reduction="mean"
            )
            loss = action_loss + self.aux_geometry_loss_weight * geometry_loss
            loss_dict["aux_geometry_loss"] = geometry_loss.item()
            loss_dict["total_loss"] = loss.item()
        if binary_gripper_logits is not None:
            start = self.n_obs_steps - 1
            gripper_commands = batch["action"][
                :, start : start + self.n_action_steps, list(self.binary_gripper_indices)
            ]
            closed_target = (gripper_commands < 0.5).to(
                binary_gripper_logits.dtype
            )
            positive_weight = torch.full(
                (len(self.binary_gripper_indices),),
                self.binary_gripper_positive_weight,
                device=binary_gripper_logits.device,
                dtype=binary_gripper_logits.dtype,
            )
            if self.binary_gripper_retention_head is not None:
                retention_active, _retention_state = (
                    self._binary_gripper_retention_mask(nobs, batch_size)
                )
                positive_weight = torch.where(
                    retention_active,
                    torch.full_like(
                        retention_active,
                        self.binary_gripper_retention_positive_weight,
                        dtype=binary_gripper_logits.dtype,
                    ),
                    torch.full_like(
                        retention_active,
                        self.binary_gripper_positive_weight,
                        dtype=binary_gripper_logits.dtype,
                    ),
                )[:, None, :]
            gripper_loss = F.binary_cross_entropy_with_logits(
                binary_gripper_logits,
                closed_target,
                pos_weight=positive_weight,
            )
            loss = loss + self.binary_gripper_loss_weight * gripper_loss
            loss_dict["binary_gripper_loss"] = gripper_loss.item()
            loss_dict["total_loss"] = loss.item()
        if rotation_action_prediction is not None:
            if "active_arm_right" not in batch:
                raise KeyError(
                    "rotation action head requires active_arm_right batch labels"
                )
            start = self.n_obs_steps - 1
            rotation_target = batch["action"][
                :, start : start + self.n_action_steps
            ].to(
                device=rotation_action_prediction.device,
                dtype=rotation_action_prediction.dtype,
            )
            recovery_gate_positive = batch.get(
                "is_recovery_gate_positive",
                batch.get("is_failure_matched_recovery"),
            )
            if self.rotation_action_head_gated and recovery_gate_positive is None:
                raise KeyError(
                    "gated rotation head requires recovery-gate labels"
                )
            geometry_valid = self._rotation_action_axis(nobs, batch_size)[1]
            rotation_valid = geometry_valid
            if self.rotation_action_head_gated:
                rotation_valid = rotation_valid & (
                    recovery_gate_positive.reshape(-1).to(
                        device=geometry_valid.device, dtype=torch.bool
                    )
                )
            rotation_loss = _active_rotation_action_loss(
                rotation_action_prediction,
                rotation_target,
                batch["active_arm_right"],
                is_recovery=batch.get("is_recovery_augmented"),
                recovery_weight=self.rotation_action_head_recovery_weight,
                valid_mask=rotation_valid,
            )
            loss = loss + self.rotation_action_head_loss_weight * rotation_loss
            loss_dict["rotation_action_head_loss"] = rotation_loss.item()
            loss_dict["total_loss"] = loss.item()
        if rotation_action_gate_logits is not None:
            recovery_gate_positive = batch.get(
                "is_recovery_gate_positive",
                batch.get("is_failure_matched_recovery"),
            )
            if recovery_gate_positive is None:
                raise KeyError(
                    "rotation gate requires recovery-gate labels"
                )
            gate_target = recovery_gate_positive.reshape(-1).to(
                device=rotation_action_gate_logits.device,
                dtype=rotation_action_gate_logits.dtype,
            )
            gate_loss = F.binary_cross_entropy_with_logits(
                rotation_action_gate_logits,
                gate_target,
                pos_weight=torch.as_tensor(
                    self.rotation_action_gate_positive_weight,
                    device=rotation_action_gate_logits.device,
                    dtype=rotation_action_gate_logits.dtype,
                ),
            )
            loss = loss + self.rotation_action_gate_loss_weight * gate_loss
            loss_dict["rotation_action_gate_loss"] = gate_loss.item()
            loss_dict["total_loss"] = loss.item()
        if rotation_action_translation_prediction is not None:
            recovery_gate_positive = batch.get(
                "is_recovery_gate_positive",
                batch.get("is_failure_matched_recovery"),
            )
            if recovery_gate_positive is None:
                raise KeyError(
                    "translation compensation requires recovery-gate labels"
                )
            start = self.n_obs_steps - 1
            translation_target = batch["action"][
                :, start : start + self.n_action_steps
            ].to(
                device=rotation_action_translation_prediction.device,
                dtype=rotation_action_translation_prediction.dtype,
            )
            geometry_valid = self._recovery_geometry_valid(nobs, batch_size)
            translation_recovery_positive = batch.get(
                "is_translation_recovery_positive",
                recovery_gate_positive,
            )
            failure_valid = geometry_valid & (
                translation_recovery_positive.reshape(-1).to(
                    device=geometry_valid.device, dtype=torch.bool
                )
            )
            translation_loss = _active_translation_action_loss(
                rotation_action_translation_prediction,
                translation_target,
                batch["active_arm_right"],
                failure_valid,
                endpoint_weight=self.rotation_action_translation_endpoint_loss_weight,
            )
            loss = (
                loss
                + self.rotation_action_translation_loss_weight * translation_loss
            )
            loss_dict["rotation_action_translation_loss"] = translation_loss.item()
            loss_dict["total_loss"] = loss.item()

        # print(f"t2-t1: {t2-t1:.3f}")
        # print(f"t3-t2: {t3-t2:.3f}")
        # print(f"t4-t3: {t4-t3:.3f}")
        # print(f"t5-t4: {t5-t4:.3f}")
        # print(f"t6-t5: {t6-t5:.3f}")

        return loss, loss_dict
