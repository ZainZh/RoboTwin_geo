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
        binary_gripper_threshold=0.5,
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
        self.binary_gripper_threshold = float(binary_gripper_threshold)
        if len(set(self.binary_gripper_indices)) != len(
            self.binary_gripper_indices
        ):
            raise ValueError("binary_gripper_indices must be unique")
        if any(index < 0 or index >= action_dim for index in self.binary_gripper_indices):
            raise ValueError("binary_gripper_indices are outside the action shape")
        if self.binary_gripper_indices and not obs_as_global_cond:
            raise ValueError("binary gripper head currently requires obs_as_global_cond")
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
        if self.binary_gripper_indices:
            binary_gripper_head = BinaryGripperHead(
                input_dim=int(obs_feature_dim) * int(n_obs_steps),
                hidden_dim=int(binary_gripper_hidden_dim),
                output_dim=int(n_action_steps)
                * len(self.binary_gripper_indices),
            )
            cprint(
                "[DP3] learned binary gripper head: "
                f"indices={self.binary_gripper_indices}, "
                f"steps={n_action_steps}, weight={self.binary_gripper_loss_weight}",
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
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self._encode_observations(this_nobs)
            aux_geometry_pred = self._predict_aux_geometry(nobs_features, B)
            if self.binary_gripper_head is not None:
                binary_gripper_logits = self.binary_gripper_head(
                    nobs_features.reshape(B, -1)
                ).reshape(B, self.n_action_steps, -1)
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

        # get prediction
        result = {
            "action": action,
            "action_pred": action_pred,
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
                binary_gripper_logits = self.binary_gripper_head(
                    nobs_features.reshape(batch_size, -1)
                ).reshape(batch_size, self.n_action_steps, -1)

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
        action_loss = action_loss * loss_mask.type(action_loss.dtype)
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
            gripper_loss = F.binary_cross_entropy_with_logits(
                binary_gripper_logits,
                closed_target,
                pos_weight=positive_weight,
            )
            loss = loss + self.binary_gripper_loss_weight * gripper_loss
            loss_dict["binary_gripper_loss"] = gripper_loss.item()
            loss_dict["total_loss"] = loss.item()

        # print(f"t2-t1: {t2-t1:.3f}")
        # print(f"t3-t2: {t3-t2:.3f}")
        # print(f"t4-t3: {t4-t3:.3f}")
        # print(f"t5-t4: {t5-t4:.3f}")
        # print(f"t6-t5: {t6-t5:.3f}")

        return loss, loss_dict
