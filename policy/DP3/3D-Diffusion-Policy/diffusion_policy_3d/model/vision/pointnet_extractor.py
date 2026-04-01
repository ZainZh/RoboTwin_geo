import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import copy

from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint
import pdb


def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


class PointNetEncoderXYZRGB(nn.Module):
    """Encoder for Pointcloud"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1024,
        use_layernorm: bool = False,
        final_norm: str = "none",
        use_projection: bool = True,
        **kwargs,
    ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256, 512]
        cprint("pointnet use_layernorm: {}".format(use_layernorm), "cyan")
        cprint("pointnet use_final_norm: {}".format(final_norm), "cyan")

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[2], block_channel[3]),
        )

        if final_norm == "layernorm":
            self.final_projection = nn.Sequential(nn.Linear(block_channel[-1], out_channels),
                                                  nn.LayerNorm(out_channels))
        elif final_norm == "none":
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x


class PointNetEncoderXYZ(nn.Module):
    """Encoder for Pointcloud"""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1024,
        use_layernorm: bool = False,
        final_norm: str = "none",
        use_projection: bool = True,
        **kwargs,
    ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256]
        cprint("[PointNetEncoderXYZ] use_layernorm: {}".format(use_layernorm), "cyan")
        cprint("[PointNetEncoderXYZ] use_final_norm: {}".format(final_norm), "cyan")

        assert in_channels == 3, cprint(f"PointNetEncoderXYZ only supports 3 channels, but got {in_channels}", "red")

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )

        if final_norm == "layernorm":
            self.final_projection = nn.Sequential(nn.Linear(block_channel[-1], out_channels),
                                                  nn.LayerNorm(out_channels))
        elif final_norm == "none":
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection = use_projection
        if not use_projection:
            self.final_projection = nn.Identity()
            cprint("[PointNetEncoderXYZ] not use projection", "yellow")

        VIS_WITH_GRAD_CAM = False
        if VIS_WITH_GRAD_CAM:
            self.gradient = None
            self.feature = None
            self.input_pointcloud = None
            self.mlp[0].register_forward_hook(self.save_input)
            self.mlp[6].register_forward_hook(self.save_feature)
            self.mlp[6].register_backward_hook(self.save_gradient)

    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x

    def save_gradient(self, module, grad_input, grad_output):
        """
        for grad-cam
        """
        self.gradient = grad_output[0]

    def save_feature(self, module, input, output):
        """
        for grad-cam
        """
        if isinstance(output, tuple):
            self.feature = output[0].detach()
        else:
            self.feature = output.detach()

    def save_input(self, module, input, output):
        """
        for grad-cam
        """
        self.input_pointcloud = input[0].detach()


class DP3Encoder(nn.Module):

    def __init__(
        self,
        observation_space: Dict,
        img_crop_shape=None,
        out_channel=256,
        state_mlp_size=(64, 64),
        state_mlp_activation_fn=nn.ReLU,
        pointcloud_encoder_cfg=None,
        use_pc_color=False,
        pointnet_type="pointnet",
    ):
        super().__init__()
        self.imagination_key = "imagin_robot"
        self.rgb_image_key = "image"
        self.primary_point_cloud_key = "point_cloud"
        self.n_output_channels = 0

        self.use_imagined_robot = self.imagination_key in observation_space.keys()
        point_cloud_keys = []
        low_dim_keys = []
        for key, shape in observation_space.items():
            if key in {self.imagination_key, self.rgb_image_key}:
                continue
            shape = tuple(shape)
            if len(shape) == 2:
                point_cloud_keys.append(key)
            elif len(shape) == 1:
                low_dim_keys.append(key)

        if self.primary_point_cloud_key in point_cloud_keys:
            point_cloud_keys = [self.primary_point_cloud_key] + [
                key for key in point_cloud_keys if key != self.primary_point_cloud_key
            ]

        if len(point_cloud_keys) == 0:
            raise RuntimeError("DP3Encoder requires at least one point cloud observation key.")

        self.point_cloud_keys = point_cloud_keys
        self.point_cloud_shapes = {key: observation_space[key] for key in self.point_cloud_keys}
        self.low_dim_keys = low_dim_keys
        self.low_dim_shapes = {key: observation_space[key] for key in self.low_dim_keys}
        self.low_dim_total_dim = 0
        for shape in self.low_dim_shapes.values():
            shape = tuple(shape)
            dim = 1
            for val in shape:
                dim *= val
            self.low_dim_total_dim += dim
        if self.use_imagined_robot:
            self.imagination_shape = observation_space[self.imagination_key]
        else:
            self.imagination_shape = None

        cprint(f"[DP3Encoder] point cloud keys: {self.point_cloud_keys}", "yellow")
        cprint(f"[DP3Encoder] point cloud shapes: {self.point_cloud_shapes}", "yellow")
        cprint(f"[DP3Encoder] low-dim keys: {self.low_dim_keys}", "yellow")
        cprint(f"[DP3Encoder] low-dim total dim: {self.low_dim_total_dim}", "yellow")
        cprint(f"[DP3Encoder] imagination point shape: {self.imagination_shape}", "yellow")

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        self.point_cloud_channel_map = {}
        self.extractors = nn.ModuleDict()
        if pointnet_type != "pointnet":
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")

        for key in self.point_cloud_keys:
            cfg = copy.deepcopy(pointcloud_encoder_cfg)
            input_channels = int(tuple(self.point_cloud_shapes[key])[-1])
            if key == self.primary_point_cloud_key and not use_pc_color and input_channels > 3:
                effective_channels = 3
            else:
                effective_channels = input_channels
            self.point_cloud_channel_map[key] = effective_channels
            cfg.in_channels = effective_channels
            if effective_channels == 3 and key == self.primary_point_cloud_key:
                extractor = PointNetEncoderXYZ(**cfg)
            else:
                extractor = PointNetEncoderXYZRGB(**cfg)
            self.extractors[key] = extractor
            self.n_output_channels += out_channel

        self.state_mlp = None
        if self.low_dim_total_dim > 0:
            if len(state_mlp_size) == 0:
                raise RuntimeError("State mlp size is empty")
            elif len(state_mlp_size) == 1:
                net_arch = []
            else:
                net_arch = state_mlp_size[:-1]
            output_dim = state_mlp_size[-1]

            self.n_output_channels += output_dim
            self.state_mlp = nn.Sequential(
                *create_mlp(self.low_dim_total_dim, output_dim, net_arch, state_mlp_activation_fn)
            )

        cprint(f"[DP3Encoder] output dim: {self.n_output_channels}", "red")

    def forward(self, observations: Dict) -> torch.Tensor:
        feat_list = []
        for key in self.point_cloud_keys:
            points = observations[key]
            assert len(points.shape) == 3, cprint(f"point cloud shape: {points.shape}, length should be 3", "red")
            if key == self.primary_point_cloud_key and self.use_imagined_robot:
                img_points = observations[self.imagination_key][..., :points.shape[-1]]  # align the last dim
                points = torch.concat([points, img_points], dim=1)
            expected_channels = int(self.point_cloud_channel_map[key])
            if points.shape[-1] != expected_channels:
                points = points[..., :expected_channels]
            feat_list.append(self.extractors[key](points))
        if self.state_mlp is not None:
            low_dim_inputs = []
            for key in self.low_dim_keys:
                value = observations[key]
                low_dim_inputs.append(value.reshape(value.shape[0], -1))
            state = torch.cat(low_dim_inputs, dim=-1)
            state_feat = self.state_mlp(state)
            feat_list.append(state_feat)
        final_feat = torch.cat(feat_list, dim=-1)
        return final_feat

    def output_shape(self):
        return self.n_output_channels
