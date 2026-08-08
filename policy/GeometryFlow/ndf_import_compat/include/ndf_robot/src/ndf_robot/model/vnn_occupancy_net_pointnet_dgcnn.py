"""Expose the repository NDF model while always retaining its vector head."""

from __future__ import annotations

from policy.DP3.third_party.ndf_robot.model.vnn_occupancy_net_pointnet_dgcnn import (  # noqa: F401
    VNNOccNet as _VNNOccNet,
)


def VNNOccNet(*args, **kwargs):
    kwargs["return_vector_features"] = True
    kwargs["vector_feature_dim"] = 16
    return _VNNOccNet(*args, **kwargs)
