from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
DP3_SCRIPTS = REPO_ROOT / "policy" / "DP3" / "scripts"
if str(DP3_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(DP3_SCRIPTS))

from ndf_feature_utils import (  # noqa: E402
    normalize_object_point_cloud_with_transform,
)
import ndf_robot.model.vnn_occupancy_net_pointnet_dgcnn as vnn_occupancy_network  # noqa: E402


class NdfPointwiseAdapter:
    """Evaluate frozen NDF descriptors at caller-selected object points."""

    scalar_dim = 256

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "cpu",
        include_vector_direction: bool = True,
    ) -> None:
        self.checkpoint = str(Path(checkpoint).expanduser().resolve())
        self.device = torch.device(device)
        try:
            state = torch.load(self.checkpoint, map_location=self.device, weights_only=True)
        except TypeError:
            state = torch.load(self.checkpoint, map_location=self.device)
        has_vector_head = any(str(key).startswith("decoder.vector_basis") for key in state)
        self.include_vector_direction = bool(include_vector_direction and has_vector_head)
        self.model = vnn_occupancy_network.VNNOccNet(
            latent_dim=self.scalar_dim,
            model_type="pointcloud",
            return_features=True,
            return_vector_features=self.include_vector_direction,
            vector_feature_dim=16 if self.include_vector_direction else 0,
            acts="last",
            sigmoid=True,
        )
        incompatible = self.model.load_state_dict(state, strict=False)
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
        allowed_vector = {
            "decoder.vector_basis.map_to_feat.weight",
            "decoder.fc_vec_alpha.weight",
            "decoder.fc_vec_alpha.bias",
        }
        if missing or (set(unexpected) - allowed_vector):
            raise RuntimeError(
                f"NDF checkpoint mismatch: missing={missing[:8]}, unexpected={unexpected[:8]}"
            )
        self.model.to(self.device).eval()
        self.output_dim = self.scalar_dim + (3 if self.include_vector_direction else 0)

    @torch.no_grad()
    def encode(
        self,
        support_points_world: np.ndarray,
        query_points_world: np.ndarray,
    ) -> np.ndarray:
        support = np.asarray(support_points_world, dtype=np.float32)
        query = np.asarray(query_points_world, dtype=np.float32)
        if support.ndim != 2 or support.shape[1] != 3:
            raise ValueError(f"support must have shape (N,3), got {support.shape}")
        if query.ndim != 2 or query.shape[1] != 3:
            raise ValueError(f"query must have shape (M,3), got {query.shape}")
        normalized_support, center, scale = normalize_object_point_cloud_with_transform(
            support
        )
        normalized_query = (query - center[None, :]) / max(float(scale), 1e-6)
        support_tensor = torch.from_numpy(normalized_support[None]).to(self.device)
        query_tensor = torch.from_numpy(normalized_query[None]).to(self.device)
        latent = self.model.extract_latent({"point_cloud": support_tensor})
        if self.include_vector_direction:
            features, vector_direction = self.model.forward_latent(
                latent, query_tensor, return_vector_features=True
            )
            if vector_direction.shape != (1, len(query), 3):
                raise RuntimeError(
                    f"unexpected NDF vector feature shape: {tuple(vector_direction.shape)}"
                )
            features = torch.cat((features, vector_direction), dim=-1)
        else:
            features = self.model.forward_latent(latent, query_tensor)
        if features.shape != (1, len(query), self.output_dim):
            raise RuntimeError(
                "unexpected NDF feature shape: "
                f"expected {(1, len(query), self.output_dim)}, got {tuple(features.shape)}"
            )
        return features[0].detach().cpu().numpy().astype(np.float32)
