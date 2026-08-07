from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
DP3_SCRIPTS = REPO_ROOT / "policy" / "DP3" / "scripts"
if str(DP3_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(DP3_SCRIPTS))

from ndf_feature_utils import (  # noqa: E402
    normalize_object_point_cloud_with_transform,
)
import ndf_robot.model.vnn_occupancy_net_pointnet_dgcnn as vnn_occupancy_network  # noqa: E402


def _load_state(checkpoint: str | Path, device: torch.device) -> dict:
    path = str(Path(checkpoint).expanduser().resolve())
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _attach_frame_head(model: nn.Module, state: dict) -> bool:
    has_beta = any(str(key).startswith("decoder.fc_vec_beta") for key in state)
    if has_beta:
        model.decoder.fc_vec_beta = nn.Linear(
            model.decoder.fc_vec_alpha.in_features,
            model.decoder.fc_vec_alpha.out_features,
        )
    return bool(has_beta)


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
        state = _load_state(self.checkpoint, self.device)
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
        self.has_frame_head = _attach_frame_head(self.model, state)
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
            if vector_direction.ndim == 4:
                vector_direction = vector_direction[..., 0, :]
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

    @torch.no_grad()
    def encode_batch(
        self,
        support_points_world: np.ndarray,
        query_points_world: np.ndarray | None = None,
    ) -> np.ndarray:
        """Evaluate equally-sized camera clouds in batches.

        Each support cloud is normalized independently, matching ``encode``.
        This is intended for frame-level policy datasets where calling NDF once
        per observation would otherwise dominate preprocessing time.
        """
        support = np.asarray(support_points_world, dtype=np.float32)
        query = support if query_points_world is None else np.asarray(
            query_points_world, dtype=np.float32
        )
        if support.ndim != 3 or support.shape[-1] != 3:
            raise ValueError(
                f"support batch must have shape (B,N,3), got {support.shape}"
            )
        if query.ndim != 3 or query.shape[0] != support.shape[0] or query.shape[-1] != 3:
            raise ValueError(
                "query batch must have shape (B,M,3) with the same batch size, "
                f"got {query.shape} for support {support.shape}"
            )
        normalized_support = []
        normalized_query = []
        for support_item, query_item in zip(support, query):
            item, center, scale = normalize_object_point_cloud_with_transform(
                support_item
            )
            normalized_support.append(item)
            normalized_query.append(
                (query_item - center[None, :]) / max(float(scale), 1e-6)
            )
        support_tensor = torch.from_numpy(np.stack(normalized_support)).to(self.device)
        query_tensor = torch.from_numpy(np.stack(normalized_query)).to(self.device)
        latent = self.model.extract_latent({"point_cloud": support_tensor})
        if self.include_vector_direction:
            features, vector_direction = self.model.forward_latent(
                latent, query_tensor, return_vector_features=True
            )
            if vector_direction.ndim == 4:
                vector_direction = vector_direction[..., 0, :]
            features = torch.cat((features, vector_direction), dim=-1)
        else:
            features = self.model.forward_latent(latent, query_tensor)
        expected = (len(support), query.shape[1], self.output_dim)
        if tuple(features.shape) != expected:
            raise RuntimeError(
                f"unexpected batched NDF shape: expected {expected}, "
                f"got {tuple(features.shape)}"
            )
        return features.detach().cpu().numpy().astype(np.float32)


class NdfFunctionalFrameAdapter:
    """Predict one task-aligned SO(3) frame from a full-frame NDF checkpoint.

    The legacy alpha head represents object Z and the added beta head represents
    object Y.  Both are scalar combinations of the same equivariant VN basis;
    Gram--Schmidt produces the right-handed frame ``R=[X,Y,Z]``.
    """

    scalar_dim = 256
    vector_feature_dim = 16

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "cpu",
    ) -> None:
        self.checkpoint = str(Path(checkpoint).expanduser().resolve())
        self.device = torch.device(device)
        state = _load_state(self.checkpoint, self.device)
        self.model = vnn_occupancy_network.VNNOccNet(
            latent_dim=self.scalar_dim,
            model_type="pointcloud",
            return_features=True,
            return_vector_features=True,
            vector_feature_dim=self.vector_feature_dim,
            acts="last",
            sigmoid=True,
        )
        if not _attach_frame_head(self.model, state):
            raise RuntimeError(
                "functional-frame checkpoint is missing decoder.fc_vec_beta"
            )
        incompatible = self.model.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "NDF functional-frame checkpoint mismatch: "
                f"missing={list(incompatible.missing_keys)[:8]}, "
                f"unexpected={list(incompatible.unexpected_keys)[:8]}"
            )
        self.model.to(self.device).eval()

    @staticmethod
    def frame_from_vectors(vectors: torch.Tensor) -> torch.Tensor:
        if vectors.ndim != 4 or vectors.shape[-2:] != (2, 3):
            raise ValueError(
                f"frame vectors must be [B,N,2,3], got {tuple(vectors.shape)}"
            )
        z_axis = F.normalize(vectors[..., 0, :].mean(dim=1), dim=-1, eps=1e-6)
        y_raw = vectors[..., 1, :].mean(dim=1)
        y_axis = y_raw - z_axis * torch.sum(y_raw * z_axis, dim=-1, keepdim=True)
        y_norm = torch.linalg.vector_norm(y_axis, dim=-1, keepdim=True)
        basis_index = torch.argmin(torch.abs(z_axis), dim=-1)
        fallback = F.one_hot(basis_index, num_classes=3).to(z_axis.dtype)
        fallback = fallback - z_axis * torch.sum(
            fallback * z_axis, dim=-1, keepdim=True
        )
        y_axis = torch.where(y_norm > 1e-5, y_axis, fallback)
        y_axis = F.normalize(y_axis, dim=-1, eps=1e-6)
        x_axis = F.normalize(torch.cross(y_axis, z_axis, dim=-1), dim=-1, eps=1e-6)
        y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=-1), dim=-1, eps=1e-6)
        return torch.stack((x_axis, y_axis, z_axis), dim=-1)

    def _normalized_tensors(
        self, support_points_world: np.ndarray
    ) -> torch.Tensor:
        support = np.asarray(support_points_world, dtype=np.float32)
        if support.ndim != 3 or support.shape[-1] != 3:
            raise ValueError(
                f"support batch must have shape (B,N,3), got {support.shape}"
            )
        normalized = [
            normalize_object_point_cloud_with_transform(item)[0]
            for item in support
        ]
        return torch.from_numpy(np.stack(normalized)).to(self.device)

    @torch.no_grad()
    def encode_frame_batch(
        self, support_points_world: np.ndarray
    ) -> np.ndarray:
        support = self._normalized_tensors(support_points_world)
        latent = self.model.extract_latent({"point_cloud": support})
        _, vectors = self.model.forward_latent(
            latent, support, return_vector_features=True
        )
        frame = self.frame_from_vectors(vectors)
        return frame.detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def encode_frame(self, support_points_world: np.ndarray) -> np.ndarray:
        value = np.asarray(support_points_world, dtype=np.float32)
        if value.ndim != 2 or value.shape[-1] != 3:
            raise ValueError(f"support must have shape (N,3), got {value.shape}")
        return self.encode_frame_batch(value[None])[0]
