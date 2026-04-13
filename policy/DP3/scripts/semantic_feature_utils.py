import json
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from object_pointcloud_utils import resample_point_cloud, strip_zero_points


REPO_ROOT = Path(__file__).resolve().parents[3]
SEM_ROOT = REPO_ROOT / "include" / "3d_semantic_train"
home_directory = os.path.expanduser('~')
if SEM_ROOT.exists() and str(SEM_ROOT) not in sys.path:
    sys.path.insert(0, str(SEM_ROOT))

from my_datasets.partnext_canonical_field import UTONIA  # noqa: E402
from models.universal_field.utonia_universal_field import UtoniaUniversalFieldNet  # noqa: E402


def _load_checkpoint(path: str | Path) -> dict:
    checkpoint_path = str(path)
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")
    except Exception:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def _build_utonia_transform(grid_size: float):
    return UTONIA.transform.Compose(
        [
            dict(
                type="GridSample",
                grid_size=float(grid_size),
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "color", "inverse"),
                feat_keys=("coord", "color", "normal"),
            ),
        ]
    )


def _normalize_support_and_query(
    support_points: np.ndarray,
    query_points: np.ndarray,
    *,
    coord_scale: float,
    center_shift_z: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    centroid = np.mean(support_points, axis=0, keepdims=True)
    support = support_points - centroid
    query = query_points - centroid

    radius = np.linalg.norm(support, axis=1).max()
    radius = max(float(radius), 1e-6)
    support = support / radius
    query = query / radius

    if center_shift_z:
        x_min, y_min, z_min = support.min(axis=0)
        x_max, y_max, _ = support.max(axis=0)
        shift = np.array(
            [(x_min + x_max) / 2.0, (y_min + y_max) / 2.0, z_min],
            dtype=np.float32,
        )
        support = support - shift
        query = query - shift

    return (support * float(coord_scale)).astype(np.float32), (query * float(coord_scale)).astype(np.float32)


def _estimate_normals(points: np.ndarray, k: int = 16) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if len(points) < 3:
        return np.zeros_like(points, dtype=np.float32)

    k = max(3, min(int(k), len(points) - 1))
    diff = points[:, None, :] - points[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dist2, np.inf)
    neighbor_idx = np.argpartition(dist2, kth=k, axis=1)[:, :k]

    normals = np.zeros_like(points, dtype=np.float32)
    for idx in range(len(points)):
        neighbors = points[neighbor_idx[idx]]
        centered = neighbors - neighbors.mean(axis=0, keepdims=True)
        cov = centered.T @ centered / max(len(neighbors), 1)
        eigvals, eigvecs = np.linalg.eigh(cov.astype(np.float64))
        normal = eigvecs[:, int(np.argmin(eigvals))].astype(np.float32)
        norm = np.linalg.norm(normal)
        if norm > 1e-6:
            normal = normal / norm
        normals[idx] = normal
    return normals.astype(np.float32)


def _move_to_device(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def load_semantic_model(
    checkpoint: str,
    *,
    device: torch.device,
) -> Dict[str, object]:
    checkpoint_path = Path(checkpoint).resolve()
    checkpoint_state = _load_checkpoint(checkpoint_path)
    checkpoint_args = dict(checkpoint_state.get("args", {}))
    canonical_label_names = list(checkpoint_state.get("canonical_label_names", []))
    if not canonical_label_names:
        labels_path = checkpoint_path.parent / "canonical_labels.json"
        canonical_label_names = json.loads(labels_path.read_text(encoding="utf-8"))

    model = UtoniaUniversalFieldNet(
        num_classes=len(canonical_label_names),
        utonia_checkpoint= f"{home_directory}/.cache/utonia/ckpt/utonia.pth",
        # utonia_checkpoint=checkpoint_args.get("utonia_checkpoint", "/home/zheng/.cache/utonia/ckpt/utonia.pth"),
        utonia_repo_id=checkpoint_args.get("utonia_repo_id", "Pointcept/Utonia"),
        utonia_upcast_levels=int(checkpoint_args.get("utonia_upcast_levels", 0)),
        freeze_utonia=True,
        adapter_hidden_dim=int(checkpoint_args.get("adapter_hidden_dim", 256)),
        branch_dim=int(checkpoint_args.get("branch_dim", 256)),
        sem_embedding_dim=int(checkpoint_args.get("sem_embedding_dim", 128)),
        geo_embedding_dim=int(checkpoint_args.get("geo_embedding_dim", 128)),
        num_anchors=int(checkpoint_args.get("num_anchors", 256)),
        rbf_sigma=float(checkpoint_args.get("rbf_sigma", 0.12)),
        triplane_resolution=int(checkpoint_args.get("triplane_resolution", 64)),
        triplane_padding=float(checkpoint_args.get("triplane_padding", 0.05)),
    )
    projector_state = checkpoint_state["projector_state_dict"]
    model.semantic_adapter.load_state_dict(projector_state["semantic_adapter"])
    if "semantic_local_fusion" in projector_state:
        model.semantic_local_fusion.load_state_dict(projector_state["semantic_local_fusion"])
    model.semantic_decoder.load_state_dict(projector_state["semantic_decoder"])
    if "encoder_state_dict" in checkpoint_state:
        model.feature_extractor.encoder.load_state_dict(checkpoint_state["encoder_state_dict"])
    model = model.to(device)
    model.eval()

    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_args": checkpoint_args,
        "canonical_label_names": canonical_label_names,
        "model": model,
        "device": device,
        "utonia_transform": _build_utonia_transform(float(checkpoint_args.get("grid_size", 0.01))),
        "sem_embedding_dim": int(checkpoint_args.get("sem_embedding_dim", 128)),
    }


@torch.no_grad()
def compute_semantic_pointwise_cloud(
    artifacts: Dict[str, object],
    object_point_cloud: np.ndarray,
    *,
    target_num_points: int,
) -> np.ndarray:
    point_cloud = strip_zero_points(np.asarray(object_point_cloud, dtype=np.float32))
    sem_dim = int(artifacts["sem_embedding_dim"])
    if len(point_cloud) == 0:
        return np.zeros((int(target_num_points), 3 + sem_dim), dtype=np.float32)

    support_pc = point_cloud.astype(np.float32)
    query_pc = resample_point_cloud(point_cloud, int(target_num_points))

    support_xyz = support_pc[:, :3].astype(np.float32)
    support_rgb = support_pc[:, 3:6].astype(np.float32) if support_pc.shape[1] >= 6 else np.zeros((len(support_pc), 3), dtype=np.float32)
    support_normals = _estimate_normals(support_xyz)
    query_world_xyz = query_pc[:, :3].astype(np.float32)

    checkpoint_args = artifacts["checkpoint_args"]
    support_normed, query_normed = _normalize_support_and_query(
        support_xyz,
        query_world_xyz,
        coord_scale=float(checkpoint_args.get("coord_scale", 1.0)),
        center_shift_z=not bool(checkpoint_args.get("disable_z_shift", False)),
    )

    utonia_input = artifacts["utonia_transform"](
        {
            "coord": support_normed.copy(),
            "color": support_rgb.copy(),
            "normal": support_normals.copy(),
        }
    )
    utonia_input = UTONIA.data.collate_fn([utonia_input])
    device = artifacts["device"]
    utonia_input = _move_to_device(utonia_input, device)

    support_t = torch.from_numpy(support_normed[None, ...]).float().to(device)
    query_t = torch.from_numpy(query_normed[None, ...]).float().to(device)
    model = artifacts["model"]
    cache = model.encode_support(
        utonia_input=utonia_input,
        support_points=support_t,
    )
    semantic_output = model.query_semantic(
        cache=cache,
        query_points=query_t,
    )
    embeddings = semantic_output["embedding"].squeeze(0).detach().cpu().numpy().astype(np.float32)
    return np.concatenate([query_world_xyz.astype(np.float32), embeddings], axis=1)
