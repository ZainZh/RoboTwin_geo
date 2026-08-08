#!/usr/bin/env python3
"""Extract official UTONIA point descriptors for the task-flow episodes.

This adapter intentionally targets the public ``Pointcept/Utonia`` inference
package instead of the repository's legacy, currently unavailable
``3d_semantic_train`` wrapper.  Input RGB in RoboTwin camera clouds is stored in
[0, 1], while UTONIA's NormalizeColor transform expects [0, 255], so RGB is
scaled exactly once before the official transform.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def estimate_outward_normals(points: np.ndarray, neighbors: int = 16) -> np.ndarray:
    value = np.asarray(points, dtype=np.float32)
    if len(value) < 4:
        return np.zeros_like(value)
    count = max(3, min(int(neighbors), len(value) - 1))
    square_distance = np.sum(
        (value[:, None, :] - value[None, :, :]) ** 2, axis=-1
    )
    np.fill_diagonal(square_distance, np.inf)
    neighbor_indices = np.argpartition(square_distance, kth=count - 1, axis=1)[
        :, :count
    ]
    normals = np.empty_like(value)
    center = value.mean(axis=0)
    for index, indices in enumerate(neighbor_indices):
        local = value[indices]
        local = local - local.mean(axis=0, keepdims=True)
        _, eigenvectors = np.linalg.eigh(local.T @ local / max(len(local), 1))
        normal = eigenvectors[:, 0].astype(np.float32)
        if float(np.dot(normal, value[index] - center)) < 0.0:
            normal = -normal
        normals[index] = normal / max(float(np.linalg.norm(normal)), 1e-8)
    return normals


def normalize_object_coordinates(points: np.ndarray) -> np.ndarray:
    value = np.asarray(points, dtype=np.float32)
    value = value - value.mean(axis=0, keepdims=True)
    radius = max(float(np.linalg.norm(value, axis=1).max()), 1e-8)
    value = value / radius
    minimum = value.min(axis=0)
    maximum = value.max(axis=0)
    shift = np.asarray(
        (
            0.5 * (minimum[0] + maximum[0]),
            0.5 * (minimum[1] + maximum[1]),
            minimum[2],
        ),
        dtype=np.float32,
    )
    return (value - shift).astype(np.float32)


def build_transform(utonia, grid_size: float):
    return utonia.transform.Compose(
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


def move_tensors(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_tensors(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_tensors(item, device) for item in value]
    return value


def upcast_to_input(point, levels: int) -> torch.Tensor:
    for _ in range(int(levels)):
        if "pooling_parent" not in point.keys():
            break
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = torch.cat((parent.feat, point.feat[inverse]), dim=-1)
        point = parent
    while "pooling_parent" in point.keys():
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = point.feat[inverse]
        point = parent
    return point.feat[point.inverse]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--utonia-root", type=Path, required=True)
    result.add_argument("--checkpoint", default="utonia")
    result.add_argument("--repo-id", default="Pointcept/Utonia")
    result.add_argument("--download-root", type=Path)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--grid-size", type=float, default=0.01)
    result.add_argument("--upcast-levels", type=int, default=4)
    result.add_argument("--normal-neighbors", type=int, default=16)
    result.add_argument("--seed", type=int, default=73)
    result.add_argument(
        "--color-mode", choices=("rgb", "zero"), default="rgb"
    )
    return result


def main() -> None:
    args = parser().parse_args()
    sys.path.insert(0, str(args.utonia_root.resolve()))
    import utonia  # type: ignore

    utonia.utils.set_seed(int(args.seed))
    custom_config = {
        "enable_flash": False,
        "enc_patch_size": [1024 for _ in range(5)],
    }
    model = utonia.load(
        args.checkpoint,
        repo_id=args.repo_id,
        download_root=(
            str(args.download_root.resolve()) if args.download_root else None
        ),
        custom_config=custom_config,
    )
    device = torch.device(args.device)
    model = model.to(device).eval()
    transform = build_transform(utonia, float(args.grid_size))

    with np.load(args.dataset, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    clouds = np.asarray(payload["points_a_xyzrgb"], dtype=np.float32)
    features = []
    for episode, cloud in enumerate(clouds):
        world_xyz = cloud[:, :3]
        coordinates = normalize_object_coordinates(world_xyz)
        colors = np.clip(cloud[:, 3:6], 0.0, 1.0) * 255.0
        if args.color_mode == "zero":
            colors = np.zeros_like(colors)
        normals = estimate_outward_normals(
            world_xyz, neighbors=int(args.normal_neighbors)
        )
        np.random.seed(int(args.seed) + episode)
        point = transform(
            {
                "coord": coordinates.copy(),
                "color": colors.astype(np.float32),
                "normal": normals.astype(np.float32),
            }
        )
        point = utonia.data.collate_fn([point])
        point = move_tensors(point, device)
        with torch.inference_mode():
            encoded = model(point)
            descriptor = upcast_to_input(encoded, int(args.upcast_levels))
        value = descriptor.detach().float().cpu().numpy().astype(np.float32)
        if value.shape[0] != cloud.shape[0]:
            raise RuntimeError(
                f"episode {episode}: expected {cloud.shape[0]} descriptors, got {value.shape[0]}"
            )
        if not np.all(np.isfinite(value)):
            raise RuntimeError(f"episode {episode}: UTONIA produced non-finite features")
        features.append(value)
        print(
            json.dumps(
                {
                    "episode": episode,
                    "points": int(len(value)),
                    "feature_dim": int(value.shape[1]),
                }
            ),
            flush=True,
        )

    payload["utonia_features"] = np.asarray(features, dtype=np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    metadata = {
        "schema_version": 1,
        "dataset": str(args.dataset.resolve()),
        "output": str(args.output.resolve()),
        "utonia_root": str(args.utonia_root.resolve()),
        "checkpoint": args.checkpoint,
        "repo_id": args.repo_id,
        "grid_size": float(args.grid_size),
        "upcast_levels": int(args.upcast_levels),
        "normal_neighbors": int(args.normal_neighbors),
        "color_mode": args.color_mode,
        "input_rgb_range": [float(clouds[:, :, 3:6].min()), float(clouds[:, :, 3:6].max())],
        "rgb_scaled_to_255_before_official_normalize_color": True,
        "episodes": int(len(features)),
        "points": int(payload["utonia_features"].shape[1]),
        "feature_dim": int(payload["utonia_features"].shape[2]),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
