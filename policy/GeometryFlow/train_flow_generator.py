#!/usr/bin/env python3
"""Predict rigid 3D object task flow from initial camera point clouds.

This is a small-data screening model, not a generative Flow-Matching model.
It predicts a sequence of rigid transforms and applies them to visible object
anchors, making every output track exactly shape preserving.  Raw, PCA, and
frozen-NDF variants use the same architecture and point samples.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .geometry import rotation_6d_to_matrix
from .train_task_flow_benchmark import FOLDS


FEATURE_DIM = 19


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class CoordinateNormalization:
    xyz_mean: np.ndarray
    xyz_std: np.ndarray
    rgb_mean: np.ndarray
    rgb_std: np.ndarray
    flow_scale: float


@dataclass
class RepresentationTransform:
    mode: str
    pca_mean: np.ndarray
    pca_std: np.ndarray
    ndf_mean: np.ndarray
    ndf_components: np.ndarray
    ndf_score_std: np.ndarray
    ndf_vector_scale: float


def safe_std(value: np.ndarray, axis=0) -> np.ndarray:
    return np.maximum(np.asarray(value.std(axis=axis), dtype=np.float32), 1e-5)


def fit_coordinate_normalization(
    payload: dict[str, np.ndarray],
    train_indices: np.ndarray,
    *,
    canonical_target_frame: bool,
) -> CoordinateNormalization:
    a_xyz = payload["points_a_xyzrgb"][train_indices, :, :3].copy()
    b_xyz = payload["points_b_xyzrgb"][train_indices, :, :3].copy()
    flow = payload["episode_flow"][train_indices].copy()
    if canonical_target_frame:
        for row, episode in enumerate(train_indices):
            frame = payload["target_frame_camera"][episode]
            a_xyz[row] = world_to_frame(a_xyz[row], frame)
            b_xyz[row] = world_to_frame(b_xyz[row], frame)
            flow[row] = world_to_frame(flow[row], frame)
    xyz = np.concatenate((a_xyz.reshape(-1, 3), b_xyz.reshape(-1, 3)), axis=0)
    rgb = np.concatenate(
        (
            payload["points_a_xyzrgb"][train_indices, :, 3:6].reshape(-1, 3),
            payload["points_b_xyzrgb"][train_indices, :, 3:6].reshape(-1, 3),
        ),
        axis=0,
    )
    displacement = flow - flow[:, :, :1]
    flow_scale = max(float(np.sqrt(np.mean(displacement**2))), 0.02)
    return CoordinateNormalization(
        xyz_mean=xyz.mean(axis=0).astype(np.float32),
        xyz_std=safe_std(xyz, axis=0),
        rgb_mean=rgb.mean(axis=0).astype(np.float32),
        rgb_std=safe_std(rgb, axis=0),
        flow_scale=flow_scale,
    )


def world_to_frame(points: np.ndarray, frame: np.ndarray) -> np.ndarray:
    value = np.asarray(points)
    transform = np.asarray(frame)
    return (value - transform[:3, 3]) @ transform[:3, :3]


def frame_to_world(points: np.ndarray, frame: np.ndarray) -> np.ndarray:
    value = np.asarray(points)
    transform = np.asarray(frame)
    return value @ transform[:3, :3].T + transform[:3, 3]


def fit_representation_transform(
    payload: dict[str, np.ndarray], train_indices: np.ndarray, mode: str
) -> RepresentationTransform:
    if mode not in {"raw", "pca", "ndf"}:
        raise ValueError(f"unknown representation {mode}")
    pca = payload["pca_features"][train_indices].reshape(-1, 3).astype(np.float64)
    pca_mean = pca.mean(axis=0)
    pca_std = np.maximum(pca.std(axis=0), 1e-5)
    ndf_mean = np.zeros((256,), dtype=np.float64)
    components = np.zeros((16, 256), dtype=np.float64)
    score_std = np.ones((16,), dtype=np.float64)
    vector_scale = 1.0
    if mode == "ndf":
        scalar = payload["ndf_features"][train_indices, :, :256].reshape(-1, 256)
        ndf_mean = scalar.mean(axis=0, dtype=np.float64)
        centered = scalar.astype(np.float64) - ndf_mean
        _, _, right = np.linalg.svd(centered, full_matrices=False)
        components = right[:16]
        scores = centered @ components.T
        score_std = np.maximum(scores.std(axis=0), 1e-5)
        vector = payload["ndf_features"][train_indices, :, 256:259]
        vector_scale = max(
            float(np.sqrt(np.mean(vector.astype(np.float64) ** 2))), 1e-5
        )
    return RepresentationTransform(
        mode=mode,
        pca_mean=pca_mean.astype(np.float32),
        pca_std=pca_std.astype(np.float32),
        ndf_mean=ndf_mean.astype(np.float32),
        ndf_components=components.astype(np.float32),
        ndf_score_std=score_std.astype(np.float32),
        ndf_vector_scale=vector_scale,
    )


def transform_representation(
    payload: dict[str, np.ndarray], transform: RepresentationTransform
) -> np.ndarray:
    episodes, points = payload["pca_features"].shape[:2]
    result = np.zeros((episodes, points, FEATURE_DIM), dtype=np.float32)
    if transform.mode == "pca":
        result[..., :3] = (
            payload["pca_features"] - transform.pca_mean[None, None]
        ) / transform.pca_std[None, None]
    elif transform.mode == "ndf":
        scalar = payload["ndf_features"][..., :256] - transform.ndf_mean[None, None]
        result[..., :16] = (scalar @ transform.ndf_components.T) / (
            transform.ndf_score_std[None, None]
        )
        result[..., 16:19] = (
            payload["ndf_features"][..., 256:259] / transform.ndf_vector_scale
        )
    return result


class FlowPredictionDataset(Dataset):
    def __init__(
        self,
        payload: dict[str, np.ndarray],
        indices: np.ndarray,
        features: np.ndarray,
        normalization: CoordinateNormalization,
        *,
        mode: str,
        training: bool,
        canonical_target_frame: bool = False,
        augmentations_per_episode: int = 1,
        point_jitter_std_m: float = 0.0,
    ) -> None:
        self.payload = payload
        self.indices = np.asarray(indices, dtype=np.int64)
        self.features = features
        self.norm = normalization
        self.mode = mode
        self.training = bool(training)
        self.canonical_target_frame = bool(canonical_target_frame)
        self.augmentations_per_episode = int(augmentations_per_episode) if training else 1
        self.point_jitter_std_m = float(point_jitter_std_m) if training else 0.0

    def __len__(self) -> int:
        return int(len(self.indices) * self.augmentations_per_episode)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        episode_index = int(self.indices[int(item) % len(self.indices)])
        a = self.payload["points_a_xyzrgb"][episode_index].copy()
        b = self.payload["points_b_xyzrgb"][episode_index].copy()
        anchors = self.payload["initial_anchors"][episode_index].copy()
        flow = self.payload["episode_flow"][episode_index].copy()
        features = self.features[episode_index].copy()
        target_frame = self.payload["target_frame_camera"][episode_index].copy()
        if self.canonical_target_frame:
            a[:, :3] = world_to_frame(a[:, :3], target_frame)
            b[:, :3] = world_to_frame(b[:, :3], target_frame)
            anchors = world_to_frame(anchors, target_frame)
            flow = world_to_frame(flow, target_frame)
            if self.mode == "ndf":
                features[:, 16:19] = features[:, 16:19] @ target_frame[:3, :3]
        if self.training and not self.canonical_target_frame:
            yaw = float(np.random.uniform(-math.pi, math.pi))
            cosine, sine = math.cos(yaw), math.sin(yaw)
            rotation = np.asarray(
                ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
                dtype=np.float32,
            )
            translation = np.asarray(
                (np.random.uniform(-0.04, 0.04), np.random.uniform(-0.04, 0.04), 0.0),
                dtype=np.float32,
            )
            a[:, :3] = a[:, :3] @ rotation.T + translation
            b[:, :3] = b[:, :3] @ rotation.T + translation
            anchors = anchors @ rotation.T + translation
            flow = flow @ rotation.T + translation
            if self.mode == "ndf":
                features[:, 16:19] = features[:, 16:19] @ rotation.T
        if self.training and self.point_jitter_std_m > 0.0:
            a[:, :3] += np.random.normal(
                scale=self.point_jitter_std_m, size=a[:, :3].shape
            ).astype(np.float32)
            b[:, :3] += np.random.normal(
                scale=self.point_jitter_std_m, size=b[:, :3].shape
            ).astype(np.float32)
        result = {
            "points_a": torch.from_numpy(
                ((a[:, :3] - self.norm.xyz_mean) / self.norm.xyz_std).astype(np.float32)
            ),
            "colors_a": torch.from_numpy(
                ((a[:, 3:6] - self.norm.rgb_mean) / self.norm.rgb_std).astype(np.float32)
            ),
            "features_a": torch.from_numpy(features.astype(np.float32)),
            "points_b": torch.from_numpy(
                ((b[:, :3] - self.norm.xyz_mean) / self.norm.xyz_std).astype(np.float32)
            ),
            "colors_b": torch.from_numpy(
                ((b[:, 3:6] - self.norm.rgb_mean) / self.norm.rgb_std).astype(np.float32)
            ),
            "anchors": torch.from_numpy(anchors.astype(np.float32)),
            "target_flow": torch.from_numpy(flow.astype(np.float32)),
            "target_frame": torch.from_numpy(target_frame.astype(np.float32)),
            "episode_index": torch.tensor(episode_index, dtype=torch.long),
        }
        return result


class PointSetEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 128) -> None:
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
        )
        self.output = nn.Sequential(
            nn.Linear(256, output_dim), nn.LayerNorm(output_dim), nn.SiLU()
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        features = self.point_mlp(points)
        pooled = torch.cat((features.mean(dim=1), features.max(dim=1).values), dim=-1)
        return self.output(pooled)


class RigidFlowGenerator(nn.Module):
    def __init__(self, flow_steps: int) -> None:
        super().__init__()
        self.flow_steps = int(flow_steps)
        self.object_encoder = PointSetEncoder(3 + 3 + FEATURE_DIM, 128)
        self.target_encoder = PointSetEncoder(6, 128)
        self.head = nn.Sequential(
            nn.Linear(256, 384),
            nn.LayerNorm(384),
            nn.SiLU(),
            nn.Dropout(0.08),
            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, (self.flow_steps - 1) * 9),
        )
        final = self.head[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        identity_6d = torch.tensor((1.0, 0.0, 0.0, 1.0, 0.0, 0.0))
        with torch.no_grad():
            final.bias.reshape(self.flow_steps - 1, 9)[:, 3:9] = identity_6d

    def forward(
        self, batch: dict[str, torch.Tensor], flow_scale: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        object_input = torch.cat(
            (batch["points_a"], batch["colors_a"], batch["features_a"]), dim=-1
        )
        target_input = torch.cat((batch["points_b"], batch["colors_b"]), dim=-1)
        context = torch.cat(
            (self.object_encoder(object_input), self.target_encoder(target_input)), dim=-1
        )
        output = self.head(context).reshape(-1, self.flow_steps - 1, 9)
        centroid_displacement = output[..., :3] * float(flow_scale)
        rotations = rotation_6d_to_matrix(output[..., 3:9])
        anchors = batch["anchors"]
        center = anchors.mean(dim=1)
        relative = anchors - center[:, None]
        transformed = torch.einsum("btij,bnj->bnti", rotations, relative)
        transformed = transformed + center[:, None, None, :] + centroid_displacement[:, None]
        flow = torch.cat((anchors[:, :, None, :], transformed), dim=2)
        identity = torch.eye(3, dtype=rotations.dtype, device=rotations.device)
        identity = identity.reshape(1, 1, 3, 3).expand(len(anchors), 1, -1, -1)
        all_rotations = torch.cat((identity, rotations), dim=1)
        return flow, all_rotations


def move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def flow_loss(
    prediction: torch.Tensor, target: torch.Tensor, flow_scale: float
) -> torch.Tensor:
    steps = prediction.shape[2]
    weights = torch.linspace(0.5, 1.5, steps, device=prediction.device)
    square = ((prediction - target) / float(flow_scale)).square().mean(dim=(1, 3))
    point_loss = (square * weights[None]).mean()
    pred_center = prediction.mean(dim=1)
    target_center = target.mean(dim=1)
    center_square = ((pred_center - target_center) / float(flow_scale)).square().mean(dim=-1)
    center_loss = (center_square * weights[None]).mean()
    return point_loss + 0.35 * center_loss


@torch.no_grad()
def validation_loss(
    model: RigidFlowGenerator,
    loader: DataLoader,
    device: torch.device,
    flow_scale: float,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        prediction, _ = model(batch, flow_scale)
        loss = flow_loss(prediction, batch["target_flow"], flow_scale)
        total += float(loss.item()) * len(prediction)
        count += int(len(prediction))
    return total / max(count, 1)


def rigid_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_zero = source - source.mean(axis=0)
    target_zero = target - target.mean(axis=0)
    u, _, right = np.linalg.svd(source_zero.T @ target_zero)
    rotation = right.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        right[-1] *= -1.0
        rotation = right.T @ u.T
    return rotation


def summarize_flow(
    prediction: np.ndarray, target: np.ndarray, episode_indices: np.ndarray
) -> dict:
    error = np.linalg.norm(prediction - target, axis=-1)
    pred_center = prediction.mean(axis=1)
    target_center = target.mean(axis=1)
    center_error = np.linalg.norm(pred_center - target_center, axis=-1)
    final_rotation_error = []
    rigid_drift = []
    per_episode = []
    for row, episode in enumerate(episode_indices):
        pred_rotation = rigid_rotation(prediction[row, :, 0], prediction[row, :, -1])
        target_rotation = rigid_rotation(target[row, :, 0], target[row, :, -1])
        angle = float(
            np.rad2deg(
                Rotation.from_matrix(pred_rotation @ target_rotation.T).magnitude()
            )
        )
        final_rotation_error.append(angle)
        initial_distance = np.linalg.norm(
            prediction[row, :, None, 0] - prediction[row, None, :, 0], axis=-1
        )
        track_distance = np.linalg.norm(
            prediction[row, :, None] - prediction[row, None, :], axis=-1
        )
        drift = float(np.max(np.abs(track_distance - initial_distance[..., None])))
        rigid_drift.append(drift)
        per_episode.append(
            {
                "episode": int(episode),
                "trajectory_anchor_epe_cm": float(error[row].mean() * 100.0),
                "final_anchor_epe_cm": float(error[row, :, -1].mean() * 100.0),
                "final_centroid_error_cm": float(center_error[row, -1] * 100.0),
                "final_rotation_error_deg": angle,
            }
        )
    final_rotation_error_array = np.asarray(final_rotation_error)
    strict = (center_error[:, -1] < 0.02) & (final_rotation_error_array < 15.0)
    return {
        "episodes": int(len(prediction)),
        "trajectory_anchor_epe_cm": float(error.mean() * 100.0),
        "late_half_anchor_epe_cm": float(error[:, :, error.shape[2] // 2 :].mean() * 100.0),
        "final_anchor_epe_cm": float(error[:, :, -1].mean() * 100.0),
        "trajectory_centroid_error_cm": float(center_error.mean() * 100.0),
        "final_centroid_error_cm": float(center_error[:, -1].mean() * 100.0),
        "final_rotation_error_deg": float(final_rotation_error_array.mean()),
        "final_2cm_15deg": float(strict.mean()),
        "maximum_predicted_rigid_distance_drift_m": float(max(rigid_drift)),
        "per_episode": per_episode,
    }


@torch.no_grad()
def predict(
    model: RigidFlowGenerator,
    loader: DataLoader,
    device: torch.device,
    flow_scale: float,
    *,
    canonical_target_frame: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_indices, all_prediction, all_target = [], [], []
    for batch in loader:
        batch = move_batch(batch, device)
        prediction, _ = model(batch, flow_scale)
        all_indices.append(batch["episode_index"].cpu().numpy())
        prediction_numpy = prediction.cpu().numpy()
        target_numpy = batch["target_flow"].cpu().numpy()
        if canonical_target_frame:
            frames = batch["target_frame"].cpu().numpy()
            prediction_numpy = np.stack(
                [frame_to_world(value, frame) for value, frame in zip(prediction_numpy, frames)]
            )
            target_numpy = np.stack(
                [frame_to_world(value, frame) for value, frame in zip(target_numpy, frames)]
            )
        all_prediction.append(prediction_numpy)
        all_target.append(target_numpy)
    indices = np.concatenate(all_indices)
    order = np.argsort(indices)
    return (
        indices[order],
        np.concatenate(all_prediction)[order],
        np.concatenate(all_target)[order],
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--representations", nargs="+", default=["raw", "pca", "ndf"])
    result.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--epochs", type=int, default=160)
    result.add_argument("--patience", type=int, default=28)
    result.add_argument("--batch-size", type=int, default=16)
    result.add_argument("--augmentations-per-episode", type=int, default=8)
    result.add_argument("--point-jitter-std-m", type=float, default=0.001)
    result.add_argument(
        "--canonical-target-frame",
        action="store_true",
        help="Estimate the target T-marker frame from camera XYZRGB and predict within it.",
    )
    result.add_argument("--learning-rate", type=float, default=8e-4)
    result.add_argument("--weight-decay", type=float, default=1e-5)
    result.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    result.add_argument("--num-workers", type=int, default=0)
    return result


def main() -> None:
    args = parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.dataset, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    device = torch.device(args.device)
    shoe_ids = payload["shoe_id"]
    flow_steps = int(payload["episode_flow"].shape[2])
    results = []
    for fold_index in args.folds:
        fold = FOLDS[int(fold_index)]
        split_indices = {
            split: np.flatnonzero(np.isin(shoe_ids, ids)) for split, ids in fold.items()
        }
        normalization = fit_coordinate_normalization(
            payload,
            split_indices["train"],
            canonical_target_frame=args.canonical_target_frame,
        )
        for representation in args.representations:
            run_seed = int(args.seed) + int(fold_index) * 100
            seed_everything(run_seed)
            representation_transform = fit_representation_transform(
                payload, split_indices["train"], representation
            )
            features = transform_representation(payload, representation_transform)
            datasets = {
                split: FlowPredictionDataset(
                    payload,
                    indices,
                    features,
                    normalization,
                    mode=representation,
                    training=split == "train",
                    canonical_target_frame=args.canonical_target_frame,
                    augmentations_per_episode=args.augmentations_per_episode,
                    point_jitter_std_m=args.point_jitter_std_m,
                )
                for split, indices in split_indices.items()
            }
            loaders = {
                split: DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    shuffle=split == "train",
                    num_workers=args.num_workers,
                    pin_memory=device.type == "cuda",
                    persistent_workers=args.num_workers > 0,
                )
                for split, dataset in datasets.items()
            }
            model = RigidFlowGenerator(flow_steps).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
            )
            best_state = None
            best_validation = float("inf")
            best_epoch = -1
            stale = 0
            history = []
            for epoch in range(int(args.epochs)):
                model.train()
                total, count = 0.0, 0
                for batch in loaders["train"]:
                    batch = move_batch(batch, device)
                    optimizer.zero_grad(set_to_none=True)
                    prediction, _ = model(batch, normalization.flow_scale)
                    loss = flow_loss(
                        prediction, batch["target_flow"], normalization.flow_scale
                    )
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    total += float(loss.item()) * len(prediction)
                    count += int(len(prediction))
                train_loss = total / max(count, 1)
                val_loss = validation_loss(
                    model, loaders["validation"], device, normalization.flow_scale
                )
                history.append(
                    {"epoch": epoch, "train": train_loss, "validation": val_loss}
                )
                if val_loss < best_validation - 1e-5:
                    best_validation = val_loss
                    best_epoch = epoch
                    best_state = copy.deepcopy(model.state_dict())
                    stale = 0
                else:
                    stale += 1
                if epoch % 10 == 0 or stale == 0:
                    print(
                        json.dumps(
                            {
                                "fold": fold_index,
                                "representation": representation,
                                "epoch": epoch,
                                "train": train_loss,
                                "validation": val_loss,
                                "best_validation": best_validation,
                            }
                        ),
                        flush=True,
                    )
                if stale >= int(args.patience):
                    break
            if best_state is None:
                raise RuntimeError("training failed to produce a checkpoint")
            model.load_state_dict(best_state)
            test_indices, test_prediction, test_target = predict(
                model,
                loaders["test"],
                device,
                normalization.flow_scale,
                canonical_target_frame=args.canonical_target_frame,
            )
            summary = summarize_flow(test_prediction, test_target, test_indices)
            result = {
                "fold": int(fold_index),
                "representation": representation,
                "seed": int(run_seed),
                "split": {key: list(value) for key, value in fold.items()},
                "parameter_count": int(sum(p.numel() for p in model.parameters())),
                "best_epoch": int(best_epoch),
                "best_validation_loss": float(best_validation),
                "canonical_target_frame": bool(args.canonical_target_frame),
                "summary": summary,
                "history": history,
            }
            stem = f"fold{fold_index}_{representation}_seed{args.seed}"
            (args.output_dir / f"{stem}.json").write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8"
            )
            np.savez_compressed(
                args.output_dir / f"{stem}_predictions.npz",
                episode_indices=test_indices,
                predicted_flow=test_prediction.astype(np.float32),
                target_flow=test_target.astype(np.float32),
            )
            torch.save(
                {
                    "model": model.state_dict(),
                    "flow_steps": flow_steps,
                    "representation": representation,
                    "normalization": asdict(normalization),
                    "representation_transform": asdict(representation_transform),
                    "canonical_target_frame": bool(args.canonical_target_frame),
                },
                args.output_dir / f"{stem}.pt",
            )
            results.append(result)
            print(
                json.dumps(
                    {
                        "fold": fold_index,
                        "representation": representation,
                        "best_epoch": best_epoch,
                        "summary": summary,
                    }
                ),
                flush=True,
            )
    summary_path = args.output_dir / f"summary_seed{args.seed}.json"
    summary_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(summary_path), "runs": len(results)}, indent=2))


if __name__ == "__main__":
    main()
