#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .geometry import rotation_angle_degrees
from .models import GraspPrediction, build_model


class GraspDataset(Dataset):
    def __init__(
        self,
        payload: dict[str, np.ndarray],
        indices: np.ndarray,
        *,
        training: bool,
        feature_mean: np.ndarray | None,
        feature_std: np.ndarray | None,
        rotate_last_feature_triplet: bool,
        yaw_augmentation_rad: float,
        translation_augmentation_m: float,
        point_noise_m: float,
        point_dropout: float,
    ) -> None:
        self.payload = payload
        self.indices = np.asarray(indices, dtype=np.int64)
        self.training = bool(training)
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.rotate_last_feature_triplet = bool(rotate_last_feature_triplet)
        self.yaw_augmentation_rad = float(yaw_augmentation_rad)
        self.translation_augmentation_m = float(translation_augmentation_m)
        self.point_noise_m = float(point_noise_m)
        self.point_dropout = float(point_dropout)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        selected = int(self.indices[index])
        points = torch.from_numpy(self.payload["points_world"][selected]).clone()
        target_keypoints = torch.from_numpy(
            self.payload["target_keypoints_world"][selected]
        ).clone()
        target_transform = torch.from_numpy(
            self.payload["target_transform_world"][selected]
        ).clone()
        features = None
        if "ndf_features" in self.payload:
            feature_value = self.payload["ndf_features"][selected]
            if self.feature_mean is not None and self.feature_std is not None:
                feature_value = (feature_value - self.feature_mean) / self.feature_std
            features = torch.from_numpy(np.asarray(feature_value, dtype=np.float32)).clone()
        if self.training:
            angle = (2.0 * torch.rand(()) - 1.0) * self.yaw_augmentation_rad
            cosine, sine = torch.cos(angle), torch.sin(angle)
            zero = torch.zeros_like(angle)
            one = torch.ones_like(angle)
            rotation = torch.stack(
                (
                    torch.stack((cosine, -sine, zero)),
                    torch.stack((sine, cosine, zero)),
                    torch.stack((zero, zero, one)),
                )
            )
            center = 0.5 * (torch.amin(points, dim=0) + torch.amax(points, dim=0))
            translation = (
                2.0 * torch.rand(3) - 1.0
            ) * self.translation_augmentation_m
            points = (points - center) @ rotation.T + center + translation
            target_keypoints = (
                (target_keypoints - center) @ rotation.T + center + translation
            )
            target_transform[:3, :3] = rotation @ target_transform[:3, :3]
            target_transform[:3, 3] = (
                rotation @ (target_transform[:3, 3] - center)
                + center
                + translation
            )
            if features is not None and self.rotate_last_feature_triplet:
                features[..., -3:] = features[..., -3:] @ rotation.T
            points = points + torch.randn_like(points) * self.point_noise_m
            if self.point_dropout > 0.0:
                keep = torch.rand((len(points),)) >= self.point_dropout
                if torch.any(keep) and not torch.all(keep):
                    replacement = int(torch.nonzero(keep, as_tuple=False)[0, 0])
                    points[~keep] = points[replacement].clone()
                    if features is not None:
                        features[~keep] = features[replacement].clone()
        result = {
            "points_world": points,
            "target_keypoints_world": target_keypoints,
            "target_transform_world": target_transform,
            "active_arm_right": torch.tensor(
                self.payload["active_arm_right"][selected], dtype=torch.float32
            ),
            "shoe_id": torch.tensor(
                self.payload["shoe_id_label_only"][selected], dtype=torch.long
            ),
            "episode_id": torch.tensor(
                self.payload["episode_id"][selected], dtype=torch.long
            ),
        }
        if features is not None:
            result["additional_features"] = features
        return result


def object_split_masks(
    shoe_ids: np.ndarray,
    train_ids: list[int],
    validation_ids: list[int],
    test_ids: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    available = set(int(value) for value in np.unique(shoe_ids))
    groups = {
        "train": set(int(value) for value in train_ids),
        "validation": set(int(value) for value in validation_ids),
        "test": set(int(value) for value in test_ids),
    }
    for name, values in groups.items():
        if not values:
            raise ValueError(f"{name} object split may not be empty")
        unknown = values - available
        if unknown:
            raise ValueError(f"{name} split has unknown shoe IDs: {sorted(unknown)}")
    if any(groups[a] & groups[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("train, validation, and test shoe IDs must be disjoint")
    assigned = set().union(*groups.values())
    if assigned != available:
        raise ValueError(f"object split must cover all IDs; missing={sorted(available-assigned)}")
    return tuple(np.isin(shoe_ids, sorted(groups[name])) for name in groups)


def target_keypoints_normalized(
    prediction: GraspPrediction, target_world: torch.Tensor
) -> torch.Tensor:
    return (
        target_world - prediction.object_center[:, None, :]
    ) / prediction.object_scale[:, None, :]


def candidate_costs(
    prediction: GraspPrediction,
    target_keypoints_world: torch.Tensor,
    target_transform_world: torch.Tensor,
    *,
    rotation_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target_normalized = target_keypoints_normalized(prediction, target_keypoints_world)
    keypoint = F.smooth_l1_loss(
        prediction.keypoints_normalized,
        target_normalized[:, None, :, :].expand_as(prediction.keypoints_normalized),
        reduction="none",
        beta=0.1,
    ).mean(dim=(-1, -2))
    target_rotation = target_transform_world[:, None, :3, :3]
    rotation = (
        prediction.transforms[..., :3, :3] - target_rotation
    ).square().mean(dim=(-1, -2))
    return keypoint + float(rotation_loss_weight) * rotation, keypoint, rotation


def gather_candidates(value: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch = torch.arange(value.shape[0], device=value.device)
    return value[batch, indices]


def grasp_loss(
    prediction: GraspPrediction,
    batch: dict[str, torch.Tensor],
    *,
    rotation_loss_weight: float,
    score_loss_weight: float,
    attention_loss_weight: float,
    attention_sigma_normalized: float,
    diversity_loss_weight: float,
    diversity_margin_normalized: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    costs, keypoint_costs, rotation_costs = candidate_costs(
        prediction,
        batch["target_keypoints_world"],
        batch["target_transform_world"],
        rotation_loss_weight=rotation_loss_weight,
    )
    winners = torch.argmin(costs.detach(), dim=1)
    winning_cost = gather_candidates(costs, winners).mean()
    score_loss = F.cross_entropy(prediction.scores, winners)
    attention_loss = torch.zeros((), device=costs.device)
    if prediction.attention is not None and attention_loss_weight > 0.0:
        target_normalized = target_keypoints_normalized(
            prediction, batch["target_keypoints_world"]
        )
        points_normalized = (
            batch["points_world"] - prediction.object_center[:, None, :]
        ) / prediction.object_scale[:, None, :]
        squared_distance = (
            target_normalized[:, :, None, :] - points_normalized[:, None, :, :]
        ).square().sum(dim=-1)
        target_attention = torch.softmax(
            -squared_distance / (2.0 * float(attention_sigma_normalized) ** 2),
            dim=-1,
        )
        winning_attention = gather_candidates(prediction.attention, winners)
        attention_loss = -(
            target_attention * torch.log(torch.clamp(winning_attention, min=1e-8))
        ).sum(dim=-1).mean()
    diversity_loss = torch.zeros((), device=costs.device)
    if prediction.keypoints_normalized.shape[1] > 1 and diversity_loss_weight > 0.0:
        pair_losses = []
        for first in range(prediction.keypoints_normalized.shape[1]):
            for second in range(first + 1, prediction.keypoints_normalized.shape[1]):
                separation = torch.sqrt(
                    (
                        prediction.keypoints_normalized[:, first]
                        - prediction.keypoints_normalized[:, second]
                    ).square().mean(dim=(-1, -2))
                    + 1e-8
                )
                pair_losses.append(
                    F.relu(float(diversity_margin_normalized) - separation)
                    / float(diversity_margin_normalized)
                )
        diversity_loss = torch.stack(pair_losses, dim=0).mean()
    total = (
        winning_cost
        + float(score_loss_weight) * score_loss
        + float(attention_loss_weight) * attention_loss
        + float(diversity_loss_weight) * diversity_loss
    )
    values = {
        "loss": float(total.detach().cpu()),
        "winning_cost": float(winning_cost.detach().cpu()),
        "keypoint_cost": float(gather_candidates(keypoint_costs, winners).mean().detach().cpu()),
        "rotation_cost": float(gather_candidates(rotation_costs, winners).mean().detach().cpu()),
        "score_loss": float(score_loss.detach().cpu()),
        "attention_loss": float(attention_loss.detach().cpu()),
        "diversity_loss": float(diversity_loss.detach().cpu()),
    }
    return total, values


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "maximum": float(array.max()),
    }


def evaluation_metrics(model, loader: DataLoader, device: torch.device) -> dict:
    records = []
    model.eval()
    with torch.no_grad():
        for cpu_batch in loader:
            batch = {key: value.to(device) for key, value in cpu_batch.items()}
            prediction = model(
                batch["points_world"],
                batch["active_arm_right"],
                batch.get("additional_features"),
            )
            target_t = batch["target_transform_world"][:, :3, 3]
            target_r = batch["target_transform_world"][:, :3, :3]
            predicted_t = prediction.transforms[..., :3, 3]
            predicted_r = prediction.transforms[..., :3, :3]
            translation = torch.linalg.norm(predicted_t - target_t[:, None, :], dim=-1)
            rotation = rotation_angle_degrees(predicted_r, target_r[:, None, :, :])
            keypoint = torch.sqrt(
                (
                    prediction.keypoints_world
                    - batch["target_keypoints_world"][:, None, :, :]
                ).square().mean(dim=(-1, -2))
            )
            top = torch.argmax(prediction.scores, dim=1)
            oracle = torch.argmin(keypoint, dim=1)
            for index in range(len(top)):
                def candidate(which: int) -> dict:
                    t_error = float(translation[index, which].cpu())
                    r_error = float(rotation[index, which].cpu())
                    return {
                        "translation_error_m": t_error,
                        "rotation_error_deg": r_error,
                        "keypoint_rmse_m": float(keypoint[index, which].cpu()),
                        "success_2cm_15deg": bool(t_error < 0.02 and r_error < 15.0),
                        "success_3cm_30deg": bool(t_error < 0.03 and r_error < 30.0),
                    }
                top_index = int(top[index])
                oracle_index = int(oracle[index])
                orthogonality = predicted_r[index].transpose(-1, -2) @ predicted_r[index]
                identity = torch.eye(3, device=device)[None]
                records.append(
                    {
                        "episode": int(batch["episode_id"][index]),
                        "shoe_id": int(batch["shoe_id"][index]),
                        "active_arm": "right" if bool(batch["active_arm_right"][index]) else "left",
                        "top1_candidate": top_index,
                        "oracle_candidate": oracle_index,
                        "top1": candidate(top_index),
                        "oracle_at_k": candidate(oracle_index),
                        "maximum_orthogonality_error": float(
                            (orthogonality - identity).abs().max().cpu()
                        ),
                        "minimum_rotation_determinant": float(
                            torch.det(predicted_r[index]).min().cpu()
                        ),
                    }
                )

    def aggregate(rows: list[dict]) -> dict:
        result = {"samples": len(rows)}
        for route in ("top1", "oracle_at_k"):
            result[route] = {
                "translation_error_m": _summary(
                    [row[route]["translation_error_m"] for row in rows]
                ),
                "rotation_error_deg": _summary(
                    [row[route]["rotation_error_deg"] for row in rows]
                ),
                "keypoint_rmse_m": _summary(
                    [row[route]["keypoint_rmse_m"] for row in rows]
                ),
                "success_2cm_15deg": float(
                    np.mean([row[route]["success_2cm_15deg"] for row in rows])
                ),
                "success_3cm_30deg": float(
                    np.mean([row[route]["success_3cm_30deg"] for row in rows])
                ),
            }
        result["maximum_orthogonality_error"] = max(
            row["maximum_orthogonality_error"] for row in rows
        )
        result["minimum_rotation_determinant"] = min(
            row["minimum_rotation_determinant"] for row in rows
        )
        return result

    by_shoe = {}
    for shoe_id in sorted({row["shoe_id"] for row in records}):
        by_shoe[str(shoe_id)] = aggregate(
            [row for row in records if row["shoe_id"] == shoe_id]
        )
    return {"aggregate": aggregate(records), "by_shoe_id": by_shoe, "records": records}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train deterministic grasp-relation models.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=(
            "direct",
            "keypoint_transport",
            "ndf_directional",
            "pca_directional",
        ),
        required=True,
    )
    parser.add_argument("--train-shoe-ids", type=int, nargs="+", required=True)
    parser.add_argument("--validation-shoe-ids", type=int, nargs="+", required=True)
    parser.add_argument("--test-shoe-ids", type=int, nargs="+", required=True)
    parser.add_argument("--num-candidates", type=int, default=2)
    parser.add_argument(
        "--feature-mode",
        choices=("raw", "all", "ndf_scalar", "ndf_vector_only"),
        default="all",
        help="Select frozen per-point features while keeping identical XYZ samples.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stop-patience", type=int, default=70)
    parser.add_argument("--rotation-loss-weight", type=float, default=0.5)
    parser.add_argument("--score-loss-weight", type=float, default=0.05)
    parser.add_argument("--attention-loss-weight", type=float, default=0.01)
    parser.add_argument("--attention-sigma-normalized", type=float, default=0.15)
    parser.add_argument("--diversity-loss-weight", type=float, default=0.02)
    parser.add_argument("--diversity-margin-normalized", type=float, default=0.12)
    parser.add_argument("--yaw-augmentation-rad", type=float, default=float(np.pi))
    parser.add_argument("--translation-augmentation-m", type=float, default=0.02)
    parser.add_argument("--point-noise-m", type=float, default=0.001)
    parser.add_argument("--point-dropout", type=float, default=0.10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    with np.load(args.dataset) as stored:
        payload = {key: np.asarray(stored[key]) for key in stored.files}
    if args.feature_mode == "raw":
        payload.pop("ndf_features", None)
    elif args.feature_mode == "ndf_scalar":
        if "ndf_features" not in payload or payload["ndf_features"].shape[-1] < 256:
            raise ValueError("ndf_scalar requires at least 256 saved NDF channels")
        payload["ndf_features"] = payload["ndf_features"][..., :256]
    elif args.feature_mode == "ndf_vector_only":
        if "ndf_features" not in payload or payload["ndf_features"].shape[-1] < 259:
            raise ValueError("ndf_vector_only requires scalar+direction NDF data")
        payload["ndf_features"] = payload["ndf_features"][..., 256:259]
    elif "ndf_features" not in payload:
        raise ValueError("feature_mode=all requires a dataset with saved pointwise features")
    shoe_ids = np.asarray(payload["shoe_id_label_only"], dtype=np.int64)
    train_mask, validation_mask, test_mask = object_split_masks(
        shoe_ids,
        args.train_shoe_ids,
        args.validation_shoe_ids,
        args.test_shoe_ids,
    )
    train_indices = np.flatnonzero(train_mask)
    validation_indices = np.flatnonzero(validation_mask)
    test_indices = np.flatnonzero(test_mask)
    feature_mean = feature_std = None
    additional_feature_dim = 0
    rotate_last_feature_triplet = args.feature_mode in {"all", "ndf_vector_only"}
    if "ndf_features" in payload:
        training_features = payload["ndf_features"][train_mask].reshape(
            -1, payload["ndf_features"].shape[-1]
        )
        feature_mean = training_features.mean(axis=0).astype(np.float32)
        feature_std = np.maximum(training_features.std(axis=0), 1e-5).astype(np.float32)
        # NDF's final vector is an equivariant unit direction, not an ordinary
        # scalar channel.  Preserve its geometry and rotate it with XYZ during
        # rigid augmentation instead of per-axis whitening it.
        if rotate_last_feature_triplet:
            feature_mean[-3:] = 0.0
            feature_std[-3:] = 1.0
        additional_feature_dim = int(training_features.shape[-1])
    dataset_kwargs = {
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "rotate_last_feature_triplet": rotate_last_feature_triplet,
        "yaw_augmentation_rad": args.yaw_augmentation_rad,
        "translation_augmentation_m": args.translation_augmentation_m,
        "point_noise_m": args.point_noise_m,
        "point_dropout": args.point_dropout,
    }
    train_loader = DataLoader(
        GraspDataset(payload, train_indices, training=True, **dataset_kwargs),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation_loader = DataLoader(
        GraspDataset(payload, validation_indices, training=False, **dataset_kwargs),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        GraspDataset(payload, test_indices, training=False, **dataset_kwargs),
        batch_size=args.batch_size,
        shuffle=False,
    )
    model = build_model(
        args.model,
        num_candidates=args.num_candidates,
        additional_feature_dim=additional_feature_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loss_kwargs = {
        "rotation_loss_weight": args.rotation_loss_weight,
        "score_loss_weight": args.score_loss_weight,
        "attention_loss_weight": args.attention_loss_weight,
        "attention_sigma_normalized": args.attention_sigma_normalized,
        "diversity_loss_weight": args.diversity_loss_weight,
        "diversity_margin_normalized": args.diversity_margin_normalized,
    }
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_rows: defaultdict[str, list[float]] = defaultdict(list)
        for cpu_batch in train_loader:
            batch = {key: value.to(device) for key, value in cpu_batch.items()}
            prediction = model(
                batch["points_world"],
                batch["active_arm_right"],
                batch.get("additional_features"),
            )
            loss, values = grasp_loss(prediction, batch, **loss_kwargs)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            for key, value in values.items():
                train_rows[key].append(value)
        model.eval()
        validation_losses = []
        with torch.no_grad():
            for cpu_batch in validation_loader:
                batch = {key: value.to(device) for key, value in cpu_batch.items()}
                prediction = model(
                    batch["points_world"],
                    batch["active_arm_right"],
                    batch.get("additional_features"),
                )
                _, values = grasp_loss(prediction, batch, **loss_kwargs)
                validation_losses.append(values["loss"])
        validation_loss = float(np.mean(validation_losses))
        if epoch == 1 or epoch % 10 == 0:
            row = {
                "epoch": epoch,
                "train_loss": float(np.mean(train_rows["loss"])),
                "validation_loss": validation_loss,
            }
            history.append(row)
            print(json.dumps(row), flush=True)
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= args.early_stop_patience:
            break
    if best_state is None:
        raise RuntimeError("training produced no valid checkpoint")
    model.load_state_dict(best_state)
    summary = {
        "schema_version": 1,
        "dataset": str(args.dataset.resolve()),
        "model": args.model,
        "feature_mode": args.feature_mode,
        "num_candidates": args.num_candidates,
        "seed": args.seed,
        "device": str(device),
        "additional_feature_dim": additional_feature_dim,
        "train_shoe_ids": sorted(int(value) for value in args.train_shoe_ids),
        "validation_shoe_ids": sorted(int(value) for value in args.validation_shoe_ids),
        "test_shoe_ids": sorted(int(value) for value in args.test_shoe_ids),
        "train_samples": int(len(train_indices)),
        "validation_samples": int(len(validation_indices)),
        "test_samples": int(len(test_indices)),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "test_metrics": evaluation_metrics(model, test_loader, device),
        "history": history,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "model_state_dict": best_state,
            "model": args.model,
            "feature_mode": args.feature_mode,
            "num_candidates": args.num_candidates,
            "additional_feature_dim": additional_feature_dim,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "training_summary": summary,
        },
        output,
    )
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), **summary}, indent=2))


if __name__ == "__main__":
    main()
