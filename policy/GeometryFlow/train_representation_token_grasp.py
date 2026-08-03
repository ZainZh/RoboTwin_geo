#!/usr/bin/env python3
"""Fair object-disjoint screen for point representations in grasp prediction.

Every condition uses the same points, target, descriptor width, model
parameters, split, optimizer, and action head.  The two aggregation modes only
change the memory exposed to the grasp queries:

``global``
    one mean/max-pooled object token;
``token``
    the same pooled token plus all aligned per-point tokens.

High-dimensional NDF and UTONIA descriptors are standardized and projected to
a common width by a PCA fitted on training objects only.  A deterministic
within-cloud descriptor shuffle is available as a causal control for local
point/feature correspondence.

This is an inexpensive grasp-relation screen, not a closed-loop policy result.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .geometry import keypoints_to_transform, normalize_object_points
from .models import GraspPrediction
from .train_grasp_relation import evaluation_metrics, grasp_loss
from .train_task_flow_benchmark import FOLDS


UTONIA_FEATURE_SLICES = {
    "utonia_s0": slice(0, 54),
    "utonia_s1": slice(54, 162),
    "utonia_s2": slice(162, 378),
    "utonia_s3": slice(378, 810),
    "utonia_s4": slice(810, 1386),
}
REPRESENTATIONS = {
    "raw",
    "rgb",
    "pca",
    "ndf",
    "ndf_vector",
    "utonia",
    *UTONIA_FEATURE_SLICES,
}
AGGREGATIONS = {"global", "token"}
CORRUPTIONS = {"clean", "point_shuffle"}
FUSIONS = {"early", "gated_residual"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_aligned_payload(
    grasp_dataset: Path, feature_dataset: Path
) -> dict[str, np.ndarray]:
    with np.load(grasp_dataset, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(feature_dataset, allow_pickle=False) as archive:
        feature_payload = {
            key: np.asarray(archive[key])
            for key in (
                "points_a_xyzrgb",
                "pca_features",
                "ndf_features",
                "utonia_features",
                "episode_id",
                "shoe_id",
                "active_arm_right",
            )
            if key in archive.files
        }
    required = {
        "points_a_xyzrgb",
        "pca_features",
        "ndf_features",
        "utonia_features",
        "episode_id",
        "shoe_id",
        "active_arm_right",
    }
    missing = required - set(feature_payload)
    if missing:
        raise KeyError(f"feature dataset is missing {sorted(missing)}")
    checks = (
        (
            "episode ID",
            payload["episode_id"],
            feature_payload["episode_id"],
            0.0,
        ),
        (
            "shoe ID",
            payload["shoe_id_label_only"],
            feature_payload["shoe_id"],
            0.0,
        ),
        (
            "active arm",
            payload["active_arm_right"],
            feature_payload["active_arm_right"],
            1e-7,
        ),
        (
            "point coordinates",
            payload["points_world"],
            feature_payload["points_a_xyzrgb"][..., :3],
            1e-7,
        ),
        (
            "NDF descriptor",
            payload["ndf_features"],
            feature_payload["ndf_features"],
            1e-7,
        ),
    )
    diagnostics = {}
    for name, first, second, tolerance in checks:
        if first.shape != second.shape:
            raise ValueError(
                f"{name} shape mismatch: {first.shape} versus {second.shape}"
            )
        difference = float(np.max(np.abs(first - second)))
        diagnostics[name] = difference
        if difference > tolerance:
            raise ValueError(
                f"{name} cache mismatch: maximum absolute difference {difference}"
            )
    payload["rgb_features"] = np.asarray(
        feature_payload["points_a_xyzrgb"][..., 3:6], dtype=np.float32
    )
    payload["pca_features"] = np.asarray(
        feature_payload["pca_features"], dtype=np.float32
    )
    payload["utonia_features"] = np.asarray(
        feature_payload["utonia_features"], dtype=np.float32
    )
    payload["_alignment_diagnostics"] = diagnostics
    return payload


def representation_features(
    payload: dict[str, np.ndarray], representation: str
) -> np.ndarray | None:
    if representation == "raw":
        return None
    if representation == "rgb":
        return payload["rgb_features"]
    if representation == "pca":
        return payload["pca_features"]
    if representation == "ndf":
        # The first 256 channels are invariant scalar descriptors.  The final
        # triplet is an equivariant vector and is screened separately.
        return payload["ndf_features"][..., :256]
    if representation == "ndf_vector":
        return payload["ndf_features"][..., 256:259]
    if representation == "utonia":
        return payload["utonia_features"]
    if representation in UTONIA_FEATURE_SLICES:
        return payload["utonia_features"][..., UTONIA_FEATURE_SLICES[representation]]
    raise ValueError(
        f"unknown representation {representation!r}; choose from "
        f"{sorted(REPRESENTATIONS)}"
    )


@dataclass
class DescriptorProjection:
    values: np.ndarray
    metadata: dict
    state: dict[str, np.ndarray | int | str | None]


def fit_descriptor_projection(
    features: np.ndarray | None,
    train_indices: np.ndarray,
    *,
    output_dim: int,
    random_state: int,
    episodes: int,
    points: int,
) -> DescriptorProjection:
    """Fit a train-object-only standardization/PCA and transform all episodes."""
    if output_dim <= 0:
        raise ValueError("output_dim must be positive")
    if features is None:
        values = np.zeros((episodes, points, output_dim), dtype=np.float32)
        return DescriptorProjection(
            values=values,
            metadata={
                "input_dim": 0,
                "output_dim": int(output_dim),
                "projection": "constant_zero",
                "explained_variance_ratio": None,
            },
            state={
                "kind": "constant_zero",
                "input_mean": None,
                "input_std": None,
                "components": None,
                "pca_mean": None,
                "pca_scale": None,
                "output_dim": int(output_dim),
            },
        )
    value = np.asarray(features, dtype=np.float32)
    if value.shape[:2] != (episodes, points):
        raise ValueError(
            f"descriptor point shape {value.shape[:2]} does not match "
            f"{(episodes, points)}"
        )
    if not np.all(np.isfinite(value)):
        raise ValueError("descriptor contains non-finite values")
    train = value[train_indices].reshape(-1, value.shape[-1]).astype(np.float64)
    input_mean = train.mean(axis=0)
    input_std = np.maximum(train.std(axis=0), 1e-5)
    train_standard = (train - input_mean) / input_std
    all_standard = (
        value.reshape(-1, value.shape[-1]).astype(np.float64) - input_mean
    ) / input_std
    explained = None
    components = None
    pca_mean = None
    pca_scale = None
    if value.shape[-1] > output_dim:
        projector = PCA(
            n_components=output_dim,
            whiten=True,
            svd_solver="randomized",
            random_state=int(random_state),
        )
        projector.fit(train_standard)
        transformed = projector.transform(all_standard)
        explained = float(projector.explained_variance_ratio_.sum())
        components = projector.components_.astype(np.float32)
        pca_mean = projector.mean_.astype(np.float32)
        pca_scale = np.sqrt(
            np.maximum(projector.explained_variance_, 1e-12)
        ).astype(np.float32)
        projection_name = "train_only_standardize_pca_whiten"
    else:
        transformed = np.zeros(
            (len(all_standard), output_dim), dtype=np.float64
        )
        transformed[:, : value.shape[-1]] = all_standard
        projection_name = "train_only_standardize_pad"
    transformed = transformed.reshape(episodes, points, output_dim).astype(
        np.float32
    )
    return DescriptorProjection(
        values=transformed,
        metadata={
            "input_dim": int(value.shape[-1]),
            "output_dim": int(output_dim),
            "projection": projection_name,
            "explained_variance_ratio": explained,
            "training_feature_rows": int(len(train)),
        },
        state={
            "kind": projection_name,
            "input_mean": input_mean.astype(np.float32),
            "input_std": input_std.astype(np.float32),
            "components": components,
            "pca_mean": pca_mean,
            "pca_scale": pca_scale,
            "output_dim": int(output_dim),
        },
    )


class RepresentationGraspDataset(Dataset):
    def __init__(
        self,
        payload: dict[str, np.ndarray],
        features: np.ndarray,
        indices: np.ndarray,
        *,
        training: bool,
        corruption: str,
        point_noise_m: float,
        point_dropout: float,
    ) -> None:
        if corruption not in CORRUPTIONS:
            raise ValueError(f"unknown corruption {corruption!r}")
        self.payload = payload
        self.features = np.asarray(features, dtype=np.float32)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.training = bool(training)
        self.corruption = str(corruption)
        self.point_noise_m = float(point_noise_m)
        self.point_dropout = float(point_dropout)

    def __len__(self) -> int:
        return int(len(self.indices))

    @staticmethod
    def point_permutation(episode: int, points: int) -> np.ndarray:
        generator = np.random.default_rng(1_000_003 + int(episode))
        return generator.permutation(points)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        episode = int(self.indices[item])
        points = torch.from_numpy(self.payload["points_world"][episode]).clone()
        features = torch.from_numpy(self.features[episode]).clone()
        if self.corruption == "point_shuffle":
            permutation = self.point_permutation(episode, len(points))
            features = features[torch.from_numpy(permutation)]
        if self.training:
            points = points + torch.randn_like(points) * self.point_noise_m
            if self.point_dropout > 0.0:
                keep = torch.rand((len(points),)) >= self.point_dropout
                if torch.any(keep) and not torch.all(keep):
                    replacement = int(torch.nonzero(keep, as_tuple=False)[0, 0])
                    points[~keep] = points[replacement].clone()
                    features[~keep] = features[replacement].clone()
        return {
            "points_world": points,
            "additional_features": features,
            "target_keypoints_world": torch.from_numpy(
                self.payload["target_keypoints_world"][episode]
            ).clone(),
            "target_transform_world": torch.from_numpy(
                self.payload["target_transform_world"][episode]
            ).clone(),
            "active_arm_right": torch.tensor(
                self.payload["active_arm_right"][episode], dtype=torch.float32
            ),
            "shoe_id": torch.tensor(
                self.payload["shoe_id_label_only"][episode], dtype=torch.long
            ),
            "episode_id": torch.tensor(
                self.payload["episode_id"][episode], dtype=torch.long
            ),
        }


class CrossAttentionBlock(nn.Module):
    def __init__(self, feature_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(feature_dim)
        self.memory_norm = nn.LayerNorm(feature_dim)
        self.attention = nn.MultiheadAttention(
            feature_dim, heads, dropout=dropout, batch_first=True
        )
        self.feed_forward_norm = nn.LayerNorm(feature_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
        )

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        normalized_query = self.query_norm(query)
        normalized_memory = self.memory_norm(memory)
        attended, _ = self.attention(
            normalized_query,
            normalized_memory,
            normalized_memory,
            need_weights=False,
        )
        query = query + attended
        return query + self.feed_forward(self.feed_forward_norm(query))


class RepresentationTokenGraspPredictor(nn.Module):
    """Query either a pooled object token or the aligned per-point token set."""

    def __init__(
        self,
        aggregation: str,
        *,
        fusion: str = "early",
        descriptor_dim: int = 64,
        feature_dim: int = 128,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.05,
        num_candidates: int = 1,
        num_keypoints: int = 7,
        maximum_keypoint_offset_normalized: float = 1.25,
    ) -> None:
        super().__init__()
        if aggregation not in AGGREGATIONS:
            raise ValueError(f"unknown aggregation {aggregation!r}")
        if fusion not in FUSIONS:
            raise ValueError(f"unknown fusion {fusion!r}")
        if num_candidates <= 0 or num_keypoints <= 0:
            raise ValueError("candidate and keypoint counts must be positive")
        self.aggregation = str(aggregation)
        self.fusion = str(fusion)
        self.num_candidates = int(num_candidates)
        self.num_keypoints = int(num_keypoints)
        self.maximum_keypoint_offset_normalized = float(
            maximum_keypoint_offset_normalized
        )
        if self.fusion == "early":
            self.point_projection = nn.Sequential(
                nn.Linear(3 + int(descriptor_dim), feature_dim),
                nn.LayerNorm(feature_dim),
                nn.SiLU(),
                nn.Linear(feature_dim, feature_dim),
                nn.LayerNorm(feature_dim),
            )
            self.xyz_projection = None
            self.descriptor_projection = None
            self.descriptor_gate = None
        else:
            self.point_projection = None
            self.xyz_projection = nn.Sequential(
                nn.Linear(3, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.SiLU(),
                nn.Linear(feature_dim, feature_dim),
                nn.LayerNorm(feature_dim),
            )
            # A zero descriptor remains exactly zero, making the raw condition
            # a clean architecture-matched baseline.
            self.descriptor_projection = nn.Sequential(
                nn.Linear(int(descriptor_dim), feature_dim, bias=False),
                nn.LayerNorm(feature_dim, elementwise_affine=False),
                nn.SiLU(),
                nn.Linear(feature_dim, feature_dim, bias=False),
            )
            # tanh(0)=0: training begins from an XYZ-only point encoder and
            # opens descriptor channels only when gradients support doing so.
            self.descriptor_gate = nn.Parameter(torch.zeros(feature_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=heads,
            dim_feedforward=feature_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.token_refiner = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
            norm=nn.LayerNorm(feature_dim),
        )
        self.pool_projection = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.SiLU(),
        )
        self.arm_embedding = nn.Embedding(2, feature_dim)
        self.queries = nn.Parameter(
            torch.randn(
                self.num_candidates, self.num_keypoints, feature_dim
            )
            * 0.02
        )
        self.cross_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(feature_dim, heads, dropout)
                for _ in range(layers)
            ]
        )
        self.output_norm = nn.LayerNorm(feature_dim)
        self.keypoint_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, 3),
            nn.Tanh(),
        )
        self.score_head = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, 1),
        )

    def forward(
        self,
        points_world: torch.Tensor,
        active_arm_right: torch.Tensor,
        additional_features: torch.Tensor | None = None,
    ) -> GraspPrediction:
        if additional_features is None:
            raise ValueError("a fixed-width descriptor tensor is required")
        points_normalized, center, scale = normalize_object_points(points_world)
        if additional_features.shape[:-1] != points_normalized.shape[:-1]:
            raise ValueError("point and descriptor dimensions differ")
        if self.fusion == "early":
            if self.point_projection is None:
                raise RuntimeError("early fusion projection was not constructed")
            point_input = self.point_projection(
                torch.cat((points_normalized, additional_features), dim=-1)
            )
        else:
            if (
                self.xyz_projection is None
                or self.descriptor_projection is None
                or self.descriptor_gate is None
            ):
                raise RuntimeError("gated residual projections were not constructed")
            point_input = self.xyz_projection(points_normalized)
            point_input = point_input + torch.tanh(self.descriptor_gate)[
                None, None, :
            ] * self.descriptor_projection(additional_features)
        point_tokens = self.token_refiner(point_input)
        pooled = self.pool_projection(
            torch.cat(
                (
                    torch.mean(point_tokens, dim=1),
                    torch.amax(point_tokens, dim=1),
                ),
                dim=-1,
            )
        )
        if self.aggregation == "global":
            memory = pooled[:, None, :]
        else:
            memory = torch.cat((pooled[:, None, :], point_tokens), dim=1)
        arm = self.arm_embedding(active_arm_right.reshape(-1).long())
        batch = len(points_world)
        queries = self.queries[None].expand(batch, -1, -1, -1)
        queries = queries + arm[:, None, None, :]
        query_shape = queries.shape
        queries = queries.reshape(batch, -1, query_shape[-1])
        for block in self.cross_blocks:
            queries = block(queries, memory)
        queries = self.output_norm(queries).reshape(query_shape)
        keypoints_normalized = (
            self.keypoint_head(queries)
            * self.maximum_keypoint_offset_normalized
        )
        keypoints_world = (
            keypoints_normalized * scale[:, None, None, :]
            + center[:, None, None, :]
        )
        transforms = keypoints_to_transform(keypoints_world)
        candidate_context = torch.mean(queries, dim=2)
        score_input = torch.cat(
            (
                candidate_context,
                pooled[:, None, :].expand(-1, self.num_candidates, -1),
                arm[:, None, :].expand(-1, self.num_candidates, -1),
            ),
            dim=-1,
        )
        scores = self.score_head(score_input).squeeze(-1)
        return GraspPrediction(
            transforms=transforms,
            keypoints_world=keypoints_world,
            scores=scores,
            keypoints_normalized=keypoints_normalized,
            object_center=center,
            object_scale=scale,
        )


def move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def validation_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_kwargs: dict,
) -> float:
    model.eval()
    total = 0.0
    samples = 0
    for cpu_batch in loader:
        batch = move_batch(cpu_batch, device)
        prediction = model(
            batch["points_world"],
            batch["active_arm_right"],
            batch["additional_features"],
        )
        loss, _ = grasp_loss(prediction, batch, **loss_kwargs)
        total += float(loss.item()) * len(batch["points_world"])
        samples += int(len(batch["points_world"]))
    return total / max(samples, 1)


def train_one(
    *,
    payload: dict[str, np.ndarray],
    projected: DescriptorProjection,
    indices: dict[str, np.ndarray],
    representation: str,
    aggregation: str,
    corruption: str,
    fold_index: int,
    base_seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict, dict]:
    run_seed = int(base_seed) + int(fold_index) * 100
    seed_everything(run_seed)
    datasets = {
        split: RepresentationGraspDataset(
            payload,
            projected.values,
            split_indices,
            training=split == "train",
            corruption=corruption,
            point_noise_m=args.point_noise_m,
            point_dropout=args.point_dropout,
        )
        for split, split_indices in indices.items()
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            generator=(
                torch.Generator().manual_seed(run_seed)
                if split == "train"
                else None
            ),
        )
        for split, dataset in datasets.items()
    }
    model = RepresentationTokenGraspPredictor(
        aggregation,
        fusion=args.fusion,
        descriptor_dim=args.descriptor_dim,
        feature_dim=args.feature_dim,
        heads=args.heads,
        layers=args.layers,
        dropout=args.dropout,
        num_candidates=args.num_candidates,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_kwargs = {
        "rotation_loss_weight": args.rotation_loss_weight,
        "score_loss_weight": args.score_loss_weight,
        "attention_loss_weight": 0.0,
        "attention_sigma_normalized": 0.15,
        "diversity_loss_weight": args.diversity_loss_weight,
        "diversity_margin_normalized": 0.12,
    }
    best_state = None
    best_validation = float("inf")
    best_epoch = -1
    stale = 0
    history = []
    for epoch in range(int(args.epochs)):
        model.train()
        train_total = 0.0
        train_samples = 0
        for cpu_batch in loaders["train"]:
            batch = move_batch(cpu_batch, device)
            prediction = model(
                batch["points_world"],
                batch["active_arm_right"],
                batch["additional_features"],
            )
            loss, _ = grasp_loss(prediction, batch, **loss_kwargs)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_total += float(loss.item()) * len(batch["points_world"])
            train_samples += int(len(batch["points_world"]))
        validation = validation_loss(
            model, loaders["validation"], device, loss_kwargs
        )
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": train_total / max(train_samples, 1),
                "validation_loss": validation,
            }
        )
        if validation < best_validation - 1e-6:
            best_validation = validation
            best_epoch = int(epoch)
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch == 0 or epoch % 20 == 0 or stale == 0:
            print(
                json.dumps(
                    {
                        "fold": int(fold_index),
                        "base_seed": int(base_seed),
                        "representation": representation,
                        "aggregation": aggregation,
                        "corruption": corruption,
                        "epoch": int(epoch),
                        "train_loss": history[-1]["train_loss"],
                        "validation_loss": validation,
                        "best_validation_loss": best_validation,
                    }
                ),
                flush=True,
            )
        if stale >= int(args.patience):
            break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    result = {
        "schema_version": 1,
        "experiment": "representation_token_grasp",
        "grasp_dataset": str(args.grasp_dataset.resolve()),
        "feature_dataset": str(args.feature_dataset.resolve()),
        "fold": int(fold_index),
        "split": {
            key: [int(value) for value in FOLDS[fold_index][key]]
            for key in ("train", "validation", "test")
        },
        "base_seed": int(base_seed),
        "run_seed": int(run_seed),
        "representation": representation,
        "aggregation": aggregation,
        "fusion": args.fusion,
        "corruption": corruption,
        "condition": (
            f"{aggregation}:{representation}"
            + ("" if args.fusion == "early" else f":{args.fusion}")
            + ("" if corruption == "clean" else f":{corruption}")
        ),
        "parameters": int(sum(value.numel() for value in model.parameters())),
        "descriptor_projection": projected.metadata,
        "train_samples": int(len(indices["train"])),
        "validation_samples": int(len(indices["validation"])),
        "test_samples": int(len(indices["test"])),
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_validation),
        "test_metrics": evaluation_metrics(model, loaders["test"], device),
        "history": history,
    }
    checkpoint = {
        "schema_version": 1,
        "model_state_dict": best_state,
        "model": {
            "aggregation": aggregation,
            "fusion": args.fusion,
            "descriptor_dim": int(args.descriptor_dim),
            "feature_dim": int(args.feature_dim),
            "heads": int(args.heads),
            "layers": int(args.layers),
            "dropout": float(args.dropout),
            "num_candidates": int(args.num_candidates),
        },
        "representation": representation,
        "corruption": corruption,
        "descriptor_projection": projected.state,
        "training_summary": result,
    }
    return result, checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grasp-dataset", type=Path, required=True)
    parser.add_argument("--feature-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--representations",
        nargs="+",
        default=["raw", "rgb", "pca", "ndf", "utonia"],
        choices=sorted(REPRESENTATIONS),
    )
    parser.add_argument(
        "--aggregations",
        nargs="+",
        default=["global", "token"],
        choices=sorted(AGGREGATIONS),
    )
    parser.add_argument(
        "--fusion",
        choices=sorted(FUSIONS),
        default="early",
        help="Early concatenation or an XYZ-preserving zero-gated descriptor residual.",
    )
    parser.add_argument(
        "--point-shuffle-representations",
        nargs="*",
        default=["ndf", "utonia"],
        choices=sorted(REPRESENTATIONS - {"raw"}),
    )
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--descriptor-dim", type=int, default=64)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--num-candidates", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--rotation-loss-weight", type=float, default=0.5)
    parser.add_argument("--score-loss-weight", type=float, default=0.0)
    parser.add_argument("--diversity-loss-weight", type=float, default=0.0)
    parser.add_argument("--point-noise-m", type=float, default=0.001)
    parser.add_argument("--point-dropout", type=float, default=0.10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--save-checkpoints", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if any(fold < 0 or fold >= len(FOLDS) for fold in args.folds):
        raise ValueError(f"folds must be in [0, {len(FOLDS) - 1}]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = load_aligned_payload(args.grasp_dataset, args.feature_dataset)
    shoe_ids = np.asarray(payload["shoe_id_label_only"], dtype=np.int64)
    episodes, points = payload["points_world"].shape[:2]
    device = torch.device(args.device)
    run_summaries = []
    for fold_index in args.folds:
        fold = FOLDS[int(fold_index)]
        indices = {
            split: np.flatnonzero(np.isin(shoe_ids, ids))
            for split, ids in fold.items()
        }
        if sum(len(value) for value in indices.values()) != episodes:
            raise ValueError(f"fold {fold_index} does not cover all episodes")
        projections = {}
        for representation in args.representations:
            projections[representation] = fit_descriptor_projection(
                representation_features(payload, representation),
                indices["train"],
                output_dim=args.descriptor_dim,
                random_state=7_919 + int(fold_index),
                episodes=episodes,
                points=points,
            )
        for base_seed in args.seeds:
            for representation in args.representations:
                projected = projections[representation]
                conditions = [
                    (aggregation, "clean")
                    for aggregation in args.aggregations
                ]
                if representation in args.point_shuffle_representations:
                    conditions.append(("token", "point_shuffle"))
                for aggregation, corruption in conditions:
                    result, checkpoint = train_one(
                        payload=payload,
                        projected=projected,
                        indices=indices,
                        representation=representation,
                        aggregation=aggregation,
                        corruption=corruption,
                        fold_index=int(fold_index),
                        base_seed=int(base_seed),
                        args=args,
                        device=device,
                    )
                    stem = (
                        f"fold{fold_index}_{aggregation}_{representation}_"
                        f"{corruption}_seed{base_seed}"
                    )
                    (args.output_dir / f"{stem}.json").write_text(
                        json.dumps(result, indent=2) + "\n", encoding="utf-8"
                    )
                    if args.save_checkpoints:
                        torch.save(checkpoint, args.output_dir / f"{stem}.pt")
                    run_summaries.append(result)
                    print(
                        json.dumps(
                            {
                                "completed": stem,
                                "parameters": result["parameters"],
                                "best_epoch": result["best_epoch"],
                                "top1": result["test_metrics"]["aggregate"]["top1"],
                            }
                        ),
                        flush=True,
                    )
    summary = {
        "schema_version": 1,
        "experiment": "representation_token_grasp",
        "alignment_diagnostics": payload["_alignment_diagnostics"],
        "runs": run_summaries,
    }
    output = args.output_dir / "runs.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "runs": len(run_summaries)}))


if __name__ == "__main__":
    main()
