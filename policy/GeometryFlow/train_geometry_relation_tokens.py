#!/usr/bin/env python3
"""Object-disjoint gate for descriptor-derived geometry-relation tokens.

The experiment deliberately stays inside a learned-policy formulation:

    query/reference point clouds
      -> frozen NDF/UTONIA descriptors
      -> differentiable soft correspondence
      -> point-wise relation tokens
      -> action queries
      -> grasp keypoints (a short manipulation target)

There is no stage label, analytic action prior, residual controller, or motion
planner in this module.  A reference demonstration supplies the task relation;
the network must transfer it to a held-out object instance.  All conditions use
the same pairs, token width, decoder, loss, and parameter count.  Shuffled
descriptor controls test whether local point/feature alignment is causal.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .geometry import (
    GRIPPER_KEYPOINTS_METERS,
    keypoints_to_transform,
    normalize_object_points,
    rotation_angle_degrees,
    transform_keypoints_torch,
)
from .models import GraspPrediction
from .train_representation_token_grasp import (
    CrossAttentionBlock,
    fit_descriptor_projection,
    load_aligned_payload,
    representation_features,
    seed_everything,
)
from .train_task_flow_benchmark import FOLDS


CONDITIONS = {
    "query_only",
    "raw_relation",
    "ndf_relation",
    "ndf_shuffled",
    "utonia_s0_relation",
    "utonia_s0_shuffled",
    "ndf_fused_relation",
    "ndf_fused_shuffled",
    "utonia_s0_fused_relation",
    "utonia_s0_fused_shuffled",
    "oracle_relation",
}


def normalized_cloud_numpy(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    lower = points.min(axis=1)
    upper = points.max(axis=1)
    center = 0.5 * (lower + upper)
    scale = np.maximum(np.max(upper - lower, axis=-1, keepdims=True), 1e-3)
    return ((points - center[:, None, :]) / scale[:, None, :]).astype(np.float32)


def oracle_object_coordinates(payload: dict[str, np.ndarray]) -> np.ndarray:
    """Privileged object-local coordinates used only as an upper-bound match."""
    points = np.asarray(payload["points_world"], dtype=np.float32)
    transforms = np.asarray(
        payload["object_transform_world_label_only"], dtype=np.float32
    )
    rotation = transforms[:, :3, :3]
    translation = transforms[:, :3, 3]
    local = np.einsum("bnj,bji->bni", points - translation[:, None, :], rotation)
    return normalized_cloud_numpy(local)


def pad_descriptor(values: np.ndarray, width: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.shape[-1] > int(width):
        raise ValueError(
            f"descriptor width {values.shape[-1]} exceeds requested width {width}"
        )
    result = np.zeros((*values.shape[:-1], int(width)), dtype=np.float32)
    result[..., : values.shape[-1]] = values
    return result


def condition_descriptors(
    payload: dict[str, np.ndarray],
    condition: str,
    train_indices: np.ndarray,
    *,
    descriptor_dim: int,
    fold_index: int,
) -> tuple[np.ndarray, dict]:
    episodes, points = payload["points_world"].shape[:2]
    if condition in {"query_only", "raw_relation"}:
        values = pad_descriptor(
            normalized_cloud_numpy(payload["points_world"]), descriptor_dim
        )
        return values, {
            "kind": "normalized_camera_xyz",
            "output_dim": int(descriptor_dim),
        }
    if condition == "oracle_relation":
        values = pad_descriptor(
            oracle_object_coordinates(payload), descriptor_dim
        )
        return values, {
            "kind": "privileged_object_local_xyz",
            "output_dim": int(descriptor_dim),
        }
    representation = "ndf" if condition.startswith("ndf_") else "utonia_s0"
    projected = fit_descriptor_projection(
        representation_features(payload, representation),
        np.asarray(train_indices, dtype=np.int64),
        output_dim=int(descriptor_dim),
        random_state=13_771 + int(fold_index),
        episodes=int(episodes),
        points=int(points),
    )
    return np.asarray(projected.values, dtype=np.float32), projected.metadata


def deterministic_point_indices(episode_id: int, total: int, count: int) -> np.ndarray:
    if count >= total:
        return np.arange(total, dtype=np.int64)
    generator = np.random.default_rng(4_000_037 + int(episode_id))
    return np.sort(generator.choice(total, size=int(count), replace=False))


def deterministic_shuffle(episode_id: int, count: int) -> np.ndarray:
    generator = np.random.default_rng(8_000_033 + int(episode_id))
    return generator.permutation(int(count))


def build_reference_pairs(
    payload: dict[str, np.ndarray],
    target_indices: np.ndarray,
    source_indices: np.ndarray,
    *,
    references_per_target: int,
) -> list[tuple[int, int]]:
    """Make fixed, representation-independent, cross-instance demo pairs."""
    shoe_ids = np.asarray(payload["shoe_id_label_only"], dtype=np.int64)
    arms = np.asarray(payload["active_arm_right"], dtype=np.int64)
    episode_ids = np.asarray(payload["episode_id"], dtype=np.int64)
    pairs: list[tuple[int, int]] = []
    for target in np.asarray(target_indices, dtype=np.int64):
        candidates = [
            int(source)
            for source in np.asarray(source_indices, dtype=np.int64)
            if int(shoe_ids[source]) != int(shoe_ids[target])
            and int(arms[source]) == int(arms[target])
        ]
        if not candidates:
            candidates = [
                int(source)
                for source in np.asarray(source_indices, dtype=np.int64)
                if int(shoe_ids[source]) != int(shoe_ids[target])
            ]
        if not candidates:
            raise ValueError(f"target episode {int(episode_ids[target])} has no reference")
        candidates.sort(
            key=lambda source: (
                (int(episode_ids[source]) * 1_000_003 + int(episode_ids[target]) * 97)
                % 2_147_483_647,
                int(episode_ids[source]),
            )
        )
        for source in candidates[: min(int(references_per_target), len(candidates))]:
            pairs.append((int(target), int(source)))
    return pairs


class GeometryRelationPairDataset(Dataset):
    def __init__(
        self,
        payload: dict[str, np.ndarray],
        descriptors: np.ndarray,
        pairs: list[tuple[int, int]],
        *,
        point_count: int,
        shuffled: bool,
        training: bool,
        point_noise_m: float,
    ) -> None:
        self.payload = payload
        self.descriptors = np.asarray(descriptors, dtype=np.float32)
        self.pairs = list(pairs)
        self.point_count = int(point_count)
        self.shuffled = bool(shuffled)
        self.training = bool(training)
        self.point_noise_m = float(point_noise_m)

    def __len__(self) -> int:
        return len(self.pairs)

    def _cloud(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        episode_id = int(self.payload["episode_id"][index])
        total = int(self.payload["points_world"].shape[1])
        selected = deterministic_point_indices(
            episode_id, total, min(self.point_count, total)
        )
        points = torch.from_numpy(
            np.asarray(self.payload["points_world"][index, selected], dtype=np.float32)
        ).clone()
        descriptors = torch.from_numpy(
            np.asarray(self.descriptors[index, selected], dtype=np.float32)
        ).clone()
        if self.shuffled:
            permutation = deterministic_shuffle(episode_id, len(selected))
            descriptors = descriptors[torch.from_numpy(permutation)]
        if self.training and self.point_noise_m > 0.0:
            points = points + torch.randn_like(points) * self.point_noise_m
        return points, descriptors

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        target, reference = self.pairs[item]
        query_points, query_descriptors = self._cloud(target)
        reference_points, reference_descriptors = self._cloud(reference)
        return {
            "points_world": query_points,
            "additional_features": query_descriptors,
            "reference_points_world": reference_points,
            "reference_features": reference_descriptors,
            "reference_keypoints_world": torch.from_numpy(
                np.asarray(
                    self.payload["target_keypoints_world"][reference],
                    dtype=np.float32,
                )
            ).clone(),
            "reference_transform_world": torch.from_numpy(
                np.asarray(
                    self.payload["target_transform_world"][reference],
                    dtype=np.float32,
                )
            ).clone(),
            "target_keypoints_world": torch.from_numpy(
                np.asarray(
                    self.payload["target_keypoints_world"][target], dtype=np.float32
                )
            ).clone(),
            "target_transform_world": torch.from_numpy(
                np.asarray(
                    self.payload["target_transform_world"][target], dtype=np.float32
                )
            ).clone(),
            "active_arm_right": torch.tensor(
                self.payload["active_arm_right"][target], dtype=torch.float32
            ),
            "shoe_id": torch.tensor(
                self.payload["shoe_id_label_only"][target], dtype=torch.long
            ),
            "episode_id": torch.tensor(
                self.payload["episode_id"][target], dtype=torch.long
            ),
            "reference_episode_id": torch.tensor(
                self.payload["episode_id"][reference], dtype=torch.long
            ),
        }


@dataclass
class GeometryRelationOutput:
    prediction: GraspPrediction
    cycle_loss: torch.Tensor
    mean_confidence: torch.Tensor
    mean_entropy: torch.Tensor
    sample_confidence: torch.Tensor
    functional_loss: torch.Tensor
    correspondence_loss: torch.Tensor
    affordance_loss: torch.Tensor


def gripper_x_flip(
    *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    value = torch.eye(4, dtype=dtype, device=device)
    value[1, 1] = -1.0
    value[2, 2] = -1.0
    return value


def flipped_gripper_keypoints(transform: torch.Tensor) -> torch.Tensor:
    flip = gripper_x_flip(dtype=transform.dtype, device=transform.device)
    flipped = transform @ flip
    return transform_keypoints_torch(flipped[..., :3, :3], flipped[..., :3, 3])


def symmetry_invariant_functional_coordinates(
    points_world: torch.Tensor,
    gripper_transform_world: torch.Tensor,
    object_scale: torch.Tensor,
) -> torch.Tensor:
    """Express surface points in an X-flip-invariant gripper frame."""
    rotation = gripper_transform_world[..., :3, :3]
    translation = gripper_transform_world[..., :3, 3]
    local = torch.einsum(
        "bnj,bji->bni", points_world - translation[:, None, :], rotation
    )
    local = local / object_scale[:, None, :]
    return torch.stack(
        (local[..., 0], local[..., 1].abs(), local[..., 2].abs()), dim=-1
    ).clamp(-3.0, 3.0)


def soft_correspondence(
    query_embedding: torch.Tensor,
    reference_embedding: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return query-to-reference weights, confidence, entropy and logits."""
    query = F.normalize(query_embedding, dim=-1, eps=1e-6)
    reference = F.normalize(reference_embedding, dim=-1, eps=1e-6)
    logits = torch.einsum("bnd,bmd->bnm", query, reference) / max(
        float(temperature), 1e-4
    )
    weights = torch.softmax(logits, dim=-1)
    entropy = -(weights * torch.log(weights.clamp_min(1e-8))).sum(dim=-1)
    entropy = entropy / np.log(max(int(reference.shape[1]), 2))
    confidence = 1.0 - entropy
    return weights, confidence, entropy, logits


class GeometryRelationTokenPolicy(nn.Module):
    """Action queries attend to explicit descriptor-derived relation tokens."""

    def __init__(
        self,
        *,
        descriptor_dim: int = 64,
        match_dim: int = 32,
        feature_dim: int = 128,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.05,
        temperature: float = 0.10,
        query_only: bool = False,
        matching_mode: str = "descriptor",
        task_refined_matching: bool = False,
        fusion_style: str = "matcher",
        strict_local_bottleneck: bool = False,
        maximum_keypoint_offset_normalized: float = 1.25,
    ) -> None:
        super().__init__()
        if matching_mode not in {"geometry", "descriptor", "fused"}:
            raise ValueError(
                "matching_mode must be geometry, descriptor, or fused"
            )
        self.query_only = bool(query_only)
        self.matching_mode = str(matching_mode)
        self.task_refined_matching = bool(task_refined_matching)
        if fusion_style not in {"matcher", "dual_memory"}:
            raise ValueError("fusion_style must be matcher or dual_memory")
        self.fusion_style = str(fusion_style)
        self.strict_local_bottleneck = bool(strict_local_bottleneck)
        if int(match_dim) < 3:
            raise ValueError("match_dim must be at least three")
        self.temperature = float(temperature)
        self.maximum_keypoint_offset_normalized = float(
            maximum_keypoint_offset_normalized
        )
        self.match_projection = nn.Sequential(
            nn.Linear(int(descriptor_dim), feature_dim),
            nn.LayerNorm(feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, int(match_dim)),
        )
        self.geometry_match_projection = nn.Sequential(
            nn.Linear(3, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, int(match_dim)),
        )
        # Fused conditions begin exactly at the Raw-XYZ matcher.  The frozen
        # descriptor can open only when action/cycle supervision supports it.
        self.descriptor_match_gate = nn.Parameter(torch.zeros(()))
        self.functional_head = nn.Sequential(
            nn.LayerNorm(int(match_dim)),
            nn.Linear(int(match_dim), int(match_dim)),
            nn.SiLU(),
            nn.Linear(int(match_dim), 3),
        )
        self.affordance_head = nn.Sequential(
            nn.LayerNorm(int(match_dim)),
            nn.Linear(int(match_dim), feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, 2 * 7 * 3),
            nn.Tanh(),
        )
        # Both physically equivalent parallel-gripper wrist-roll modes are
        # represented.  This prevents arbitrary demonstration label flips from
        # corrupting the correspondence interface while keeping action output
        # fully learned.
        relation_dim = 9 + 42 + 2 * int(match_dim) + 3
        self.relation_projection = nn.Sequential(
            nn.Linear(relation_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=heads,
            dim_feedforward=feature_dim * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.relation_encoder = nn.TransformerEncoder(
            encoder_layer, layers, norm=nn.LayerNorm(feature_dim)
        )
        # The semantic branch is deliberately separate from the Raw-XYZ
        # relation memory.  Its zero gate makes a shared Raw checkpoint an
        # exact functional baseline, while clean and shuffled descriptors can
        # be forked from precisely the same policy state.
        self.semantic_relation_projection = nn.Sequential(
            nn.Linear(relation_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim),
        )
        semantic_encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=heads,
            dim_feedforward=feature_dim * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.semantic_relation_encoder = nn.TransformerEncoder(
            semantic_encoder_layer, layers, norm=nn.LayerNorm(feature_dim)
        )
        self.pool_projection = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.SiLU(),
        )
        self.arm_embedding = nn.Embedding(2, feature_dim)
        self.action_queries = nn.Parameter(torch.randn(7, feature_dim) * 0.02)
        self.cross_blocks = nn.ModuleList(
            [CrossAttentionBlock(feature_dim, heads, dropout) for _ in range(layers)]
        )
        self.semantic_cross_blocks = nn.ModuleList(
            [CrossAttentionBlock(feature_dim, heads, dropout) for _ in range(layers)]
        )
        self.output_norm = nn.LayerNorm(feature_dim)
        self.keypoint_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, 3),
            nn.Tanh(),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> GeometryRelationOutput:
        query_points = batch["points_world"]
        reference_points = batch["reference_points_world"]
        query_xyz, query_center, query_scale = normalize_object_points(query_points)
        reference_xyz, reference_center, reference_scale = normalize_object_points(
            reference_points
        )
        query_descriptor = self.match_projection(batch["additional_features"])
        reference_descriptor = self.match_projection(batch["reference_features"])
        query_functional = self.functional_head(query_descriptor)
        reference_functional = self.functional_head(reference_descriptor)
        query_affordance_offsets = self.affordance_head(query_descriptor)
        query_affordance_offsets = (
            query_affordance_offsets
            * self.maximum_keypoint_offset_normalized
        )
        if self.task_refined_matching:
            query_descriptor_for_match = F.pad(
                query_functional, (0, query_descriptor.shape[-1] - 3)
            )
            reference_descriptor_for_match = F.pad(
                reference_functional, (0, reference_descriptor.shape[-1] - 3)
            )
        else:
            query_descriptor_for_match = query_descriptor
            reference_descriptor_for_match = reference_descriptor
        query_geometry = self.geometry_match_projection(query_xyz)
        reference_geometry = self.geometry_match_projection(reference_xyz)
        semantic_weights = semantic_confidence = semantic_entropy = None
        semantic_logits = None
        if self.matching_mode == "geometry":
            query_embedding = query_geometry
            reference_embedding = reference_geometry
        elif self.matching_mode == "descriptor":
            query_embedding = query_descriptor_for_match
            reference_embedding = reference_descriptor_for_match
        elif self.fusion_style == "dual_memory":
            # Raw correspondence remains the immutable policy backbone.  The
            # descriptor-derived correspondence is exposed only through the
            # separately gated semantic memory below.
            query_embedding = query_geometry
            reference_embedding = reference_geometry
            (
                semantic_weights,
                semantic_confidence,
                semantic_entropy,
                semantic_logits,
            ) = soft_correspondence(
                query_descriptor_for_match,
                reference_descriptor_for_match,
                temperature=self.temperature,
            )
        else:
            gate = torch.tanh(self.descriptor_match_gate)
            query_embedding = query_geometry + gate * query_descriptor_for_match
            reference_embedding = (
                reference_geometry + gate * reference_descriptor_for_match
            )
        weights, confidence, entropy, logits = soft_correspondence(
            query_embedding,
            reference_embedding,
            temperature=self.temperature,
        )
        matched_reference_xyz = torch.bmm(weights, reference_xyz)
        matched_reference_embedding = torch.bmm(weights, reference_embedding)
        reference_keypoints = (
            batch["reference_keypoints_world"] - reference_center[:, None, :]
        ) / reference_scale[:, None, :]
        reference_keypoints_flipped_world = flipped_gripper_keypoints(
            batch["reference_transform_world"]
        )
        reference_keypoints_flipped = (
            reference_keypoints_flipped_world - reference_center[:, None, :]
        ) / reference_scale[:, None, :]
        reference_offsets_original = (
            reference_keypoints[:, None, :, :] - reference_xyz[:, :, None, :]
        )
        reference_offsets_flipped = (
            reference_keypoints_flipped[:, None, :, :]
            - reference_xyz[:, :, None, :]
        )
        reference_offsets = torch.cat(
            (reference_offsets_original, reference_offsets_flipped), dim=2
        )
        matched_offsets = torch.einsum(
            "bnm,bmkc->bnkc", weights, reference_offsets
        ).flatten(start_dim=-2)
        expected_similarity = torch.sum(weights * logits, dim=-1, keepdim=True)
        confidence_token = confidence[..., None]
        entropy_token = entropy[..., None]
        relation_parts = [
            query_xyz,
            matched_reference_xyz,
            query_xyz - matched_reference_xyz,
            matched_offsets,
            query_embedding,
            matched_reference_embedding,
            expected_similarity,
            confidence_token,
            entropy_token,
        ]
        if self.query_only:
            relation_parts = [
                query_xyz,
                *[torch.zeros_like(value) for value in relation_parts[1:]],
            ]
        relation = torch.cat(relation_parts, dim=-1)
        memory = self.relation_encoder(self.relation_projection(relation))
        pooled = self.pool_projection(
            torch.cat((memory.mean(dim=1), memory.amax(dim=1)), dim=-1)
        )
        memory = torch.cat((pooled[:, None, :], memory), dim=1)
        arm = self.arm_embedding(batch["active_arm_right"].reshape(-1).long())
        queries = self.action_queries[None, :, :] + arm[:, None, :]
        for block in self.cross_blocks:
            queries = block(queries, memory)
        if self.matching_mode == "fused" and self.fusion_style == "dual_memory":
            assert semantic_weights is not None
            assert semantic_confidence is not None
            assert semantic_entropy is not None
            assert semantic_logits is not None
            semantic_matched_xyz = torch.bmm(semantic_weights, reference_xyz)
            semantic_matched_embedding = torch.bmm(
                semantic_weights, reference_descriptor_for_match
            )
            semantic_matched_offsets = torch.einsum(
                "bnm,bmkc->bnkc", semantic_weights, reference_offsets
            ).flatten(start_dim=-2)
            semantic_expected_similarity = torch.sum(
                semantic_weights * semantic_logits, dim=-1, keepdim=True
            )
            if self.strict_local_bottleneck:
                centered_xyz = query_xyz - query_xyz.mean(dim=1, keepdim=True)
                centered_functional = (
                    query_functional
                    - query_functional.mean(dim=1, keepdim=True)
                )
                point_feature_interaction = torch.einsum(
                    "bnc,bnd->bncd", centered_xyz, centered_functional
                ).flatten(start_dim=-2)
                centered_affordance = (
                    query_affordance_offsets
                    - query_affordance_offsets.mean(dim=1, keepdim=True)
                )
                centered_query_descriptor = (
                    query_descriptor_for_match
                    - query_descriptor_for_match.mean(dim=1, keepdim=True)
                )
                centered_matched_descriptor = (
                    semantic_matched_embedding
                    - semantic_matched_embedding.mean(dim=1, keepdim=True)
                )
                semantic_relation = torch.cat(
                    (
                        point_feature_interaction,
                        centered_affordance,
                        centered_query_descriptor,
                        centered_matched_descriptor,
                        semantic_expected_similarity
                        - semantic_expected_similarity.mean(dim=1, keepdim=True),
                        semantic_confidence[..., None]
                        - semantic_confidence[..., None].mean(
                            dim=1, keepdim=True
                        ),
                        semantic_entropy[..., None]
                        - semantic_entropy[..., None].mean(
                            dim=1, keepdim=True
                        ),
                    ),
                    dim=-1,
                )
            else:
                semantic_relation = torch.cat(
                    (
                        query_xyz,
                        semantic_matched_xyz,
                        query_xyz - semantic_matched_xyz,
                        query_affordance_offsets,
                        query_descriptor_for_match,
                        semantic_matched_embedding,
                        semantic_expected_similarity,
                        semantic_confidence[..., None],
                        semantic_entropy[..., None],
                    ),
                    dim=-1,
                )
            uniform_weights = torch.full_like(
                semantic_weights,
                1.0 / float(reference_xyz.shape[1]),
            )
            null_matched_xyz = torch.bmm(uniform_weights, reference_xyz)
            if self.strict_local_bottleneck:
                null_relation = torch.zeros_like(semantic_relation)
            else:
                null_relation = torch.cat(
                    (
                        query_xyz,
                        null_matched_xyz,
                        query_xyz - null_matched_xyz,
                        torch.zeros_like(query_affordance_offsets),
                        torch.zeros_like(query_descriptor_for_match),
                        torch.zeros_like(semantic_matched_embedding),
                        torch.zeros_like(semantic_expected_similarity),
                        torch.zeros_like(semantic_confidence[..., None]),
                        torch.ones_like(semantic_entropy[..., None]),
                    ),
                    dim=-1,
                )
            semantic_memory = self.semantic_relation_encoder(
                self.semantic_relation_projection(semantic_relation)
            )
            null_memory = self.semantic_relation_encoder(
                self.semantic_relation_projection(null_relation)
            )
            semantic_pooled = self.pool_projection(
                torch.cat(
                    (
                        semantic_memory.mean(dim=1),
                        semantic_memory.amax(dim=1),
                    ),
                    dim=-1,
                )
            )
            semantic_memory = torch.cat(
                (semantic_pooled[:, None, :], semantic_memory), dim=1
            )
            null_pooled = self.pool_projection(
                torch.cat(
                    (null_memory.mean(dim=1), null_memory.amax(dim=1)),
                    dim=-1,
                )
            )
            null_memory = torch.cat(
                (null_pooled[:, None, :], null_memory), dim=1
            )
            semantic_queries = queries
            null_queries = queries
            for block in self.semantic_cross_blocks:
                semantic_queries = block(semantic_queries, semantic_memory)
                null_queries = block(null_queries, null_memory)
            queries = queries + torch.tanh(self.descriptor_match_gate) * (
                semantic_queries - null_queries
            )
        keypoints_normalized = self.keypoint_head(self.output_norm(queries))
        keypoints_normalized = (
            keypoints_normalized * self.maximum_keypoint_offset_normalized
        )
        keypoints_world = (
            keypoints_normalized * query_scale[:, None, :]
            + query_center[:, None, :]
        )
        transforms = keypoints_to_transform(keypoints_world)
        prediction = GraspPrediction(
            transforms=transforms[:, None, :, :],
            keypoints_world=keypoints_world[:, None, :, :],
            scores=torch.zeros(
                (len(query_points), 1),
                dtype=query_points.dtype,
                device=query_points.device,
            ),
            keypoints_normalized=keypoints_normalized[:, None, :, :],
            object_center=query_center,
            object_scale=query_scale,
        )
        if self.query_only:
            cycle_loss = torch.zeros((), dtype=query_points.dtype, device=query_points.device)
        else:
            reverse = torch.softmax(logits.transpose(1, 2), dim=-1)
            reconstructed = torch.bmm(weights, torch.bmm(reverse, query_xyz))
            cycle_loss = F.smooth_l1_loss(reconstructed, query_xyz, beta=0.05)
        functional_loss = torch.zeros(
            (), dtype=query_points.dtype, device=query_points.device
        )
        correspondence_loss = torch.zeros(
            (), dtype=query_points.dtype, device=query_points.device
        )
        affordance_loss = torch.zeros(
            (), dtype=query_points.dtype, device=query_points.device
        )
        if (
            "target_transform_world" in batch
            and "reference_transform_world" in batch
        ):
            query_functional_target = symmetry_invariant_functional_coordinates(
                query_points,
                batch["target_transform_world"],
                query_scale,
            )
            reference_functional_target = symmetry_invariant_functional_coordinates(
                reference_points,
                batch["reference_transform_world"],
                reference_scale,
            )
            functional_loss = 0.5 * (
                F.smooth_l1_loss(
                    query_functional, query_functional_target, beta=0.1
                )
                + F.smooth_l1_loss(
                    reference_functional, reference_functional_target, beta=0.1
                )
            )
            target_keypoints = torch.stack(
                (
                    batch["target_keypoints_world"],
                    flipped_gripper_keypoints(batch["target_transform_world"]),
                ),
                dim=1,
            )
            target_keypoints = (
                target_keypoints - query_center[:, None, None, :]
            ) / query_scale[:, None, None, :]
            target_affordance_offsets = (
                target_keypoints[:, None, :, :, :]
                - query_xyz[:, :, None, None, :]
            )
            predicted_affordance = query_affordance_offsets.reshape(
                len(query_points), query_points.shape[1], 2, 7, 3
            )
            direct_affordance = F.smooth_l1_loss(
                predicted_affordance,
                target_affordance_offsets,
                reduction="none",
                beta=0.1,
            ).mean(dim=(1, 2, 3, 4))
            swapped_affordance = F.smooth_l1_loss(
                predicted_affordance,
                target_affordance_offsets.flip(dims=(2,)),
                reduction="none",
                beta=0.1,
            ).mean(dim=(1, 2, 3, 4))
            affordance_loss = torch.minimum(
                direct_affordance, swapped_affordance
            ).mean()
            if semantic_weights is not None and semantic_logits is not None:
                transported_query = torch.bmm(
                    semantic_weights, reference_functional_target
                )
                semantic_reverse = torch.softmax(
                    semantic_logits.transpose(1, 2), dim=-1
                )
                transported_reference = torch.bmm(
                    semantic_reverse, query_functional_target
                )
                correspondence_loss = 0.5 * (
                    F.smooth_l1_loss(
                        transported_query, query_functional_target, beta=0.1
                    )
                    + F.smooth_l1_loss(
                        transported_reference,
                        reference_functional_target,
                        beta=0.1,
                    )
                )
        return GeometryRelationOutput(
            prediction=prediction,
            cycle_loss=cycle_loss,
            mean_confidence=confidence.mean(),
            mean_entropy=entropy.mean(),
            sample_confidence=confidence.mean(dim=1),
            functional_loss=functional_loss,
            correspondence_loss=correspondence_loss,
            affordance_loss=affordance_loss,
        )


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def symmetry_aware_grasp_loss(
    prediction: GraspPrediction,
    batch: dict[str, torch.Tensor],
    *,
    rotation_loss_weight: float = 0.5,
) -> torch.Tensor:
    """Minimize over the two equivalent local-X wrist-roll labels."""
    target_original = batch["target_keypoints_world"]
    target_flipped = flipped_gripper_keypoints(batch["target_transform_world"])
    targets = torch.stack((target_original, target_flipped), dim=1)
    targets_normalized = (
        targets - prediction.object_center[:, None, None, :]
    ) / prediction.object_scale[:, None, None, :]
    predicted_keypoints = prediction.keypoints_normalized[:, 0, None, :, :]
    keypoint_cost = F.smooth_l1_loss(
        predicted_keypoints.expand_as(targets_normalized),
        targets_normalized,
        reduction="none",
        beta=0.1,
    ).mean(dim=(-1, -2))
    target_rotation_original = batch["target_transform_world"][:, :3, :3]
    flip_rotation = gripper_x_flip(
        dtype=target_rotation_original.dtype,
        device=target_rotation_original.device,
    )[:3, :3]
    target_rotation_flipped = target_rotation_original @ flip_rotation
    target_rotations = torch.stack(
        (target_rotation_original, target_rotation_flipped), dim=1
    )
    predicted_rotation = prediction.transforms[:, 0, None, :3, :3]
    rotation_cost = (
        predicted_rotation - target_rotations
    ).square().mean(dim=(-1, -2))
    return torch.min(
        keypoint_cost + float(rotation_loss_weight) * rotation_cost, dim=1
    ).values.mean()


@torch.no_grad()
def validation_loss(
    model: GeometryRelationTokenPolicy,
    loader: DataLoader,
    device: torch.device,
    *,
    cycle_weight: float,
    functional_aux_weight: float,
    correspondence_aux_weight: float,
    affordance_aux_weight: float,
) -> tuple[float, float, float, float, float, float, float]:
    model.eval()
    total = policy_total = confidence = entropy = 0.0
    functional = correspondence = affordance = 0.0
    samples = 0
    for cpu_batch in loader:
        batch = move_batch(cpu_batch, device)
        output = model(batch)
        action_loss = symmetry_aware_grasp_loss(output.prediction, batch)
        policy_loss = action_loss + float(cycle_weight) * output.cycle_loss
        loss = policy_loss + float(functional_aux_weight) * output.functional_loss
        loss = loss + float(correspondence_aux_weight) * output.correspondence_loss
        loss = loss + float(affordance_aux_weight) * output.affordance_loss
        count = len(batch["points_world"])
        total += float(loss.item()) * count
        policy_total += float(policy_loss.item()) * count
        confidence += float(output.mean_confidence.item()) * count
        entropy += float(output.mean_entropy.item()) * count
        functional += float(output.functional_loss.item()) * count
        correspondence += float(output.correspondence_loss.item()) * count
        affordance += float(output.affordance_loss.item()) * count
        samples += count
    return (
        total / max(samples, 1),
        policy_total / max(samples, 1),
        confidence / max(samples, 1),
        entropy / max(samples, 1),
        functional / max(samples, 1),
        correspondence / max(samples, 1),
        affordance / max(samples, 1),
    )


def metric_record(
    predicted_keypoints: torch.Tensor,
    target_keypoints: torch.Tensor,
    target_transform: torch.Tensor,
) -> dict[str, float | bool]:
    predicted_transform = keypoints_to_transform(predicted_keypoints)
    translation = float(
        torch.linalg.norm(
            predicted_transform[:3, 3] - target_transform[:3, 3]
        ).item()
    )
    rotation = float(
        rotation_angle_degrees(
            predicted_transform[None, :3, :3], target_transform[None, :3, :3]
        )[0].item()
    )
    target_flipped = target_transform @ gripper_x_flip(
        dtype=target_transform.dtype, device=target_transform.device
    )
    symmetry_rotation = min(
        rotation,
        float(
            rotation_angle_degrees(
                predicted_transform[None, :3, :3],
                target_flipped[None, :3, :3],
            )[0].item()
        ),
    )
    keypoint = float(
        torch.sqrt((predicted_keypoints - target_keypoints).square().mean()).item()
    )
    flipped_target_keypoints = flipped_gripper_keypoints(target_transform)
    symmetry_keypoint = min(
        keypoint,
        float(
            torch.sqrt(
                (predicted_keypoints - flipped_target_keypoints).square().mean()
            ).item()
        ),
    )
    return {
        "translation_error_m": translation,
        "rotation_error_deg": rotation,
        "keypoint_rmse_m": keypoint,
        "symmetry_rotation_error_deg": symmetry_rotation,
        "symmetry_keypoint_rmse_m": symmetry_keypoint,
        "success_2cm_15deg": bool(translation < 0.02 and rotation < 15.0),
        "success_3cm_30deg": bool(translation < 0.03 and rotation < 30.0),
        "symmetry_success_2cm_15deg": bool(
            translation < 0.02 and symmetry_rotation < 15.0
        ),
        "symmetry_success_3cm_30deg": bool(
            translation < 0.03 and symmetry_rotation < 30.0
        ),
    }


def summarize_metric_records(records: list[dict]) -> dict:
    result: dict[str, object] = {"samples": len(records)}
    for key in (
        "translation_error_m",
        "rotation_error_deg",
        "keypoint_rmse_m",
        "symmetry_rotation_error_deg",
        "symmetry_keypoint_rmse_m",
    ):
        values = np.asarray([row[key] for row in records], dtype=np.float64)
        result[key] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.9)),
        }
    for key in (
        "success_2cm_15deg",
        "success_3cm_30deg",
        "symmetry_success_2cm_15deg",
        "symmetry_success_3cm_30deg",
    ):
        result[key] = float(np.mean([row[key] for row in records]))
    return result


@torch.no_grad()
def evaluate(
    model: GeometryRelationTokenPolicy,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    pair_records: list[dict] = []
    grouped: defaultdict[int, list[dict]] = defaultdict(list)
    confidences: list[float] = []
    for cpu_batch in loader:
        batch = move_batch(cpu_batch, device)
        output = model(batch)
        predicted = output.prediction.keypoints_world[:, 0]
        for index in range(len(predicted)):
            episode = int(batch["episode_id"][index].item())
            row = metric_record(
                predicted[index],
                batch["target_keypoints_world"][index],
                batch["target_transform_world"][index],
            )
            row.update(
                {
                    "episode_id": episode,
                    "shoe_id": int(batch["shoe_id"][index].item()),
                    "reference_episode_id": int(
                        batch["reference_episode_id"][index].item()
                    ),
                }
            )
            pair_records.append(row)
            grouped[episode].append(
                {
                    "predicted": predicted[index].detach().cpu(),
                    "target_keypoints": batch["target_keypoints_world"][index]
                    .detach()
                    .cpu(),
                    "target_transform": batch["target_transform_world"][index]
                    .detach()
                    .cpu(),
                    "shoe_id": int(batch["shoe_id"][index].item()),
                    "confidence": float(output.sample_confidence[index].item()),
                }
            )
        confidences.append(float(output.mean_confidence.item()))
    ensemble_records = []
    for episode, rows in sorted(grouped.items()):
        # Align the equivalent local-X gripper modes before ensembling.  This
        # avoids averaging two physically equivalent 180-degree labels into an
        # invalid pose.  Selection uses predictions only, never target labels.
        anchor_transform = keypoints_to_transform(rows[0]["predicted"])
        aligned = []
        weights = []
        flip = gripper_x_flip(
            dtype=anchor_transform.dtype, device=anchor_transform.device
        )
        for row in rows:
            transform = keypoints_to_transform(row["predicted"])
            flipped_transform = transform @ flip
            original_angle = rotation_angle_degrees(
                transform[None, :3, :3], anchor_transform[None, :3, :3]
            )[0]
            flipped_angle = rotation_angle_degrees(
                flipped_transform[None, :3, :3], anchor_transform[None, :3, :3]
            )[0]
            selected_transform = (
                flipped_transform if flipped_angle < original_angle else transform
            )
            aligned.append(
                transform_keypoints_torch(
                    selected_transform[:3, :3], selected_transform[:3, 3]
                )
            )
            weights.append(max(float(row["confidence"]), 1e-3))
        weight = torch.as_tensor(weights, dtype=aligned[0].dtype)
        weight = weight / weight.sum()
        predicted = torch.sum(torch.stack(aligned) * weight[:, None, None], dim=0)
        record = metric_record(
            predicted, rows[0]["target_keypoints"], rows[0]["target_transform"]
        )
        record.update(
            {
                "episode_id": int(episode),
                "shoe_id": int(rows[0]["shoe_id"]),
                "references": len(rows),
            }
        )
        ensemble_records.append(record)
    return {
        "pair": summarize_metric_records(pair_records),
        "ensemble": summarize_metric_records(ensemble_records),
        "mean_correspondence_confidence": float(np.mean(confidences)),
        "ensemble_records": ensemble_records,
    }


def train_condition(
    *,
    payload: dict[str, np.ndarray],
    descriptors: np.ndarray,
    pairs: dict[str, list[tuple[int, int]]],
    condition: str,
    fold_index: int,
    seed: int,
    projection_metadata: dict,
    args: argparse.Namespace,
    device: torch.device,
    initial_state: dict[str, torch.Tensor] | None = None,
) -> tuple[dict, dict[str, torch.Tensor]]:
    run_seed = int(seed) + 100 * int(fold_index)
    seed_everything(run_seed)
    shuffled = condition.endswith("_shuffled")
    datasets = {
        split: GeometryRelationPairDataset(
            payload,
            descriptors,
            split_pairs,
            point_count=args.point_count,
            shuffled=shuffled,
            training=split == "train",
            point_noise_m=args.point_noise_m,
        )
        for split, split_pairs in pairs.items()
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
    if "_fused_" in condition:
        matching_mode = "fused"
    elif condition in {"query_only", "raw_relation"}:
        matching_mode = "geometry"
    else:
        matching_mode = "descriptor"
    use_functional_refinement = bool(
        (
            float(args.functional_aux_weight) > 0.0
            or float(args.correspondence_aux_weight) > 0.0
        )
        and condition not in {"query_only", "raw_relation"}
    )
    functional_aux_weight = (
        float(args.functional_aux_weight) if use_functional_refinement else 0.0
    )
    correspondence_aux_weight = (
        float(args.correspondence_aux_weight)
        if use_functional_refinement and args.fusion_style == "dual_memory"
        else 0.0
    )
    affordance_aux_weight = (
        float(args.affordance_aux_weight)
        if condition not in {"query_only", "raw_relation"}
        else 0.0
    )
    model = GeometryRelationTokenPolicy(
        descriptor_dim=args.descriptor_dim,
        match_dim=args.match_dim,
        feature_dim=args.feature_dim,
        heads=args.heads,
        layers=args.layers,
        dropout=args.dropout,
        temperature=args.temperature,
        query_only=condition == "query_only",
        matching_mode=matching_mode,
        task_refined_matching=use_functional_refinement,
        fusion_style=args.fusion_style,
        strict_local_bottleneck=args.strict_local_bottleneck,
    ).to(device)
    if initial_state is not None:
        if matching_mode != "fused":
            raise ValueError("shared Raw initialization is only valid for fused runs")
        model.load_state_dict(initial_state, strict=True)
    fused_warmup = (
        min(int(args.fusion_warmup_epochs), int(args.epochs) - 1)
        if matching_mode == "fused"
        and (initial_state is None or args.fusion_style == "dual_memory")
        else 0
    )
    if matching_mode == "fused" and fused_warmup > 0:
        model.descriptor_match_gate.requires_grad_(False)
    if initial_state is not None and args.freeze_fused_base:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.match_projection.parameters():
            parameter.requires_grad_(True)
        for parameter in model.functional_head.parameters():
            parameter.requires_grad_(True)
        for parameter in model.affordance_head.parameters():
            parameter.requires_grad_(True)
        if args.fusion_style == "dual_memory":
            for parameter in model.semantic_relation_projection.parameters():
                parameter.requires_grad_(True)
            for parameter in model.semantic_relation_encoder.parameters():
                parameter.requires_grad_(True)
            for parameter in model.semantic_cross_blocks.parameters():
                parameter.requires_grad_(True)
        model.descriptor_match_gate.requires_grad_(fused_warmup == 0)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("no trainable parameters remain")
    optimizer_parameters = list(trainable_parameters)
    if (
        matching_mode == "fused"
        and fused_warmup > 0
        and all(
            parameter is not model.descriptor_match_gate
            for parameter in optimizer_parameters
        )
    ):
        # Keep the frozen gate in the optimizer so it can be enabled without
        # rebuilding optimizer state after semantic correspondence pretraining.
        optimizer_parameters.append(model.descriptor_match_gate)
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    cycle_weight = 0.0 if condition == "query_only" else args.cycle_weight
    best_state = None
    best_validation = float("inf")
    best_epoch = -1
    stale = 0
    history = []
    if initial_state is not None:
        (
            initial_objective,
            initial_validation,
            confidence,
            entropy,
            functional,
            correspondence,
            affordance,
        ) = validation_loss(
            model,
            loaders["validation"],
            device,
            cycle_weight=cycle_weight,
            functional_aux_weight=functional_aux_weight,
            correspondence_aux_weight=correspondence_aux_weight,
            affordance_aux_weight=affordance_aux_weight,
        )
        best_state = copy.deepcopy(model.state_dict())
        best_validation = float(initial_validation)
        history.append(
            {
                "epoch": -1,
                "train_loss": None,
                "validation_loss": float(initial_validation),
                "validation_objective": float(initial_objective),
                "confidence": confidence,
                "entropy": entropy,
                "functional_loss": functional,
                "correspondence_loss": correspondence,
                "affordance_loss": affordance,
                "descriptor_match_gate": float(
                    torch.tanh(model.descriptor_match_gate).detach().cpu()
                ),
                "phase": "shared_raw_initialization",
            }
        )
    for epoch in range(int(args.epochs)):
        if matching_mode == "fused" and epoch == fused_warmup:
            if args.fixed_fusion_gate is None:
                model.descriptor_match_gate.requires_grad_(True)
            else:
                fixed_gate = float(args.fixed_fusion_gate)
                if not 0.0 < fixed_gate < 1.0:
                    raise ValueError("--fixed-fusion-gate must be in (0,1)")
                with torch.no_grad():
                    model.descriptor_match_gate.fill_(
                        float(np.arctanh(fixed_gate))
                    )
                model.descriptor_match_gate.requires_grad_(False)
            # Give the descriptor residual a full patience window.  The best
            # pre-unfreeze Raw checkpoint is retained as a safe fallback.
            stale = 0
        model.train()
        train_total = 0.0
        train_samples = 0
        for cpu_batch in loaders["train"]:
            batch = move_batch(cpu_batch, device)
            output = model(batch)
            action_loss = symmetry_aware_grasp_loss(output.prediction, batch)
            loss = action_loss + float(cycle_weight) * output.cycle_loss
            loss = loss + functional_aux_weight * output.functional_loss
            loss = (
                loss
                + correspondence_aux_weight * output.correspondence_loss
            )
            loss = loss + affordance_aux_weight * output.affordance_loss
            if matching_mode == "fused":
                loss = loss + float(args.descriptor_gate_l2) * torch.tanh(
                    model.descriptor_match_gate
                ).square()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            count = len(batch["points_world"])
            train_total += float(loss.item()) * count
            train_samples += count
        (
            validation_objective,
            validation,
            confidence,
            entropy,
            functional,
            correspondence,
            affordance,
        ) = validation_loss(
            model,
            loaders["validation"],
            device,
            cycle_weight=cycle_weight,
            functional_aux_weight=functional_aux_weight,
            correspondence_aux_weight=correspondence_aux_weight,
            affordance_aux_weight=affordance_aux_weight,
        )
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": train_total / max(train_samples, 1),
                "validation_loss": validation,
                "validation_objective": validation_objective,
                "confidence": confidence,
                "entropy": entropy,
                "functional_loss": functional,
                "correspondence_loss": correspondence,
                "affordance_loss": affordance,
                "descriptor_match_gate": float(
                    torch.tanh(model.descriptor_match_gate).detach().cpu()
                ),
                "phase": (
                    "semantic_correspondence_pretrain"
                    if matching_mode == "fused" and epoch < fused_warmup
                    else "descriptor_residual"
                    if matching_mode == "fused"
                    else matching_mode
                ),
            }
        )
        if validation < best_validation - 1e-6:
            best_validation = validation
            best_epoch = int(epoch)
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch == 0 or epoch % 20 == 0:
            print(
                json.dumps(
                    {
                        "condition": condition,
                        "fold": int(fold_index),
                        "seed": int(seed),
                        "epoch": int(epoch),
                        "train": history[-1]["train_loss"],
                        "validation": validation,
                        "best": best_validation,
                        "stale": stale,
                        "gate": history[-1]["descriptor_match_gate"],
                        "phase": history[-1]["phase"],
                    }
                ),
                flush=True,
            )
        may_stop = not (
            matching_mode == "fused" and epoch < fused_warmup
        )
        if may_stop and stale >= int(args.patience):
            break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return {
        "schema_version": 1,
        "experiment": "geometry_relation_token_grasp",
        "condition": condition,
        "fold": int(fold_index),
        "seed": int(seed),
        "split": {key: list(FOLDS[fold_index][key]) for key in FOLDS[fold_index]},
        "parameters": int(sum(value.numel() for value in model.parameters())),
        "trainable_parameters": int(
            sum(value.numel() for value in model.parameters() if value.requires_grad)
        ),
        "matching_mode": matching_mode,
        "fusion_style": args.fusion_style,
        "strict_local_bottleneck": bool(args.strict_local_bottleneck),
        "task_refined_matching": use_functional_refinement,
        "functional_aux_weight": functional_aux_weight,
        "correspondence_aux_weight": correspondence_aux_weight,
        "affordance_aux_weight": affordance_aux_weight,
        "descriptor_match_gate": float(
            torch.tanh(model.descriptor_match_gate).detach().cpu()
        ),
        "fixed_fusion_gate": args.fixed_fusion_gate,
        "fusion_warmup_epochs": int(fused_warmup),
        "shared_raw_initialization": bool(initial_state is not None),
        "frozen_fused_base": bool(
            initial_state is not None and args.freeze_fused_base
        ),
        "projection": projection_metadata,
        "pair_counts": {key: len(value) for key, value in pairs.items()},
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_validation),
        "test_metrics": evaluate(model, loaders["test"], device),
        "history": history,
    }, copy.deepcopy(best_state)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grasp-dataset", type=Path, required=True)
    parser.add_argument("--feature-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=[
            "query_only",
            "raw_relation",
            "ndf_relation",
            "ndf_shuffled",
            "utonia_s0_relation",
            "utonia_s0_shuffled",
            "ndf_fused_relation",
            "ndf_fused_shuffled",
            "utonia_s0_fused_relation",
            "utonia_s0_fused_shuffled",
            "oracle_relation",
        ],
        choices=sorted(CONDITIONS),
    )
    parser.add_argument("--folds", nargs="+", type=int, default=[0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--descriptor-dim", type=int, default=64)
    parser.add_argument("--match-dim", type=int, default=32)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--point-count", type=int, default=128)
    parser.add_argument("--references-per-target", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--cycle-weight", type=float, default=0.05)
    parser.add_argument("--fusion-warmup-epochs", type=int, default=80)
    parser.add_argument("--descriptor-gate-l2", type=float, default=1e-3)
    parser.add_argument("--fixed-fusion-gate", type=float)
    parser.add_argument("--functional-aux-weight", type=float, default=0.0)
    parser.add_argument("--correspondence-aux-weight", type=float, default=0.0)
    parser.add_argument("--affordance-aux-weight", type=float, default=0.0)
    parser.add_argument(
        "--fusion-style",
        choices=("matcher", "dual_memory"),
        default="matcher",
        help="Fuse descriptors inside matching or as a gated second relation memory.",
    )
    parser.add_argument("--strict-local-bottleneck", action="store_true")
    parser.add_argument("--shared-raw-init", action="store_true")
    parser.add_argument("--freeze-fused-base", action="store_true")
    parser.add_argument("--point-noise-m", type=float, default=0.001)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--cuda-memory-fraction",
        type=float,
        default=0.05,
        help="Per-process guard relative to the visible GPU memory.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if any(fold < 0 or fold >= len(FOLDS) for fold in args.folds):
        raise ValueError(f"folds must be in [0,{len(FOLDS)-1}]")
    unknown = set(args.conditions) - CONDITIONS
    if unknown:
        raise ValueError(f"unknown conditions: {sorted(unknown)}")
    if args.freeze_fused_base and not args.shared_raw_init:
        raise ValueError("--freeze-fused-base requires --shared-raw-init")
    if args.shared_raw_init and "raw_relation" not in args.conditions:
        raise ValueError("--shared-raw-init requires raw_relation in --conditions")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = load_aligned_payload(args.grasp_dataset, args.feature_dataset)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.set_per_process_memory_fraction(
            float(args.cuda_memory_fraction), device=device
        )
    shoe_ids = np.asarray(payload["shoe_id_label_only"], dtype=np.int64)
    all_results = []
    for fold_index in args.folds:
        split_indices = {
            key: np.flatnonzero(np.isin(shoe_ids, ids))
            for key, ids in FOLDS[int(fold_index)].items()
        }
        pairs = {
            "train": build_reference_pairs(
                payload,
                split_indices["train"],
                split_indices["train"],
                references_per_target=args.references_per_target,
            ),
            "validation": build_reference_pairs(
                payload,
                split_indices["validation"],
                split_indices["train"],
                references_per_target=args.references_per_target,
            ),
            "test": build_reference_pairs(
                payload,
                split_indices["test"],
                split_indices["train"],
                references_per_target=args.references_per_target,
            ),
        }
        descriptor_cache = {}
        condition_order = list(args.conditions)
        if args.shared_raw_init:
            condition_order = ["raw_relation"] + [
                value for value in condition_order if value != "raw_relation"
            ]
        shared_raw_states: dict[int, dict[str, torch.Tensor]] = {}
        for condition in condition_order:
            base = condition.replace("_shuffled", "_relation")
            cache_key = base
            if cache_key not in descriptor_cache:
                descriptor_cache[cache_key] = condition_descriptors(
                    payload,
                    condition,
                    split_indices["train"],
                    descriptor_dim=args.descriptor_dim,
                    fold_index=int(fold_index),
                )
            descriptors, metadata = descriptor_cache[cache_key]
            for seed in args.seeds:
                initial_state = (
                    shared_raw_states.get(int(seed))
                    if args.shared_raw_init and "_fused_" in condition
                    else None
                )
                if args.shared_raw_init and "_fused_" in condition and initial_state is None:
                    raise RuntimeError(
                        f"missing shared Raw state for fold {fold_index}, seed {seed}"
                    )
                result, best_state = train_condition(
                    payload=payload,
                    descriptors=descriptors,
                    pairs=pairs,
                    condition=condition,
                    fold_index=int(fold_index),
                    seed=int(seed),
                    projection_metadata=metadata,
                    args=args,
                    device=device,
                    initial_state=initial_state,
                )
                if args.shared_raw_init and condition == "raw_relation":
                    shared_raw_states[int(seed)] = best_state
                stem = f"fold{fold_index}_{condition}_seed{seed}"
                (args.output_dir / f"{stem}.json").write_text(
                    json.dumps(result, indent=2) + "\n", encoding="utf-8"
                )
                all_results.append(result)
                metric = result["test_metrics"]["ensemble"]
                print(
                    json.dumps(
                        {
                            "completed": stem,
                            "translation_cm": 100.0
                            * metric["translation_error_m"]["mean"],
                            "rotation_deg": metric["rotation_error_deg"]["mean"],
                            "success_3cm_30deg": metric["success_3cm_30deg"],
                        }
                    ),
                    flush=True,
                )
    summary = {
        "schema_version": 1,
        "experiment": "geometry_relation_token_grasp",
        "alignment_diagnostics": payload["_alignment_diagnostics"],
        "runs": all_results,
    }
    output = args.output_dir / "runs.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "runs": len(all_results)}))


if __name__ == "__main__":
    main()
