#!/usr/bin/env python3
"""Object-disjoint falsification test for interaction-centric flow tokens.

This experiment intentionally avoids a simulator pose or oracle flow at policy
inference.  The only model inputs are current A/B camera point clouds and robot
proprioception.  Simulator-derived rigid object flow is a label-only auxiliary
target.  A wrong-episode target is included as a causal control.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from .canonicalize_task_flow_dataset import (
    canonicalize,
    episode_frames_from_sample_frame9,
)
from .functional_action_frame import (
    action14_from_target_frame,
    action14_to_target_frame,
    frame9_rotation,
    pose9_rotation,
    relative_frame9_to_target_frame,
    transform_eef_state20_to_target_frame,
)
from .hand_object_frame import (
    HAND_OBJECT_TRANSLATION_SCALE_M,
    active_hand_object_frame11_from_goal_relative,
)
from .diffusion_action_decoder import stateless_normal_like
from .interaction_flow_tokens import (
    InteractionFlowOutput,
    InteractionFlowTokenPolicy,
    PointTokenActionPolicy,
    attention_entropy,
    correspondence_flow_loss,
    motion_correspondence_flow_loss,
    rigid_flow_loss,
    rotation_geodesic_loss,
    unordered_flow_chamfer_loss,
)
from .train_task_flow_benchmark import (
    FOLDS,
    TaskFlowDataset,
    fit_normalization,
    move_batch,
    remaining_flow_from_current,
    seed_everything,
    summarize_errors,
    weighted_action_loss,
)


CONDITIONS = {
    "point_tokens",
    "interaction_tokens",
    "interaction_flow",
    "interaction_flow_shuffled",
    "interaction_flow_conditioned",
    "interaction_functional",
    "interaction_functional_flow",
    "interaction_functional_direct",
    "interaction_functional_direct_flow",
    "interaction_functional_dual_frame_flow",
    "interaction_functional_correction_adapter_flow",
    "interaction_functional_equivariant_adapter_flow",
    "interaction_functional_local_equivariant_adapter_flow",
    "interaction_functional_hybrid_equivariant_adapter_flow",
    "interaction_functional_biframe_adapter_flow",
    "interaction_functional_gated_biframe_adapter_flow",
    "interaction_functional_grasp_adapter_flow",
    "interaction_functional_gated",
    "interaction_functional_gated_flow",
    "interaction_functional_relative",
    "interaction_relative_flow",
    "interaction_diffusion",
    "interaction_relative_diffusion",
}


def resolve_policy_object_split(
    train_ids: list[int] | tuple[int, ...] | None,
    validation_ids: list[int] | tuple[int, ...] | None,
    test_ids: list[int] | tuple[int, ...] | None,
) -> dict[str, tuple[int, ...]] | None:
    """Validate an optional explicit object-disjoint policy split."""

    values = (train_ids, validation_ids, test_ids)
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(
            "train-ids, validation-ids, and test-ids must be supplied together"
        )
    result = {
        "train": tuple(int(value) for value in train_ids),
        "validation": tuple(int(value) for value in validation_ids),
        "test": tuple(int(value) for value in test_ids),
    }
    sets = {name: set(ids) for name, ids in result.items()}
    for name, ids in result.items():
        if not ids or len(ids) != len(sets[name]):
            raise ValueError(f"{name} object IDs must be nonempty and unique")
    if (
        sets["train"] & sets["validation"]
        or sets["train"] & sets["test"]
        or sets["validation"] & sets["test"]
    ):
        raise ValueError("train, validation, and test object IDs must be disjoint")
    return result


def policy_phase_keep_mask(
    payload: dict[str, np.ndarray], policy_phase: str
) -> np.ndarray:
    """Return the training/deployment-scope mask without adding phase inputs.

    ``pre_release`` keeps relation states whose demonstrated release is still
    in the future.  Their action chunks may contain the release transition, so
    the policy can still learn it while excluding post-release observations
    that deployment never asks the policy to recover from.
    """

    sample_count = len(payload["shoe_id"])
    if policy_phase == "all":
        return np.ones(sample_count, dtype=bool)
    if "relation_phase" not in payload:
        raise ValueError(
            f"--policy-phase={policy_phase} requires relation_phase"
        )
    relation = np.asarray(payload["relation_phase"], dtype=np.float32)
    if policy_phase == "relation":
        return np.isclose(relation, 1.0)
    if policy_phase == "pre_relation":
        return np.isclose(relation, 0.0)
    if policy_phase == "pre_release":
        if "steps_to_release" not in payload:
            raise ValueError(
                "--policy-phase=pre_release requires steps_to_release"
            )
        steps_to_release = np.asarray(payload["steps_to_release"])
        return np.isclose(relation, 1.0) & (steps_to_release > 0)
    raise ValueError(f"unknown policy phase {policy_phase!r}")


def task_aligned_point_coordinates(
    points_a: np.ndarray,
    points_b: np.ndarray,
    goal_frame9: np.ndarray,
    relative_frame9: np.ndarray,
    *,
    scale_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Express A/B points in their paired functional object frames.

    ``relative_frame9`` encodes goal_R @ current_R.T and goal-current
    translation.  It therefore reconstructs the predicted current frame from
    the camera goal without any simulator pose at inference.
    """

    if float(scale_m) <= 0.0:
        raise ValueError("scale_m must be positive")
    source = np.asarray(points_a, dtype=np.float64)
    target = np.asarray(points_b, dtype=np.float64)
    goal = np.asarray(goal_frame9, dtype=np.float64)
    relative = np.asarray(relative_frame9, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] < 3:
        raise ValueError(f"points_a must be [points,channels], got {source.shape}")
    if target.ndim != 2 or target.shape[1] < 3:
        raise ValueError(f"points_b must be [points,channels], got {target.shape}")
    if goal.shape != (9,) or relative.shape != (9,):
        raise ValueError("goal_frame9 and relative_frame9 must each have shape [9]")
    goal_rotation = frame9_rotation(goal)
    relative_rotation = frame9_rotation(relative)
    source_rotation = relative_rotation.T @ goal_rotation
    source_position = goal[:3] - relative[:3]
    source_local = (source[:, :3] - source_position) @ source_rotation
    target_local = (target[:, :3] - goal[:3]) @ goal_rotation
    return (
        (source_local / float(scale_m)).astype(np.float32),
        (target_local / float(scale_m)).astype(np.float32),
    )


class InteractionFlowDataset(TaskFlowDataset):
    """Attach correct and deterministic wrong-episode future point-set labels."""

    def __init__(
        self,
        payload: dict[str, np.ndarray],
        indices: np.ndarray,
        normalization,
        *,
        flow_steps: int,
        target_horizon_mode: str = "terminal",
        target_frame_mode: str = "normal",
        source_local_frame_mode: str = "normal",
        functional_frame_translation_mode: str = "absolute",
        functional_point_coordinates: bool = False,
        zero_global_functional_frame_token: bool = False,
    ) -> None:
        super().__init__(
            payload,
            indices,
            normalization,
            flow_context_mode="remaining",
        )
        if flow_steps < 2:
            raise ValueError("flow_steps must be at least two")
        if target_horizon_mode not in {"terminal", "action_chunk"}:
            raise ValueError(
                "target_horizon_mode must be 'terminal' or 'action_chunk'"
            )
        self.target_horizon_mode = str(target_horizon_mode)
        if target_frame_mode not in {"normal", "zero", "shuffled"}:
            raise ValueError(
                "target_frame_mode must be 'normal', 'zero', or 'shuffled'"
            )
        self.target_frame_mode = str(target_frame_mode)
        if source_local_frame_mode not in {"normal", "zero", "shuffled"}:
            raise ValueError(
                "source_local_frame_mode must be normal, zero, or shuffled"
            )
        self.source_local_frame_mode = str(source_local_frame_mode)
        if functional_frame_translation_mode not in {"absolute", "delta"}:
            raise ValueError(
                "functional_frame_translation_mode must be 'absolute' or 'delta'"
            )
        self.functional_frame_translation_mode = str(
            functional_frame_translation_mode
        )
        self.functional_point_coordinates = bool(functional_point_coordinates)
        self.zero_global_functional_frame_token = bool(
            zero_global_functional_frame_token
        )
        if self.functional_point_coordinates and "goal_frame9" not in payload:
            raise KeyError("functional point coordinates require goal_frame9")
        if self.zero_global_functional_frame_token and not self.functional_point_coordinates:
            raise ValueError(
                "zeroing only the global token requires functional point coordinates"
            )
        self.action_horizon = int(payload["action"].shape[1])
        self.episode_frames = {
            int(episode): int(payload["frame_index"][payload["episode_index"] == episode].max())
            + 2
            for episode in np.unique(payload["episode_index"])
        }
        source_steps = int(payload["episode_flow"].shape[2])
        self.flow_time_indices = np.rint(
            np.linspace(0, source_steps - 1, int(flow_steps))
        ).astype(np.int64)
        episodes = sorted(
            np.unique(payload["episode_index"][self.indices]).astype(np.int64).tolist()
        )
        episode_shoes = np.asarray(payload["episode_shoe_id"], dtype=np.int64)
        self._target_cache: dict[tuple[int, int], torch.Tensor] = {}
        self._rotation_cache: dict[tuple[int, int], torch.Tensor] = {}
        self.wrong_episode: dict[int, int] = {}
        self.episode_representative = {
            int(episode): int(
                np.flatnonzero(payload["episode_index"] == episode)[0]
            )
            for episode in episodes
        }
        for position, episode in enumerate(episodes):
            for offset in range(1, len(episodes) + 1):
                candidate = episodes[(position + offset) % len(episodes)]
                if episode_shoes[candidate] != episode_shoes[episode]:
                    self.wrong_episode[episode] = candidate
                    break
            if episode not in self.wrong_episode:
                # Small object-disjoint pilots can contain one identity in a
                # validation/test split but several independently perturbed
                # episodes.  Prefer a cross-object corruption above; when that
                # is impossible, use another episode of the same object rather
                # than blocking every non-shuffled condition as well.
                alternatives = [value for value in episodes if value != episode]
                if not alternatives:
                    raise ValueError(
                        "wrong-flow control needs at least two episodes"
                    )
                self.wrong_episode[episode] = alternatives[0]
        self.wrong_frame_index: dict[int, int] = {}
        if "target_frame9" in payload:
            # A relative current-to-goal frame varies within an episode.  Match
            # the wrong-frame control by perturbation magnitude and normalized
            # trajectory progress so that corruption changes the relation, not
            # merely the difficulty or time step.
            selected = np.asarray(self.indices, dtype=np.int64)
            frames = np.asarray(payload["target_frame9"][selected], dtype=np.float64)
            translation_norm = np.linalg.norm(frames[:, :3], axis=-1)
            rotations = frame9_rotation(frames)
            traces = np.trace(rotations, axis1=-2, axis2=-1)
            rotation_angle = np.arccos(np.clip((traces - 1.0) * 0.5, -1.0, 1.0))
            progress = np.empty(len(selected), dtype=np.float64)
            for position, sample_index in enumerate(selected):
                sample_episode = int(payload["episode_index"][sample_index])
                rows = selected[payload["episode_index"][selected] == sample_episode]
                frame_values = np.asarray(payload["frame_index"][rows], dtype=np.float64)
                denominator = max(float(frame_values.max() - frame_values.min()), 1.0)
                progress[position] = (
                    float(payload["frame_index"][sample_index]) - float(frame_values.min())
                ) / denominator
            difficulty = np.stack(
                (translation_norm, rotation_angle, progress), axis=-1
            )
            scale = np.maximum(difficulty.std(axis=0), np.asarray((1e-3, 1e-2, 0.1)))
            selected_shoes = np.asarray(payload["shoe_id"][selected], dtype=np.int64)
            for position, sample_index in enumerate(selected):
                candidates = np.flatnonzero(selected_shoes != selected_shoes[position])
                if not len(candidates):
                    selected_episodes = np.asarray(
                        payload["episode_index"][selected], dtype=np.int64
                    )
                    candidates = np.flatnonzero(
                        selected_episodes != selected_episodes[position]
                    )
                if not len(candidates):
                    raise ValueError(
                        "wrong-frame control needs at least two episodes"
                    )
                cost = np.abs(
                    (difficulty[candidates] - difficulty[position]) / scale
                ).sum(axis=-1)
                self.wrong_frame_index[int(sample_index)] = int(
                    selected[candidates[int(np.argmin(cost))]]
                )

    @staticmethod
    def _rigid_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        source_zero = source - source.mean(axis=0)
        target_zero = target - target.mean(axis=0)
        left, _, right = np.linalg.svd(source_zero.T @ target_zero)
        rotation = right.T @ left.T
        if np.linalg.det(rotation) < 0.0:
            right[-1] *= -1.0
            rotation = right.T @ left.T
        return rotation

    @staticmethod
    def _interpolate_flow(flow: np.ndarray, fractions: np.ndarray) -> np.ndarray:
        value = np.asarray(flow, dtype=np.float64)
        position = np.clip(fractions, 0.0, 1.0) * (value.shape[1] - 1)
        lower = np.floor(position).astype(np.int64)
        upper = np.minimum(lower + 1, value.shape[1] - 1)
        weight = position - lower
        return (
            value[:, lower] * (1.0 - weight)[None, :, None]
            + value[:, upper] * weight[None, :, None]
        )

    def _action_chunk_flow(
        self,
        flow: np.ndarray,
        current_anchors: np.ndarray,
        *,
        frame: int,
        episode_frames: int,
    ) -> np.ndarray:
        final_frame = max(int(episode_frames) - 1, 1)
        query_frames = np.linspace(
            float(frame),
            float(min(frame + self.action_horizon, final_frame)),
            len(self.flow_time_indices),
        )
        future = self._interpolate_flow(flow, query_frames / final_frame)
        reference = future[:, 0]
        reference_center = reference.mean(axis=0)
        current = np.asarray(current_anchors, dtype=np.float64)
        current_center = current.mean(axis=0)
        current_local = current - current_center
        output = []
        for step in range(future.shape[1]):
            rotation = self._rigid_rotation(reference, future[:, step])
            center_delta = future[:, step].mean(axis=0) - reference_center
            output.append(current_local @ rotation.T + current_center + center_delta)
        result = np.stack(output, axis=1)
        result[:, 0] = current
        return result.astype(np.float32)

    def _target(
        self,
        flow: np.ndarray,
        current_anchors: np.ndarray,
        *,
        sample_index: int,
    ) -> torch.Tensor:
        if self.target_horizon_mode == "terminal":
            remaining = remaining_flow_from_current(flow, current_anchors)
            selected = remaining[:, self.flow_time_indices]
        else:
            episode = int(self.payload["episode_index"][sample_index])
            selected = self._action_chunk_flow(
                flow,
                current_anchors,
                frame=int(self.payload["frame_index"][sample_index]),
                episode_frames=self.episode_frames[episode],
            )
        normalized = (selected - self.norm.xyz_mean) / self.norm.xyz_std
        return torch.from_numpy(normalized.astype(np.float32))

    def _cached_target(
        self,
        *,
        sample_index: int,
        episode: int,
        current_anchors: np.ndarray,
    ) -> torch.Tensor:
        key = (int(sample_index), int(episode))
        if key not in self._target_cache:
            target = self._target(
                self.payload["episode_flow"][episode],
                current_anchors,
                sample_index=sample_index,
            )
            self._target_cache[key] = target
            metric = target.numpy() * self.norm.xyz_std[None, None, :]
            rotation = self._rigid_rotation(metric[:, 0], metric[:, -1])
            self._rotation_cache[key] = torch.from_numpy(
                rotation.astype(np.float32)
            )
        return self._target_cache[key]

    def _cached_rotation(
        self,
        *,
        sample_index: int,
        episode: int,
        current_anchors: np.ndarray,
    ) -> torch.Tensor:
        key = (int(sample_index), int(episode))
        if key not in self._rotation_cache:
            self._cached_target(
                sample_index=sample_index,
                episode=episode,
                current_anchors=current_anchors,
            )
        return self._rotation_cache[key]

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        result = super().__getitem__(item)
        index = int(self.indices[item])
        recovery_key = next(
            (
                key
                for key in ("is_recovery_sample", "is_recovery_augmented")
                if key in self.payload
            ),
            None,
        )
        if recovery_key is not None:
            result["is_recovery"] = torch.tensor(
                float(self.payload[recovery_key][index]), dtype=torch.float32
            )
        if "source_axis3" in self.payload:
            result["source_axis3"] = torch.from_numpy(
                np.asarray(self.payload["source_axis3"][index], dtype=np.float32)
            )
        if "source_frame6_columns" in self.payload:
            result["source_frame6_columns"] = torch.from_numpy(
                np.asarray(
                    self.payload["source_frame6_columns"][index],
                    dtype=np.float32,
                )
            )
        if "target_axis3" in self.payload:
            result["target_axis3"] = torch.from_numpy(
                np.asarray(self.payload["target_axis3"][index], dtype=np.float32)
            )
        episode = int(self.payload["episode_index"][index])
        if "target_frame9" in self.payload:
            if self.target_frame_mode == "zero":
                frame = np.zeros(9, dtype=np.float32)
                frame_metric = frame.copy()
                frame_confidence = 0.0
            else:
                frame_index = (
                    index
                    if self.target_frame_mode == "normal"
                    else self.wrong_frame_index[index]
                )
                frame = np.asarray(
                    self.payload["target_frame9"][frame_index],
                    dtype=np.float32,
                ).copy()
                frame_metric = frame.copy()
                if self.functional_frame_translation_mode == "absolute":
                    frame[:3] = (
                        frame[:3] - self.norm.xyz_mean
                    ) / self.norm.xyz_std
                else:
                    frame[:3] = frame[:3] / self.norm.xyz_std
                frame_confidence = float(
                    self.payload.get(
                        "target_frame_confidence",
                        np.ones(len(self.payload["target_frame9"]), dtype=np.float32),
                    )[frame_index]
                )
            result["target_frame9"] = torch.from_numpy(frame)
            result["target_frame_confidence"] = torch.tensor(
                frame_confidence, dtype=torch.float32
            )
            if self.functional_point_coordinates:
                if self.target_frame_mode == "zero" or frame_confidence <= 0.0:
                    source_local = np.zeros(
                        (len(self.payload["points_a"][index]), 3), dtype=np.float32
                    )
                    target_local = np.zeros(
                        (len(self.payload["points_b"][index]), 3), dtype=np.float32
                    )
                else:
                    source_local, target_local = task_aligned_point_coordinates(
                        self.payload["points_a"][index],
                        self.payload["points_b"][index],
                        self.payload["goal_frame9"][index],
                        frame_metric,
                        scale_m=float(np.asarray(self.norm.xyz_std).mean()),
                    )
                result["points_a"] = torch.cat(
                    (result["points_a"], torch.from_numpy(source_local)), dim=-1
                )
                result["points_b"] = torch.cat(
                    (result["points_b"], torch.from_numpy(target_local)), dim=-1
                )
                if self.zero_global_functional_frame_token:
                    result["target_frame9"] = torch.zeros(9, dtype=torch.float32)
                    result["target_frame_confidence"] = torch.tensor(
                        0.0, dtype=torch.float32
                    )
            if "source_frame6_columns" in self.payload:
                if self.source_local_frame_mode == "zero":
                    source_local_frame = np.zeros(9, dtype=np.float32)
                else:
                    local_index = (
                        index
                        if self.source_local_frame_mode == "normal"
                        else self.wrong_frame_index[index]
                    )
                    local_relative_frame = (
                        frame_metric
                        if self.source_local_frame_mode == "normal"
                        else np.asarray(
                            self.payload["target_frame9"][local_index],
                            dtype=np.float32,
                        )
                    )
                    source_frame9 = np.concatenate(
                        (
                            np.zeros(3, dtype=np.float32),
                            np.asarray(
                                self.payload["source_frame6_columns"][local_index],
                                dtype=np.float32,
                            ),
                        )
                    )
                    source_local_frame = relative_frame9_to_target_frame(
                        local_relative_frame, source_frame9
                    )
                    source_local_frame[:3] /= float(
                        np.asarray(self.norm.xyz_std, dtype=np.float32).mean()
                    )
                result["target_frame9_source_local"] = torch.from_numpy(
                    source_local_frame
                )
            if all(
                key in self.payload
                for key in ("goal_frame9", "current_eef_pose7")
            ):
                state20 = np.asarray(self.payload["state"][index], dtype=np.float32)
                hand_object_frame = active_hand_object_frame11_from_goal_relative(
                    np.asarray(self.payload["goal_frame9"][index], dtype=np.float32),
                    np.asarray(self.payload["target_frame9"][index], dtype=np.float32),
                    state20,
                    translation_scale_m=HAND_OBJECT_TRANSLATION_SCALE_M,
                )
                result["hand_object_frame11"] = torch.from_numpy(
                    hand_object_frame
                )
                result["hand_object_frame_confidence"] = torch.tensor(
                    float(
                        self.payload.get(
                            "source_frame_confidence",
                            np.ones(
                                len(self.payload["target_frame9"]),
                                dtype=np.float32,
                            ),
                        )[index]
                    ),
                    dtype=torch.float32,
                )
            if "target_frame9_local" in self.payload:
                if self.target_frame_mode == "zero":
                    local_frame = np.zeros(9, dtype=np.float32)
                else:
                    local_frame = np.asarray(
                        self.payload["target_frame9_local"][frame_index],
                        dtype=np.float32,
                    ).copy()
                    local_frame[:3] /= self.norm.xyz_std
                result["target_frame9_local"] = torch.from_numpy(local_frame)
        current_anchors = self.payload["current_anchors"][index]
        result["target_flow"] = self._cached_target(
            sample_index=index,
            episode=episode,
            current_anchors=current_anchors,
        )
        result["shuffled_target_flow"] = self._cached_target(
            sample_index=index,
            episode=self.wrong_episode[episode],
            current_anchors=current_anchors,
        )
        result["target_rotation"] = self._cached_rotation(
            sample_index=index,
            episode=episode,
            current_anchors=current_anchors,
        )
        result["shuffled_target_rotation"] = self._cached_rotation(
            sample_index=index,
            episode=self.wrong_episode[episode],
            current_anchors=current_anchors,
        )
        return result


def build_model(
    condition: str,
    *,
    horizon: int,
    flow_steps: int,
    num_anchors: int,
    feature_dim: int,
    layers: int,
    flow_parameterization: str,
    xyz_std: np.ndarray,
    action_std: np.ndarray | None = None,
    predict_rotation: bool = False,
    point_channels: int = 3,
    target_color_adapter: bool = False,
    source_geometry_adapter: bool = False,
    shared_geometry_adapter: bool = False,
    functional_coordinate_adapter: bool = False,
    source_axis_relation_adapter: bool = False,
    source_axis_gate_initial_value: float = 0.0,
    target_axis_relation_adapter: bool = False,
    hand_object_frame_token: bool = False,
    diffusion_steps: int = 100,
    diffusion_inference_steps: int = 10,
    diffusion_prediction_type: str = "sample",
) -> nn.Module:
    if condition == "point_tokens":
        return PointTokenActionPolicy(
            horizon=horizon,
            feature_dim=feature_dim,
            layers=layers,
            point_channels=point_channels,
            target_color_adapter=target_color_adapter,
            source_geometry_adapter=source_geometry_adapter,
            shared_geometry_adapter=shared_geometry_adapter,
        )
    if condition in CONDITIONS:
        # Keep the auxiliary head instantiated for all interaction conditions so
        # parameter count is identical.  Only the declared flow conditions train it.
        return InteractionFlowTokenPolicy(
            horizon=horizon,
            flow_steps=flow_steps,
            num_anchors=num_anchors,
            feature_dim=feature_dim,
            layers=layers,
            predict_flow=True,
            predict_rotation=predict_rotation,
            flow_parameterization=flow_parameterization,
            xyz_std=[float(value) for value in xyz_std],
            action_std=(
                None
                if action_std is None
                else [float(value) for value in action_std]
            ),
            point_channels=point_channels,
            target_color_adapter=target_color_adapter,
            source_geometry_adapter=source_geometry_adapter,
            shared_geometry_adapter=shared_geometry_adapter,
            functional_coordinate_adapter=functional_coordinate_adapter,
            source_axis_relation_adapter=source_axis_relation_adapter,
            source_axis_gate_initial_value=source_axis_gate_initial_value,
            target_axis_relation_adapter=target_axis_relation_adapter,
            condition_action_on_flow=condition == "interaction_flow_conditioned",
            functional_frame_token=condition in {
                "interaction_functional",
                "interaction_functional_flow",
                "interaction_functional_direct",
                "interaction_functional_direct_flow",
                "interaction_functional_dual_frame_flow",
                "interaction_functional_correction_adapter_flow",
                "interaction_functional_equivariant_adapter_flow",
                "interaction_functional_local_equivariant_adapter_flow",
                "interaction_functional_hybrid_equivariant_adapter_flow",
                "interaction_functional_biframe_adapter_flow",
                "interaction_functional_gated_biframe_adapter_flow",
                "interaction_functional_grasp_adapter_flow",
                "interaction_functional_gated",
                "interaction_functional_gated_flow",
                "interaction_functional_relative",
                "interaction_relative_flow",
                "interaction_relative_diffusion",
            },
            functional_frame_in_scene=condition in {
                "interaction_functional_direct",
                "interaction_functional_direct_flow",
                "interaction_functional_dual_frame_flow",
                "interaction_functional_correction_adapter_flow",
                "interaction_functional_equivariant_adapter_flow",
                "interaction_functional_local_equivariant_adapter_flow",
                "interaction_functional_hybrid_equivariant_adapter_flow",
                "interaction_functional_biframe_adapter_flow",
                "interaction_functional_gated_biframe_adapter_flow",
                "interaction_functional_grasp_adapter_flow",
                "interaction_functional_gated",
                "interaction_functional_gated_flow",
                "interaction_functional_relative",
                "interaction_relative_flow",
                "interaction_relative_diffusion",
            },
            functional_frame_gated_scene=condition in {
                "interaction_functional_gated",
                "interaction_functional_gated_flow",
                "interaction_functional_relative",
                "interaction_relative_flow",
                "interaction_relative_diffusion",
            },
            functional_frame_as_memory=condition not in {
                "interaction_functional_gated",
                "interaction_functional_gated_flow",
                "interaction_functional_relative",
                "interaction_relative_flow",
                "interaction_relative_diffusion",
            },
            dual_functional_frame_token=(
                condition == "interaction_functional_dual_frame_flow"
            ),
            functional_relative_flow=condition in {
                "interaction_functional_relative",
                "interaction_relative_flow",
                "interaction_relative_diffusion",
            },
            recovery_action_adapter=(
                condition
                in {
                    "interaction_functional_correction_adapter_flow",
                    "interaction_functional_equivariant_adapter_flow",
                    "interaction_functional_local_equivariant_adapter_flow",
                    "interaction_functional_hybrid_equivariant_adapter_flow",
                    "interaction_functional_biframe_adapter_flow",
                    "interaction_functional_gated_biframe_adapter_flow",
                    "interaction_functional_grasp_adapter_flow",
                }
            ),
            recovery_action_frame=(
                "hybrid_source_rotation"
                if condition
                == "interaction_functional_hybrid_equivariant_adapter_flow"
                else "source_functional"
                if condition in {
                    "interaction_functional_equivariant_adapter_flow",
                    "interaction_functional_local_equivariant_adapter_flow",
                }
                else "world"
            ),
            recovery_local_frame_token=(
                condition in {
                    "interaction_functional_local_equivariant_adapter_flow",
                    "interaction_functional_hybrid_equivariant_adapter_flow",
                    "interaction_functional_biframe_adapter_flow",
                    "interaction_functional_gated_biframe_adapter_flow",
                }
            ),
            recovery_local_frame_gated_fusion=(
                condition
                == "interaction_functional_gated_biframe_adapter_flow"
            ),
            hand_object_frame_token=(
                hand_object_frame_token
                or condition == "interaction_functional_grasp_adapter_flow"
            ),
            action_decoder_type=(
                "diffusion"
                if condition in {
                    "interaction_diffusion",
                    "interaction_relative_diffusion",
                }
                else "regression"
            ),
            diffusion_steps=diffusion_steps,
            diffusion_inference_steps=diffusion_inference_steps,
            diffusion_prediction_type=diffusion_prediction_type,
        )
    raise ValueError(f"unknown condition {condition!r}")


def condition_uses_flow_loss(condition: str) -> bool:
    return condition in {
        "interaction_flow",
        "interaction_flow_shuffled",
        "interaction_flow_conditioned",
        "interaction_functional_flow",
        "interaction_functional_direct_flow",
        "interaction_functional_dual_frame_flow",
        "interaction_functional_correction_adapter_flow",
        "interaction_functional_equivariant_adapter_flow",
        "interaction_functional_local_equivariant_adapter_flow",
        "interaction_functional_hybrid_equivariant_adapter_flow",
        "interaction_functional_biframe_adapter_flow",
        "interaction_functional_gated_biframe_adapter_flow",
        "interaction_functional_grasp_adapter_flow",
        "interaction_functional_gated_flow",
        "interaction_relative_flow",
        "interaction_diffusion",
        "interaction_relative_diffusion",
    }


def condition_uses_diffusion(condition: str) -> bool:
    return condition in {
        "interaction_diffusion",
        "interaction_relative_diffusion",
    }


def condition_flow_target(
    condition: str, batch: dict[str, torch.Tensor]
) -> torch.Tensor:
    if condition == "interaction_flow_shuffled":
        return batch["shuffled_target_flow"]
    return batch["target_flow"]


def condition_rotation_target(
    condition: str, batch: dict[str, torch.Tensor]
) -> torch.Tensor:
    if condition == "interaction_flow_shuffled":
        return batch["shuffled_target_rotation"]
    return batch["target_rotation"]


def flow_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mode: str,
    motion_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mode == "unordered":
        geometry = unordered_flow_chamfer_loss(prediction, target)
    elif mode == "correspondence":
        geometry = correspondence_flow_loss(prediction, target)
    elif mode == "hybrid":
        geometry = correspondence_flow_loss(prediction, target)
        geometry = geometry + 0.25 * unordered_flow_chamfer_loss(
            prediction, target
        )
    elif mode == "motion":
        geometry = motion_correspondence_flow_loss(
            prediction, target, motion_scale=motion_scale
        )
    else:
        raise ValueError(f"unknown flow loss mode {mode!r}")
    return geometry, rigid_flow_loss(prediction)


def _rotation_matrix_from_rotvec(rotation_vector: torch.Tensor) -> torch.Tensor:
    """Differentiable exponential map from an axis-angle vector to SO(3)."""
    theta_squared = torch.sum(rotation_vector.square(), dim=-1, keepdim=True)
    theta = torch.sqrt(theta_squared.clamp_min(1e-16))
    # Taylor branches keep both the value and gradient well behaved at zero.
    coefficient_a = torch.where(
        theta_squared < 1e-8,
        1.0 - theta_squared / 6.0 + theta_squared.square() / 120.0,
        torch.sin(theta) / theta,
    )
    coefficient_b = torch.where(
        theta_squared < 1e-8,
        0.5 - theta_squared / 24.0 + theta_squared.square() / 720.0,
        (1.0 - torch.cos(theta)) / theta_squared.clamp_min(1e-16),
    )
    x, y, z = rotation_vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    skew = torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero), dim=-1
    ).reshape(rotation_vector.shape[:-1] + (3, 3))
    identity = torch.eye(
        3, dtype=rotation_vector.dtype, device=rotation_vector.device
    ).expand(rotation_vector.shape[:-1] + (3, 3))
    return (
        identity
        + coefficient_a[..., None] * skew
        + coefficient_b[..., None] * (skew @ skew)
    )


def active_endpoint_rotation_geodesic_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    active_arm_right: torch.Tensor,
    action_mean: torch.Tensor | np.ndarray,
    action_std: torch.Tensor | np.ndarray,
    *,
    action_frame_eef: bool = False,
    sample_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """SO(3) error after composing the active arm's complete action chunk.

    The policy predicts normalized 14-D bimanual actions.  This loss first
    restores physical rotvec units, then composes the active arm's increments
    exactly as deployment does.  It therefore penalizes accumulated endpoint
    orientation rather than treating three rotvec coordinates independently.
    """
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError(
            "prediction and target must have identical [batch,horizon,14] shape"
        )
    mean = torch.as_tensor(
        action_mean, dtype=prediction.dtype, device=prediction.device
    ).reshape(1, 1, -1)
    std = torch.as_tensor(
        action_std, dtype=prediction.dtype, device=prediction.device
    ).reshape(1, 1, -1)
    if mean.shape[-1] != prediction.shape[-1] or std.shape[-1] != prediction.shape[-1]:
        raise ValueError("action normalization dimension does not match prediction")
    prediction_physical = prediction * std + mean
    target_physical = target * std + mean
    batch, horizon = prediction.shape[:2]
    active_offset = torch.where(
        active_arm_right.reshape(-1) >= 0.5,
        torch.full((batch,), 10, device=prediction.device, dtype=torch.long),
        torch.full((batch,), 3, device=prediction.device, dtype=torch.long),
    )
    rotation_index = active_offset[:, None, None].expand(batch, horizon, 3)
    rotation_index = rotation_index + torch.arange(
        3, device=prediction.device, dtype=torch.long
    ).reshape(1, 1, 3)
    prediction_rotvec = torch.gather(prediction_physical, 2, rotation_index)
    target_rotvec = torch.gather(target_physical, 2, rotation_index)
    prediction_delta = _rotation_matrix_from_rotvec(prediction_rotvec)
    target_delta = _rotation_matrix_from_rotvec(target_rotvec)
    prediction_endpoint = torch.eye(
        3, dtype=prediction.dtype, device=prediction.device
    ).expand(batch, 3, 3)
    target_endpoint = prediction_endpoint.clone()
    for step in range(horizon):
        if action_frame_eef:
            prediction_endpoint = prediction_endpoint @ prediction_delta[:, step]
            target_endpoint = target_endpoint @ target_delta[:, step]
        else:
            prediction_endpoint = prediction_delta[:, step] @ prediction_endpoint
            target_endpoint = target_delta[:, step] @ target_endpoint
    relative = prediction_endpoint @ target_endpoint.transpose(-1, -2)
    vee = 0.5 * torch.stack(
        (
            relative[:, 2, 1] - relative[:, 1, 2],
            relative[:, 0, 2] - relative[:, 2, 0],
            relative[:, 1, 0] - relative[:, 0, 1],
        ),
        dim=-1,
    )
    sine = torch.sqrt(torch.sum(vee.square(), dim=-1) + 1e-12)
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(
        -1.0, 1.0
    )
    angle = torch.atan2(sine, cosine)
    if sample_mask is not None:
        keep = sample_mask.reshape(-1).to(dtype=torch.bool, device=prediction.device)
        if not torch.any(keep):
            return prediction.sum() * 0.0
        angle = angle[keep]
    return angle.mean()


def validation_action_objective(
    metrics: dict[str, float | None], endpoint_rotation_weight: float
) -> float:
    result = float(metrics["action"])
    if endpoint_rotation_weight > 0.0:
        endpoint = metrics.get("endpoint_rotation_rad")
        if endpoint is None:
            raise RuntimeError("endpoint rotation objective was not computed")
        result += float(endpoint_rotation_weight) * float(endpoint)
    return result


def initialize_recovery_adapter_from_decoder(model: nn.Module) -> None:
    """Clone the trained nominal decoder into an exact-zero residual adapter.

    A randomly initialized deep residual decoder is an unnecessary source of
    seed variance: its zero action head initially hides random queries and
    attention blocks, but those random features become active after the first
    optimizer steps.  Cloning the nominal decoder supplies an action-aligned
    representation while re-zeroing the final head preserves an exact no-op.
    """
    adapter = getattr(model, "recovery_action_adapter", None)
    decoder = getattr(model, "decoder", None)
    if adapter is None:
        raise ValueError("model has no recovery action adapter to initialize")
    if decoder is None:
        raise ValueError("model has no nominal action decoder to clone")
    adapter.load_state_dict(decoder.state_dict(), strict=True)
    nn.init.zeros_(adapter.action_head.weight)
    nn.init.zeros_(adapter.action_head.bias)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    condition: str,
    *,
    rotation_action_loss_weight: float = 1.0,
    endpoint_rotation_action_loss_weight: float = 0.0,
    action_mean: torch.Tensor | np.ndarray | None = None,
    action_std: torch.Tensor | np.ndarray | None = None,
    action_frame_eef: bool = False,
) -> dict[str, float | None]:
    model.eval()
    action_total = 0.0
    correct_flow_total = 0.0
    supervised_flow_total = 0.0
    rigidity_total = 0.0
    rotation_total = 0.0
    endpoint_rotation_total = 0.0
    count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        action_target = batch["action"]
        if condition_uses_diffusion(condition):
            noisy, timestep, action_target = model.decoder.training_inputs(
                batch["action"], sample_index=batch["index"]
            )
            batch["diffusion_noisy_action"] = noisy
            batch["diffusion_timestep"] = timestep
        output = model(batch)
        batch_count = int(len(action_target))
        action_loss = weighted_action_loss(
            output.action,
            action_target,
            batch["active_arm_right"],
            batch["relation_phase"],
            active_arm_weight=1.0,
            relation_weight=1.0,
            rotation_channel_weight=rotation_action_loss_weight,
        )
        action_total += float(action_loss.item()) * batch_count
        if endpoint_rotation_action_loss_weight > 0.0:
            if action_mean is None or action_std is None:
                raise ValueError(
                    "endpoint rotation validation requires action normalization"
                )
            endpoint_rotation = active_endpoint_rotation_geodesic_loss(
                output.action,
                action_target,
                batch["active_arm_right"],
                action_mean,
                action_std,
                action_frame_eef=action_frame_eef,
            )
            endpoint_rotation_total += float(endpoint_rotation.item()) * batch_count
        count += batch_count
        if output.predicted_flow is not None:
            correct = unordered_flow_chamfer_loss(
                output.predicted_flow, batch["target_flow"]
            )
            supervised = unordered_flow_chamfer_loss(
                output.predicted_flow, condition_flow_target(condition, batch)
            )
            rigidity = rigid_flow_loss(output.predicted_flow)
            correct_flow_total += float(correct.item()) * batch_count
            supervised_flow_total += float(supervised.item()) * batch_count
            rigidity_total += float(rigidity.item()) * batch_count
        if output.predicted_rotation is not None:
            rotation = rotation_geodesic_loss(
                output.predicted_rotation, batch["target_rotation"]
            )
            rotation_total += float(rotation.item()) * batch_count
    return {
        "action": action_total / max(count, 1),
        "correct_flow": correct_flow_total / max(count, 1) if correct_flow_total else None,
        "supervised_flow": (
            supervised_flow_total / max(count, 1) if supervised_flow_total else None
        ),
        "rigidity": rigidity_total / max(count, 1) if rigidity_total else None,
        "rotation_rad": (
            rotation_total / max(count, 1) if rotation_total else None
        ),
        "endpoint_rotation_rad": (
            endpoint_rotation_total / max(count, 1)
            if endpoint_rotation_action_loss_weight > 0.0
            else None
        ),
    }


def normalized_flow_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    xyz_std: np.ndarray,
) -> dict[str, float]:
    scale = torch.as_tensor(xyz_std, dtype=prediction.dtype, device=prediction.device)
    prediction_m = prediction * scale
    target_m = target * scale
    pred_time = prediction_m.permute(0, 2, 1, 3)
    target_time = target_m.permute(0, 2, 1, 3)
    distance = torch.cdist(pred_time, target_time)
    symmetric = 0.5 * (
        distance.min(dim=3).values.mean(dim=2)
        + distance.min(dim=2).values.mean(dim=2)
    )
    with torch.no_grad():
        current_index = torch.cdist(
            prediction_m[:, :, 0], target_m[:, :, 0]
        ).argmin(dim=-1)
        gather_index = current_index[:, :, None, None].expand(
            -1, -1, target_m.shape[2], 3
        )
        matched_target = torch.gather(target_m, dim=1, index=gather_index)
        predicted_endpoint_motion = prediction_m[:, :, -1] - prediction_m[:, :, 0]
        target_endpoint_motion = matched_target[:, :, -1] - matched_target[:, :, 0]
        endpoint_motion_error = torch.linalg.vector_norm(
            predicted_endpoint_motion - target_endpoint_motion, dim=-1
        ).mean()
    return {
        "trajectory_chamfer_cm": float(symmetric.mean().item() * 100.0),
        "endpoint_chamfer_cm": float(symmetric[:, -1].mean().item() * 100.0),
        "endpoint_motion_epe_cm": float(endpoint_motion_error.item() * 100.0),
    }


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    normalization,
    action_frame: str = "world",
) -> tuple[np.ndarray, np.ndarray, dict[str, float | None]]:
    model.eval()
    indices = []
    actions = []
    flow_metrics = []
    entropy_a = []
    entropy_b = []
    anchor_spread = []
    rotation_error = []
    for batch in loader:
        batch = move_batch(batch, device)
        if condition_uses_diffusion(getattr(model, "experiment_condition", "")):
            batch["diffusion_initial_noise"] = stateless_normal_like(
                batch["action"], batch["index"], salt=97.0
            )
        output = model(batch)
        action = (
            output.action.cpu().numpy()
            * normalization.action_std[None, None, :]
            + normalization.action_mean[None, None, :]
        )
        if action_frame == "target":
            action = action14_from_target_frame(
                action, batch["target_frame9"].cpu().numpy()
            )
        elif action_frame != "world":
            raise ValueError(f"unknown action frame {action_frame!r}")
        actions.append(action)
        indices.append(batch["index"].cpu().numpy())
        if output.predicted_flow is not None:
            flow_metrics.append(
                normalized_flow_metrics(
                    output.predicted_flow,
                    batch["target_flow"],
                    normalization.xyz_std,
                )
            )
        if output.predicted_rotation is not None:
            rotation_error.append(
                float(
                    rotation_geodesic_loss(
                        output.predicted_rotation, batch["target_rotation"]
                    ).item()
                )
            )
        if output.attention_a is not None:
            entropy_a.append(float(attention_entropy(output.attention_a).item()))
            entropy_b.append(float(attention_entropy(output.attention_b).item()))
            spread = torch.cdist(output.anchor_a, output.anchor_a)
            pair_mask = ~torch.eye(
                spread.shape[-1], dtype=torch.bool, device=spread.device
            )[None]
            xyz_scale = torch.as_tensor(
                normalization.xyz_std,
                dtype=output.anchor_a.dtype,
                device=output.anchor_a.device,
            )
            spread_m = torch.cdist(
                output.anchor_a * xyz_scale,
                output.anchor_a * xyz_scale,
            )
            anchor_spread.append(float(spread_m[pair_mask.expand_as(spread_m)].mean().item()))
    diagnostic: dict[str, float | None] = {
        "trajectory_chamfer_cm": None,
        "endpoint_chamfer_cm": None,
        "endpoint_motion_epe_cm": None,
        "attention_entropy_a": float(np.mean(entropy_a)) if entropy_a else None,
        "attention_entropy_b": float(np.mean(entropy_b)) if entropy_b else None,
        "anchor_pair_distance_cm": (
            float(np.mean(anchor_spread) * 100.0) if anchor_spread else None
        ),
        "relative_rotation_error_deg": (
            float(np.rad2deg(np.mean(rotation_error)))
            if rotation_error
            else None
        ),
    }
    if flow_metrics:
        for key in (
            "trajectory_chamfer_cm",
            "endpoint_chamfer_cm",
            "endpoint_motion_epe_cm",
        ):
            diagnostic[key] = float(np.mean([value[key] for value in flow_metrics]))
    return np.concatenate(indices), np.concatenate(actions), diagnostic


def summarize_phases(
    payload: dict[str, np.ndarray],
    indices: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, dict[str, float]]:
    result = {"all": summarize_errors(payload, indices, prediction)}
    for name, value in (("pre_relation", 0.0), ("relation", 1.0)):
        keep = np.isclose(payload["relation_phase"][indices], value)
        if np.any(keep):
            result[name] = summarize_errors(
                payload, indices[keep], prediction[keep]
            )
    return result


def oracle_relative_frame9(
    payload: dict[str, np.ndarray],
    *,
    legacy_pose9_decode: bool = False,
) -> np.ndarray:
    """Privileged ceiling: per-frame error to the intended task goal.

    The old implementation used the demonstrated trajectory endpoint, which
    leaks where that particular expert rollout happened to stop.  These labels
    are instead computed by the dataset builder from current A/B poses and the
    task relation.  ``legacy_pose9_decode`` is retained only for API/checkpoint
    compatibility and intentionally has no effect.
    """
    del legacy_pose9_decode
    required = (
        "goal_translation_error_xyz_m",
        "goal_rotation_error_rotvec",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(
            "intended-goal oracle frame requires dataset labels "
            f"{missing}; refusing to fall back to an expert endpoint"
        )
    translation = np.asarray(
        payload["goal_translation_error_xyz_m"], dtype=np.float32
    )
    rotation_vector = np.asarray(
        payload["goal_rotation_error_rotvec"], dtype=np.float32
    )
    delta_rotation = Rotation.from_rotvec(rotation_vector).as_matrix()
    return np.concatenate(
        (
            translation,
            delta_rotation[..., :, 0],
            delta_rotation[..., :, 1],
        ),
        axis=-1,
    ).astype(np.float32)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument(
        "--conditions",
        nargs="+",
        default=[
            "point_tokens",
            "interaction_tokens",
            "interaction_flow",
            "interaction_flow_shuffled",
        ],
    )
    result.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    result.add_argument(
        "--train-ids",
        nargs="+",
        type=int,
        help=(
            "Explicit training object IDs. Requires validation-ids and test-ids; "
            "use --folds 0 so output filenames remain deployment-compatible."
        ),
    )
    result.add_argument("--validation-ids", nargs="+", type=int)
    result.add_argument("--test-ids", nargs="+", type=int)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--epochs", type=int, default=120)
    result.add_argument("--patience", type=int, default=18)
    result.add_argument("--batch-size", type=int, default=256)
    result.add_argument("--learning-rate", type=float, default=3e-4)
    result.add_argument("--weight-decay", type=float, default=1e-5)
    result.add_argument("--feature-dim", type=int, default=128)
    result.add_argument("--layers", type=int, default=2)
    result.add_argument("--num-anchors", type=int, default=16)
    result.add_argument("--flow-steps", type=int, default=4)
    result.add_argument("--diffusion-steps", type=int, default=100)
    result.add_argument("--diffusion-inference-steps", type=int, default=10)
    result.add_argument(
        "--diffusion-prediction-type",
        choices=("sample", "epsilon"),
        default="sample",
    )
    result.add_argument("--geometry-loss-weight", type=float, default=0.1)
    result.add_argument("--rigidity-loss-weight", type=float, default=0.02)
    result.add_argument("--rotation-loss-weight", type=float, default=0.0)
    result.add_argument("--target-color-adapter", action="store_true")
    result.add_argument(
        "--source-geometry-adapter",
        action="store_true",
        help=(
            "Treat point channels after XYZ as source-object descriptors and "
            "inject them through a zero-start gated token adapter."
        ),
    )
    result.add_argument(
        "--shared-geometry-adapter",
        action="store_true",
        help=(
            "Project frozen point descriptors for both A and B through one "
            "zero-start gated adapter, so relation attention can compare them "
            "in a shared feature space."
        ),
    )
    result.add_argument(
        "--source-axis-relation-adapter",
        action="store_true",
        help=(
            "Inject source_axis3 through a zero-start gated relation adapter "
            "with explicit axial and cross-displacement terms."
        ),
    )
    result.add_argument(
        "--source-axis-gate-initial-value",
        type=float,
        default=0.0,
        help=(
            "Initial pre-tanh gate for task-aligned source-axis relation tokens. "
            "A positive value avoids starving the token projection while the "
            "all-zero axis control remains an exact no-op."
        ),
    )
    result.add_argument(
        "--target-axis-relation-adapter",
        action="store_true",
        help=(
            "Add target_axis3 and explicit source-target/delta relations through "
            "a zero-start gated adapter. Requires --source-axis-relation-adapter."
        ),
    )
    result.add_argument(
        "--flow-loss-mode",
        choices=("unordered", "correspondence", "hybrid", "motion"),
        default="unordered",
    )
    result.add_argument("--motion-loss-scale", type=float, default=10.0)
    result.add_argument(
        "--flow-parameterization",
        choices=("free", "rigid_se3"),
        default="free",
    )
    result.add_argument(
        "--target-horizon-mode",
        choices=("terminal", "action_chunk"),
        default="terminal",
    )
    result.add_argument(
        "--geometry-phase",
        choices=("all", "relation"),
        default="all",
    )
    result.add_argument("--active-arm-loss-weight", type=float, default=1.0)
    result.add_argument("--relation-loss-weight", type=float, default=1.0)
    result.add_argument("--rotation-action-loss-weight", type=float, default=1.0)
    result.add_argument(
        "--endpoint-rotation-action-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for the active arm's SO(3) geodesic error after composing "
            "the complete predicted action chunk. Unlike per-channel rotvec "
            "MSE, this directly supervises the deployed endpoint orientation."
        ),
    )
    result.add_argument(
        "--action-frame", choices=("world", "target"), default="world"
    )
    result.add_argument(
        "--policy-frame",
        choices=(
            "world",
            "goal_action",
            "goal_state_action",
            "goal_geometry",
            "goal",
        ),
        default="world",
        help=(
            "Coordinate system shared by point clouds, proprioception and "
            "action labels. 'goal' uses the deployable camera marker goal frame."
        ),
    )
    result.add_argument(
        "--functional-frame-source",
        choices=("target_camera", "oracle_relative", "dataset_relative"),
        default="target_camera",
    )
    result.add_argument(
        "--functional-point-coordinates",
        action="store_true",
        help=(
            "Append per-point coordinates in the NDF current functional frame "
            "for A and the camera goal frame for B. The explicit SE(3) token is "
            "retained, yielding task-aligned local and global geometry levels."
        ),
    )
    result.add_argument(
        "--functional-coordinate-adapter",
        action="store_true",
        help=(
            "Encode XYZ/RGB through the baseline point encoder and inject only "
            "the final three functional coordinates through a shared zero-start "
            "gated adapter. Requires --functional-point-coordinates."
        ),
    )
    result.add_argument(
        "--zero-global-functional-frame-token",
        action="store_true",
        help=(
            "Keep task-aligned per-point coordinates but zero the global SE(3) "
            "token. This is a paired two-level contribution ablation."
        ),
    )
    result.add_argument(
        "--target-frame-mode",
        choices=("normal", "zero", "shuffled"),
        default="normal",
        help=(
            "Train, validate, and test with the correct frame, an exact-zero "
            "capacity control, or a difficulty/progress-matched wrong frame."
        ),
    )
    result.add_argument("--max-train-samples", type=int)
    result.add_argument(
        "--sample-mode",
        choices=("all", "nominal", "recovery"),
        default="all",
        help="Optionally train/evaluate only nominal or recovery rows.",
    )
    result.add_argument(
        "--policy-phase",
        choices=("all", "pre_relation", "relation", "pre_release"),
        default="all",
        help=(
            "Optionally scope the learned policy to one observable task segment. "
            "pre_release keeps relation rows before the demonstrated release. "
            "This filters the training/evaluation rows; it does not add a phase "
            "prediction head or provide phase labels at deployment."
        ),
    )
    result.add_argument("--initialize-from-checkpoint", type=Path)
    result.add_argument(
        "--initialize-recovery-adapter-from-decoder",
        action="store_true",
        help=(
            "After loading the nominal checkpoint, clone its trained action "
            "decoder into the recovery adapter and re-zero the adapter output "
            "head. This remains an exact initial no-op while removing random "
            "deep-adapter initialization."
        ),
    )
    result.add_argument("--train-geometry-adapter-only", action="store_true")
    result.add_argument(
        "--train-recovery-action-adapter-only",
        action="store_true",
        help=(
            "Freeze an initialized nominal policy and train only the zero-start "
            "geometry-conditioned action adapter."
        ),
    )
    result.add_argument(
        "--train-hand-object-adapter-only",
        action="store_true",
        help=(
            "Freeze an initialized recovery policy and train only the new "
            "camera/proprioceptive hand-object SE(3) projection."
        ),
    )
    result.add_argument(
        "--train-target-axis-adapter-only",
        action="store_true",
        help=(
            "Freeze an initialized source-axis policy and train only the new "
            "target-axis relation adapter."
        ),
    )
    result.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    result.add_argument("--num-workers", type=int, default=2)
    result.add_argument(
        "--episode-balanced-sampling",
        action="store_true",
        help="Weight each training frame inversely by its episode length.",
    )
    result.add_argument(
        "--recovery-target-fraction",
        type=float,
        help=(
            "When nominal and synthetic correction rows are mixed, sample "
            "correction rows with this total probability. Validation/test and "
            "normalization remain nominal-only."
        ),
    )
    result.add_argument(
        "--minimum-recovery-train-target-confidence",
        type=float,
        help=(
            "Exclude recovery rows below this camera-geometry confidence from "
            "training only. Validation and test rows remain untouched. This "
            "matches the recovery adapter's deployment gate: a zero-confidence "
            "row has an exactly zero adapter gradient and must not consume the "
            "fixed recovery sampling budget."
        ),
    )
    result.add_argument(
        "--recovery-validation-weight",
        type=float,
        help=(
            "Add this weight times a separately averaged real-recovery "
            "validation loss to the nominal validation objective. Recovery "
            "rows remain excluded from ordinary validation/test metrics."
        ),
    )
    result.add_argument(
        "--maximum-nominal-validation-relative-degradation",
        type=float,
        help=(
            "Hard checkpoint-selection constraint relative to the initialized "
            "policy's nominal validation loss (for example 0.05 permits 5%%)."
        ),
    )
    result.add_argument(
        "--nominal-adapter-distillation-weight",
        type=float,
        help=(
            "For recovery-adapter-only training, learn BC only on recovery "
            "rows and penalize the gated adapter residual on nominal rows."
        ),
    )
    return result


def main() -> None:
    args = parser().parse_args()
    explicit_object_split = resolve_policy_object_split(
        args.train_ids, args.validation_ids, args.test_ids
    )
    if explicit_object_split is not None and args.folds != [0]:
        raise ValueError("an explicit object split requires exactly --folds 0")
    if float(args.endpoint_rotation_action_loss_weight) < 0.0:
        raise ValueError(
            "--endpoint-rotation-action-loss-weight must be nonnegative"
        )
    if args.functional_point_coordinates and args.functional_frame_source != "dataset_relative":
        raise ValueError(
            "functional point coordinates require --functional-frame-source=dataset_relative"
        )
    if args.zero_global_functional_frame_token and not args.functional_point_coordinates:
        raise ValueError(
            "--zero-global-functional-frame-token requires "
            "--functional-point-coordinates"
        )
    if args.functional_coordinate_adapter and not args.functional_point_coordinates:
        raise ValueError(
            "--functional-coordinate-adapter requires "
            "--functional-point-coordinates"
        )
    if float(args.endpoint_rotation_action_loss_weight) > 0.0 and any(
        condition_uses_diffusion(condition) for condition in args.conditions
    ):
        raise ValueError(
            "endpoint rotation action loss currently requires a regression decoder"
        )
    if args.recovery_target_fraction is not None and not (
        0.0 < float(args.recovery_target_fraction) < 1.0
    ):
        raise ValueError("--recovery-target-fraction must lie in (0,1)")
    if args.minimum_recovery_train_target_confidence is not None and not (
        0.0 <= float(args.minimum_recovery_train_target_confidence) <= 1.0
    ):
        raise ValueError(
            "--minimum-recovery-train-target-confidence must lie in [0,1]"
        )
    if args.recovery_validation_weight is not None and not (
        float(args.recovery_validation_weight) > 0.0
    ):
        raise ValueError("--recovery-validation-weight must be positive")
    if args.maximum_nominal_validation_relative_degradation is not None and not (
        float(args.maximum_nominal_validation_relative_degradation) >= 0.0
    ):
        raise ValueError(
            "--maximum-nominal-validation-relative-degradation must be nonnegative"
        )
    if (
        args.maximum_nominal_validation_relative_degradation is not None
        and args.initialize_from_checkpoint is None
    ):
        raise ValueError(
            "nominal degradation constraint requires checkpoint initialization"
        )
    if args.nominal_adapter_distillation_weight is not None:
        if float(args.nominal_adapter_distillation_weight) <= 0.0:
            raise ValueError(
                "--nominal-adapter-distillation-weight must be positive"
            )
        if not (
            args.train_recovery_action_adapter_only
            or args.train_hand_object_adapter_only
        ):
            raise ValueError(
                "nominal adapter distillation requires "
                "an action- or hand-object-adapter-only training mode"
            )
    unknown = set(args.conditions) - CONDITIONS
    if unknown:
        raise ValueError(f"unknown conditions: {sorted(unknown)}")
    if args.train_geometry_adapter_only and args.initialize_from_checkpoint is None:
        raise ValueError(
            "--train-geometry-adapter-only requires --initialize-from-checkpoint"
        )
    if (
        args.train_recovery_action_adapter_only
        and args.initialize_from_checkpoint is None
    ):
        raise ValueError(
            "--train-recovery-action-adapter-only requires "
            "--initialize-from-checkpoint"
        )
    if (
        args.initialize_recovery_adapter_from_decoder
        and not args.train_recovery_action_adapter_only
    ):
        raise ValueError(
            "decoder-cloned recovery initialization requires "
            "--train-recovery-action-adapter-only"
        )
    if (
        args.train_hand_object_adapter_only
        and args.initialize_from_checkpoint is None
    ):
        raise ValueError(
            "--train-hand-object-adapter-only requires "
            "--initialize-from-checkpoint"
        )
    if (
        args.train_target_axis_adapter_only
        and args.initialize_from_checkpoint is None
    ):
        raise ValueError(
            "--train-target-axis-adapter-only requires --initialize-from-checkpoint"
        )
    adapter_modes = (
        args.train_geometry_adapter_only,
        args.train_recovery_action_adapter_only,
        args.train_hand_object_adapter_only,
        args.train_target_axis_adapter_only,
    )
    if sum(bool(value) for value in adapter_modes) > 1:
        raise ValueError("adapter-only training modes are mutually exclusive")
    if args.target_axis_relation_adapter and not args.source_axis_relation_adapter:
        raise ValueError(
            "--target-axis-relation-adapter requires --source-axis-relation-adapter"
        )
    if args.train_target_axis_adapter_only and not args.target_axis_relation_adapter:
        raise ValueError(
            "--train-target-axis-adapter-only requires --target-axis-relation-adapter"
        )
    if args.train_hand_object_adapter_only and args.conditions != [
        "interaction_functional_grasp_adapter_flow"
    ]:
        raise ValueError(
            "--train-hand-object-adapter-only requires exactly the "
            "interaction_functional_grasp_adapter_flow condition"
        )
    if args.initialize_from_checkpoint is not None and len(args.conditions) != 1:
        raise ValueError(
            "checkpoint initialization currently requires exactly one condition"
        )
    if (
        any(
            condition in {
                "interaction_relative_flow",
                "interaction_relative_diffusion",
            }
            for condition in args.conditions
        )
        and args.functional_frame_source != "oracle_relative"
    ):
        raise ValueError(
            "relative-flow conditions currently require oracle_relative; "
            "replace it with the camera/NDF relative estimator before deployment"
        )
    if any(fold < 0 or fold >= len(FOLDS) for fold in args.folds):
        raise ValueError(f"folds must lie in [0,{len(FOLDS) - 1}]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.dataset, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    action_frame_eef = bool(
        int(np.asarray(payload.get("action_frame_eef", 0)).reshape(-1)[0])
    )
    # Keep the untouched robot-coordinate labels for evaluating variants that
    # predict local actions without canonicalizing the complete trajectory.
    world_metric_payload = dict(payload)
    functional_frame_translation_mode = "absolute"
    if args.functional_frame_source == "oracle_relative":
        payload["target_frame9"] = oracle_relative_frame9(payload)
        functional_frame_translation_mode = "delta"
    elif args.functional_frame_source == "dataset_relative":
        if "target_frame9" not in payload:
            raise ValueError(
                "dataset_relative requires a precomputed target_frame9"
            )
        functional_frame_translation_mode = "delta"
    if any(
        condition
        in {
            "interaction_functional_equivariant_adapter_flow",
            "interaction_functional_local_equivariant_adapter_flow",
            "interaction_functional_hybrid_equivariant_adapter_flow",
            "interaction_functional_biframe_adapter_flow",
            "interaction_functional_gated_biframe_adapter_flow",
        }
        for condition in args.conditions
    ):
        if args.policy_frame != "world" or args.action_frame != "world":
            raise ValueError(
                "source-functional recovery predicts local residuals but applies "
                "them to the nominal world-action policy; both policy/action "
                "frames must therefore be world"
            )
        if "source_frame6_columns" not in payload:
            raise KeyError(
                "source-functional recovery requires dataset "
                "source_frame6_columns from the current NDF estimate"
            )
        source_frame6 = np.asarray(payload["source_frame6_columns"])
        if source_frame6.shape != (len(payload["shoe_id"]), 6):
            raise ValueError(
                "source_frame6_columns must have shape [samples,6], got "
                f"{source_frame6.shape}"
            )
    if "interaction_functional_dual_frame_flow" in args.conditions:
        if args.policy_frame != "world":
            raise ValueError(
                "dual-frame tokens retain base-frame robot control and require "
                "--policy-frame world"
            )
        if "goal_frame9" not in payload or "target_frame9" not in payload:
            raise KeyError(
                "dual-frame tokens require goal_frame9 and target_frame9"
            )
        payload["target_frame9_local"] = relative_frame9_to_target_frame(
            payload["target_frame9"], payload["goal_frame9"]
        )
    if args.policy_frame != "world":
        if args.functional_frame_source != "dataset_relative":
            raise ValueError(
                "goal-aligned policy frames require dataset_relative functional geometry"
            )
        if args.action_frame != "world":
            raise ValueError(
                "goal-aligned policy frames already transform actions; "
                "use --action-frame world"
            )
        if "goal_frame9" not in payload:
            raise KeyError("goal-aligned policy frame requires goal_frame9")
        if args.policy_frame in {"goal", "goal_geometry"}:
            frames = episode_frames_from_sample_frame9(payload, "goal_frame9")
            canonical = canonicalize(payload, frames)
            if args.policy_frame == "goal":
                payload = canonical
            else:
                for key in (
                    "points_a",
                    "points_b",
                    "current_anchors",
                    "current_object_pose9",
                    "episode_flow",
                    "episode_pose",
                    "target_frame9",
                    "goal_frame9",
                    "goal_translation_error_xyz_m",
                    "goal_rotation_error_rotvec",
                ):
                    if key in canonical:
                        payload[key] = canonical[key]
        else:
            payload["action"] = action14_to_target_frame(
                payload["action"], payload["goal_frame9"]
            )
            if args.policy_frame == "goal_state_action":
                payload["state"] = transform_eef_state20_to_target_frame(
                    payload["state"], payload["goal_frame9"]
                )
    if args.action_frame == "target":
        if args.functional_frame_source != "target_camera":
            raise ValueError(
                "target action frame requires target_camera functional frame"
            )
        if "target_frame9" not in payload:
            raise ValueError("target action frame requires target_frame9")
        payload["action"] = action14_to_target_frame(
            payload["action"], payload["target_frame9"]
        )
    metadata_path = args.dataset.with_suffix(".json")
    dataset_metadata = {}
    if metadata_path.is_file():
        dataset_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if args.source_axis_relation_adapter and "source_axis3" not in payload:
        raise ValueError("source-axis adapter requires dataset source_axis3")
    if args.target_axis_relation_adapter and "target_axis3" not in payload:
        raise ValueError("target-axis adapter requires dataset target_axis3")
    target_axis_fold = dataset_metadata.get("target_axis_fold")
    if target_axis_fold is not None and any(
        int(fold) != int(target_axis_fold) for fold in args.folds
    ):
        raise ValueError(
            f"target-axis dataset is fold {target_axis_fold}, requested {args.folds}"
        )
    shoe_ids = payload["shoe_id"]
    recovery_key = next(
        (
            key
            for key in ("is_recovery_augmented", "is_recovery_sample")
            if key in payload
        ),
        None,
    )
    recovery_mask = np.asarray(
        payload[recovery_key]
        if recovery_key is not None
        else np.zeros(len(shoe_ids), dtype=np.float32)
    ) >= 0.5
    if args.sample_mode != "all" and recovery_key is None:
        raise ValueError(
            f"--sample-mode={args.sample_mode} requires a recovery indicator"
        )
    sample_keep = np.ones(len(shoe_ids), dtype=bool)
    if args.sample_mode == "nominal":
        sample_keep = np.asarray(payload[recovery_key]) < 0.5
    elif args.sample_mode == "recovery":
        sample_keep = np.asarray(payload[recovery_key]) >= 0.5
    sample_keep &= policy_phase_keep_mask(payload, args.policy_phase)
    horizon = int(payload["action"].shape[1])
    point_channels = int(payload["points_a"].shape[-1]) + (
        3 if args.functional_point_coordinates else 0
    )
    device = torch.device(args.device)
    all_results = []

    for fold_index in args.folds:
        fold = (
            explicit_object_split
            if explicit_object_split is not None
            else FOLDS[int(fold_index)]
        )
        masks = {
            split: np.flatnonzero(np.isin(shoe_ids, ids) & sample_keep)
            for split, ids in fold.items()
        }
        recovery_validation_indices = np.empty((0,), dtype=np.int64)
        if recovery_key is not None and args.sample_mode == "all":
            recovery_validation_indices = masks["validation"][
                recovery_mask[masks["validation"]]
            ]
            for split in ("validation", "test"):
                masks[split] = masks[split][~recovery_mask[masks[split]]]
        if args.recovery_validation_weight is not None and not len(
            recovery_validation_indices
        ):
            raise RuntimeError(
                "recovery validation objective requires recovery rows on the "
                "fold validation objects"
            )
        low_confidence_recovery_train_samples_dropped = 0
        if args.minimum_recovery_train_target_confidence is not None:
            if recovery_key is None:
                raise ValueError(
                    "recovery confidence filtering requires a recovery indicator"
                )
            if "target_frame_confidence" not in payload:
                raise KeyError(
                    "recovery confidence filtering requires "
                    "target_frame_confidence"
                )
            train_indices = masks["train"]
            train_recovery = recovery_mask[train_indices]
            train_confidence = np.asarray(
                payload["target_frame_confidence"], dtype=np.float32
            )[train_indices]
            low_confidence_recovery = train_recovery & (
                train_confidence
                < float(args.minimum_recovery_train_target_confidence)
            )
            low_confidence_recovery_train_samples_dropped = int(
                low_confidence_recovery.sum()
            )
            masks["train"] = train_indices[~low_confidence_recovery]
        if args.max_train_samples is not None and len(masks["train"]) > args.max_train_samples:
            generator = np.random.default_rng(int(args.seed) + 1000 * int(fold_index))
            masks["train"] = np.sort(
                generator.choice(
                    masks["train"], int(args.max_train_samples), replace=False
                )
            )
        nominal_train = masks["train"][~recovery_mask[masks["train"]]]
        recovery_train = masks["train"][recovery_mask[masks["train"]]]
        normalization_indices = (
            nominal_train
            if recovery_key is not None
            and args.sample_mode == "all"
            and len(nominal_train)
            else masks["train"]
        )
        normalization = fit_normalization(payload, normalization_indices)
        datasets = {
            split: InteractionFlowDataset(
                payload,
                indices,
                normalization,
                flow_steps=args.flow_steps,
                target_horizon_mode=args.target_horizon_mode,
                target_frame_mode=args.target_frame_mode,
                functional_frame_translation_mode=(
                    functional_frame_translation_mode
                ),
                functional_point_coordinates=args.functional_point_coordinates,
                zero_global_functional_frame_token=(
                    args.zero_global_functional_frame_token
                ),
            )
            for split, indices in masks.items()
        }
        if len(recovery_validation_indices):
            datasets["validation_recovery"] = InteractionFlowDataset(
                payload,
                recovery_validation_indices,
                normalization,
                flow_steps=args.flow_steps,
                target_horizon_mode=args.target_horizon_mode,
                target_frame_mode=args.target_frame_mode,
                functional_frame_translation_mode=(
                    functional_frame_translation_mode
                ),
                functional_point_coordinates=args.functional_point_coordinates,
                zero_global_functional_frame_token=(
                    args.zero_global_functional_frame_token
                ),
            )
        train_sampler = None
        if args.episode_balanced_sampling or args.recovery_target_fraction is not None:
            train_episode = np.asarray(
                payload["episode_index"][masks["train"]], dtype=np.int64
            )
            weights_np = np.ones(len(train_episode), dtype=np.float64)
            if args.episode_balanced_sampling:
                unique_episode, episode_count = np.unique(
                    train_episode, return_counts=True
                )
                inverse_count = {
                    int(episode): 1.0 / int(count)
                    for episode, count in zip(unique_episode, episode_count)
                }
                weights_np = np.asarray(
                    [inverse_count[int(episode)] for episode in train_episode],
                    dtype=np.float64,
                )
            if args.recovery_target_fraction is not None:
                if recovery_key is None or not len(nominal_train) or not len(recovery_train):
                    raise RuntimeError(
                        "recovery sampling requires nominal and recovery training rows"
                    )
                train_recovery = recovery_mask[masks["train"]]
                nominal_mass = float(weights_np[~train_recovery].sum())
                recovery_mass = float(weights_np[train_recovery].sum())
                target_fraction = float(args.recovery_target_fraction)
                weights_np[train_recovery] *= (
                    target_fraction
                    * nominal_mass
                    / ((1.0 - target_fraction) * recovery_mass)
                )
            weights = torch.as_tensor(weights_np, dtype=torch.double)
            train_sampler = WeightedRandomSampler(
                weights,
                num_samples=len(weights),
                replacement=True,
                generator=torch.Generator().manual_seed(
                    int(args.seed) + int(fold_index) * 100 + 37_719
                ),
            )
        loaders = {
            split: DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=split == "train" and train_sampler is None,
                sampler=train_sampler if split == "train" else None,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
                persistent_workers=args.num_workers > 0,
                generator=(
                    torch.Generator().manual_seed(0)
                    if split == "train"
                    else None
                ),
            )
            for split, dataset in datasets.items()
        }

        for condition in args.conditions:
            run_seed = int(args.seed) + int(fold_index) * 100
            seed_everything(run_seed)
            loaders["train"].generator.manual_seed(run_seed + 48127)
            model = build_model(
                condition,
                horizon=horizon,
                flow_steps=args.flow_steps,
                num_anchors=args.num_anchors,
                feature_dim=args.feature_dim,
                layers=args.layers,
                flow_parameterization=args.flow_parameterization,
                xyz_std=normalization.xyz_std,
                action_std=normalization.action_std,
                predict_rotation=args.rotation_loss_weight > 0.0,
                point_channels=point_channels,
                target_color_adapter=args.target_color_adapter,
                source_geometry_adapter=args.source_geometry_adapter,
                shared_geometry_adapter=args.shared_geometry_adapter,
                functional_coordinate_adapter=args.functional_coordinate_adapter,
                source_axis_relation_adapter=args.source_axis_relation_adapter,
                source_axis_gate_initial_value=args.source_axis_gate_initial_value,
                target_axis_relation_adapter=args.target_axis_relation_adapter,
                diffusion_steps=args.diffusion_steps,
                diffusion_inference_steps=args.diffusion_inference_steps,
                diffusion_prediction_type=args.diffusion_prediction_type,
            ).to(device)
            model.experiment_condition = condition
            initialization_condition = None
            if args.initialize_from_checkpoint is not None:
                initialization = torch.load(
                    args.initialize_from_checkpoint,
                    map_location=device,
                    weights_only=False,
                )
                initialization_condition = str(
                    initialization.get("condition", "unknown")
                )
                incompatible = model.load_state_dict(
                    initialization["model"], strict=False
                )
                allowed_missing_prefix = (
                    "functional_frame_",
                    "functional_relative_",
                    "target_axis_relation_projection.",
                    "recovery_action_adapter.",
                    "hand_object_frame_projection.",
                    "recovery_local_frame_projection.",
                    "recovery_local_frame_gate",
                    "action_std",
                )
                disallowed_missing = [
                    name
                    for name in incompatible.missing_keys
                    if not name.startswith(allowed_missing_prefix)
                ]
                if disallowed_missing or incompatible.unexpected_keys:
                    raise RuntimeError(
                        "initialization checkpoint/model mismatch: "
                        f"missing={disallowed_missing}, "
                        f"unexpected={incompatible.unexpected_keys}"
                    )
                if args.initialize_recovery_adapter_from_decoder:
                    initialize_recovery_adapter_from_decoder(model)
            if args.train_geometry_adapter_only:
                trainable = 0
                for name, parameter in model.named_parameters():
                    parameter.requires_grad_(
                        name.startswith("functional_frame_")
                        or name.startswith("functional_relative_")
                        or name.startswith("source_geometry_projection.")
                        or name.startswith("functional_coordinate_projection.")
                        or name.startswith("source_axis_relation_projection.")
                    )
                    if parameter.requires_grad:
                        trainable += parameter.numel()
                if trainable == 0:
                    raise RuntimeError(
                        "geometry-adapter-only run has no trainable adapter parameters"
                    )
            elif args.train_hand_object_adapter_only:
                trainable = 0
                for name, parameter in model.named_parameters():
                    parameter.requires_grad_(
                        name.startswith("hand_object_frame_projection.")
                    )
                    if parameter.requires_grad:
                        trainable += parameter.numel()
                if trainable == 0:
                    raise RuntimeError(
                        "hand-object-adapter-only run has no trainable parameters"
                    )
            elif args.train_recovery_action_adapter_only:
                trainable = 0
                for name, parameter in model.named_parameters():
                    parameter.requires_grad_(
                        name.startswith("recovery_action_adapter.")
                        or name.startswith("recovery_local_frame_projection.")
                        or name.startswith("recovery_local_frame_gate")
                        or (
                            condition
                            == "interaction_functional_grasp_adapter_flow"
                            and name.startswith("hand_object_frame_projection.")
                        )
                    )
                    if parameter.requires_grad:
                        trainable += parameter.numel()
                if trainable == 0:
                    raise RuntimeError(
                        "recovery-action-adapter-only run has no trainable parameters"
                    )
            elif args.train_target_axis_adapter_only:
                trainable = 0
                for name, parameter in model.named_parameters():
                    parameter.requires_grad_(
                        name.startswith("target_axis_relation_projection.")
                    )
                    if parameter.requires_grad:
                        trainable += parameter.numel()
                if trainable == 0:
                    raise RuntimeError(
                        "target-axis-adapter-only run has no trainable parameters"
                    )
            parameter_count = sum(value.numel() for value in model.parameters())
            trainable_parameter_count = sum(
                value.numel() for value in model.parameters() if value.requires_grad
            )
            optimizer = torch.optim.AdamW(
                [value for value in model.parameters() if value.requires_grad],
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
            best_state = None
            best_validation = float("inf")
            best_epoch = -1
            stale = 0
            history = []
            nominal_validation_reference = None
            nominal_validation_limit = None
            if args.initialize_from_checkpoint is not None:
                initial_validation = validate(
                    model,
                    loaders["validation"],
                    device,
                    condition,
                    rotation_action_loss_weight=(
                        args.rotation_action_loss_weight
                    ),
                    endpoint_rotation_action_loss_weight=(
                        args.endpoint_rotation_action_loss_weight
                    ),
                    action_mean=normalization.action_mean,
                    action_std=normalization.action_std,
                    action_frame_eef=action_frame_eef,
                )
                initial_recovery_validation = (
                    validate(
                        model,
                        loaders["validation_recovery"],
                        device,
                        condition,
                        rotation_action_loss_weight=(
                            args.rotation_action_loss_weight
                        ),
                        endpoint_rotation_action_loss_weight=(
                            args.endpoint_rotation_action_loss_weight
                        ),
                        action_mean=normalization.action_mean,
                        action_std=normalization.action_std,
                        action_frame_eef=action_frame_eef,
                    )
                    if args.recovery_validation_weight is not None
                    else None
                )
                initial_nominal_objective = validation_action_objective(
                    initial_validation,
                    args.endpoint_rotation_action_loss_weight,
                )
                initial_recovery_objective = (
                    None
                    if initial_recovery_validation is None
                    else validation_action_objective(
                        initial_recovery_validation,
                        args.endpoint_rotation_action_loss_weight,
                    )
                )
                best_validation = initial_nominal_objective + (
                    0.0
                    if initial_recovery_objective is None
                    else float(args.recovery_validation_weight)
                    * initial_recovery_objective
                )
                nominal_validation_reference = initial_nominal_objective
                nominal_validation_limit = (
                    None
                    if args.maximum_nominal_validation_relative_degradation is None
                    else nominal_validation_reference
                    * (
                        1.0
                        + float(
                            args.maximum_nominal_validation_relative_degradation
                        )
                    )
                )
                best_state = copy.deepcopy(model.state_dict())
                history.append(
                    {
                        "epoch": -1,
                        "train_action": None,
                        "train_flow": None,
                        "train_rigidity": None,
                        "train_rotation_rad": None,
                        "train_endpoint_rotation_rad": None,
                        "validation": initial_validation,
                        "recovery_validation": initial_recovery_validation,
                        "validation_objective": best_validation,
                        "nominal_validation_eligible": True,
                    }
                )
                print(
                    json.dumps(
                        {
                            "fold": int(fold_index),
                            "condition": condition,
                            "epoch": -1,
                            "validation_action": best_validation,
                            "nominal_validation_action": float(
                                initial_validation["action"]
                            ),
                            "nominal_validation_endpoint_rotation_rad": (
                                initial_validation["endpoint_rotation_rad"]
                            ),
                            "recovery_validation_action": (
                                None
                                if initial_recovery_validation is None
                                else float(initial_recovery_validation["action"])
                            ),
                            "nominal_validation_limit": nominal_validation_limit,
                            "best_validation": best_validation,
                            "trainable_parameters": int(
                                trainable_parameter_count
                            ),
                        }
                    ),
                    flush=True,
                )
            for epoch in range(int(args.epochs)):
                model.train()
                train_action_total = 0.0
                train_flow_total = 0.0
                train_rigid_total = 0.0
                train_rotation_total = 0.0
                train_endpoint_rotation_total = 0.0
                train_endpoint_rotation_count = 0
                train_count = 0
                for batch in loaders["train"]:
                    batch = move_batch(batch, device)
                    optimizer.zero_grad(set_to_none=True)
                    action_target = batch["action"]
                    if condition_uses_diffusion(condition):
                        noisy, timestep, action_target = (
                            model.decoder.training_inputs(batch["action"])
                        )
                        batch["diffusion_noisy_action"] = noisy
                        batch["diffusion_timestep"] = timestep
                    output = model(batch)
                    if args.nominal_adapter_distillation_weight is not None:
                        if "is_recovery" not in batch:
                            raise RuntimeError(
                                "nominal adapter distillation requires recovery labels"
                            )
                        if output.recovery_action_residual is None:
                            raise RuntimeError(
                                "adapter distillation requires an action residual"
                            )
                        recovery_rows = batch["is_recovery"] >= 0.5
                        nominal_rows = ~recovery_rows
                        if torch.any(recovery_rows):
                            recovery_action_loss = weighted_action_loss(
                                output.action[recovery_rows],
                                action_target[recovery_rows],
                                batch["active_arm_right"][recovery_rows],
                                batch["relation_phase"][recovery_rows],
                                active_arm_weight=args.active_arm_loss_weight,
                                relation_weight=args.relation_loss_weight,
                                rotation_channel_weight=(
                                    args.rotation_action_loss_weight
                                ),
                            )
                        else:
                            recovery_action_loss = output.action.sum() * 0.0
                        if torch.any(nominal_rows):
                            nominal_distillation = output.recovery_action_residual[
                                nominal_rows
                            ].square().mean()
                        else:
                            nominal_distillation = output.action.sum() * 0.0
                        action_loss = recovery_action_loss + float(
                            args.nominal_adapter_distillation_weight
                        ) * nominal_distillation
                    else:
                        action_loss = weighted_action_loss(
                            output.action,
                            action_target,
                            batch["active_arm_right"],
                            batch["relation_phase"],
                            active_arm_weight=args.active_arm_loss_weight,
                            relation_weight=args.relation_loss_weight,
                            rotation_channel_weight=(
                                args.rotation_action_loss_weight
                            ),
                        )
                    endpoint_rotation_action_loss = torch.zeros((), device=device)
                    if float(args.endpoint_rotation_action_loss_weight) > 0.0:
                        endpoint_sample_mask = (
                            recovery_rows
                            if args.nominal_adapter_distillation_weight is not None
                            else None
                        )
                        endpoint_rotation_action_loss = (
                            active_endpoint_rotation_geodesic_loss(
                                output.action,
                                batch["action"],
                                batch["active_arm_right"],
                                normalization.action_mean,
                                normalization.action_std,
                                action_frame_eef=action_frame_eef,
                                sample_mask=endpoint_sample_mask,
                            )
                        )
                    loss = action_loss + float(
                        args.endpoint_rotation_action_loss_weight
                    ) * endpoint_rotation_action_loss
                    geometry_loss = torch.zeros((), device=device)
                    rigidity = torch.zeros((), device=device)
                    rotation_loss = torch.zeros((), device=device)
                    if condition_uses_flow_loss(condition):
                        if output.predicted_flow is None:
                            raise RuntimeError(
                                "flow-supervised condition did not predict a flow"
                            )
                        geometry_prediction = output.predicted_flow
                        geometry_target = condition_flow_target(condition, batch)
                        if args.geometry_phase == "relation":
                            geometry_keep = batch["relation_phase"] >= 0.5
                            if torch.any(geometry_keep):
                                geometry_prediction = geometry_prediction[geometry_keep]
                                geometry_target = geometry_target[geometry_keep]
                            else:
                                geometry_prediction = geometry_prediction[:0]
                                geometry_target = geometry_target[:0]
                        if len(geometry_prediction) == 0:
                            geometry_loss = output.predicted_flow.sum() * 0.0
                            rigidity = geometry_loss
                        else:
                            geometry_loss, rigidity = flow_losses(
                                geometry_prediction,
                                geometry_target,
                                args.flow_loss_mode,
                                args.motion_loss_scale,
                            )
                        rigidity_weight = (
                            0.0
                            if args.flow_parameterization == "rigid_se3"
                            else float(args.rigidity_loss_weight)
                        )
                        loss = (
                            loss
                            + float(args.geometry_loss_weight) * geometry_loss
                            + rigidity_weight * rigidity
                        )
                        if float(args.rotation_loss_weight) > 0.0:
                            if output.predicted_rotation is None:
                                raise RuntimeError(
                                    "rotation-supervised run did not predict a rotation"
                                )
                            rotation_prediction = output.predicted_rotation
                            rotation_target = condition_rotation_target(
                                condition, batch
                            )
                            if args.geometry_phase == "relation":
                                rotation_prediction = rotation_prediction[
                                    geometry_keep
                                ]
                                rotation_target = rotation_target[geometry_keep]
                            if len(rotation_prediction) == 0:
                                rotation_loss = (
                                    output.predicted_rotation.sum() * 0.0
                                )
                            else:
                                rotation_loss = rotation_geodesic_loss(
                                    rotation_prediction, rotation_target
                                )
                            loss = loss + float(
                                args.rotation_loss_weight
                            ) * rotation_loss
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    batch_count = int(len(batch["action"]))
                    train_action_total += float(action_loss.item()) * batch_count
                    train_flow_total += float(geometry_loss.item()) * batch_count
                    train_rigid_total += float(rigidity.item()) * batch_count
                    train_rotation_total += (
                        float(rotation_loss.item()) * batch_count
                    )
                    endpoint_batch_count = (
                        int(recovery_rows.sum().item())
                        if args.nominal_adapter_distillation_weight is not None
                        else batch_count
                    )
                    train_endpoint_rotation_total += (
                        float(endpoint_rotation_action_loss.item())
                        * endpoint_batch_count
                    )
                    train_endpoint_rotation_count += endpoint_batch_count
                    train_count += batch_count
                validation = validate(
                    model,
                    loaders["validation"],
                    device,
                    condition,
                    rotation_action_loss_weight=(
                        args.rotation_action_loss_weight
                    ),
                    endpoint_rotation_action_loss_weight=(
                        args.endpoint_rotation_action_loss_weight
                    ),
                    action_mean=normalization.action_mean,
                    action_std=normalization.action_std,
                    action_frame_eef=action_frame_eef,
                )
                recovery_validation = (
                    validate(
                        model,
                        loaders["validation_recovery"],
                        device,
                        condition,
                        rotation_action_loss_weight=(
                            args.rotation_action_loss_weight
                        ),
                        endpoint_rotation_action_loss_weight=(
                            args.endpoint_rotation_action_loss_weight
                        ),
                        action_mean=normalization.action_mean,
                        action_std=normalization.action_std,
                        action_frame_eef=action_frame_eef,
                    )
                    if args.recovery_validation_weight is not None
                    else None
                )
                row = {
                    "epoch": int(epoch),
                    "train_action": train_action_total / max(train_count, 1),
                    "train_flow": train_flow_total / max(train_count, 1),
                    "train_rigidity": train_rigid_total / max(train_count, 1),
                    "train_rotation_rad": (
                        train_rotation_total / max(train_count, 1)
                    ),
                    "train_endpoint_rotation_rad": (
                        train_endpoint_rotation_total
                        / max(train_endpoint_rotation_count, 1)
                    ),
                    "validation": validation,
                    "recovery_validation": recovery_validation,
                }
                history.append(row)
                nominal_validation_objective = validation_action_objective(
                    validation,
                    args.endpoint_rotation_action_loss_weight,
                )
                recovery_validation_objective = (
                    None
                    if recovery_validation is None
                    else validation_action_objective(
                        recovery_validation,
                        args.endpoint_rotation_action_loss_weight,
                    )
                )
                validation_action = nominal_validation_objective + (
                    0.0
                    if recovery_validation_objective is None
                    else float(args.recovery_validation_weight)
                    * recovery_validation_objective
                )
                row["validation_objective"] = validation_action
                nominal_validation_eligible = bool(
                    nominal_validation_limit is None
                    or nominal_validation_objective <= nominal_validation_limit
                )
                row["nominal_validation_eligible"] = nominal_validation_eligible
                if (
                    nominal_validation_eligible
                    and validation_action < best_validation - 1e-5
                ):
                    best_validation = validation_action
                    best_epoch = epoch
                    best_state = copy.deepcopy(model.state_dict())
                    stale = 0
                else:
                    stale += 1
                if epoch % 10 == 0 or stale == 0:
                    print(
                        json.dumps(
                            {
                                "fold": int(fold_index),
                                "condition": condition,
                                "epoch": int(epoch),
                                "train_action": row["train_action"],
                                "train_flow": row["train_flow"],
                                "train_rotation_rad": row["train_rotation_rad"],
                                "train_endpoint_rotation_rad": row[
                                    "train_endpoint_rotation_rad"
                                ],
                                "validation_action": validation_action,
                                "nominal_validation_action": float(
                                    validation["action"]
                                ),
                                "nominal_validation_endpoint_rotation_rad": (
                                    validation["endpoint_rotation_rad"]
                                ),
                                "recovery_validation_action": (
                                    None
                                    if recovery_validation is None
                                    else float(recovery_validation["action"])
                                ),
                                "nominal_validation_eligible": (
                                    nominal_validation_eligible
                                ),
                                "nominal_validation_limit": (
                                    nominal_validation_limit
                                ),
                                "best_validation": best_validation,
                            }
                        ),
                        flush=True,
                    )
                if stale >= int(args.patience):
                    break
            if best_state is None:
                raise RuntimeError("training did not produce a checkpoint")
            model.load_state_dict(best_state)
            test_indices, prediction, geometry_diagnostic = predict(
                model,
                loaders["test"],
                device,
                normalization,
                action_frame=args.action_frame,
            )
            order = np.argsort(test_indices)
            test_indices = test_indices[order]
            prediction = prediction[order]
            metric_payload = payload
            if args.policy_frame in {"goal_action", "goal_state_action"}:
                prediction = action14_from_target_frame(
                    prediction,
                    world_metric_payload["goal_frame9"][test_indices],
                )
                metric_payload = world_metric_payload
            per_shoe = {
                str(int(shoe)): summarize_errors(
                    metric_payload,
                    test_indices[
                        metric_payload["shoe_id"][test_indices] == shoe
                    ],
                    prediction[
                        metric_payload["shoe_id"][test_indices] == shoe
                    ],
                )
                for shoe in sorted(
                    np.unique(metric_payload["shoe_id"][test_indices]).tolist()
                )
            }
            result = {
                "condition": condition,
                "fold": int(fold_index),
                "seed": int(run_seed),
                "base_seed": int(args.seed),
                "parameters": int(parameter_count),
                "trainable_parameters": int(trainable_parameter_count),
                "split": {key: list(value) for key, value in fold.items()},
                "train_samples": int(len(masks["train"])),
                "recovery_train_samples": int(len(recovery_train)),
                "low_confidence_recovery_train_samples_dropped": int(
                    low_confidence_recovery_train_samples_dropped
                ),
                "best_epoch": int(best_epoch),
                "best_validation_normalized_mse": float(best_validation),
                "best_validation_objective_value": float(best_validation),
                "recovery_validation_samples": int(
                    len(recovery_validation_indices)
                ),
                "recovery_validation_weight": (
                    None
                    if args.recovery_validation_weight is None
                    else float(args.recovery_validation_weight)
                ),
                "nominal_validation_reference": nominal_validation_reference,
                "nominal_validation_limit": nominal_validation_limit,
                "best_validation_objective": (
                    "nominal_plus_weighted_recovery_action_and_endpoint_so3"
                    if args.recovery_validation_weight is not None
                    and args.endpoint_rotation_action_loss_weight > 0.0
                    else "action_and_endpoint_so3"
                    if args.endpoint_rotation_action_loss_weight > 0.0
                    else "nominal_plus_weighted_recovery_normalized_action_mse"
                    if args.recovery_validation_weight is not None
                    else f"fixed_noise_{args.diffusion_prediction_type}_mse"
                    if condition_uses_diffusion(condition)
                    else "normalized_action_mse"
                ),
                "summary": summarize_phases(
                    metric_payload, test_indices, prediction
                ),
                "geometry_diagnostic": geometry_diagnostic,
                "per_shoe": per_shoe,
                "history": history,
                "config": dict(vars(args)),
                "initialization_condition": initialization_condition,
            }
            # pathlib objects are not JSON serializable.
            result["config"]["dataset"] = str(args.dataset)
            result["config"]["output_dir"] = str(args.output_dir)
            if args.initialize_from_checkpoint is not None:
                result["config"]["initialize_from_checkpoint"] = str(
                    args.initialize_from_checkpoint
                )
            result["config"]["point_channels"] = int(point_channels)
            source_geometry_gate = None
            adapter = getattr(model, "source_geometry_projection", None)
            if adapter is not None:
                source_geometry_gate = float(
                    torch.tanh(adapter.gate.detach()).cpu()
                )
            result["source_geometry_gate"] = source_geometry_gate
            functional_coordinate_gate = None
            coordinate_adapter = getattr(
                model, "functional_coordinate_projection", None
            )
            if coordinate_adapter is not None:
                functional_coordinate_gate = float(
                    torch.tanh(coordinate_adapter.gate.detach()).cpu()
                )
            result["functional_coordinate_gate"] = functional_coordinate_gate
            source_axis_gate = None
            axis_adapter = getattr(model, "source_axis_relation_projection", None)
            if axis_adapter is not None:
                source_axis_gate = float(
                    torch.tanh(axis_adapter.gate.detach()).cpu()
                )
            result["source_axis_relation_gate"] = source_axis_gate
            target_axis_gate = None
            target_axis_adapter = getattr(
                model, "target_axis_relation_projection", None
            )
            if target_axis_adapter is not None:
                target_axis_gate = float(
                    torch.tanh(target_axis_adapter.gate.detach()).cpu()
                )
            result["target_axis_relation_gate"] = target_axis_gate
            functional_frame_gate = getattr(
                model, "functional_frame_gate", None
            )
            result["functional_frame_gate"] = (
                None
                if functional_frame_gate is None
                else float(torch.tanh(functional_frame_gate.detach()).cpu())
            )
            functional_relative_gate = getattr(
                model, "functional_relative_gate", None
            )
            result["functional_relative_gate"] = (
                None
                if functional_relative_gate is None
                else float(torch.tanh(functional_relative_gate.detach()).cpu())
            )
            dual_frame_gate = getattr(
                model, "dual_functional_frame_gate", None
            )
            result["dual_functional_frame_gate"] = (
                None
                if dual_frame_gate is None
                else float(torch.tanh(dual_frame_gate.detach()).cpu())
            )
            result["config"]["target_color_priority"] = bool(
                dataset_metadata.get("target_color_priority", False)
            )
            result["config"]["cache_target_observation"] = bool(
                dataset_metadata.get("cache_target_observation", False)
            )
            result_path = args.output_dir / (
                f"fold{fold_index}_{condition}_seed{args.seed}.json"
            )
            result_path.write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8"
            )
            torch.save(
                {
                    "model": model.state_dict(),
                    "condition": condition,
                    "horizon": horizon,
                    "flow_steps": int(args.flow_steps),
                    "num_anchors": int(args.num_anchors),
                    "feature_dim": int(args.feature_dim),
                    "layers": int(args.layers),
                    "point_channels": int(point_channels),
                    "target_color_adapter": bool(args.target_color_adapter),
                    "source_geometry_adapter": bool(args.source_geometry_adapter),
                    "shared_geometry_adapter": bool(args.shared_geometry_adapter),
                    "functional_coordinate_adapter": bool(
                        args.functional_coordinate_adapter
                    ),
                    "source_axis_relation_adapter": bool(
                        args.source_axis_relation_adapter
                    ),
                    "source_axis_gate_initial_value": float(
                        args.source_axis_gate_initial_value
                    ),
                    "target_axis_relation_adapter": bool(
                        args.target_axis_relation_adapter
                    ),
                    "hand_object_frame_token": bool(
                        condition == "interaction_functional_grasp_adapter_flow"
                    ),
                    "hand_object_translation_scale_m": float(
                        HAND_OBJECT_TRANSLATION_SCALE_M
                    ),
                    "target_axis_marker_local3": dataset_metadata.get(
                        "target_axis_marker_local3"
                    ),
                    "axis_relation_encoding": (
                        "source_target_axis_delta_v1"
                        if args.target_axis_relation_adapter
                        else "source_axis_delta_v1"
                        if args.source_axis_relation_adapter
                        else None
                    ),
                    "target_color_priority": bool(
                        dataset_metadata.get("target_color_priority", False)
                    ),
                    "cache_target_observation": bool(
                        dataset_metadata.get("cache_target_observation", False)
                    ),
                    "flow_loss_mode": str(args.flow_loss_mode),
                    "flow_parameterization": str(args.flow_parameterization),
                    "target_horizon_mode": str(args.target_horizon_mode),
                    "geometry_phase": str(args.geometry_phase),
                    "sample_mode": str(args.sample_mode),
                    "policy_phase": str(args.policy_phase),
                    "recovery_target_fraction": (
                        None
                        if args.recovery_target_fraction is None
                        else float(args.recovery_target_fraction)
                    ),
                    "normalization_uses_nominal_training_only": bool(
                        recovery_key is not None
                        and args.sample_mode == "all"
                        and len(nominal_train)
                    ),
                    "motion_loss_scale": float(args.motion_loss_scale),
                    "predict_rotation": bool(args.rotation_loss_weight > 0.0),
                    "rotation_loss_weight": float(args.rotation_loss_weight),
                    "rotation_action_loss_weight": float(
                        args.rotation_action_loss_weight
                    ),
                    "endpoint_rotation_action_loss_weight": float(
                        args.endpoint_rotation_action_loss_weight
                    ),
                    "action_decoder_type": (
                        "diffusion"
                        if condition_uses_diffusion(condition)
                        else "regression"
                    ),
                    "diffusion_steps": int(args.diffusion_steps),
                    "diffusion_inference_steps": int(
                        args.diffusion_inference_steps
                    ),
                    "diffusion_prediction_type": str(
                        args.diffusion_prediction_type
                    ),
                    "action_frame": str(args.action_frame),
                    "recovery_action_frame": (
                        "hybrid_source_rotation"
                        if condition
                        == "interaction_functional_hybrid_equivariant_adapter_flow"
                        else "source_functional"
                        if condition in {
                            "interaction_functional_equivariant_adapter_flow",
                            "interaction_functional_local_equivariant_adapter_flow",
                        }
                        else "world"
                    ),
                    "recovery_local_frame_token": bool(
                        condition
                        in {
                            "interaction_functional_local_equivariant_adapter_flow",
                            "interaction_functional_hybrid_equivariant_adapter_flow",
                            "interaction_functional_biframe_adapter_flow",
                            "interaction_functional_gated_biframe_adapter_flow",
                        }
                    ),
                    "recovery_local_frame_gated_fusion": bool(
                        condition
                        == "interaction_functional_gated_biframe_adapter_flow"
                    ),
                    "policy_frame": str(args.policy_frame),
                    "functional_frame_source": str(
                        args.functional_frame_source
                    ),
                    "functional_frame_translation_mode": str(
                        functional_frame_translation_mode
                    ),
                    "functional_point_coordinates": bool(
                        args.functional_point_coordinates
                    ),
                    "zero_global_functional_frame_token": bool(
                        args.zero_global_functional_frame_token
                    ),
                    "functional_frame_encoding": (
                        "intended_goal_error_se3_columns_v3"
                        if args.functional_frame_source == "oracle_relative"
                        else dataset_metadata.get(
                            "functional_frame_encoding",
                            "dataset_relative_se3_columns_v1",
                        )
                        if args.functional_frame_source == "dataset_relative"
                        else "target_frame9_columns_v1"
                    ),
                    "normalization": normalization.__dict__,
                    "parameters": int(parameter_count),
                    "trainable_parameters": int(trainable_parameter_count),
                    "initialization_condition": initialization_condition,
                },
                args.output_dir / f"fold{fold_index}_{condition}_seed{args.seed}.pt",
            )
            all_results.append(result)
            print(
                json.dumps(
                    {
                        "condition": condition,
                        "fold": int(fold_index),
                        "parameters": int(parameter_count),
                        "best_epoch": int(best_epoch),
                        "summary": result["summary"],
                        "geometry_diagnostic": geometry_diagnostic,
                    }
                ),
                flush=True,
            )
    summary_path = args.output_dir / f"summary_seed{args.seed}.json"
    summary_path.write_text(
        json.dumps(all_results, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(summary_path), "runs": len(all_results)}))


if __name__ == "__main__":
    main()
