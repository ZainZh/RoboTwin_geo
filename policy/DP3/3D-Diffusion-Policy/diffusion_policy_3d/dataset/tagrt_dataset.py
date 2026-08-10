"""Row-wise TAGRT action-chunk dataset for a DP3 backbone.

The GeometryFlow multi-goal archives already contain one camera observation,
one task-aligned relation, and one six-step Cartesian action chunk per row.
This dataset preserves that supervision instead of pretending the independent
counterfactual rows are consecutive simulator frames.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch

from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)


def _rotation_from_columns(columns6: np.ndarray) -> np.ndarray:
    value = np.asarray(columns6, dtype=np.float64).reshape(6)
    first = value[:3]
    first /= max(float(np.linalg.norm(first)), 1e-8)
    second = value[3:] - first * float(np.dot(first, value[3:]))
    second /= max(float(np.linalg.norm(second)), 1e-8)
    third = np.cross(first, second)
    return np.stack((first, second, third), axis=1)


def task_aligned_local_coordinates(
    points_a: np.ndarray,
    points_b: np.ndarray,
    goal_frame9: np.ndarray,
    relative_frame9: np.ndarray,
    *,
    scale_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Match TAGRT-v1 local functional coordinates without simulator poses."""

    goal = np.asarray(goal_frame9, dtype=np.float64).reshape(9)
    relative = np.asarray(relative_frame9, dtype=np.float64).reshape(9)
    goal_rotation = _rotation_from_columns(goal[3:])
    relative_rotation = _rotation_from_columns(relative[3:])
    source_rotation = relative_rotation.T @ goal_rotation
    source_position = goal[:3] - relative[:3]
    source_local = (
        np.asarray(points_a, dtype=np.float64)[:, :3] - source_position
    ) @ source_rotation
    target_local = (
        np.asarray(points_b, dtype=np.float64)[:, :3] - goal[:3]
    ) @ goal_rotation
    return (
        (source_local / float(scale_m)).astype(np.float32),
        (target_local / float(scale_m)).astype(np.float32),
    )


def task_aligned_anchor_flow_tokens(
    points_a: np.ndarray,
    state20: np.ndarray,
    goal_frame9: np.ndarray,
    relative_frame9: np.ndarray,
    *,
    scale_m: float,
) -> np.ndarray:
    """Build per-anchor action-flow tokens without simulator object poses.

    Each token contains the current anchor relative to both end effectors and
    the remaining rigid displacement of that anchor.  Both EEF positions come
    from ordinary proprioception; the learned policy still decides which arm
    acts from its gripper state.
    """

    goal = np.asarray(goal_frame9, dtype=np.float64).reshape(9)
    relative = np.asarray(relative_frame9, dtype=np.float64).reshape(9)
    goal_rotation = _rotation_from_columns(goal[3:])
    relative_rotation = _rotation_from_columns(relative[3:])
    source_rotation = relative_rotation.T @ goal_rotation
    source_position = goal[:3] - relative[:3]
    anchors = np.asarray(points_a, dtype=np.float64)[:, :3]
    anchor_local = (anchors - source_position) @ source_rotation
    desired_anchors = anchor_local @ goal_rotation.T + goal[:3]
    flow = desired_anchors - anchors
    state = np.asarray(state20, dtype=np.float64).reshape(20)
    left_offset = anchors - state[:3]
    right_offset = anchors - state[10:13]
    return (
        np.concatenate((left_offset, right_offset, flow), axis=-1)
        / float(scale_m)
    ).astype(np.float32)


def _ids(values: Iterable[int]) -> np.ndarray:
    return np.asarray(sorted({int(value) for value in values}), dtype=np.int64)


def _episode_level_shuffle(
    indices: np.ndarray, payload: dict[str, np.ndarray]
) -> np.ndarray:
    """Pair every row with a same-phase row from a different episode.

    A one-row roll is not a valid intervention for sequential archives because
    the donor is normally the adjacent frame from the same trajectory.  This
    mapping preserves the coarse pre/post-grasp phase while changing the task
    goal, and matches rows by normalized progress within the donor episode.
    """

    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) < 2:
        return indices.copy()
    if "episode_id" not in payload:
        return np.roll(indices, max(1, len(indices) // 2))
    episodes = np.asarray(payload["episode_id"], dtype=np.int64)
    shoes = np.asarray(
        payload.get("shoe_id", np.zeros(len(episodes), dtype=np.int64)),
        dtype=np.int64,
    )
    phases = np.asarray(
        payload.get("relation_phase", np.ones(len(episodes), dtype=np.float32))
    ) >= 0.5
    donors = np.empty_like(indices)
    for phase in (False, True):
        phase_positions = np.flatnonzero(phases[indices] == phase)
        if not len(phase_positions):
            continue
        phase_rows = indices[phase_positions]
        episode_values = sorted(np.unique(episodes[phase_rows]).astype(int))
        if len(episode_values) < 2:
            donors[phase_positions] = np.roll(
                phase_rows, max(1, len(phase_rows) // 2)
            )
            continue
        groups = {
            episode: phase_rows[episodes[phase_rows] == episode]
            for episode in episode_values
        }
        for source_position, source_episode in enumerate(episode_values):
            source_rows = groups[source_episode]
            source_shoe = int(shoes[source_rows[0]])
            donor_episode = None
            # Prefer a different object as well as a different trajectory.
            for offset in range(1, len(episode_values) + 1):
                candidate = episode_values[
                    (source_position + offset) % len(episode_values)
                ]
                if candidate != source_episode and int(shoes[groups[candidate][0]]) != source_shoe:
                    donor_episode = candidate
                    break
            if donor_episode is None:
                donor_episode = episode_values[
                    (source_position + 1) % len(episode_values)
                ]
            donor_rows = groups[donor_episode]
            if len(source_rows) == 1:
                donor_offsets = np.asarray([len(donor_rows) // 2], dtype=np.int64)
            else:
                donor_offsets = np.rint(
                    np.linspace(0, len(donor_rows) - 1, len(source_rows))
                ).astype(np.int64)
            row_to_donor = dict(zip(source_rows.tolist(), donor_rows[donor_offsets].tolist()))
            for row in source_rows:
                output_position = int(np.flatnonzero(indices == row)[0])
                donors[output_position] = int(row_to_donor[int(row)])
    return donors


class TaskAlignedGeometryDataset(BaseDataset):
    """Object-disjoint DP3 view of a frozen TAGRT multi-goal archive."""

    OBSERVATION_KEYS = (
        "point_cloud",
        "point_cloud_B",
        "agent_pos",
        "tagrt_local",
        "tagrt_global",
        "tagrt_confidence",
    )

    def __init__(
        self,
        npz_path: str,
        horizon: int = 8,
        n_obs_steps: int = 3,
        n_action_steps: int = 6,
        train_object_ids=(2, 3, 4, 7, 8, 9),
        val_object_ids=(1, 6),
        test_object_ids=(0, 5),
        split: str = "train",
        condition_mode: str = "correct",
        use_color: bool = False,
        history_stride: int = 1,
        geometry_scale_m: float | None = None,
        include_anchor_flow: bool = False,
        task_name=None,
    ):
        super().__init__()
        self.task_name = task_name
        self.path = Path(npz_path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        if int(horizon) < int(n_obs_steps) - 1 + int(n_action_steps):
            raise ValueError(
                "horizon must contain the DP3 action window starting at "
                "n_obs_steps-1"
            )
        if condition_mode not in {"correct", "zero", "shuffled"}:
            raise ValueError("condition_mode must be correct, zero, or shuffled")
        self.horizon = int(horizon)
        self.n_obs_steps = int(n_obs_steps)
        self.n_action_steps = int(n_action_steps)
        self.condition_mode = str(condition_mode)
        self.use_color = bool(use_color)
        if int(history_stride) <= 0:
            raise ValueError("history_stride must be positive")
        self.history_stride = int(history_stride)
        self.include_anchor_flow = bool(include_anchor_flow)
        self.observation_keys = self.OBSERVATION_KEYS + (
            ("tagrt_anchor_flow",) if self.include_anchor_flow else ()
        )
        with np.load(self.path, allow_pickle=False) as archive:
            required = {
                "points_a",
                "points_b",
                "state",
                "action",
                "shoe_id",
                "goal_frame9",
                "target_frame9",
            }
            missing = sorted(required - set(archive.files))
            if missing:
                raise KeyError(f"TAGRT archive is missing {missing}")
            self.payload = {key: np.asarray(archive[key]) for key in archive.files}
        count = len(self.payload["action"])
        if any(len(value) != count for key, value in self.payload.items() if key != "episode_flow" and key != "episode_pose" and key != "episode_shoe_id"):
            raise ValueError("TAGRT row arrays have inconsistent lengths")
        self.history_indices = self._build_history_indices(count)

        self.train_object_ids = _ids(train_object_ids)
        self.val_object_ids = _ids(val_object_ids)
        self.test_object_ids = _ids(test_object_ids)
        if (
            np.intersect1d(self.train_object_ids, self.val_object_ids).size
            or np.intersect1d(self.train_object_ids, self.test_object_ids).size
            or np.intersect1d(self.val_object_ids, self.test_object_ids).size
        ):
            raise ValueError("train/validation/test object IDs must be disjoint")
        shoe_ids = np.asarray(self.payload["shoe_id"], dtype=np.int64)
        self.split_indices = {
            "train": np.flatnonzero(np.isin(shoe_ids, self.train_object_ids)),
            "validation": np.flatnonzero(np.isin(shoe_ids, self.val_object_ids)),
            "test": np.flatnonzero(np.isin(shoe_ids, self.test_object_ids)),
        }
        if split not in self.split_indices:
            raise ValueError(f"unknown split {split!r}")
        self.split = str(split)
        self.active_indices = self.split_indices[self.split]
        if len(self.active_indices) == 0:
            raise ValueError(f"TAGRT {self.split} split is empty")

        train_points = np.concatenate(
            (
                self.payload["points_a"][self.split_indices["train"], :, :3],
                self.payload["points_b"][self.split_indices["train"], :, :3],
            ),
            axis=1,
        )
        xyz_std = np.std(train_points.reshape(-1, 3), axis=0)
        computed_geometry_scale_m = float(np.mean(np.maximum(xyz_std, 1e-4)))
        if geometry_scale_m is not None and float(geometry_scale_m) <= 0.0:
            raise ValueError("geometry_scale_m must be positive")
        self.geometry_scale_m = (
            computed_geometry_scale_m
            if geometry_scale_m is None
            else float(geometry_scale_m)
        )
        self.computed_geometry_scale_m = computed_geometry_scale_m
        # A deterministic, stage-matched cross-episode permutation for
        # wrong-relation controls.
        self.shuffled_indices = {}
        for name, indices in self.split_indices.items():
            self.shuffled_indices[name] = _episode_level_shuffle(
                indices, self.payload
            )

    def __len__(self) -> int:
        return int(len(self.active_indices))

    def _build_history_indices(self, count: int) -> np.ndarray:
        """Return causal, episode-local observation histories for every row.

        The row-wise archive stores one current observation per action chunk.
        Earlier versions repeated that observation across ``n_obs_steps``, which
        made online deployment with a real history a train/test mismatch.  At an
        episode boundary we left-pad with the first available frame, matching
        the online runtime reset contract.
        """
        episode_key = self.payload.get("history_episode_index")
        if episode_key is None:
            episode_key = self.payload.get("episode_index")
        if episode_key is None:
            episode_key = self.payload.get("episode_id")
        if episode_key is None:
            episode_key = np.arange(count, dtype=np.int64)
        episode_key = np.asarray(episode_key).reshape(-1)
        frame_index = np.asarray(
            self.payload.get(
                "history_frame_index",
                self.payload.get("frame_index", np.arange(count, dtype=np.int64)),
            )
        ).reshape(-1)
        output = np.empty((count, self.n_obs_steps), dtype=np.int64)
        for episode in np.unique(episode_key):
            rows = np.flatnonzero(episode_key == episode)
            rows = rows[np.argsort(frame_index[rows], kind="stable")]
            for position, row in enumerate(rows):
                positions = np.arange(
                    position - self.history_stride * (self.n_obs_steps - 1),
                    position + 1,
                    self.history_stride,
                    dtype=np.int64,
                )
                positions = np.clip(positions, 0, position)
                output[int(row)] = rows[positions]
        return output

    def get_validation_dataset(self):
        result = copy.copy(self)
        result.split = "validation"
        result.active_indices = self.split_indices["validation"]
        return result

    def get_test_dataset(self, condition_mode: str | None = None):
        result = copy.copy(self)
        result.split = "test"
        result.active_indices = self.split_indices["test"]
        if condition_mode is not None:
            if condition_mode not in {"correct", "zero", "shuffled"}:
                raise ValueError(condition_mode)
            result.condition_mode = str(condition_mode)
        return result

    def _geometry(
        self, row: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        if self.condition_mode == "zero":
            local = np.zeros(
                (len(self.payload["points_a"][row]) + len(self.payload["points_b"][row]), 4),
                dtype=np.float32,
            )
            anchor_flow = (
                np.zeros((len(self.payload["points_a"][row]), 9), dtype=np.float32)
                if self.include_anchor_flow
                else None
            )
            return local, np.zeros(9, dtype=np.float32), np.zeros(1, dtype=np.float32), anchor_flow
        geometry_row = row
        if self.condition_mode == "shuffled":
            position = int(np.flatnonzero(self.active_indices == row)[0])
            geometry_row = int(self.shuffled_indices[self.split][position])
        relative = np.asarray(self.payload["target_frame9"][geometry_row], dtype=np.float32)
        confidence = float(
            np.asarray(
                self.payload.get(
                    "target_frame_confidence",
                    np.ones(len(self.payload["action"]), dtype=np.float32),
                )
            )[geometry_row]
        )
        source_local, target_local = task_aligned_local_coordinates(
            self.payload["points_a"][row],
            self.payload["points_b"][row],
            self.payload["goal_frame9"][geometry_row],
            relative,
            scale_m=self.geometry_scale_m,
        )
        local = np.concatenate(
            (
                np.concatenate(
                    (source_local, -np.ones((len(source_local), 1), dtype=np.float32)),
                    axis=-1,
                ),
                np.concatenate(
                    (target_local, np.ones((len(target_local), 1), dtype=np.float32)),
                    axis=-1,
                ),
            ),
            axis=0,
        )
        global_relation = relative.copy()
        global_relation[:3] /= self.geometry_scale_m
        if confidence <= 0.0:
            local.fill(0.0)
            global_relation.fill(0.0)
        anchor_flow = None
        if self.include_anchor_flow:
            anchor_flow = task_aligned_anchor_flow_tokens(
                self.payload["points_a"][row],
                self.payload["state"][row],
                self.payload["goal_frame9"][geometry_row],
                relative,
                scale_m=self.geometry_scale_m,
            )
            if confidence <= 0.0:
                anchor_flow.fill(0.0)
        return (
            local.astype(np.float32),
            global_relation.astype(np.float32),
            np.asarray([np.clip(confidence, 0.0, 1.0)], dtype=np.float32),
            anchor_flow,
        )

    def _action_horizon(self, row: int) -> np.ndarray:
        chunk = np.asarray(self.payload["action"][row], dtype=np.float32)
        if len(chunk) < self.n_action_steps:
            raise ValueError("stored TAGRT action chunk is shorter than n_action_steps")
        output = np.empty((self.horizon, chunk.shape[-1]), dtype=np.float32)
        start = self.n_obs_steps - 1
        output[:start] = chunk[0]
        output[start : start + self.n_action_steps] = chunk[: self.n_action_steps]
        output[start + self.n_action_steps :] = chunk[self.n_action_steps - 1]
        return output

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = int(self.active_indices[int(index)])
        channels = 6 if self.use_color else 3
        history = []
        for history_row in self.history_indices[row]:
            history_row = int(history_row)
            local, global_relation, confidence, anchor_flow = self._geometry(history_row)
            item = {
                    "point_cloud": np.asarray(
                        self.payload["points_a"][history_row, :, :channels],
                        dtype=np.float32,
                    ),
                    "point_cloud_B": np.asarray(
                        self.payload["points_b"][history_row, :, :channels],
                        dtype=np.float32,
                    ),
                    "agent_pos": np.asarray(
                        self.payload["state"][history_row], dtype=np.float32
                    ),
                    "tagrt_local": local,
                    "tagrt_global": global_relation,
                    "tagrt_confidence": confidence,
                }
            if self.include_anchor_flow:
                item["tagrt_anchor_flow"] = anchor_flow
            history.append(item)
        observations = {}
        for key in self.observation_keys:
            causal = np.stack([item[key] for item in history], axis=0)
            if self.horizon > self.n_obs_steps:
                causal = np.concatenate(
                    (
                        causal,
                        np.repeat(
                            causal[-1:], self.horizon - self.n_obs_steps, axis=0
                        ),
                    ),
                    axis=0,
                )
            observations[key] = torch.from_numpy(causal.astype(np.float32))
        return {
            "obs": observations,
            "action": torch.from_numpy(self._action_horizon(row)),
            "is_recovery_augmented": torch.as_tensor(
                float(
                    np.asarray(
                        self.payload.get(
                            "is_recovery_augmented",
                            np.zeros(len(self.payload["action"]), dtype=np.float32),
                        )
                    )[row]
                ),
                dtype=torch.float32,
            ),
            "is_failure_matched_recovery": torch.as_tensor(
                float(
                    np.asarray(
                        self.payload.get(
                            "is_failure_matched_recovery",
                            np.zeros(len(self.payload["action"]), dtype=np.float32),
                        )
                    )[row]
                ),
                dtype=torch.float32,
            ),
            "is_recovery_gate_positive": torch.as_tensor(
                float(
                    np.asarray(
                        self.payload.get(
                            "is_recovery_gate_positive",
                            self.payload.get(
                                "is_failure_matched_recovery",
                                np.zeros(
                                    len(self.payload["action"]), dtype=np.float32
                                ),
                            ),
                        )
                    )[row]
                ),
                dtype=torch.float32,
            ),
            "is_translation_recovery_positive": torch.as_tensor(
                float(
                    np.asarray(
                        self.payload.get(
                            "is_translation_recovery_positive",
                            self.payload.get(
                                "is_recovery_gate_positive",
                                np.zeros(
                                    len(self.payload["action"]), dtype=np.float32
                                ),
                            ),
                        )
                    )[row]
                ),
                dtype=torch.float32,
            ),
            "active_arm_right": torch.as_tensor(
                float(
                    np.asarray(
                        self.payload.get(
                            "active_arm_right",
                            np.ones(len(self.payload["action"]), dtype=np.float32),
                        )
                    )[row]
                ),
                dtype=torch.float32,
            ),
        }

    def get_normalizer(self, mode="limits", **kwargs):
        train = self.split_indices["train"]
        channels = 6 if self.use_color else 3
        action = np.stack([self._action_horizon(int(row)) for row in train])
        data = {
            "action": action,
            "point_cloud": self.payload["points_a"][train, :, :channels],
            "point_cloud_B": self.payload["points_b"][train, :, :channels],
            "agent_pos": self.payload["state"][train],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        geometry_keys = ["tagrt_local", "tagrt_global", "tagrt_confidence"]
        if self.include_anchor_flow:
            geometry_keys.append("tagrt_anchor_flow")
        for key in geometry_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_identity()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(
            np.stack(
                [self._action_horizon(int(row)) for row in self.active_indices]
            )
        )
