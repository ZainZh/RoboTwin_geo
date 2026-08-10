from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch


DP3_ROOT = Path(__file__).resolve().parents[1] / "DP3" / "3D-Diffusion-Policy"
sys.path.insert(0, str(DP3_ROOT))

from diffusion_policy_3d.dataset.tagrt_dataset import (  # noqa: E402
    TaskAlignedGeometryDataset,
)


class TaskAlignedGeometryDatasetTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        path = Path(self.temp.name) / "rows.npz"
        count = 6
        points_a = np.zeros((count, 4, 6), dtype=np.float32)
        points_b = np.zeros((count, 4, 6), dtype=np.float32)
        for row in range(count):
            points_a[row, :, :3] = np.eye(4, 3) + row * 0.01
            points_b[row, :, :3] = np.eye(4, 3) + 0.1 + row * 0.01
        frame = np.tile(
            np.asarray([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float32),
            (count, 1),
        )
        relative = frame.copy()
        relative[:, 0] = np.linspace(0.01, 0.06, count)
        np.savez_compressed(
            path,
            points_a=points_a,
            points_b=points_b,
            state=np.zeros((count, 20), dtype=np.float32),
            action=np.ones((count, 6, 14), dtype=np.float32),
            shoe_id=np.asarray([2, 3, 1, 6, 0, 5]),
            goal_frame9=frame,
            target_frame9=relative,
            target_frame_confidence=np.ones(count, dtype=np.float32),
            episode_id=np.arange(count, dtype=np.int64),
            relation_phase=np.asarray([0, 0, 0, 0, 1, 1], dtype=np.float32),
        )
        self.path = path

    def tearDown(self):
        self.temp.cleanup()

    def dataset(self, **kwargs):
        return TaskAlignedGeometryDataset(
            str(self.path),
            train_object_ids=[2, 3],
            val_object_ids=[1, 6],
            test_object_ids=[0, 5],
            **kwargs,
        )

    def test_shapes_and_action_alignment(self):
        data = self.dataset()[0]
        self.assertEqual(tuple(data["obs"]["point_cloud"].shape), (8, 4, 3))
        self.assertEqual(tuple(data["obs"]["tagrt_local"].shape), (8, 8, 4))
        self.assertEqual(tuple(data["obs"]["tagrt_global"].shape), (8, 9))
        self.assertEqual(tuple(data["action"].shape), (8, 14))
        torch.testing.assert_close(data["action"][2:], torch.ones(6, 14))

    def test_object_splits_and_zero_control(self):
        dataset = self.dataset(condition_mode="zero")
        self.assertEqual(len(dataset), 2)
        self.assertEqual(len(dataset.get_validation_dataset()), 2)
        test = dataset.get_test_dataset()
        self.assertEqual(len(test), 2)
        sample = test[0]["obs"]
        torch.testing.assert_close(sample["tagrt_local"], torch.zeros_like(sample["tagrt_local"]))
        torch.testing.assert_close(sample["tagrt_global"], torch.zeros_like(sample["tagrt_global"]))

    def test_geometry_normalizer_is_identity(self):
        normalizer = self.dataset().get_normalizer()
        relation = torch.randn(2, 9)
        torch.testing.assert_close(
            normalizer["tagrt_global"].normalize(relation), relation
        )

    def test_shuffle_changes_episode_and_preserves_phase(self):
        dataset = self.dataset(condition_mode="shuffled").get_test_dataset()
        for row, donor in zip(dataset.active_indices, dataset.shuffled_indices["test"]):
            self.assertNotEqual(
                int(dataset.payload["episode_id"][row]),
                int(dataset.payload["episode_id"][donor]),
            )
            self.assertEqual(
                bool(dataset.payload["relation_phase"][row] >= 0.5),
                bool(dataset.payload["relation_phase"][donor] >= 0.5),
            )

    def test_history_is_causal_and_episode_local(self):
        path = Path(self.temp.name) / "temporal_rows.npz"
        count = 8
        points = np.zeros((count, 4, 6), dtype=np.float32)
        state = np.zeros((count, 20), dtype=np.float32)
        for row in range(count):
            points[row, :, 0] = float(row)
            state[row, 0] = float(row)
        frame = np.tile(
            np.asarray([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float32),
            (count, 1),
        )
        np.savez_compressed(
            path,
            points_a=points,
            points_b=points,
            state=state,
            action=np.ones((count, 6, 14), dtype=np.float32),
            shoe_id=np.asarray([2, 2, 2, 2, 1, 6, 0, 5]),
            goal_frame9=frame,
            target_frame9=frame,
            target_frame_confidence=np.ones(count, dtype=np.float32),
            episode_id=np.asarray([10, 10, 10, 11, 20, 21, 30, 31]),
            frame_index=np.asarray([0, 1, 2, 0, 0, 0, 0, 0]),
            relation_phase=np.zeros(count, dtype=np.float32),
        )
        dataset = TaskAlignedGeometryDataset(
            str(path),
            train_object_ids=[2],
            val_object_ids=[1, 6],
            test_object_ids=[0, 5],
        )
        np.testing.assert_array_equal(dataset.history_indices[0], [0, 0, 0])
        np.testing.assert_array_equal(dataset.history_indices[1], [0, 0, 1])
        np.testing.assert_array_equal(dataset.history_indices[2], [0, 1, 2])
        np.testing.assert_array_equal(dataset.history_indices[3], [3, 3, 3])
        sample = dataset[2]["obs"]["agent_pos"]
        torch.testing.assert_close(sample[:3, 0], torch.tensor([0.0, 1.0, 2.0]))
        torch.testing.assert_close(sample[3:, 0], torch.full((5,), 2.0))

    def test_history_stride_matches_chunked_online_observation_cadence(self):
        path = Path(self.temp.name) / "strided_temporal_rows.npz"
        count = 15
        points = np.zeros((count, 4, 6), dtype=np.float32)
        state = np.zeros((count, 20), dtype=np.float32)
        for row in range(count):
            points[row, :, 0] = float(row)
            state[row, 0] = float(row)
        frame = np.tile(
            np.asarray([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float32),
            (count, 1),
        )
        np.savez_compressed(
            path,
            points_a=points,
            points_b=points,
            state=state,
            action=np.ones((count, 6, 14), dtype=np.float32),
            shoe_id=np.full(count, 2, dtype=np.int64),
            goal_frame9=frame,
            target_frame9=frame,
            target_frame_confidence=np.ones(count, dtype=np.float32),
            episode_id=np.full(count, 10, dtype=np.int64),
            frame_index=np.arange(count, dtype=np.int64),
            relation_phase=np.zeros(count, dtype=np.float32),
        )
        dataset = TaskAlignedGeometryDataset(
            str(path),
            train_object_ids=[2],
            val_object_ids=[1],
            test_object_ids=[0],
            history_stride=6,
        )
        np.testing.assert_array_equal(dataset.history_indices[14], [2, 8, 14])
        np.testing.assert_array_equal(dataset.history_indices[7], [0, 1, 7])
        np.testing.assert_array_equal(dataset.history_indices[3], [0, 0, 3])

    def test_synthetic_history_identity_prevents_donor_frame_leakage(self):
        path = Path(self.temp.name) / "synthetic_history_rows.npz"
        count = 5
        points = np.zeros((count, 4, 6), dtype=np.float32)
        frame = np.tile(
            np.asarray([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float32),
            (count, 1),
        )
        np.savez_compressed(
            path,
            points_a=points,
            points_b=points,
            state=np.zeros((count, 20), dtype=np.float32),
            action=np.ones((count, 6, 14), dtype=np.float32),
            shoe_id=np.asarray([2, 2, 2, 1, 0]),
            goal_frame9=frame,
            target_frame9=frame,
            episode_id=np.asarray([10, 10, 10, 20, 30]),
            frame_index=np.asarray([0, 1, 1, 0, 0]),
            history_episode_index=np.asarray([10, 10, -1, 20, 30]),
            history_frame_index=np.asarray([0, 1, 0, 0, 0]),
            relation_phase=np.ones(count, dtype=np.float32),
        )
        dataset = TaskAlignedGeometryDataset(
            str(path),
            train_object_ids=[2],
            val_object_ids=[1],
            test_object_ids=[0],
        )
        np.testing.assert_array_equal(dataset.history_indices[1], [0, 0, 1])
        np.testing.assert_array_equal(dataset.history_indices[2], [2, 2, 2])


if __name__ == "__main__":
    unittest.main()
