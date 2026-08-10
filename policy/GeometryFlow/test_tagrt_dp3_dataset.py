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
    task_aligned_anchor_flow_tokens,
)
from diffusion_policy_3d.policy.dp3 import (  # noqa: E402
    FlowConsistentAnchorTranslationHead,
    _active_translation_action_loss,
    _active_rotation_action_loss,
    _frame9_rotation_rotvec,
    _recovery_action_channel_weights,
    _replace_closed_arm_rotations,
    _replace_closed_arm_translations,
)


class TaskAlignedGeometryDatasetTest(unittest.TestCase):
    def test_flow_consistent_head_chunk_sums_to_learned_endpoint(self):
        head = FlowConsistentAnchorTranslationHead(
            in_channels=9,
            hidden_dim=16,
            global_dim=9,
            confidence_dim=1,
            action_steps=6,
        )
        tokens = torch.randn(2, 8, 9)
        relation = torch.randn(2, 9)
        confidence = torch.ones(2, 1)
        output = head(tokens, relation, confidence)
        summary = torch.cat(
            (
                tokens.mean(1),
                tokens.std(1, unbiased=False),
                tokens.min(1).values,
                tokens.max(1).values,
                relation,
                confidence,
            ),
            dim=-1,
        )
        endpoint = head.endpoint(head.trunk(summary)).reshape(2, 2, 3)
        torch.testing.assert_close(output.sum(dim=1), endpoint)

    def test_anchor_flow_tokens_encode_eef_offsets_and_rigid_displacement(self):
        points = np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=np.float32)
        state = np.zeros(20, dtype=np.float32)
        state[10:13] = [0.2, 0.0, 0.0]
        goal = np.asarray([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float32)
        relation = goal.copy()
        relation[:3] = [0.01, 0.0, 0.0]
        tokens = task_aligned_anchor_flow_tokens(
            points, state, goal, relation, scale_m=0.1
        )
        self.assertEqual(tokens.shape, (2, 9))
        np.testing.assert_allclose(tokens[:, 6:9], [[0.1, 0, 0], [0.1, 0, 0]])
        np.testing.assert_allclose(tokens[:, 3:6], (points - state[10:13]) / 0.1)

    def test_dataset_option_exposes_anchor_flow_with_identity_normalizer(self):
        dataset = self.dataset(include_anchor_flow=True)
        sample = dataset[0]["obs"]["tagrt_anchor_flow"]
        self.assertEqual(tuple(sample.shape), (dataset.horizon, 4, 9))
        normalizer = dataset.get_normalizer()
        torch.testing.assert_close(
            normalizer["tagrt_anchor_flow"].normalize(sample), sample
        )

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
        self.assertEqual(float(data["is_recovery_augmented"]), 0.0)
        self.assertEqual(float(data["is_failure_matched_recovery"]), 0.0)
        self.assertEqual(float(data["is_recovery_gate_positive"]), 0.0)
        self.assertEqual(float(data["active_arm_right"]), 1.0)

    def test_recovery_rotation_weights_preserve_nominal_and_select_active_arm(self):
        weights = _recovery_action_channel_weights(
            torch.tensor([0.0, 1.0, 1.0]),
            torch.tensor([0.0, 0.0, 1.0]),
            horizon=2,
            action_dim=14,
            rotation_weight=4.0,
        )
        torch.testing.assert_close(weights[0], torch.ones((2, 14)))
        self.assertEqual(int(torch.count_nonzero(weights[1])), 6)
        self.assertEqual(int(torch.count_nonzero(weights[2])), 6)
        torch.testing.assert_close(weights[1, :, 3:6], torch.full((2, 3), 4.0))
        torch.testing.assert_close(weights[2, :, 10:13], torch.full((2, 3), 4.0))

    def test_factorized_rotation_replacement_preserves_translation_and_gripper(self):
        action = torch.arange(2 * 3 * 14, dtype=torch.float32).reshape(2, 3, 14)
        prediction = torch.full((2, 3, 2, 3), -7.0)
        replaced = _replace_closed_arm_rotations(
            action, prediction, torch.tensor([[True, False], [False, True]])
        )
        torch.testing.assert_close(replaced[0, :, 3:6], prediction[0, :, 0])
        torch.testing.assert_close(replaced[0, :, 10:13], action[0, :, 10:13])
        torch.testing.assert_close(replaced[1, :, 3:6], action[1, :, 3:6])
        torch.testing.assert_close(replaced[1, :, 10:13], prediction[1, :, 1])
        preserved = [0, 1, 2, 6, 7, 8, 9, 13]
        torch.testing.assert_close(replaced[..., preserved], action[..., preserved])

    def test_gated_translation_replacement_preserves_rotation_and_gripper(self):
        action = torch.arange(2 * 3 * 14, dtype=torch.float32).reshape(2, 3, 14)
        prediction = torch.full((2, 3, 2, 3), -5.0)
        replaced = _replace_closed_arm_translations(
            action, prediction, torch.tensor([[True, False], [False, True]])
        )
        torch.testing.assert_close(replaced[0, :, :3], prediction[0, :, 0])
        torch.testing.assert_close(replaced[0, :, 7:10], action[0, :, 7:10])
        torch.testing.assert_close(replaced[1, :, :3], action[1, :, :3])
        torch.testing.assert_close(replaced[1, :, 7:10], prediction[1, :, 1])
        preserved = [3, 4, 5, 6, 10, 11, 12, 13]
        torch.testing.assert_close(replaced[..., preserved], action[..., preserved])

    def test_translation_endpoint_loss_penalizes_cumulative_error(self):
        prediction = torch.zeros((1, 2, 2, 3), dtype=torch.float32)
        target = torch.zeros((1, 2, 14), dtype=torch.float32)
        target[0, :, :3] = 1.0
        per_step = _active_translation_action_loss(
            prediction,
            target,
            torch.tensor([False]),
            torch.tensor([True]),
            endpoint_weight=0.0,
        )
        with_endpoint = _active_translation_action_loss(
            prediction,
            target,
            torch.tensor([False]),
            torch.tensor([True]),
            endpoint_weight=1.0,
        )
        self.assertAlmostEqual(float(per_step), 1.0)
        self.assertAlmostEqual(float(with_endpoint), 5.0)

    def test_rotation_head_loss_selects_active_arm_and_weights_recovery(self):
        prediction = torch.zeros((2, 2, 2, 3), dtype=torch.float32)
        target = torch.zeros((2, 2, 14), dtype=torch.float32)
        target[0, :, 3:6] = 1.0
        target[0, :, 10:13] = 20.0
        target[1, :, 3:6] = 30.0
        target[1, :, 10:13] = 2.0
        loss = _active_rotation_action_loss(
            prediction,
            target,
            torch.tensor([False, True]),
            is_recovery=torch.tensor([False, True]),
            recovery_weight=3.0,
        )
        self.assertAlmostEqual(float(loss), (1.0 + 3.0 * 4.0) / 4.0)

    def test_frame9_rotvec_preserves_signed_axis(self):
        angle = torch.tensor(torch.pi / 3.0)
        cosine = torch.cos(angle)
        sine = torch.sin(angle)
        frame = torch.tensor(
            [0.0, 0.0, 0.0, cosine, sine, 0.0, -sine, cosine, 0.0]
        )
        rotvec = _frame9_rotation_rotvec(frame)
        torch.testing.assert_close(
            rotvec, torch.tensor([0.0, 0.0, angle]), atol=1.0e-5, rtol=1.0e-5
        )
        torch.testing.assert_close(
            _frame9_rotation_rotvec(torch.zeros(9)), torch.zeros(3)
        )

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

    def test_explicit_geometry_scale_freezes_token_contract(self):
        dataset = self.dataset(geometry_scale_m=0.25)
        self.assertAlmostEqual(dataset.geometry_scale_m, 0.25)
        self.assertNotAlmostEqual(dataset.computed_geometry_scale_m, 0.25)
        with self.assertRaisesRegex(ValueError, "geometry_scale_m"):
            self.dataset(geometry_scale_m=0.0)

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
