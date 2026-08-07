from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch import nn

from .deploy_interaction_flow import (
    _path_list,
    _pose7_wxyz_matrix,
    _privileged_oracle_relative_frame9,
)
from .interaction_flow_runtime import InteractionFlowRuntime
from .online_ndf_functional_frame import OnlineNdfFunctionalFrameProvider
from .train_interaction_flow_tokens import build_model


class RecordingEncoder:
    def __init__(self, rotation: np.ndarray) -> None:
        self.rotation = np.asarray(rotation, dtype=np.float32)
        self.inputs: list[np.ndarray] = []

    def encode_frame(self, support_points_world: np.ndarray) -> np.ndarray:
        self.inputs.append(np.asarray(support_points_world).copy())
        return self.rotation.copy()


class RecordingMarker:
    def __init__(self) -> None:
        self.inputs: list[np.ndarray] = []

    def __call__(self, point_cloud: np.ndarray):
        self.inputs.append(np.asarray(point_cloud).copy())
        frame = np.eye(4, dtype=np.float32)
        frame[:3, 3] = np.asarray([0.8, -0.2, 0.5], dtype=np.float32)
        return frame, {"marker_points": 40, "fit_score_m2": 1.0e-6}


class CapturingModel(nn.Module):
    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base
        self.last_batch = None

    def forward(self, batch):
        self.last_batch = {key: value.detach().cpu().clone() for key, value in batch.items()}
        return self.base(batch)


class FakePose:
    def __init__(self, position, quaternion_wxyz):
        self.p = np.asarray(position, dtype=np.float32)
        self.q = np.asarray(quaternion_wxyz, dtype=np.float32)


class FakeActor:
    def __init__(self, pose):
        self.pose = pose

    def get_functional_point(self, _index, _kind):
        return self.pose


def make_provider(*, allow_oracle=False):
    encoders = [RecordingEncoder(np.eye(3)) for _ in range(3)]
    marker = RecordingMarker()
    provider = OnlineNdfFunctionalFrameProvider(
        frame_encoders=encoders,
        current_origin_offset_local3=[0.1, 0.0, 0.0],
        goal_position_offset_marker_local3=[0.0, 0.2, 0.0],
        goal_rotation_offset_marker_local9=np.eye(3),
        confidence_threshold_deg=5.0,
        marker_frame_estimator=marker,
        allow_privileged_oracle=allow_oracle,
    )
    return provider, encoders, marker


class DenseInteractionRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.checkpoint = Path(self.temporary.name) / "dense.pt"
        self.normalization = {
            "state_mean": np.zeros(20, dtype=np.float32),
            "state_std": np.ones(20, dtype=np.float32),
            "action_mean": np.zeros(14, dtype=np.float32),
            "action_std": np.ones(14, dtype=np.float32),
            "xyz_mean": np.zeros(3, dtype=np.float32),
            "xyz_std": np.asarray([2.0, 4.0, 8.0], dtype=np.float32),
        }
        model = build_model(
            "interaction_functional_direct_flow",
            horizon=2,
            flow_steps=2,
            num_anchors=4,
            feature_dim=32,
            layers=1,
            flow_parameterization="free",
            xyz_std=self.normalization["xyz_std"],
            point_channels=3,
        )
        torch.save(
            {
                "model": model.state_dict(),
                "condition": "interaction_functional_direct_flow",
                "horizon": 2,
                "flow_steps": 2,
                "num_anchors": 4,
                "feature_dim": 32,
                "layers": 1,
                "point_channels": 3,
                "flow_parameterization": "free",
                "action_frame": "world",
                "functional_frame_encoding": (
                    "camera_ndf_current_marker_goal_se3_columns_v1"
                ),
                "functional_frame_translation_mode": "delta",
                "normalization": self.normalization,
            },
            self.checkpoint,
        )
        generator = np.random.default_rng(31)
        self.source = generator.normal(size=(180, 3)).astype(np.float32)
        self.target = generator.normal(size=(170, 6)).astype(np.float32)
        self.target[:, 3:6] = np.asarray([0.9, 0.8, 0.05])

    def tearDown(self):
        self.temporary.cleanup()

    def runtime(self, *, mode="clean", allow_oracle=False):
        provider, encoders, marker = make_provider(allow_oracle=allow_oracle)
        runtime = InteractionFlowRuntime(
            self.checkpoint,
            device="cpu",
            observation_points=128,
            dense_frame_mode=mode,
            allow_privileged_oracle=allow_oracle,
            dense_frame_provider=provider,
        )
        runtime.model = CapturingModel(runtime.model)
        return runtime, provider, encoders, marker

    def test_clean_uses_same_fps_a_raw_rgb_b_confidence_and_latch(self):
        runtime, provider, encoders, marker = self.runtime()
        expected_a = runtime._points(self.source)[:, :3]
        action = runtime.predict(
            operated_point_cloud=self.source,
            target_point_cloud=self.target,
            eef_state20=np.zeros(20, dtype=np.float32),
        )
        self.assertEqual(action.shape, (2, 14))
        self.assertTrue(np.all(np.isfinite(action)))
        for encoder in encoders:
            np.testing.assert_array_equal(encoder.inputs[0], expected_a)
            self.assertEqual(encoder.inputs[0].shape, (128, 3))
        self.assertEqual(marker.inputs[0].shape, (170, 6))
        np.testing.assert_array_equal(marker.inputs[0], self.target)
        batch = runtime.model.last_batch
        self.assertEqual(float(batch["target_frame_confidence"][0]), 1.0)
        estimate = provider.estimate(
            current_point_cloud=expected_a,
            target_point_cloud=self.target + 20.0,
        )
        np.testing.assert_allclose(
            batch["target_frame9"][0].numpy(),
            estimate.policy_frame9(self.normalization["xyz_std"]),
            atol=1e-6,
        )
        runtime.predict(
            operated_point_cloud=self.source,
            target_point_cloud=self.target + 10.0,
            eef_state20=np.zeros(20, dtype=np.float32),
        )
        self.assertEqual(len(marker.inputs), 1)
        self.assertEqual(runtime.last_diagnostic["dense_combined_confidence"], 1.0)
        runtime.reset()
        self.assertFalse(provider.target_latched)

    def test_dense_recovery_adapter_checkpoint_is_deployable(self):
        checkpoint = Path(self.temporary.name) / "dense_recovery_adapter.pt"
        model = build_model(
            "interaction_functional_correction_adapter_flow",
            horizon=2,
            flow_steps=2,
            num_anchors=4,
            feature_dim=32,
            layers=1,
            flow_parameterization="free",
            xyz_std=self.normalization["xyz_std"],
            point_channels=3,
        )
        torch.save(
            {
                "model": model.state_dict(),
                "condition": "interaction_functional_correction_adapter_flow",
                "horizon": 2,
                "flow_steps": 2,
                "num_anchors": 4,
                "feature_dim": 32,
                "layers": 1,
                "point_channels": 3,
                "flow_parameterization": "free",
                "action_frame": "world",
                "functional_frame_encoding": (
                    "camera_ndf_current_marker_goal_se3_columns_v1"
                ),
                "functional_frame_translation_mode": "delta",
                "normalization": self.normalization,
            },
            checkpoint,
        )
        provider, _encoders, _marker = make_provider()
        runtime = InteractionFlowRuntime(
            checkpoint,
            device="cpu",
            observation_points=128,
            dense_frame_mode="clean",
            dense_frame_provider=provider,
        )
        action = runtime.predict(
            operated_point_cloud=self.source,
            target_point_cloud=self.target,
            eef_state20=np.zeros(20, dtype=np.float32),
        )
        self.assertEqual(action.shape, (2, 14))
        self.assertTrue(np.all(np.isfinite(action)))

    def test_hand_object_frame_checkpoint_builds_camera_proprio_token(self):
        checkpoint = Path(self.temporary.name) / "dense_hand_object_adapter.pt"
        model = build_model(
            "interaction_functional_grasp_adapter_flow",
            horizon=2,
            flow_steps=2,
            num_anchors=4,
            feature_dim=32,
            layers=1,
            flow_parameterization="free",
            xyz_std=self.normalization["xyz_std"],
            point_channels=3,
        )
        torch.save(
            {
                "model": model.state_dict(),
                "condition": "interaction_functional_grasp_adapter_flow",
                "horizon": 2,
                "flow_steps": 2,
                "num_anchors": 4,
                "feature_dim": 32,
                "layers": 1,
                "point_channels": 3,
                "flow_parameterization": "free",
                "action_frame": "world",
                "hand_object_frame_token": True,
                "hand_object_translation_scale_m": 0.2,
                "functional_frame_encoding": (
                    "camera_ndf_current_marker_goal_se3_columns_v1"
                ),
                "functional_frame_translation_mode": "delta",
                "normalization": self.normalization,
            },
            checkpoint,
        )
        provider, _encoders, _marker = make_provider()
        runtime = InteractionFlowRuntime(
            checkpoint,
            device="cpu",
            observation_points=128,
            dense_frame_mode="clean",
            dense_frame_provider=provider,
        )
        runtime.model = CapturingModel(runtime.model)
        identity6 = np.eye(3, dtype=np.float32)[:, :2].reshape(6)
        state = np.concatenate(
            (
                [0.1, 0.0, 0.0],
                identity6,
                [0.0],
                [-0.2, 0.0, 0.0],
                identity6,
                [1.0],
            )
        ).astype(np.float32)
        action = runtime.predict(
            operated_point_cloud=self.source,
            target_point_cloud=self.target,
            eef_state20=state,
        )
        self.assertTrue(np.all(np.isfinite(action)))
        token = runtime.model.last_batch["hand_object_frame11"][0].numpy()
        self.assertEqual(token.shape, (11,))
        np.testing.assert_array_equal(token[-2:], [1.0, 0.0])
        self.assertEqual(
            float(runtime.model.last_batch["hand_object_frame_confidence"][0]),
            1.0,
        )
        self.assertGreater(
            float(runtime.last_diagnostic["hand_object_translation_norm_cm"]),
            0.0,
        )

    def test_zero_is_exact_and_skips_ndf_and_marker(self):
        runtime, _provider, encoders, marker = self.runtime(mode="zero")
        runtime.predict(
            operated_point_cloud=self.source,
            target_point_cloud=self.target,
            eef_state20=np.zeros(20, dtype=np.float32),
        )
        batch = runtime.model.last_batch
        np.testing.assert_array_equal(batch["target_frame9"][0], np.zeros(9))
        self.assertEqual(float(batch["target_frame_confidence"][0]), 0.0)
        self.assertEqual(sum(len(item.inputs) for item in encoders), 0)
        self.assertEqual(len(marker.inputs), 0)

    def test_dense_guards_point_count_mode_and_oracle_authority(self):
        provider, _encoders, _marker = make_provider()
        with self.assertRaisesRegex(ValueError, "explicit dense_frame_mode"):
            InteractionFlowRuntime(
                self.checkpoint,
                observation_points=128,
                dense_frame_provider=provider,
            )
        with self.assertRaisesRegex(ValueError, "exactly 128"):
            InteractionFlowRuntime(
                self.checkpoint,
                observation_points=64,
                dense_frame_mode="clean",
                dense_frame_provider=provider,
            )
        with self.assertRaises(PermissionError):
            InteractionFlowRuntime(
                self.checkpoint,
                observation_points=128,
                dense_frame_mode="oracle",
                dense_frame_provider=provider,
            )

    def test_dense_metadata_runtime_accepts_top_two_and_rejects_single_model(self):
        provider, _encoders, _marker = make_provider()
        factory_path = (
            "policy.GeometryFlow.interaction_flow_runtime."
            "OnlineNdfFunctionalFrameProvider.from_metadata"
        )
        with patch(factory_path, return_value=provider) as factory:
            runtime = InteractionFlowRuntime(
                self.checkpoint,
                observation_points=128,
                dense_frame_mode="clean",
                dense_frame_checkpoints=["seed0.pth", "seed1.pth"],
                dense_camera_metadata="camera.json",
                dense_ensemble_metadata="top2.json",
            )
        self.assertIs(runtime.dense_frame_provider, provider)
        factory.assert_called_once()
        with self.assertRaisesRegex(ValueError, "at least two"):
            InteractionFlowRuntime(
                self.checkpoint,
                observation_points=128,
                dense_frame_mode="clean",
                dense_frame_checkpoints=["seed0.pth"],
                dense_camera_metadata="camera.json",
                dense_ensemble_metadata="single.json",
            )

    def test_oracle_is_explicit_and_normalized(self):
        runtime, _provider, _encoders, _marker = self.runtime(
            mode="oracle", allow_oracle=True
        )
        oracle = np.concatenate(
            ([0.2, -0.4, 0.8], np.eye(3)[:, 0], np.eye(3)[:, 1])
        ).astype(np.float32)
        runtime.predict(
            operated_point_cloud=self.source,
            target_point_cloud=self.target,
            eef_state20=np.zeros(20, dtype=np.float32),
            oracle_relative_frame9_metric=oracle,
        )
        expected = oracle.copy()
        expected[:3] /= self.normalization["xyz_std"]
        np.testing.assert_allclose(
            runtime.model.last_batch["target_frame9"][0], expected
        )
        self.assertEqual(
            float(runtime.model.last_batch["target_frame_confidence"][0]), 1.0
        )

    def test_deploy_parsing_and_privileged_pose_conversion(self):
        self.assertEqual(_path_list("a.pth,b.pth,c.pth"), ["a.pth", "b.pth", "c.pth"])
        self.assertEqual(_path_list(["a", "b", "c"]), ["a", "b", "c"])
        current_rotation = Rotation.from_euler("x", 90, degrees=True).as_quat()
        target_rotation = Rotation.from_euler("z", 90, degrees=True).as_quat()
        current = np.concatenate(
            ([0.2, -0.1, 0.4], current_rotation[[3, 0, 1, 2]])
        )
        target = np.concatenate(
            ([0.1, 0.2, 0.3], target_rotation[[3, 0, 1, 2]])
        )
        goal_t_a_from_b = np.eye(4)
        goal_t_a_from_b[:3, 3] = [0.04, -0.03, 0.02]
        observation = {
            "task_state": {
                "object_pose_A": current,
                "object_pose_B": target,
                "goal_T_A_from_B_oracle": goal_t_a_from_b.reshape(-1),
            }
        }
        frame = _privileged_oracle_relative_frame9(observation)
        world_from_b = _pose7_wxyz_matrix(target)
        expected_goal = world_from_b @ np.linalg.inv(goal_t_a_from_b)
        expected_current = _pose7_wxyz_matrix(current)
        np.testing.assert_allclose(
            frame[:3], expected_goal[:3, 3] - expected_current[:3, 3], atol=1e-6
        )
        expected_rotation = expected_goal[:3, :3] @ expected_current[:3, :3].T
        np.testing.assert_allclose(frame[3:6], expected_rotation[:, 0], atol=1e-6)
        np.testing.assert_allclose(frame[6:9], expected_rotation[:, 1], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
