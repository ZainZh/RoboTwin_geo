from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from .online_hanging_mug_camera_rotation import (
    OnlineHangingMugCameraRotationProvider,
)


class _SequenceEncoder:
    def __init__(self, frames):
        self.frames = [np.asarray(value, dtype=np.float32) for value in frames]
        self.index = 0

    def encode_frame(self, _points):
        value = self.frames[min(self.index, len(self.frames) - 1)]
        self.index += 1
        return value.copy()


def _cloud():
    rng = np.random.default_rng(7)
    return rng.normal(size=(256, 3)).astype(np.float32) * (0.1, 0.05, 0.03)


def _provider(encoders, *, goal_rotation=None):
    return OnlineHangingMugCameraRotationProvider(
        reference_target_cloud=_cloud(),
        reference_goal_rotation=(
            np.eye(3) if goal_rotation is None else goal_rotation
        ),
        source_confidence_threshold_deg=5.0,
        registration_confidence_threshold_m=1e-4,
        frame_encoders=encoders,
        registration_points=128,
    )


class OnlineHangingMugCameraRotationTest(unittest.TestCase):
    def test_clean_token_is_camera_rotation_with_exact_zero_translation(self):
        current = Rotation.from_euler("z", 10.0, degrees=True).as_matrix()
        goal = Rotation.from_euler("z", 25.0, degrees=True).as_matrix()
        provider = _provider(
            [_SequenceEncoder([current]), _SequenceEncoder([current])],
            goal_rotation=goal,
        )
        cloud = _cloud()
        estimate = provider.estimate(
            current_point_cloud=cloud,
            target_point_cloud=cloud,
            mode="clean",
        )
        expected = goal @ current.T
        np.testing.assert_array_equal(estimate.frame9_metric[:3], np.zeros(3))
        np.testing.assert_allclose(
            estimate.frame9_metric[3:6], expected[:, 0], atol=1e-5
        )
        np.testing.assert_allclose(
            estimate.frame9_metric[6:9], expected[:, 1], atol=1e-5
        )
        self.assertEqual(estimate.combined_confidence, 1.0)

    def test_temporal_stabilization_removes_proper_axis_sign_flip(self):
        flipped = np.diag((-1.0, -1.0, 1.0)).astype(np.float32)
        encoders = [
            _SequenceEncoder([np.eye(3), flipped]),
            _SequenceEncoder([np.eye(3), flipped]),
        ]
        provider = _provider(encoders)
        cloud = _cloud()
        first = provider.estimate(
            current_point_cloud=cloud, target_point_cloud=cloud, mode="clean"
        )
        second = provider.estimate(
            current_point_cloud=cloud, target_point_cloud=cloud, mode="clean"
        )
        np.testing.assert_allclose(first.source_rotation, np.eye(3), atol=1e-6)
        np.testing.assert_allclose(second.source_rotation, np.eye(3), atol=1e-6)
        self.assertEqual(second.source_confidence, 1.0)

    def test_zero_mode_never_runs_camera_encoder_or_uses_oracle(self):
        class _FailEncoder:
            def encode_frame(self, _points):
                raise AssertionError("zero intervention must bypass frame estimation")

        provider = _provider([_FailEncoder(), _FailEncoder()])
        estimate = provider.estimate(
            current_point_cloud=_cloud(), target_point_cloud=_cloud(), mode="zero"
        )
        np.testing.assert_array_equal(estimate.frame9_metric, np.zeros(9))
        self.assertEqual(estimate.combined_confidence, 0.0)
        self.assertFalse(estimate.target_latched)


if __name__ == "__main__":
    unittest.main()
