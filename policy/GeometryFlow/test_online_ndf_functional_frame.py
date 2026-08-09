from __future__ import annotations

import json
from pathlib import Path
import tempfile
import numpy as np
from scipy.spatial.transform import Rotation
import unittest
from unittest.mock import patch

from .online_ndf_functional_frame import OnlineNdfFunctionalFrameProvider


class StubFrameEncoder:
    def __init__(self, rotation: np.ndarray) -> None:
        self.rotation = np.asarray(rotation, dtype=np.float32)
        self.calls = 0

    def encode_frame(self, support_points_world: np.ndarray) -> np.ndarray:
        self.calls += 1
        assert support_points_world.shape[-1] == 3
        return self.rotation.copy()


class SequenceFrameEncoder:
    def __init__(self, rotations: list[np.ndarray]) -> None:
        self.rotations = [np.asarray(value, dtype=np.float32) for value in rotations]
        self.calls = 0

    def encode_frame(self, support_points_world: np.ndarray) -> np.ndarray:
        assert support_points_world.shape[-1] == 3
        value = self.rotations[min(self.calls, len(self.rotations) - 1)]
        self.calls += 1
        return value.copy()


def marker_estimator(rotation: np.ndarray, position: np.ndarray):
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float32)
    transform[:3, 3] = np.asarray(position, dtype=np.float32)

    def estimate(_cloud):
        return transform.copy(), {"marker_points": 25, "fit_score_m2": 1.0e-6}

    return estimate


def provider(encoders, *, marker_rotation=None, marker_position=None, **kwargs):
    return OnlineNdfFunctionalFrameProvider(
        frame_encoders=encoders,
        current_origin_offset_local3=kwargs.pop(
            "current_origin_offset_local3", [0.1, -0.02, 0.03]
        ),
        goal_position_offset_marker_local3=kwargs.pop(
            "goal_position_offset_marker_local3", [0.04, 0.05, -0.01]
        ),
        goal_rotation_offset_marker_local9=kwargs.pop(
            "goal_rotation_offset_marker_local9", np.eye(3)
        ),
        confidence_threshold_deg=kwargs.pop("confidence_threshold_deg", 5.0),
        marker_frame_estimator=marker_estimator(
            np.eye(3) if marker_rotation is None else marker_rotation,
            [1.0, 2.0, 3.0] if marker_position is None else marker_position,
        ),
        **kwargs,
    )


def clouds():
    current = np.asarray(
        [[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, 0.3, 0.0]],
        dtype=np.float32,
    )
    target = np.column_stack(
        (
            current + np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
            np.tile(np.asarray([0.9, 0.8, 0.05]), (len(current), 1)),
        )
    ).astype(np.float32)
    return current, target


def test_clean_matches_training_relative_frame_contract():
    current_rotation = Rotation.from_euler("z", 30.0, degrees=True).as_matrix()
    marker_rotation = Rotation.from_euler("x", -20.0, degrees=True).as_matrix()
    goal_offset_rotation = Rotation.from_euler("y", 10.0, degrees=True).as_matrix()
    encoders = [StubFrameEncoder(current_rotation) for _ in range(3)]
    runtime = provider(
        encoders,
        marker_rotation=marker_rotation,
        marker_position=[1.0, 2.0, 3.0],
        goal_rotation_offset_marker_local9=goal_offset_rotation,
    )
    current_cloud, target_cloud = clouds()
    estimate = runtime.estimate(
        current_point_cloud=current_cloud,
        target_point_cloud=target_cloud,
        mode="clean",
    )
    current_position = (
        current_cloud.mean(axis=0)
        + current_rotation @ np.asarray([0.1, -0.02, 0.03])
    )
    goal_position = (
        np.asarray([1.0, 2.0, 3.0])
        + marker_rotation @ np.asarray([0.04, 0.05, -0.01])
    )
    relative_rotation = (
        marker_rotation @ goal_offset_rotation @ current_rotation.T
    )
    expected = np.concatenate(
        (
            goal_position - current_position,
            relative_rotation[:, 0],
            relative_rotation[:, 1],
        )
    )
    np.testing.assert_allclose(estimate.frame9_metric, expected, atol=1e-6)
    normalized = estimate.policy_frame9([2.0, 4.0, 8.0])
    np.testing.assert_allclose(normalized[:3], expected[:3] / [2.0, 4.0, 8.0])
    np.testing.assert_allclose(normalized[3:], expected[3:], atol=1e-6)
    assert estimate.combined_confidence == 1.0
    assert estimate.source_disagreement_deg < 1e-4


def test_inverse_preserves_confidence_and_magnitude_but_reverses_relation():
    current_rotation = Rotation.from_euler("zyx", [25.0, -10.0, 5.0], degrees=True).as_matrix()
    encoders = [StubFrameEncoder(current_rotation) for _ in range(2)]
    runtime = provider(encoders)
    current, target = clouds()
    clean = runtime.estimate(
        current_point_cloud=current,
        target_point_cloud=target,
        mode="clean",
    )
    inverse = runtime.estimate(
        current_point_cloud=current,
        target_point_cloud=target,
        mode="inverse",
    )
    clean_rotation = np.column_stack(
        (
            clean.frame9_metric[3:6],
            clean.frame9_metric[6:9],
            np.cross(clean.frame9_metric[3:6], clean.frame9_metric[6:9]),
        )
    )
    inverse_rotation = np.column_stack(
        (
            inverse.frame9_metric[3:6],
            inverse.frame9_metric[6:9],
            np.cross(inverse.frame9_metric[3:6], inverse.frame9_metric[6:9]),
        )
    )
    np.testing.assert_allclose(inverse.frame9_metric[:3], -clean.frame9_metric[:3])
    np.testing.assert_allclose(inverse_rotation, clean_rotation.T, atol=1e-6)
    assert inverse.combined_confidence == clean.combined_confidence
    assert inverse.mode == "inverse"


def test_ensemble_disagreement_exactly_masks_frame():
    encoders = [
        StubFrameEncoder(Rotation.from_euler("z", value, degrees=True).as_matrix())
        for value in (0.0, 1.0, 90.0)
    ]
    runtime = provider(encoders, confidence_threshold_deg=5.0)
    current_cloud, target_cloud = clouds()
    estimate = runtime.estimate(
        current_point_cloud=current_cloud,
        target_point_cloud=target_cloud,
    )
    assert estimate.source_disagreement_deg > 80.0
    assert estimate.source_confidence == 0.0
    assert estimate.target_confidence == 1.0
    assert estimate.combined_confidence == 0.0


def test_medoid_aggregation_rejects_one_flipped_member():
    rotations = [
        Rotation.from_euler("z", value, degrees=True).as_matrix()
        for value in (1.0, 3.0, 179.0)
    ]
    runtime = provider(
        [StubFrameEncoder(value) for value in rotations],
        aggregation="medoid",
        confidence_threshold_deg=180.0,
    )
    current_cloud, target_cloud = clouds()
    estimate = runtime.estimate(
        current_point_cloud=current_cloud,
        target_point_cloud=target_cloud,
    )
    angle = Rotation.from_matrix(estimate.source_rotation).as_euler(
        "xyz", degrees=True
    )[2]
    assert abs(float(angle) - 3.0) < 1e-4


def test_temporal_sign_jump_is_corrected_but_confidence_masked():
    sequence = [
        np.eye(3),
        Rotation.from_euler("z", 179.0, degrees=True).as_matrix(),
    ]
    runtime = provider(
        [SequenceFrameEncoder(sequence), SequenceFrameEncoder(sequence)],
        aggregation="medoid",
        confidence_threshold_deg=180.0,
        temporal_stabilization=True,
    )
    current, target = clouds()
    first = runtime.estimate(
        current_point_cloud=current, target_point_cloud=target
    )
    second = runtime.estimate(
        current_point_cloud=current, target_point_cloud=target
    )
    assert first.source_confidence == 1.0
    assert second.source_confidence == 0.0
    assert second.source_temporal_ambiguous
    assert second.source_temporal_corrected
    angle = Rotation.from_matrix(second.source_rotation).magnitude()
    assert float(np.rad2deg(angle)) < 5.0
    runtime.reset()
    assert runtime._previous_source_rotation is None


def test_combined_confidence_masks_unreliable_marker():
    encoder = StubFrameEncoder(np.eye(3))

    def unreliable_marker(_cloud):
        return np.eye(4, dtype=np.float32), {
            "marker_points": 7,
            "fit_score_m2": 1.0e-3,
        }

    runtime = OnlineNdfFunctionalFrameProvider(
        frame_encoders=[encoder, encoder],
        current_origin_offset_local3=[0.0, 0.0, 0.0],
        goal_position_offset_marker_local3=[0.0, 0.0, 0.0],
        goal_rotation_offset_marker_local9=np.eye(3),
        confidence_threshold_deg=1.0,
        marker_frame_estimator=unreliable_marker,
    )
    current, target = clouds()
    runtime.latch_target(target)
    estimate = runtime.estimate(
        current_point_cloud=current, target_point_cloud=target
    )
    assert estimate.source_confidence == 1.0
    assert estimate.target_confidence == 0.0
    assert estimate.combined_confidence == 0.0


def test_target_frame_is_early_latched():
    encoder = StubFrameEncoder(np.eye(3))
    calls = []

    def estimate(cloud):
        calls.append(np.asarray(cloud).copy())
        transform = np.eye(4, dtype=np.float32)
        transform[:3, 3] = np.asarray(cloud)[:, :3].mean(axis=0)
        return transform, {"marker_points": 30, "fit_score_m2": 1.0e-6}

    runtime = OnlineNdfFunctionalFrameProvider(
        frame_encoders=[encoder, encoder],
        current_origin_offset_local3=[0.0, 0.0, 0.0],
        goal_position_offset_marker_local3=[0.0, 0.0, 0.0],
        goal_rotation_offset_marker_local9=np.eye(3),
        confidence_threshold_deg=1.0,
        marker_frame_estimator=estimate,
    )
    current, target = clouds()
    first = runtime.estimate(
        current_point_cloud=current, target_point_cloud=target
    )
    second = runtime.estimate(
        current_point_cloud=current,
        target_point_cloud=target + np.asarray([10, 0, 0, 0, 0, 0]),
    )
    assert len(calls) == 1
    np.testing.assert_array_equal(runtime.cached_target_cloud, target)
    np.testing.assert_allclose(first.frame9_metric, second.frame9_metric)
    runtime.reset()
    assert not runtime.target_latched


def test_zero_is_exact_and_skips_geometry_models():
    encoders = [StubFrameEncoder(np.eye(3)), StubFrameEncoder(np.eye(3))]
    runtime = provider(encoders)
    current, target = clouds()
    estimate = runtime.estimate(
        current_point_cloud=current,
        target_point_cloud=target,
        mode="zero",
    )
    np.testing.assert_array_equal(estimate.frame9_metric, np.zeros(9))
    assert estimate.combined_confidence == 0.0
    assert sum(item.calls for item in encoders) == 0
    assert not runtime.target_latched


def test_oracle_ceiling_is_disabled_by_default_and_explicit_when_enabled():
    encoders = [StubFrameEncoder(np.eye(3)), StubFrameEncoder(np.eye(3))]
    current, target = clouds()
    frame = np.concatenate(([0.1, -0.2, 0.3], np.eye(3)[:, 0], np.eye(3)[:, 1]))
    runtime = provider(encoders)
    with unittest.TestCase().assertRaises(PermissionError):
        runtime.estimate(
            current_point_cloud=current,
            target_point_cloud=target,
            mode="oracle",
            oracle_relative_frame9_metric=frame,
        )
    runtime = provider(encoders, allow_privileged_oracle=True)
    estimate = runtime.estimate(
        current_point_cloud=current,
        target_point_cloud=target,
        mode="oracle",
        oracle_relative_frame9_metric=frame,
    )
    np.testing.assert_allclose(estimate.frame9_metric, frame)
    assert estimate.combined_confidence == 1.0


class OnlineNdfFunctionalFrameTest(unittest.TestCase):
    def test_clean_contract(self):
        test_clean_matches_training_relative_frame_contract()

    def test_disagreement_mask(self):
        test_ensemble_disagreement_exactly_masks_frame()

    def test_medoid_aggregation(self):
        test_medoid_aggregation_rejects_one_flipped_member()

    def test_temporal_sign_jump(self):
        test_temporal_sign_jump_is_corrected_but_confidence_masked()

    def test_inverse_relation(self):
        test_inverse_preserves_confidence_and_magnitude_but_reverses_relation()

    def test_combined_confidence(self):
        test_combined_confidence_masks_unreliable_marker()

    def test_early_latch(self):
        test_target_frame_is_early_latched()

    def test_zero(self):
        test_zero_is_exact_and_skips_geometry_models()

    def test_oracle_guard(self):
        test_oracle_ceiling_is_disabled_by_default_and_explicit_when_enabled()

    def test_metadata_requires_checkpoint_count_to_match_selected_ensemble(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            camera = root / "camera.json"
            ensemble = root / "ensemble.json"
            camera.write_text(
                json.dumps(
                    {
                        "fold": 0,
                        "deployable": True,
                        "functional_frame_encoding": (
                            "camera_ndf_current_marker_goal_se3_columns_v1"
                        ),
                        "current_origin_offset_local3": [0.0, 0.0, 0.0],
                        "goal_position_offset_marker_local3": [0.0, 0.0, 0.0],
                        "goal_rotation_offset_marker_local9": np.eye(3).tolist(),
                    }
                ),
                encoding="utf-8",
            )
            ensemble.write_text(
                json.dumps(
                    {
                        "fold": 0,
                        "confidence_threshold_deg": 7.0,
                        "selection_top_k": 2,
                        "selected_seeds": [0, 1],
                        "frame_predictions": ["seed0.npz", "seed1.npz"],
                    }
                ),
                encoding="utf-8",
            )
            adapter_path = (
                "policy.GeometryFlow.online_ndf_functional_frame."
                "NdfFunctionalFrameAdapter"
            )
            with patch(adapter_path, side_effect=lambda *_args, **_kwargs: StubFrameEncoder(np.eye(3))):
                runtime = OnlineNdfFunctionalFrameProvider.from_metadata(
                    ndf_checkpoints=["seed0.pth", "seed1.pth"],
                    camera_metadata=camera,
                    ensemble_metadata=ensemble,
                )
            self.assertEqual(len(runtime.encoders), 2)
            with self.assertRaisesRegex(ValueError, "expects 2"):
                OnlineNdfFunctionalFrameProvider.from_metadata(
                    ndf_checkpoints=["seed0.pth", "seed1.pth", "seed2.pth"],
                    camera_metadata=camera,
                    ensemble_metadata=ensemble,
                )


if __name__ == "__main__":
    unittest.main()
