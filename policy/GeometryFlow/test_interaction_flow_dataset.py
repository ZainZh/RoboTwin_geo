from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .functional_action_frame import frame9_rotation
from .train_interaction_flow_tokens import (
    InteractionFlowDataset,
    oracle_relative_frame9,
    policy_phase_keep_mask,
)
from .train_task_flow_benchmark import Normalization


def _frame(translation: list[float], rotation_vector: list[float]) -> np.ndarray:
    rotation = Rotation.from_rotvec(rotation_vector).as_matrix()
    return np.concatenate((translation, rotation[:, 0], rotation[:, 1])).astype(
        np.float32
    )


def _payload() -> dict[str, np.ndarray]:
    generator = np.random.default_rng(7)
    identity_pose9 = np.asarray(
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32
    )
    anchors = generator.normal(size=(4, 3)).astype(np.float32)
    flow = np.stack(
        (
            anchors,
            anchors + np.asarray([0.01, 0.0, 0.0], dtype=np.float32),
            anchors + np.asarray([0.02, 0.0, 0.0], dtype=np.float32),
        ),
        axis=1,
    )
    frames = np.stack(
        (
            _frame([1.0, 0.0, 0.0], [0.0, 0.0, 0.1]),
            _frame([0.5, 0.0, 0.0], [0.0, 0.0, 0.2]),
            _frame([0.0, 1.0, 0.0], [0.1, 0.0, 0.0]),
            _frame([0.0, 0.5, 0.0], [0.2, 0.0, 0.0]),
        )
    )
    return {
        "points_a": generator.normal(size=(4, 6, 3)).astype(np.float32),
        "points_b": generator.normal(size=(4, 6, 3)).astype(np.float32),
        "state": np.zeros((4, 20), dtype=np.float32),
        "action": np.zeros((4, 2, 14), dtype=np.float32),
        "current_anchors": np.repeat(anchors[None], 4, axis=0),
        "current_object_pose9": np.repeat(identity_pose9[None], 4, axis=0),
        "active_arm_right": np.ones(4, dtype=np.float32),
        "relation_phase": np.ones(4, dtype=np.float32),
        "episode_index": np.asarray([0, 0, 1, 1], dtype=np.int64),
        "frame_index": np.asarray([0, 1, 0, 1], dtype=np.int64),
        "shoe_id": np.asarray([0, 0, 1, 1], dtype=np.int64),
        "episode_shoe_id": np.asarray([0, 1], dtype=np.int64),
        "episode_flow": np.repeat(flow[None], 2, axis=0),
        "episode_pose": np.repeat(identity_pose9[None, None], 6, axis=0).reshape(
            2, 3, 9
        ),
        "target_frame9": frames,
        "source_frame6_columns": np.repeat(
            np.asarray([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], dtype=np.float32),
            4,
            axis=0,
        ),
    }


def _normalization() -> Normalization:
    return Normalization(
        state_mean=np.zeros(20, dtype=np.float32),
        state_std=np.ones(20, dtype=np.float32),
        action_mean=np.zeros(14, dtype=np.float32),
        action_std=np.ones(14, dtype=np.float32),
        xyz_mean=np.zeros(3, dtype=np.float32),
        xyz_std=np.ones(3, dtype=np.float32),
    )


def test_dynamic_functional_frame_uses_current_sample() -> None:
    payload = _payload()
    dataset = InteractionFlowDataset(
        payload,
        np.arange(4),
        _normalization(),
        flow_steps=3,
        target_frame_mode="normal",
        functional_frame_translation_mode="delta",
    )
    np.testing.assert_allclose(dataset[0]["target_frame9"], payload["target_frame9"][0])
    np.testing.assert_allclose(dataset[1]["target_frame9"], payload["target_frame9"][1])
    assert not np.allclose(dataset[0]["target_frame9"], dataset[1]["target_frame9"])
    assert float(dataset[0]["target_frame_confidence"]) == 1.0
    np.testing.assert_allclose(
        dataset[0]["source_frame6_columns"],
        payload["source_frame6_columns"][0],
    )
    np.testing.assert_allclose(
        dataset[0]["target_frame9_source_local"],
        payload["target_frame9"][0],
        atol=1e-6,
    )


def test_zero_frame_also_zeros_confidence() -> None:
    payload = _payload()
    payload["target_frame_confidence"] = np.ones(4, dtype=np.float32)
    dataset = InteractionFlowDataset(
        payload,
        np.arange(4),
        _normalization(),
        flow_steps=3,
        target_frame_mode="zero",
        functional_frame_translation_mode="delta",
    )
    assert np.allclose(dataset[0]["target_frame9"], 0.0)
    assert float(dataset[0]["target_frame_confidence"]) == 0.0


def test_source_local_frame_can_be_zeroed_independently() -> None:
    payload = _payload()
    dataset = InteractionFlowDataset(
        payload,
        np.arange(4),
        _normalization(),
        flow_steps=3,
        target_frame_mode="normal",
        source_local_frame_mode="zero",
        functional_frame_translation_mode="delta",
    )
    assert not np.allclose(dataset[0]["target_frame9"], 0.0)
    assert np.allclose(dataset[0]["target_frame9_source_local"], 0.0)


def test_wrong_frame_matches_progress_and_error_magnitude() -> None:
    payload = _payload()
    dataset = InteractionFlowDataset(
        payload,
        np.arange(4),
        _normalization(),
        flow_steps=3,
        target_frame_mode="shuffled",
        functional_frame_translation_mode="delta",
    )
    np.testing.assert_allclose(dataset[0]["target_frame9"], payload["target_frame9"][2])
    np.testing.assert_allclose(dataset[1]["target_frame9"], payload["target_frame9"][3])


def test_oracle_frame_uses_intended_goal_error_labels() -> None:
    translation = np.asarray([[0.1, -0.2, 0.3], [-0.4, 0.0, 0.2]], dtype=np.float32)
    rotation_vector = np.asarray(
        [[0.0, 0.0, 0.3], [0.2, -0.1, 0.0]], dtype=np.float32
    )
    payload = {
        "goal_translation_error_xyz_m": translation,
        "goal_rotation_error_rotvec": rotation_vector,
        # Deliberately nonsensical expert endpoints: they must be ignored.
        "episode_pose": np.full((2, 3, 9), 99.0, dtype=np.float32),
    }
    frame = oracle_relative_frame9(payload)
    np.testing.assert_allclose(frame[:, :3], translation, atol=1e-7)
    np.testing.assert_allclose(
        frame9_rotation(frame), Rotation.from_rotvec(rotation_vector).as_matrix(), atol=1e-6
    )


def test_oracle_frame_refuses_expert_endpoint_fallback() -> None:
    try:
        oracle_relative_frame9({"episode_pose": np.zeros((1, 2, 9), dtype=np.float32)})
    except KeyError as error:
        assert "refusing to fall back" in str(error)
    else:
        raise AssertionError("expected missing intended-goal labels to fail")


def test_pre_release_phase_excludes_post_release_relation_rows() -> None:
    payload = {
        "shoe_id": np.zeros(5, dtype=np.int64),
        "relation_phase": np.asarray([0, 1, 1, 1, 0], dtype=np.float32),
        "steps_to_release": np.asarray([4, 3, 1, 0, 0], dtype=np.float32),
    }
    np.testing.assert_array_equal(
        policy_phase_keep_mask(payload, "pre_release"),
        np.asarray([False, True, True, False, False]),
    )


def test_pre_release_phase_requires_release_distance() -> None:
    payload = {
        "shoe_id": np.zeros(2, dtype=np.int64),
        "relation_phase": np.ones(2, dtype=np.float32),
    }
    try:
        policy_phase_keep_mask(payload, "pre_release")
    except ValueError as error:
        assert "steps_to_release" in str(error)
    else:
        raise AssertionError("expected missing steps_to_release to fail")
