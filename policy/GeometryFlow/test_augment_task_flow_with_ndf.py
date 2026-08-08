from __future__ import annotations

import numpy as np

from .augment_task_flow_with_ndf import (
    deterministic_isotropic_axis,
    episode_shuffle,
    mean_vector_axis,
    point_shuffle,
    select_descriptor,
)


def test_isotropic_axis_is_unit_length_and_episode_constant() -> None:
    episode = np.array([0, 0, 1, 1, 3], dtype=np.int64)
    first = deterministic_isotropic_axis(episode)
    second = deterministic_isotropic_axis(episode)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(np.linalg.norm(first, axis=-1), 1.0, atol=1e-6)
    np.testing.assert_array_equal(first[0], first[1])
    np.testing.assert_array_equal(first[2], first[3])
    assert not np.array_equal(first[0], first[2])


def test_point_shuffle_preserves_per_sample_descriptor_multiset() -> None:
    features = np.arange(2 * 5 * 3, dtype=np.float32).reshape(2, 5, 3)
    shuffled = point_shuffle(features, np.array([0, 1]), np.array([3, 7]))
    assert not np.array_equal(features, shuffled)
    for source, target in zip(features, shuffled):
        assert np.array_equal(np.sort(source, axis=0), np.sort(target, axis=0))


def test_episode_shuffle_uses_a_different_shoe() -> None:
    features = np.stack(
        [np.full((4, 3), value, dtype=np.float32) for value in range(4)]
    )
    episode = np.array([0, 0, 1, 1])
    frame = np.array([0, 3, 0, 4])
    shuffled = episode_shuffle(features, episode, frame, np.array([2, 7]))
    assert np.all(shuffled[:2] >= 2.0)
    assert np.all(shuffled[2:] < 2.0)


def test_projective_vector_is_sign_invariant() -> None:
    features = np.zeros((2, 4, 259), dtype=np.float32)
    features[0, :, -3:] = np.array([0.2, -0.5, 0.7])
    features[1] = features[0]
    features[1, :, -3:] *= -1.0
    projective = select_descriptor(features, "projective_vector")
    assert projective.shape == (2, 4, 6)
    assert np.array_equal(projective[0], projective[1])


def test_mean_vector_axis_is_signed_normalized_and_zero_safe() -> None:
    features = np.zeros((2, 4, 259), dtype=np.float32)
    features[0, :, -3:] = np.asarray([0.0, 0.0, 2.0], dtype=np.float32)
    axis = mean_vector_axis(features)
    np.testing.assert_allclose(axis[0], [0.0, 0.0, 1.0])
    np.testing.assert_array_equal(axis[1], np.zeros(3, dtype=np.float32))
