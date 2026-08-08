from __future__ import annotations

import numpy as np

from .evaluate_axis_counterfactual import assert_counterfactual_alignment


def payload() -> dict[str, np.ndarray]:
    return {
        "episode_index": np.array([0, 1]),
        "frame_index": np.array([2, 3]),
        "shoe_id": np.array([0, 5]),
        "state": np.zeros((2, 20), dtype=np.float32),
        "action": np.zeros((2, 6, 14), dtype=np.float32),
        "relation_phase": np.ones(2, dtype=np.float32),
        "points_a": np.zeros((2, 4, 6), dtype=np.float32),
        "points_b": np.zeros((2, 4, 6), dtype=np.float32),
        "source_axis3": np.zeros((2, 3), dtype=np.float32),
    }


def test_counterfactual_alignment_allows_only_feature_changes() -> None:
    reference = payload()
    candidate = {key: value.copy() for key, value in reference.items()}
    candidate["source_axis3"][:] = 1.0
    candidate["points_a"][..., 3:] = 2.0
    assert_counterfactual_alignment(reference, candidate)


def test_counterfactual_alignment_rejects_xyz_changes() -> None:
    reference = payload()
    candidate = {key: value.copy() for key, value in reference.items()}
    candidate["points_a"][0, 0, 0] = 1.0
    try:
        assert_counterfactual_alignment(reference, candidate)
    except ValueError as error:
        assert "protected XYZ" in str(error)
    else:
        raise AssertionError("XYZ change was not rejected")
