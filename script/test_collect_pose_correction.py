from __future__ import annotations

import numpy as np

from script.collect_pose_correction import (
    parse_levels,
    perturb_pose,
    perturbation_seed_for_scene,
)


def test_parse_pose_correction_levels() -> None:
    levels = parse_levels("1:3:5:20,3:6:20:45")
    assert levels == (((0.01, 0.03), (5.0, 20.0)), ((0.03, 0.06), (20.0, 45.0)))


def test_pose_correction_perturbation_respects_bounds_and_moves_outward() -> None:
    final = np.asarray((0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
    pre = np.asarray((0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0))
    perturbed, diagnostic = perturb_pose(
        final,
        pre,
        (0.03, 0.06),
        (20.0, 45.0),
        np.random.default_rng(7),
    )
    assert 0.03 <= diagnostic["translation_m"] <= 0.06
    assert 20.0 <= diagnostic["rotation_deg"] <= 45.0
    assert perturbed[2] > 0.0
    assert abs(float(np.linalg.norm(perturbed[3:])) - 1.0) < 1e-6


def test_pose_correction_perturbation_is_exactly_replayable_per_scene() -> None:
    final = np.asarray((0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
    pre = np.asarray((0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0))
    seed = perturbation_seed_for_scene(20260805, 173)
    first = perturb_pose(
        final,
        pre,
        (0.01, 0.03),
        (5.0, 20.0),
        np.random.default_rng(seed),
    )
    replay = perturb_pose(
        final,
        pre,
        (0.01, 0.03),
        (5.0, 20.0),
        np.random.default_rng(seed),
    )
    assert np.array_equal(first[0], replay[0])
    assert first[1] == replay[1]
