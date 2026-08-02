#!/usr/bin/env python3
"""Audit whether expert recovery rotation follows the supplied rigid flow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .train_flow_generator import rigid_rotation


def distribution(values: np.ndarray) -> dict[str, float | None]:
    value = np.asarray(values, dtype=np.float64)
    if not len(value):
        return {key: None for key in ("mean", "p50", "p90", "max")}
    return {
        "mean": float(value.mean()),
        "p50": float(np.quantile(value, 0.5)),
        "p90": float(np.quantile(value, 0.9)),
        "max": float(value.max()),
    }


def summarize(
    payload: dict[str, np.ndarray], mask: np.ndarray
) -> dict[str, object]:
    indices = np.flatnonzero(mask)
    endpoint = []
    for index in indices:
        episode = int(payload["episode_index"][index])
        endpoint.append(
            Rotation.from_matrix(
                rigid_rotation(
                    payload["current_anchors"][index],
                    payload["episode_flow"][episode, :, -1],
                )
            ).as_rotvec()
        )
    endpoint = np.asarray(endpoint, dtype=np.float64)
    action = np.asarray(payload["action"][indices], dtype=np.float64)
    active_right = np.asarray(payload["active_arm_right"][indices]) >= 0.5
    active = np.where(
        active_right[:, None, None], action[..., 7:14], action[..., :7]
    )
    action_rotation = active[..., 3:6]
    action_magnitude = np.linalg.norm(action_rotation, axis=-1)
    endpoint_magnitude = np.linalg.norm(endpoint, axis=-1)
    valid = (action_magnitude[:, 0] >= np.deg2rad(0.1)) & (
        endpoint_magnitude >= np.deg2rad(5.0)
    )
    cosine = np.sum(action_rotation[:, 0] * endpoint, axis=-1) / np.maximum(
        action_magnitude[:, 0] * endpoint_magnitude, 1e-8
    )
    return {
        "samples": int(len(indices)),
        "endpoint_rotation_deg": distribution(np.rad2deg(endpoint_magnitude)),
        "first_action_rotation_deg": distribution(
            np.rad2deg(action_magnitude[:, 0])
        ),
        "horizon_max_action_rotation_deg": distribution(
            np.rad2deg(action_magnitude.max(axis=1))
        ),
        "direction_valid_samples": int(valid.sum()),
        "first_action_to_flow_cosine": distribution(cosine[valid]),
        "fraction_positive_direction": (
            float(np.mean(cosine[valid] > 0.0)) if np.any(valid) else None
        ),
        "fraction_cosine_over_half": (
            float(np.mean(cosine[valid] > 0.5)) if np.any(valid) else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--new-recovery-episode-start", type=int, default=70)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.dataset, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    recovery = np.asarray(payload["is_recovery_sample"]) >= 0.5
    relation = np.asarray(payload["relation_phase"]) >= 0.5
    episode = np.asarray(payload["episode_index"], dtype=np.int64)
    groups = {
        "nominal": relation & ~recovery,
        "old_recovery": relation
        & recovery
        & (episode < int(args.new_recovery_episode_start)),
        "new_camera_relation_recovery": relation
        & recovery
        & (episode >= int(args.new_recovery_episode_start)),
    }
    result = {
        "dataset": str(args.dataset.resolve()),
        "new_recovery_episode_start": int(args.new_recovery_episode_start),
        "groups": {
            name: summarize(payload, mask) for name, mask in groups.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
