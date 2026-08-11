#!/usr/bin/env python3
"""Audit a collected dense-control batch and emit an immutable candidate manifest."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


EPISODE_PATTERN = re.compile(r"episode(\d+)\.hdf5$")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data-root", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--expected-episodes", type=int, default=None)
    return result


def episode_files(data_root: Path) -> list[tuple[int, Path]]:
    result = []
    for path in (data_root / "data").glob("episode*.hdf5"):
        match = EPISODE_PATTERN.fullmatch(path.name)
        if match:
            result.append((int(match.group(1)), path))
    return sorted(result)


def load_seed_list(data_root: Path) -> list[int]:
    return [int(value) for value in (data_root / "seed.txt").read_text().split()]


def arm_name(scene_info: dict, episode_index: int) -> str:
    value = str(scene_info[f"episode_{episode_index}"]["info"]["{a}"]).lower()
    if "left" in value:
        return "left"
    if "right" in value:
        return "right"
    raise ValueError(f"episode {episode_index} has unknown active arm {value!r}")


def audit_episode(
    episode_index: int,
    path: Path,
    *,
    seed: int,
    scene_info: dict,
) -> dict:
    errors = []
    with h5py.File(path, "r") as archive:
        required = (
            "dense_control/position",
            "dense_control/arm_velocity",
            "dense_control/observation_last_step_index",
            "task_state/shoe_id",
        )
        missing = [name for name in required if name not in archive]
        if missing:
            return {
                "episode_index": episode_index,
                "seed": seed,
                "path": str(path.resolve()),
                "structurally_valid": False,
                "errors": [f"missing datasets: {missing}"],
            }
        position = np.asarray(archive["dense_control/position"], dtype=np.float32)
        velocity = np.asarray(archive["dense_control/arm_velocity"], dtype=np.float32)
        observation_index = np.asarray(
            archive["dense_control/observation_last_step_index"], dtype=np.int64
        ).reshape(-1)
        shoe_ids = np.asarray(archive["task_state/shoe_id"], dtype=np.int64).reshape(-1)
    if position.ndim != 2 or position.shape[1] != 14:
        errors.append(f"dense position shape is {position.shape}, expected [N,14]")
    if velocity.ndim != 2 or velocity.shape[1] != 12:
        errors.append(f"dense velocity shape is {velocity.shape}, expected [N,12]")
    if len(position) != len(velocity):
        errors.append("dense position/velocity lengths differ")
    if not np.isfinite(position).all() or not np.isfinite(velocity).all():
        errors.append("dense controls contain non-finite values")
    if not len(observation_index):
        errors.append("observation-to-control index is empty")
    elif np.any(np.diff(observation_index) < 0):
        errors.append("observation-to-control index is not monotonic")
    elif int(observation_index[-1]) >= len(position):
        errors.append("final observation-to-control index exceeds dense trace")
    unique_shoes = np.unique(shoe_ids)
    if len(unique_shoes) != 1:
        errors.append(f"shoe id is not constant: {unique_shoes.tolist()}")
        shoe_id = None
    else:
        shoe_id = int(unique_shoes[0])
    return {
        "episode_index": int(episode_index),
        "seed": int(seed),
        "shoe_id": shoe_id,
        "active_arm": arm_name(scene_info, episode_index),
        "observations": int(len(observation_index)),
        "dense_controls": int(len(position)),
        "path": str(path.resolve()),
        "bytes": int(path.stat().st_size),
        "structurally_valid": not errors,
        "errors": errors,
    }


def main() -> None:
    args = parser().parse_args()
    data_root = args.data_root.expanduser().resolve()
    seeds = load_seed_list(data_root)
    scene_info = json.loads((data_root / "scene_info.json").read_text(encoding="utf-8"))
    files = episode_files(data_root)
    episodes = []
    for episode_index, path in files:
        if episode_index >= len(seeds):
            raise ValueError(f"episode {episode_index} has no seed entry")
        episodes.append(
            audit_episode(
                episode_index,
                path,
                seed=seeds[episode_index],
                scene_info=scene_info,
            )
        )
    valid = [item for item in episodes if item["structurally_valid"]]
    distribution = Counter(
        f"shoe{item['shoe_id']}_{item['active_arm']}" for item in valid
    )
    expected = args.expected_episodes
    manifest = {
        "schema_version": 1,
        "data_root": str(data_root),
        "expected_episodes": expected,
        "found_episodes": len(episodes),
        "structurally_valid_episodes": len(valid),
        "complete": expected is None or len(episodes) == expected,
        "distribution": dict(sorted(distribution.items())),
        "episodes": episodes,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "candidate_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "candidate_seeds.json").write_text(
        json.dumps(
            {"accepted_seeds": [item["seed"] for item in valid]}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: manifest[key] for key in manifest if key != "episodes"}))


if __name__ == "__main__":
    main()
