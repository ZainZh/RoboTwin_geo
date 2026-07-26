#!/usr/bin/env python3
"""Build a point-cloud-free mean-goal ablation from training episodes."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from data_path_utils import raw_task_data_dir
from object_pointcloud_utils import load_hdf5


def _episode_index(path: Path) -> int:
    match = re.fullmatch(r"episode(\d+)\.hdf5", path.name)
    if match is None:
        raise ValueError(f"unexpected episode filename {path.name!r}")
    return int(match.group(1))


def _project_to_so3(matrix: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.asarray(matrix, dtype=np.float64).reshape(3, 3))
    correction = np.eye(3, dtype=np.float64)
    correction[2, 2] = np.linalg.det(u @ vt)
    return u @ correction @ vt


def _rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first @ second.T
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Average training-set oracle T_A_from_B without reading point clouds "
            "or object IDs. The result is the constant-goal control ablation."
        )
    )
    parser.add_argument("task_name")
    parser.add_argument("task_config")
    parser.add_argument("expert_data_num", type=int)
    parser.add_argument("--output", required=True)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    load_dir = raw_task_data_dir(args.task_name, args.task_config)
    episode_paths = sorted(
        (load_dir / "data").glob("episode*.hdf5"), key=_episode_index
    )
    episode_paths = [
        path for path in episode_paths if _episode_index(path) < args.expert_data_num
    ]
    if len(episode_paths) != int(args.expert_data_num):
        raise RuntimeError(
            f"expected {args.expert_data_num} episodes, found {len(episode_paths)} "
            f"under {load_dir / 'data'}"
        )

    goals = []
    episodes = []
    for path in episode_paths:
        episode = load_hdf5(str(path))
        task_state = episode.get("task_state")
        if not isinstance(task_state, dict) or "goal_T_A_from_B_oracle" not in task_state:
            raise RuntimeError(
                f"{path} is missing task_state/goal_T_A_from_B_oracle"
            )
        values = np.asarray(
            task_state["goal_T_A_from_B_oracle"], dtype=np.float64
        ).reshape(-1, 4, 4)
        # Oracle relation is episode-constant. Taking one value gives every
        # episode equal weight regardless of trajectory length.
        goals.append(values[0])
        episodes.append(_episode_index(path))

    goals_array = np.stack(goals)
    mean_goal = np.eye(4, dtype=np.float64)
    mean_goal[:3, :3] = _project_to_so3(goals_array[:, :3, :3].mean(axis=0))
    mean_goal[:3, 3] = goals_array[:, :3, 3].mean(axis=0)
    translation_errors = np.linalg.norm(
        goals_array[:, :3, 3] - mean_goal[:3, 3][None, :], axis=1
    )
    rotation_errors = np.asarray(
        [
            _rotation_error_deg(goal[:3, :3], mean_goal[:3, :3])
            for goal in goals_array
        ],
        dtype=np.float64,
    )

    result = {
        "schema_version": 1,
        "type": "constant_goal",
        "goal_T_A_from_B": mean_goal.tolist(),
        "source": {
            "task_name": args.task_name,
            "task_config": args.task_config,
            "expert_data_num": int(args.expert_data_num),
            "episodes": episodes,
            "uses_pointcloud": False,
            "uses_object_id": False,
        },
        "training_set_dispersion": {
            "translation_mean_m": float(translation_errors.mean()),
            "translation_max_m": float(translation_errors.max()),
            "rotation_mean_deg": float(rotation_errors.mean()),
            "rotation_max_deg": float(rotation_errors.max()),
        },
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"wrote constant-goal estimator: {output}")


if __name__ == "__main__":
    main()
