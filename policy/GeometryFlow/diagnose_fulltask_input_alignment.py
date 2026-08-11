#!/usr/bin/env python3
"""Compare one archived full-task input against deployment preprocessing.

This is an interface diagnostic.  It deliberately injects the archived NDF
frame so point sampling and proprioception can be checked independently of the
online frame estimator.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from .deploy_fulltask_tagrt_v2 import FullTaskTAGRTV2Runtime
from .functional_action_frame import frame9_rotation


class ArchivedFrameProvider:
    def __init__(self, relative: np.ndarray, goal: np.ndarray) -> None:
        self.relative = np.asarray(relative, dtype=np.float32)
        self.goal_transform = np.eye(4, dtype=np.float32)
        self.goal_transform[:3, :3] = frame9_rotation(goal)
        self.goal_transform[:3, 3] = np.asarray(goal[:3], dtype=np.float32)

    def reset(self) -> None:
        return None

    def estimate(self, **_) -> SimpleNamespace:
        return SimpleNamespace(
            goal_transform=self.goal_transform,
            frame9_metric=self.relative,
            combined_confidence=1.0,
            source_disagreement_deg=0.0,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hdf5", required=True)
    parser.add_argument("--episode", type=int, required=True)
    args = parser.parse_args()

    with np.load(args.dataset, allow_pickle=False) as archive:
        dataset = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(args.source_dataset, allow_pickle=False) as archive:
        source = {key: np.asarray(archive[key]) for key in archive.files}
    row = int(np.flatnonzero(dataset["episode_id"] == args.episode)[0])
    source_row = int(np.flatnonzero(source["episode_id"] == args.episode)[0])
    provider = ArchivedFrameProvider(
        source["target_frame9"][source_row], source["goal_frame9"][source_row]
    )
    with h5py.File(args.hdf5, "r") as root:
        observation = {
            "object_pointcloud": {
                "{A}": np.asarray(root["object_pointcloud/{A}"][0]),
                "{B}": np.asarray(root["object_pointcloud/{B}"][0]),
            },
            "endpose": {
                key: np.asarray(root[f"endpose/{key}"][0])
                for key in root["endpose"]
            },
        }
    runtime = FullTaskTAGRTV2Runtime(
        args.checkpoint, frame_provider=provider, device="cpu"
    )
    frame = runtime._normalized_frame(observation)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    statistics = checkpoint["statistics"]
    fields = (
        ("scene", "point_mean", "point_std"),
        ("points_a", "point_mean", "point_std"),
        ("points_b", "point_mean", "point_std"),
        ("local", "local_mean", "local_std"),
        ("global", "global_mean", "global_std"),
        ("state", "state_mean", "state_std"),
    )
    for key, mean_key, std_key in fields:
        deployed = (
            frame[key] * np.asarray(statistics[std_key])
            + np.asarray(statistics[mean_key])
        )
        archived = dataset[key][row, -1]
        print(
            f"{key}: mean_absolute_input_difference="
            f"{float(np.mean(np.abs(deployed - archived))):.8f}"
        )
    runtime.reset()
    prediction = runtime.predict(observation)
    print("deployment_preprocessed_first_action14:", prediction[0].tolist())
    print("archived_target_first_action14:", dataset["action"][row, 0].tolist())


if __name__ == "__main__":
    main()
