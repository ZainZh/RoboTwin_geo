#!/usr/bin/env python3
"""Freeze the one-reference camera calibration used by hanging-mug runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from .build_grasp_dataset import valid_xyz
from .functional_action_frame import pose9_rotation


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", type=Path, required=True)
    result.add_argument("--data-dir", type=Path, required=True)
    result.add_argument("--camera-metadata", type=Path, required=True)
    result.add_argument("--ensemble-metadata", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--overwrite", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    camera = json.loads(args.camera_metadata.read_text(encoding="utf-8"))
    ensemble = json.loads(args.ensemble_metadata.read_text(encoding="utf-8"))
    reference_episode = int(camera["reference_episode"])
    if not bool(camera.get("deployable_rotation", False)):
        raise ValueError("camera metadata is not marked deployable for rotation")
    with np.load(args.dataset, allow_pickle=False) as archive:
        episode_ids = np.asarray(archive["episode_id"], dtype=np.int64)
        current_pose9 = np.asarray(archive["current_object_pose9"])
        goal_rotvec = np.asarray(archive["goal_rotation_error_rotvec"])
    rows = np.flatnonzero(episode_ids == reference_episode)
    if not len(rows):
        raise ValueError(f"reference episode {reference_episode} is absent")
    row = int(rows[0])
    current_rotation = pose9_rotation(current_pose9)[row]
    reference_goal_rotation = (
        Rotation.from_rotvec(goal_rotvec[row]).as_matrix() @ current_rotation
    )
    hdf5_path = args.data_dir / f"episode{reference_episode}.hdf5"
    with h5py.File(hdf5_path, "r") as root:
        reference_cloud = valid_xyz(root["object_pointcloud/{B}"][0])
    registration_points = 512
    if len(reference_cloud) > registration_points:
        indices = np.linspace(
            0, len(reference_cloud) - 1, registration_points, dtype=np.int64
        )
        reference_cloud = reference_cloud[indices]
    threshold = float(camera["diagnostics"]["registration_train_p95_m"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        reference_target_cloud=np.asarray(reference_cloud, dtype=np.float32),
        reference_goal_rotation=np.asarray(reference_goal_rotation, dtype=np.float32),
        registration_confidence_threshold_m=np.float32(threshold),
        registration_points=np.int64(registration_points),
        icp_iterations=np.int64(30),
        icp_trim_fraction=np.float32(0.8),
        temporal_maximum_step_deg=np.float32(
            camera["temporal_maximum_step_deg"]
        ),
        temporal_minimum_margin_deg=np.float32(
            camera["temporal_minimum_margin_deg"]
        ),
    )
    metadata = {
        "schema_version": 1,
        "method": "hanging_mug_single_train_reference_camera_rotation_calibration",
        "dataset": str(args.dataset.resolve()),
        "reference_hdf5": str(hdf5_path.resolve()),
        "reference_episode": reference_episode,
        "reference_object_id": int(camera["diagnostics"]["reference_object_id"]),
        "registration_confidence_threshold_m": threshold,
        "source_confidence_threshold_deg": float(
            ensemble["confidence_threshold_deg"]
        ),
        "translation_token": "exact zero",
        "uses_test_labels": False,
        "uses_simulator_state_at_runtime": False,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), **metadata}, indent=2))


if __name__ == "__main__":
    main()
