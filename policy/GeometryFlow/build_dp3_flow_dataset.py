#!/usr/bin/env python3
"""Add camera-reanchored predicted object-flow tracks to a DP3 replay buffer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import zarr

from .build_flow_prediction_dataset import valid_point_cloud
from .evaluate_flow_transport import align_pca_signs, pca_frame
from .target_frame import estimate_geometry_marker_frame
from .train_flow_generator import rigid_rotation, world_to_frame


FLOW_KEY = "object_flow"


def interpolate_flow(flow: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    value = np.asarray(flow, dtype=np.float64)
    query = np.clip(np.asarray(fractions, dtype=np.float64), 0.0, 1.0)
    position = query * (value.shape[1] - 1)
    lower = np.floor(position).astype(np.int64)
    upper = np.minimum(lower + 1, value.shape[1] - 1)
    weight = position - lower
    return (
        value[:, lower] * (1.0 - weight)[None, :, None]
        + value[:, upper] * weight[None, :, None]
    )


def camera_reanchored_anchors(
    initial_cloud: np.ndarray, current_cloud: np.ndarray, initial_anchors: np.ndarray
) -> np.ndarray:
    initial_center, initial_basis, initial_coordinates = pca_frame(initial_cloud)
    current_center, current_basis, current_coordinates = pca_frame(current_cloud)
    _, current_basis = align_pca_signs(
        initial_coordinates, current_coordinates, current_basis
    )
    intrinsic = (np.asarray(initial_anchors) - initial_center) @ initial_basis
    return intrinsic @ current_basis.T + current_center


def remaining_rigid_flow(
    episode_flow: np.ndarray,
    current_anchors: np.ndarray,
    progress: float,
    steps: int,
) -> np.ndarray:
    future = interpolate_flow(
        episode_flow, np.linspace(float(progress), 1.0, int(steps))
    )
    reference = future[:, 0]
    reference_center = reference.mean(axis=0)
    current = np.asarray(current_anchors, dtype=np.float64)
    current_center = current.mean(axis=0)
    local = current - current_center
    output = []
    for step in range(future.shape[1]):
        rotation = rigid_rotation(reference, future[:, step])
        center_delta = future[:, step].mean(axis=0) - reference_center
        output.append(local @ rotation.T + current_center + center_delta)
    result = np.stack(output, axis=1)
    result[:, 0] = current
    return result


def flow_track_in_target_frame(
    episode_flow: np.ndarray,
    initial_cloud: np.ndarray,
    current_cloud: np.ndarray,
    target_cloud_xyzrgb: np.ndarray,
    progress: float,
    target_frame: np.ndarray | None = None,
) -> np.ndarray:
    current_anchors = camera_reanchored_anchors(
        initial_cloud, current_cloud, episode_flow[:, 0]
    )
    remaining = remaining_rigid_flow(
        episode_flow, current_anchors, progress, episode_flow.shape[1]
    )
    if target_frame is None:
        target_frame, _ = estimate_geometry_marker_frame(target_cloud_xyzrgb)
    anchors_target = world_to_frame(current_anchors, target_frame)
    remaining_target = world_to_frame(remaining, target_frame)
    return np.concatenate(
        (anchors_target, (remaining_target - anchors_target[:, None]).reshape(len(current_anchors), -1)),
        axis=-1,
    ).astype(np.float32)


def load_predictions(path: Path, episodes: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if "predicted_flow" in archive:
            predicted = archive["predicted_flow"].astype(np.float32)
        elif "episode_flow" in archive:
            predicted = archive["episode_flow"].astype(np.float32)
        elif "target_flow" in archive:
            predicted = archive["target_flow"].astype(np.float32)
        else:
            raise KeyError(
                "prediction archive must contain predicted_flow, episode_flow, or target_flow"
            )
        indices = (
            archive["episode_indices"].astype(np.int64)
            if "episode_indices" in archive
            else np.arange(len(predicted), dtype=np.int64)
        )
    if set(indices.tolist()) != set(range(int(episodes))):
        raise ValueError("prediction archive must contain one flow for every episode")
    ordered = np.empty_like(predicted)
    ordered[indices] = predicted
    return ordered


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--base-zarr", type=Path, required=True)
    result.add_argument("--base-meta", type=Path, required=True)
    result.add_argument("--raw-data-dir", type=Path, required=True)
    result.add_argument("--prediction-dataset", type=Path, required=True)
    result.add_argument("--flow-predictions", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--flow-key", default=FLOW_KEY)
    result.add_argument("--zero-flow", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    metadata = json.loads(args.base_meta.read_text(encoding="utf-8"))
    episode_rows = metadata["episodes"]
    with np.load(args.prediction_dataset, allow_pickle=False) as archive:
        initial_clouds = np.asarray(archive["points_a_xyzrgb"][:, :, :3], dtype=np.float32)
    predictions = load_predictions(args.flow_predictions, len(episode_rows))

    tracks = []
    diagnostics = []
    for episode, row in enumerate(episode_rows):
        frames = [int(value) for value in row["source_frame_indices"]]
        with h5py.File(args.raw_data_dir / f"episode{episode}.hdf5", "r") as root:
            frame_count = int(root["object_pointcloud/{A}"].shape[0])
            initial_b = valid_point_cloud(root["object_pointcloud/{B}"][0])
            cached_target_frame, target_diagnostic = estimate_geometry_marker_frame(initial_b)
            for frame in frames:
                current_a = valid_point_cloud(root["object_pointcloud/{A}"][frame])
                progress = frame / max(frame_count - 1, 1)
                track = flow_track_in_target_frame(
                    predictions[episode],
                    initial_clouds[episode],
                    current_a[:, :3],
                    initial_b,
                    progress,
                    target_frame=cached_target_frame,
                )
                if args.zero_flow:
                    track = np.zeros_like(track)
                if not np.all(np.isfinite(track)):
                    raise RuntimeError(f"non-finite flow track in episode={episode}, frame={frame}")
                tracks.append(track)
        diagnostics.append(
            {
                "episode": episode,
                "frames": len(frames),
                "first_frame": frames[0],
                "last_frame": frames[-1],
                "cached_target_fit_score_m2": float(target_diagnostic["fit_score_m2"]),
            }
        )
        print(json.dumps(diagnostics[-1]), flush=True)

    source = zarr.open(str(args.base_zarr.resolve()), mode="r")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    destination = zarr.open(str(args.output.resolve()), mode="w")
    zarr.copy_store(source.store, destination.store, if_exists="replace")
    flow_value = np.asarray(tracks, dtype=np.float32)
    expected_frames = int(source["meta/episode_ends"][-1])
    if len(flow_value) != expected_frames:
        raise RuntimeError(f"built {len(flow_value)} flow frames, expected {expected_frames}")
    destination["data"].create_dataset(
        args.flow_key,
        data=flow_value,
        chunks=(min(100, len(flow_value)),) + flow_value.shape[1:],
        compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=1),
        overwrite=False,
    )
    output_meta = {
        "schema_version": 1,
        "base_zarr": str(args.base_zarr.resolve()),
        "prediction_dataset": str(args.prediction_dataset.resolve()),
        "flow_predictions": str(args.flow_predictions.resolve()),
        "output": str(args.output.resolve()),
        "flow_key": args.flow_key,
        "flow_shape": list(flow_value.shape[1:]),
        "frames": int(len(flow_value)),
        "episodes": int(len(episode_rows)),
        "zero_flow": bool(args.zero_flow),
        "camera_only_reanchoring": True,
        "target_frame_source": "segmented B XYZRGB marker",
        "diagnostics": diagnostics,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(output_meta, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: output_meta[key] for key in ("output", "flow_shape", "frames", "zero_flow")}, indent=2))


if __name__ == "__main__":
    main()
