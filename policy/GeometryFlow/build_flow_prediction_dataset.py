#!/usr/bin/env python3
"""Build camera-only inputs for predicting an episode-level 3D object flow.

The future object track is privileged supervision constructed by
``build_task_flow_dataset.py``.  Model inputs contain only the initial
segmented camera point clouds.  Frozen NDF features are evaluated on the same
sampled operated-object points as the raw/PCA controls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from .build_grasp_dataset import identify_active_arm
from .evaluate_demo_transport import pca_canonical_xyz
from .geometry import pose7_wxyz_to_matrix
from .target_frame import estimate_geometry_marker_frame


DEFAULT_DATA_DIR = Path(
    "/shared2/sz/robotwin_data/data/place_shoe_geometry_marker/"
    "demo_clean_3d_object_pc_geometry_marker/data"
)


def farthest_point_indices(points: np.ndarray, count: int) -> np.ndarray:
    """Deterministic FPS matching ``deterministic_farthest_points``."""
    value = np.asarray(points, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != 3 or len(value) == 0:
        raise ValueError(f"expected non-empty (N,3) points, got {value.shape}")
    if len(value) < count:
        return np.arange(int(count), dtype=np.int64) % len(value)
    selected = np.zeros((int(count),), dtype=np.int64)
    minimum_distance = np.full((len(value),), np.inf, dtype=np.float32)
    center = value.mean(axis=0)
    farthest = int(np.argmax(np.sum((value - center) ** 2, axis=1)))
    for index in range(int(count)):
        selected[index] = farthest
        distance = np.sum((value - value[farthest]) ** 2, axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
        farthest = int(np.argmax(minimum_distance))
    return selected


def valid_point_cloud(point_cloud: np.ndarray) -> np.ndarray:
    value = np.asarray(point_cloud, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] < 6:
        raise ValueError(f"expected point cloud (N,C>=6), got {value.shape}")
    keep = np.isfinite(value[:, :6]).all(axis=1)
    keep &= ~np.isclose(value[:, :3], 0.0).all(axis=1)
    return value[keep, :6]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    result.add_argument("--oracle-flow", type=Path, required=True)
    result.add_argument(
        "--ndf-cache",
        type=Path,
        help="Optional point-aligned NDF cache; omit for a raw/PCA-only library.",
    )
    result.add_argument("--episodes", type=int, default=50)
    result.add_argument("--object-points", type=int, default=256)
    result.add_argument("--target-points", type=int, default=1024)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--overwrite", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {output}")
    with np.load(args.oracle_flow, allow_pickle=False) as archive:
        oracle = {key: archive[key] for key in archive.files}
    ndf = None
    ndf_by_episode = None
    if args.ndf_cache is not None:
        with np.load(args.ndf_cache, allow_pickle=False) as archive:
            ndf = {key: archive[key] for key in archive.files}
        ndf_by_episode = {
            int(episode): index
            for index, episode in enumerate(ndf["episode_id"].tolist())
        }
    episode_flow = np.asarray(oracle["episode_flow"], dtype=np.float32)
    if len(episode_flow) < int(args.episodes):
        raise ValueError("oracle flow contains fewer episodes than requested")

    arrays: dict[str, list[np.ndarray | int | float]] = {
        "points_a_xyzrgb": [],
        "points_b_xyzrgb": [],
        "pca_features": [],
        "initial_anchors": [],
        "episode_flow": [],
        "episode_id": [],
        "shoe_id": [],
        "active_arm_right": [],
        "target_frame_camera": [],
        "marker_params_label_only": [],
    }
    if ndf is not None:
        arrays["ndf_features"] = []
    diagnostics = []
    for episode in range(int(args.episodes)):
        path = args.data_dir / f"episode{episode}.hdf5"
        with h5py.File(path, "r") as root:
            full_a = valid_point_cloud(root["object_pointcloud/{A}"][0])
            full_b = valid_point_cloud(root["object_pointcloud/{B}"][0])
            a_indices = farthest_point_indices(full_a[:, :3], int(args.object_points))
            points_a = full_a[a_indices]
            if len(full_b) >= int(args.target_points):
                # B is already a deterministic segmented cloud.  Retaining its
                # order avoids throwing away the sparse high-contrast marker.
                points_b = full_b[: int(args.target_points)]
            else:
                repeat = np.arange(int(args.target_points)) % len(full_b)
                points_b = full_b[repeat]
            ndf_alignment = None
            cache_index = None
            if ndf is not None:
                cache_index = ndf_by_episode[episode]
                cache_points = np.asarray(
                    ndf["points_world"][cache_index], dtype=np.float32
                )
                if cache_points.shape != points_a[:, :3].shape:
                    raise ValueError(f"episode {episode}: NDF/raw point shape mismatch")
                ndf_alignment = float(
                    np.max(np.abs(cache_points - points_a[:, :3]))
                )
                if ndf_alignment > 2e-6:
                    raise ValueError(
                        f"episode {episode}: NDF/raw sampled points differ by "
                        f"{ndf_alignment:.3g} m"
                    )
            anchors = episode_flow[episode, :, 0]
            anchor_alignment = float(
                np.max(np.abs(anchors - points_a[: len(anchors), :3]))
            )
            if anchor_alignment > 2e-6:
                raise ValueError(
                    f"episode {episode}: flow anchors/FPS points differ by "
                    f"{anchor_alignment:.3g} m"
                )
            active_arm, _ = identify_active_arm(root)
            active_right = float(active_arm == "right")
            shoe_id = int(np.asarray(root["task_state/shoe_id"][0]).reshape(-1)[0])
            if ndf is not None:
                if active_right != float(ndf["active_arm_right"][cache_index]):
                    raise ValueError(f"episode {episode}: active-arm cache mismatch")
                if shoe_id != int(ndf["shoe_id_label_only"][cache_index]):
                    raise ValueError(f"episode {episode}: shoe-ID cache mismatch")
            marker = np.asarray(
                root["task_state/anchor_geometry_params"][0], dtype=np.float32
            ).reshape(3)
            target_frame, target_diagnostic = estimate_geometry_marker_frame(full_b)
            marker_local = np.eye(4, dtype=np.float64)
            marker_local[:3, :3] = Rotation.from_euler("z", marker[2]).as_matrix()
            marker_local[:3, 3] = (marker[0], marker[1], 0.0)
            target_frame_label = (
                pose7_wxyz_to_matrix(root["task_state/object_pose_B"][0]) @ marker_local
            )
            target_translation_error = float(
                np.linalg.norm(target_frame[:3, 3] - target_frame_label[:3, 3])
            )
            target_rotation_error = float(
                np.rad2deg(
                    Rotation.from_matrix(
                        target_frame[:3, :3] @ target_frame_label[:3, :3].T
                    ).magnitude()
                )
            )
            color = points_b[:, 3:6]
            marker_mask = color[:, 0] + color[:, 1] - 2.0 * color[:, 2] > 0.65
            marker_count = int(marker_mask.sum())

            arrays["points_a_xyzrgb"].append(points_a.astype(np.float32))
            arrays["points_b_xyzrgb"].append(points_b.astype(np.float32))
            arrays["pca_features"].append(
                pca_canonical_xyz(points_a[:, :3]).astype(np.float32)
            )
            if ndf is not None:
                arrays["ndf_features"].append(
                    np.asarray(ndf["ndf_features"][cache_index], dtype=np.float32)
                )
            arrays["initial_anchors"].append(anchors.astype(np.float32))
            arrays["episode_flow"].append(episode_flow[episode].astype(np.float32))
            arrays["episode_id"].append(episode)
            arrays["shoe_id"].append(shoe_id)
            arrays["active_arm_right"].append(active_right)
            arrays["target_frame_camera"].append(target_frame.astype(np.float32))
            arrays["marker_params_label_only"].append(marker)
            diagnostics.append(
                {
                    "episode": episode,
                    "shoe_id": shoe_id,
                    "active_arm": "right" if active_right >= 0.5 else "left",
                    "marker_points_in_B": marker_count,
                    "ndf_raw_alignment_max_m": ndf_alignment,
                    "flow_anchor_alignment_max_m": anchor_alignment,
                    "target_frame_translation_error_m_label_only": target_translation_error,
                    "target_frame_rotation_error_deg_label_only": target_rotation_error,
                    "target_frame_fit_score_m2": target_diagnostic["fit_score_m2"],
                }
            )
            print(json.dumps(diagnostics[-1]), flush=True)

    integer_keys = {"episode_id", "shoe_id"}
    payload = {
        key: np.asarray(value, dtype=np.int64 if key in integer_keys else np.float32)
        for key, value in arrays.items()
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    metadata = {
        "schema_version": 1,
        "output": str(output),
        "data_dir": str(args.data_dir.resolve()),
        "oracle_flow": str(args.oracle_flow.resolve()),
        "ndf_cache": (
            None if args.ndf_cache is None else str(args.ndf_cache.resolve())
        ),
        "episodes": int(len(payload["episode_id"])),
        "object_points": int(args.object_points),
        "target_points": int(args.target_points),
        "flow_anchors": int(payload["episode_flow"].shape[1]),
        "flow_steps": int(payload["episode_flow"].shape[2]),
        "deployment_inputs": [
            "initial segmented operated-object XYZRGB camera cloud",
            "initial segmented target-object XYZRGB camera cloud",
            "optional frozen NDF descriptor or deterministic PCA coordinates",
        ],
        "privileged_supervision_or_diagnostics": [
            "future simulator object track",
            "shoe ID used only for object-disjoint split",
            "marker parameters used only for diagnostics",
            "active-arm label stored but not passed to the flow model",
        ],
        "target_frame_camera_summary": {
            "translation_error_m_mean_label_only": float(
                np.mean(
                    [
                        row["target_frame_translation_error_m_label_only"]
                        for row in diagnostics
                    ]
                )
            ),
            "rotation_error_deg_mean_label_only": float(
                np.mean(
                    [
                        row["target_frame_rotation_error_deg_label_only"]
                        for row in diagnostics
                    ]
                )
            ),
            "rotation_error_deg_max_label_only": float(
                np.max(
                    [
                        row["target_frame_rotation_error_deg_label_only"]
                        for row in diagnostics
                    ]
                )
            ),
        },
        "diagnostics": diagnostics,
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "episodes": metadata["episodes"],
                "minimum_marker_points": min(
                    row["marker_points_in_B"] for row in diagnostics
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
