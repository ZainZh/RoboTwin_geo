import argparse
import os
from pathlib import Path

import numpy as np
import open3d as o3d
import zarr


def infer_episode_and_local_frame(episode_ends: np.ndarray, global_frame_idx: int):
    episode_ends = np.asarray(episode_ends, dtype=np.int64)
    episode_idx = int(np.searchsorted(episode_ends, global_frame_idx, side="right"))
    episode_start = 0 if episode_idx == 0 else int(episode_ends[episode_idx - 1])
    local_frame_idx = int(global_frame_idx - episode_start)
    return episode_idx, local_frame_idx


def make_o3d_cloud(points: np.ndarray, color) -> o3d.geometry.PointCloud:
    points = np.asarray(points, dtype=np.float32)
    xyz = points[:, :3]
    valid = ~(np.isclose(xyz, 0.0).all(axis=1))
    xyz = xyz[valid]
    pcd = o3d.geometry.PointCloud()
    if len(xyz) == 0:
        return pcd
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    colors = np.tile(np.asarray(color, dtype=np.float64)[None, :], (len(xyz), 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def describe_cloud(name: str, points: np.ndarray):
    xyz = np.asarray(points, dtype=np.float32)[:, :3]
    valid = ~(np.isclose(xyz, 0.0).all(axis=1))
    xyz = xyz[valid]
    if len(xyz) == 0:
        print(f"{name}: empty")
        return
    extent = xyz.max(axis=0) - xyz.min(axis=0)
    centroid = xyz.mean(axis=0)
    print(
        f"{name}: count={len(xyz)}, centroid={np.round(centroid, 4).tolist()}, "
        f"extent={np.round(extent, 4).tolist()}"
    )


def main():
    parser = argparse.ArgumentParser(description="Visualize placeholder point clouds stored in a DP3 zarr.")
    parser.add_argument("zarr_path", type=str)
    parser.add_argument("--frame_idx", type=int, default=100, help="Global frame index in flattened zarr data.")
    parser.add_argument("--placeholders", type=str, default="A,B")
    parser.add_argument("--show_merged", action="store_true")
    parser.add_argument("--export_dir", type=str, default="")
    parser.add_argument("--no_show", action="store_true")
    args = parser.parse_args()

    root = zarr.open(str(Path(args.zarr_path).resolve()), mode="r")
    data = root["data"]
    episode_ends = np.asarray(root["meta"]["episode_ends"])
    episode_idx, local_frame_idx = infer_episode_and_local_frame(episode_ends, int(args.frame_idx))
    print(f"global_frame_idx={args.frame_idx}, episode={episode_idx}, local_frame={local_frame_idx}")

    placeholder_labels = [item.strip() for item in str(args.placeholders).split(",") if item.strip()]
    dataset_names = []
    for label in placeholder_labels:
        dataset_names.append((label, f"object_point_cloud_{label.strip('{}')}"))

    colors = [
        [0.95, 0.35, 0.35],
        [0.25, 0.55, 0.95],
        [0.35, 0.85, 0.45],
        [0.9, 0.8, 0.2],
    ]
    geometries = []

    export_dir = Path(args.export_dir).resolve() if args.export_dir else None
    if export_dir is not None:
        export_dir.mkdir(parents=True, exist_ok=True)

    for idx, (label, dataset_name) in enumerate(dataset_names):
        if dataset_name not in data:
            print(f"{dataset_name}: missing in zarr")
            continue
        cloud = np.asarray(data[dataset_name][args.frame_idx], dtype=np.float32)
        describe_cloud(dataset_name, cloud)
        pcd = make_o3d_cloud(cloud, colors[idx % len(colors)])
        geometries.append(pcd)
        if export_dir is not None:
            out_path = export_dir / f"{dataset_name}_frame{args.frame_idx}.ply"
            o3d.io.write_point_cloud(str(out_path), pcd)

    if args.show_merged and "point_cloud" in data:
        merged = np.asarray(data["point_cloud"][args.frame_idx], dtype=np.float32)
        describe_cloud("point_cloud", merged)
        merged_pcd = make_o3d_cloud(merged, [0.7, 0.7, 0.7])
        geometries.append(merged_pcd)
        if export_dir is not None:
            out_path = export_dir / f"point_cloud_frame{args.frame_idx}.ply"
            o3d.io.write_point_cloud(str(out_path), merged_pcd)

    if not args.no_show and len(geometries) > 0:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        geometries.append(axis)
        o3d.visualization.draw_geometries(geometries)


if __name__ == "__main__":
    main()
