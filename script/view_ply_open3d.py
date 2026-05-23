import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

# DEFAULT_PLY = ["/home/zheng/github/3d_semantic_train/tools/outputs/visualizations_universal_field_query_count_stability/mug/direct_mesh_01af543e10c8478c9632394e176a3a50/surface_sem_embedding_joint_pca_q5000.ply"]
# DEFAULT_PLY = ["/home/zheng/github/3d_semantic_train/tools/outputs/visualizations_universal_field/mug/direct_point_cloud_frame_000013_A/surface_sem_embedding_joint_pca.ply"]
DEFAULT_PLY = ["/home/zheng/github/RoboTwin_geo/outputs/semantic_field_dataset_viz/episode0_frame13_scene_semantic_overlay.ply"]


def load_point_cloud(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Point cloud file not found: {path}")

    point_cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(point_cloud.points)
    if points.size == 0:
        raise ValueError(f"No points were loaded from: {path}")
    if not point_cloud.has_colors():
        print(f"Warning: {path} has no vertex colors; Open3D will show default color.")
    return point_cloud


def collect_ply_paths(args):
    paths = list(args.ply)
    if args.ply_option:
        for group in args.ply_option:
            for item in group:
                paths.extend(path.strip() for path in item.split(",") if path.strip())
    return paths or DEFAULT_PLY


def main():
    parser = argparse.ArgumentParser(
        description="View one or more PLY point clouds with Open3D vertex colors."
    )
    parser.add_argument("ply", nargs="*", help="PLY point cloud file path(s)")
    parser.add_argument(
        "--ply",
        dest="ply_option",
        nargs="+",
        default=None,
        help="PLY point cloud file path(s); comma-separated paths are also accepted",
    )
    parser.add_argument(
        "--point_size",
        type=float,
        default=3.0,
        help="render point size in the Open3D viewer",
    )
    args = parser.parse_args()

    point_clouds = [load_point_cloud(path) for path in collect_ply_paths(args)]

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(window_name="Open3D PLY Viewer")
    for point_cloud in point_clouds:
        visualizer.add_geometry(point_cloud)

    render_option = visualizer.get_render_option()
    render_option.point_size = args.point_size
    render_option.background_color = np.asarray([1.0, 1.0, 1.0])

    visualizer.run()
    visualizer.destroy_window()


if __name__ == "__main__":
    main()
