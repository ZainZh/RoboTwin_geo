import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

# DEFAULT_PLY = ["/home/zheng/github/3d_semantic_train/tools/outputs/visualizations_universal_field_query_count_stability/mug/direct_mesh_01af543e10c8478c9632394e176a3a50/surface_sem_embedding_joint_pca_q5000.ply"]
# DEFAULT_PLY = ["/home/zheng/github/RoboTwin_geo/script/real_zed_collection/outputs/real_zed_collection/raw_colored_pointclouds/beat_cube/0/fused/frame_000035.ply"]
# DEFAULT_PLY = ["/home/zheng/github/RoboTwin_geo/outputs/semantic_field_dataset_viz/view_pour_water/episode1_frame0_scene_semantic_overlay.ply"]
DEFAULT_PLY = ["/home/zheng/github/RoboTwin_geo/outputs/semantic_field_dataset_viz/view_grasp_mug/episode7_frame0_scene_dinov2_overlay.ply"]
# DEFAULT_PLY = ["/home/zheng/github/RoboTwin_geo/outputs/semantic_field_dataset_viz/view_grasp_mug/episode0_frame0_scene_utonia_overlay.ply"]
# DEFAULT_PLY = ["/home/zheng/github/RoboTwin_geo/outputs/semantic_field_dataset_viz/view_pour_water/episode1_frame0_scene_original_color.ply"]

DEFAULT_VIEW_STATUS = r'''{
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : false,
	"trajectory" :
	[
		{
			"boundingbox_max" : [ 2.281740665435791, 0.84336256980895996, 2.9999933242797852 ],
			"boundingbox_min" : [ -2.0944585800170898, -1.5530669689178467, -0.2132609486579895 ],
			"field_of_view" : 60.0,
			"front" : [ 0.82536791582361468, 0.062230351785553022, 0.56115522526804384 ],
			"lookat" : [ 0.0045460694301429167, -0.13615938517039466, 0.035368909388652794 ],
			"up" : [ -0.56303842285749539, 0.016963397975656894, 0.82625660512655741 ],
			"zoom" : 0.12000000000000001
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}'''

# DEFAULT_VIEW_STATUS = r'''{
# 	"class_name" : "ViewTrajectory",
# 	"interval" : 29,
# 	"is_loop" : false,
# 	"trajectory" :
# 	[
# 		{
# 			"boundingbox_max" : [ 2.281740665435791, 0.84336256980895996, 2.9999933242797852 ],
# 			"boundingbox_min" : [ -2.0944585800170898, -1.5530669689178467, -0.2132609486579895 ],
# 			"field_of_view" : 60.0,
# 			"front" : [ 0.017775568391123615, 0.084816570687139589, -0.99623801297945169 ],
# 			"lookat" : [ -0.04674126035239063, -0.067287431299280484, 0.004294938229810182 ],
# 			"up" : [ 0.0072045158680923722, -0.99638043929153897, -0.084700148455042187 ],
# 			"zoom" : 0.02
# 		}
# 	],
# 	"version_major" : 1,
# 	"version_minor" : 0
# }'''
#
DEFAULT_VIEW_STATUS_FILE = ""

VIEW_PRESETS = {
    # Global frame convention used by the real-ZED processed point clouds:
    # X horizontal, Y depth, Z up. Look from -Y toward the scene with Z up.
    "global": {
        "front": [0.0, -1.0, 0.0],
        "up": [0.0, 0.0, 1.0],
        "zoom": 0.55,
    },
    "top": {
        "front": [0.0, 0.0, -1.0],
        "up": [0.0, -1.0, 0.0],
        "zoom": 0.65,
    },
}


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


def parse_vec3(value, *, name):
    items = [item.strip() for item in str(value).replace(",", " ").split() if item.strip()]
    if len(items) != 3:
        raise argparse.ArgumentTypeError(f"{name} must contain exactly 3 numbers, got: {value!r}")
    try:
        return [float(item) for item in items]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must contain numeric values, got: {value!r}") from exc


def normalize_vec3(value, *, name):
    vector = np.asarray(value, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        raise ValueError(f"{name} cannot be a zero vector.")
    return (vector / norm).astype(np.float64).tolist()


def compute_scene_center_from_arrays(point_arrays):
    valid_arrays = []
    for points in point_arrays:
        arr = np.asarray(points, dtype=np.float32)
        if arr.size == 0:
            continue
        arr = arr.reshape(-1, 3)
        valid_arrays.append(arr)
    if not valid_arrays:
        return np.zeros(3, dtype=np.float32)
    stacked = np.concatenate(valid_arrays, axis=0)
    return ((stacked.min(axis=0) + stacked.max(axis=0)) * 0.5).astype(np.float32)


def compute_scene_center(point_clouds):
    return compute_scene_center_from_arrays([np.asarray(point_cloud.points) for point_cloud in point_clouds])


def apply_view_preset(view_control, *, preset, lookat, zoom=None, front=None, up=None):
    preset_name = str(preset)
    if preset_name == "none":
        return
    if preset_name not in VIEW_PRESETS:
        raise ValueError(f"Unsupported view preset: {preset}")

    preset_cfg = VIEW_PRESETS[preset_name]
    front_vec = normalize_vec3(front if front is not None else preset_cfg["front"], name="front")
    up_vec = normalize_vec3(up if up is not None else preset_cfg["up"], name="up")
    lookat_vec = np.asarray(lookat, dtype=np.float64).reshape(3).tolist()
    zoom_value = float(preset_cfg["zoom"] if zoom is None else zoom)

    view_control.set_front(front_vec)
    view_control.set_up(up_vec)
    view_control.set_lookat(lookat_vec)
    view_control.set_zoom(zoom_value)


def read_view_status_file(path):
    view_path = Path(path).expanduser()
    if not view_path.is_file():
        raise FileNotFoundError(f"View status file not found: {view_path}")
    return view_path.read_text(encoding="utf-8").strip()


def resolve_view_status(*, view_status="", view_status_file=""):
    if view_status_file:
        return read_view_status_file(view_status_file)
    return str(view_status or "").strip()


def apply_view_status(visualizer, *, view_status="", view_status_file=""):
    status = resolve_view_status(view_status=view_status, view_status_file=view_status_file)
    if not status:
        return False
    visualizer.set_view_status(status)
    return True


def format_view_status_report(view_status):
    status = str(view_status).strip()
    return (
        "\n[Open3D current view status]\n"
        "Paste this into script/view_ply_open3d.py to make it the permanent default:\n\n"
        f"DEFAULT_VIEW_STATUS = r'''{status}'''\n\n"
        "Or save it to a file and launch with:\n"
        "python script/view_ply_open3d.py your.ply --view_status_file /path/to/view_status.json\n"
    )


def register_print_view_status_key(visualizer, *, key="P", output_path=""):
    key_text = str(key or "P").strip()
    if len(key_text) != 1:
        raise ValueError(f"print view key must be a single character, got: {key!r}")
    output = str(output_path or "").strip()

    def _callback(vis):
        status = vis.get_view_status()
        print(format_view_status_report(status), flush=True)
        if output:
            output_file = Path(output).expanduser()
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(str(status).strip() + "\n", encoding="utf-8")
            print(f"[Open3D current view status] wrote: {output_file}", flush=True)
        return False

    visualizer.register_key_callback(ord(key_text.upper()), _callback)


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
    parser.add_argument(
        "--view_preset",
        choices=["global", "top", "none"],
        default="global",
        help="initial Open3D camera preset; global keeps a stable global-frame view",
    )
    parser.add_argument("--front", type=lambda value: parse_vec3(value, name="front"), default=None)
    parser.add_argument("--up", type=lambda value: parse_vec3(value, name="up"), default=None)
    parser.add_argument("--lookat", type=lambda value: parse_vec3(value, name="lookat"), default=None)
    parser.add_argument("--zoom", type=float, default=None)
    parser.add_argument(
        "--view_status",
        default=DEFAULT_VIEW_STATUS,
        help="Open3D view status JSON, usually pasted from pressing the print-view key",
    )
    parser.add_argument(
        "--view_status_file",
        default=DEFAULT_VIEW_STATUS_FILE,
        help="Path to Open3D view status JSON saved from the print-view key",
    )
    parser.add_argument(
        "--print_view_key",
        default="P",
        help="single key that prints the current Open3D view status to the terminal",
    )
    parser.add_argument(
        "--view_status_output",
        default="",
        help="optional path to also save the current view status whenever --print_view_key is pressed",
    )
    args = parser.parse_args()

    point_clouds = [load_point_cloud(path) for path in collect_ply_paths(args)]

    visualizer = o3d.visualization.VisualizerWithKeyCallback()
    visualizer.create_window(window_name="Open3D PLY Viewer")
    for point_cloud in point_clouds:
        visualizer.add_geometry(point_cloud)

    render_option = visualizer.get_render_option()
    render_option.point_size = args.point_size
    render_option.background_color = np.asarray([1.0, 1.0, 1.0])

    register_print_view_status_key(
        visualizer,
        key=args.print_view_key,
        output_path=args.view_status_output,
    )

    lookat = np.asarray(args.lookat, dtype=np.float32) if args.lookat is not None else compute_scene_center(point_clouds)
    visualizer.poll_events()
    visualizer.update_renderer()
    if not apply_view_status(
        visualizer,
        view_status=args.view_status,
        view_status_file=args.view_status_file,
    ):
        apply_view_preset(
            visualizer.get_view_control(),
            preset=args.view_preset,
            lookat=lookat,
            zoom=args.zoom,
            front=args.front,
            up=args.up,
        )

    visualizer.run()
    visualizer.destroy_window()


if __name__ == "__main__":
    main()
