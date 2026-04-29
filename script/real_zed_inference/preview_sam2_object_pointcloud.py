#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from script.real_zed_inference.real_dp3_inference import (  # noqa: E402
    DEFAULT_ROBOT_CAMERA_CALIBRATION_DIR,
    DEFAULT_SAM2_CHECKPOINT,
    DEFAULT_SAM2_ROOT,
    DEFAULT_WORKSPACE_CALIBRATION,
    add_sam2_object_pointclouds,
    build_robotwin_observation,
    format_timings,
    load_sam2_runtime,
    maybe_show_cameras,
    parse_placeholder_list,
    print_info,
    profile_section,
    start_zed_cameras,
)
from script.real_zed_collection.real_zed_utils import deterministic_resample  # noqa: E402
from script.real_zed_collection.workspace_crop_utils import WorkspaceBounds, invert_transform  # noqa: E402


PLACEHOLDER_COLORS = (
    (0.95, 0.25, 0.20),
    (0.20, 0.45, 1.00),
    (0.25, 0.85, 0.35),
    (0.95, 0.78, 0.18),
    (0.80, 0.35, 0.95),
)


def valid_point_rows(point_cloud: np.ndarray) -> np.ndarray:
    pc = np.asarray(point_cloud, dtype=np.float32)
    if pc.ndim != 2 or pc.shape[1] < 3 or pc.shape[0] == 0:
        return np.zeros((0, 6), dtype=np.float32)
    if pc.shape[1] < 6:
        pad = np.zeros((pc.shape[0], 6 - pc.shape[1]), dtype=np.float32)
        pc = np.concatenate([pc, pad], axis=1)
    pc = pc[:, :6]
    xyz = pc[:, :3]
    finite = np.isfinite(xyz).all(axis=1)
    non_zero = ~np.isclose(xyz, 0.0).all(axis=1)
    return pc[finite & non_zero].astype(np.float32)


def prepare_placeholder_clouds(
    object_pointcloud: Mapping[str, np.ndarray],
    *,
    placeholders: Sequence[str],
    color_mode: str = "placeholder",
) -> dict[str, np.ndarray]:
    mode = str(color_mode).strip().lower()
    if mode not in {"placeholder", "rgb"}:
        raise ValueError("color_mode must be one of: placeholder, rgb")
    out: dict[str, np.ndarray] = {}
    for idx, placeholder in enumerate([str(item) for item in placeholders]):
        pc = valid_point_rows(np.asarray(object_pointcloud.get(placeholder, np.zeros((0, 6))), dtype=np.float32))
        if len(pc) > 0 and mode == "placeholder":
            color = np.asarray(PLACEHOLDER_COLORS[idx % len(PLACEHOLDER_COLORS)], dtype=np.float32)
            pc = pc.copy()
            pc[:, 3:6] = color[None, :]
        out[placeholder] = pc
    return out


def describe_clouds(clouds: Mapping[str, np.ndarray]) -> str:
    parts = []
    for placeholder, cloud in clouds.items():
        pc = valid_point_rows(cloud)
        if len(pc) == 0:
            parts.append(f"{placeholder}: empty")
            continue
        centroid = np.mean(pc[:, :3], axis=0)
        extent = np.max(pc[:, :3], axis=0) - np.min(pc[:, :3], axis=0)
        parts.append(
            f"{placeholder}: n={len(pc)} centroid={np.round(centroid, 3).tolist()} "
            f"extent={np.round(extent, 3).tolist()}"
        )
    return " | ".join(parts)


def workspace_box_corners_in_output_frame(
    bounds: WorkspaceBounds,
    *,
    t_output_from_workspace: np.ndarray,
) -> np.ndarray:
    xs = [float(bounds.x_min), float(bounds.x_max)]
    ys = [float(bounds.y_min), float(bounds.y_max)]
    zs = [float(bounds.z_min), float(bounds.z_max)]
    corners = np.asarray([[x, y, z] for x in xs for y in ys for z in zs], dtype=np.float64)
    tf = np.asarray(t_output_from_workspace, dtype=np.float64).reshape(4, 4)
    corners_h = np.concatenate([corners, np.ones((8, 1), dtype=np.float64)], axis=1)
    return (tf @ corners_h.T).T[:, :3].astype(np.float32)


def workspace_box_edges() -> list[tuple[int, int]]:
    return [
        (0, 1),
        (0, 2),
        (0, 4),
        (3, 1),
        (3, 2),
        (3, 7),
        (5, 1),
        (5, 4),
        (5, 7),
        (6, 2),
        (6, 4),
        (6, 7),
    ]


def output_from_workspace_transform(live) -> np.ndarray:
    if not live.labels:
        return np.eye(4, dtype=np.float32)
    label = live.labels[0]
    return (
        np.asarray(live.t_output_from_cam_by_label[label], dtype=np.float64).reshape(4, 4)
        @ invert_transform(np.asarray(live.t_workspace_from_cam_by_label[label], dtype=np.float64).reshape(4, 4))
    ).astype(np.float32)


class Open3DLiveViewer:
    def __init__(
        self,
        *,
        placeholders: Sequence[str],
        show_scene: bool,
        workspace_box_points: np.ndarray | None,
        point_size: float,
        window_width: int,
        window_height: int,
        coordinate_frame_size: float,
        initial_zoom: float,
    ) -> None:
        import open3d as o3d

        self.o3d = o3d
        self.placeholders = [str(item) for item in placeholders]
        self.show_scene = bool(show_scene)
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(
            window_name="SAM2 Object Point Clouds",
            width=int(window_width),
            height=int(window_height),
        )
        render_option = self.vis.get_render_option()
        render_option.point_size = float(point_size)
        render_option.background_color = np.asarray([0.02, 0.02, 0.02], dtype=np.float64)
        self.object_geometries = {
            placeholder: o3d.geometry.PointCloud()
            for placeholder in self.placeholders
        }
        for pcd in self.object_geometries.values():
            self.vis.add_geometry(pcd)
        self.scene_geometry = o3d.geometry.PointCloud() if self.show_scene else None
        if self.scene_geometry is not None:
            self.vis.add_geometry(self.scene_geometry)
        self.workspace_box = None
        if workspace_box_points is not None and len(workspace_box_points) == 8:
            self.workspace_box = o3d.geometry.LineSet()
            self.workspace_box.points = o3d.utility.Vector3dVector(np.asarray(workspace_box_points, dtype=np.float64))
            self.workspace_box.lines = o3d.utility.Vector2iVector(np.asarray(workspace_box_edges(), dtype=np.int32))
            colors = np.tile(np.asarray([[0.9, 0.9, 0.9]], dtype=np.float64), (12, 1))
            self.workspace_box.colors = o3d.utility.Vector3dVector(colors)
            self.vis.add_geometry(self.workspace_box)
        self.vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(coordinate_frame_size)))
        if float(initial_zoom) > 0:
            self.vis.get_view_control().set_zoom(float(initial_zoom))

    def _set_cloud(self, pcd, cloud: np.ndarray, *, scene: bool = False) -> None:
        pc = valid_point_rows(cloud)
        if scene and len(pc) > 0:
            pc = pc.copy()
            pc[:, 3:6] = np.asarray([0.55, 0.55, 0.55], dtype=np.float32)
        pcd.points = self.o3d.utility.Vector3dVector(pc[:, :3].astype(np.float64))
        colors = np.clip(pc[:, 3:6], 0.0, 1.0) if len(pc) > 0 else np.zeros((0, 3), dtype=np.float32)
        pcd.colors = self.o3d.utility.Vector3dVector(colors.astype(np.float64))

    def update(
        self,
        *,
        object_clouds: Mapping[str, np.ndarray],
        scene_cloud: np.ndarray | None = None,
        reset_view: bool = False,
    ) -> bool:
        for placeholder in self.placeholders:
            self._set_cloud(self.object_geometries[placeholder], object_clouds.get(placeholder, np.zeros((0, 6))))
            self.vis.update_geometry(self.object_geometries[placeholder])
        if self.scene_geometry is not None:
            self._set_cloud(self.scene_geometry, np.zeros((0, 6)) if scene_cloud is None else scene_cloud, scene=True)
            self.vis.update_geometry(self.scene_geometry)
        if bool(reset_view):
            self.vis.reset_view_point(True)
        alive = self.vis.poll_events()
        self.vis.update_renderer()
        return bool(alive)

    def close(self) -> None:
        self.vis.destroy_window()


def default_robot_camera_calibration(output_frame: str) -> str:
    frame = str(output_frame)
    if frame not in {"left_base", "right_base"}:
        return ""
    arm = frame.split("_", 1)[0]
    return str(DEFAULT_ROBOT_CAMERA_CALIBRATION_DIR / f"robot_camera_apriltag_{arm}_global.yaml")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preview real-time SAM2 object masks reconstructed as workspace-filtered point clouds."
    )
    parser.add_argument("--object_placeholders", default="{A},{B}")
    parser.add_argument("--calibration_path", default=str(DEFAULT_WORKSPACE_CALIBRATION))
    parser.add_argument("--frame_mode", choices=["reference_camera", "workspace"], default="workspace")
    parser.add_argument("--output_frame", choices=["source", "workspace", "left_base", "right_base"], default="source")
    parser.add_argument("--robot_camera_calibration_path", default="")
    parser.add_argument("--serial_remap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--camera_labels", default="global,left,right")
    parser.add_argument("--zed_serials", default="")
    parser.add_argument("--zed_resolution", default="HD720")
    parser.add_argument("--zed_fps", type=int, default=15)
    parser.add_argument("--zed_depth_mode", default="NEURAL")
    parser.add_argument("--save_rgb_width", type=int, default=0)
    parser.add_argument("--save_rgb_height", type=int, default=0)
    parser.add_argument("--camera_warmup_timeout_sec", type=float, default=10.0)
    parser.add_argument("--scene_point_num", type=int, default=1024)
    parser.add_argument("--object_point_num", type=int, default=1024)
    parser.add_argument("--scene_preview_num", type=int, default=4096)
    parser.add_argument("--min_depth_m", type=float, default=0.05)
    parser.add_argument("--max_depth_m", type=float, default=3.0)

    parser.add_argument("--workspace_crop_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--workspace_crop_x_min", type=float, default=-0.35)
    parser.add_argument("--workspace_crop_x_max", type=float, default=0.35)
    parser.add_argument("--workspace_crop_y_min", type=float, default=-0.35)
    parser.add_argument("--workspace_crop_y_max", type=float, default=0.35)
    parser.add_argument("--workspace_crop_z_min", type=float, default=0.0)
    parser.add_argument("--workspace_crop_z_max", type=float, default=0.5)
    parser.add_argument("--workspace_crop_margin_px", type=int, default=32)
    parser.add_argument("--workspace_crop_resize_rgb", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--sam2_root", default=str(DEFAULT_SAM2_ROOT))
    parser.add_argument("--sam2_config", default="sam2.1/sam2.1_hiera_b+.yaml")
    parser.add_argument("--sam2_checkpoint", default=str(DEFAULT_SAM2_CHECKPOINT))
    parser.add_argument("--sam2_device", default="cuda:0")
    parser.add_argument("--sam2_autocast_dtype", default="bfloat16")
    parser.add_argument("--sam2_image_width", type=int, default=640)
    parser.add_argument("--sam2_bbox_prompt_path", default="")
    parser.add_argument("--sam2_interactive_init", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sam2_min_mask_points", type=int, default=16)

    parser.add_argument("--color_mode", choices=["placeholder", "rgb"], default="placeholder")
    parser.add_argument("--no_open3d", action="store_true", help="Disable Open3D rendering and only print reconstruction FPS.")
    parser.add_argument("--point_size", type=float, default=6.0)
    parser.add_argument("--open3d_width", type=int, default=1600)
    parser.add_argument("--open3d_height", type=int, default=1000)
    parser.add_argument("--coordinate_frame_size", type=float, default=0.12)
    parser.add_argument("--open3d_initial_zoom", type=float, default=0.55)
    parser.add_argument("--show_workspace_box", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reset_view_each_frame", action="store_true")
    parser.add_argument("--show_scene", action="store_true")
    parser.add_argument("--show_img", action="store_true")
    parser.add_argument("--preview_hz", type=float, default=10.0)
    parser.add_argument("--max_frames", type=int, default=-1)
    parser.add_argument("--print_interval", type=int, default=10)
    parser.add_argument("--profile_timing", action="store_true")
    return parser


def run_preview(args: argparse.Namespace) -> None:
    placeholders = parse_placeholder_list(args.object_placeholders)
    if not placeholders:
        raise ValueError("--object_placeholders must contain at least one placeholder, e.g. '{A}'")
    if str(args.output_frame) in {"left_base", "right_base"} and not str(args.robot_camera_calibration_path).strip():
        args.robot_camera_calibration_path = default_robot_camera_calibration(str(args.output_frame))

    print(f"[INFO] placeholders={placeholders}")
    print(f"[INFO] output_frame={args.output_frame}")
    print(f"[INFO] workspace_crop_enabled={bool(args.workspace_crop_enabled)}")
    print(f"[INFO] sam2_image_width={int(args.sam2_image_width)} zed_resolution={args.zed_resolution}")
    print(f"[INFO] open3d_enabled={not bool(args.no_open3d)}")
    if args.robot_camera_calibration_path:
        print(f"[INFO] robot_camera_calibration_path={args.robot_camera_calibration_path}")

    live = start_zed_cameras(args)
    tracker_factory, extract_all_fn, loaded_prompts = load_sam2_runtime(args, placeholders, live.labels)
    tracking_state_by_camera: dict[str, Any] = {}
    bbox_prompts_by_camera: dict[str, dict[str, object]] = dict(loaded_prompts)
    viewer: Open3DLiveViewer | None = None
    workspace_box_points = None
    if bool(args.show_workspace_box) and live.workspace_bounds is not None:
        workspace_box_points = workspace_box_corners_in_output_frame(
            live.workspace_bounds,
            t_output_from_workspace=output_from_workspace_transform(live),
        )
    frame_idx = 0
    period = 0.0 if float(args.preview_hz) <= 0 else 1.0 / float(args.preview_hz)
    try:
        while int(args.max_frames) < 0 or frame_idx < int(args.max_frames):
            loop_start = time.perf_counter()
            timings: dict[str, float] = {}
            with profile_section(timings, "build_obs", bool(args.profile_timing)):
                observation, dense_scene = build_robotwin_observation(
                    args=args,
                    live=live,
                    joint_vector=np.zeros(14, dtype=np.float32),
                )
            with profile_section(timings, "sam2_object_pc", bool(args.profile_timing)):
                observation = add_sam2_object_pointclouds(
                    observation=observation,
                    dense_scene_pointcloud=dense_scene,
                    args=args,
                    placeholders=placeholders,
                    camera_names=live.labels,
                    tracker_factory=tracker_factory,
                    extract_all_fn=extract_all_fn,
                    tracking_state_by_camera=tracking_state_by_camera,
                    bbox_prompts_by_camera=bbox_prompts_by_camera,
                )
            prepared = prepare_placeholder_clouds(
                observation.get("object_pointcloud", {}),
                placeholders=placeholders,
                color_mode=str(args.color_mode),
            )
            scene_preview = (
                deterministic_resample(dense_scene, int(args.scene_preview_num))
                if bool(args.show_scene)
                else None
            )
            if not bool(args.no_open3d):
                if viewer is None:
                    viewer = Open3DLiveViewer(
                        placeholders=placeholders,
                        show_scene=bool(args.show_scene),
                        workspace_box_points=workspace_box_points,
                        point_size=float(args.point_size),
                        window_width=int(args.open3d_width),
                        window_height=int(args.open3d_height),
                        coordinate_frame_size=float(args.coordinate_frame_size),
                        initial_zoom=float(args.open3d_initial_zoom),
                    )
                if not viewer.update(
                    object_clouds=prepared,
                    scene_cloud=scene_preview,
                    reset_view=bool(args.reset_view_each_frame),
                ):
                    break
            if bool(args.show_img):
                maybe_show_cameras(observation, live.labels)
            frame_idx += 1
            if frame_idx == 1 or frame_idx % max(1, int(args.print_interval)) == 0:
                elapsed = time.perf_counter() - loop_start
                fps = 1.0 / max(elapsed, 1e-6)
                msg = f"[PREVIEW {frame_idx:05d}] fps={fps:.2f} {describe_clouds(prepared)}"
                if bool(args.profile_timing):
                    msg += f" timings={format_timings(timings)}"
                print_info(msg)
            elapsed = time.perf_counter() - loop_start
            if period > 0 and elapsed < period:
                time.sleep(period - elapsed)
    except KeyboardInterrupt:
        print("[INFO] interrupted")
    finally:
        if viewer is not None:
            viewer.close()
        live.stop()
        if bool(args.show_img):
            cv2.destroyAllWindows()


def main() -> None:
    args = build_arg_parser().parse_args()
    run_preview(args)


if __name__ == "__main__":
    main()
