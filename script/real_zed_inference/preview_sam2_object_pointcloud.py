#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
import threading
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
    camera_frame_to_output_pc,
    format_timings,
    maybe_show_cameras,
    parse_placeholder_list,
    print_info,
    rgb_to_depth_shape,
    snapshot_frames,
    start_zed_cameras,
)
from script.real_zed_collection.real_zed_utils import deterministic_resample, merge_point_clouds  # noqa: E402
from script.real_zed_collection.workspace_crop_utils import WorkspaceBounds, invert_transform  # noqa: E402
from object_pointcloud_utils import merge_object_point_clouds, resample_point_cloud  # noqa: E402
from sam2_pointcloud_utils import (  # noqa: E402
    Sam2CameraTrackingState,
    _scale_prompt_spec_to_image,
    _select_depth_points_by_mask,
    _resize_keep_aspect,
    build_sam2_tracker_factory,
    fast_merge_object_point_clouds,
    load_sam2_bbox_prompt_file,
    run_camera_tasks_parallel,
)


PLACEHOLDER_COLORS = (
    (0.95, 0.25, 0.20),
    (0.20, 0.45, 1.00),
    (0.25, 0.85, 0.35),
    (0.95, 0.78, 0.18),
    (0.80, 0.35, 0.95),
)


class TimingRecorder:
    def __init__(self, *, enabled: bool = False) -> None:
        self.enabled = bool(enabled)
        self.timings: dict[str, float] = {}
        self._lock = threading.Lock()

    def add(self, name: str, seconds: float) -> None:
        if self.enabled:
            with self._lock:
                self.timings[str(name)] = self.timings.get(str(name), 0.0) + float(seconds)

    def format(self) -> str:
        return format_timings(self.timings)


class timed_stage:
    def __init__(self, timer: TimingRecorder, name: str) -> None:
        self.timer = timer
        self.name = str(name)
        self.start = 0.0

    def __enter__(self):
        if self.timer.enabled:
            self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.timer.enabled:
            self.timer.add(self.name, time.perf_counter() - self.start)


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


def workspace_bounds_array(bounds: WorkspaceBounds | None) -> np.ndarray | None:
    if bounds is None:
        return None
    return np.asarray(
        [
            bounds.x_min,
            bounds.x_max,
            bounds.y_min,
            bounds.y_max,
            bounds.z_min,
            bounds.z_max,
        ],
        dtype=np.float32,
    )


def build_preview_observation(
    *,
    args: argparse.Namespace,
    live,
    timer: TimingRecorder,
    build_scene: bool = True,
) -> tuple[dict[str, Any], np.ndarray]:
    with timed_stage(timer, "snapshot_frames"):
        frames_by_label = snapshot_frames(live)
    camera_obs: dict[str, dict[str, Any]] = {}
    bounds_arr = workspace_bounds_array(live.workspace_bounds)

    def build_camera(label: str):
        with timed_stage(timer, f"build_obs.{label}"):
            calib = live.calibrations[live.label_to_calib[label]]
            frame = frames_by_label[label]
            with timed_stage(timer, f"build_obs.{label}.align"):
                depth_m = np.asarray(frame["depth_m"], dtype=np.float32)
                rgb_aligned = rgb_to_depth_shape(np.asarray(frame["rgb"], dtype=np.uint8), depth_m.shape)
                camera_matrix = np.asarray(frame.get("camera_matrix", calib.camera_matrix), dtype=np.float32).reshape(3, 3)
                t_output_from_cam = np.asarray(live.t_output_from_cam_by_label[label], dtype=np.float32).reshape(4, 4)
                t_workspace_from_cam = np.asarray(live.t_workspace_from_cam_by_label[label], dtype=np.float32).reshape(4, 4)
            scene_chunk = None
            if bool(build_scene):
                with timed_stage(timer, f"build_obs.{label}.depth_to_pc"):
                    frame_for_pc = dict(frame)
                    frame_for_pc["rgb"] = rgb_aligned
                    scene_chunk = camera_frame_to_output_pc(
                        camera_frame=frame_for_pc,
                        camera_matrix=camera_matrix,
                        t_workspace_from_cam=t_workspace_from_cam,
                        workspace_bounds=live.workspace_bounds,
                        t_output_from_cam=t_output_from_cam,
                        min_depth_m=float(args.min_depth_m),
                        max_depth_m=float(args.max_depth_m),
                    )
            return label, {
                "rgb": rgb_aligned.astype(np.uint8),
                "depth": depth_m.astype(np.float32),
                "intrinsic_cv": camera_matrix.astype(np.float32),
                "cam2world_gl": t_output_from_cam.astype(np.float32),
                "t_workspace_from_cam": t_workspace_from_cam.astype(np.float32),
                "workspace_bounds_m": bounds_arr,
            }, scene_chunk

    results = run_camera_tasks_parallel(
        live.labels,
        build_camera,
        max_workers=int(getattr(args, "parallel_camera_workers", 0)),
    )
    scene_chunks: list[np.ndarray] = []
    for label in live.labels:
        _label, camera_info, scene_chunk = results[str(label)]
        camera_obs[str(_label)] = camera_info
        if scene_chunk is not None:
            scene_chunks.append(scene_chunk)

    if bool(build_scene):
        with timed_stage(timer, "build_obs.merge_scene"):
            dense_scene = merge_point_clouds(scene_chunks)
        with timed_stage(timer, "build_obs.resample_scene"):
            scene_point_cloud = deterministic_resample(dense_scene, int(args.scene_point_num))
    else:
        dense_scene = np.zeros((0, 6), dtype=np.float32)
        scene_point_cloud = np.zeros((int(args.scene_point_num), 6), dtype=np.float32)
    observation = {
        "joint_action": {"vector": np.zeros(14, dtype=np.float32)},
        "pointcloud": scene_point_cloud.astype(np.float32),
        "observation": camera_obs,
    }
    return observation, dense_scene.astype(np.float32)


def load_preview_sam2_runtime(args: argparse.Namespace, placeholders: Sequence[str], camera_names: Sequence[str]):
    tracker_factory = build_sam2_tracker_factory(
        placeholders=placeholders,
        sam2_root=args.sam2_root,
        config=args.sam2_config,
        checkpoint=args.sam2_checkpoint,
        device=args.sam2_device,
        autocast_dtype=args.sam2_autocast_dtype,
    )
    prompt_path = str(args.sam2_bbox_prompt_path or "").strip()
    prompts = (
        load_sam2_bbox_prompt_file(prompt_path, camera_names=camera_names, placeholders=placeholders)
        if prompt_path
        else {}
    )
    return tracker_factory, prompts


def interactive_boxes_for_camera(camera_name: str, placeholders: Sequence[str], image: np.ndarray):
    from sam2_pointcloud_utils import select_bbox_for_image

    return {
        str(placeholder): select_bbox_for_image(str(camera_name), str(placeholder), image)
        for placeholder in placeholders
    }


def extract_preview_object_pointclouds_sam2(
    observation: Mapping[str, Any],
    *,
    placeholders: Sequence[str],
    camera_names: Sequence[str],
    tracker_factory,
    tracking_state_by_camera: dict[str, Sam2CameraTrackingState],
    bbox_prompts_by_camera: dict[str, dict[str, object]],
    target_num_points: int,
    min_mask_points: int,
    sam2_image_width: int,
    min_depth_m: float,
    max_depth_m: float,
    interactive_init: bool,
    object_resample_mode: str,
    parallel_camera_workers: int | None,
    timer: TimingRecorder,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    placeholder_list = [str(item) for item in placeholders]
    camera_name_list = [str(item) for item in camera_names]
    empty_clouds = {
        placeholder: np.zeros((int(target_num_points), 6), dtype=np.float32)
        for placeholder in placeholder_list
    }
    selected_by_placeholder: dict[str, list[np.ndarray]] = {placeholder: [] for placeholder in placeholder_list}
    camera_meta: dict[str, Any] = {}
    camera_obs = observation.get("observation", {}) if isinstance(observation, Mapping) else {}

    if bool(interactive_init):
        for camera_name in camera_name_list:
            state = tracking_state_by_camera.get(camera_name)
            if state is not None and state.initialized:
                continue
            boxes = dict(bbox_prompts_by_camera.get(camera_name, {}))
            if boxes:
                continue
            camera_info = camera_obs.get(camera_name)
            if camera_info is None:
                continue
            image = np.asarray(camera_info.get("rgb"), dtype=np.uint8)
            if image.ndim != 3:
                continue
            with timed_stage(timer, f"sam2.{camera_name}.resize_for_prompt"):
                tracker_image = _resize_keep_aspect(image, int(sam2_image_width))
            bbox_prompts_by_camera[camera_name] = interactive_boxes_for_camera(camera_name, placeholder_list, tracker_image)

    state_lock = threading.Lock()

    def process_camera(camera_name: str):
        camera_key = str(camera_name)
        selected: dict[str, list[np.ndarray]] = {placeholder: [] for placeholder in placeholder_list}
        camera_info = camera_obs.get(camera_key)
        if camera_info is None:
            return camera_key, selected, {"mode": "missing_camera"}, False
        image = np.asarray(camera_info.get("rgb"), dtype=np.uint8)
        if image.ndim != 3:
            return camera_key, selected, {"mode": "missing_rgb"}, False
        with timed_stage(timer, f"sam2.{camera_key}.resize"):
            tracker_image = _resize_keep_aspect(image, int(sam2_image_width))

        with state_lock:
            state = tracking_state_by_camera.setdefault(camera_key, Sam2CameraTrackingState())
        if state.tracker is None:
            with timed_stage(timer, f"sam2.{camera_key}.create_tracker"):
                state.tracker = tracker_factory(camera_key)
        if not state.initialized:
            boxes = dict(bbox_prompts_by_camera.get(camera_key, {}))
            missing = [placeholder for placeholder in placeholder_list if placeholder not in boxes]
            if missing:
                return camera_key, selected, {"mode": "missing_bbox_prompts", "missing_placeholders": missing}, False
            boxes = {
                placeholder: _scale_prompt_spec_to_image(prompt_spec, tracker_image.shape[:2])
                for placeholder, prompt_spec in boxes.items()
            }
            with timed_stage(timer, f"sam2.{camera_key}.initialize"):
                masks = state.tracker.initialize(tracker_image, boxes)
            state.initialized = True
            mode = "initialized"
        else:
            with timed_stage(timer, f"sam2.{camera_key}.track"):
                masks = state.tracker.track(tracker_image)
            mode = "tracked"

        per_placeholder_meta: dict[str, Any] = {}
        camera_selected = False
        for placeholder in placeholder_list:
            mask = np.asarray(masks.get(placeholder, np.zeros(tracker_image.shape[:2], dtype=bool))).astype(bool)
            with timed_stage(timer, f"object_pc.{camera_key}.{placeholder}.lift_depth"):
                points, meta = _select_depth_points_by_mask(
                    mask=mask,
                    depth=camera_info.get("depth"),
                    rgb=image,
                    intrinsic_cv=camera_info.get("intrinsic_cv"),
                    cam2world_gl=camera_info.get("cam2world_gl"),
                    t_workspace_from_cam=camera_info.get("t_workspace_from_cam"),
                    workspace_bounds_m=camera_info.get("workspace_bounds_m"),
                    min_depth_m=float(min_depth_m),
                    max_depth_m=float(max_depth_m),
                    min_points=int(min_mask_points),
            )
            if points is not None:
                selected[placeholder].append(points)
                camera_selected = True
            per_placeholder_meta[placeholder] = meta
        return camera_key, selected, {"mode": mode, "placeholders": per_placeholder_meta}, camera_selected

    active_camera_count = 0
    camera_results = run_camera_tasks_parallel(
        camera_name_list,
        process_camera,
        max_workers=parallel_camera_workers,
    )
    for camera_name in camera_name_list:
        if camera_name not in camera_results:
            continue
        camera_key, selected, meta, camera_selected = camera_results[camera_name]
        camera_meta[camera_key] = meta
        for placeholder, clouds in selected.items():
            selected_by_placeholder[placeholder].extend(clouds)
        if camera_selected:
            active_camera_count += 1

    out: dict[str, np.ndarray] = {}
    for placeholder, clouds in selected_by_placeholder.items():
        if not clouds:
            out[placeholder] = empty_clouds[placeholder]
            continue
        mode = str(object_resample_mode).strip().lower()
        if mode == "fast":
            with timed_stage(timer, f"object_pc.{placeholder}.fast_merge"):
                out[placeholder] = fast_merge_object_point_clouds(clouds, target_num_points=int(target_num_points))
        else:
            with timed_stage(timer, f"object_pc.{placeholder}.fps_merge"):
                merged = merge_object_point_clouds(clouds, target_num_points=int(target_num_points))
            with timed_stage(timer, f"object_pc.{placeholder}.resample"):
                out[placeholder] = resample_point_cloud(merged, int(target_num_points))

    return out, {
        "mode": "sam2_mask_depth_profiled",
        "object_resample_mode": str(object_resample_mode),
        "camera_count": int(active_camera_count),
        "cameras": camera_meta,
    }


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
    parser.add_argument(
        "--object_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip dense full-scene point-cloud construction unless --show_scene is also set.",
    )
    parser.add_argument(
        "--parallel_camera_workers",
        type=int,
        default=0,
        help="Per-camera worker count; <=0 uses one worker per active camera, 1 disables parallelism.",
    )

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
    parser.add_argument("--online_object_resample", choices=["fast", "fps"], default="fast")
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
    parser.add_argument("--print_interval", type=int, default=2)
    parser.add_argument("--profile_timing", action="store_true",default=True)
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
    print(f"[INFO] online_object_resample={args.online_object_resample}")
    print(f"[INFO] object_only={bool(args.object_only)} parallel_camera_workers={int(args.parallel_camera_workers)}")
    print(f"[INFO] open3d_enabled={not bool(args.no_open3d)}")
    if args.robot_camera_calibration_path:
        print(f"[INFO] robot_camera_calibration_path={args.robot_camera_calibration_path}")

    live = start_zed_cameras(args)
    tracker_factory, loaded_prompts = load_preview_sam2_runtime(args, placeholders, live.labels)
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
            timer = TimingRecorder(enabled=bool(args.profile_timing))
            build_scene = bool(args.show_scene) or not bool(args.object_only)
            observation, dense_scene = build_preview_observation(
                args=args,
                live=live,
                timer=timer,
                build_scene=build_scene,
            )
            with timed_stage(timer, "sam2_object_pc.total"):
                object_pointcloud, _meta = extract_preview_object_pointclouds_sam2(
                    observation,
                    placeholders=placeholders,
                    camera_names=live.labels,
                    tracker_factory=tracker_factory,
                    tracking_state_by_camera=tracking_state_by_camera,
                    bbox_prompts_by_camera=bbox_prompts_by_camera,
                    target_num_points=int(args.object_point_num),
                    min_mask_points=int(args.sam2_min_mask_points),
                    sam2_image_width=int(args.sam2_image_width),
                    min_depth_m=float(args.min_depth_m),
                    max_depth_m=float(args.max_depth_m),
                    interactive_init=bool(args.sam2_interactive_init),
                    object_resample_mode=str(args.online_object_resample),
                    parallel_camera_workers=int(args.parallel_camera_workers),
                    timer=timer,
                )
            with timed_stage(timer, "prepare_clouds"):
                prepared = prepare_placeholder_clouds(
                    object_pointcloud,
                    placeholders=placeholders,
                    color_mode=str(args.color_mode),
                )
            with timed_stage(timer, "scene_preview"):
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
                with timed_stage(timer, "open3d_update"):
                    if not viewer.update(
                        object_clouds=prepared,
                        scene_cloud=scene_preview,
                        reset_view=bool(args.reset_view_each_frame),
                    ):
                        break
            if bool(args.show_img):
                with timed_stage(timer, "show_img"):
                    maybe_show_cameras(observation, live.labels)
            frame_idx += 1
            if frame_idx == 1 or frame_idx % max(1, int(args.print_interval)) == 0:
                elapsed = time.perf_counter() - loop_start
                fps = 1.0 / max(elapsed, 1e-6)
                msg = f"[PREVIEW {frame_idx:05d}] fps={fps:.2f} {describe_clouds(prepared)}"
                if bool(args.profile_timing):
                    timer.add("loop_total", elapsed)
                    msg += f" timings={timer.format()}"
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
