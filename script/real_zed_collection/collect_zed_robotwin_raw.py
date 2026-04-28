#!/usr/bin/env python3

from __future__ import annotations

import datetime
import os
import argparse
from queue import Empty, Full, Queue
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from script.real_zed_collection.real_zed_utils import (
    calibration_label_map_from_manifest,
    ensure_dir,
    load_three_zed_calibration,
    write_json,
)
from script.real_zed_collection.workspace_crop_utils import WorkspaceBounds, apply_workspace_crop_to_camera_frame


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "real_zed_collection.yaml"

ANSI_BLUE = "\033[94m"
ANSI_RESET = "\033[0m"


def parse_camera_labels(value: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


@dataclass
class Args:
    robot_port: int = 6001
    hostname: str = "127.0.0.1"
    save_data_path: str = "./real_data"
    project_name: str = "robotwin_real_zed"
    calibration_path: str = ""
    camera_labels: str = "global,left,right"
    zed_serials: list[int] = field(default_factory=list)
    zed_resolution: str = "HD1080"
    zed_fps: int = 15
    zed_depth_mode: str = "NEURAL"
    save_rgb_width: int = 640
    save_rgb_height: int = 360
    save_frequency_hz: float = 5.0
    writer_queue_size: int = 8
    manifest_flush_interval: int = 10
    compress_camera_frames: bool = False
    show_img: bool = False
    save_zed_xyzrgba: bool = False
    robot_camera_calibration_paths: list[str] = field(default_factory=list)
    workspace_calibration_path: str = ""
    workspace_crop_enabled: bool = False
    workspace_crop_x_min: float = -0.75
    workspace_crop_x_max: float = 0.75
    workspace_crop_y_min: float = -0.55
    workspace_crop_y_max: float = 0.75
    workspace_crop_z_min: float = 0.25
    workspace_crop_z_max: float = 1.25
    workspace_crop_margin_px: int = 32
    workspace_crop_resize_rgb: bool = False
    workspace_crop_debug_full_frame_interval: int = 0
    max_frames: int = -1


def _extract_config_path(argv: list[str]) -> tuple[Path | None, list[str]]:
    config_path = DEFAULT_CONFIG_PATH if DEFAULT_CONFIG_PATH.exists() else None
    remaining_args: list[str] = []
    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if arg == "--config":
            if idx + 1 >= len(argv):
                raise SystemExit("--config requires a path.")
            config_path = Path(argv[idx + 1]).expanduser().resolve()
            idx += 2
            continue
        if arg.startswith("--config="):
            config_path = Path(arg.split("=", 1)[1]).expanduser().resolve()
            idx += 1
            continue
        remaining_args.append(arg)
        idx += 1
    return config_path, remaining_args


def _load_args_defaults(config_path: Path | None) -> Args:
    if config_path is None:
        return Args()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")

    allowed = {item.name for item in fields(Args)}
    unknown = sorted(str(key) for key in data.keys() if str(key) not in allowed)
    if unknown:
        raise ValueError(f"Unknown config keys in {config_path}: {unknown}")

    defaults = Args(**{name: data[name] for name in allowed if name in data})
    print(f"[INFO] Loaded collection config: {config_path}")
    return defaults


def _parse_args_without_tyro(defaults: Args, cli_args: list[str]) -> Args:
    parser = argparse.ArgumentParser(description="Collect raw real-robot three-ZED data for RoboTwin/DP3.")
    for item in fields(Args):
        name = item.name
        default_value = getattr(defaults, name)
        flag = f"--{name}"
        if isinstance(default_value, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=default_value)
        elif isinstance(default_value, int) and not isinstance(default_value, bool):
            parser.add_argument(flag, type=int, default=default_value)
        elif isinstance(default_value, float):
            parser.add_argument(flag, type=float, default=default_value)
        elif isinstance(default_value, list):
            item_type = int if name == "zed_serials" else str
            parser.add_argument(flag, nargs="*", type=item_type, default=default_value)
        else:
            parser.add_argument(flag, type=str, default=str(default_value))
    parsed = parser.parse_args(cli_args)
    return Args(**vars(parsed))


def _blue_text(text: str) -> str:
    if hasattr(sys.stdout, "isatty") and sys.stdout.isatty():
        return f"{ANSI_BLUE}{text}{ANSI_RESET}"
    return text


def _resolve_user_path(path: str | Path) -> Path:
    return Path(os.path.expandvars(str(path))).expanduser().resolve()


def _print_args_summary(args: Args, config_path: Path | None, save_root: Path) -> None:
    print("[INFO] Collection configuration summary:")
    print(f"  - config_path: {config_path if config_path is not None else 'None (Args defaults)'}")
    print(f"  - output_root: {save_root}")
    for item in fields(Args):
        print(f"  - {item.name}: {getattr(args, item.name)}")


def parse_path_list(value: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def _snapshot_calibration_file(calibration_path: str | Path, episode_dir: Path, dst_name: str = "calibration_snapshot.yaml") -> str:
    src = _resolve_user_path(calibration_path)
    if not src.exists():
        raise FileNotFoundError(f"Calibration file does not exist: {src}")
    dst = episode_dir / str(dst_name)
    shutil.copy2(src, dst)
    return dst.name


def _snapshot_robot_camera_calibrations(paths: list[str], episode_dir: Path) -> list[dict[str, str]]:
    snapshots: list[dict[str, str]] = []
    if not paths:
        return snapshots
    snapshot_dir = episode_dir / "robot_camera_calibration_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    used_names: set[str] = set()
    for idx, raw_path in enumerate(paths):
        src = _resolve_user_path(raw_path)
        if not src.exists():
            raise FileNotFoundError(f"Robot-camera calibration file does not exist: {src}")
        dst_name = src.name
        if dst_name in used_names:
            dst_name = f"{idx:02d}_{dst_name}"
        used_names.add(dst_name)
        dst = snapshot_dir / dst_name
        shutil.copy2(src, dst)
        snapshots.append(
            {
                "source_path": str(src),
                "snapshot_path": str(dst.relative_to(episode_dir)),
            }
        )
    return snapshots


# Same button-state semantics as include/xtrainer_clover/experiments/run_control.py.
what_to_do = np.array(([0, 0, 0], [0, 0, 0]))
dt_time = np.array([20240507161455])
is_falling = np.array([0])


def button_monitor_realtime(agent) -> None:
    last_keys_status = np.array(([0, 0], [0, 0]))
    start_press_status = np.array(([0, 0], [0, 0]))
    keys_press_count = np.array(([0, 0, 0], [0, 0, 0]))
    tic = time.time()

    while True:
        now_keys = agent.get_keys()
        dev_keys = now_keys - last_keys_status
        for i in range(2):
            if dev_keys[i, 0] == -1:
                tic = time.time()
                start_press_status[i, 0] = 1
            if dev_keys[i, 0] == 1 and start_press_status[i, 0]:
                start_press_status[i, 0] = 0
                toc = time.time()
                if toc - tic < 0.5:
                    keys_press_count[i, 0] += 1
                    what_to_do[i, 0] = 1 if keys_press_count[i, 0] % 2 == 1 else 0
                    print(f"ButtonA: [{i}] {'unlock' if what_to_do[i, 0] else 'lock'}", what_to_do)
                elif toc - tic > 1:
                    keys_press_count[i, 1] += 1
                    what_to_do[i, 1] = 1 if keys_press_count[i, 1] % 2 == 1 else 0
                    print(f"ButtonA: [{i}] {'servo' if what_to_do[i, 1] else 'stop servo'}")

        for i in range(2):
            if dev_keys[i, 1] == -1:
                start_press_status[i, 1] = 1
            if dev_keys[i, 1] == 1:
                start_press_status[i, 1] = 0
                if keys_press_count[0, 2] % 2 == 1:
                    if keys_press_count[0, 1] % 2 == 1 or keys_press_count[1, 1] % 2 == 1:
                        what_to_do[0, 2] = 1
                        now_time = datetime.datetime.now()
                        dt_time[0] = int(now_time.strftime("%Y%m%d%H%M%S"))
                        keys_press_count[0, 2] += 1
                else:
                    what_to_do[0, 2] = 0
                    keys_press_count[0, 2] += 1

        last_keys_status = now_keys


class SharedZedFrame:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.frame: dict[str, Any] | None = None
        self.error: str | None = None

    def update(self, frame: dict[str, Any]) -> None:
        with self.lock:
            self.frame = frame
            self.error = None

    def set_error(self, error: str) -> None:
        with self.lock:
            self.error = error

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            if self.error:
                raise RuntimeError(self.error)
            if self.frame is None:
                raise RuntimeError("ZED frame is not ready yet.")
            return {key: np.copy(value) if isinstance(value, np.ndarray) else value for key, value in self.frame.items()}


@dataclass
class WriteTask:
    kind: str
    episode_dir: Path
    metadata: dict[str, Any] | None = None
    frame_idx: int = -1
    obs: dict[str, Any] | None = None
    action: np.ndarray | None = None
    camera_frames: dict[str, dict[str, Any]] | None = None


def _camera_matrix_from_zed_info(info) -> np.ndarray:
    calib = info.camera_configuration.calibration_parameters
    left = calib.left_cam
    return np.array(
        [[left.fx, 0.0, left.cx], [0.0, left.fy, left.cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _resize_rgb_for_storage(
    rgb: np.ndarray,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    src_h, src_w = rgb.shape[:2]
    if target_width <= 0 or target_height <= 0 or (src_w == target_width and src_h == target_height):
        return rgb.astype(np.uint8)
    return cv2.resize(rgb, (int(target_width), int(target_height)), interpolation=cv2.INTER_AREA).astype(np.uint8)


def _workspace_bounds_from_args(args: Args) -> WorkspaceBounds:
    return WorkspaceBounds(
        x_min=float(args.workspace_crop_x_min),
        x_max=float(args.workspace_crop_x_max),
        y_min=float(args.workspace_crop_y_min),
        y_max=float(args.workspace_crop_y_max),
        z_min=float(args.workspace_crop_z_min),
        z_max=float(args.workspace_crop_z_max),
    )


def _build_robot_snapshot(obs: dict[str, Any], action: np.ndarray) -> dict[str, np.ndarray]:
    joint_positions = np.asarray(obs.get("joint_positions", action), dtype=np.float32).reshape(-1).copy()
    joint_velocities = np.asarray(obs.get("joint_velocities", np.zeros_like(joint_positions)), dtype=np.float32).reshape(-1).copy()
    return {
        "joint_positions": joint_positions,
        "joint_velocities": joint_velocities,
    }


def _write_task_queue_loop(task_queue: Queue, manifest_flush_interval: int, compress_camera_frames: bool) -> None:
    manifests: dict[Path, dict[str, Any]] = {}
    flush_interval = max(1, int(manifest_flush_interval))
    pending_since_flush: dict[Path, int] = {}

    while True:
        task = task_queue.get()
        try:
            if task is None:
                break
            if not isinstance(task, WriteTask):
                continue

            if task.kind == "start_episode":
                metadata = dict(task.metadata or {})
                metadata.setdefault("frames", [])
                manifests[task.episode_dir] = metadata
                pending_since_flush[task.episode_dir] = 0
                _write_manifest(task.episode_dir, metadata)
                continue

            if task.kind == "frame":
                manifest = manifests.get(task.episode_dir)
                if manifest is None or task.obs is None or task.action is None or task.camera_frames is None:
                    continue
                frame_record = _save_raw_frame(
                    episode_dir=task.episode_dir,
                    frame_idx=int(task.frame_idx),
                    obs=task.obs,
                    action=np.asarray(task.action, dtype=np.float32),
                    camera_frames=task.camera_frames,
                    compress_camera_frames=bool(compress_camera_frames),
                )
                manifest["frames"].append(frame_record)
                pending_since_flush[task.episode_dir] = pending_since_flush.get(task.episode_dir, 0) + 1
                if pending_since_flush[task.episode_dir] >= flush_interval:
                    _write_manifest(task.episode_dir, manifest)
                    pending_since_flush[task.episode_dir] = 0
                continue

            if task.kind == "end_episode":
                manifest = manifests.get(task.episode_dir)
                if manifest is None:
                    continue
                if task.metadata:
                    manifest.update(task.metadata)
                _write_manifest(task.episode_dir, manifest)
                manifests.pop(task.episode_dir, None)
                pending_since_flush.pop(task.episode_dir, None)
        finally:
            task_queue.task_done()

    for episode_dir, manifest in manifests.items():
        _write_manifest(episode_dir, manifest)


def dh_transformation_matrix(theta, d, a, alpha):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
    return np.array(
        [
            [cos_theta, -sin_theta * cos_alpha, sin_theta * sin_alpha, a * cos_theta],
            [sin_theta, cos_theta * cos_alpha, -cos_theta * sin_alpha, a * sin_theta],
            [0, sin_alpha, cos_alpha, d],
            [0, 0, 0, 1],
        ]
    )


def claw_width(coef):
    claw_servo = 2.3818 - coef * 1.5401
    cos_claw_servo = np.cos(claw_servo)
    return 0.03 * cos_claw_servo + 0.5 * np.sqrt(0.0036 * cos_claw_servo**2 + 0.0028)


def forward_kinematics(q0, q1, q2, q3, q4, q5, y):
    dh_params = [
        (q0, 0.2234, 0, np.pi / 2),
        (q1 - np.pi / 2, 0, -0.280, 0),
        (q2, 0, -0.225, 0),
        (q3 - np.pi / 2, 0.1175, 0, np.pi / 2),
        (q4, 0.120, 0, -np.pi / 2),
        (q5, 0.088, 0, 0),
    ]
    t = np.eye(4)
    for params in dh_params:
        t = np.dot(t, dh_transformation_matrix(*params))
    t_tool = np.eye(4)
    t_tool[:3, 3] = np.array([0, y, 0.2])
    return np.dot(t, t_tool)[:3, 3]


def calculate_vel_pos(action, last_action, total_time):
    total_time = max(float(total_time), 1e-6)
    claw_left = claw_width(action[6])
    claw_right = claw_width(action[13])
    positions = {}
    vel = {}
    for side in ["left", "right"]:
        for paw in ["left", "right"]:
            coef = 1 if paw == "left" else -1
            claw = (claw_left if side == "left" else claw_right) * coef
            current_fk = forward_kinematics(*(action[0:6] if side == "left" else action[7:13]), claw)
            last_fk = forward_kinematics(*(last_action[0:6] if side == "left" else last_action[7:13]), claw)
            positions[f"{side}_{paw}"] = current_fk
            vel[f"{side}_{paw}"] = (current_fk - last_fk) / total_time
    return positions, vel


def is_within_safe_position(position, x_range, y_range, z_min):
    return x_range[0] <= position[0] <= x_range[1] and y_range[0] <= position[1] <= y_range[1] and position[2] > z_min


def check_pose_protection(positions, vel, servo_state):
    protect_err = False
    positions_mm = {key: value * 1000 for key, value in positions.items()}
    x_range_left = (-450, 290)
    x_range_right = (-290, 450)
    y_range = (-750, -160)
    z_range_left = 44
    z_range_right = 42

    if servo_state[0, 1]:
        if vel["left_left"][2] < -1 or vel["left_right"][2] < -1:
            print("[Warn]:The left robot speed of the TCP is moving too fast!")
            protect_err = True
        if not all(
            is_within_safe_position(positions_mm[pos], x_range_left, y_range, z_range_left)
            for pos in ["left_left", "left_right"]
        ):
            print("[Warn]:The left arm is out of the safe zone!")
            protect_err = True

    if servo_state[1, 1]:
        if vel["right_left"][2] < -1 or vel["right_right"][2] < -1:
            print("[Warn]:The right robot speed of the TCP is moving too fast!")
            protect_err = True
        if not all(
            is_within_safe_position(positions_mm[pos], x_range_right, y_range, z_range_right)
            for pos in ["right_left", "right_right"]
        ):
            print("[Warn]:The right arm is out of the safe zone!")
            protect_err = True
    return protect_err


def check_joint_safety(action):
    protect_err = False
    if not (action[2] < 0):
        print("[Warn]:The J3 joints of the left robotic arm are out of the safe position!")
        protect_err = True
    if not (action[9] > 0):
        print("[Warn]:The J3 joints of the right robotic arm are out of the safe position!")
        protect_err = True
    if protect_err:
        print(action)
    return protect_err


def _resolution_enum(sl, name: str):
    return {
        "HD2K": sl.RESOLUTION.HD2K,
        "HD1080": sl.RESOLUTION.HD1080,
        "HD720": sl.RESOLUTION.HD720,
        "VGA": sl.RESOLUTION.VGA,
    }.get(str(name).upper(), sl.RESOLUTION.HD720)


def _depth_mode_enum(sl, name: str):
    return {
        "NONE": sl.DEPTH_MODE.NONE,
        "PERFORMANCE": sl.DEPTH_MODE.PERFORMANCE,
        "QUALITY": sl.DEPTH_MODE.QUALITY,
        "ULTRA": sl.DEPTH_MODE.ULTRA,
        "NEURAL": getattr(sl.DEPTH_MODE, "NEURAL", sl.DEPTH_MODE.QUALITY),
        "NEURAL_PLUS": getattr(sl.DEPTH_MODE, "NEURAL_PLUS", getattr(sl.DEPTH_MODE, "NEURAL", sl.DEPTH_MODE.QUALITY)),
    }.get(str(name).upper(), getattr(sl.DEPTH_MODE, "NEURAL", sl.DEPTH_MODE.QUALITY))


def zed_capture_loop(
    *,
    label: str,
    serial: int,
    resolution: str,
    fps: int,
    depth_mode: str,
    save_rgb_width: int,
    save_rgb_height: int,
    save_xyzrgba: bool,
    workspace_crop_enabled: bool,
    t_workspace_from_cam: np.ndarray | None,
    workspace_camera_matrix: np.ndarray | None,
    workspace_bounds: WorkspaceBounds | None,
    workspace_crop_margin_px: int,
    workspace_crop_resize_rgb: bool,
    workspace_crop_debug_full_frame_interval: int,
    shared: SharedZedFrame,
    stop_event: threading.Event,
) -> None:
    try:
        import pyzed.sl as sl
    except Exception as e:
        shared.set_error(f"pyzed.sl import failed for camera {label}: {e}")
        return

    zed = sl.Camera()
    init = sl.InitParameters()
    init.set_from_serial_number(int(serial))
    init.camera_resolution = _resolution_enum(sl, resolution)
    init.camera_fps = int(fps)
    init.depth_mode = _depth_mode_enum(sl, depth_mode)
    init.coordinate_units = sl.UNIT.METER
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        shared.set_error(f"Failed to open ZED label={label}, serial={serial}: {status}")
        return

    camera_matrix = _camera_matrix_from_zed_info(zed.get_camera_information())
    runtime = sl.RuntimeParameters()
    image_mat = sl.Mat()
    depth_mat = sl.Mat()
    xyzrgba_mat = sl.Mat()
    frame_idx = 0
    try:
        while not stop_event.is_set():
            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                continue
            zed.retrieve_image(image_mat, sl.VIEW.LEFT)
            zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
            image_raw = image_mat.get_data()
            depth_raw = depth_mat.get_data()
            if image_raw is None or depth_raw is None:
                continue
            bgr = image_raw[:, :, :3] if image_raw.ndim == 3 and image_raw.shape[2] >= 3 else image_raw
            rgb = cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2RGB)
            depth_m = np.asarray(depth_raw, dtype=np.float32)
            save_camera_matrix = np.asarray(camera_matrix, dtype=np.float32)
            frame_extra: dict[str, Any] = {}
            if workspace_crop_enabled:
                if t_workspace_from_cam is None or workspace_bounds is None:
                    raise RuntimeError(f"Workspace crop enabled for {label}, but transform/bounds are missing.")
                projection_camera_matrix = (
                    np.asarray(workspace_camera_matrix, dtype=np.float64).reshape(3, 3)
                    if workspace_camera_matrix is not None
                    else camera_matrix
                )
                cropped = apply_workspace_crop_to_camera_frame(
                    rgb=rgb,
                    depth_m=depth_m,
                    camera_matrix=projection_camera_matrix,
                    t_workspace_from_cam=t_workspace_from_cam,
                    bounds=workspace_bounds,
                    margin_px=int(workspace_crop_margin_px),
                    resize_rgb_width=int(save_rgb_width) if bool(workspace_crop_resize_rgb) else 0,
                    resize_rgb_height=int(save_rgb_height) if bool(workspace_crop_resize_rgb) else 0,
                )
                rgb_save = np.asarray(cropped.pop("rgb"), dtype=np.uint8)
                depth_m = np.asarray(cropped.pop("depth_m"), dtype=np.float32)
                save_camera_matrix = np.asarray(cropped.pop("camera_matrix"), dtype=np.float32)
                frame_extra.update(cropped)
                frame_extra["zed_camera_matrix"] = np.asarray(camera_matrix, dtype=np.float32)
                debug_interval = int(workspace_crop_debug_full_frame_interval)
                if debug_interval > 0 and frame_idx % debug_interval == 0:
                    frame_extra["full_rgb_debug"] = _resize_rgb_for_storage(
                        rgb=rgb,
                        target_width=int(save_rgb_width),
                        target_height=int(save_rgb_height),
                    )
                    frame_extra["full_depth_m_debug"] = np.asarray(depth_raw, dtype=np.float32)
            else:
                rgb_save = _resize_rgb_for_storage(
                    rgb=rgb,
                    target_width=int(save_rgb_width),
                    target_height=int(save_rgb_height),
                )
            frame = {
                "label": label,
                "serial": int(serial),
                "frame_index": int(frame_idx),
                "timestamp_unix_sec": time.time(),
                "rgb": rgb_save,
                "depth_m": depth_m.astype(np.float32),
                "camera_matrix": save_camera_matrix,
            }
            frame.update(frame_extra)
            if save_xyzrgba:
                zed.retrieve_measure(xyzrgba_mat, sl.MEASURE.XYZRGBA)
                xyzrgba = xyzrgba_mat.get_data()
                if xyzrgba is not None:
                    xyzrgba_arr = np.asarray(xyzrgba, dtype=np.float32)
                    frame["xyzrgba"] = xyzrgba_arr.astype(np.float32)
            shared.update(frame)
            frame_idx += 1
    finally:
        zed.close()


def _resolve_cameras(args: Args) -> tuple[list[str], list[int]]:
    labels = parse_camera_labels(args.camera_labels)
    serials = [int(x) for x in args.zed_serials]
    if args.calibration_path:
        calib = load_three_zed_calibration(args.calibration_path)
        if not labels:
            labels = list(calib.keys())
        if not serials:
            serials = [int(calib[label].serial_number) for label in labels]
    if len(labels) != 3:
        raise ValueError(f"First version expects exactly 3 camera labels, got {labels}")
    if len(serials) != 3:
        raise ValueError("Provide exactly 3 ZED serials or a calibration file with serial_number entries.")
    print(f"[INFO] Collection camera label -> serial mapping: {dict(zip(labels, serials))}")
    return labels, serials


def _save_raw_frame(
    *,
    episode_dir: Path,
    frame_idx: int,
    obs: dict[str, Any],
    action: np.ndarray,
    camera_frames: dict[str, dict[str, Any]],
    compress_camera_frames: bool,
) -> dict[str, Any]:
    robot_path = episode_dir / f"robot_{frame_idx:06d}.npz"
    joint_positions = np.asarray(obs.get("joint_positions", action), dtype=np.float32).reshape(-1)
    if joint_positions.shape[0] != 14:
        raise ValueError(f"Expected 14-dim joint_positions, got {joint_positions.shape}")
    np.savez(
        robot_path,
        timestamp_unix_sec=np.asarray(time.time(), dtype=np.float64),
        joint_vector=joint_positions,
        joint_positions=joint_positions,
        joint_velocities=np.asarray(obs.get("joint_velocities", np.zeros_like(joint_positions)), dtype=np.float32),
        control=np.asarray(action, dtype=np.float32),
    )

    camera_rel_paths = {}
    for label, frame in camera_frames.items():
        camera_path = episode_dir / f"{label}_{frame_idx:06d}.npz"
        payload = {
            "timestamp_unix_sec": np.asarray(frame["timestamp_unix_sec"], dtype=np.float64),
            "camera_frame_index": np.asarray(frame["frame_index"], dtype=np.int64),
            "rgb": np.asarray(frame["rgb"], dtype=np.uint8),
            "depth_m": np.asarray(frame["depth_m"], dtype=np.float32),
            "camera_matrix": np.asarray(frame["camera_matrix"], dtype=np.float32),
        }
        for optional_key, dtype in (
            ("rgb_camera_matrix", np.float32),
            ("original_camera_matrix", np.float32),
            ("depth_crop_box_xyxy", np.int32),
            ("rgb_crop_box_xyxy", np.int32),
            ("original_depth_shape_hw", np.int32),
            ("original_rgb_shape_hw", np.int32),
            ("workspace_bounds_m", np.float32),
            ("t_workspace_from_camera", np.float32),
            ("zed_camera_matrix", np.float32),
            ("full_rgb_debug", np.uint8),
            ("full_depth_m_debug", np.float32),
            ("xyzrgba", np.float32),
        ):
            if optional_key in frame:
                payload[optional_key] = np.asarray(frame[optional_key], dtype=dtype)
        if compress_camera_frames:
            np.savez_compressed(camera_path, **payload)
        else:
            np.savez(camera_path, **payload)
        camera_rel_paths[label] = camera_path.name

    return {
        "frame_index": int(frame_idx),
        "timestamp_unix_sec": time.time(),
        "robot": robot_path.name,
        "cameras": camera_rel_paths,
    }


def _write_manifest(episode_dir: Path, manifest: dict[str, Any]) -> None:
    write_json(episode_dir / "manifest.json", manifest)


def main(args: Args, config_path: Path | None = None) -> None:
    include_dir = Path(__file__).resolve().parents[2] / "include" / "xtrainer_clover"
    sys.path.append(str(include_dir))
    from dobot_control.agents.agent import BimanualAgent
    from dobot_control.agents.dobot_agent import DobotAgent
    from dobot_control.env import RobotEnv
    from dobot_control.robots.robot_node import ZMQClientRobot
    from scripts.manipulate_utils import (
        dynamic_approach,
        load_ini_data_hands,
        robot_pose_init,
        servo_action_check,
        set_light,
    )

    labels, serials = _resolve_cameras(args)
    save_root = ensure_dir(_resolve_user_path(args.save_data_path) / args.project_name / "real_zed_raw")
    _print_args_summary(args, config_path, save_root)
    record_period_sec = 0.0 if float(args.save_frequency_hz) <= 0.0 else 1.0 / float(args.save_frequency_hz)
    workspace_bounds: WorkspaceBounds | None = None
    workspace_calib: dict[str, Any] = {}
    workspace_calibration_path = ""
    if bool(args.workspace_crop_enabled):
        workspace_bounds = _workspace_bounds_from_args(args)
        workspace_calibration_path = str(_resolve_user_path(args.workspace_calibration_path or args.calibration_path))
        workspace_calib = load_three_zed_calibration(workspace_calibration_path, frame_mode="workspace")
        workspace_label_by_raw_label = calibration_label_map_from_manifest(
            {"camera_labels": labels, "camera_serials": dict(zip(labels, serials))},
            workspace_calib,
            labels,
        )
        missing_workspace = [calib_label for calib_label in workspace_label_by_raw_label.values() if calib_label not in workspace_calib]
        if missing_workspace:
            raise ValueError(f"Workspace calibration missing camera labels: {missing_workspace}")
        print(f"[INFO] Workspace crop enabled with calibration: {workspace_calibration_path}")
        print(f"[INFO] Workspace bounds: {workspace_bounds.as_dict()}")
        if workspace_label_by_raw_label != {label: label for label in labels}:
            print(f"[WARN] Camera label/serial order differs from calibration. Using remap: {workspace_label_by_raw_label}")
    else:
        workspace_label_by_raw_label = {label: label for label in labels}

    robot_camera_calibration_paths = parse_path_list(args.robot_camera_calibration_paths)
    robot_camera_calibration_abs_paths = [str(_resolve_user_path(path)) for path in robot_camera_calibration_paths]
    for path in robot_camera_calibration_abs_paths:
        if not Path(path).exists():
            raise FileNotFoundError(f"Robot-camera calibration file does not exist: {path}")
    if robot_camera_calibration_abs_paths:
        print("[INFO] Robot-camera calibration snapshots enabled:")
        for path in robot_camera_calibration_abs_paths:
            print(f"  - {path}")

    stop_event = threading.Event()
    shared_by_label = {label: SharedZedFrame() for label in labels}
    writer_queue: Queue = Queue(maxsize=max(1, int(args.writer_queue_size)))
    writer_thread = threading.Thread(
        target=_write_task_queue_loop,
        kwargs={
            "task_queue": writer_queue,
            "manifest_flush_interval": int(args.manifest_flush_interval),
            "compress_camera_frames": bool(args.compress_camera_frames),
        },
        daemon=False,
    )
    writer_thread.start()
    camera_threads = []
    for label, serial in zip(labels, serials):
        thread = threading.Thread(
            target=zed_capture_loop,
            kwargs={
                "label": label,
                "serial": serial,
                "resolution": args.zed_resolution,
                "fps": int(args.zed_fps),
                "depth_mode": args.zed_depth_mode,
                "save_rgb_width": int(args.save_rgb_width),
                "save_rgb_height": int(args.save_rgb_height),
                "save_xyzrgba": bool(args.save_zed_xyzrgba),
                "workspace_crop_enabled": bool(args.workspace_crop_enabled),
                "t_workspace_from_cam": (
                    workspace_calib[workspace_label_by_raw_label[label]].t_world_from_cam.astype(np.float64)
                    if bool(args.workspace_crop_enabled)
                    else None
                ),
                "workspace_camera_matrix": (
                    workspace_calib[workspace_label_by_raw_label[label]].camera_matrix.astype(np.float64)
                    if bool(args.workspace_crop_enabled)
                    else None
                ),
                "workspace_bounds": workspace_bounds,
                "workspace_crop_margin_px": int(args.workspace_crop_margin_px),
                "workspace_crop_resize_rgb": bool(args.workspace_crop_resize_rgb),
                "workspace_crop_debug_full_frame_interval": int(args.workspace_crop_debug_full_frame_interval),
                "shared": shared_by_label[label],
                "stop_event": stop_event,
            },
            daemon=True,
        )
        thread.start()
        camera_threads.append(thread)
    time.sleep(2.0)
    print(f"ZED camera threads initialized: {dict(zip(labels, serials))}")

    _, hands_dict = load_ini_data_hands()
    left_agent = DobotAgent(which_hand="LEFT", dobot_config=hands_dict["HAND_LEFT"])
    right_agent = DobotAgent(which_hand="RIGHT", dobot_config=hands_dict["HAND_RIGHT"])
    agent = BimanualAgent(left_agent, right_agent)

    robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
    env = RobotEnv(robot_client)
    env.set_do_status([1, 0])
    env.set_do_status([2, 0])
    env.set_do_status([3, 0])
    robot_pose_init(env)
    print("robot init success")

    last_status = np.array(([0, 0, 0], [0, 0, 0]))
    button_thread = threading.Thread(target=button_monitor_realtime, args=(agent,), daemon=True)
    button_thread.start()
    print("button thread init success")

    start_servo = False
    curr_light = "dark"
    total_time = 0.04
    safe_limit = 0
    frame_count = 0
    active_episode_dir: Path | None = None
    record_frame_index = 0
    dropped_frame_count = 0
    last_drop_warn_sec = 0.0
    next_record_time = 0.0

    try:
        while True:
            tic = time.time()
            for thread, label in zip(camera_threads, labels):
                if not thread.is_alive():
                    raise RuntimeError(f"ZED thread stopped: {label}")
            if np.any(is_falling):
                raise RuntimeError("sensor detection")

            action = agent.act({})
            dev_what_to_do = what_to_do.copy() - last_status
            last_status = what_to_do.copy()

            for i in range(2):
                if dev_what_to_do[i, 0] != 0:
                    agent.set_torque(i, not what_to_do[i, 0])

            if dev_what_to_do[0, 1] == 1 or dev_what_to_do[1, 1] == 1:
                print("dynamic approach")
                for i in range(2):
                    if what_to_do[i, 1]:
                        agent.set_torque(i, True)
                flag_in = np.array([what_to_do[0, 1], what_to_do[1, 1]])
                last_action = dynamic_approach(env, agent, flag_in)
                for i in range(2):
                    if what_to_do[i, 0] and what_to_do[i, 1]:
                        agent.set_torque(i, False)
                start_servo = True
                obs = env.get_obs()
                if curr_light != "green":
                    curr_light = set_light(env, "yellow", 1)

            if dev_what_to_do[0, 1] == -1 or dev_what_to_do[1, 1] == -1:
                if what_to_do[0, 1] == 0 and what_to_do[1, 1] == 0:
                    set_light(env, "green", 0)

            if (what_to_do[0, 1] or what_to_do[1, 1]) and start_servo:
                action = agent.act({})
                flag_in = np.array([what_to_do[0, 1], what_to_do[1, 1]])
                err3, action = servo_action_check(action, last_action, flag_in)
                assert err3 != 0, set_light(env, "red", 1)
                if safe_limit < 1:
                    safe_limit += 1
                else:
                    positions, vel = calculate_vel_pos(action, last_action, total_time)
                    protect_err = [
                        check_pose_protection(positions, vel, what_to_do),
                        check_joint_safety(action),
                    ]
                    if any(protect_err):
                        set_light(env, "red", 1)
                        time.sleep(1)
                        raise RuntimeError("Safety protection triggered.")

                if dev_what_to_do[0, 2] == 1:
                    curr_light = set_light(env, "green", 1)
                    active_episode_dir = ensure_dir(save_root / f"episode_{int(dt_time[0])}")
                    calibration_snapshot_rel = ""
                    if args.calibration_path:
                        calibration_snapshot_rel = _snapshot_calibration_file(args.calibration_path, active_episode_dir)
                    workspace_snapshot_rel = ""
                    if workspace_calibration_path and workspace_calibration_path != str(_resolve_user_path(args.calibration_path)):
                        workspace_snapshot_rel = _snapshot_calibration_file(
                            workspace_calibration_path,
                            active_episode_dir,
                            dst_name="workspace_calibration_snapshot.yaml",
                        )
                    robot_camera_calibration_snapshots = _snapshot_robot_camera_calibrations(
                        robot_camera_calibration_abs_paths,
                        active_episode_dir,
                    )
                    print(_blue_text(f"[RECORD] START -> {active_episode_dir}"))
                    writer_queue.put(
                        WriteTask(
                            kind="start_episode",
                            episode_dir=active_episode_dir,
                            metadata={
                                "format": "robotwin_real_zed_raw_v1",
                                "created_at_unix_sec": time.time(),
                                "camera_labels": labels,
                                "camera_serials": dict(zip(labels, serials)),
                                "calibration_path": str(_resolve_user_path(args.calibration_path)) if args.calibration_path else "",
                                "calibration_snapshot_path": calibration_snapshot_rel,
                                "workspace_calibration_path": workspace_calibration_path,
                                "workspace_calibration_snapshot_path": workspace_snapshot_rel,
                                "robot_camera_calibration_paths": robot_camera_calibration_abs_paths,
                                "robot_camera_calibration_snapshots": robot_camera_calibration_snapshots,
                                "workspace_crop_enabled": bool(args.workspace_crop_enabled),
                                "workspace_crop_bounds_m": workspace_bounds.as_dict() if workspace_bounds is not None else {},
                                "workspace_crop_margin_px": int(args.workspace_crop_margin_px),
                                "workspace_crop_resize_rgb": bool(args.workspace_crop_resize_rgb),
                                "workspace_crop_debug_full_frame_interval": int(
                                    args.workspace_crop_debug_full_frame_interval
                                ),
                                "camera_label_to_calibration_label": workspace_label_by_raw_label,
                                "save_rgb_width": int(args.save_rgb_width),
                                "save_rgb_height": int(args.save_rgb_height),
                                "save_frequency_hz": float(args.save_frequency_hz),
                                "compress_camera_frames": bool(args.compress_camera_frames),
                                "dropped_frames_due_to_backpressure": 0,
                            },
                        )
                    )
                    record_frame_index = 0
                    dropped_frame_count = 0
                    next_record_time = 0.0
                elif dev_what_to_do[0, 2] == -1:
                    curr_light = set_light(env, "yellow", 1)
                    if active_episode_dir is not None:
                        print(
                            _blue_text(
                                f"[RECORD] STOP -> {active_episode_dir} | queued_frames={record_frame_index} | dropped_frames={dropped_frame_count}"
                            )
                        )
                        writer_queue.put(
                            WriteTask(
                                kind="end_episode",
                                episode_dir=active_episode_dir,
                                metadata={"dropped_frames_due_to_backpressure": int(dropped_frame_count)},
                            )
                        )
                    active_episode_dir = None

                if what_to_do[0, 2] == 1:
                    now_sec = time.time()
                    if active_episode_dir is not None and now_sec >= next_record_time:
                        camera_frames = {label: shared_by_label[label].snapshot() for label in labels}
                        try:
                            writer_queue.put_nowait(
                                WriteTask(
                                    kind="frame",
                                    episode_dir=active_episode_dir,
                                    frame_idx=int(record_frame_index),
                                    obs=_build_robot_snapshot(obs, action),
                                    action=np.asarray(action, dtype=np.float32).copy(),
                                    camera_frames=camera_frames,
                                )
                            )
                            frame_count += 1
                            record_frame_index += 1
                            next_record_time = now_sec + record_period_sec
                            print(f"queued raw frame {frame_count} -> {active_episode_dir}", end="\r")
                        except Full:
                            dropped_frame_count += 1
                            if now_sec - last_drop_warn_sec > 1.0:
                                print(
                                    f"[WARN] Writer queue full, dropping recorded frames to protect control loop. "
                                    f"dropped={dropped_frame_count}"
                                )
                                last_drop_warn_sec = now_sec
                            next_record_time = now_sec + record_period_sec
                    if args.max_frames > 0 and frame_count >= int(args.max_frames):
                        break

                obs = env.step(action, flag_in)
                obs["joint_positions"][6] = action[6]
                obs["joint_positions"][13] = action[13]
                last_action = action
            else:
                start_servo = False
                safe_limit = 0

            if args.show_img:
                panels = []
                for label in labels:
                    rgb = shared_by_label[label].snapshot()["rgb"]
                    panels.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                canvas = cv2.hconcat([cv2.resize(panel, (480, 270)) for panel in panels])
                cv2.imshow("real_zed_collection", canvas)
                cv2.waitKey(1)

            total_time = time.time() - tic
    finally:
        stop_event.set()
        if active_episode_dir is not None:
            print(
                _blue_text(
                    f"[RECORD] STOP -> {active_episode_dir} | queued_frames={record_frame_index} | dropped_frames={dropped_frame_count}"
                )
            )
            writer_queue.put(
                WriteTask(
                    kind="end_episode",
                    episode_dir=active_episode_dir,
                    metadata={"dropped_frames_due_to_backpressure": int(dropped_frame_count)},
                )
            )
        writer_queue.put(None)
        writer_thread.join()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    config_path, cli_args = _extract_config_path(sys.argv[1:])
    default_args = _load_args_defaults(config_path)
    try:
        import tyro
    except ModuleNotFoundError:
        parsed_args = _parse_args_without_tyro(default_args, cli_args)
    else:
        parsed_args = tyro.cli(Args, args=cli_args, default=default_args)
    main(parsed_args, config_path=config_path)
