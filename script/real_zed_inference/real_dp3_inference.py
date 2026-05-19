#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import csv
import os
import re
import select
import sys
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DP3_ROOT = REPO_ROOT / "policy" / "DP3"
DP3_SCRIPTS_ROOT = DP3_ROOT / "scripts"
DEFAULT_WORKSPACE_CALIBRATION = REPO_ROOT / "script" / "real_zed_collection" / "calibration" / "three_camera_workspace_extrinsics.yaml"
DEFAULT_ROBOT_CAMERA_CALIBRATION_DIR = REPO_ROOT / "script" / "real_zed_collection" / "calibration"
DEFAULT_SAM2_ROOT = Path(os.environ.get("SAM2_STREAMING_ROOT", str(REPO_ROOT / "include" / "SAM2_streaming"))).expanduser()
DEFAULT_SAM2_CHECKPOINT = Path(
    # os.environ.get("SAM2_CHECKPOINT", str(Path.home() / "DataModel" / "sam2.1" / "sam2.1_hiera_large.pt"))
    os.environ.get("SAM2_CHECKPOINT", str(Path.home() / "DataModel" / "sam2.1" / "sam2.1_hiera_base_plus.pt"))
).expanduser()
DEFAULT_SEMANTIC_CKPT_A = Path(
    os.environ.get(
        "SEMANTIC_CKPT_A",
        str(Path.home() / "github" / "3d_semantic_train" / "outputs" / "utonia_universal_field" / "Mug_semantic" / "mug.pt"),
    )
).expanduser()
ARM_JOINT_INDICES = np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12], dtype=np.int64)
GRIPPER_INDICES = np.asarray([6, 13], dtype=np.int64)

def _preprocess_print(*args):
    """Preprocess the input for colorful printing.

    Args:
        args (Any|None): One or more any type arguments to print.

    Returns:
        str Msg to print.
    """
    str_args = ""
    for a in args:
        if isinstance(a, np.ndarray):
            str_args += "\n" + np.array2string(a, separator=", ")
        else:
            str_args += " " + str(a)
    separate_with_newline = str_args.split("\n")
    extra_whitespaces_removed = []
    for b in separate_with_newline:
        extra_whitespaces_removed.append(" ".join(b.split()))
    return "\n".join(extra_whitespaces_removed)


def print_debug(*args):
    """Print information with green."""
    print("".join(["\033[1m\033[92m", _preprocess_print(*args), "\033[0m"]))


def print_info(*args):
    """Print information with sky blue."""
    print("".join(["\033[1m\033[94m", _preprocess_print(*args), "\033[0m"]))


def print_warning(*args):
    """Print a warning with yellow."""
    print("".join(["\033[1m\033[93m", _preprocess_print(*args), "\033[0m"]))


def print_error(*args):
    """Print error with red."""
    print("".join(["\033[1m\033[91m", _preprocess_print(*args), "\033[0m"]))


class KeyboardCommandListener:
    """Non-blocking terminal command listener for long real-robot runs."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        reset_key: str = "r",
        quit_key: str = "q",
        stream=None,
    ):
        self.enabled = bool(enabled)
        self.reset_key = str(reset_key or "r")[:1].lower()
        self.quit_key = str(quit_key or "q")[:1].lower()
        self.stream = sys.stdin if stream is None else stream
        self._commands: deque[str] = deque()
        self._condition = threading.Condition()
        self._reset_event = threading.Event()
        self._quit_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._old_termios = None

    @property
    def reset_event(self) -> threading.Event:
        return self._reset_event

    def reset_requested(self) -> bool:
        return self._reset_event.is_set()

    def quit_requested(self) -> bool:
        return self._quit_event.is_set()

    def clear_reset_request(self) -> None:
        self._reset_event.clear()

    def pop_commands(self) -> list[str]:
        with self._condition:
            commands = list(self._commands)
            self._commands.clear()
            return commands

    def handle_key(self, key: str) -> None:
        normalized = str(key or "")[:1].lower()
        if not normalized:
            return
        with self._condition:
            if normalized == self.reset_key:
                self._commands.append("reset")
                self._reset_event.set()
                self._condition.notify_all()
            elif normalized == self.quit_key:
                self._commands.append("quit")
                self._quit_event.set()
                self._condition.notify_all()

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        if not hasattr(self.stream, "fileno") or not self.stream.isatty():
            print_warning("[WARN] Keyboard reset disabled because stdin is not a TTY.")
            return
        try:
            import termios
            import tty

            fd = self.stream.fileno()
            self._old_termios = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        except Exception as exc:
            print_warning(f"[WARN] Keyboard reset disabled; failed to configure terminal: {exc}")
            self._old_termios = None
            return
        self._thread = threading.Thread(target=self._run, name="KeyboardCommandListener", daemon=True)
        self._thread.start()
        print_info(
            f"[INFO] Keyboard commands enabled: press '{self.reset_key}' to reset episode, "
            f"'{self.quit_key}' to quit."
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._old_termios is not None:
            try:
                import termios

                termios.tcsetattr(self.stream.fileno(), termios.TCSADRAIN, self._old_termios)
            except Exception as exc:
                print_warning(f"[WARN] Failed to restore terminal mode: {exc}")
            self._old_termios = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                ready, _, _ = select.select([self.stream], [], [], 0.05)
                if ready:
                    key = self.stream.read(1)
                    self.handle_key(key)
            except Exception as exc:
                print_warning(f"[WARN] Keyboard command listener stopped: {exc}")
                return



for path in (REPO_ROOT, DP3_ROOT, DP3_SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from script.real_zed_collection.collect_zed_robotwin_raw import (  # noqa: E402
    Args as CollectionArgs,
    SharedZedFrame,
    _resolve_cameras,
    zed_capture_loop,
)
from script.real_zed_collection.real_zed_utils import (  # noqa: E402
    calibration_label_map_from_manifest,
    depth_rgb_to_point_cloud,
    deterministic_resample,
    load_three_zed_calibration,
    merge_point_clouds,
    transform_point_cloud,
)
from script.real_zed_collection.postprocess.postprocess_raw_to_robotwin_hdf5 import (  # noqa: E402
    _load_yaml_mapping,
    _output_frame_transforms,
)
from script.real_zed_collection.workspace_crop_utils import WorkspaceBounds, invert_transform  # noqa: E402
from sam2_pointcloud_utils import run_camera_tasks_parallel  # noqa: E402
from eef_action_utils import (  # noqa: E402
    action20_to_eef14,
    eef14_world_to_base,
    joint14_to_eef14,
    load_world_from_base_transforms,
)


def parse_placeholder_list(text: str | Sequence[str]) -> list[str]:
    if isinstance(text, str):
        return [item.strip() for item in text.split(",") if item.strip()]
    return [str(item).strip() for item in text if str(item).strip()]


def parse_camera_labels(text: str | Sequence[str]) -> list[str]:
    if isinstance(text, str):
        return [item.strip() for item in text.split(",") if item.strip()]
    return [str(item).strip() for item in text if str(item).strip()]


def parse_serials(text: str | Sequence[int] | Sequence[str]) -> list[int]:
    if text is None:
        return []
    if isinstance(text, str):
        if not text.strip():
            return []
        return [int(item.strip()) for item in text.split(",") if item.strip()]
    return [int(item) for item in text]


def resolve_path(path: str | Path) -> Path:
    return Path(os.path.expandvars(str(path))).expanduser().resolve()


def make_workspace_bounds(args: argparse.Namespace) -> WorkspaceBounds:
    return WorkspaceBounds(
        x_min=float(args.workspace_crop_x_min),
        x_max=float(args.workspace_crop_x_max),
        y_min=float(args.workspace_crop_y_min),
        y_max=float(args.workspace_crop_y_max),
        z_min=float(args.workspace_crop_z_min),
        z_max=float(args.workspace_crop_z_max),
    )


def crop_point_cloud_by_bounds(point_cloud: np.ndarray, bounds: WorkspaceBounds | None) -> np.ndarray:
    pc = np.asarray(point_cloud, dtype=np.float32)
    if bounds is None or pc.size == 0:
        return pc
    mask = (
        (pc[:, 0] >= float(bounds.x_min))
        & (pc[:, 0] <= float(bounds.x_max))
        & (pc[:, 1] >= float(bounds.y_min))
        & (pc[:, 1] <= float(bounds.y_max))
        & (pc[:, 2] >= float(bounds.z_min))
        & (pc[:, 2] <= float(bounds.z_max))
    )
    return pc[mask]


def rgb_to_depth_shape(rgb: np.ndarray, depth_shape_hw: tuple[int, int]) -> np.ndarray:
    rgb_arr = np.asarray(rgb, dtype=np.uint8)
    if rgb_arr.shape[:2] == depth_shape_hw:
        return rgb_arr
    return cv2.resize(rgb_arr, (int(depth_shape_hw[1]), int(depth_shape_hw[0])), interpolation=cv2.INTER_LINEAR)


def camera_frame_to_world_pc(
    *,
    camera_frame: Mapping[str, Any],
    camera_matrix: np.ndarray,
    t_world_from_cam: np.ndarray,
    min_depth_m: float,
    max_depth_m: float,
) -> np.ndarray:
    rgb = np.asarray(camera_frame["rgb"], dtype=np.uint8)
    depth_m = np.asarray(camera_frame["depth_m"], dtype=np.float32)
    rgb = rgb_to_depth_shape(rgb, depth_m.shape)
    pc_cam = depth_rgb_to_point_cloud(
        depth_m=depth_m,
        rgb=rgb,
        camera_matrix=np.asarray(camera_matrix, dtype=np.float32).reshape(3, 3),
        min_depth_m=float(min_depth_m),
        max_depth_m=float(max_depth_m),
    )
    return transform_point_cloud(pc_cam, np.asarray(t_world_from_cam, dtype=np.float32).reshape(4, 4))


def camera_frame_to_output_pc(
    *,
    camera_frame: Mapping[str, Any],
    camera_matrix: np.ndarray,
    t_workspace_from_cam: np.ndarray,
    workspace_bounds: WorkspaceBounds | None,
    t_output_from_cam: np.ndarray,
    min_depth_m: float,
    max_depth_m: float,
) -> np.ndarray:
    rgb = np.asarray(camera_frame["rgb"], dtype=np.uint8)
    depth_m = np.asarray(camera_frame["depth_m"], dtype=np.float32)
    rgb = rgb_to_depth_shape(rgb, depth_m.shape)
    pc_cam = depth_rgb_to_point_cloud(
        depth_m=depth_m,
        rgb=rgb,
        camera_matrix=np.asarray(camera_matrix, dtype=np.float32).reshape(3, 3),
        min_depth_m=float(min_depth_m),
        max_depth_m=float(max_depth_m),
    )
    if workspace_bounds is None:
        return transform_point_cloud(pc_cam, np.asarray(t_output_from_cam, dtype=np.float32).reshape(4, 4))
    pc_workspace = transform_point_cloud(pc_cam, np.asarray(t_workspace_from_cam, dtype=np.float32).reshape(4, 4))
    pc_workspace = crop_point_cloud_by_bounds(pc_workspace, workspace_bounds)
    if pc_workspace.size == 0:
        return np.zeros((0, 6), dtype=np.float32)
    t_output_from_workspace = np.asarray(t_output_from_cam, dtype=np.float64).reshape(4, 4) @ invert_transform(
        np.asarray(t_workspace_from_cam, dtype=np.float64).reshape(4, 4)
    )
    return transform_point_cloud(pc_workspace, t_output_from_workspace)


@dataclass
class LiveZedCameras:
    labels: list[str]
    serials: list[int]
    calibrations: dict[str, Any]
    label_to_calib: dict[str, str]
    workspace_bounds: WorkspaceBounds | None
    output_frame: str
    t_workspace_from_cam_by_label: dict[str, np.ndarray]
    t_output_from_cam_by_label: dict[str, np.ndarray]
    shared_by_label: dict[str, SharedZedFrame]
    stop_event: threading.Event
    threads: list[threading.Thread]

    def stop(self) -> None:
        self.stop_event.set()
        for thread in self.threads:
            thread.join(timeout=2.0)


def start_zed_cameras(args: argparse.Namespace) -> LiveZedCameras:
    if bool(args.workspace_crop_enabled) and str(args.frame_mode) != "workspace":
        raise ValueError("--workspace_crop_enabled requires --frame_mode workspace")
    calibration_path = resolve_path(args.calibration_path)
    collection_args = CollectionArgs(
        calibration_path=str(calibration_path),
        camera_labels=args.camera_labels,
        zed_serials=parse_serials(args.zed_serials),
        zed_resolution=args.zed_resolution,
        zed_fps=int(args.zed_fps),
        zed_depth_mode=args.zed_depth_mode,
        zed_auto_exposure=bool(args.zed_auto_exposure),
        zed_exposure=int(args.zed_exposure),
        zed_gain=int(args.zed_gain),
        zed_whitebalance_temp=int(args.zed_whitebalance_temp),
        save_rgb_width=int(args.save_rgb_width),
        save_rgb_height=int(args.save_rgb_height),
        workspace_crop_enabled=bool(args.workspace_crop_enabled),
        workspace_crop_x_min=float(args.workspace_crop_x_min),
        workspace_crop_x_max=float(args.workspace_crop_x_max),
        workspace_crop_y_min=float(args.workspace_crop_y_min),
        workspace_crop_y_max=float(args.workspace_crop_y_max),
        workspace_crop_z_min=float(args.workspace_crop_z_min),
        workspace_crop_z_max=float(args.workspace_crop_z_max),
        workspace_crop_margin_px=int(args.workspace_crop_margin_px),
        workspace_crop_resize_rgb=bool(args.workspace_crop_resize_rgb),
        workspace_crop_debug_full_frame_interval=0,
    )
    labels, serials = _resolve_cameras(collection_args)
    calibrations = load_three_zed_calibration(calibration_path, frame_mode=args.frame_mode)
    manifest_like = {"camera_labels": labels, "camera_serials": dict(zip(labels, serials))}
    label_to_calib = (
        calibration_label_map_from_manifest(manifest_like, calibrations, labels)
        if bool(args.serial_remap)
        else {label: label for label in labels}
    )
    missing = [calib_label for calib_label in label_to_calib.values() if calib_label not in calibrations]
    if missing:
        raise ValueError(f"Calibration is missing labels required by live cameras: {missing}")

    workspace_bounds = make_workspace_bounds(args) if bool(args.workspace_crop_enabled) else None
    robot_camera_calibration = load_robot_camera_calibration_for_output_frame(args)
    output_frame, t_output_from_cam_by_label = _output_frame_transforms(
        calib=calibrations,
        labels=labels,
        label_to_calib=label_to_calib,
        output_frame=args.output_frame,
        robot_camera_calibration=robot_camera_calibration,
    )
    t_workspace_from_cam_by_label = {
        label: np.asarray(calibrations[label_to_calib[label]].t_world_from_cam, dtype=np.float64).reshape(4, 4)
        for label in labels
    }
    shared_by_label = {label: SharedZedFrame() for label in labels}
    stop_event = threading.Event()
    threads: list[threading.Thread] = []
    for label, serial in zip(labels, serials):
        calib_label = label_to_calib[label]
        thread = threading.Thread(
            target=zed_capture_loop,
            kwargs={
                "label": label,
                "serial": int(serial),
                "resolution": args.zed_resolution,
                "fps": int(args.zed_fps),
                "depth_mode": args.zed_depth_mode,
                "zed_auto_exposure": bool(args.zed_auto_exposure),
                "zed_exposure": int(args.zed_exposure),
                "zed_gain": int(args.zed_gain),
                "zed_whitebalance_temp": int(args.zed_whitebalance_temp),
                "save_rgb_width": int(args.save_rgb_width),
                "save_rgb_height": int(args.save_rgb_height),
                "save_xyzrgba": False,
                "workspace_crop_enabled": bool(args.workspace_crop_enabled),
                "t_workspace_from_cam": (
                    calibrations[calib_label].t_world_from_cam.astype(np.float64)
                    if bool(args.workspace_crop_enabled)
                    else None
                ),
                # Use the live ZED intrinsics for image-space ROI projection so
                # changing --zed_resolution does not reuse stale calibration intrinsics.
                "workspace_camera_matrix": None,
                "workspace_bounds": workspace_bounds,
                "workspace_crop_margin_px": int(args.workspace_crop_margin_px),
                "workspace_crop_resize_rgb": bool(args.workspace_crop_resize_rgb),
                "workspace_crop_debug_full_frame_interval": 0,
                "shared": shared_by_label[label],
                "stop_event": stop_event,
            },
            daemon=True,
        )
        thread.start()
        threads.append(thread)

    live = LiveZedCameras(
        labels=labels,
        serials=serials,
        calibrations=calibrations,
        label_to_calib=label_to_calib,
        workspace_bounds=workspace_bounds,
        output_frame=output_frame,
        t_workspace_from_cam_by_label=t_workspace_from_cam_by_label,
        t_output_from_cam_by_label=t_output_from_cam_by_label,
        shared_by_label=shared_by_label,
        stop_event=stop_event,
        threads=threads,
    )
    print(f"[INFO] Live point cloud output_frame={output_frame}")
    wait_for_cameras(live, timeout_sec=float(args.camera_warmup_timeout_sec))
    return live


def load_robot_camera_calibration_for_output_frame(args: argparse.Namespace) -> dict[str, Any] | None:
    output_frame = str(args.output_frame)
    if output_frame in {"source", "workspace"}:
        return None
    if output_frame not in {"left_base", "right_base"}:
        raise ValueError("output_frame must be one of: source, workspace, left_base, right_base")
    arm = output_frame.split("_", 1)[0]
    if str(args.robot_camera_calibration_path).strip():
        path = resolve_path(args.robot_camera_calibration_path)
    else:
        path = DEFAULT_ROBOT_CAMERA_CALIBRATION_DIR / f"robot_camera_apriltag_{arm}_global.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"output_frame={output_frame!r} requires robot-camera calibration, but file does not exist: {path}"
        )
    data = _load_yaml_mapping(path)
    if str(data.get("arm", "")).strip() != arm:
        raise ValueError(f"Robot-camera calibration arm mismatch: expected {arm}, got {data.get('arm')} from {path}")
    data = dict(data)
    data["_path"] = str(path)
    print(f"[INFO] Using robot-camera calibration for {output_frame}: {path}")
    return data


def wait_for_cameras(live: LiveZedCameras, timeout_sec: float) -> None:
    deadline = time.time() + float(timeout_sec)
    ready: set[str] = set()
    last_error = ""
    while time.time() < deadline:
        for label in live.labels:
            if label in ready:
                continue
            try:
                live.shared_by_label[label].snapshot()
                ready.add(label)
            except RuntimeError as exc:
                last_error = str(exc)
        if len(ready) == len(live.labels):
            print(f"[INFO] ZED cameras ready: {live.labels}")
            return
        time.sleep(0.05)
    raise TimeoutError(f"Timed out waiting for ZED cameras. ready={sorted(ready)}, last_error={last_error}")


def snapshot_frames(live: LiveZedCameras) -> dict[str, dict[str, Any]]:
    return {label: live.shared_by_label[label].snapshot() for label in live.labels}


def build_robotwin_observation(
    *,
    args: argparse.Namespace,
    live: LiveZedCameras,
    agent_vector: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    frames_by_label = snapshot_frames(live)
    camera_obs: dict[str, dict[str, Any]] = {}

    def build_camera(label: str):
        calib = live.calibrations[live.label_to_calib[label]]
        frame = frames_by_label[label]
        depth_m = np.asarray(frame["depth_m"], dtype=np.float32)
        rgb_aligned = rgb_to_depth_shape(np.asarray(frame["rgb"], dtype=np.uint8), depth_m.shape)
        camera_matrix = np.asarray(frame.get("camera_matrix", calib.camera_matrix), dtype=np.float32).reshape(3, 3)
        t_output_from_cam = np.asarray(live.t_output_from_cam_by_label[label], dtype=np.float32).reshape(4, 4)
        t_workspace_from_cam = np.asarray(live.t_workspace_from_cam_by_label[label], dtype=np.float32).reshape(4, 4)

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
        return label, scene_chunk, {
            "rgb": rgb_aligned.astype(np.uint8),
            "depth": depth_m.astype(np.float32),
            "intrinsic_cv": camera_matrix.astype(np.float32),
            # SAM2 projection utilities expect world -> camera here.
            "extrinsic_cv": invert_transform(t_output_from_cam).astype(np.float32),
            "cam2world_gl": t_output_from_cam.astype(np.float32),
            "t_workspace_from_cam": t_workspace_from_cam.astype(np.float32),
            "workspace_bounds_m": (
                np.asarray(
                    [
                        live.workspace_bounds.x_min,
                        live.workspace_bounds.x_max,
                        live.workspace_bounds.y_min,
                        live.workspace_bounds.y_max,
                        live.workspace_bounds.z_min,
                        live.workspace_bounds.z_max,
                    ],
                    dtype=np.float32,
                )
                if live.workspace_bounds is not None
                else None
            ),
        }

    scene_chunks: list[np.ndarray] = []
    results = run_camera_tasks_parallel(
        live.labels,
        build_camera,
        max_workers=int(getattr(args, "parallel_camera_workers", 0)),
    )
    for label in live.labels:
        _label, scene_chunk, camera_info = results[str(label)]
        scene_chunks.append(scene_chunk)
        camera_obs[str(_label)] = camera_info

    dense_scene = merge_point_clouds(scene_chunks)
    scene_point_cloud = deterministic_resample(dense_scene, int(args.scene_point_num))
    observation = {
        "joint_action": {"vector": np.asarray(agent_vector, dtype=np.float32).reshape(14)},
        "pointcloud": scene_point_cloud.astype(np.float32),
        "observation": camera_obs,
    }
    return observation, dense_scene.astype(np.float32)


def maybe_show_cameras(live_observation: dict[str, Any], labels: Sequence[str]) -> None:
    images = []
    for label in labels:
        image = live_observation.get("observation", {}).get(label, {}).get("rgb")
        if image is None:
            continue
        arr = np.asarray(image, dtype=np.uint8)
        cv2.putText(arr, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        images.append(arr)
    if not images:
        return
    min_h = min(img.shape[0] for img in images)
    resized = [
        cv2.resize(img, (int(img.shape[1] * min_h / img.shape[0]), min_h), interpolation=cv2.INTER_AREA)
        if img.shape[0] != min_h
        else img
        for img in images
    ]
    canvas = np.hstack(resized)
    cv2.imshow("real_zed_dp3_rgb", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)


def load_sam2_runtime(args: argparse.Namespace, placeholders: Sequence[str], camera_names: Sequence[str]):
    from sam2_pointcloud_utils import (  # noqa: PLC0415
        build_sam2_tracker_factory,
        extract_placeholder_point_clouds_sam2_online,
        load_sam2_bbox_prompt_file,
    )

    tracker_factory = build_sam2_tracker_factory(
        placeholders=placeholders,
        sam2_root=args.sam2_root,
        config=args.sam2_config,
        checkpoint=args.sam2_checkpoint,
        device=args.sam2_device,
        autocast_dtype=args.sam2_autocast_dtype,
    )
    prompts: dict[str, dict[str, object]] = {}
    prompt_path = str(args.sam2_bbox_prompt_path or "").strip()
    if prompt_path:
        prompts = load_sam2_bbox_prompt_file(
            prompt_path,
            camera_names=camera_names,
            placeholders=placeholders,
        )
    return tracker_factory, extract_placeholder_point_clouds_sam2_online, prompts


def add_sam2_object_pointclouds(
    *,
    observation: dict[str, Any],
    dense_scene_pointcloud: np.ndarray,
    args: argparse.Namespace,
    placeholders: Sequence[str],
    camera_names: Sequence[str],
    tracker_factory,
    extract_all_fn,
    tracking_state_by_camera: dict[str, Any],
    bbox_prompts_by_camera: dict[str, dict[str, object]],
) -> dict[str, Any]:
    sam2_observation = dict(observation)
    sam2_observation["pointcloud"] = dense_scene_pointcloud.astype(np.float32)
    object_pointcloud, meta = extract_all_fn(
        sam2_observation,
        placeholders=placeholders,
        camera_names=camera_names,
        tracker_factory=tracker_factory,
        tracking_state_by_camera=tracking_state_by_camera,
        bbox_prompts_by_camera=bbox_prompts_by_camera,
        target_num_points=int(args.object_point_num),
        min_mask_points=int(args.sam2_min_mask_points),
        interactive_init=bool(args.sam2_interactive_init),
        sam2_image_width=int(args.sam2_image_width),
        min_depth_m=float(args.min_depth_m),
        max_depth_m=float(args.max_depth_m),
        object_resample_mode=str(args.online_object_resample),
        parallel_camera_workers=int(getattr(args, "parallel_camera_workers", 0)),
    )
    observation["object_pointcloud"] = object_pointcloud
    observation["sam2_meta"] = meta
    return observation


@contextmanager
def pushd(path: Path):
    old_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


@contextmanager
def profile_section(timings: dict[str, float], name: str, enabled: bool):
    if not enabled:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = timings.get(name, 0.0) + (time.perf_counter() - start)


def format_timings(timings: Mapping[str, float]) -> str:
    return " ".join(f"{name}={seconds * 1000.0:.1f}ms" for name, seconds in timings.items())


def should_print_step_timing(args: argparse.Namespace, step: int) -> bool:
    if not bool(args.profile_timing):
        return False
    interval = max(1, int(args.profile_timing_interval))
    return int(step) == 1 or int(step) % interval == 0


def encoded_obs_with_agent_vector(encoded_obs: Mapping[str, Any], agent_vector: np.ndarray) -> dict[str, Any]:
    out = dict(encoded_obs)
    out["agent_pos"] = np.asarray(agent_vector, dtype=np.float32).reshape(14)
    return out


def action_dim_for_mode(args: argparse.Namespace) -> int:
    return 20 if str(getattr(args, "action_mode", "joint")) == "eef_absolute6d" else 14


def load_eef_world_from_base_transforms(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    if str(getattr(args, "action_mode", "joint")) != "eef_absolute6d":
        return np.eye(4, dtype=np.float64), np.eye(4, dtype=np.float64)
    return load_world_from_base_transforms(
        calibration_path=getattr(args, "eef_calibration_path", ""),
        frame_mode=getattr(args, "eef_frame_mode", "right_base"),
        left_robot_camera_calibration_path=getattr(args, "left_robot_camera_calibration_path", ""),
        right_robot_camera_calibration_path=getattr(args, "right_robot_camera_calibration_path", ""),
    )


def agent_vector_from_joint_action(
    joint_action: np.ndarray,
    args: argparse.Namespace,
    t_world_from_left_base: np.ndarray,
    t_world_from_right_base: np.ndarray,
) -> np.ndarray:
    if str(getattr(args, "action_mode", "joint")) != "eef_absolute6d":
        return np.asarray(joint_action, dtype=np.float32).reshape(14)
    return joint14_to_eef14(
        joint_action,
        t_world_from_left_base=t_world_from_left_base,
        t_world_from_right_base=t_world_from_right_base,
        tool_x_m=float(getattr(args, "eef_tool_x_m", 0.0)),
        tool_y_m=float(getattr(args, "eef_tool_y_m", 0.0)),
        tool_z_m=float(getattr(args, "eef_tool_z_m", 0.197)),
    )


def eef_policy_action_to_joint_action(
    action20: np.ndarray,
    *,
    env,
    t_world_from_left_base: np.ndarray,
    t_world_from_right_base: np.ndarray,
) -> np.ndarray:
    if env is None:
        raise RuntimeError("EEF action mode requires a robot environment for Dobot IK conversion.")
    eef_world = action20_to_eef14(action20)
    eef_base = eef14_world_to_base(
        eef_world,
        t_world_from_left_base=t_world_from_left_base,
        t_world_from_right_base=t_world_from_right_base,
    )
    try:
        ik_joints = np.asarray(env.get_ik(eef_base), dtype=np.float32).reshape(-1)
    except Exception as exc:
        raise RuntimeError(
            "Dobot IK failed while converting EEF policy action. "
            f"eef_world={np.round(eef_world, 4).tolist()} "
            f"eef_base={np.round(eef_base, 4).tolist()} "
            f"cause={type(exc).__name__}: {exc}"
        ) from exc
    if ik_joints.shape[0] == 14:
        out = ik_joints.copy()
        out[6] = eef_base[6]
        out[13] = eef_base[13]
        return clamp_action(out)
    if ik_joints.shape[0] != 12:
        raise RuntimeError(f"Expected Dobot IK to return 12 or 14 values for bimanual arms, got {ik_joints.shape}")
    out = np.zeros(14, dtype=np.float32)
    out[:6] = ik_joints[:6]
    out[6] = eef_base[6]
    out[7:13] = ik_joints[6:12]
    out[13] = eef_base[13]
    return clamp_action(out)


def policy_actions_to_joint_actions(
    raw_actions: np.ndarray,
    *,
    env,
    args: argparse.Namespace,
    t_world_from_left_base: np.ndarray,
    t_world_from_right_base: np.ndarray,
) -> np.ndarray:
    actions = np.asarray(raw_actions, dtype=np.float32).reshape(-1, action_dim_for_mode(args))
    if str(getattr(args, "action_mode", "joint")) != "eef_absolute6d":
        return np.asarray([clamp_action(action) for action in actions], dtype=np.float32)
    return np.asarray(
        [
            eef_policy_action_to_joint_action(
                action,
                env=env,
                t_world_from_left_base=t_world_from_left_base,
                t_world_from_right_base=t_world_from_right_base,
            )
            for action in actions
        ],
        dtype=np.float32,
    )


def derive_dp3_settings(args: argparse.Namespace) -> tuple[str, str]:
    if args.config_name:
        config_name = str(args.config_name)
    elif args.mode == "semantic_pointwise_hybrid":
        config_name = "robot_dp3_semantic_pointwise_hybrid"
    else:
        config_name = "robot_dp3"

    if args.ckpt_setting:
        ckpt_setting = str(args.ckpt_setting)
    elif args.mode == "semantic_pointwise_hybrid":
        ckpt_setting = f"{args.task_config}-objpc-semantic-pointwise-hybrid"
    else:
        ckpt_setting = str(args.task_config)
    return config_name, ckpt_setting


def resolve_dp3_checkpoint_path(args: argparse.Namespace, ckpt_setting: str) -> Path:
    suffix = "_w_rgb" if bool(args.use_rgb) else ""
    return (
        DP3_ROOT
        / "checkpoints"
        / f"{args.task_name}-{ckpt_setting}-{args.expert_data_num}{suffix}_{args.seed}"
        / f"{args.checkpoint_num}.ckpt"
    )


def load_checkpoint_payload_for_validation(checkpoint_path: Path) -> Any | None:
    if not checkpoint_path.is_file():
        return None
    try:
        import dill  # noqa: PLC0415
        import torch  # noqa: PLC0415

        return torch.load(checkpoint_path.open("rb"), pickle_module=dill, map_location="cpu")
    except Exception as exc:
        print_warning(f"Could not pre-validate DP3 checkpoint structure at {checkpoint_path}: {exc}")
        return None


def semantic_pointcloud_placeholders_from_checkpoint_payload(checkpoint_payload: Any) -> set[str]:
    if not isinstance(checkpoint_payload, Mapping):
        return set()
    state_dicts = checkpoint_payload.get("state_dicts", {})
    candidates: list[Mapping[str, Any]] = []
    if isinstance(state_dicts, Mapping):
        candidates.extend(value for value in state_dicts.values() if isinstance(value, Mapping))
    candidates.append(checkpoint_payload)

    placeholders: set[str] = set()
    pattern = re.compile(r"(?:^|\.)semantic_point_cloud_([^.\s]+)\.")
    for state_dict in candidates:
        for key in state_dict.keys():
            if not isinstance(key, str):
                continue
            for match in pattern.finditer(key):
                placeholders.add("{" + match.group(1) + "}")
    return placeholders


def _optional_arg_is_set(value: Any) -> bool:
    return value not in {None, "", "none"}


def semantic_pointcloud_placeholders_from_args(args: argparse.Namespace) -> set[str]:
    placeholders: set[str] = set()
    if _optional_arg_is_set(getattr(args, "semantic_ckpt_A", None)):
        placeholders.add("{A}")
    if _optional_arg_is_set(getattr(args, "semantic_ckpt_B", None)):
        placeholders.add("{B}")
    return placeholders


def validate_semantic_checkpoint_branches(
    args: argparse.Namespace,
    checkpoint_payload: Any,
    checkpoint_path: str | Path,
) -> None:
    if args.mode != "semantic_pointwise_hybrid":
        return
    required_placeholders = semantic_pointcloud_placeholders_from_checkpoint_payload(checkpoint_payload)
    provided_placeholders = semantic_pointcloud_placeholders_from_args(args)
    extra_placeholders = provided_placeholders - required_placeholders
    if extra_placeholders:
        extra_keys = ", ".join(sorted(extra_placeholders))
        required_keys = ", ".join(sorted(required_placeholders)) if required_placeholders else "none"
        raise RuntimeError(
            "Semantic checkpoint/config mismatch: "
            f"checkpoint '{checkpoint_path}' was not trained with semantic point-cloud branch(es) {extra_keys}; "
            f"checkpoint semantic branch(es): {required_keys}. "
            "Use the checkpoint trained with matching semantic branches, or do not pass those semantic checkpoints."
        )
    if not required_placeholders:
        return

    placeholder_arg_names = {
        "{A}": "semantic_ckpt_A",
        "{B}": "semantic_ckpt_B",
    }
    missing_args = []
    for placeholder in sorted(required_placeholders):
        arg_name = placeholder_arg_names.get(placeholder)
        if arg_name is None:
            continue
        if not _optional_arg_is_set(getattr(args, arg_name, None)):
            missing_args.append(arg_name)
    if missing_args:
        missing_flags = ", ".join(f"--{name}" for name in missing_args)
        required_keys = ", ".join(sorted(required_placeholders))
        raise RuntimeError(
            "Semantic checkpoint/config mismatch: "
            f"checkpoint '{checkpoint_path}' contains semantic point-cloud branch(es) {required_keys}, "
            f"but inference did not provide {missing_flags}. "
            "Pass the matching semantic checkpoint(s), or use a DP3 checkpoint trained without those branch(es)."
        )


def load_dp3_model(args: argparse.Namespace):
    if args.gpu_id is not None and str(args.gpu_id).strip() != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    from deploy_policy import encode_obs, get_model  # noqa: PLC0415

    config_name, ckpt_setting = derive_dp3_settings(args)
    ckpt_file = resolve_dp3_checkpoint_path(args, ckpt_setting)
    payload = load_checkpoint_payload_for_validation(ckpt_file)
    if payload is not None:
        validate_semantic_checkpoint_branches(args, payload, ckpt_file)
    usr_args: dict[str, Any] = {
        "policy_name": "DP3",
        "task_name": args.task_name,
        "task_config": args.task_config,
        "ckpt_setting": ckpt_setting,
        "expert_data_num": str(args.expert_data_num),
        "seed": str(args.seed),
        "config_name": config_name,
        "checkpoint_num": str(args.checkpoint_num),
        "use_rgb": bool(args.use_rgb),
        "object_placeholders": args.object_placeholders,
        "semantic_ckpt_A": args.semantic_ckpt_A,
        "semantic_ckpt_B": args.semantic_ckpt_B,
        "semantic_device": args.semantic_device,
        "semantic_point_num": str(args.semantic_point_num),
    }
    with pushd(DP3_ROOT):
        model = get_model(usr_args)
    if hasattr(model, "env_runner"):
        model.env_runner.reset_obs()
    print(f"[INFO] Loaded DP3 model config={config_name}, ckpt_setting={ckpt_setting}")
    return model, encode_obs


def add_xtrainer_to_path() -> None:
    candidates = [
        REPO_ROOT / "include" / "xtrainer_clover",
        Path("/home/zheng/github/xtrainer_clover"),
    ]
    for candidate in candidates:
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


def build_robot_env(args: argparse.Namespace):
    if bool(args.no_robot):
        print("[INFO] Robot disabled by --no_robot; using zero/fallback joint state.")
        return None
    add_xtrainer_to_path()
    from dobot_control.env import RobotEnv  # noqa: PLC0415
    from dobot_control.robots.robot_node import ZMQClientRobot  # noqa: PLC0415

    print(f"[INFO] Connecting robot client at {args.hostname}:{int(args.robot_port)} ...")
    robot_client = ZMQClientRobot(port=int(args.robot_port), host=str(args.hostname))
    print("[INFO] Robot ZMQ client created; initializing RobotEnv ...")
    env = RobotEnv(robot_client)
    try:
        print("[INFO] Setting robot light DO status ...")
        env.set_do_status([1, 0])
        env.set_do_status([2, 0])
        env.set_do_status([3, 0])
    except Exception as exc:
        print(f"[WARN] Failed to set robot light DO status: {exc}")
    print("[INFO] Robot client initialized")
    return env


def configure_robot_servo_params(env, args: argparse.Namespace):
    if env is None:
        return None
    servo_j_t = float(getattr(args, "servo_j_t", 0.0))
    servo_j_gain = int(getattr(args, "servo_j_gain", 0))
    if servo_j_t <= 0.0 and servo_j_gain <= 0:
        return None
    kwargs = {
        "servo_j_t": servo_j_t if servo_j_t > 0.0 else None,
        "servo_j_gain": servo_j_gain if servo_j_gain > 0 else None,
    }
    try:
        result = env.set_servo_params(**kwargs)
    except Exception as exc:
        raise RuntimeError(
            "Failed to set Dobot ServoJ parameters through the robot server. "
            "Restart the xtrainer ZMQ robot server after updating RoboTwin_geo, "
            "or pass --servo_j_t 0 --servo_j_gain 0 to skip this controller-level setting."
        ) from exc
    print(f"[INFO] Dobot ServoJ params set: {result}")
    return result


def interpolate_robot(env, start: np.ndarray, target: np.ndarray, *, max_step: float, flag: np.ndarray) -> None:
    max_delta = float(np.max(np.abs(target - start))) if start.size and target.size else 0.0
    steps = max(1, min(int(max_delta / max(float(max_step), 1e-6)), 150))
    for joints in np.linspace(start, target, steps):
        env.step(np.asarray(joints, dtype=np.float32), flag)


def reset_robot_to_photo_pose(env) -> np.ndarray:
    safe_left = np.deg2rad([-90, 30, -110, 20, 90, 90, 0])
    safe_right = np.deg2rad([90, -30, 110, -20, -90, -90, 0])
    photo_left = np.deg2rad([-90, 0, -90, 0, 90, 90, 57])
    photo_right = np.deg2rad([90, 0, 90, 0, -90, -90, 57])
    flag = np.asarray([1, 1], dtype=np.float32)
    print("[INFO] Reading robot joints before reset ...")
    current = np.asarray(env.get_obs()["joint_positions"], dtype=np.float32).reshape(14)
    safe = np.concatenate([safe_left, safe_right]).astype(np.float32)
    photo = np.concatenate([photo_left, photo_right]).astype(np.float32)
    print("[INFO] Moving robot to safe pose ...")
    interpolate_robot(env, current, safe, max_step=0.001, flag=flag)
    time.sleep(0.5)
    print("[INFO] Moving robot to initial photo pose ...")
    interpolate_robot(env, safe, photo, max_step=0.001, flag=flag)
    time.sleep(0.5)
    print("[INFO] Robot reset complete.")
    return photo.astype(np.float32)


def current_joint_vector(env, fallback: np.ndarray, *, use_last_gripper: bool = True) -> np.ndarray:
    if env is None:
        return np.asarray(fallback, dtype=np.float32).reshape(14).copy()
    print("[TRACE] Reading robot joint observation ...", flush=True)
    obs = env.get_obs()
    joints = np.asarray(obs["joint_positions"], dtype=np.float32).reshape(14)
    if use_last_gripper:
        joints[6] = float(fallback[6])
        joints[13] = float(fallback[13])
    return joints


def reset_model_observation_history(model) -> None:
    if hasattr(model, "env_runner") and hasattr(model.env_runner, "reset_obs"):
        model.env_runner.reset_obs()


def initialize_episode_action(
    *,
    env,
    model,
    args: argparse.Namespace,
    allow_robot_reset: bool,
    fallback_action: np.ndarray | None = None,
) -> np.ndarray:
    reset_model_observation_history(model)
    if env is not None and bool(args.execute) and bool(allow_robot_reset):
        action = reset_robot_to_photo_pose(env)
    elif env is not None:
        print("[INFO] Reading initial robot joint state ...")
        action = np.asarray(env.get_obs()["joint_positions"], dtype=np.float32).reshape(14).copy()
    elif fallback_action is not None:
        action = np.asarray(fallback_action, dtype=np.float32).reshape(14).copy()
    else:
        action = np.zeros(14, dtype=np.float32)
    action[6] = float(args.initial_left_gripper)
    action[13] = float(args.initial_right_gripper)
    return clamp_action(action)


def restore_sam2_prompt_records(target: dict[str, dict[str, object]], source: Mapping[str, dict[str, object]]) -> None:
    target.clear()
    target.update(copy.deepcopy(dict(source)))


def handle_requested_episode_reset(
    *,
    env,
    model,
    args: argparse.Namespace,
    keyboard_listener: KeyboardCommandListener,
    sam2_tracking_state_by_camera: dict[str, Any],
    sam2_bbox_prompts_by_camera: dict[str, dict[str, object]],
    sam2_loaded_prompts_by_camera: Mapping[str, dict[str, object]],
    fallback_action: np.ndarray,
) -> np.ndarray:
    for command in keyboard_listener.pop_commands():
        if command == "reset":
            print_info("[INFO] Keyboard reset requested.")
        elif command == "quit":
            print_info("[INFO] Keyboard quit requested.")
    keyboard_listener.clear_reset_request()
    if bool(getattr(args, "reset_sam2_on_keyboard_reset", False)):
        sam2_tracking_state_by_camera.clear()
        restore_sam2_prompt_records(sam2_bbox_prompts_by_camera, sam2_loaded_prompts_by_camera)
        print_info("[INFO] Cleared SAM2 tracking state for keyboard reset.")
    last_action = initialize_episode_action(
        env=env,
        model=model,
        args=args,
        allow_robot_reset=True,
        fallback_action=fallback_action,
    )
    print_info(f"[INFO] Episode reset complete. Restarting up to {int(args.max_steps)} steps.")
    return last_action


def clamp_action(action: np.ndarray) -> np.ndarray:
    out = np.asarray(action, dtype=np.float32).reshape(14).copy()
    out[6] = float(np.clip(out[6], 0.0, 1.0))
    out[13] = float(np.clip(out[13], 0.0, 1.0))
    return out


def limit_action_delta_for_execution(
    action: np.ndarray,
    last_action: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    out = np.asarray(action, dtype=np.float32).reshape(14).copy()
    last = np.asarray(last_action, dtype=np.float32).reshape(14)
    joint_limit = float(getattr(args, "max_executed_joint_delta", 0.0))
    gripper_limit = float(getattr(args, "max_executed_gripper_delta", 0.0))

    if joint_limit > 0.0:
        delta = np.clip(out[ARM_JOINT_INDICES] - last[ARM_JOINT_INDICES], -joint_limit, joint_limit)
        out[ARM_JOINT_INDICES] = last[ARM_JOINT_INDICES] + delta

    if gripper_limit > 0.0:
        for idx in GRIPPER_INDICES:
            delta = float(np.clip(float(out[idx] - last[idx]), -gripper_limit, gripper_limit))
            out[idx] = float(last[idx] + delta)

    return clamp_action(out)


def limit_action_delta_change_for_execution(
    action: np.ndarray,
    last_action: np.ndarray,
    previous_command_delta: np.ndarray | None,
    args: argparse.Namespace,
) -> np.ndarray:
    out = np.asarray(action, dtype=np.float32).reshape(14).copy()
    last = np.asarray(last_action, dtype=np.float32).reshape(14)
    if previous_command_delta is None:
        prev_delta = np.zeros(14, dtype=np.float32)
    else:
        prev_delta = np.asarray(previous_command_delta, dtype=np.float32).reshape(14)
    joint_change_limit = float(getattr(args, "max_executed_joint_delta_change", 0.0))
    gripper_change_limit = float(getattr(args, "max_executed_gripper_delta_change", 0.0))

    if joint_change_limit > 0.0:
        current_delta = out[ARM_JOINT_INDICES] - last[ARM_JOINT_INDICES]
        delta_change = np.clip(
            current_delta - prev_delta[ARM_JOINT_INDICES],
            -joint_change_limit,
            joint_change_limit,
        )
        out[ARM_JOINT_INDICES] = last[ARM_JOINT_INDICES] + prev_delta[ARM_JOINT_INDICES] + delta_change

    if gripper_change_limit > 0.0:
        for idx in GRIPPER_INDICES:
            current_delta = float(out[idx] - last[idx])
            delta_change = float(
                np.clip(
                    current_delta - float(prev_delta[idx]),
                    -gripper_change_limit,
                    gripper_change_limit,
                )
            )
            out[idx] = float(last[idx] + float(prev_delta[idx]) + delta_change)

    return clamp_action(out)


def prepare_action_for_execution(
    action: np.ndarray,
    last_action: np.ndarray,
    args: argparse.Namespace,
    *,
    previous_command_delta: np.ndarray | None = None,
) -> np.ndarray:
    command = clamp_action(action)
    command = limit_action_delta_for_execution(command, last_action, args)
    command = limit_action_delta_change_for_execution(command, last_action, previous_command_delta, args)
    return clamp_action(command)


def build_execution_substeps(
    last_action: np.ndarray,
    target_action: np.ndarray,
    args: argparse.Namespace,
) -> list[np.ndarray]:
    start = np.asarray(last_action, dtype=np.float32).reshape(14)
    target = np.asarray(target_action, dtype=np.float32).reshape(14)
    substeps = int(getattr(args, "execution_substeps", 1))
    if substeps <= 1:
        return [clamp_action(target)]
    return [
        clamp_action(start + (target - start) * (float(i) / float(substeps)))
        for i in range(1, substeps + 1)
    ]


def action_delta_summary(action: np.ndarray, last_action: np.ndarray) -> tuple[float, float]:
    current = np.asarray(action, dtype=np.float32).reshape(14)
    last = np.asarray(last_action, dtype=np.float32).reshape(14)
    arm_delta = float(np.max(np.abs(current[ARM_JOINT_INDICES] - last[ARM_JOINT_INDICES])))
    gripper_delta = float(np.max(np.abs(current[GRIPPER_INDICES] - last[GRIPPER_INDICES])))
    return arm_delta, gripper_delta


def action_delta_change_summary(
    action: np.ndarray,
    last_action: np.ndarray,
    previous_command_delta: np.ndarray | None,
) -> tuple[float, float]:
    current = np.asarray(action, dtype=np.float32).reshape(14)
    last = np.asarray(last_action, dtype=np.float32).reshape(14)
    if previous_command_delta is None:
        prev_delta = np.zeros(14, dtype=np.float32)
    else:
        prev_delta = np.asarray(previous_command_delta, dtype=np.float32).reshape(14)
    current_delta = current - last
    arm_delta_change = float(np.max(np.abs(current_delta[ARM_JOINT_INDICES] - prev_delta[ARM_JOINT_INDICES])))
    gripper_delta_change = float(np.max(np.abs(current_delta[GRIPPER_INDICES] - prev_delta[GRIPPER_INDICES])))
    return arm_delta_change, gripper_delta_change


def build_action_diagnostic_row(
    *,
    step: int,
    raw_policy_action: np.ndarray,
    command_action: np.ndarray,
    observed_after_step: np.ndarray,
    action_before_step: np.ndarray,
    previous_command_delta: np.ndarray | None = None,
    target_period_sec: float,
    step_elapsed_sec: float,
) -> dict[str, float | int | bool]:
    policy_arm_delta, policy_gripper_delta = action_delta_summary(raw_policy_action, action_before_step)
    command_arm_delta, command_gripper_delta = action_delta_summary(command_action, action_before_step)
    observed_arm_delta, observed_gripper_delta = action_delta_summary(observed_after_step, action_before_step)
    command_arm_delta_change, command_gripper_delta_change = action_delta_change_summary(
        command_action,
        action_before_step,
        previous_command_delta,
    )
    follow_error_arm, follow_error_gripper = action_delta_summary(observed_after_step, command_action)
    target_period_sec = float(target_period_sec)
    step_elapsed_sec = float(step_elapsed_sec)
    return {
        "step": int(step),
        "policy_arm_delta": policy_arm_delta,
        "command_arm_delta": command_arm_delta,
        "observed_arm_delta": observed_arm_delta,
        "policy_gripper_delta": policy_gripper_delta,
        "command_gripper_delta": command_gripper_delta,
        "observed_gripper_delta": observed_gripper_delta,
        "command_arm_delta_change": command_arm_delta_change,
        "command_gripper_delta_change": command_gripper_delta_change,
        "command_follow_error_arm": follow_error_arm,
        "command_follow_error_gripper": follow_error_gripper,
        "target_period_sec": target_period_sec,
        "step_elapsed_sec": step_elapsed_sec,
        "control_overrun": bool(target_period_sec > 0.0 and step_elapsed_sec > target_period_sec),
    }


def maybe_check_action_safety(action: np.ndarray, last_action: np.ndarray, args: argparse.Namespace, *, first_action: bool) -> None:
    if bool(args.disable_action_delta_safety):
        return
    if first_action and bool(args.interpolate_first_action):
        return
    max_delta = float(np.max(np.abs(action - last_action)))
    if max_delta > float(args.max_action_delta):
        raise RuntimeError(
            f"Action delta safety stop: max_delta={max_delta:.4f} > {float(args.max_action_delta):.4f}. "
            "Pass --disable_action_delta_safety to override."
        )


def maybe_check_xyz_safety(env, args: argparse.Namespace) -> None:
    if env is None or not bool(args.execute) or bool(args.disable_xyz_safety):
        return
    pos = np.asarray(env.get_XYZrxryrz_state(), dtype=np.float32).reshape(-1)
    if pos.shape[0] < 9:
        print(f"[WARN] Cannot run XYZ safety check, expected >=9 values but got {pos.shape}")
        return
    left_x_min = float(getattr(args, "xyz_left_x_min", -450.0))
    left_x_max = float(getattr(args, "xyz_left_x_max", 450.0))
    right_x_min = float(getattr(args, "xyz_right_x_min", -450.0))
    right_x_max = float(getattr(args, "xyz_right_x_max", 450.0))
    y_min = float(getattr(args, "xyz_y_min", -750.0))
    y_max = float(getattr(args, "xyz_y_max", -160.0))
    left_z_min = float(getattr(args, "xyz_left_z_min", 25.0))
    right_z_min = float(getattr(args, "xyz_right_z_min", 23.0))
    ok = (
        (pos[0] >= left_x_min)
        and (pos[0] <= left_x_max)
        and (pos[1] >= y_min)
        and (pos[1] <= y_max)
        and (pos[2] > left_z_min)
        and (pos[6] >= right_x_min)
        and (pos[6] <= right_x_max)
        and (pos[7] >= y_min)
        and (pos[7] <= y_max)
        and (pos[8] > right_z_min)
    )
    if not ok:
        raise RuntimeError(
            "XYZ safety stop. "
            f"Current robot XYZrxryrz state: {pos.tolist()} "
            f"limits={{left_x:[{left_x_min},{left_x_max}], right_x:[{right_x_min},{right_x_max}], "
            f"y:[{y_min},{y_max}], left_z>{left_z_min}, right_z>{right_z_min}}}"
        )


def execute_action(
    *,
    env,
    action: np.ndarray,
    last_action: np.ndarray,
    args: argparse.Namespace,
    first_action: bool,
    previous_command_delta: np.ndarray | None = None,
) -> np.ndarray:
    action = prepare_action_for_execution(
        action,
        last_action,
        args,
        previous_command_delta=previous_command_delta,
    )
    maybe_check_action_safety(action, last_action, args, first_action=first_action)
    maybe_check_xyz_safety(env, args)
    flag = np.asarray([1, 1], dtype=np.float32)
    if env is not None and bool(args.execute):
        if first_action and bool(args.interpolate_first_action):
            interpolate_robot(env, last_action, action, max_step=float(args.first_action_interp_step), flag=flag)
            substeps = [action]
        else:
            substeps = build_execution_substeps(last_action, action, args)
        obs = None
        sleep_sec = max(0.0, float(getattr(args, "execution_substep_sleep_sec", 0.0)))
        for idx, substep in enumerate(substeps):
            obs = env.step(substep.astype(np.float32), flag)
            if sleep_sec > 0.0 and idx < len(substeps) - 1:
                time.sleep(sleep_sec)
        if obs is None:
            obs = {"joint_positions": action.astype(np.float32)}
        joints = np.asarray(obs["joint_positions"], dtype=np.float32).reshape(14)
        joints[6] = action[6]
        joints[13] = action[13]
        return joints
    return action.copy()


class AsyncActionController:
    def __init__(
        self,
        *,
        env,
        args: argparse.Namespace,
        initial_action: np.ndarray,
        external_stop_event: threading.Event | None = None,
    ):
        self.env = env
        self.args = args
        self.external_stop_event = external_stop_event
        self._latest_action = clamp_action(initial_action)
        self._previous_command_delta = np.zeros(14, dtype=np.float32)
        self._last_target_action = self._latest_action.copy()
        self._has_target_action = False
        self._pending_actions: deque[np.ndarray] = deque()
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._step_count = 0
        self._first_action = False
        self._error: BaseException | None = None
        self._diagnostic_file = None
        self._diagnostic_writer = None

    @property
    def step_count(self) -> int:
        with self._condition:
            return int(self._step_count)

    def latest_action(self) -> np.ndarray:
        with self._condition:
            return self._latest_action.copy()

    def start(self) -> None:
        if self._thread is not None:
            return
        diagnostics_path = str(getattr(self.args, "action_diagnostics_csv", "")).strip()
        if diagnostics_path:
            path = resolve_path(diagnostics_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._diagnostic_file = path.open("w", newline="", encoding="utf-8")
            print(f"[INFO] Writing async action diagnostics CSV: {path}")
        self._thread = threading.Thread(target=self._run, name="AsyncActionController", daemon=True)
        self._thread.start()

    def submit_actions(self, actions: Sequence[np.ndarray]) -> None:
        action_list = [clamp_action(np.asarray(action, dtype=np.float32).reshape(14)) for action in actions]
        if not action_list:
            return
        with self._condition:
            if bool(getattr(self.args, "async_control_replace_buffer", True)):
                self._pending_actions.clear()
            self._pending_actions.extend(action_list)
            self._has_target_action = True
            self._condition.notify_all()

    def wait_until_steps(self, step_count: int, timeout_sec: float) -> bool:
        deadline = time.time() + float(timeout_sec)
        with self._condition:
            while self._step_count < int(step_count):
                self._raise_if_error_locked()
                remaining = deadline - time.time()
                if remaining <= 0.0:
                    return False
                self._condition.wait(timeout=remaining)
            return True

    def raise_if_error(self) -> None:
        with self._condition:
            self._raise_if_error_locked()

    def _raise_if_error_locked(self) -> None:
        if self._error is not None:
            raise RuntimeError("Async control thread failed") from self._error

    def stop(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self._diagnostic_file is not None:
            self._diagnostic_file.close()
            self._diagnostic_file = None
        self.raise_if_error()

    def _next_action(self) -> tuple[np.ndarray | None, bool]:
        max_idle_repeats = int(getattr(self.args, "async_control_max_idle_repeats", -1))
        with self._condition:
            if self._pending_actions:
                action = self._pending_actions.popleft()
                self._last_target_action = action.copy()
                return action, False
            if not self._has_target_action:
                return None, True
            if max_idle_repeats == 0:
                return None, True
            return self._last_target_action.copy(), True

    def _write_diagnostic(self, row: Mapping[str, Any]) -> None:
        if self._diagnostic_file is None:
            return
        if self._diagnostic_writer is None:
            self._diagnostic_writer = csv.DictWriter(self._diagnostic_file, fieldnames=list(row.keys()))
            self._diagnostic_writer.writeheader()
        self._diagnostic_writer.writerow(row)
        self._diagnostic_file.flush()

    def _run(self) -> None:
        hz = float(getattr(self.args, "async_control_hz", 0.0))
        if hz <= 0.0:
            hz = float(getattr(self.args, "control_hz", 0.0))
        period = 0.0 if hz <= 0.0 else 1.0 / hz
        idle_repeats = 0
        try:
            while (
                not self._stop_event.is_set()
                and not (self.external_stop_event is not None and self.external_stop_event.is_set())
                and self.step_count < int(getattr(self.args, "max_steps", 200))
            ):
                loop_start = time.perf_counter()
                raw_action, idle = self._next_action()
                if raw_action is None:
                    time.sleep(min(period, 0.01) if period > 0.0 else 0.001)
                    continue
                if idle:
                    idle_repeats += 1
                    max_idle_repeats = int(getattr(self.args, "async_control_max_idle_repeats", -1))
                    if max_idle_repeats > 0 and idle_repeats > max_idle_repeats:
                        time.sleep(min(period, 0.01) if period > 0.0 else 0.001)
                        continue
                else:
                    idle_repeats = 0

                with self._condition:
                    action_before_step = self._latest_action.copy()
                    previous_command_delta = self._previous_command_delta.copy()

                command_action = prepare_action_for_execution(
                    raw_action,
                    action_before_step,
                    self.args,
                    previous_command_delta=previous_command_delta,
                )
                cmd_arm_delta, cmd_gripper_delta = action_delta_summary(command_action, action_before_step)
                cmd_arm_delta_change, cmd_gripper_delta_change = action_delta_change_summary(
                    command_action,
                    action_before_step,
                    previous_command_delta,
                )
                observed_after_step = execute_action(
                    env=self.env,
                    action=raw_action,
                    last_action=action_before_step,
                    args=self.args,
                    first_action=False,
                    previous_command_delta=previous_command_delta,
                )
                step_elapsed = time.perf_counter() - loop_start
                with self._condition:
                    self._latest_action = observed_after_step.copy()
                    self._previous_command_delta = command_action - action_before_step
                    self._step_count += 1
                    step = int(self._step_count)
                    self._condition.notify_all()
                pred_arm_delta, pred_gripper_delta = action_delta_summary(raw_action, action_before_step)
                exec_arm_delta, exec_gripper_delta = action_delta_summary(observed_after_step, action_before_step)
                row = build_action_diagnostic_row(
                    step=step,
                    raw_policy_action=raw_action,
                    command_action=command_action,
                    observed_after_step=observed_after_step,
                    action_before_step=action_before_step,
                    previous_command_delta=previous_command_delta,
                    target_period_sec=period,
                    step_elapsed_sec=step_elapsed,
                )
                self._write_diagnostic(row)
                print(
                    f"[ASYNC STEP {step:04d}] "
                    f"idle={idle} left_gripper={observed_after_step[6]:.3f} right_gripper={observed_after_step[13]:.3f} "
                    f"pred_arm_delta={pred_arm_delta:.4f} cmd_arm_delta={cmd_arm_delta:.4f} "
                    f"cmd_arm_delta_change={cmd_arm_delta_change:.4f} exec_arm_delta={exec_arm_delta:.4f} "
                    f"pred_gripper_delta={pred_gripper_delta:.3f} cmd_gripper_delta={cmd_gripper_delta:.3f} "
                    f"cmd_gripper_delta_change={cmd_gripper_delta_change:.3f} exec_gripper_delta={exec_gripper_delta:.3f}",
                    flush=True,
                )
                elapsed = time.perf_counter() - loop_start
                if period > 0.0 and elapsed < period:
                    time.sleep(period - elapsed)
        except BaseException as exc:
            with self._condition:
                self._error = exc
                self._condition.notify_all()


def run_real_inference(args: argparse.Namespace) -> None:
    if bool(args.execute) and bool(args.no_robot):
        raise ValueError("--execute cannot be used together with --no_robot")

    print("[INFO] Starting real DP3 inference.")
    placeholders = parse_placeholder_list(args.object_placeholders)
    print("[INFO] Loading DP3 model ...")
    model, encode_obs = load_dp3_model(args)
    needs_sam2_objpc = args.mode == "semantic_pointwise_hybrid" or bool(args.enable_sam2_objpc)

    sam2_runtime = None
    sam2_tracking_state_by_camera: dict[str, Any] = {}
    sam2_bbox_prompts_by_camera: dict[str, dict[str, object]] = {}
    sam2_loaded_prompts_by_camera: dict[str, dict[str, object]] = {}

    print("[INFO] Starting ZED cameras ...")
    live = start_zed_cameras(args)
    t_world_from_left_base, t_world_from_right_base = load_eef_world_from_base_transforms(args)
    env = None
    keyboard_listener = KeyboardCommandListener(
        enabled=bool(args.keyboard_control),
        reset_key=str(args.keyboard_reset_key),
        quit_key=str(args.keyboard_quit_key),
    )
    action_diagnostic_file = None
    action_diagnostic_writer = None
    try:
        print("[INFO] Initializing robot interface ...")
        env = build_robot_env(args)
        configure_robot_servo_params(env, args)
        keyboard_listener.start()
        last_action = initialize_episode_action(
            env=env,
            model=model,
            args=args,
            allow_robot_reset=not bool(args.skip_robot_reset),
        )

        if needs_sam2_objpc:
            print("[INFO] Loading SAM2 runtime for object point clouds ...")
            sam2_runtime = load_sam2_runtime(args, placeholders, live.labels)
            _, _, loaded_prompts = sam2_runtime
            sam2_loaded_prompts_by_camera = copy.deepcopy(loaded_prompts)
            restore_sam2_prompt_records(sam2_bbox_prompts_by_camera, sam2_loaded_prompts_by_camera)

        if not bool(args.execute):
            print("[INFO] Dry-run mode: model actions are printed but not sent to the robot. Pass --execute to move.")

        if bool(args.async_control):
            if bool(args.reobserve_each_action):
                print_warning("[WARN] --reobserve_each_action is ignored when --async_control is enabled.")
            controller: AsyncActionController | None = None
            try:
                while (controller is None or controller.step_count < int(args.max_steps)) and not keyboard_listener.quit_requested():
                    if keyboard_listener.reset_requested():
                        if controller is not None:
                            controller.stop()
                            controller = None
                        last_action = handle_requested_episode_reset(
                            env=env,
                            model=model,
                            args=args,
                            keyboard_listener=keyboard_listener,
                            sam2_tracking_state_by_camera=sam2_tracking_state_by_camera,
                            sam2_bbox_prompts_by_camera=sam2_bbox_prompts_by_camera,
                            sam2_loaded_prompts_by_camera=sam2_loaded_prompts_by_camera,
                            fallback_action=last_action,
                        )
                        continue
                    if controller is not None:
                        controller.raise_if_error()
                        joints = controller.latest_action()
                    else:
                        joints = last_action.copy()
                    agent_vector = agent_vector_from_joint_action(
                        joints,
                        args,
                        t_world_from_left_base,
                        t_world_from_right_base,
                    )
                    chunk_timings: dict[str, float] = {}
                    print("[TRACE] Building live point-cloud observation ...", flush=True)
                    with profile_section(chunk_timings, "build_obs_pre", bool(args.profile_timing)):
                        observation, dense_scene = build_robotwin_observation(args=args, live=live, agent_vector=agent_vector)
                    if keyboard_listener.reset_requested() or keyboard_listener.quit_requested():
                        continue
                    if needs_sam2_objpc:
                        print("[TRACE] Updating SAM2 object point clouds ...", flush=True)
                        tracker_factory, extract_all_fn, _ = sam2_runtime
                        with profile_section(chunk_timings, "sam2_pre", bool(args.profile_timing)):
                            observation = add_sam2_object_pointclouds(
                                observation=observation,
                                dense_scene_pointcloud=dense_scene,
                                args=args,
                                placeholders=placeholders,
                                camera_names=live.labels,
                                tracker_factory=tracker_factory,
                                extract_all_fn=extract_all_fn,
                                tracking_state_by_camera=sam2_tracking_state_by_camera,
                                bbox_prompts_by_camera=sam2_bbox_prompts_by_camera,
                            )
                    if keyboard_listener.reset_requested() or keyboard_listener.quit_requested():
                        continue
                    if bool(args.show_img):
                        with profile_section(chunk_timings, "show_img_pre", bool(args.profile_timing)):
                            maybe_show_cameras(observation, live.labels)

                    with profile_section(chunk_timings, "encode_pre", bool(args.profile_timing)):
                        encoded_obs = encode_obs(observation, model)
                    model.update_obs(encoded_obs)
                    print("[TRACE] Running DP3 policy inference ...", flush=True)
                    with profile_section(chunk_timings, "dp3_policy", bool(args.profile_timing)):
                        raw_actions = np.asarray(model.get_action(), dtype=np.float32).reshape(-1, action_dim_for_mode(args))
                        actions = policy_actions_to_joint_actions(
                            raw_actions,
                            env=env,
                            args=args,
                            t_world_from_left_base=t_world_from_left_base,
                            t_world_from_right_base=t_world_from_right_base,
                        )
                    if keyboard_listener.reset_requested() or keyboard_listener.quit_requested():
                        continue
                    if controller is None:
                        controller = AsyncActionController(
                            env=env,
                            args=args,
                            initial_action=joints,
                            external_stop_event=keyboard_listener.reset_event,
                        )
                        controller.start()
                    controller.submit_actions(actions)
                    if bool(args.profile_timing):
                        chunk_timings["chunk_total"] = sum(chunk_timings.values())
                        print_info(
                            f"[TIMING async_chunk@step {controller.step_count + 1:04d}] "
                            f"{format_timings(chunk_timings)} actions_submitted={len(actions)} "
                            f"control_steps={controller.step_count}",
                        )
                if controller is not None:
                    controller.raise_if_error()
                if keyboard_listener.quit_requested():
                    keyboard_listener.pop_commands()
                    print_info("[INFO] Keyboard quit requested. Stopping real inference.")
            finally:
                if controller is not None:
                    controller.stop()
            return

        action_step = 0
        first_action = True
        previous_command_delta = np.zeros(14, dtype=np.float32)
        rate_sec = 0.0 if float(args.control_hz) <= 0 else 1.0 / float(args.control_hz)
        if str(args.action_diagnostics_csv).strip():
            diagnostics_path = resolve_path(args.action_diagnostics_csv)
            diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
            action_diagnostic_file = diagnostics_path.open("w", newline="", encoding="utf-8")
            print(f"[INFO] Writing action diagnostics CSV: {diagnostics_path}")
        while action_step < int(args.max_steps):
            if keyboard_listener.quit_requested():
                keyboard_listener.pop_commands()
                print_info("[INFO] Keyboard quit requested. Stopping real inference.")
                return
            if keyboard_listener.reset_requested():
                last_action = handle_requested_episode_reset(
                    env=env,
                    model=model,
                    args=args,
                    keyboard_listener=keyboard_listener,
                    sam2_tracking_state_by_camera=sam2_tracking_state_by_camera,
                    sam2_bbox_prompts_by_camera=sam2_bbox_prompts_by_camera,
                    sam2_loaded_prompts_by_camera=sam2_loaded_prompts_by_camera,
                    fallback_action=last_action,
                )
                action_step = 0
                first_action = True
                previous_command_delta = np.zeros(14, dtype=np.float32)
                continue
            chunk_timings: dict[str, float] = {}
            with profile_section(chunk_timings, "robot_obs_pre", bool(args.profile_timing)):
                joints = current_joint_vector(env, last_action)
            print("[TRACE] Building live point-cloud observation ...", flush=True)
            with profile_section(chunk_timings, "build_obs_pre", bool(args.profile_timing)):
                agent_vector = agent_vector_from_joint_action(
                    joints,
                    args,
                    t_world_from_left_base,
                    t_world_from_right_base,
                )
                observation, dense_scene = build_robotwin_observation(args=args, live=live, agent_vector=agent_vector)
            if keyboard_listener.reset_requested() or keyboard_listener.quit_requested():
                continue
            if needs_sam2_objpc:
                print("[TRACE] Updating SAM2 object point clouds ...", flush=True)
                tracker_factory, extract_all_fn, _ = sam2_runtime
                with profile_section(chunk_timings, "sam2_pre", bool(args.profile_timing)):
                    observation = add_sam2_object_pointclouds(
                        observation=observation,
                        dense_scene_pointcloud=dense_scene,
                        args=args,
                        placeholders=placeholders,
                        camera_names=live.labels,
                        tracker_factory=tracker_factory,
                        extract_all_fn=extract_all_fn,
                            tracking_state_by_camera=sam2_tracking_state_by_camera,
                            bbox_prompts_by_camera=sam2_bbox_prompts_by_camera,
                        )
            if keyboard_listener.reset_requested() or keyboard_listener.quit_requested():
                continue
            if bool(args.show_img):
                with profile_section(chunk_timings, "show_img_pre", bool(args.profile_timing)):
                    maybe_show_cameras(observation, live.labels)

            with profile_section(chunk_timings, "encode_pre", bool(args.profile_timing)):
                encoded_obs = encode_obs(observation, model)
            model.update_obs(encoded_obs)
            print("[TRACE] Running DP3 policy inference ...", flush=True)
            with profile_section(chunk_timings, "dp3_policy", bool(args.profile_timing)):
                raw_actions = np.asarray(model.get_action(), dtype=np.float32).reshape(-1, action_dim_for_mode(args))
                actions = policy_actions_to_joint_actions(
                    raw_actions,
                    env=env,
                    args=args,
                    t_world_from_left_base=t_world_from_left_base,
                    t_world_from_right_base=t_world_from_right_base,
                )
            if keyboard_listener.reset_requested() or keyboard_listener.quit_requested():
                continue
            if bool(args.profile_timing):
                chunk_timings["chunk_total"] = sum(chunk_timings.values())
                print_info(
                    f"[TIMING chunk@step {action_step + 1:04d}] "
                    f"{format_timings(chunk_timings)} actions={len(actions)} "
                    f"reobserve_each_action={bool(args.reobserve_each_action)}",
                )

            for action in actions:
                if action_step >= int(args.max_steps):
                    break
                if keyboard_listener.reset_requested() or keyboard_listener.quit_requested():
                    break
                action_loop_start = time.time()
                step_timings: dict[str, float] = {}
                step_wall_start = time.perf_counter()
                last_action_before_step = last_action.copy()
                pred_arm_delta, pred_gripper_delta = action_delta_summary(action, last_action_before_step)
                command_action = prepare_action_for_execution(
                    action,
                    last_action_before_step,
                    args,
                    previous_command_delta=previous_command_delta,
                )
                cmd_arm_delta, cmd_gripper_delta = action_delta_summary(command_action, last_action_before_step)
                cmd_arm_delta_change, cmd_gripper_delta_change = action_delta_change_summary(
                    command_action,
                    last_action_before_step,
                    previous_command_delta,
                )
                with profile_section(step_timings, "execute", bool(args.profile_timing)):
                    last_action = execute_action(
                        env=env,
                        action=action,
                        last_action=last_action_before_step,
                        args=args,
                        first_action=first_action,
                        previous_command_delta=previous_command_delta,
                    )
                exec_arm_delta, exec_gripper_delta = action_delta_summary(last_action, last_action_before_step)
                first_action = False
                action_step += 1
                step_elapsed_for_diag = time.perf_counter() - step_wall_start
                diagnostic_row = build_action_diagnostic_row(
                    step=action_step,
                    raw_policy_action=action,
                    command_action=command_action,
                    observed_after_step=last_action,
                    action_before_step=last_action_before_step,
                    previous_command_delta=previous_command_delta,
                    target_period_sec=rate_sec,
                    step_elapsed_sec=step_elapsed_for_diag,
                )
                previous_command_delta = command_action - last_action_before_step
                if action_diagnostic_file is not None:
                    if action_diagnostic_writer is None:
                        action_diagnostic_writer = csv.DictWriter(action_diagnostic_file, fieldnames=list(diagnostic_row.keys()))
                        action_diagnostic_writer.writeheader()
                    action_diagnostic_writer.writerow(diagnostic_row)
                    action_diagnostic_file.flush()
                print(
                    f"[STEP {action_step:04d}] "
                    f"left_gripper={last_action[6]:.3f} right_gripper={last_action[13]:.3f} "
                    f"pred_arm_delta={pred_arm_delta:.4f} cmd_arm_delta={cmd_arm_delta:.4f} "
                    f"cmd_arm_delta_change={cmd_arm_delta_change:.4f} exec_arm_delta={exec_arm_delta:.4f} "
                    f"pred_gripper_delta={pred_gripper_delta:.3f} cmd_gripper_delta={cmd_gripper_delta:.3f} "
                    f"cmd_gripper_delta_change={cmd_gripper_delta_change:.3f} exec_gripper_delta={exec_gripper_delta:.3f} "
                    f"execute={bool(args.execute)}"
                )

                if bool(args.reobserve_each_action):
                    with profile_section(step_timings, "robot_obs", bool(args.profile_timing)):
                        joints = current_joint_vector(env, last_action)
                    with profile_section(step_timings, "build_obs", bool(args.profile_timing)):
                        agent_vector = agent_vector_from_joint_action(
                            joints,
                            args,
                            t_world_from_left_base,
                            t_world_from_right_base,
                        )
                        observation, dense_scene = build_robotwin_observation(args=args, live=live, agent_vector=agent_vector)
                    if needs_sam2_objpc:
                        tracker_factory, extract_all_fn, _ = sam2_runtime
                        with profile_section(step_timings, "sam2", bool(args.profile_timing)):
                            observation = add_sam2_object_pointclouds(
                                observation=observation,
                                dense_scene_pointcloud=dense_scene,
                                args=args,
                                placeholders=placeholders,
                                camera_names=live.labels,
                                tracker_factory=tracker_factory,
                                extract_all_fn=extract_all_fn,
                                tracking_state_by_camera=sam2_tracking_state_by_camera,
                                bbox_prompts_by_camera=sam2_bbox_prompts_by_camera,
                            )
                    with profile_section(step_timings, "encode", bool(args.profile_timing)):
                        encoded_obs = encode_obs(observation, model)
                else:
                    with profile_section(step_timings, "reuse_obs", bool(args.profile_timing)):
                        encoded_obs = encoded_obs_with_agent_vector(
                            encoded_obs,
                            agent_vector_from_joint_action(
                                last_action,
                                args,
                                t_world_from_left_base,
                                t_world_from_right_base,
                            ),
                        )
                model.update_obs(encoded_obs)
                if bool(args.show_img):
                    with profile_section(step_timings, "show_img", bool(args.profile_timing)):
                        maybe_show_cameras(observation, live.labels)
                elapsed = time.time() - action_loop_start
                with profile_section(step_timings, "sleep", bool(args.profile_timing)):
                    if rate_sec > 0 and elapsed < rate_sec:
                        time.sleep(rate_sec - elapsed)
                if bool(args.profile_timing):
                    step_timings["step_total"] = time.perf_counter() - step_wall_start
                    if should_print_step_timing(args, action_step):
                        overrun = max(0.0, elapsed - rate_sec) if rate_sec > 0 else 0.0
                        print_info(
                            f"[TIMING step {action_step:04d}] {format_timings(step_timings)} "
                            f"target_period={rate_sec * 1000.0:.1f}ms overrun={overrun * 1000.0:.1f}ms",
                        )
    finally:
        if action_diagnostic_file is not None:
            action_diagnostic_file.close()
        keyboard_listener.stop()
        live.stop()
        if bool(args.show_img):
            cv2.destroyAllWindows()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RoboTwin DP3 policies on the real three-ZED + xtrainer robot setup.")
    parser.add_argument("--mode", choices=["baseline", "semantic_pointwise_hybrid"], default="baseline")
    parser.add_argument("--task_name", default="grasp_mug")
    parser.add_argument("--task_config", default="demo_real_zed_sam2_objpc")
    parser.add_argument("--ckpt_setting", default="")
    parser.add_argument("--expert_data_num", default="57")
    parser.add_argument("--seed", default="0")
    parser.add_argument("--checkpoint_num", default="3000")
    parser.add_argument("--config_name", default="")
    parser.add_argument("--gpu_id", default="0")
    parser.add_argument("--use_rgb", action="store_true")
    parser.add_argument(
        "--action_mode",
        choices=["joint", "eef_absolute6d"],
        default="joint",
        help="Policy action representation. eef_absolute6d decodes 20D EEF actions in --eef_frame_mode through Dobot IK before ServoJ.",
    )

    parser.add_argument("--semantic_ckpt_A", default=str(DEFAULT_SEMANTIC_CKPT_A))
    parser.add_argument("--semantic_ckpt_B", default="none")
    parser.add_argument("--semantic_device", default="cuda:0")
    parser.add_argument("--semantic_point_num", type=int, default=128)
    parser.add_argument("--object_placeholders", default="{A},{B}")

    parser.add_argument("--calibration_path", default=str(DEFAULT_WORKSPACE_CALIBRATION))
    parser.add_argument("--frame_mode", choices=["reference_camera", "workspace"], default="workspace")
    parser.add_argument("--output_frame", choices=["source", "workspace", "left_base", "right_base"], default="source")
    parser.add_argument("--robot_camera_calibration_path", default="")
    parser.add_argument("--eef_calibration_path", default=str(DEFAULT_WORKSPACE_CALIBRATION))
    parser.add_argument(
        "--eef_frame_mode",
        choices=["reference_camera", "workspace", "left_base", "right_base"],
        default="right_base",
    )
    parser.add_argument(
        "--left_robot_camera_calibration_path",
        default=str(DEFAULT_ROBOT_CAMERA_CALIBRATION_DIR / "robot_camera_apriltag_left_global.yaml"),
    )
    parser.add_argument(
        "--right_robot_camera_calibration_path",
        default=str(DEFAULT_ROBOT_CAMERA_CALIBRATION_DIR / "robot_camera_apriltag_right_global.yaml"),
    )
    parser.add_argument("--eef_tool_x_m", type=float, default=0.0)
    parser.add_argument("--eef_tool_y_m", type=float, default=0.0)
    parser.add_argument("--eef_tool_z_m", type=float, default=0.197)
    parser.add_argument("--serial_remap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--camera_labels",
        default="",
        help="Comma-separated live ZED labels. Empty uses labels from --calibration_path.",
    )
    parser.add_argument("--zed_serials", default="")
    parser.add_argument("--zed_resolution", default="HD720")
    parser.add_argument("--zed_fps", type=int, default=15)
    parser.add_argument("--zed_depth_mode", default="NEURAL")
    parser.add_argument(
        "--zed_auto_exposure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable ZED auto exposure/gain during real inference. Use --no-zed_auto_exposure for fixed exposure/gain.",
    )
    parser.add_argument("--zed_exposure", type=int, default=22)
    parser.add_argument("--zed_gain", type=int, default=12)
    parser.add_argument(
        "--zed_whitebalance_temp",
        type=int,
        default=0,
        help="ZED white-balance temperature in Kelvin. Use 0 for automatic white balance.",
    )
    parser.add_argument("--save_rgb_width", type=int, default=0)
    parser.add_argument("--save_rgb_height", type=int, default=0)
    parser.add_argument("--camera_warmup_timeout_sec", type=float, default=30.0)
    parser.add_argument("--scene_point_num", type=int, default=1024)
    parser.add_argument("--object_point_num", type=int, default=1024)
    parser.add_argument("--min_depth_m", type=float, default=0.05)
    parser.add_argument("--max_depth_m", type=float, default=3.0)
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

    parser.add_argument("--enable_sam2_objpc", action="store_true")
    parser.add_argument("--sam2_root", default=str(DEFAULT_SAM2_ROOT))
    # parser.add_argument("--sam2_config", default="sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--sam2_config", default="sam2.1/sam2.1_hiera_b+.yaml")
    parser.add_argument("--sam2_checkpoint", default=str(DEFAULT_SAM2_CHECKPOINT))
    parser.add_argument("--sam2_device", default="cuda:0")
    parser.add_argument("--sam2_autocast_dtype", default="bfloat16")
    parser.add_argument("--sam2_image_width", type=int, default=640)
    parser.add_argument("--online_object_resample", choices=["fast", "fps"], default="fast")
    parser.add_argument("--sam2_bbox_prompt_path", default="")
    parser.add_argument("--sam2_interactive_init", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sam2_min_mask_points", type=int, default=16)
    parser.add_argument(
        "--reset_sam2_on_keyboard_reset",
        action="store_true",
        help="Clear online SAM2 tracking state on keyboard reset; defaults to keeping active tracks/prompts.",
    )

    parser.add_argument("--robot_port", type=int, default=6001)
    parser.add_argument("--hostname", default="127.0.0.1")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--no_robot", action="store_true")
    parser.add_argument("--skip_robot_reset", action="store_true")
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--keyboard_control", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keyboard_reset_key", default="r")
    parser.add_argument("--keyboard_quit_key", default="q")
    parser.add_argument("--control_hz", type=float, default=10.0)
    parser.add_argument("--async_control", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--async_control_hz",
        type=float,
        default=0.0,
        help="Fixed-rate async robot-control thread frequency; <=0 reuses --control_hz.",
    )
    parser.add_argument(
        "--async_control_max_idle_repeats",
        type=int,
        default=-1,
        help="How many times async control repeats the last target when the action buffer is empty; -1 repeats until stop.",
    )
    parser.add_argument(
        "--async_control_replace_buffer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replace stale pending actions whenever a new policy chunk is submitted.",
    )
    parser.add_argument("--initial_left_gripper", type=float, default=1.0)
    parser.add_argument("--initial_right_gripper", type=float, default=1.0)
    parser.add_argument("--interpolate_first_action", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--first_action_interp_step", type=float, default=0.001)
    parser.add_argument(
        "--execution_substeps",
        type=int,
        default=1,
        help="Split each executed policy action into this many smaller robot commands; 1 restores old behavior.",
    )
    parser.add_argument(
        "--execution_substep_sleep_sec",
        type=float,
        default=0.0,
        help="Sleep between execution substeps to reduce start/stop jerk.",
    )
    parser.add_argument(
        "--servo_j_t",
        type=float,
        default=0.06,
        help="Dobot ServoJ trajectory time parameter. Use 0 to leave the robot-server default unchanged.",
    )
    parser.add_argument(
        "--servo_j_gain",
        type=int,
        default=300,
        help="Dobot ServoJ gain parameter. Use 0 to leave the robot-server default unchanged.",
    )
    parser.add_argument("--max_executed_joint_delta", type=float, default=0.12)
    parser.add_argument("--max_executed_gripper_delta", type=float, default=0.2)
    parser.add_argument(
        "--max_executed_joint_delta_change",
        type=float,
        default=0.0,
        help="Limit per-step change of executed arm-joint delta; 0 disables acceleration-style limiting.",
    )
    parser.add_argument(
        "--max_executed_gripper_delta_change",
        type=float,
        default=0.0,
        help="Limit per-step change of executed gripper delta; 0 disables acceleration-style limiting.",
    )
    parser.add_argument("--max_action_delta", type=float, default=0.35)
    parser.add_argument("--disable_action_delta_safety", action="store_true")
    parser.add_argument("--disable_xyz_safety", action="store_true")
    parser.add_argument("--xyz_left_x_min", type=float, default=-450.0)
    parser.add_argument("--xyz_left_x_max", type=float, default=450.0)
    parser.add_argument("--xyz_right_x_min", type=float, default=-450.0)
    parser.add_argument("--xyz_right_x_max", type=float, default=450.0)
    parser.add_argument("--xyz_y_min", type=float, default=-750.0)
    parser.add_argument("--xyz_y_max", type=float, default=-160.0)
    parser.add_argument("--xyz_left_z_min", type=float, default=25.0)
    parser.add_argument("--xyz_right_z_min", type=float, default=23.0)
    parser.add_argument("--show_img", action="store_true")
    parser.add_argument("--reobserve_each_action", action="store_true")
    parser.add_argument("--profile_timing", action="store_true")
    parser.add_argument("--profile_timing_interval", type=int, default=1)
    parser.add_argument(
        "--action_diagnostics_csv",
        default="",
        help="Optional CSV path for policy-vs-command-vs-observed action delta diagnostics.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_real_inference(args)


if __name__ == "__main__":
    main()
