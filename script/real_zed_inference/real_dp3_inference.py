#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
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
    joint_vector: np.ndarray,
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
        "joint_action": {"vector": np.asarray(joint_vector, dtype=np.float32).reshape(14)},
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


def encoded_obs_with_joint_vector(encoded_obs: Mapping[str, Any], joint_vector: np.ndarray) -> dict[str, Any]:
    out = dict(encoded_obs)
    out["agent_pos"] = np.asarray(joint_vector, dtype=np.float32).reshape(14)
    return out


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


def load_dp3_model(args: argparse.Namespace):
    if args.gpu_id is not None and str(args.gpu_id).strip() != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    from deploy_policy import encode_obs, get_model  # noqa: PLC0415

    config_name, ckpt_setting = derive_dp3_settings(args)
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
        arm_indices = np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12], dtype=np.int64)
        delta = np.clip(out[arm_indices] - last[arm_indices], -joint_limit, joint_limit)
        out[arm_indices] = last[arm_indices] + delta

    if gripper_limit > 0.0:
        for idx in (6, 13):
            delta = float(np.clip(float(out[idx] - last[idx]), -gripper_limit, gripper_limit))
            out[idx] = float(last[idx] + delta)

    return clamp_action(out)


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
    arm_indices = np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12], dtype=np.int64)
    gripper_indices = np.asarray([6, 13], dtype=np.int64)
    arm_delta = float(np.max(np.abs(current[arm_indices] - last[arm_indices])))
    gripper_delta = float(np.max(np.abs(current[gripper_indices] - last[gripper_indices])))
    return arm_delta, gripper_delta


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
    ok = (
        (pos[0] > -410)
        and (pos[0] < 210)
        and (pos[1] > -700)
        and (pos[1] < -210)
        and (pos[2] > 42)
        and (pos[6] < 410)
        and (pos[6] > -210)
        and (pos[7] > -700)
        and (pos[7] < -210)
        and (pos[8] > 42)
    )
    if not ok:
        raise RuntimeError(f"XYZ safety stop. Current robot XYZrxryrz state: {pos.tolist()}")


def execute_action(
    *,
    env,
    action: np.ndarray,
    last_action: np.ndarray,
    args: argparse.Namespace,
    first_action: bool,
) -> np.ndarray:
    action = clamp_action(action)
    action = limit_action_delta_for_execution(action, last_action, args)
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

    print("[INFO] Starting ZED cameras ...")
    live = start_zed_cameras(args)
    env = None
    try:
        print("[INFO] Initializing robot interface ...")
        env = build_robot_env(args)
        configure_robot_servo_params(env, args)
        if env is not None and bool(args.execute) and not bool(args.skip_robot_reset):
            last_action = reset_robot_to_photo_pose(env)
        elif env is not None:
            print("[INFO] Reading initial robot joint state ...")
            raw = np.asarray(env.get_obs()["joint_positions"], dtype=np.float32).reshape(14)
            last_action = raw.copy()
        else:
            last_action = np.zeros(14, dtype=np.float32)
        last_action[6] = float(args.initial_left_gripper)
        last_action[13] = float(args.initial_right_gripper)

        if needs_sam2_objpc:
            print("[INFO] Loading SAM2 runtime for object point clouds ...")
            sam2_runtime = load_sam2_runtime(args, placeholders, live.labels)
            _, _, loaded_prompts = sam2_runtime
            sam2_bbox_prompts_by_camera.update(loaded_prompts)

        if not bool(args.execute):
            print("[INFO] Dry-run mode: model actions are printed but not sent to the robot. Pass --execute to move.")

        action_step = 0
        first_action = True
        rate_sec = 0.0 if float(args.control_hz) <= 0 else 1.0 / float(args.control_hz)
        while action_step < int(args.max_steps):
            chunk_timings: dict[str, float] = {}
            with profile_section(chunk_timings, "robot_obs_pre", bool(args.profile_timing)):
                joints = current_joint_vector(env, last_action)
            print("[TRACE] Building live point-cloud observation ...", flush=True)
            with profile_section(chunk_timings, "build_obs_pre", bool(args.profile_timing)):
                observation, dense_scene = build_robotwin_observation(args=args, live=live, joint_vector=joints)
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
            if bool(args.show_img):
                with profile_section(chunk_timings, "show_img_pre", bool(args.profile_timing)):
                    maybe_show_cameras(observation, live.labels)

            with profile_section(chunk_timings, "encode_pre", bool(args.profile_timing)):
                encoded_obs = encode_obs(observation, model)
            model.update_obs(encoded_obs)
            print("[TRACE] Running DP3 policy inference ...", flush=True)
            with profile_section(chunk_timings, "dp3_policy", bool(args.profile_timing)):
                actions = np.asarray(model.get_action(), dtype=np.float32).reshape(-1, 14)
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
                action_loop_start = time.time()
                step_timings: dict[str, float] = {}
                step_wall_start = time.perf_counter()
                last_action_before_step = last_action.copy()
                pred_arm_delta, pred_gripper_delta = action_delta_summary(action, last_action_before_step)
                with profile_section(step_timings, "execute", bool(args.profile_timing)):
                    last_action = execute_action(
                        env=env,
                        action=action,
                        last_action=last_action_before_step,
                        args=args,
                        first_action=first_action,
                    )
                exec_arm_delta, exec_gripper_delta = action_delta_summary(last_action, last_action_before_step)
                first_action = False
                action_step += 1
                print(
                    f"[STEP {action_step:04d}] "
                    f"left_gripper={last_action[6]:.3f} right_gripper={last_action[13]:.3f} "
                    f"pred_arm_delta={pred_arm_delta:.4f} exec_arm_delta={exec_arm_delta:.4f} "
                    f"pred_gripper_delta={pred_gripper_delta:.3f} exec_gripper_delta={exec_gripper_delta:.3f} "
                    f"execute={bool(args.execute)}"
                )

                if bool(args.reobserve_each_action):
                    with profile_section(step_timings, "robot_obs", bool(args.profile_timing)):
                        joints = current_joint_vector(env, last_action)
                    with profile_section(step_timings, "build_obs", bool(args.profile_timing)):
                        observation, dense_scene = build_robotwin_observation(args=args, live=live, joint_vector=joints)
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
                        encoded_obs = encoded_obs_with_joint_vector(encoded_obs, last_action)
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

    parser.add_argument("--semantic_ckpt_A", default=str(DEFAULT_SEMANTIC_CKPT_A))
    parser.add_argument("--semantic_ckpt_B", default="none")
    parser.add_argument("--semantic_device", default="cuda:0")
    parser.add_argument("--semantic_point_num", type=int, default=128)
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

    parser.add_argument("--robot_port", type=int, default=6001)
    parser.add_argument("--hostname", default="127.0.0.1")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--no_robot", action="store_true")
    parser.add_argument("--skip_robot_reset", action="store_true")
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--control_hz", type=float, default=10.0)
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
    parser.add_argument("--max_action_delta", type=float, default=0.35)
    parser.add_argument("--disable_action_delta_safety", action="store_true")
    parser.add_argument("--disable_xyz_safety", action="store_true")
    parser.add_argument("--show_img", action="store_true")
    parser.add_argument("--reobserve_each_action", action="store_true")
    parser.add_argument("--profile_timing", action="store_true")
    parser.add_argument("--profile_timing_interval", type=int, default=1)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_real_inference(args)


if __name__ == "__main__":
    main()
