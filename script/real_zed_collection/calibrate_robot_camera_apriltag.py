#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from script.real_zed_collection.calibrate_three_zed_extrinsics import (
    _capture_frames,
    _configure_zed_image_controls,
    _open_zed,
    load_collection_camera_mapping,
)
from script.real_zed_collection.collect_zed_robotwin_raw import (
    calculate_vel_pos,
    check_joint_safety,
    check_pose_protection,
)


what_to_do = np.array(([0, 0, 0], [0, 0, 0]))


@dataclass
class AprilTagDetection:
    tag_id: int
    marker_dictionary: str
    corners: np.ndarray
    camera_from_tag: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray


@dataclass
class CameraDetectionSnapshot:
    frame: np.ndarray
    detection: AprilTagDetection | None
    timestamp_unix_sec: float


class LatestCameraDetection:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot: CameraDetectionSnapshot | None = None
        self._error: BaseException | None = None

    def update(self, snapshot: CameraDetectionSnapshot) -> None:
        with self._lock:
            self._snapshot = snapshot
            self._error = None

    def set_error(self, error: BaseException) -> None:
        with self._lock:
            self._error = error

    def snapshot(self) -> CameraDetectionSnapshot | None:
        with self._lock:
            if self._error is not None:
                raise RuntimeError("Camera detection worker failed.") from self._error
            return self._snapshot


AUTO_ARUCO_DICTIONARIES = [
    "DICT_4X4_50",
    "DICT_4X4_100",
    "DICT_4X4_250",
    "DICT_5X5_50",
    "DICT_5X5_100",
    "DICT_5X5_250",
    "DICT_6X6_50",
    "DICT_6X6_100",
    "DICT_6X6_250",
    "DICT_7X7_50",
    "DICT_7X7_100",
    "DICT_7X7_250",
    "DICT_ARUCO_ORIGINAL",
    "DICT_APRILTAG_16H5",
    "DICT_APRILTAG_25H9",
    "DICT_APRILTAG_36H10",
    "DICT_APRILTAG_36H11",
]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_three_zed = repo_root / "script" / "real_zed_collection" / "calibration" / "three_camera_charuco_extrinsics.yaml"
    default_collection_config = repo_root / "script" / "real_zed_collection" / "configs" / "real_zed_collection.yaml"
    default_session_root = repo_root / "outputs" / "real_zed_collection" / "robot_camera_calibration"

    parser = argparse.ArgumentParser(
        description=(
            "Collect ArUco/AprilTag marker-on-gripper poses and solve robot-base <-> ZED-camera calibration. "
            "Teleop behavior follows the real-ZED collection script: Button A unlock/servo, Button B captures one sample."
        )
    )
    parser.add_argument("--arm", choices=("left", "right"), default="right")
    parser.add_argument("--camera_label", default="global")
    parser.add_argument(
        "--collection_config",
        default=str(default_collection_config),
        help="Default source for camera_label->serial mapping. Pass empty string to fall back to --calibration_path.",
    )
    parser.add_argument("--calibration_path", default=str(default_three_zed))
    parser.add_argument("--zed_serial", type=int, default=0, help="Overrides --camera_label lookup when >0.")
    parser.add_argument("--zed_resolution", default="HD1080")
    parser.add_argument("--zed_fps", type=int, default=15)
    parser.add_argument("--zed_auto_exposure", action="store_true", default=False)
    parser.add_argument("--zed_exposure", type=int, default=22)
    parser.add_argument("--zed_gain", type=int, default=12)
    parser.add_argument("--zed_whitebalance_temp", type=int, default=None)
    parser.add_argument("--tag_id", type=int, default=4, help="Use the first detected marker when negative.")
    parser.add_argument(
        "--marker_dictionary",
        default="DICT_APRILTAG_36h11",
        help="OpenCV ArUco dictionary name, e.g. DICT_4X4_50, DICT_5X5_100, DICT_APRILTAG_36h11, or auto.",
    )
    parser.add_argument("--tag_size_m", type=float, default=0.04)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--method", choices=("SHAH", "LI"), default="SHAH")
    parser.add_argument("--robot_port", type=int, default=6001)
    parser.add_argument("--hostname", default="127.0.0.1")
    parser.add_argument("--skip_robot_pose_init", action="store_true", default=False)
    parser.add_argument("--pose_xyz_unit", choices=("mm", "m"), default="mm")
    parser.add_argument(
        "--pose_rotation_mode",
        choices=("euler_deg", "euler_rad", "rotvec_deg", "rotvec_rad"),
        default="euler_deg",
    )
    parser.add_argument("--pose_euler_order", default="xyz")
    parser.add_argument("--disable_outlier_rejection", action="store_true", default=False)
    parser.add_argument("--outlier_rotation_threshold_deg", type=float, default=30.0)
    parser.add_argument("--outlier_translation_threshold_m", type=float, default=0.05)
    parser.add_argument("--outlier_rejection_passes", type=int, default=4)
    parser.add_argument("--session_root", default=str(default_session_root))
    parser.add_argument("--output_config", default="")
    parser.add_argument(
        "--recompute_from_yaml",
        default="",
        help="Re-solve an existing result YAML from saved raw samples without collecting new data.",
    )
    parser.add_argument("--show", action="store_true", default=True)
    parser.add_argument("--no_show", dest="show", action="store_false")
    parser.add_argument("--window_name", default="robot_camera_apriltag_calibration")
    parser.add_argument("--window_width", type=int, default=1280)
    parser.add_argument("--window_height", type=int, default=720)
    parser.add_argument("--axis_length_m", type=float, default=0.05)
    parser.add_argument("--camera_poll_interval_sec", type=float, default=0.0)
    return parser.parse_args()


def as_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = transform[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ transform[:3, 3]
    return out


def _axis_rotation(axis: str, angle_rad: float) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    if axis == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)
    if axis == "y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)
    if axis == "z":
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    raise ValueError(f"Unsupported Euler axis: {axis!r}")


def rotation_from_euler(angles: np.ndarray, order: str) -> np.ndarray:
    order = str(order).lower()
    if sorted(order) != ["x", "y", "z"] or len(order) != 3:
        raise ValueError(f"Euler order must be a permutation of xyz, got {order!r}")
    rotation = np.eye(3, dtype=np.float64)
    # Dobot GetPose() reports fixed-axis Euler angles; with column-vector
    # transforms, order xyz composes as Rz @ Ry @ Rx.
    for axis, angle in zip(order, np.asarray(angles, dtype=np.float64).reshape(3)):
        rotation = _axis_rotation(axis, float(angle)) @ rotation
    return rotation


def pose_xyzrxryrz_to_transform(
    pose_xyzrxryrz: np.ndarray,
    *,
    xyz_unit: str = "mm",
    rotation_mode: str = "euler_deg",
    euler_order: str = "xyz",
) -> np.ndarray:
    pose = np.asarray(pose_xyzrxryrz, dtype=np.float64).reshape(6)
    xyz = pose[:3].copy()
    if xyz_unit == "mm":
        xyz *= 0.001
    elif xyz_unit != "m":
        raise ValueError(f"Unsupported xyz unit: {xyz_unit!r}")

    rot_raw = pose[3:6].copy()
    if rotation_mode == "euler_deg":
        rotation = rotation_from_euler(np.deg2rad(rot_raw), euler_order)
    elif rotation_mode == "euler_rad":
        rotation = rotation_from_euler(rot_raw, euler_order)
    elif rotation_mode == "rotvec_deg":
        rotation = cv2.Rodrigues(np.deg2rad(rot_raw))[0]
    elif rotation_mode == "rotvec_rad":
        rotation = cv2.Rodrigues(rot_raw)[0]
    else:
        raise ValueError(f"Unsupported rotation mode: {rotation_mode!r}")
    return as_transform(rotation, xyz)


def _rotation_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    delta = invert_transform(a) @ np.asarray(b, dtype=np.float64).reshape(4, 4)
    trace = float(np.trace(delta[:3, :3]))
    cos_angle = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def _translation_error_m(a: np.ndarray, b: np.ndarray) -> float:
    delta = invert_transform(a) @ np.asarray(b, dtype=np.float64).reshape(4, 4)
    return float(np.linalg.norm(delta[:3, 3]))


def _to_cv_rot_trans(transforms: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    rotations: list[np.ndarray] = []
    translations: list[np.ndarray] = []
    for transform in transforms:
        mat = np.asarray(transform, dtype=np.float64).reshape(4, 4)
        rotations.append(mat[:3, :3].copy())
        translations.append(mat[:3, 3].reshape(3, 1).copy())
    return rotations, translations


def _method_id(method: str) -> int:
    method = str(method).upper()
    if method == "SHAH":
        return cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH
    if method == "LI":
        return cv2.CALIB_ROBOT_WORLD_HAND_EYE_LI
    raise ValueError(f"Unsupported robot-world-hand-eye method: {method!r}")


def solve_robot_camera_calibration(samples: list[dict[str, Any]], method: str = "SHAH") -> dict[str, Any]:
    if len(samples) < 3:
        raise ValueError("At least 3 pose samples are required; 15-20 diverse poses are recommended.")

    base_from_gripper = [np.asarray(sample["base_from_gripper"], dtype=np.float64).reshape(4, 4) for sample in samples]
    camera_from_tag = [np.asarray(sample["camera_from_tag"], dtype=np.float64).reshape(4, 4) for sample in samples]

    # OpenCV solves AX=ZB as robot-world-hand-eye. For a static camera and
    # AprilTag rigidly attached to the gripper, use algorithm "world" as the
    # physical camera frame and algorithm "camera" as the moving AprilTag frame.
    tag_from_camera = [invert_transform(transform) for transform in camera_from_tag]
    gripper_from_base = [invert_transform(transform) for transform in base_from_gripper]

    r_world2cam, t_world2cam = _to_cv_rot_trans(tag_from_camera)
    r_base2gripper, t_base2gripper = _to_cv_rot_trans(gripper_from_base)
    r_camera_from_base, t_camera_from_base, r_tag_from_gripper, t_tag_from_gripper = cv2.calibrateRobotWorldHandEye(
        r_world2cam,
        t_world2cam,
        r_base2gripper,
        t_base2gripper,
        method=_method_id(method),
    )

    camera_from_base = as_transform(r_camera_from_base, t_camera_from_base)
    base_from_camera = invert_transform(camera_from_base)
    tag_from_gripper = as_transform(r_tag_from_gripper, t_tag_from_gripper)

    predicted_camera_from_tag = [
        camera_from_base @ bfg @ invert_transform(tag_from_gripper) for bfg in base_from_gripper
    ]
    translation_errors = [
        _translation_error_m(pred, obs) for pred, obs in zip(predicted_camera_from_tag, camera_from_tag)
    ]
    rotation_errors = [_rotation_error_deg(pred, obs) for pred, obs in zip(predicted_camera_from_tag, camera_from_tag)]

    return {
        "method": str(method).upper(),
        "sample_count": int(len(samples)),
        "camera_from_base": camera_from_base,
        "base_from_camera": base_from_camera,
        "tag_from_gripper": tag_from_gripper,
        "mean_translation_error_m": float(np.mean(translation_errors)),
        "max_translation_error_m": float(np.max(translation_errors)),
        "mean_rotation_error_deg": float(np.mean(rotation_errors)),
        "max_rotation_error_deg": float(np.max(rotation_errors)),
        "per_sample_translation_error_m": [float(x) for x in translation_errors],
        "per_sample_rotation_error_deg": [float(x) for x in rotation_errors],
    }


def solve_robot_camera_calibration_robust(
    samples: list[dict[str, Any]],
    *,
    method: str = "SHAH",
    reject_outliers: bool = True,
    rotation_threshold_deg: float = 30.0,
    translation_threshold_m: float = 0.20,
    max_passes: int = 2,
) -> dict[str, Any]:
    initial_result = solve_robot_camera_calibration(samples, method=method)
    active_indices = list(range(len(samples)))
    rejected_indices: list[int] = []
    result = initial_result

    if reject_outliers:
        for _ in range(max(0, int(max_passes))):
            scores = []
            for local_idx, (trans_err, rot_err) in enumerate(
                zip(result["per_sample_translation_error_m"], result["per_sample_rotation_error_deg"])
            ):
                trans_score = float(trans_err) / max(float(translation_threshold_m), 1e-12)
                rot_score = float(rot_err) / max(float(rotation_threshold_deg), 1e-12)
                scores.append(max(trans_score, rot_score))
            worst_local_idx = int(np.argmax(scores))
            if float(scores[worst_local_idx]) <= 1.0:
                break
            next_active = [
                sample_idx
                for local_idx, sample_idx in enumerate(active_indices)
                if local_idx != worst_local_idx
            ]
            if len(next_active) < 3:
                break
            rejected_indices.append(active_indices[worst_local_idx])
            active_indices = next_active
            result = solve_robot_camera_calibration([samples[idx] for idx in active_indices], method=method)

    result = dict(result)
    result["raw_sample_count"] = int(len(samples))
    result["accepted_sample_indices"] = [int(idx) for idx in active_indices]
    result["rejected_sample_indices"] = sorted({int(idx) for idx in rejected_indices})
    result["outlier_rejection_enabled"] = bool(reject_outliers)
    result["outlier_rotation_threshold_deg"] = float(rotation_threshold_deg)
    result["outlier_translation_threshold_m"] = float(translation_threshold_m)
    result["initial_mean_translation_error_m"] = float(initial_result["mean_translation_error_m"])
    result["initial_max_translation_error_m"] = float(initial_result["max_translation_error_m"])
    result["initial_mean_rotation_error_deg"] = float(initial_result["mean_rotation_error_deg"])
    result["initial_max_rotation_error_deg"] = float(initial_result["max_rotation_error_deg"])
    result["initial_per_sample_translation_error_m"] = initial_result["per_sample_translation_error_m"]
    result["initial_per_sample_rotation_error_deg"] = initial_result["per_sample_rotation_error_deg"]
    return result


def _solve_from_args(samples: list[dict[str, Any]], args: argparse.Namespace, method: str) -> dict[str, Any]:
    return solve_robot_camera_calibration_robust(
        samples,
        method=method,
        reject_outliers=not bool(getattr(args, "disable_outlier_rejection", False)),
        rotation_threshold_deg=float(getattr(args, "outlier_rotation_threshold_deg", 30.0)),
        translation_threshold_m=float(getattr(args, "outlier_translation_threshold_m", 0.20)),
        max_passes=int(getattr(args, "outlier_rejection_passes", 2)),
    )


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return data


def _resolve_camera_serial(args: argparse.Namespace) -> int:
    if int(args.zed_serial) > 0:
        return int(args.zed_serial)
    _config_labels, config_serial_by_label = load_collection_camera_mapping(getattr(args, "collection_config", ""))
    if args.camera_label in config_serial_by_label:
        return int(config_serial_by_label[args.camera_label])
    cfg = _load_yaml(args.calibration_path)
    cameras = cfg.get("cameras", {})
    if not isinstance(cameras, dict) or args.camera_label not in cameras:
        raise ValueError(f"Camera label {args.camera_label!r} not found in {args.calibration_path}")
    serial = int((cameras[args.camera_label] or {}).get("serial_number", 0))
    if serial <= 0:
        raise ValueError(f"Camera label {args.camera_label!r} has no valid serial_number in {args.calibration_path}")
    return serial


def _normalize_aruco_dictionary_name(dictionary_name: str) -> str:
    raw = str(dictionary_name).strip()
    if not raw:
        raise ValueError("marker dictionary name cannot be empty.")
    normalized = raw.upper().replace("-", "_").replace(" ", "")
    if not normalized.startswith("DICT_"):
        normalized = f"DICT_{normalized}"
    return normalized


def candidate_aruco_dictionary_names(marker_dictionary: str) -> list[str]:
    raw = str(marker_dictionary).strip().lower()
    if raw == "auto":
        return [name for name in AUTO_ARUCO_DICTIONARIES if hasattr(cv2.aruco, name)]
    return [_normalize_aruco_dictionary_name(marker_dictionary)]


def resolve_aruco_dictionary_id(dictionary_name: str) -> int:
    normalized = _normalize_aruco_dictionary_name(dictionary_name)
    if not hasattr(cv2.aruco, normalized):
        raise ValueError(
            f"Unsupported OpenCV ArUco dictionary {dictionary_name!r}. "
            "Examples: DICT_4X4_50, DICT_5X5_100, DICT_6X6_250, DICT_APRILTAG_36h11."
        )
    value = getattr(cv2.aruco, normalized)
    if not isinstance(value, int):
        raise ValueError(f"OpenCV attribute is not an ArUco dictionary id: {normalized}")
    return int(value)


def _make_aruco_detector(marker_dictionary: str):
    dictionary = cv2.aruco.getPredefinedDictionary(resolve_aruco_dictionary_id(marker_dictionary))
    if hasattr(cv2.aruco, "ArucoDetector"):
        params = cv2.aruco.DetectorParameters()
        return dictionary, cv2.aruco.ArucoDetector(dictionary, params)
    return dictionary, None


def detect_apriltag_pose(
    bgr: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    *,
    tag_size_m: float,
    tag_id: int = -1,
    marker_dictionary: str = "DICT_APRILTAG_36h11",
) -> AprilTagDetection | None:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    for dictionary_name in candidate_aruco_dictionary_names(marker_dictionary):
        dictionary, detector = _make_aruco_detector(dictionary_name)
        if detector is not None:
            corners, ids, _rejected = detector.detectMarkers(gray)
        else:
            corners, ids, _rejected = cv2.aruco.detectMarkers(gray, dictionary)
        if ids is None or len(ids) == 0:
            continue

        flat_ids = ids.reshape(-1).astype(int)
        selected_idx = 0
        if int(tag_id) >= 0:
            matches = np.where(flat_ids == int(tag_id))[0]
            if len(matches) == 0:
                continue
            selected_idx = int(matches[0])

        marker_corners = [corners[selected_idx]]
        rvecs, tvecs, _obj_points = cv2.aruco.estimatePoseSingleMarkers(
            marker_corners,
            float(tag_size_m),
            np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3),
            np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1),
        )
        rvec = np.asarray(rvecs[0], dtype=np.float64).reshape(3)
        tvec = np.asarray(tvecs[0], dtype=np.float64).reshape(3)
        rotation = cv2.Rodrigues(rvec)[0]
        return AprilTagDetection(
            tag_id=int(flat_ids[selected_idx]),
            marker_dictionary=dictionary_name,
            corners=np.asarray(corners[selected_idx], dtype=np.float64),
            camera_from_tag=as_transform(rotation, tvec),
            rvec=rvec,
            tvec=tvec,
        )
    return None


def draw_detection(
    bgr: np.ndarray,
    detection: AprilTagDetection | None,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    axis_length_m: float,
    status_lines: list[str],
) -> np.ndarray:
    vis = bgr.copy()
    if detection is not None:
        cv2.aruco.drawDetectedMarkers(vis, [detection.corners.astype(np.float32)], np.array([[detection.tag_id]]))
        cv2.drawFrameAxes(
            vis,
            np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3),
            np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1),
            detection.rvec,
            detection.tvec,
            float(axis_length_m),
        )
    y = 28
    for line in status_lines:
        cv2.putText(vis, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        y += 28
    return vis


def resize_for_display(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    if max_width <= 0 or max_height <= 0:
        return image
    height, width = image.shape[:2]
    scale = min(float(max_width) / float(width), float(max_height) / float(height), 1.0)
    if scale >= 1.0:
        return image
    out_width = max(1, int(round(width * scale)))
    out_height = max(1, int(round(height * scale)))
    return cv2.resize(image, (out_width, out_height), interpolation=cv2.INTER_AREA)


def camera_detection_loop(
    stream,
    args: argparse.Namespace,
    latest: LatestCameraDetection,
    stop_event: threading.Event,
    *,
    capture_frames_fn=_capture_frames,
    detect_pose_fn=detect_apriltag_pose,
) -> None:
    try:
        while not stop_event.is_set():
            frames = capture_frames_fn([stream])
            frame = frames[0] if frames else None
            if frame is None:
                stop_event.wait(0.01)
                continue
            detection = detect_pose_fn(
                frame,
                stream.camera_matrix,
                stream.dist_coeffs,
                tag_size_m=float(args.tag_size_m),
                tag_id=int(args.tag_id),
                marker_dictionary=str(args.marker_dictionary),
            )
            latest.update(
                CameraDetectionSnapshot(
                    frame=frame,
                    detection=detection,
                    timestamp_unix_sec=float(time.time()),
                )
            )
            poll_interval = float(getattr(args, "camera_poll_interval_sec", 0.0))
            if poll_interval > 0.0:
                stop_event.wait(poll_interval)
    except BaseException as exc:
        latest.set_error(exc)
        stop_event.set()


def start_camera_detection_worker(
    stream,
    args: argparse.Namespace,
    latest: LatestCameraDetection,
    stop_event: threading.Event,
    *,
    capture_frames_fn=_capture_frames,
    detect_pose_fn=detect_apriltag_pose,
) -> threading.Thread:
    thread = threading.Thread(
        target=camera_detection_loop,
        kwargs={
            "stream": stream,
            "args": args,
            "latest": latest,
            "stop_event": stop_event,
            "capture_frames_fn": capture_frames_fn,
            "detect_pose_fn": detect_pose_fn,
        },
        daemon=True,
    )
    thread.start()
    return thread


def _apply_button_state_transition(
    *,
    now_keys: np.ndarray,
    last_keys_status: np.ndarray,
    start_press_status: np.ndarray,
    keys_press_count: np.ndarray,
    command_state: np.ndarray,
    timestamp: float,
    button_a_press_start: list[float] | None = None,
) -> float:
    if button_a_press_start is None:
        button_a_press_start = [float(timestamp)]
    now_keys = np.asarray(now_keys, dtype=np.int64)
    dev_keys = now_keys - last_keys_status
    for i in range(2):
        if dev_keys[i, 0] == -1:
            button_a_press_start[0] = float(timestamp)
            start_press_status[i, 0] = 1
        if dev_keys[i, 0] == 1 and start_press_status[i, 0]:
            start_press_status[i, 0] = 0
            duration = float(timestamp) - float(button_a_press_start[0])
            if duration < 0.5:
                keys_press_count[i, 0] += 1
                command_state[i, 0] = 1 if keys_press_count[i, 0] % 2 == 1 else 0
                print(f"ButtonA: [{i}] {'unlock' if command_state[i, 0] else 'lock'}", command_state)
            elif duration > 1:
                keys_press_count[i, 1] += 1
                command_state[i, 1] = 1 if keys_press_count[i, 1] % 2 == 1 else 0
                print(f"ButtonA: [{i}] {'servo' if command_state[i, 1] else 'stop servo'}")

    for i in range(2):
        if dev_keys[i, 1] == -1:
            start_press_status[i, 1] = 1
        if dev_keys[i, 1] == 1 and start_press_status[i, 1]:
            start_press_status[i, 1] = 0
            keys_press_count[0, 2] += 1
            command_state[0, 2] = 1

    last_keys_status[...] = now_keys
    return float(button_a_press_start[0])


def button_monitor_realtime(agent) -> None:
    last_keys_status = np.asarray(agent.get_keys(), dtype=np.int64).copy()
    start_press_status = np.array(([0, 0], [0, 0]))
    keys_press_count = np.array(([0, 0, 0], [0, 0, 0]))
    button_a_press_start = [time.time()]

    while True:
        _apply_button_state_transition(
            now_keys=agent.get_keys(),
            last_keys_status=last_keys_status,
            start_press_status=start_press_status,
            keys_press_count=keys_press_count,
            command_state=what_to_do,
            timestamp=time.time(),
            button_a_press_start=button_a_press_start,
        )


def _arm_slice(arm: str) -> slice:
    return slice(0, 6) if str(arm) == "left" else slice(6, 12)


def _matrix_to_list(mat: np.ndarray) -> list[list[float]]:
    return np.asarray(mat, dtype=np.float64).tolist()


def _sample_to_jsonable(sample: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in sample.items():
        if isinstance(value, np.ndarray):
            out[key] = value.tolist()
        elif isinstance(value, (np.float32, np.float64)):
            out[key] = float(value)
        elif isinstance(value, (np.int32, np.int64)):
            out[key] = int(value)
        else:
            out[key] = value
    return out


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def _default_output_config(args: argparse.Namespace) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return (
        repo_root
        / "script"
        / "real_zed_collection"
        / "calibration"
        / f"robot_camera_apriltag_{args.arm}_{args.camera_label}.yaml"
    )


def _recompute_output_config(args: argparse.Namespace, source_path: Path) -> Path:
    if str(args.output_config).strip():
        return Path(args.output_config).expanduser().resolve()
    return source_path.with_name(f"{source_path.stem}_recomputed.yaml")


def recompute_saved_calibration(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.recompute_from_yaml).expanduser().resolve()
    payload = _load_yaml(input_path)
    raw_samples = payload.get("samples", [])
    if not isinstance(raw_samples, list) or len(raw_samples) < 3:
        raise ValueError(f"Expected at least 3 saved samples in {input_path}")

    xyz_unit = str(payload.get("pose_xyz_unit", args.pose_xyz_unit))
    rotation_mode = str(payload.get("pose_rotation_mode", args.pose_rotation_mode))
    euler_order = str(payload.get("pose_euler_order", args.pose_euler_order))
    method = str(payload.get("method", args.method))

    samples: list[dict[str, Any]] = []
    jsonable_samples: list[dict[str, Any]] = []
    for sample in raw_samples:
        pose6 = np.asarray(sample["robot_pose_xyzrxryrz"], dtype=np.float64).reshape(6)
        base_from_gripper = pose_xyzrxryrz_to_transform(
            pose6,
            xyz_unit=xyz_unit,
            rotation_mode=rotation_mode,
            euler_order=euler_order,
        )
        camera_from_tag = np.asarray(sample["camera_from_tag"], dtype=np.float64).reshape(4, 4)
        updated = dict(sample)
        updated["base_from_gripper"] = base_from_gripper
        samples.append(
            {
                "base_from_gripper": base_from_gripper,
                "camera_from_tag": camera_from_tag,
            }
        )
        jsonable_samples.append(_sample_to_jsonable(updated))

    result = _solve_from_args(samples, args, method)
    payload = dict(payload)
    payload["recomputed_at_unix_sec"] = float(time.time())
    payload["recomputed_from_yaml"] = str(input_path)
    payload["pose_xyz_unit"] = xyz_unit
    payload["pose_rotation_mode"] = rotation_mode
    payload["pose_euler_order"] = euler_order
    payload["method"] = result["method"]
    payload["sample_count"] = int(result["sample_count"])
    payload["t_camera_from_base"] = _matrix_to_list(result["camera_from_base"])
    payload["t_base_from_camera"] = _matrix_to_list(result["base_from_camera"])
    payload["t_tag_from_gripper"] = _matrix_to_list(result["tag_from_gripper"])
    payload["mean_translation_error_m"] = float(result["mean_translation_error_m"])
    payload["max_translation_error_m"] = float(result["max_translation_error_m"])
    payload["mean_rotation_error_deg"] = float(result["mean_rotation_error_deg"])
    payload["max_rotation_error_deg"] = float(result["max_rotation_error_deg"])
    payload["per_sample_translation_error_m"] = result["per_sample_translation_error_m"]
    payload["per_sample_rotation_error_deg"] = result["per_sample_rotation_error_deg"]
    payload["raw_sample_count"] = int(result["raw_sample_count"])
    payload["accepted_sample_indices"] = result["accepted_sample_indices"]
    payload["rejected_sample_indices"] = result["rejected_sample_indices"]
    payload["outlier_rejection_enabled"] = bool(result["outlier_rejection_enabled"])
    payload["outlier_rotation_threshold_deg"] = float(result["outlier_rotation_threshold_deg"])
    payload["outlier_translation_threshold_m"] = float(result["outlier_translation_threshold_m"])
    payload["initial_mean_translation_error_m"] = float(result["initial_mean_translation_error_m"])
    payload["initial_max_translation_error_m"] = float(result["initial_max_translation_error_m"])
    payload["initial_mean_rotation_error_deg"] = float(result["initial_mean_rotation_error_deg"])
    payload["initial_max_rotation_error_deg"] = float(result["initial_max_rotation_error_deg"])
    payload["initial_per_sample_translation_error_m"] = result["initial_per_sample_translation_error_m"]
    payload["initial_per_sample_rotation_error_deg"] = result["initial_per_sample_rotation_error_deg"]
    payload["samples"] = jsonable_samples

    output_path = _recompute_output_config(args, input_path)
    _write_yaml(output_path, payload)
    print(f"[INFO] Wrote recomputed calibration YAML: {output_path}")
    print(
        "[INFO] Residuals: "
        f"mean_trans={result['mean_translation_error_m']:.6f}m, "
        f"max_trans={result['max_translation_error_m']:.6f}m, "
        f"mean_rot={result['mean_rotation_error_deg']:.4f}deg, "
        f"max_rot={result['max_rotation_error_deg']:.4f}deg"
    )
    if result["rejected_sample_indices"]:
        print(f"[WARN] Rejected outlier samples: {result['rejected_sample_indices']}")
    return payload


def _load_robot_modules():
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / "include" / "xtrainer_clover",
        Path.home() / "github" / "xtrainer_clover",
    ]
    for candidate in candidates:
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.append(str(candidate))
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

    return {
        "BimanualAgent": BimanualAgent,
        "DobotAgent": DobotAgent,
        "RobotEnv": RobotEnv,
        "ZMQClientRobot": ZMQClientRobot,
        "dynamic_approach": dynamic_approach,
        "load_ini_data_hands": load_ini_data_hands,
        "robot_pose_init": robot_pose_init,
        "servo_action_check": servo_action_check,
        "set_light": set_light,
    }


def _init_robot(args: argparse.Namespace):
    modules = _load_robot_modules()
    _, hands_dict = modules["load_ini_data_hands"]()
    left_agent = modules["DobotAgent"](which_hand="LEFT", dobot_config=hands_dict["HAND_LEFT"])
    right_agent = modules["DobotAgent"](which_hand="RIGHT", dobot_config=hands_dict["HAND_RIGHT"])
    agent = modules["BimanualAgent"](left_agent, right_agent)
    robot_client = modules["ZMQClientRobot"](port=int(args.robot_port), host=args.hostname)
    env = modules["RobotEnv"](robot_client)
    env.set_do_status([1, 0])
    env.set_do_status([2, 0])
    env.set_do_status([3, 0])
    if not bool(args.skip_robot_pose_init):
        modules["robot_pose_init"](env)
    return env, agent, modules


def _capture_robot_pose_sample(env, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    pose_all = np.asarray(env.get_XYZrxryrz_state(), dtype=np.float64).reshape(-1)
    pose6 = pose_all[_arm_slice(args.arm)].copy()
    base_from_gripper = pose_xyzrxryrz_to_transform(
        pose6,
        xyz_unit=args.pose_xyz_unit,
        rotation_mode=args.pose_rotation_mode,
        euler_order=args.pose_euler_order,
    )
    return pose6, base_from_gripper


def run_interactive_collection(args: argparse.Namespace) -> dict[str, Any]:
    serial = _resolve_camera_serial(args)
    session_name = time.strftime("%Y%m%d_%H%M%S") + f"_{args.arm}_{args.camera_label}"
    session_dir = Path(args.session_root).expanduser().resolve() / session_name
    session_dir.mkdir(parents=True, exist_ok=True)

    stream = _open_zed(serial, args.zed_resolution, int(args.zed_fps))
    stream.label = str(args.camera_label)
    zed_settings = _configure_zed_image_controls(
        stream.camera,
        bool(args.zed_auto_exposure),
        int(args.zed_exposure),
        int(args.zed_gain),
        args.zed_whitebalance_temp,
    )
    camera_stop_event = threading.Event()
    latest_camera = LatestCameraDetection()
    camera_thread = start_camera_detection_worker(stream, args, latest_camera, camera_stop_event)

    env, agent, modules = _init_robot(args)
    what_to_do[:] = 0
    button_thread = threading.Thread(target=button_monitor_realtime, args=(agent,), daemon=True)
    button_thread.start()

    last_status = np.array(([0, 0, 0], [0, 0, 0]))
    start_servo = False
    last_action = None
    total_time = 0.04
    safe_limit = 0
    samples: list[dict[str, Any]] = []

    print("[INFO] Robot-camera ArUco/AprilTag calibration started.")
    print("[INFO] Button A short press: unlock/lock. Button A long press: servo on/off. Button B: capture one sample.")
    print(f"[INFO] Need {int(args.samples)} samples. Session dir: {session_dir}")

    try:
        while len(samples) < int(args.samples):
            camera_snapshot = latest_camera.snapshot()
            vis = None
            if camera_snapshot is not None:
                detection = camera_snapshot.detection
                status_lines = [
                    f"arm={args.arm} camera={args.camera_label} sample={len(samples)}/{args.samples}",
                    (
                        f"dict={args.marker_dictionary if detection is None else detection.marker_dictionary} "
                        f"tag={'none' if detection is None else detection.tag_id} size={args.tag_size_m:.4f}m"
                    ),
                    f"camera_age={time.time() - camera_snapshot.timestamp_unix_sec:.3f}s | Button B: capture current pose",
                ]
                vis = draw_detection(
                    camera_snapshot.frame,
                    detection,
                    stream.camera_matrix,
                    stream.dist_coeffs,
                    float(args.axis_length_m),
                    status_lines,
                )
                if bool(args.show):
                    cv2.imshow(
                        str(args.window_name),
                        resize_for_display(vis, int(args.window_width), int(args.window_height)),
                    )
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        raise KeyboardInterrupt
            elif bool(args.show):
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    raise KeyboardInterrupt

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
                last_action = modules["dynamic_approach"](env, agent, flag_in)
                for i in range(2):
                    if what_to_do[i, 0] and what_to_do[i, 1]:
                        agent.set_torque(i, False)
                start_servo = True
                modules["set_light"](env, "yellow", 1)

            if dev_what_to_do[0, 1] == -1 or dev_what_to_do[1, 1] == -1:
                if what_to_do[0, 1] == 0 and what_to_do[1, 1] == 0:
                    modules["set_light"](env, "green", 0)

            if dev_what_to_do[0, 2] == 1:
                what_to_do[0, 2] = 0
                capture_snapshot = latest_camera.snapshot()
                if capture_snapshot is None or capture_snapshot.detection is None:
                    print("[WARN] Capture skipped: no requested marker detected in the latest camera frame.")
                    modules["set_light"](env, "red", 1)
                    time.sleep(0.2)
                    modules["set_light"](env, "yellow", 1)
                else:
                    detection = capture_snapshot.detection
                    pose6, base_from_gripper = _capture_robot_pose_sample(env, args)
                    sample_idx = len(samples)
                    sample = {
                        "index": int(sample_idx),
                        "timestamp_unix_sec": float(time.time()),
                        "arm": str(args.arm),
                        "camera_label": str(args.camera_label),
                        "tag_id": int(detection.tag_id),
                        "marker_dictionary": str(detection.marker_dictionary),
                        "robot_pose_xyzrxryrz": pose6,
                        "base_from_gripper": base_from_gripper,
                        "camera_from_tag": detection.camera_from_tag,
                        "tag_rvec": detection.rvec,
                        "tag_tvec_m": detection.tvec,
                        "camera_timestamp_unix_sec": float(capture_snapshot.timestamp_unix_sec),
                        "camera_frame_age_sec": float(time.time() - capture_snapshot.timestamp_unix_sec),
                    }
                    samples.append(sample)
                    if vis is None:
                        vis = draw_detection(
                            capture_snapshot.frame,
                            detection,
                            stream.camera_matrix,
                            stream.dist_coeffs,
                            float(args.axis_length_m),
                            [
                                f"arm={args.arm} camera={args.camera_label} sample={len(samples)}/{args.samples}",
                                f"dict={detection.marker_dictionary} tag={detection.tag_id} size={args.tag_size_m:.4f}m",
                                "captured",
                            ],
                        )
                    cv2.imwrite(str(session_dir / f"sample_{sample_idx:03d}.png"), vis)
                    _write_json(session_dir / f"sample_{sample_idx:03d}.json", _sample_to_jsonable(sample))
                    print(f"[CAPTURE] sample {len(samples)}/{args.samples}: tag={detection.tag_id}")
                    modules["set_light"](env, "green", 1)
                    time.sleep(0.15)
                    modules["set_light"](env, "yellow", 1)

            if (what_to_do[0, 1] or what_to_do[1, 1]) and start_servo:
                step_start = time.time()
                action = agent.act({})
                flag_in = np.array([what_to_do[0, 1], what_to_do[1, 1]])
                if last_action is not None:
                    err3, action = modules["servo_action_check"](action, last_action, flag_in)
                    if err3 == 0:
                        modules["set_light"](env, "red", 1)
                        raise RuntimeError("servo_action_check failed.")
                    if safe_limit < 1:
                        safe_limit += 1
                    else:
                        positions, vel = calculate_vel_pos(action, last_action, total_time)
                        protect_err = [
                            check_pose_protection(positions, vel, what_to_do),
                            check_joint_safety(action),
                        ]
                        if any(protect_err):
                            modules["set_light"](env, "red", 1)
                            time.sleep(1)
                            raise RuntimeError("Safety protection triggered.")
                env.step(action, flag_in)
                last_action = action
                total_time = max(time.time() - step_start, 1e-6)

        result = _solve_from_args(samples, args, args.method)
        output_config = Path(args.output_config).expanduser().resolve() if args.output_config else _default_output_config(args)
        payload = {
            "format": "robot_camera_apriltag_calibration_v1",
            "created_at_unix_sec": float(time.time()),
            "arm": str(args.arm),
            "camera_label": str(args.camera_label),
            "camera_serial": int(serial),
            "calibration_path": str(Path(args.calibration_path).expanduser().resolve()),
            "session_dir": str(session_dir),
            "tag_id": int(args.tag_id),
            "marker_dictionary": str(args.marker_dictionary),
            "tag_size_m": float(args.tag_size_m),
            "pose_xyz_unit": str(args.pose_xyz_unit),
            "pose_rotation_mode": str(args.pose_rotation_mode),
            "pose_euler_order": str(args.pose_euler_order),
            "zed_settings": zed_settings,
            "camera_matrix": _matrix_to_list(stream.camera_matrix),
            "dist_coeffs": np.asarray(stream.dist_coeffs, dtype=np.float64).reshape(-1).tolist(),
            "method": result["method"],
            "sample_count": int(result["sample_count"]),
            "t_camera_from_base": _matrix_to_list(result["camera_from_base"]),
            "t_base_from_camera": _matrix_to_list(result["base_from_camera"]),
            "t_tag_from_gripper": _matrix_to_list(result["tag_from_gripper"]),
            "mean_translation_error_m": float(result["mean_translation_error_m"]),
            "max_translation_error_m": float(result["max_translation_error_m"]),
            "mean_rotation_error_deg": float(result["mean_rotation_error_deg"]),
            "max_rotation_error_deg": float(result["max_rotation_error_deg"]),
            "per_sample_translation_error_m": result["per_sample_translation_error_m"],
            "per_sample_rotation_error_deg": result["per_sample_rotation_error_deg"],
            "raw_sample_count": int(result["raw_sample_count"]),
            "accepted_sample_indices": result["accepted_sample_indices"],
            "rejected_sample_indices": result["rejected_sample_indices"],
            "outlier_rejection_enabled": bool(result["outlier_rejection_enabled"]),
            "outlier_rotation_threshold_deg": float(result["outlier_rotation_threshold_deg"]),
            "outlier_translation_threshold_m": float(result["outlier_translation_threshold_m"]),
            "initial_mean_translation_error_m": float(result["initial_mean_translation_error_m"]),
            "initial_max_translation_error_m": float(result["initial_max_translation_error_m"]),
            "initial_mean_rotation_error_deg": float(result["initial_mean_rotation_error_deg"]),
            "initial_max_rotation_error_deg": float(result["initial_max_rotation_error_deg"]),
            "initial_per_sample_translation_error_m": result["initial_per_sample_translation_error_m"],
            "initial_per_sample_rotation_error_deg": result["initial_per_sample_rotation_error_deg"],
            "samples": [_sample_to_jsonable(sample) for sample in samples],
        }
        _write_yaml(output_config, payload)
        _write_yaml(session_dir / "robot_camera_apriltag_result.yaml", payload)
        print(f"[INFO] Wrote calibration YAML: {output_config}")
        print(
            "[INFO] Residuals: "
            f"mean_trans={result['mean_translation_error_m']:.6f}m, "
            f"max_trans={result['max_translation_error_m']:.6f}m, "
            f"mean_rot={result['mean_rotation_error_deg']:.4f}deg, "
            f"max_rot={result['max_rotation_error_deg']:.4f}deg"
        )
        if result["rejected_sample_indices"]:
            print(f"[WARN] Rejected outlier samples: {result['rejected_sample_indices']}")
        return payload
    finally:
        camera_stop_event.set()
        camera_thread.join(timeout=2.0)
        try:
            stream.camera.close()
        except Exception:
            pass
        if bool(args.show):
            cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    if str(args.recompute_from_yaml).strip():
        recompute_saved_calibration(args)
    else:
        run_interactive_collection(args)


if __name__ == "__main__":
    main()
