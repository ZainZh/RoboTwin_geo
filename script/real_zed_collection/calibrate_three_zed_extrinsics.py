#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import select
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

try:
    import pyzed.sl as sl
except Exception:
    sl = None


@dataclass
class CameraStream:
    label: str
    serial: int
    camera: sl.Camera
    runtime: sl.RuntimeParameters
    image_mat: sl.Mat
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray


@dataclass
class PositionEstimate:
    index: int
    avg_cam_from_board: dict[str, np.ndarray]
    sample_count: int
    board_world_xyz_ref_m: list[float]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_output = repo_root / "script" / "real_zed_collection" / "calibration" / "three_camera_charuco_extrinsics.yaml"
    default_collection_config = repo_root / "script" / "real_zed_collection" / "configs" / "real_zed_collection.yaml"
    default_geometry_repo = Path.home() / "github" / "geometry_awareness_manipulation"

    parser = argparse.ArgumentParser(
        description=(
            "Interactive multi-pose calibration for three ZED cameras using one shared Charuco board. "
            "The output is a self-contained YAML with intrinsics and transforms into a reference camera frame."
        )
    )
    parser.add_argument("--serials", type=int, nargs="*", default=None, metavar="SN")
    parser.add_argument("--labels", type=str, nargs=3, default=None, metavar=("L0", "L1", "L2"))
    parser.add_argument(
        "--collection_config",
        type=str,
        default=str(default_collection_config),
        help="Default source for camera_labels/zed_serials when --serials is omitted. Pass empty string to disable.",
    )
    parser.add_argument("--reference_label", type=str, default="global")
    # parser.add_argument("--charuco_config_name", type=str, default="charuco_A4")
    parser.add_argument("--charuco_config_name", type=str, default="charuco_300_9x14_20mm_15mm")
    parser.add_argument(
        "--charuco_config_path",
        type=str,
        default="",
        help="Optional explicit Charuco YAML path. If omitted, --charuco_config_name is resolved from geometry repo.",
    )
    parser.add_argument(
        "--geometry_repo",
        type=str,
        default=str(default_geometry_repo),
        help="Used only to find config/device/<charuco_config_name>.yaml when --charuco_config_path is empty.",
    )
    parser.add_argument("--zed_resolution", type=str, default="HD1080")
    parser.add_argument("--zed_fps", type=int, default=15)
    parser.add_argument("--zed_auto_exposure", action="store_true", default=False)
    parser.add_argument("--zed_exposure", type=int, default=22)
    parser.add_argument("--zed_gain", type=int, default=12)
    parser.add_argument("--zed_whitebalance_temp", type=int, default=None)
    parser.add_argument("--num_positions", type=int, default=25)
    parser.add_argument("--samples_per_position", type=int, default=1)
    parser.add_argument("--position_timeout_sec", type=float, default=15.0)
    parser.add_argument("--output_config", type=str, default=str(default_output))
    parser.add_argument("--show", action="store_true", default=True)
    parser.add_argument("--no-show", dest="show", action="store_false")
    parser.add_argument("--window_name", type=str, default="three_zed_charuco_calibration")
    parser.add_argument("--window_width", type=int, default=2600)
    parser.add_argument("--window_height", type=int, default=900)
    parser.add_argument("--panel_height", type=int, default=800)
    parser.add_argument("--axis_length", type=float, default=0.035)
    return parser.parse_args()


def _parse_camera_labels(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def load_collection_camera_mapping(config_path: str | Path | None) -> tuple[list[str], dict[str, int]]:
    if config_path is None or not str(config_path).strip():
        return [], {}
    path = Path(config_path).expanduser()
    if not path.exists():
        return [], {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Collection config must be a YAML mapping: {path}")
    labels = _parse_camera_labels(data.get("camera_labels"))
    serials = [int(item) for item in (data.get("zed_serials") or [])]
    if not labels and not serials:
        return [], {}
    if len(labels) != len(serials):
        raise ValueError(
            f"Collection config camera_labels and zed_serials length mismatch: "
            f"{path} labels={labels} serials={serials}"
        )
    if len(set(labels)) != len(labels):
        raise ValueError(f"Collection config camera_labels must be unique: {labels}")
    if len(set(serials)) != len(serials):
        raise ValueError(f"Collection config zed_serials must be unique: {serials}")
    return labels, dict(zip(labels, serials))


def _as_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def _invert_transform(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = transform[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ transform[:3, 3]
    return out


def _mat_to_quat(rotation: np.ndarray) -> np.ndarray:
    r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = np.trace(r)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r[2, 1] - r[1, 2]) / s
        qy = (r[0, 2] - r[2, 0]) / s
        qz = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        qw = (r[2, 1] - r[1, 2]) / s
        qx = 0.25 * s
        qy = (r[0, 1] + r[1, 0]) / s
        qz = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        qw = (r[0, 2] - r[2, 0]) / s
        qx = (r[0, 1] + r[1, 0]) / s
        qy = 0.25 * s
        qz = (r[1, 2] + r[2, 1]) / s
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        qw = (r[1, 0] - r[0, 1]) / s
        qx = (r[0, 2] + r[2, 0]) / s
        qy = (r[1, 2] + r[2, 1]) / s
        qz = 0.25 * s
    quat = np.array([qx, qy, qz, qw], dtype=np.float64)
    return quat / (np.linalg.norm(quat) + 1e-12)


def _quat_to_mat(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quat, dtype=np.float64).reshape(4)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _average_transform(transforms: list[np.ndarray]) -> np.ndarray:
    if not transforms:
        return np.eye(4, dtype=np.float64)
    mats = [np.asarray(x, dtype=np.float64).reshape(4, 4) for x in transforms]
    translations = np.stack([x[:3, 3] for x in mats], axis=0)
    quats = np.stack([_mat_to_quat(x[:3, :3]) for x in mats], axis=0)
    ref = quats[0]
    for i in range(len(quats)):
        if float(np.dot(ref, quats[i])) < 0.0:
            quats[i] *= -1.0
    quat = np.mean(quats, axis=0)
    quat = quat / (np.linalg.norm(quat) + 1e-12)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = _quat_to_mat(quat)
    out[:3, 3] = np.median(translations, axis=0)
    return out


def _se3_residual(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    delta = _invert_transform(a) @ np.asarray(b, dtype=np.float64).reshape(4, 4)
    r = delta[:3, :3]
    cos_angle = np.clip((np.trace(r) - 1.0) * 0.5, -1.0, 1.0)
    angle_deg = float(np.degrees(np.arccos(cos_angle)))
    trans_m = float(np.linalg.norm(delta[:3, 3]))
    return angle_deg, trans_m


def _extract_serial_from_device_info(device) -> int | None:
    if device is None:
        return None
    if hasattr(device, "serial_number"):
        try:
            return int(getattr(device, "serial_number"))
        except Exception:
            pass
    if isinstance(device, dict):
        for key in ("serial_number", "serial", "sn"):
            if key in device:
                try:
                    return int(device[key])
                except Exception:
                    pass
    try:
        return int(device)
    except Exception:
        return None


def _discover_connected_serials() -> list[int]:
    if sl is None:
        return []
    try:
        devices = sl.Camera.get_device_list()
    except Exception:
        devices = []
    serials = []
    for dev in devices:
        serial = _extract_serial_from_device_info(dev)
        if serial is not None and serial > 0:
            serials.append(int(serial))
    return sorted(set(serials))


def _log_label_serial_mapping(labels: list[str], serials: list[int], serial_source: str) -> None:
    print(f"[INFO] Camera label -> serial mapping ({serial_source}):")
    for label, serial in zip(labels, serials):
        print(f"  - {label}: {serial}")


def _resolve_charuco_config_path(args: argparse.Namespace) -> Path | None:
    if str(args.charuco_config_path).strip():
        return Path(args.charuco_config_path).expanduser().resolve()
    geometry_repo = Path(args.geometry_repo).expanduser()
    candidate = geometry_repo / "config" / "device" / f"{args.charuco_config_name}.yaml"
    if candidate.exists():
        return candidate.resolve()
    local_charuco_candidate = Path(__file__).resolve().parent / "charuco_config" / f"{args.charuco_config_name}.yaml"
    if local_charuco_candidate.exists():
        return local_charuco_candidate.resolve()
    local_candidate = Path(__file__).resolve().parent / "configs" / f"{args.charuco_config_name}.yaml"
    if local_candidate.exists():
        return local_candidate.resolve()
    return None


def _resolve_aruco_dictionary_name(dictionary_value: str | None, required_markers: int) -> str:
    default_name = "DICT_5X5_100"
    if not dictionary_value:
        return default_name

    raw = str(dictionary_value).strip().upper()
    normalized = raw.replace("*", "X").replace("-", "_").replace(" ", "")
    if not normalized.startswith("DICT_"):
        normalized = f"DICT_{normalized}"

    if hasattr(cv2.aruco, normalized):
        return normalized

    family_aliases = {
        "DICT_4X4": ["DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000"],
        "DICT_5X5": ["DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000"],
        "DICT_6X6": ["DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000"],
        "DICT_7X7": ["DICT_7X7_50", "DICT_7X7_100", "DICT_7X7_250", "DICT_7X7_1000"],
    }
    options = family_aliases.get(normalized)
    if not options:
        return default_name

    for option in options:
        try:
            capacity = int(option.rsplit("_", 1)[-1])
        except Exception:
            continue
        if capacity >= required_markers and hasattr(cv2.aruco, option):
            return option
    return next((option for option in options if hasattr(cv2.aruco, option)), default_name)


def _charuco_board_from_config(args: argparse.Namespace) -> tuple[cv2.aruco_CharucoBoard, cv2.aruco_Dictionary, dict]:
    cfg_path = _resolve_charuco_config_path(args)
    if cfg_path is not None and cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {
            "squares_x": 7,
            "squares_y": 9,
            "square_length_mm": 30,
            "marker_length_mm": 22,
            "dictionary": "DICT_5X5_100",
        }
    squares_x = int(cfg["squares_x"])
    squares_y = int(cfg["squares_y"])
    square_len_m = float(cfg["square_length_mm"]) / 1000.0
    marker_len_m = float(cfg["marker_length_mm"]) / 1000.0
    required_markers = max(1, (squares_x * squares_y + 1) // 2)
    dictionary_name = _resolve_aruco_dictionary_name(cfg.get("dictionary"), required_markers)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    if hasattr(cv2.aruco.CharucoBoard, "create"):
        board = cv2.aruco.CharucoBoard.create(squares_x, squares_y, square_len_m, marker_len_m, dictionary)
    else:
        board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_len_m, marker_len_m, dictionary)
    return board, dictionary, {"path": None if cfg_path is None else str(cfg_path), "resolved_dictionary": dictionary_name, **cfg}


def _detect_charuco_pose(
    img_bgr: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    board: cv2.aruco_CharucoBoard,
    dictionary: cv2.aruco_Dictionary,
) -> tuple[bool, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    try:
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        corners, ids, rejected = detector.detectMarkers(gray)
    except Exception:
        try:
            params = cv2.aruco.DetectorParameters_create()
        except Exception:
            params = None
        if params is None:
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)

    if ids is None or len(ids) == 0:
        return False, None, None, None

    try:
        refined = cv2.aruco.refineDetectedMarkers(
            image=gray,
            board=board,
            detectedCorners=corners,
            detectedIds=ids,
            rejectedCorners=rejected if rejected is not None else [],
            cameraMatrix=camera_matrix,
            distCoeffs=dist_coeffs,
        )
        if isinstance(refined, tuple) and len(refined) >= 3:
            corners, ids, rejected = refined[:3]
    except Exception:
        pass

    _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        markerCorners=corners,
        markerIds=ids,
        image=gray,
        board=board,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs,
    )
    if charuco_corners is None or charuco_ids is None or len(charuco_corners) < 6:
        return False, None, None, None

    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    try:
        result = cv2.aruco.estimatePoseCharucoBoard(
            charucoCorners=charuco_corners,
            charucoIds=charuco_ids,
            board=board,
            cameraMatrix=camera_matrix,
            distCoeffs=dist_coeffs,
            rvec=rvec,
            tvec=tvec,
        )
    except cv2.error:
        return False, None, None, None
    if isinstance(result, tuple):
        ok, rvec, tvec = result
    else:
        ok = bool(result)
    if not ok:
        return False, None, None, None

    rotation = cv2.Rodrigues(rvec)[0]
    t_cam_from_board = _as_transform(rotation, tvec.reshape(3))
    return True, t_cam_from_board, np.asarray(rvec).reshape(3), np.asarray(tvec).reshape(3)


def _open_zed(serial: int, resolution: str, fps: int) -> CameraStream:
    zed = sl.Camera()
    init = sl.InitParameters()
    init.set_from_serial_number(int(serial))
    res_map = {
        "HD2K": sl.RESOLUTION.HD2K,
        "HD1080": sl.RESOLUTION.HD1080,
        "HD720": sl.RESOLUTION.HD720,
        "VGA": sl.RESOLUTION.VGA,
    }
    init.camera_resolution = res_map.get(str(resolution).upper(), sl.RESOLUTION.HD1080)
    init.camera_fps = int(fps)
    init.depth_mode = getattr(sl.DEPTH_MODE, "NEURAL", sl.DEPTH_MODE.QUALITY)
    init.coordinate_units = sl.UNIT.METER
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open ZED serial={serial}: {status}")

    info = zed.get_camera_information()
    calib_raw = getattr(info, "calibration_parameters_raw", None)
    calib = calib_raw if calib_raw is not None else info.camera_configuration.calibration_parameters
    left = calib.left_cam
    k = np.array(
        [[left.fx, 0.0, left.cx], [0.0, left.fy, left.cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    try:
        d = np.asarray(left.disto, dtype=np.float64).reshape(-1)
        if d.size > 5:
            d = d[:5]
        d = d.reshape(-1, 1)
    except Exception:
        d = np.zeros((5, 1), dtype=np.float64)

    runtime = sl.RuntimeParameters()
    runtime.enable_depth = False
    return CameraStream(
        label="",
        serial=int(serial),
        camera=zed,
        runtime=runtime,
        image_mat=sl.Mat(),
        camera_matrix=k,
        dist_coeffs=d,
    )


def _clamp_int(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(max_value, int(value)))


def _set_camera_setting(camera: sl.Camera, setting_name: str, value: int) -> bool:
    setting = getattr(sl.VIDEO_SETTINGS, setting_name, None)
    if setting is None:
        return False
    try:
        return camera.set_camera_settings(setting, int(value)) == sl.ERROR_CODE.SUCCESS
    except Exception:
        return False


def _configure_zed_image_controls(
    camera: sl.Camera,
    auto_exposure: bool,
    exposure: int,
    gain: int,
    whitebalance_temp: Optional[int],
) -> dict[str, int | bool | None]:
    applied: dict[str, int | bool | None] = {
        "auto_exposure": bool(auto_exposure),
        "exposure": None,
        "gain": None,
        "whitebalance_auto": True,
        "whitebalance_temp": None,
    }
    _set_camera_setting(camera, "AEC_AGC", 1 if auto_exposure else 0)
    if not auto_exposure:
        exp = _clamp_int(exposure, 0, 100)
        gn = _clamp_int(gain, 0, 100)
        _set_camera_setting(camera, "EXPOSURE", exp)
        _set_camera_setting(camera, "GAIN", gn)
        applied["exposure"] = exp
        applied["gain"] = gn
    if whitebalance_temp is not None:
        wb = _clamp_int(whitebalance_temp, 2800, 6500)
        _set_camera_setting(camera, "WHITEBALANCE_AUTO", 0)
        _set_camera_setting(camera, "WHITEBALANCE_TEMPERATURE", wb)
        applied["whitebalance_auto"] = False
        applied["whitebalance_temp"] = wb
    else:
        _set_camera_setting(camera, "WHITEBALANCE_AUTO", 1)
    return applied


def _to_float_list(mat: np.ndarray) -> list[list[float]]:
    return np.asarray(mat, dtype=np.float64).tolist()


def _capture_frames(streams: list[CameraStream]) -> list[Optional[np.ndarray]]:
    frames: list[Optional[np.ndarray]] = []
    for stream in streams:
        if stream.camera.grab(stream.runtime) != sl.ERROR_CODE.SUCCESS:
            frames.append(None)
            continue
        stream.camera.retrieve_image(stream.image_mat, sl.VIEW.LEFT)
        raw = stream.image_mat.get_data()
        if raw is None:
            frames.append(None)
            continue
        if raw.ndim == 3 and raw.shape[2] == 4:
            bgr = raw[:, :, :3].copy()
        elif raw.ndim == 3 and raw.shape[2] == 3:
            bgr = raw.copy()
        else:
            frames.append(None)
            continue
        frames.append(np.ascontiguousarray(bgr))
    return frames


def _detect_frames(
    streams: list[CameraStream],
    frames: list[Optional[np.ndarray]],
    board,
    dictionary,
) -> tuple[list[Optional[np.ndarray]], list[Optional[np.ndarray]], list[Optional[np.ndarray]], bool]:
    poses: list[Optional[np.ndarray]] = []
    rvecs: list[Optional[np.ndarray]] = []
    tvecs: list[Optional[np.ndarray]] = []
    ok_all = True
    for stream, frame in zip(streams, frames):
        if frame is None:
            ok_all = False
            poses.append(None)
            rvecs.append(None)
            tvecs.append(None)
            continue
        ok, t_cam_from_board, rvec, tvec = _detect_charuco_pose(
            frame,
            stream.camera_matrix,
            stream.dist_coeffs,
            board,
            dictionary,
        )
        if not ok or t_cam_from_board is None:
            ok_all = False
            poses.append(None)
            rvecs.append(None)
            tvecs.append(None)
            continue
        poses.append(t_cam_from_board)
        rvecs.append(rvec)
        tvecs.append(tvec)
    return poses, rvecs, tvecs, ok_all


def _consume_terminal_enter() -> bool:
    try:
        if not sys.stdin or sys.stdin.closed or not sys.stdin.isatty():
            return False
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not ready:
            return False
        return sys.stdin.readline() != ""
    except Exception:
        return False


def _draw_axes_safe(
    img: np.ndarray,
    k: np.ndarray,
    d: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    axis_length: float,
) -> None:
    try:
        axis_pts = np.array(
            [
                [0.0, 0.0, 0.0],
                [axis_length, 0.0, 0.0],
                [0.0, axis_length, 0.0],
                [0.0, 0.0, axis_length],
            ],
            dtype=np.float64,
        )
        proj, _ = cv2.projectPoints(axis_pts, rvec.reshape(3, 1), tvec.reshape(3, 1), k, d)
        proj2 = proj.reshape(-1, 2)
        h, w = img.shape[:2]
        in_frame = (
            np.isfinite(proj2).all()
            and np.all(proj2[:, 0] >= 0)
            and np.all(proj2[:, 0] < w)
            and np.all(proj2[:, 1] >= 0)
            and np.all(proj2[:, 1] < h)
        )
        if in_frame:
            cv2.drawFrameAxes(img, k, d, rvec.reshape(3, 1), tvec.reshape(3, 1), axis_length)
    except Exception:
        pass


def _compute_rel_from_positions(
    position_estimates: list[PositionEstimate],
    labels: list[str],
    ref_label: str,
) -> dict[str, np.ndarray]:
    if not position_estimates:
        return {}
    rel_samples: dict[str, list[np.ndarray]] = {label: [] for label in labels}
    for pos in position_estimates:
        t_ref_from_board = pos.avg_cam_from_board[ref_label]
        for label in labels:
            t_board_from_cam = _invert_transform(pos.avg_cam_from_board[label])
            t_ref_from_cam = t_ref_from_board @ t_board_from_cam
            rel_samples[label].append(t_ref_from_cam)
    out: dict[str, np.ndarray] = {}
    for label in labels:
        out[label] = np.eye(4, dtype=np.float64) if label == ref_label else _average_transform(rel_samples[label])
    return out


def _format_xyz(xyz: Optional[np.ndarray]) -> str:
    if xyz is None:
        return "n/a"
    return f"[{xyz[0]:+.3f}, {xyz[1]:+.3f}, {xyz[2]:+.3f}]"


def _draw_triplet_view(
    streams: list[CameraStream],
    frames: list[Optional[np.ndarray]],
    poses: list[Optional[np.ndarray]],
    rvecs: list[Optional[np.ndarray]],
    tvecs: list[Optional[np.ndarray]],
    ref_label: str,
    running_rel: dict[str, np.ndarray],
    pose_idx: int,
    num_positions: int,
    sample_done: int,
    sample_target: int,
    stage_text: str,
    panel_height: int,
    axis_length: float,
) -> Optional[np.ndarray]:
    panels: list[np.ndarray] = []
    ref_idx = [s.label for s in streams].index(ref_label)
    ref_pose = poses[ref_idx]
    for idx, stream in enumerate(streams):
        frame = frames[idx]
        if frame is None:
            panel = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(panel, f"{stream.label} (no frame)", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            panels.append(panel)
            continue

        vis = frame.copy()
        pose = poses[idx]
        rvec = rvecs[idx]
        tvec = tvecs[idx]
        detected = pose is not None
        if detected and rvec is not None and tvec is not None:
            _draw_axes_safe(vis, stream.camera_matrix, stream.dist_coeffs, rvec, tvec, axis_length=axis_length)

        cam_xyz = None if pose is None else pose[:3, 3]
        if pose is not None and stream.label in running_rel:
            world_xyz = (running_rel[stream.label] @ pose)[:3, 3]
        elif ref_pose is not None:
            world_xyz = ref_pose[:3, 3]
        else:
            world_xyz = None

        lines = [
            f"{stream.label} | SN:{stream.serial}",
            f"detect: {'OK' if detected else 'NO'}",
            f"board@cam(m): {_format_xyz(cam_xyz)}",
            f"board@world(ref,m): {_format_xyz(world_xyz)}",
        ]
        y = 30
        for li, txt in enumerate(lines):
            color = (0, 255, 0) if li <= 1 and detected else (255, 255, 255)
            cv2.putText(vis, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            y += 24
        panels.append(vis)

    if not panels:
        return None
    target_h = max(200, int(panel_height))
    resized = []
    for panel in panels:
        scale = target_h / float(panel.shape[0])
        resized.append(cv2.resize(panel, (int(round(panel.shape[1] * scale)), target_h)))
    canvas = cv2.hconcat(resized)
    head_1 = f"Pose {pose_idx}/{num_positions} | samples {sample_done}/{sample_target} | {stage_text}"
    head_2 = "Press Enter in terminal (or in window) to confirm pose. Press 'q' to quit."
    cv2.putText(canvas, head_1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, head_2, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2, cv2.LINE_AA)
    return canvas


def _collect_samples_for_pose(
    streams: list[CameraStream],
    board,
    dictionary,
    ref_label: str,
    running_rel: dict[str, np.ndarray],
    pose_idx: int,
    num_positions: int,
    samples_per_position: int,
    timeout_sec: float,
    show: bool,
    window_name: str,
    panel_height: int,
    axis_length: float,
    seed_pose: Optional[list[Optional[np.ndarray]]] = None,
) -> tuple[Optional[PositionEstimate], bool]:
    per_cam_samples: dict[str, list[np.ndarray]] = {s.label: [] for s in streams}
    if seed_pose is not None and all(p is not None for p in seed_pose):
        for stream, pose in zip(streams, seed_pose):
            per_cam_samples[stream.label].append(np.asarray(pose, dtype=np.float64).reshape(4, 4))

    t0 = time.time()
    while len(per_cam_samples[streams[0].label]) < samples_per_position:
        frames = _capture_frames(streams)
        poses, rvecs, tvecs, ok_all = _detect_frames(streams, frames, board, dictionary)
        if ok_all:
            for stream, pose in zip(streams, poses):
                per_cam_samples[stream.label].append(np.asarray(pose, dtype=np.float64).reshape(4, 4))

        sample_done = len(per_cam_samples[streams[0].label])
        if show:
            canvas = _draw_triplet_view(
                streams=streams,
                frames=frames,
                poses=poses,
                rvecs=rvecs,
                tvecs=tvecs,
                ref_label=ref_label,
                running_rel=running_rel,
                pose_idx=pose_idx,
                num_positions=num_positions,
                sample_done=sample_done,
                sample_target=samples_per_position,
                stage_text="collecting",
                panel_height=panel_height,
                axis_length=axis_length,
            )
            if canvas is not None:
                cv2.imshow(window_name, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return None, True

        if time.time() - t0 > timeout_sec:
            print(f"[WARN] Pose {pose_idx}: timeout while collecting samples ({sample_done}/{samples_per_position}).")
            return None, False

    avg_cam_from_board = {stream.label: _average_transform(per_cam_samples[stream.label]) for stream in streams}
    return (
        PositionEstimate(
            index=pose_idx,
            avg_cam_from_board=avg_cam_from_board,
            sample_count=samples_per_position,
            board_world_xyz_ref_m=avg_cam_from_board[ref_label][:3, 3].tolist(),
        ),
        False,
    )


def main() -> None:
    args = parse_args()
    if sl is None:
        raise RuntimeError("pyzed.sl is not available. Please install ZED SDK Python API.")

    config_labels, config_serial_by_label = load_collection_camera_mapping(args.collection_config)
    labels = [str(x) for x in (args.labels or config_labels or ["global", "left", "right"])]
    if len(labels) != 3 or len(set(labels)) != 3:
        raise ValueError("Please provide exactly 3 unique labels.")
    if args.reference_label not in labels:
        raise ValueError(f"reference_label must be one of {labels}")
    if int(args.num_positions) <= 0:
        raise ValueError("num_positions must be > 0.")
    if int(args.samples_per_position) <= 0:
        raise ValueError("samples_per_position must be > 0.")

    serials_cli = [] if args.serials is None else [int(x) for x in args.serials]
    discovered_serials = _discover_connected_serials()
    if serials_cli:
        if len(serials_cli) != len(labels):
            raise ValueError(f"When using --serials, provide exactly {len(labels)} serials.")
        serials = serials_cli
        serial_source = "cli"
    elif config_serial_by_label:
        missing_labels = [label for label in labels if label not in config_serial_by_label]
        if missing_labels:
            raise ValueError(
                f"Collection config {args.collection_config} is missing serials for labels: {missing_labels}"
            )
        serials = [int(config_serial_by_label[label]) for label in labels]
        serial_source = f"collection_config:{Path(args.collection_config).expanduser().resolve()}"
    else:
        if len(discovered_serials) < len(labels):
            raise RuntimeError(
                f"Auto-discovery found {len(discovered_serials)} camera(s), but {len(labels)} are required. "
                f"Found: {discovered_serials}"
            )
        serials = discovered_serials[: len(labels)]
        serial_source = "auto_discovery"
        print(f"[WARN] Auto-discovered serials: {discovered_serials}; using {serials}")
        print("[WARN] Auto-discovery binds sorted serials to labels by position. Prefer explicit --serials.")
    if len(set(serials)) != len(serials):
        raise ValueError("serials must be unique.")
    _log_label_serial_mapping(labels, serials, serial_source)

    board, dictionary, charuco_cfg = _charuco_board_from_config(args)
    show = bool(args.show)
    if show and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        print("[WARN] No GUI display detected, disabling --show.")
        show = False

    streams: list[CameraStream] = []
    image_control_by_label: dict[str, dict[str, int | bool | None]] = {}
    try:
        for label, serial in zip(labels, serials):
            stream = _open_zed(serial=serial, resolution=args.zed_resolution, fps=args.zed_fps)
            image_controls = _configure_zed_image_controls(
                camera=stream.camera,
                auto_exposure=bool(args.zed_auto_exposure),
                exposure=int(args.zed_exposure),
                gain=int(args.zed_gain),
                whitebalance_temp=args.zed_whitebalance_temp,
            )
            stream.label = label
            streams.append(stream)
            image_control_by_label[label] = image_controls
            print(f"[INFO] Opened camera label={label}, serial={serial}")

        if show:
            cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(args.window_name, int(args.window_width), int(args.window_height))

        position_estimates: list[PositionEstimate] = []
        user_abort = False
        for pose_idx in range(1, int(args.num_positions) + 1):
            print(f"\n[INFO] Pose {pose_idx}/{args.num_positions}: move board, then press Enter.")
            pose_done = False
            running_rel = _compute_rel_from_positions(position_estimates, labels, args.reference_label)
            while not pose_done:
                while True:
                    frames = _capture_frames(streams)
                    poses, rvecs, tvecs, ok_all = _detect_frames(streams, frames, board, dictionary)
                    key = -1
                    if show:
                        canvas = _draw_triplet_view(
                            streams=streams,
                            frames=frames,
                            poses=poses,
                            rvecs=rvecs,
                            tvecs=tvecs,
                            ref_label=args.reference_label,
                            running_rel=running_rel,
                            pose_idx=pose_idx,
                            num_positions=int(args.num_positions),
                            sample_done=0,
                            sample_target=int(args.samples_per_position),
                            stage_text="waiting_enter",
                            panel_height=int(args.panel_height),
                            axis_length=float(args.axis_length),
                        )
                        if canvas is not None:
                            cv2.imshow(args.window_name, canvas)
                        key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        user_abort = True
                        break
                    enter_pressed = _consume_terminal_enter() or key in (10, 13)
                    if not enter_pressed:
                        continue
                    if not ok_all:
                        print("[WARN] Enter pressed but not all 3 cameras detected the board. Reposition and retry.")
                        break
                    est, aborted = _collect_samples_for_pose(
                        streams=streams,
                        board=board,
                        dictionary=dictionary,
                        ref_label=args.reference_label,
                        running_rel=running_rel,
                        pose_idx=pose_idx,
                        num_positions=int(args.num_positions),
                        samples_per_position=int(args.samples_per_position),
                        timeout_sec=float(args.position_timeout_sec),
                        show=show,
                        window_name=args.window_name,
                        panel_height=int(args.panel_height),
                        axis_length=float(args.axis_length),
                        seed_pose=poses,
                    )
                    if aborted:
                        user_abort = True
                        break
                    if est is None:
                        break
                    position_estimates.append(est)
                    print(f"[INFO] Pose {pose_idx} accepted. board@ref = {np.asarray(est.board_world_xyz_ref_m)}")
                    pose_done = True
                    break
                if user_abort or pose_done:
                    break
            if user_abort:
                break

        if not position_estimates:
            raise RuntimeError("No pose was successfully collected.")

        rel_samples: dict[str, list[np.ndarray]] = {label: [] for label in labels}
        for pos in position_estimates:
            t_ref_from_board = pos.avg_cam_from_board[args.reference_label]
            for label in labels:
                t_board_from_cam = _invert_transform(pos.avg_cam_from_board[label])
                rel_samples[label].append(t_ref_from_board @ t_board_from_cam)

        final_rel = {
            label: (np.eye(4, dtype=np.float64) if label == args.reference_label else _average_transform(rel_samples[label]))
            for label in labels
        }
        residual_info: dict[str, dict[str, float]] = {}
        for label in labels:
            rot_err = []
            trans_err = []
            for sample in rel_samples[label]:
                r_deg, t_m = _se3_residual(sample, final_rel[label])
                rot_err.append(r_deg)
                trans_err.append(t_m)
            residual_info[label] = {
                "mean_rotation_deg": float(np.mean(rot_err)),
                "mean_translation_m": float(np.mean(trans_err)),
                "median_rotation_deg": float(np.median(rot_err)),
                "median_translation_m": float(np.median(trans_err)),
            }

        relative_to_ref: dict[str, dict[str, list[list[float]]]] = {}
        for label in labels:
            t_ref_from_cam = final_rel[label]
            relative_to_ref[label] = {
                "t_ref_from_cam": _to_float_list(t_ref_from_cam),
                "t_cam_from_ref": _to_float_list(_invert_transform(t_ref_from_cam)),
            }

        first_pose = position_estimates[0]
        output_payload: dict = {
            "type": "three_camera_charuco_extrinsics",
            "calibration_mode": "interactive_multi_pose_enter",
            "charuco_config_name": str(args.charuco_config_name),
            "charuco_config": charuco_cfg,
            "zed_resolution": str(args.zed_resolution),
            "zed_fps": int(args.zed_fps),
            "zed_image_control_request": {
                "auto_exposure": bool(args.zed_auto_exposure),
                "exposure": int(args.zed_exposure),
                "gain": int(args.zed_gain),
                "whitebalance_temp": None if args.zed_whitebalance_temp is None else int(args.zed_whitebalance_temp),
            },
            "serial_source": serial_source,
            "requested_serials": serials_cli,
            "discovered_serials": discovered_serials,
            "num_positions_target": int(args.num_positions),
            "num_positions_collected": int(len(position_estimates)),
            "samples_per_position_target": int(args.samples_per_position),
            "position_timeout_sec": float(args.position_timeout_sec),
            "reference_camera": str(args.reference_label),
            "generated_at_unix_sec": float(time.time()),
            "cameras": {},
            "relative_to_reference": relative_to_ref,
            "positions": [
                {
                    "index": int(pos.index),
                    "sample_count": int(pos.sample_count),
                    "board_world_xyz_ref_m": [float(x) for x in pos.board_world_xyz_ref_m],
                }
                for pos in position_estimates
            ],
        }
        for stream in streams:
            label = stream.label
            t_cam_from_board = first_pose.avg_cam_from_board[label]
            output_payload["cameras"][label] = {
                "serial_number": int(stream.serial),
                "camera_matrix": _to_float_list(stream.camera_matrix),
                "dist_coeffs": np.asarray(stream.dist_coeffs, dtype=np.float64).reshape(-1).tolist(),
                "image_controls": image_control_by_label.get(label, {}),
                "t_cam_from_board": _to_float_list(t_cam_from_board),
                "t_board_from_cam": _to_float_list(_invert_transform(t_cam_from_board)),
                "residual": residual_info[label],
            }

        out_path = Path(args.output_config).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(output_payload, f, sort_keys=False, allow_unicode=False)

        print(f"\n[INFO] Saved extrinsics to: {out_path}")
        print("[INFO] Relative residual summary across positions:")
        for label in labels:
            r = residual_info[label]
            print(f"  - {label}: rot_mean={r['mean_rotation_deg']:.3f} deg, trans_mean={r['mean_translation_m']:.4f} m")
    finally:
        for stream in streams:
            try:
                stream.camera.close()
            except Exception:
                pass
        if show:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


if __name__ == "__main__":
    main()
