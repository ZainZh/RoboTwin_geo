#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - import availability depends on runtime image stack.
    cv2 = None

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from script.real_zed_collection.export_raw_rgb_images import parse_csv, resolve_raw_episode_dir
from script.real_zed_collection.real_zed_utils import (
    calibration_label_map_from_manifest,
    deterministic_resample,
    ensure_dir,
    load_three_zed_calibration,
    merge_point_clouds,
    read_json,
    transform_point_cloud,
    write_json,
)


def _load_frame_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _rgb_to_depth_shape(rgb: np.ndarray, depth_shape: tuple[int, int]) -> np.ndarray:
    rgb_arr = np.asarray(rgb, dtype=np.uint8)
    if rgb_arr.shape[:2] == depth_shape:
        return rgb_arr
    if cv2 is None:
        raise RuntimeError("cv2 is required when rgb/depth shapes differ.")
    return cv2.resize(rgb_arr, (int(depth_shape[1]), int(depth_shape[0])), interpolation=cv2.INTER_LINEAR)


def _frame_to_point_cloud(
    *,
    camera_frame: dict[str, np.ndarray],
    camera_matrix: np.ndarray,
    rgb_key: str,
    depth_key: str,
    min_depth_m: float,
    max_depth_m: float,
) -> np.ndarray:
    if depth_key in camera_frame:
        depth_m = np.asarray(camera_frame[depth_key], dtype=np.float32)
    elif depth_key == "depth_m" and "depth_mm" in camera_frame:
        depth_m = np.asarray(camera_frame["depth_mm"], dtype=np.float32) / 1000.0
    else:
        raise KeyError(f"Camera frame missing depth key {depth_key!r}. Available keys: {sorted(camera_frame)}")
    if rgb_key not in camera_frame:
        raise KeyError(f"Camera frame missing RGB key {rgb_key!r}. Available keys: {sorted(camera_frame)}")
    rgb = _rgb_to_depth_shape(np.asarray(camera_frame[rgb_key], dtype=np.uint8), depth_m.shape)
    k = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)

    valid = np.isfinite(depth_m) & (depth_m > float(min_depth_m)) & (depth_m < float(max_depth_m))
    ys, xs = np.where(valid)
    if xs.size == 0:
        return np.zeros((0, 6), dtype=np.float32)

    z = depth_m[ys, xs].astype(np.float64)
    x = (xs.astype(np.float64) - float(k[0, 2])) * z / float(k[0, 0])
    y = (ys.astype(np.float64) - float(k[1, 2])) * z / float(k[1, 1])
    xyz = np.stack([x, y, z], axis=1).astype(np.float32)
    colors = rgb[ys, xs, :3].astype(np.float32) / 255.0
    return np.concatenate([xyz, colors], axis=1).astype(np.float32)


def _resolve_calibration_path(raw_episode_dir: Path, manifest: dict[str, Any], override: str, frame_mode: str) -> Path:
    if str(override).strip():
        path = Path(override).expanduser().resolve()
    else:
        if str(frame_mode) == "workspace":
            snapshot_rel = str(
                manifest.get("workspace_calibration_snapshot_path", "")
                or manifest.get("calibration_snapshot_path", "")
            ).strip()
            source_path = str(
                manifest.get("workspace_calibration_path", "")
                or manifest.get("calibration_path", "")
            ).strip()
        else:
            snapshot_rel = str(manifest.get("calibration_snapshot_path", "")).strip()
            source_path = str(manifest.get("calibration_path", "")).strip()
        if snapshot_rel:
            path = (raw_episode_dir / snapshot_rel).resolve()
        elif source_path:
            path = Path(source_path).expanduser().resolve()
        else:
            raise ValueError("Missing calibration path. Pass --calibration_path or include it in manifest.json.")
    if not path.exists():
        raise FileNotFoundError(f"Calibration file does not exist: {path}")
    return path


def _resolve_camera_labels(manifest: dict[str, Any], requested_labels: Sequence[str] | None, calib_labels: Sequence[str]) -> list[str]:
    labels = [str(label) for label in (requested_labels or []) if str(label)]
    if labels:
        return labels
    manifest_labels = manifest.get("camera_labels", [])
    if isinstance(manifest_labels, list) and manifest_labels:
        return [str(label) for label in manifest_labels]
    return [str(label) for label in calib_labels]


def _apply_xyz_crop(
    point_cloud: np.ndarray,
    *,
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    z_min: float | None = None,
    z_max: float | None = None,
) -> np.ndarray:
    pc = np.asarray(point_cloud, dtype=np.float32)
    if pc.size == 0:
        return np.zeros((0, 6), dtype=np.float32)
    mask = np.ones((len(pc),), dtype=bool)
    if x_min is not None:
        mask &= pc[:, 0] >= float(x_min)
    if x_max is not None:
        mask &= pc[:, 0] <= float(x_max)
    if y_min is not None:
        mask &= pc[:, 1] >= float(y_min)
    if y_max is not None:
        mask &= pc[:, 1] <= float(y_max)
    if z_min is not None:
        mask &= pc[:, 2] >= float(z_min)
    if z_max is not None:
        mask &= pc[:, 2] <= float(z_max)
    return pc[mask]


def _resample_for_export(point_cloud: np.ndarray, point_num: int) -> np.ndarray:
    pc = np.asarray(point_cloud, dtype=np.float32)
    return deterministic_resample(pc, int(point_num)) if int(point_num) > 0 else pc


def write_colored_ply(path: str | Path, point_cloud: np.ndarray) -> Path:
    out_path = Path(path).expanduser()
    pc = np.asarray(point_cloud, dtype=np.float32)
    if pc.ndim != 2 or pc.shape[1] < 6:
        raise ValueError(f"Expected Nx6 point cloud, got {pc.shape}")

    xyz = pc[:, :3].astype(np.float32)
    rgb = np.clip(np.round(pc[:, 3:6] * 255.0), 0, 255).astype(np.uint8)
    vertex = np.empty(
        len(pc),
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertex["x"] = xyz[:, 0]
    vertex["y"] = xyz[:, 1]
    vertex["z"] = xyz[:, 2]
    vertex["red"] = rgb[:, 0]
    vertex["green"] = rgb[:, 1]
    vertex["blue"] = rgb[:, 2]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertex)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(header)
        f.write(vertex.tobytes())
    return out_path


def export_raw_colored_pointclouds(
    *,
    raw_episode_dir: str | Path,
    output_dir: str | Path,
    camera_labels: Sequence[str] | None = None,
    frame_stride: int = 1,
    max_frames: int | None = None,
    export_mode: str = "fused",
    calibration_path: str = "",
    frame_mode: str = "reference_camera",
    intrinsics_source: str = "frame",
    disable_serial_remap: bool = False,
    rgb_key: str = "rgb",
    depth_key: str = "depth_m",
    min_depth_m: float = 0.05,
    max_depth_m: float = 3.0,
    point_num: int = 800000,
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    z_min: float | None = None,
    z_max: float | None = None,
) -> dict[str, Any]:
    raw_episode = Path(raw_episode_dir).expanduser().resolve()
    manifest = read_json(raw_episode / "manifest.json")
    frames = manifest.get("frames", [])
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"No frames found in {raw_episode / 'manifest.json'}")
    mode = str(export_mode)
    if mode not in {"fused", "per_camera", "both"}:
        raise ValueError("export_mode must be one of: fused, per_camera, both")
    calibration = _resolve_calibration_path(raw_episode, manifest, calibration_path, frame_mode=frame_mode)
    calib = load_three_zed_calibration(calibration, frame_mode=frame_mode)
    labels = _resolve_camera_labels(manifest, camera_labels, list(calib.keys()))
    label_to_calib = (
        {label: label for label in labels}
        if bool(disable_serial_remap)
        else calibration_label_map_from_manifest(manifest, calib, labels)
    )
    stride = max(1, int(frame_stride))
    limit = None if max_frames is None or int(max_frames) <= 0 else int(max_frames)
    selected_frames = frames[::stride]
    if limit is not None:
        selected_frames = selected_frames[:limit]
    out_root = ensure_dir(output_dir)

    saved_count = 0
    saved_by_mode = {"fused": 0, "per_camera": 0}
    frame_summaries: list[dict[str, Any]] = []
    for local_idx, frame in enumerate(selected_frames):
        frame_index = int(frame.get("frame_index", local_idx * stride))
        cameras = frame.get("cameras", {})
        if not isinstance(cameras, dict):
            raise ValueError(f"Invalid cameras record for frame {frame_index}: {cameras!r}")
        chunks = []
        per_camera_counts = {}
        for label in labels:
            if label not in cameras:
                raise KeyError(f"Frame {frame_index} has no camera {label!r}. Available: {sorted(cameras)}")
            calib_label = label_to_calib[label]
            camera_frame = _load_frame_npz(raw_episode / str(cameras[label]))
            frame_camera_matrix = (
                np.asarray(camera_frame["camera_matrix"], dtype=np.float32).reshape(3, 3)
                if "camera_matrix" in camera_frame
                else calib[calib_label].camera_matrix.astype(np.float32)
            )
            camera_matrix = frame_camera_matrix if str(intrinsics_source) == "frame" else calib[calib_label].camera_matrix
            pc_camera = _frame_to_point_cloud(
                camera_frame=camera_frame,
                camera_matrix=camera_matrix,
                rgb_key=rgb_key,
                depth_key=depth_key,
                min_depth_m=float(min_depth_m),
                max_depth_m=float(max_depth_m),
            )
            pc_world = transform_point_cloud(pc_camera, calib[calib_label].t_world_from_cam)
            pc_world = _apply_xyz_crop(
                pc_world,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                z_min=z_min,
                z_max=z_max,
            )
            chunks.append(pc_world)
            per_camera_counts[label] = int(len(pc_world))
            if mode in {"per_camera", "both"}:
                exported = _resample_for_export(pc_world, int(point_num))
                write_colored_ply(out_root / label / f"frame_{frame_index:06d}.ply", exported)
                saved_count += 1
                saved_by_mode["per_camera"] += 1
        merged = merge_point_clouds(chunks)
        if mode in {"fused", "both"}:
            exported = _resample_for_export(merged, int(point_num))
            write_colored_ply(out_root / "fused" / f"frame_{frame_index:06d}.ply", exported)
            saved_count += 1
            saved_by_mode["fused"] += 1
        frame_summaries.append(
            {
                "frame_index": frame_index,
                "per_camera_points": per_camera_counts,
                "merged_points": int(len(merged)),
            }
        )

    summary = {
        "raw_episode_dir": str(raw_episode),
        "output_dir": str(out_root),
        "camera_labels": labels,
        "export_mode": mode,
        "calibration_path": str(calibration),
        "frame_mode": str(frame_mode),
        "intrinsics_source": str(intrinsics_source),
        "rgb_key": str(rgb_key),
        "depth_key": str(depth_key),
        "frame_stride": stride,
        "max_frames": limit,
        "point_num": int(point_num),
        "saved_count": saved_count,
        "saved_by_mode": saved_by_mode,
        "frames": frame_summaries,
    }
    write_json(out_root / "export_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export colored point clouds from one raw real-ZED task episode.")
    parser.add_argument("--task_name", default="view_pour_water_new", help="Task/project name, e.g. grasp_mug.")
    parser.add_argument("--episode", default="4", help="Episode index, episode directory name, timestamp, or full path.")
    parser.add_argument("--raw_root", default="", help="Raw root. Defaults to /media/$USER/Extreme SSD/geo_mani_data/<task>/real_zed_raw.")
    parser.add_argument("--output_dir", default="", help="Output directory. Defaults under outputs/real_zed_collection/raw_colored_pointclouds/.")
    parser.add_argument("--camera_labels", default="", help="Comma-separated camera labels. Empty means manifest camera_labels.")
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=0, help="0 means export all selected frames.")
    parser.add_argument("--export_mode", default="fused", choices=["fused", "per_camera", "both"])
    parser.add_argument("--calibration_path", default="")
    parser.add_argument("--frame_mode", default="reference_camera", choices=["reference_camera", "workspace"])
    parser.add_argument("--intrinsics_source", default="frame", choices=["frame", "calibration"])
    parser.add_argument("--disable_serial_remap", action="store_true", default=False)
    parser.add_argument("--rgb_key", default="rgb", choices=["rgb", "full_rgb_debug"])
    parser.add_argument("--depth_key", default="depth_m", choices=["depth_m", "full_depth_m_debug"])
    parser.add_argument("--min_depth_m", type=float, default=0.05)
    parser.add_argument("--max_depth_m", type=float, default=3.0)
    parser.add_argument("--point_num", type=int, default=800000, help="<=0 keeps all points.")
    parser.add_argument("--x_min", type=float, default=None)
    parser.add_argument("--x_max", type=float, default=None)
    parser.add_argument("--y_min", type=float, default=None)
    parser.add_argument("--y_max", type=float, default=None)
    parser.add_argument("--z_min", type=float, default=None)
    parser.add_argument("--z_max", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_episode_dir = resolve_raw_episode_dir(args.task_name, args.episode, raw_root=args.raw_root or None)
    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else Path("outputs") / "real_zed_collection" / "raw_colored_pointclouds" / args.task_name / args.episode
    )
    summary = export_raw_colored_pointclouds(
        raw_episode_dir=raw_episode_dir,
        output_dir=output_dir,
        camera_labels=parse_csv(args.camera_labels),
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        export_mode=args.export_mode,
        calibration_path=args.calibration_path,
        frame_mode=args.frame_mode,
        intrinsics_source=args.intrinsics_source,
        disable_serial_remap=args.disable_serial_remap,
        rgb_key=args.rgb_key,
        depth_key=args.depth_key,
        min_depth_m=args.min_depth_m,
        max_depth_m=args.max_depth_m,
        point_num=args.point_num,
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
        z_min=args.z_min,
        z_max=args.z_max,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
