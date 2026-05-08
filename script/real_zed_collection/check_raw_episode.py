#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from script.real_zed_collection.real_zed_utils import load_three_zed_calibration, read_json

try:
    import cv2
except Exception:
    cv2 = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect one raw real_zed_collection episode and report depth/point-cloud health.")
    parser.add_argument("--raw_episode_dir", required=True, type=str)
    parser.add_argument("--calibration_path", default="", type=str)
    parser.add_argument("--min_depth_m", default=0.05, type=float)
    parser.add_argument("--max_depth_m", default=3.0, type=float)
    parser.add_argument("--warn_valid_ratio_below", default=0.70, type=float)
    parser.add_argument("--warn_points_below", default=100000, type=int)
    parser.add_argument("--print_every_frame", action="store_true", default=False)
    return parser.parse_args()


def _load_frame_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _resize_rgb_to_depth(rgb: np.ndarray, depth_shape: tuple[int, int]) -> np.ndarray:
    if rgb.shape[:2] == depth_shape:
        return rgb
    if cv2 is None:
        raise RuntimeError("cv2 is required when rgb/depth shapes differ.")
    return cv2.resize(rgb, (depth_shape[1], depth_shape[0]), interpolation=cv2.INTER_LINEAR)


def _count_valid_depth(depth_m: np.ndarray, min_depth_m: float, max_depth_m: float) -> tuple[int, float]:
    valid = np.isfinite(depth_m) & (depth_m > float(min_depth_m)) & (depth_m < float(max_depth_m))
    return int(valid.sum()), float(valid.mean())


def _resolve_calibration_path(raw_episode_dir: Path, manifest: dict, override: str) -> Path:
    if str(override).strip():
        calib_path = Path(override).expanduser().resolve()
    else:
        snapshot_rel = str(manifest.get("calibration_snapshot_path", "")).strip()
        if snapshot_rel:
            calib_path = (raw_episode_dir / snapshot_rel).resolve()
        else:
            source_path = str(manifest.get("calibration_path", "")).strip()
            if not source_path:
                raise ValueError("Missing calibration path. Pass --calibration_path or include it in manifest.json.")
            calib_path = Path(source_path).expanduser().resolve()
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration file does not exist: {calib_path}")
    return calib_path


def main() -> None:
    args = parse_args()
    raw_episode_dir = Path(args.raw_episode_dir).expanduser().resolve()
    manifest_path = raw_episode_dir / "manifest.json"
    manifest = read_json(manifest_path)
    frames = manifest.get("frames", [])
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"No frames found in {manifest_path}")

    calibration_path = _resolve_calibration_path(raw_episode_dir, manifest, args.calibration_path)
    calib = load_three_zed_calibration(calibration_path)

    labels = [str(x) for x in manifest.get("camera_labels", list(calib.keys()))]
    missing = [label for label in labels if label not in calib]
    if missing:
        raise ValueError(f"Calibration missing camera labels: {missing}")

    print(f"[INFO] Episode: {raw_episode_dir}")
    print(f"[INFO] Manifest: {manifest_path}")
    print(f"[INFO] Calibration: {calibration_path}")
    print(f"[INFO] Frames: {len(frames)}")
    print(f"[INFO] Cameras: {labels}")
    print(f"[INFO] Camera serials: {manifest.get('camera_serials', {})}")
    print(f"[INFO] Saved RGB size: {manifest.get('save_rgb_width', 'n/a')}x{manifest.get('save_rgb_height', 'n/a')}")
    print(f"[INFO] Save frequency: {manifest.get('save_frequency_hz', 'n/a')} Hz")
    print(f"[INFO] Dropped frames: {manifest.get('dropped_frames_due_to_backpressure', 'n/a')}")

    valid_ratio_by_label: dict[str, list[float]] = {label: [] for label in labels}
    point_count_by_label: dict[str, list[int]] = {label: [] for label in labels}
    merged_point_counts: list[int] = []
    robot_joint_norms: list[float] = []
    robot_velocity_norms: list[float] = []
    warnings: list[str] = []

    for frame in frames:
        frame_index = int(frame.get("frame_index", len(merged_point_counts)))
        robot = _load_frame_npz(raw_episode_dir / str(frame["robot"]))
        joint_vector = np.asarray(robot.get("joint_vector", robot.get("joint_positions")), dtype=np.float32).reshape(-1)
        joint_vel = np.asarray(robot.get("joint_velocities", np.zeros_like(joint_vector)), dtype=np.float32).reshape(-1)
        robot_joint_norms.append(float(np.linalg.norm(joint_vector)))
        robot_velocity_norms.append(float(np.linalg.norm(joint_vel)))

        per_frame_counts: dict[str, int] = {}
        per_frame_ratios: dict[str, float] = {}
        for label in labels:
            camera_rel = str(frame["cameras"][label])
            camera_frame = _load_frame_npz(raw_episode_dir / camera_rel)
            depth_m = (
                np.asarray(camera_frame["depth_m"], dtype=np.float32)
                if "depth_m" in camera_frame
                else np.asarray(camera_frame["depth_mm"], dtype=np.float32) / 1000.0
            )
            rgb = np.asarray(camera_frame["rgb"], dtype=np.uint8)
            _ = _resize_rgb_to_depth(rgb, depth_m.shape)
            valid_count, valid_ratio = _count_valid_depth(depth_m, args.min_depth_m, args.max_depth_m)
            valid_ratio_by_label[label].append(valid_ratio)
            point_count_by_label[label].append(valid_count)
            per_frame_counts[label] = valid_count
            per_frame_ratios[label] = valid_ratio

            if valid_ratio < float(args.warn_valid_ratio_below):
                warnings.append(
                    f"frame {frame_index} camera {label}: valid depth ratio {valid_ratio:.3f} < {float(args.warn_valid_ratio_below):.3f}"
                )
            if valid_count < int(args.warn_points_below):
                warnings.append(
                    f"frame {frame_index} camera {label}: valid point count {valid_count} < {int(args.warn_points_below)}"
                )

        merged_count = int(sum(per_frame_counts.values()))
        merged_point_counts.append(merged_count)
        if args.print_every_frame:
            print(
                f"frame {frame_index:03d} | "
                + " | ".join(
                    f"{label}: ratio={per_frame_ratios[label]:.3f}, points={per_frame_counts[label]}"
                    for label in labels
                )
                + f" | merged={merged_count}"
            )

    print("\n[SUMMARY] Per-camera depth health")
    for label in labels:
        ratios = np.asarray(valid_ratio_by_label[label], dtype=np.float64)
        counts = np.asarray(point_count_by_label[label], dtype=np.int64)
        print(
            f"  - {label}: valid_ratio min/mean/max = {ratios.min():.3f}/{ratios.mean():.3f}/{ratios.max():.3f}; "
            f"valid_points min/mean/max = {counts.min()}/{int(round(counts.mean()))}/{counts.max()}"
        )

    merged_arr = np.asarray(merged_point_counts, dtype=np.int64)
    joint_norm_arr = np.asarray(robot_joint_norms, dtype=np.float64)
    vel_norm_arr = np.asarray(robot_velocity_norms, dtype=np.float64)
    print("\n[SUMMARY] Episode-level")
    print(
        f"  - merged_points min/mean/max = {merged_arr.min()}/{int(round(merged_arr.mean()))}/{merged_arr.max()}"
    )
    print(
        f"  - robot_joint_norm min/mean/max = {joint_norm_arr.min():.3f}/{joint_norm_arr.mean():.3f}/{joint_norm_arr.max():.3f}"
    )
    print(
        f"  - joint_velocity_norm min/mean/max = {vel_norm_arr.min():.3f}/{vel_norm_arr.mean():.3f}/{vel_norm_arr.max():.3f}"
    )

    if warnings:
        print("\n[WARNINGS]")
        for item in warnings[:50]:
            print(f"  - {item}")
        if len(warnings) > 50:
            print(f"  - ... {len(warnings) - 50} more")
    else:
        print("\n[WARNINGS]\n  - none")


if __name__ == "__main__":
    main()
