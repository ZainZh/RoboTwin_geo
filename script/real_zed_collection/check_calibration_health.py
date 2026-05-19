#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from script.real_zed_collection.calibrate_three_zed_extrinsics import (
    _average_transform,
    _capture_frames,
    _charuco_board_from_config,
    _configure_zed_image_controls,
    _detect_frames,
    _draw_triplet_view,
    _invert_transform,
    _open_zed,
    _se3_residual,
    _to_float_list,
    sl,
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_calibration = repo_root / "script" / "real_zed_collection" / "calibration" / "three_camera_charuco_extrinsics.yaml"
    default_output = repo_root / "script" / "real_zed_collection" / "calibration" / "three_camera_workspace_extrinsics.yaml"
    default_geometry_repo = Path.home() / "github" / "geometry_awareness_manipulation"

    parser = argparse.ArgumentParser(
        description=(
            "Define a stable workspace/world frame from a deliberately placed Charuco board, "
            "and check whether the existing three-ZED calibration still agrees across cameras."
        )
    )
    parser.add_argument("--calibration_path", type=str, default=str(default_calibration))
    parser.add_argument("--output_config", type=str, default=str(default_output))
    parser.add_argument("--labels", type=str, nargs="*", default=None)
    parser.add_argument("--serials", type=int, nargs="*", default=None)
    parser.add_argument("--reference_label", type=str, default="")
    parser.add_argument("--charuco_config_name", type=str, default="charuco_300_9x14_20mm_15mm")
    # parser.add_argument("--charuco_config_name", type=str, default="charuco_A4")
    parser.add_argument("--charuco_config_path", type=str, default="")
    parser.add_argument("--geometry_repo", type=str, default=str(default_geometry_repo))
    parser.add_argument("--zed_resolution", type=str, default="HD1080")
    parser.add_argument("--zed_fps", type=int, default=15)
    parser.add_argument("--zed_auto_exposure", action="store_true", default=False)
    parser.add_argument("--zed_exposure", type=int, default=22)
    parser.add_argument("--zed_gain", type=int, default=12)
    parser.add_argument("--zed_whitebalance_temp", type=int, default=None)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--timeout_sec", type=float, default=30.0)
    parser.add_argument("--show", action="store_true", default=True)
    parser.add_argument("--no-show", dest="show", action="store_false")
    parser.add_argument("--window_name", type=str, default="three_zed_workspace_anchor")
    parser.add_argument("--panel_height", type=int, default=800)
    parser.add_argument("--axis_length", type=float, default=0.035)
    parser.add_argument(
        "--workspace_bbox",
        type=float,
        nargs=6,
        default=[-0.35, 0.35, -0.25, 0.45, 0.0, 0.45],
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX", "Z_MIN", "Z_MAX"),
        help="Default physical crop box in the new workspace frame, in meters.",
    )
    parser.add_argument("--max_translation_residual_m", type=float, default=0.02)
    parser.add_argument("--max_rotation_residual_deg", type=float, default=3.0)
    parser.add_argument(
        "--flip_workspace_z",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Default true because Charuco board +Z often points into the table. "
            "When enabled, workspace +Z is board -Z while preserving a right-handed frame."
        ),
    )
    parser.add_argument("--fail_on_bad_check", action="store_true", default=False)
    return parser.parse_args()


def _load_base_calibration(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Calibration YAML must contain a mapping: {path}")
    if "cameras" not in data or "relative_to_reference" not in data:
        raise ValueError(f"Calibration YAML missing cameras/relative_to_reference: {path}")
    return data


def _resolve_labels_serials(args: argparse.Namespace, base_cfg: dict) -> tuple[list[str], list[int], str]:
    cameras = base_cfg.get("cameras", {})
    if not isinstance(cameras, dict) or not cameras:
        raise ValueError("Base calibration has no cameras.")

    labels = [str(x) for x in args.labels] if args.labels else [str(x) for x in cameras.keys()]
    # if len(labels) != 3:
    #     raise ValueError(f"Expected exactly 3 labels, got {labels}")
    missing = [label for label in labels if label not in cameras]
    if missing:
        raise ValueError(f"Labels missing from base calibration: {missing}")

    if args.serials:
        serials = [int(x) for x in args.serials]
    else:
        serials = [int(cameras[label].get("serial_number", 0)) for label in labels]
    if len(serials) != len(labels) or any(x <= 0 for x in serials):
        raise ValueError("Provide three valid serials or use a calibration file with serial_number entries.")

    reference_label = str(args.reference_label).strip() or str(base_cfg.get("reference_camera", labels[0]))
    if reference_label not in labels:
        raise ValueError(f"reference_label must be one of {labels}, got {reference_label!r}")
    return labels, serials, reference_label


def _as_transform(raw, name: str) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float64)
    if arr.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4, got {arr.shape}")
    return arr


def _ref_from_cam_transforms(base_cfg: dict, labels: list[str]) -> dict[str, np.ndarray]:
    rel = base_cfg.get("relative_to_reference", {})
    if not isinstance(rel, dict):
        raise ValueError("Base calibration relative_to_reference must be a mapping.")
    out = {}
    for label in labels:
        if label not in rel or "t_ref_from_cam" not in rel[label]:
            raise ValueError(f"Missing relative_to_reference.{label}.t_ref_from_cam")
        out[label] = _as_transform(rel[label]["t_ref_from_cam"], f"relative_to_reference.{label}.t_ref_from_cam")
    return out


def _workspace_from_board_transform(flip_z: bool = True) -> np.ndarray:
    """Return T_board_from_workspace for the desired workspace axis convention."""
    transform = np.eye(4, dtype=np.float64)
    if flip_z:
        transform[:3, :3] = np.diag([1.0, -1.0, -1.0])
    return transform


def _collect_anchor_samples(
    *,
    streams,
    board,
    dictionary,
    ref_from_cam: dict[str, np.ndarray],
    reference_label: str,
    sample_count: int,
    timeout_sec: float,
    show: bool,
    window_name: str,
    panel_height: int,
    axis_length: float,
) -> dict[str, list[np.ndarray]]:
    samples: dict[str, list[np.ndarray]] = {stream.label: [] for stream in streams}
    running_rel = {stream.label: ref_from_cam[stream.label] for stream in streams}
    t0 = time.time()
    while len(samples[reference_label]) < int(sample_count):
        frames = _capture_frames(streams)
        poses, rvecs, tvecs, ok_all = _detect_frames(streams, frames, board, dictionary)
        if ok_all:
            for stream, t_cam_from_board in zip(streams, poses):
                if t_cam_from_board is None:
                    continue
                t_ref_from_board = ref_from_cam[stream.label] @ np.asarray(t_cam_from_board, dtype=np.float64).reshape(4, 4)
                samples[stream.label].append(t_ref_from_board)
            print(f"[INFO] accepted sample {len(samples[reference_label])}/{sample_count}", end="\r")

        if show:
            canvas = _draw_triplet_view(
                streams=streams,
                frames=frames,
                poses=poses,
                rvecs=rvecs,
                tvecs=tvecs,
                ref_label=reference_label,
                running_rel=running_rel,
                pose_idx=1,
                num_positions=1,
                sample_done=len(samples[reference_label]),
                sample_target=int(sample_count),
                stage_text="workspace anchor",
                panel_height=int(panel_height),
                axis_length=float(axis_length),
            )
            if canvas is not None:
                cv2.imshow(window_name, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                raise KeyboardInterrupt("User aborted workspace anchor capture.")

        if time.time() - t0 > float(timeout_sec):
            raise TimeoutError(
                f"Timed out while collecting workspace anchor samples: "
                f"{len(samples[reference_label])}/{sample_count}"
            )
    print()
    return samples


def _health_from_samples(samples: dict[str, list[np.ndarray]], anchor_ref_from_workspace: np.ndarray) -> tuple[dict, bool]:
    per_camera = {}
    ok = True
    for label, transforms in samples.items():
        rot_err = []
        trans_err = []
        for t_ref_from_workspace in transforms:
            r_deg, t_m = _se3_residual(t_ref_from_workspace, anchor_ref_from_workspace)
            rot_err.append(float(r_deg))
            trans_err.append(float(t_m))
        per_camera[label] = {
            "sample_count": int(len(transforms)),
            "mean_rotation_deg": float(np.mean(rot_err)),
            "max_rotation_deg": float(np.max(rot_err)),
            "mean_translation_m": float(np.mean(trans_err)),
            "max_translation_m": float(np.max(trans_err)),
        }
    return per_camera, ok


def main() -> None:
    args = parse_args()
    if sl is None:
        raise RuntimeError("pyzed.sl is not available. Please install the ZED SDK Python API.")
    if args.show and not os.environ.get("DISPLAY"):
        print("[WARN] No GUI display detected, disabling --show.")
        args.show = False
    if int(args.samples) <= 0:
        raise ValueError("--samples must be positive.")

    base_path = Path(args.calibration_path).expanduser().resolve()
    base_cfg = _load_base_calibration(base_path)
    labels, serials, reference_label = _resolve_labels_serials(args, base_cfg)
    ref_from_cam = _ref_from_cam_transforms(base_cfg, labels)
    board, dictionary, charuco_cfg = _charuco_board_from_config(args)

    streams = []
    try:
        for label, serial in zip(labels, serials):
            stream = _open_zed(serial, args.zed_resolution, int(args.zed_fps))
            stream.label = label
            _configure_zed_image_controls(
                stream.camera,
                auto_exposure=bool(args.zed_auto_exposure),
                exposure=int(args.zed_exposure),
                gain=int(args.zed_gain),
                whitebalance_temp=args.zed_whitebalance_temp,
            )
            streams.append(stream)
        time.sleep(1.0)

        samples = _collect_anchor_samples(
            streams=streams,
            board=board,
            dictionary=dictionary,
            ref_from_cam=ref_from_cam,
            reference_label=reference_label,
            sample_count=int(args.samples),
            timeout_sec=float(args.timeout_sec),
            show=bool(args.show),
            window_name=str(args.window_name),
            panel_height=int(args.panel_height),
            axis_length=float(args.axis_length),
        )
        t_board_from_workspace = _workspace_from_board_transform(flip_z=bool(args.flip_workspace_z))
        all_ref_from_board = [t for transforms in samples.values() for t in transforms]
        t_ref_from_board = _average_transform(all_ref_from_board)
        t_ref_from_workspace = t_ref_from_board @ t_board_from_workspace
        t_workspace_from_ref = _invert_transform(t_ref_from_workspace)
        per_camera, _ = _health_from_samples(samples, t_ref_from_board)
        max_rot = max(v["max_rotation_deg"] for v in per_camera.values())
        max_trans = max(v["max_translation_m"] for v in per_camera.values())
        health_ok = max_rot <= float(args.max_rotation_residual_deg) and max_trans <= float(args.max_translation_residual_m)

        rel_to_workspace = {}
        for label in labels:
            t_workspace_from_cam = t_workspace_from_ref @ ref_from_cam[label]
            rel_to_workspace[label] = {
                "t_workspace_from_cam": _to_float_list(t_workspace_from_cam),
                "t_cam_from_workspace": _to_float_list(_invert_transform(t_workspace_from_cam)),
            }

        output_payload = dict(base_cfg)
        output_payload["workspace"] = {
            "type": "charuco_workspace_anchor",
            "source_calibration_path": str(base_path),
            "reference_camera": str(reference_label),
            "generated_at_unix_sec": float(time.time()),
            "charuco_config": charuco_cfg,
            "axis_convention": {
                "source_frame": "charuco_board",
                "flip_workspace_z": bool(args.flip_workspace_z),
                "t_board_from_workspace": _to_float_list(t_board_from_workspace),
                "description": (
                    "workspace +Z is board -Z; x is preserved and y is flipped to keep a right-handed frame"
                    if bool(args.flip_workspace_z)
                    else "workspace axes match the Charuco board axes"
                ),
            },
            "t_ref_from_board": _to_float_list(t_ref_from_board),
            "t_ref_from_workspace": _to_float_list(t_ref_from_workspace),
            "t_workspace_from_ref": _to_float_list(t_workspace_from_ref),
            "bbox_m": {
                "x_min": float(args.workspace_bbox[0]),
                "x_max": float(args.workspace_bbox[1]),
                "y_min": float(args.workspace_bbox[2]),
                "y_max": float(args.workspace_bbox[3]),
                "z_min": float(args.workspace_bbox[4]),
                "z_max": float(args.workspace_bbox[5]),
            },
            "health_check": {
                "ok": bool(health_ok),
                "max_rotation_residual_deg": float(max_rot),
                "max_translation_residual_m": float(max_trans),
                "threshold_rotation_deg": float(args.max_rotation_residual_deg),
                "threshold_translation_m": float(args.max_translation_residual_m),
                "per_camera": per_camera,
            },
        }
        output_payload["relative_to_workspace"] = rel_to_workspace

        # out_path = Path(args.output_config).expanduser().resolve()
        # out_path.parent.mkdir(parents=True, exist_ok=True)
        # with out_path.open("w", encoding="utf-8") as f:
        #     yaml.safe_dump(output_payload, f, sort_keys=False, allow_unicode=False)

        status = "OK" if health_ok else "WARN"
        # print(f"[{status}] workspace anchor saved: {out_path}")
        print(f"[{status}] max residual: rotation={max_rot:.3f} deg, translation={max_trans:.4f} m")
        if args.fail_on_bad_check and not health_ok:
            raise SystemExit(2)
    finally:
        for stream in streams:
            try:
                stream.camera.close()
            except Exception:
                pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()
