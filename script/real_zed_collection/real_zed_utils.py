from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from script.real_zed_collection.workspace_crop_utils import invert_transform


@dataclass(frozen=True)
class CameraCalibration:
    label: str
    serial_number: int
    camera_matrix: np.ndarray
    t_world_from_cam: np.ndarray


def ensure_dir(path: str | Path) -> Path:
    out = Path(path).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return data


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _matrix(value: Any, shape: tuple[int, int], name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    return arr


def load_three_zed_calibration(path: str | Path, frame_mode: str = "reference_camera") -> dict[str, CameraCalibration]:
    """Load three-ZED YAML and return transforms into the selected world frame.

    `frame_mode="reference_camera"` returns transforms into the calibration
    reference camera. `frame_mode="workspace"` returns transforms into a
    workspace/world frame saved by the workspace-anchor script.
    """
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid calibration YAML: {path}")
    frame_mode = str(frame_mode)
    if frame_mode not in {"reference_camera", "workspace"}:
        raise ValueError("frame_mode must be one of: reference_camera, workspace")

    cameras_raw = cfg.get("cameras", {})
    rel_raw = cfg.get("relative_to_reference", {})
    if not isinstance(cameras_raw, dict) or not cameras_raw:
        raise ValueError("Calibration YAML missing non-empty 'cameras'.")

    t_workspace_from_ref: np.ndarray | None = None
    if frame_mode == "workspace":
        workspace_raw = cfg.get("workspace", cfg.get("workspace_frame", {}))
        if not isinstance(workspace_raw, dict) or not workspace_raw:
            raise ValueError(f"Calibration YAML has no workspace frame: {path}")
        if "t_workspace_from_ref" in workspace_raw:
            t_workspace_from_ref = _matrix(workspace_raw["t_workspace_from_ref"], (4, 4), "workspace.t_workspace_from_ref")
        elif "t_workspace_from_reference" in workspace_raw:
            t_workspace_from_ref = _matrix(
                workspace_raw["t_workspace_from_reference"],
                (4, 4),
                "workspace.t_workspace_from_reference",
            )
        elif "t_ref_from_workspace" in workspace_raw:
            t_workspace_from_ref = invert_transform(
                _matrix(workspace_raw["t_ref_from_workspace"], (4, 4), "workspace.t_ref_from_workspace")
            )
        else:
            raise ValueError(
                "Calibration YAML workspace frame must contain t_workspace_from_ref, "
                "t_workspace_from_reference, or t_ref_from_workspace."
            )

    out: dict[str, CameraCalibration] = {}
    for label, camera_info in cameras_raw.items():
        if not isinstance(camera_info, dict):
            continue
        rel_info = rel_raw.get(label, {}) if isinstance(rel_raw, dict) else {}
        t_world_from_cam = rel_info.get("t_ref_from_cam")
        if t_world_from_cam is None:
            if str(label) == str(cfg.get("reference_camera", "")):
                t_world_from_cam = np.eye(4)
            else:
                t_world_from_board = _matrix(
                    cameras_raw[str(cfg["reference_camera"])]["t_cam_from_board"],
                    (4, 4),
                    "t_cam_from_board",
                )
                t_board_from_cam = _matrix(camera_info["t_board_from_cam"], (4, 4), "t_board_from_cam")
                t_world_from_cam = t_world_from_board @ t_board_from_cam
        t_world_from_cam = _matrix(t_world_from_cam, (4, 4), "t_ref_from_cam")
        if t_workspace_from_ref is not None:
            t_world_from_cam = t_workspace_from_ref @ t_world_from_cam
        out[str(label)] = CameraCalibration(
            label=str(label),
            serial_number=int(camera_info.get("serial_number", 0)),
            camera_matrix=_matrix(camera_info.get("camera_matrix", np.eye(3)), (3, 3), "camera_matrix"),
            t_world_from_cam=t_world_from_cam,
        )
    return out


def depth_rgb_to_point_cloud(
    *,
    depth_m: np.ndarray,
    rgb: np.ndarray,
    camera_matrix: np.ndarray,
    mask: np.ndarray | None = None,
    min_depth_m: float = 0.05,
    max_depth_m: float = 3.0,
) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    rgb_arr = np.asarray(rgb)
    if depth.ndim != 2:
        raise ValueError(f"depth_m must be HxW, got {depth.shape}")
    if rgb_arr.shape[:2] != depth.shape:
        raise ValueError(f"rgb/depth shape mismatch: rgb={rgb_arr.shape[:2]}, depth={depth.shape}")
    valid = np.isfinite(depth) & (depth > float(min_depth_m)) & (depth < float(max_depth_m))
    if mask is not None:
        mask_arr = np.asarray(mask)
        if mask_arr.shape != depth.shape:
            raise ValueError(f"mask/depth shape mismatch: mask={mask_arr.shape}, depth={depth.shape}")
        valid &= mask_arr.astype(bool)

    ys, xs = np.where(valid)
    if xs.size == 0:
        return np.zeros((0, 6), dtype=np.float32)

    k = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])
    z = depth[ys, xs].astype(np.float64)
    x = (xs.astype(np.float64) - cx) * z / fx
    y = (ys.astype(np.float64) - cy) * z / fy
    xyz = np.stack([x, y, z], axis=1)
    colors = rgb_arr[ys, xs, :3].astype(np.float32) / 255.0
    return np.concatenate([xyz.astype(np.float32), colors.astype(np.float32)], axis=1)


def transform_point_cloud(point_cloud: np.ndarray, transform: np.ndarray) -> np.ndarray:
    pc = np.asarray(point_cloud, dtype=np.float32)
    if pc.size == 0:
        return np.zeros((0, 6), dtype=np.float32)
    if pc.ndim != 2 or pc.shape[1] < 3:
        raise ValueError(f"Expected point cloud shape NxC with C>=3, got {pc.shape}")
    tf = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    xyz_h = np.concatenate([pc[:, :3].astype(np.float64), np.ones((len(pc), 1), dtype=np.float64)], axis=1)
    xyz = (tf @ xyz_h.T).T[:, :3].astype(np.float32)
    if pc.shape[1] > 3:
        return np.concatenate([xyz, pc[:, 3:].astype(np.float32)], axis=1)
    return xyz


def merge_point_clouds(chunks: Iterable[np.ndarray]) -> np.ndarray:
    cleaned = []
    for chunk in chunks:
        pc = np.asarray(chunk, dtype=np.float32)
        if pc.ndim == 2 and pc.shape[0] > 0:
            cleaned.append(pc[:, :6] if pc.shape[1] >= 6 else pc)
    if not cleaned:
        return np.zeros((0, 6), dtype=np.float32)
    return np.concatenate(cleaned, axis=0).astype(np.float32)


def deterministic_resample(point_cloud: np.ndarray, target_num: int) -> np.ndarray:
    pc = np.asarray(point_cloud, dtype=np.float32)
    if pc.ndim != 2:
        raise ValueError(f"Expected point cloud NxC, got {pc.shape}")
    if pc.shape[1] < 6:
        pad = np.zeros((pc.shape[0], 6 - pc.shape[1]), dtype=np.float32)
        pc = np.concatenate([pc, pad], axis=1)
    pc = pc[:, :6]
    target_num = int(target_num)
    if target_num <= 0:
        return pc.astype(np.float32)
    if len(pc) == 0:
        return np.zeros((target_num, 6), dtype=np.float32)
    if len(pc) == target_num:
        return pc.astype(np.float32)
    if len(pc) > target_num:
        idx = np.linspace(0, len(pc) - 1, target_num).round().astype(np.int64)
        return pc[idx].astype(np.float32)
    reps = target_num - len(pc)
    idx = np.arange(reps, dtype=np.int64) % len(pc)
    return np.concatenate([pc, pc[idx]], axis=0).astype(np.float32)
