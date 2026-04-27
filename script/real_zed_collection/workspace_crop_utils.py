from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None


@dataclass(frozen=True)
class WorkspaceBounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def as_dict(self) -> dict[str, float]:
        return {
            "x_min": float(self.x_min),
            "x_max": float(self.x_max),
            "y_min": float(self.y_min),
            "y_max": float(self.y_max),
            "z_min": float(self.z_min),
            "z_max": float(self.z_max),
        }


def invert_transform(transform: np.ndarray) -> np.ndarray:
    t = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = t[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ t[:3, 3]
    return out


def workspace_bbox_corners(bounds: WorkspaceBounds) -> np.ndarray:
    xs = [float(bounds.x_min), float(bounds.x_max)]
    ys = [float(bounds.y_min), float(bounds.y_max)]
    zs = [float(bounds.z_min), float(bounds.z_max)]
    return np.asarray([[x, y, z] for x in xs for y in ys for z in zs], dtype=np.float64)


def _clip_roi(x0: int, y0: int, x1: int, y1: int, width: int, height: int) -> tuple[int, int, int, int]:
    x0 = max(0, min(int(x0), int(width)))
    x1 = max(0, min(int(x1), int(width)))
    y0 = max(0, min(int(y0), int(height)))
    y1 = max(0, min(int(y1), int(height)))
    if x1 <= x0 or y1 <= y0:
        return 0, 0, int(width), int(height)
    return x0, y0, x1, y1


def project_workspace_bbox_to_roi(
    *,
    camera_matrix: np.ndarray,
    image_shape_hw: tuple[int, int],
    t_workspace_from_cam: np.ndarray,
    bounds: WorkspaceBounds,
    margin_px: int = 0,
    min_positive_depth_m: float = 1e-4,
) -> tuple[int, int, int, int]:
    """Project a workspace 3D crop box into one camera and return xyxy pixel ROI.

    The returned box is exclusive on x1/y1 so it can be used directly for NumPy slicing.
    """
    height, width = int(image_shape_hw[0]), int(image_shape_hw[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid image shape: {image_shape_hw}")

    k = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    t_cam_from_workspace = invert_transform(t_workspace_from_cam)
    corners_workspace = workspace_bbox_corners(bounds)
    corners_h = np.concatenate([corners_workspace, np.ones((len(corners_workspace), 1), dtype=np.float64)], axis=1)
    corners_cam = (t_cam_from_workspace @ corners_h.T).T[:, :3]
    valid = np.isfinite(corners_cam).all(axis=1) & (corners_cam[:, 2] > float(min_positive_depth_m))
    if not np.any(valid):
        return 0, 0, width, height

    pts = corners_cam[valid]
    u = k[0, 0] * pts[:, 0] / pts[:, 2] + k[0, 2]
    v = k[1, 1] * pts[:, 1] / pts[:, 2] + k[1, 2]
    finite = np.isfinite(u) & np.isfinite(v)
    if not np.any(finite):
        return 0, 0, width, height

    margin = int(margin_px)
    x0 = int(np.floor(float(np.min(u[finite])))) - margin
    y0 = int(np.floor(float(np.min(v[finite])))) - margin
    x1 = int(np.ceil(float(np.max(u[finite])))) + margin + 1
    y1 = int(np.ceil(float(np.max(v[finite])))) + margin + 1
    return _clip_roi(x0, y0, x1, y1, width, height)


def crop_camera_matrix(camera_matrix: np.ndarray, roi_xyxy: tuple[int, int, int, int]) -> np.ndarray:
    k = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3).copy()
    x0, y0, _, _ = [int(v) for v in roi_xyxy]
    k[0, 2] -= float(x0)
    k[1, 2] -= float(y0)
    return k.astype(np.float32)


def resize_camera_matrix(camera_matrix: np.ndarray, old_shape_hw: tuple[int, int], new_shape_hw: tuple[int, int]) -> np.ndarray:
    old_h, old_w = int(old_shape_hw[0]), int(old_shape_hw[1])
    new_h, new_w = int(new_shape_hw[0]), int(new_shape_hw[1])
    if old_h <= 0 or old_w <= 0 or new_h <= 0 or new_w <= 0:
        raise ValueError(f"Invalid resize shape: old={old_shape_hw}, new={new_shape_hw}")
    k = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3).copy()
    sx = float(new_w) / float(old_w)
    sy = float(new_h) / float(old_h)
    k[0, 0] *= sx
    k[0, 2] *= sx
    k[1, 1] *= sy
    k[1, 2] *= sy
    return k.astype(np.float32)


def _resize_rgb(rgb: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    if target_width <= 0 or target_height <= 0:
        return rgb.astype(np.uint8)
    if rgb.shape[1] == int(target_width) and rgb.shape[0] == int(target_height):
        return rgb.astype(np.uint8)
    if cv2 is None:
        raise RuntimeError("cv2 is required to resize cropped RGB.")
    return cv2.resize(rgb, (int(target_width), int(target_height)), interpolation=cv2.INTER_AREA).astype(np.uint8)


def apply_workspace_crop_to_camera_frame(
    *,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    camera_matrix: np.ndarray,
    t_workspace_from_cam: np.ndarray,
    bounds: WorkspaceBounds,
    margin_px: int = 0,
    resize_rgb_width: int = 0,
    resize_rgb_height: int = 0,
) -> dict[str, Any]:
    rgb_arr = np.asarray(rgb, dtype=np.uint8)
    depth_arr = np.asarray(depth_m, dtype=np.float32)
    if depth_arr.ndim != 2:
        raise ValueError(f"depth_m must be HxW, got {depth_arr.shape}")
    if rgb_arr.shape[:2] != depth_arr.shape:
        raise ValueError(f"Workspace crop expects raw RGB/depth to have the same shape, got {rgb_arr.shape[:2]} and {depth_arr.shape}")

    roi = project_workspace_bbox_to_roi(
        camera_matrix=camera_matrix,
        image_shape_hw=depth_arr.shape,
        t_workspace_from_cam=t_workspace_from_cam,
        bounds=bounds,
        margin_px=margin_px,
    )
    x0, y0, x1, y1 = roi
    rgb_crop_full = rgb_arr[y0:y1, x0:x1]
    depth_crop = depth_arr[y0:y1, x0:x1]
    k_crop = crop_camera_matrix(camera_matrix, roi)

    rgb_out = _resize_rgb(rgb_crop_full, int(resize_rgb_width), int(resize_rgb_height))
    if rgb_out.shape[:2] == rgb_crop_full.shape[:2]:
        k_rgb = k_crop.copy()
    else:
        k_rgb = resize_camera_matrix(k_crop, rgb_crop_full.shape[:2], rgb_out.shape[:2])

    return {
        "rgb": rgb_out.astype(np.uint8),
        "depth_m": depth_crop.astype(np.float32),
        "camera_matrix": k_crop.astype(np.float32),
        "rgb_camera_matrix": k_rgb.astype(np.float32),
        "original_camera_matrix": np.asarray(camera_matrix, dtype=np.float32).reshape(3, 3),
        "depth_crop_box_xyxy": np.asarray(roi, dtype=np.int32),
        "rgb_crop_box_xyxy": np.asarray(roi, dtype=np.int32),
        "original_depth_shape_hw": np.asarray(depth_arr.shape, dtype=np.int32),
        "original_rgb_shape_hw": np.asarray(rgb_arr.shape[:2], dtype=np.int32),
        "workspace_bounds_m": np.asarray(
            [bounds.x_min, bounds.x_max, bounds.y_min, bounds.y_max, bounds.z_min, bounds.z_max],
            dtype=np.float32,
        ),
        "t_workspace_from_camera": np.asarray(t_workspace_from_cam, dtype=np.float32).reshape(4, 4),
    }
