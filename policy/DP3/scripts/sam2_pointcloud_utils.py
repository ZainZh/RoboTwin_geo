from __future__ import annotations

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from object_pointcloud_utils import (
    ensure_point_cloud_channels,
    merge_object_point_clouds,
    resample_point_cloud,
    strip_zero_points,
)


@dataclass
class Sam2CameraTrackingState:
    tracker: object | None = None
    initialized: bool = False


def parse_camera_list(text: Optional[str], *, default: Sequence[str] = ("head_camera", "front_camera")) -> List[str]:
    if text is None or str(text).strip() == "":
        return [str(item) for item in default]
    return [item.strip() for item in str(text).split(",") if item.strip()]


def _read_bbox_prompt_json(path_or_root: str | Path) -> dict:
    path = Path(path_or_root).expanduser().resolve()
    if path.is_dir():
        path = path / "sam2_bbox_prompts.json"
    if not path.is_file():
        raise FileNotFoundError(f"SAM2 bbox prompt file does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_sam2_bbox_prompt_file(
    path_or_root: str | Path,
    *,
    camera_names: Sequence[str],
    placeholders: Sequence[str],
) -> Dict[str, Dict[str, object]]:
    data = _read_bbox_prompt_json(path_or_root)
    records = data.get("records", {})
    if not isinstance(records, dict):
        raise ValueError(f"Invalid SAM2 bbox prompt file: {path_or_root}")
    placeholder_filter = {str(item) for item in placeholders}
    out: Dict[str, Dict[str, object]] = {}
    for camera_name in camera_names:
        per_camera = records.get(str(camera_name), {})
        if not isinstance(per_camera, dict):
            continue
        out[str(camera_name)] = {}
        for placeholder, item in per_camera.items():
            if str(placeholder) not in placeholder_filter or not isinstance(item, dict):
                continue
            bbox = item.get("bbox_xyxy")
            points = item.get("points_xy", item.get("points"))
            labels = item.get("point_labels", item.get("labels"))
            if bbox is not None:
                bbox_record = [int(v) for v in bbox[:4]]
                image_shape = item.get("image_shape_hw")
                if image_shape is not None and len(image_shape) >= 2:
                    out[str(camera_name)][str(placeholder)] = {
                        "bbox_xyxy": bbox_record,
                        "image_shape_hw": [int(image_shape[0]), int(image_shape[1])],
                    }
                else:
                    out[str(camera_name)][str(placeholder)] = bbox_record
            elif points is not None:
                point_list = [[int(round(float(x))), int(round(float(y)))] for x, y in points]
                if labels is None:
                    label_list = [1 for _ in point_list]
                else:
                    label_list = [1 if int(v) > 0 else 0 for v in labels]
                if len(point_list) > 0 and len(point_list) == len(label_list):
                    prompt_record = {
                        "prompt_type": "point",
                        "points_xy": point_list,
                        "point_labels": label_list,
                    }
                    image_shape = item.get("image_shape_hw")
                    if image_shape is not None and len(image_shape) >= 2:
                        prompt_record["image_shape_hw"] = [int(image_shape[0]), int(image_shape[1])]
                    out[str(camera_name)][str(placeholder)] = prompt_record
    return out


def build_sam2_tracker_factory(
    *,
    placeholders: Sequence[str],
    sam2_root: str | Path,
    config: str,
    checkpoint: str | Path,
    device: str,
    autocast_dtype: str | None = "bfloat16",
) -> Callable[[str], object]:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from script.real_zed_collection.sam2_tracking_utils import make_sam2_tracker_factory

    return make_sam2_tracker_factory(
        placeholders=placeholders,
        sam2_root=sam2_root,
        config=config,
        checkpoint=checkpoint,
        device=device,
        autocast_dtype=autocast_dtype,
    )


def _project_points_with_matrix(
    points_world: np.ndarray,
    intrinsic_cv: np.ndarray,
    extrinsic_cv: np.ndarray,
    image_hw: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image_hw
    pts_h = np.concatenate([points_world, np.ones((len(points_world), 1), dtype=np.float32)], axis=1)
    pts_cam = (extrinsic_cv @ pts_h.T).T[:, :3]
    z = pts_cam[:, 2]
    valid = z > 1e-6
    uv = np.full((len(points_world), 2), -1, dtype=np.int64)
    if not np.any(valid):
        return uv, valid
    pts_cam_valid = pts_cam[valid]
    uv_float = (intrinsic_cv @ pts_cam_valid.T).T
    uv_float = uv_float[:, :2] / np.clip(uv_float[:, 2:3], 1e-6, None)
    u = np.round(uv_float[:, 0]).astype(np.int64)
    v = np.round(uv_float[:, 1]).astype(np.int64)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    valid_idx = np.nonzero(valid)[0]
    valid[valid_idx] &= inside
    uv[valid_idx[inside], 0] = u[inside]
    uv[valid_idx[inside], 1] = v[inside]
    return uv, valid


def _select_scene_points_by_mask(
    *,
    scene_point_cloud: np.ndarray,
    mask: np.ndarray,
    intrinsic_cv: np.ndarray,
    extrinsic_cv: np.ndarray,
    min_points: int,
) -> Tuple[np.ndarray | None, Dict[str, object]]:
    mask_arr = np.asarray(mask).astype(bool)
    uv, valid = _project_points_with_matrix(
        scene_point_cloud[:, :3],
        np.asarray(intrinsic_cv, dtype=np.float32),
        np.asarray(extrinsic_cv, dtype=np.float32),
        mask_arr.shape,
    )
    valid_idx = np.nonzero(valid)[0]
    if len(valid_idx) == 0:
        return None, {"mode": "no_projected_points", "projected_points": 0, "mask_pixels": int(mask_arr.sum())}
    projected_uv = uv[valid_idx]
    inside = mask_arr[projected_uv[:, 1], projected_uv[:, 0]]
    selected_idx = valid_idx[inside]
    if len(selected_idx) < int(min_points):
        return None, {
            "mode": "too_few_mask_points",
            "projected_points": int(len(valid_idx)),
            "selected_points": int(len(selected_idx)),
            "mask_pixels": int(mask_arr.sum()),
        }
    selected = scene_point_cloud[selected_idx]
    return selected.astype(np.float32), {
        "mode": "mask_projected",
        "projected_points": int(len(valid_idx)),
        "selected_points": int(len(selected_idx)),
        "mask_pixels": int(mask_arr.sum()),
    }


def _resize_keep_aspect(image: np.ndarray, target_width: int) -> np.ndarray:
    width = int(target_width)
    image_arr = np.asarray(image, dtype=np.uint8)
    if width <= 0 or image_arr.shape[1] <= width:
        return image_arr
    scale = float(width) / float(image_arr.shape[1])
    height = max(1, int(round(float(image_arr.shape[0]) * scale)))
    return cv2.resize(image_arr, (width, height), interpolation=cv2.INTER_AREA).astype(np.uint8)


def _scale_prompt_spec_to_image(prompt_spec, image_shape_hw: tuple[int, int]):
    if not isinstance(prompt_spec, Mapping):
        return prompt_spec
    old_shape = prompt_spec.get("image_shape_hw")
    if old_shape is None or len(old_shape) < 2:
        return prompt_spec
    old_h, old_w = float(old_shape[0]), float(old_shape[1])
    new_h, new_w = float(image_shape_hw[0]), float(image_shape_hw[1])
    if old_h <= 0 or old_w <= 0:
        return prompt_spec
    sx = new_w / old_w
    sy = new_h / old_h
    out = dict(prompt_spec)
    if "bbox_xyxy" in out:
        x0, y0, x1, y1 = [float(v) for v in out["bbox_xyxy"][:4]]
        out["bbox_xyxy"] = [
            int(round(x0 * sx)),
            int(round(y0 * sy)),
            int(round(x1 * sx)),
            int(round(y1 * sy)),
        ]
    points = out.get("points_xy", out.get("points"))
    if points is not None:
        out["points_xy"] = [
            [int(round(float(x) * sx)), int(round(float(y) * sy))]
            for x, y in points
        ]
    out["image_shape_hw"] = [int(new_h), int(new_w)]
    return out


def _transform_point_cloud(point_cloud: np.ndarray, transform: np.ndarray) -> np.ndarray:
    pc = np.asarray(point_cloud, dtype=np.float32)
    if pc.size == 0:
        return np.zeros((0, 6), dtype=np.float32)
    tf = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    xyz_h = np.concatenate([pc[:, :3].astype(np.float64), np.ones((len(pc), 1), dtype=np.float64)], axis=1)
    xyz = (tf @ xyz_h.T).T[:, :3].astype(np.float32)
    return np.concatenate([xyz, pc[:, 3:6].astype(np.float32)], axis=1)


def _parse_workspace_bounds(value) -> tuple[float, float, float, float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        try:
            return (
                float(value["x_min"]),
                float(value["x_max"]),
                float(value["y_min"]),
                float(value["y_max"]),
                float(value["z_min"]),
                float(value["z_max"]),
            )
        except Exception:
            return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < 6:
        return None
    return tuple(float(v) for v in arr[:6])


def _crop_point_cloud_xyz(point_cloud: np.ndarray, bounds: tuple[float, float, float, float, float, float] | None) -> np.ndarray:
    pc = np.asarray(point_cloud, dtype=np.float32)
    if bounds is None or pc.size == 0:
        return pc
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    mask = (
        (pc[:, 0] >= x_min)
        & (pc[:, 0] <= x_max)
        & (pc[:, 1] >= y_min)
        & (pc[:, 1] <= y_max)
        & (pc[:, 2] >= z_min)
        & (pc[:, 2] <= z_max)
    )
    return pc[mask]


def fast_resample_point_cloud(point_cloud: np.ndarray, target_num_points: int) -> np.ndarray:
    pc = strip_zero_points(point_cloud)
    target = int(target_num_points)
    if target <= 0:
        return pc.astype(np.float32)
    if len(pc) == 0:
        return np.zeros((target, 6), dtype=np.float32)
    if len(pc) == target:
        return pc.astype(np.float32)
    if len(pc) > target:
        idx = np.linspace(0, len(pc) - 1, target).round().astype(np.int64)
        return pc[idx].astype(np.float32)
    reps = target - len(pc)
    idx = np.arange(reps, dtype=np.int64) % len(pc)
    return np.concatenate([pc, pc[idx]], axis=0).astype(np.float32)


def fast_merge_object_point_clouds(point_clouds: Sequence[np.ndarray], target_num_points: int) -> np.ndarray:
    cleaned = []
    for point_cloud in point_clouds:
        pc = strip_zero_points(point_cloud)
        if len(pc) > 0:
            cleaned.append(pc)
    if not cleaned:
        return np.zeros((int(target_num_points), 6), dtype=np.float32)
    merged = cleaned[0] if len(cleaned) == 1 else np.concatenate(cleaned, axis=0)
    return fast_resample_point_cloud(merged, int(target_num_points))


def _resolve_camera_workers(camera_names: Sequence[str], max_workers: int | None = None) -> int:
    count = len([str(item) for item in camera_names])
    if count <= 1:
        return 1
    if max_workers is None or int(max_workers) <= 0:
        return count
    return max(1, min(int(max_workers), count))


def run_camera_tasks_parallel(
    camera_names: Sequence[str],
    worker: Callable[[str], object],
    *,
    max_workers: int | None = None,
) -> Dict[str, object]:
    names = [str(item) for item in camera_names]
    if not names:
        return {}
    worker_count = _resolve_camera_workers(names, max_workers=max_workers)
    if worker_count <= 1:
        return {name: worker(name) for name in names}
    results: Dict[str, object] = {}
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_name = {executor.submit(worker, name): name for name in names}
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            results[name] = future.result()
    return {name: results[name] for name in names if name in results}


def _select_depth_points_by_mask(
    *,
    mask: np.ndarray,
    depth: np.ndarray,
    rgb: np.ndarray,
    intrinsic_cv: np.ndarray,
    cam2world_gl: np.ndarray,
    min_depth_m: float,
    max_depth_m: float,
    min_points: int,
    t_workspace_from_cam: np.ndarray | None = None,
    workspace_bounds_m=None,
) -> Tuple[np.ndarray | None, Dict[str, object]]:
    depth_arr = np.asarray(depth, dtype=np.float32)
    if depth_arr.ndim != 2:
        return None, {"mode": "invalid_depth", "mask_pixels": int(np.asarray(mask).astype(bool).sum())}
    mask_arr = np.asarray(mask).astype(bool)
    if mask_arr.shape != depth_arr.shape:
        mask_arr = cv2.resize(mask_arr.astype(np.uint8), (depth_arr.shape[1], depth_arr.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    rgb_arr = np.asarray(rgb, dtype=np.uint8)
    if rgb_arr.shape[:2] != depth_arr.shape:
        rgb_arr = cv2.resize(rgb_arr, (depth_arr.shape[1], depth_arr.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.uint8)

    valid = (
        mask_arr
        & np.isfinite(depth_arr)
        & (depth_arr > float(min_depth_m))
        & (depth_arr < float(max_depth_m))
    )
    ys, xs = np.where(valid)
    if xs.size == 0:
        return None, {
            "mode": "too_few_mask_depth_points",
            "mask_pixels": int(mask_arr.sum()),
            "depth_points": 0,
            "workspace_points": 0,
        }

    k = np.asarray(intrinsic_cv, dtype=np.float64).reshape(3, 3)
    z = depth_arr[ys, xs].astype(np.float64)
    x = (xs.astype(np.float64) - float(k[0, 2])) * z / float(k[0, 0])
    y = (ys.astype(np.float64) - float(k[1, 2])) * z / float(k[1, 1])
    xyz_cam = np.stack([x, y, z], axis=1).astype(np.float32)
    colors = rgb_arr[ys, xs, :3].astype(np.float32) / 255.0
    pc_cam = np.concatenate([xyz_cam, colors], axis=1).astype(np.float32)

    bounds = _parse_workspace_bounds(workspace_bounds_m)
    if t_workspace_from_cam is not None and bounds is not None:
        pc_workspace = _transform_point_cloud(pc_cam, np.asarray(t_workspace_from_cam, dtype=np.float32).reshape(4, 4))
        pc_workspace = _crop_point_cloud_xyz(pc_workspace, bounds)
        workspace_points = int(len(pc_workspace))
        if workspace_points < int(min_points):
            return None, {
                "mode": "too_few_mask_depth_points",
                "mask_pixels": int(mask_arr.sum()),
                "depth_points": int(len(pc_cam)),
                "workspace_points": workspace_points,
            }
        t_output_from_workspace = np.asarray(cam2world_gl, dtype=np.float64).reshape(4, 4) @ np.linalg.inv(
            np.asarray(t_workspace_from_cam, dtype=np.float64).reshape(4, 4)
        )
        selected = _transform_point_cloud(pc_workspace, t_output_from_workspace)
    else:
        selected = _transform_point_cloud(pc_cam, np.asarray(cam2world_gl, dtype=np.float32).reshape(4, 4))
        workspace_points = int(len(selected))
        if workspace_points < int(min_points):
            return None, {
                "mode": "too_few_mask_depth_points",
                "mask_pixels": int(mask_arr.sum()),
                "depth_points": int(len(pc_cam)),
                "workspace_points": workspace_points,
            }

    return selected.astype(np.float32), {
        "mode": "mask_depth_lifted",
        "mask_pixels": int(mask_arr.sum()),
        "depth_points": int(len(pc_cam)),
        "workspace_points": int(workspace_points),
        "selected_points": int(len(selected)),
    }


def _decode_rgb(image) -> np.ndarray | None:
    if image is None:
        return None
    if isinstance(image, (bytes, bytearray, np.bytes_)):
        encoded = bytes(image).rstrip(b"\0")
        if len(encoded) == 0:
            return None
        arr = np.frombuffer(encoded, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image_arr = np.asarray(image)
    if image_arr.ndim != 3:
        return None
    return image_arr.astype(np.uint8)


def _normalize_bbox_xyxy(bbox_xyxy: Sequence[float], image_shape_hw: tuple[int, int]) -> List[int]:
    height, width = int(image_shape_hw[0]), int(image_shape_hw[1])
    x0, y0, x1, y1 = [int(round(float(v))) for v in bbox_xyxy]
    lo_x, hi_x = sorted((x0, x1))
    lo_y, hi_y = sorted((y0, y1))
    lo_x = max(0, min(lo_x, max(0, width - 1)))
    hi_x = max(0, min(hi_x, width))
    lo_y = max(0, min(lo_y, max(0, height - 1)))
    hi_y = max(0, min(hi_y, height))
    if hi_x <= lo_x or hi_y <= lo_y:
        raise ValueError(f"Invalid SAM2 bbox after clipping: {bbox_xyxy}")
    return [int(lo_x), int(lo_y), int(hi_x), int(hi_y)]


def _try_normalize_bbox_xyxy(bbox_xyxy: Sequence[float] | None, image_shape_hw: tuple[int, int]) -> List[int] | None:
    if bbox_xyxy is None:
        return None
    try:
        return _normalize_bbox_xyxy(bbox_xyxy, image_shape_hw)
    except ValueError:
        return None


def _display_scale_for_image(image_shape_hw: tuple[int, int], *, max_width: int = 1280, max_height: int = 720) -> float:
    height, width = int(image_shape_hw[0]), int(image_shape_hw[1])
    if height <= 0 or width <= 0:
        return 1.0
    return float(min(1.0, float(max_width) / float(width), float(max_height) / float(height)))


def _display_to_image_point(
    x: int,
    y: int,
    *,
    scale: float,
    image_shape_hw: tuple[int, int],
) -> tuple[int, int]:
    height, width = int(image_shape_hw[0]), int(image_shape_hw[1])
    safe_scale = max(float(scale), 1e-6)
    image_x = int(round(float(x) / safe_scale))
    image_y = int(round(float(y) / safe_scale))
    image_x = max(0, min(image_x, max(0, width - 1)))
    image_y = max(0, min(image_y, max(0, height - 1)))
    return image_x, image_y


def select_bbox_for_image(label: str, placeholder: str, image: np.ndarray) -> List[int]:
    image = np.asarray(image, dtype=np.uint8)
    drawing = False
    start: tuple[int, int] | None = None
    current: tuple[int, int] | None = None
    bbox: tuple[int, int, int, int] | None = None
    status = ""
    window = f"SAM2 eval bbox: {label} {placeholder}"
    scale = _display_scale_for_image(image.shape[:2])
    display_size = (
        max(1, int(round(image.shape[1] * scale))),
        max(1, int(round(image.shape[0] * scale))),
    )

    def on_mouse(event, x, y, _flags, _param):
        nonlocal drawing, start, current, bbox, status
        point = _display_to_image_point(int(x), int(y), scale=scale, image_shape_hw=image.shape[:2])
        current = point
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start = point
            bbox = None
            status = ""
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            current = point
        elif event == cv2.EVENT_LBUTTONUP and drawing and start is not None:
            drawing = False
            candidate = (start[0], start[1], point[0], point[1])
            if _try_normalize_bbox_xyxy(candidate, image.shape[:2]) is None:
                bbox = None
                status = "Invalid bbox: drag a non-zero area rectangle."
            else:
                bbox = candidate
                status = ""

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, display_size[0], display_size[1])
    cv2.setMouseCallback(window, on_mouse)
    while True:
        vis = image.copy()
        if drawing and start is not None and current is not None:
            cv2.rectangle(vis, start, current, (0, 255, 255), 2)
        if bbox is not None:
            normalized_bbox = _try_normalize_bbox_xyxy(bbox, image.shape[:2])
            if normalized_bbox is not None:
                x0, y0, x1, y1 = normalized_bbox
                cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 220, 255), 2)
        display = cv2.resize(vis, display_size, interpolation=cv2.INTER_AREA) if scale < 1.0 else vis
        cv2.putText(
            display,
            f"Draw {placeholder} in {label}; Enter/Space/s save, r reset, q/Esc abort",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
        )
        if status:
            cv2.putText(
                display,
                status,
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 80, 80),
                2,
            )
        cv2.imshow(window, cv2.cvtColor(display, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(20) & 0xFF
        normalized_bbox = _try_normalize_bbox_xyxy(bbox, image.shape[:2])
        if key in (13, 32, ord("s")) and normalized_bbox is not None:
            cv2.destroyWindow(window)
            return normalized_bbox
        if key in (13, 32, ord("s")) and normalized_bbox is None:
            status = "No valid bbox yet: drag a rectangle before saving."
        if key in (27, ord("q")):
            cv2.destroyWindow(window)
            raise KeyboardInterrupt("SAM2 eval bbox selection aborted")
        if key == ord("r"):
            bbox = None
            start = None
            current = None
            drawing = False


def _interactive_boxes_for_camera(
    *,
    camera_name: str,
    placeholders: Sequence[str],
    image: np.ndarray,
) -> Dict[str, List[int]]:
    return {
        str(placeholder): select_bbox_for_image(camera_name, str(placeholder), image)
        for placeholder in placeholders
    }


def extract_placeholder_point_clouds_sam2_online(
    observation: dict,
    *,
    placeholders: Sequence[str],
    camera_names: Sequence[str],
    tracker_factory: Callable[[str], object],
    tracking_state_by_camera: Dict[str, Sam2CameraTrackingState],
    bbox_prompts_by_camera: Mapping[str, Mapping[str, object]] | None,
    target_num_points: int,
    min_mask_points: int = 16,
    interactive_init: bool = False,
    sam2_image_width: int = 0,
    min_depth_m: float = 0.05,
    max_depth_m: float = 3.0,
    object_resample_mode: str = "fps",
    parallel_camera_workers: int | None = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    placeholder_list = [str(item) for item in placeholders]
    camera_name_list = [str(item) for item in camera_names]
    empty_clouds = {
        placeholder: np.zeros((int(target_num_points), 6), dtype=np.float32)
        for placeholder in placeholder_list
    }

    camera_obs = observation.get("observation", {})
    selected_by_placeholder: Dict[str, List[np.ndarray]] = {placeholder: [] for placeholder in placeholder_list}
    camera_meta: Dict[str, Dict[str, object]] = {}
    if bbox_prompts_by_camera is None:
        bbox_prompts_mutable: Dict[str, Dict[str, object]] = {}
    elif isinstance(bbox_prompts_by_camera, dict):
        bbox_prompts_mutable = bbox_prompts_by_camera
    else:
        bbox_prompts_mutable = {
            str(camera): dict(prompts)
            for camera, prompts in bbox_prompts_by_camera.items()
        }

    if bool(interactive_init):
        for camera_name in camera_name_list:
            camera_info = camera_obs.get(camera_name)
            if camera_info is None:
                continue
            state = tracking_state_by_camera.get(camera_name)
            if state is not None and state.initialized:
                continue
            boxes = dict(bbox_prompts_mutable.get(camera_name, {}))
            if boxes:
                continue
            image = _decode_rgb(camera_info.get("rgb"))
            if image is None:
                continue
            tracker_image = _resize_keep_aspect(image, int(sam2_image_width))
            bbox_prompts_mutable[camera_name] = _interactive_boxes_for_camera(
                camera_name=camera_name,
                placeholders=placeholder_list,
                image=tracker_image,
            )

    state_lock = threading.Lock()
    scene_lock = threading.Lock()
    scene_pc_cache: np.ndarray | None = None

    def get_scene_pc() -> np.ndarray:
        nonlocal scene_pc_cache
        if scene_pc_cache is not None:
            return scene_pc_cache
        with scene_lock:
            if scene_pc_cache is None:
                scene_pc = ensure_point_cloud_channels(
                    observation.get("pointcloud", np.zeros((0, 6), dtype=np.float32)),
                    channels=6,
                )
                scene_pc_cache = strip_zero_points(scene_pc)
        return scene_pc_cache

    def process_camera(camera_name: str) -> Tuple[str, Dict[str, List[np.ndarray]], Dict[str, object], bool]:
        camera_key = str(camera_name)
        selected: Dict[str, List[np.ndarray]] = {placeholder: [] for placeholder in placeholder_list}
        camera_info = camera_obs.get(camera_key)
        if camera_info is None:
            return camera_key, selected, {"mode": "missing_camera"}, False
        image = _decode_rgb(camera_info.get("rgb"))
        intrinsic_cv = camera_info.get("intrinsic_cv")
        extrinsic_cv = camera_info.get("extrinsic_cv")
        if image is None or intrinsic_cv is None:
            return camera_key, selected, {"mode": "missing_camera_data"}, False
        tracker_image = _resize_keep_aspect(image, int(sam2_image_width))

        with state_lock:
            state = tracking_state_by_camera.setdefault(camera_key, Sam2CameraTrackingState())
        if state.tracker is None:
            state.tracker = tracker_factory(camera_key)
        if not state.initialized:
            boxes = dict(bbox_prompts_mutable.get(camera_key, {}))
            missing_boxes = [placeholder for placeholder in placeholder_list if placeholder not in boxes]
            if missing_boxes:
                return camera_key, selected, {
                    "mode": "missing_bbox_prompts",
                    "missing_placeholders": missing_boxes,
                }, False
            boxes = {
                placeholder: _scale_prompt_spec_to_image(prompt_spec, tracker_image.shape[:2])
                for placeholder, prompt_spec in boxes.items()
            }
            masks = state.tracker.initialize(tracker_image, boxes)
            state.initialized = True
            mode = "initialized"
        else:
            masks = state.tracker.track(tracker_image)
            mode = "tracked"

        per_placeholder_meta: Dict[str, object] = {}
        camera_selected = False
        depth = camera_info.get("depth", camera_info.get("depth_m"))
        cam2world_gl = camera_info.get("cam2world_gl")
        can_lift_depth = depth is not None and cam2world_gl is not None
        for placeholder in placeholder_list:
            mask = np.asarray(masks.get(placeholder, np.zeros(tracker_image.shape[:2], dtype=bool))).astype(bool)
            if can_lift_depth:
                points, meta = _select_depth_points_by_mask(
                    mask=mask,
                    depth=depth,
                    rgb=image,
                    intrinsic_cv=intrinsic_cv,
                    cam2world_gl=cam2world_gl,
                    t_workspace_from_cam=camera_info.get("t_workspace_from_cam"),
                    workspace_bounds_m=camera_info.get("workspace_bounds_m"),
                    min_depth_m=float(min_depth_m),
                    max_depth_m=float(max_depth_m),
                    min_points=int(min_mask_points),
                )
            elif extrinsic_cv is not None:
                scene_pc = get_scene_pc()
                if len(scene_pc) > 0:
                    points, meta = _select_scene_points_by_mask(
                        scene_point_cloud=scene_pc,
                        mask=mask,
                        intrinsic_cv=intrinsic_cv,
                        extrinsic_cv=extrinsic_cv,
                        min_points=int(min_mask_points),
                    )
                else:
                    points, meta = None, {
                        "mode": "missing_depth_and_scene_projection_data",
                        "mask_pixels": int(mask.sum()),
                    }
            else:
                points, meta = None, {
                    "mode": "missing_depth_and_scene_projection_data",
                    "mask_pixels": int(mask.sum()),
                }
            if points is not None:
                selected[placeholder].append(points)
                camera_selected = True
            per_placeholder_meta[placeholder] = meta
        return camera_key, selected, {
            "mode": mode,
            "placeholders": per_placeholder_meta,
        }, camera_selected

    active_camera_count = 0
    camera_results = run_camera_tasks_parallel(
        camera_name_list,
        process_camera,
        max_workers=parallel_camera_workers,
    )
    for camera_name in camera_name_list:
        if camera_name not in camera_results:
            continue
        camera_key, selected, meta, camera_selected = camera_results[camera_name]
        camera_meta[camera_key] = meta
        for placeholder, clouds in selected.items():
            selected_by_placeholder[placeholder].extend(clouds)
        if camera_selected:
            active_camera_count += 1

    out: Dict[str, np.ndarray] = {}
    for placeholder, selected_clouds in selected_by_placeholder.items():
        if not selected_clouds:
            out[placeholder] = empty_clouds[placeholder]
            continue
        if str(object_resample_mode).strip().lower() == "fast":
            out[placeholder] = fast_merge_object_point_clouds(selected_clouds, target_num_points=int(target_num_points))
        else:
            merged = merge_object_point_clouds(selected_clouds, target_num_points=int(target_num_points))
            out[placeholder] = resample_point_cloud(merged, int(target_num_points))

    return out, {
        "mode": "sam2_projected",
        "object_resample_mode": str(object_resample_mode),
        "camera_count": int(active_camera_count),
        "cameras": camera_meta,
    }


__all__ = [
    "Sam2CameraTrackingState",
    "build_sam2_tracker_factory",
    "extract_placeholder_point_clouds_sam2_online",
    "fast_merge_object_point_clouds",
    "fast_resample_point_cloud",
    "load_sam2_bbox_prompt_file",
    "parse_camera_list",
    "run_camera_tasks_parallel",
    "select_bbox_for_image",
]
