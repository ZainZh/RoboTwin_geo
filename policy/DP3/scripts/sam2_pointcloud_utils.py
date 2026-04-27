from __future__ import annotations

import json
import sys
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
                out[str(camera_name)][str(placeholder)] = [int(v) for v in bbox[:4]]
            elif points is not None:
                point_list = [[int(round(float(x))), int(round(float(y)))] for x, y in points]
                if labels is None:
                    label_list = [1 for _ in point_list]
                else:
                    label_list = [1 if int(v) > 0 else 0 for v in labels]
                if len(point_list) > 0 and len(point_list) == len(label_list):
                    out[str(camera_name)][str(placeholder)] = {
                        "prompt_type": "point",
                        "points_xy": point_list,
                        "point_labels": label_list,
                    }
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


def select_bbox_for_image(label: str, placeholder: str, image: np.ndarray) -> List[int]:
    drawing = False
    start: tuple[int, int] | None = None
    current: tuple[int, int] | None = None
    bbox: tuple[int, int, int, int] | None = None
    window = f"SAM2 eval bbox: {label} {placeholder}"

    def on_mouse(event, x, y, _flags, _param):
        nonlocal drawing, start, current, bbox
        current = (int(x), int(y))
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start = (int(x), int(y))
            bbox = None
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            current = (int(x), int(y))
        elif event == cv2.EVENT_LBUTTONUP and drawing and start is not None:
            drawing = False
            bbox = (start[0], start[1], int(x), int(y))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        vis = image.copy()
        if drawing and start is not None and current is not None:
            cv2.rectangle(vis, start, current, (0, 255, 255), 2)
        if bbox is not None:
            x0, y0, x1, y1 = _normalize_bbox_xyxy(bbox, image.shape[:2])
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 220, 255), 2)
        cv2.putText(
            vis,
            f"Draw {placeholder} in {label}; Enter/Space/s save, r reset, q/Esc abort",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
        )
        cv2.imshow(window, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 32, ord("s")) and bbox is not None:
            cv2.destroyWindow(window)
            return _normalize_bbox_xyxy(bbox, image.shape[:2])
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
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    scene_pc = ensure_point_cloud_channels(observation["pointcloud"], channels=6)
    scene_pc = strip_zero_points(scene_pc)
    placeholder_list = [str(item) for item in placeholders]
    empty_clouds = {
        placeholder: np.zeros((int(target_num_points), 6), dtype=np.float32)
        for placeholder in placeholder_list
    }
    if len(scene_pc) == 0:
        return empty_clouds, {"mode": "empty_scene"}

    camera_obs = observation.get("observation", {})
    selected_by_placeholder: Dict[str, List[np.ndarray]] = {placeholder: [] for placeholder in placeholder_list}
    camera_meta: Dict[str, Dict[str, object]] = {}
    bbox_prompts_by_camera = bbox_prompts_by_camera or {}
    active_camera_count = 0

    for camera_name in camera_names:
        camera_info = camera_obs.get(camera_name)
        if camera_info is None:
            camera_meta[str(camera_name)] = {"mode": "missing_camera"}
            continue
        image = _decode_rgb(camera_info.get("rgb"))
        intrinsic_cv = camera_info.get("intrinsic_cv")
        extrinsic_cv = camera_info.get("extrinsic_cv")
        if image is None or intrinsic_cv is None or extrinsic_cv is None:
            camera_meta[str(camera_name)] = {"mode": "missing_camera_data"}
            continue

        state = tracking_state_by_camera.setdefault(str(camera_name), Sam2CameraTrackingState())
        if state.tracker is None:
            state.tracker = tracker_factory(str(camera_name))
        if not state.initialized:
            boxes = dict(bbox_prompts_by_camera.get(str(camera_name), {}))
            if not boxes and bool(interactive_init):
                boxes = _interactive_boxes_for_camera(
                    camera_name=str(camera_name),
                    placeholders=placeholder_list,
                    image=image,
                )
                bbox_prompts_by_camera[str(camera_name)] = boxes
            missing_boxes = [placeholder for placeholder in placeholder_list if placeholder not in boxes]
            if missing_boxes:
                camera_meta[str(camera_name)] = {
                    "mode": "missing_bbox_prompts",
                    "missing_placeholders": missing_boxes,
                }
                continue
            masks = state.tracker.initialize(image, boxes)
            state.initialized = True
            mode = "initialized"
        else:
            masks = state.tracker.track(image)
            mode = "tracked"

        per_placeholder_meta: Dict[str, object] = {}
        camera_selected = False
        for placeholder in placeholder_list:
            mask = np.asarray(masks.get(placeholder, np.zeros(image.shape[:2], dtype=bool))).astype(bool)
            points, meta = _select_scene_points_by_mask(
                scene_point_cloud=scene_pc,
                mask=mask,
                intrinsic_cv=intrinsic_cv,
                extrinsic_cv=extrinsic_cv,
                min_points=int(min_mask_points),
            )
            if points is not None:
                selected_by_placeholder[placeholder].append(points)
                camera_selected = True
            per_placeholder_meta[placeholder] = meta
        camera_meta[str(camera_name)] = {
            "mode": mode,
            "placeholders": per_placeholder_meta,
        }
        if camera_selected:
            active_camera_count += 1

    out: Dict[str, np.ndarray] = {}
    for placeholder, selected_clouds in selected_by_placeholder.items():
        if not selected_clouds:
            out[placeholder] = empty_clouds[placeholder]
            continue
        merged = merge_object_point_clouds(selected_clouds, target_num_points=int(target_num_points))
        out[placeholder] = resample_point_cloud(merged, int(target_num_points))

    return out, {
        "mode": "sam2_projected",
        "camera_count": int(active_camera_count),
        "cameras": camera_meta,
    }


__all__ = [
    "Sam2CameraTrackingState",
    "build_sam2_tracker_factory",
    "extract_placeholder_point_clouds_sam2_online",
    "load_sam2_bbox_prompt_file",
    "parse_camera_list",
    "select_bbox_for_image",
]
