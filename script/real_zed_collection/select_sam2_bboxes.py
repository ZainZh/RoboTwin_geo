#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from script.real_zed_collection.real_zed_utils import ensure_dir, read_json, write_json
from script.real_zed_collection.sam2_tracking_utils import (
    DEFAULT_SAM2_CHECKPOINT,
    DEFAULT_SAM2_CONFIG,
    DEFAULT_SAM2_ROOT,
    make_sam2_tracker_factory,
)
from script.real_zed_collection.select_camera_workspace_masks import _load_rgb, _resolve_raw_episode


DEFAULT_RAW_ROOT = "/media/clover/Extreme SSD/geo_mani_data/grasp_mug/real_zed_raw"
DEFAULT_OUTPUT_ROOT = "/media/clover/Extreme SSD/geo_mani_data/grasp_mug/sam2_bbox_prompts"
DISPLAY_HEADER_HEIGHT = 104
DEFAULT_DISPLAY_SCALE = 1.8


def normalize_bbox_xyxy(bbox_xyxy: Sequence[float], image_shape_hw: tuple[int, int]) -> list[int]:
    height, width = int(image_shape_hw[0]), int(image_shape_hw[1])
    x0, y0, x1, y1 = [int(round(float(v))) for v in bbox_xyxy]
    lo_x, hi_x = sorted((x0, x1))
    lo_y, hi_y = sorted((y0, y1))
    lo_x = max(0, min(lo_x, max(0, width - 1)))
    hi_x = max(0, min(hi_x, width))
    lo_y = max(0, min(lo_y, max(0, height - 1)))
    hi_y = max(0, min(hi_y, height))
    if hi_x <= lo_x or hi_y <= lo_y:
        raise ValueError(f"Invalid bbox after clipping: {bbox_xyxy} for image shape {image_shape_hw}")
    return [int(lo_x), int(lo_y), int(hi_x), int(hi_y)]


def try_normalize_bbox_xyxy(bbox_xyxy: Sequence[float] | None, image_shape_hw: tuple[int, int]) -> list[int] | None:
    if bbox_xyxy is None:
        return None
    try:
        return normalize_bbox_xyxy(bbox_xyxy, image_shape_hw)
    except ValueError:
        return None


def normalize_prompt_record(prompt_spec: Any, image_shape_hw: tuple[int, int]) -> dict[str, object]:
    shape = [int(image_shape_hw[0]), int(image_shape_hw[1])]
    if isinstance(prompt_spec, Mapping):
        prompt_type = str(prompt_spec.get("prompt_type", "")).strip().lower()
        bbox = prompt_spec.get("bbox_xyxy")
        points = prompt_spec.get("points_xy", prompt_spec.get("points"))
        labels = prompt_spec.get("point_labels", prompt_spec.get("labels"))
    else:
        prompt_type = "bbox"
        bbox = prompt_spec
        points = None
        labels = None

    record: dict[str, object] = {"image_shape_hw": shape}
    normalized_bbox = try_normalize_bbox_xyxy(bbox, image_shape_hw) if bbox is not None else None
    if normalized_bbox is not None:
        record["bbox_xyxy"] = normalized_bbox

    points_out: list[list[int]] = []
    labels_out: list[int] = []
    if points is not None:
        points_arr = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if labels is None:
            labels_arr = np.ones((points_arr.shape[0],), dtype=np.int32)
        else:
            labels_arr = np.asarray(labels, dtype=np.int32).reshape(-1)
        if labels_arr.shape[0] != points_arr.shape[0]:
            raise ValueError("SAM2 point prompt labels must match points.")
        height, width = int(image_shape_hw[0]), int(image_shape_hw[1])
        for point, label in zip(points_arr, labels_arr):
            x = max(0, min(int(round(float(point[0]))), max(0, width - 1)))
            y = max(0, min(int(round(float(point[1]))), max(0, height - 1)))
            points_out.append([x, y])
            labels_out.append(1 if int(label) > 0 else 0)
    if points_out:
        record["points_xy"] = points_out
        record["point_labels"] = labels_out

    if "bbox_xyxy" not in record and "points_xy" not in record:
        raise ValueError(f"Prompt has neither valid bbox nor points: {prompt_spec}")
    if not prompt_type:
        prompt_type = "point" if "points_xy" in record and "bbox_xyxy" not in record else "bbox"
    record["prompt_type"] = prompt_type
    return record


def save_sam2_bbox_prompts(
    *,
    output_root: str | Path,
    raw_episode_dir: str | Path,
    frame_index: int,
    image_shapes_by_camera: Mapping[str, Sequence[int]],
    boxes_by_camera: Mapping[str, Mapping[str, object]],
) -> Path:
    output_root = ensure_dir(output_root)
    records: dict[str, dict[str, dict[str, object]]] = {}
    for camera_label, per_placeholder in boxes_by_camera.items():
        shape = tuple(int(v) for v in image_shapes_by_camera[camera_label][:2])
        records[str(camera_label)] = {}
        for placeholder, prompt_spec in per_placeholder.items():
            records[str(camera_label)][str(placeholder)] = normalize_prompt_record(prompt_spec, shape)
    payload = {
        "raw_episode_dir": str(raw_episode_dir),
        "frame_index": int(frame_index),
        "camera_labels": sorted(records.keys()),
        "records": records,
    }
    path = output_root / "sam2_bbox_prompts.json"
    write_json(path, payload)
    return path


def load_sam2_prompt_records(
    path_or_root: str | Path,
    *,
    camera_labels: Sequence[str] | None = None,
    placeholders: Sequence[str] | None = None,
) -> dict[str, dict[str, dict[str, object]]]:
    path = Path(path_or_root).expanduser().resolve()
    if path.is_dir():
        path = path / "sam2_bbox_prompts.json"
    data = read_json(path)
    records = data.get("records", {})
    if not isinstance(records, dict):
        raise ValueError(f"Invalid SAM2 bbox prompt file: {path}")
    labels = [str(item) for item in (camera_labels or records.keys())]
    placeholder_filter = None if placeholders is None else {str(item) for item in placeholders}
    out: dict[str, dict[str, dict[str, object]]] = {}
    for label in labels:
        per_camera = records.get(label, {})
        if not isinstance(per_camera, dict):
            continue
        out[label] = {}
        for placeholder, item in per_camera.items():
            if placeholder_filter is not None and str(placeholder) not in placeholder_filter:
                continue
            if not isinstance(item, dict):
                continue
            image_shape = item.get("image_shape_hw", [1_000_000, 1_000_000])
            out[label][str(placeholder)] = normalize_prompt_record(item, tuple(int(v) for v in image_shape[:2]))
    return out


def load_sam2_bbox_prompts(
    path_or_root: str | Path,
    *,
    camera_labels: Sequence[str] | None = None,
    placeholders: Sequence[str] | None = None,
) -> dict[str, dict[str, list[int]]]:
    records = load_sam2_prompt_records(path_or_root, camera_labels=camera_labels, placeholders=placeholders)
    out: dict[str, dict[str, list[int]]] = {}
    for label, per_camera in records.items():
        out[label] = {}
        for placeholder, item in per_camera.items():
            bbox = item.get("bbox_xyxy")
            if bbox is not None:
                out[label][placeholder] = [int(v) for v in bbox]
    return out


def find_raw_episode_dirs(raw_root: str | Path) -> list[Path]:
    root = Path(raw_root).expanduser().resolve()
    episodes = sorted(path for path in root.glob("episode_*") if (path / "manifest.json").exists())
    if not episodes:
        raise FileNotFoundError(f"No raw episodes found under: {root}")
    return episodes


def _draw_existing_boxes(image: np.ndarray, boxes: Mapping[str, object]) -> np.ndarray:
    vis = image.copy()
    colors = {
        "{A}": (255, 48, 48),
        "A": (255, 48, 48),
        "{B}": (48, 96, 255),
        "B": (48, 96, 255),
    }
    for placeholder, prompt in boxes.items():
        bbox = prompt.get("bbox_xyxy") if isinstance(prompt, Mapping) else prompt
        color = colors.get(str(placeholder), (255, 255, 0))
        if bbox is not None:
            x0, y0, x1, y1 = [int(v) for v in bbox]
            cv2.rectangle(vis, (x0, y0), (x1, y1), color=(int(color[2]), int(color[1]), int(color[0])), thickness=2)
            cv2.putText(vis, str(placeholder), (x0 + 4, max(16, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if isinstance(prompt, Mapping):
            for point, label in zip(prompt.get("points_xy", []), prompt.get("point_labels", [])):
                x, y = [int(v) for v in point[:2]]
                point_color = (0, 255, 0) if int(label) > 0 else (255, 48, 48)
                cv2.circle(vis, (x, y), 5, point_color, -1)
                cv2.putText(vis, str(placeholder), (x + 6, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, point_color, 2)
    return vis


def _display_to_image_point(
    x: int,
    y: int,
    *,
    display_scale: float,
    image_shape_hw: tuple[int, int],
) -> tuple[int, int] | None:
    image_y = int(y) - DISPLAY_HEADER_HEIGHT
    if image_y < 0:
        return None
    height, width = int(image_shape_hw[0]), int(image_shape_hw[1])
    image_x = int(round(float(x) / float(display_scale)))
    image_y = int(round(float(image_y) / float(display_scale)))
    image_x = max(0, min(image_x, max(0, width - 1)))
    image_y = max(0, min(image_y, max(0, height - 1)))
    return image_x, image_y


def _render_selection_canvas(
    image: np.ndarray,
    *,
    display_scale: float,
    episode_label: str,
    camera_label: str,
    placeholder: str,
    prompt_mode: str,
    start: tuple[int, int] | None,
    current: tuple[int, int] | None,
    bbox: tuple[int, int, int, int] | None,
    points_xy: Sequence[Sequence[int]],
    point_labels: Sequence[int],
    preview_mask: np.ndarray | None,
    preview_status: str,
    drawing: bool,
) -> np.ndarray:
    vis = image.copy()
    if preview_mask is not None:
        mask = np.asarray(preview_mask).astype(bool)
        if mask.shape != vis.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        if bool(mask.any()):
            color = np.asarray([255, 180, 0], dtype=np.float32)
            dimmed = (vis.astype(np.float32) * 0.45).clip(0, 255).astype(np.uint8)
            vis = np.where(mask[..., None], vis, dimmed)
            vis[mask] = (0.35 * vis[mask].astype(np.float32) + 0.65 * color).clip(0, 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, (255, 255, 255), 2)
    if drawing and start is not None and current is not None:
        cv2.rectangle(vis, start, current, (0, 255, 255), 2)
    if bbox is not None:
        normalized = try_normalize_bbox_xyxy(bbox, image.shape[:2])
        if normalized is not None:
            x0, y0, x1, y1 = normalized
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 220, 255), 2)
        else:
            cv2.circle(vis, (int(bbox[0]), int(bbox[1])), 4, (255, 128, 0), -1)
    for point, label in zip(points_xy, point_labels):
        x, y = [int(v) for v in point[:2]]
        color = (0, 255, 0) if int(label) > 0 else (255, 48, 48)
        cv2.circle(vis, (x, y), 5, color, -1)
        cv2.circle(vis, (x, y), 8, (255, 255, 255), 1)

    height, width = image.shape[:2]
    display_width = max(1, int(round(width * display_scale)))
    display_height = max(1, int(round(height * display_scale)))
    display_image = cv2.resize(vis, (display_width, display_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((DISPLAY_HEADER_HEIGHT + display_height, display_width, 3), dtype=np.uint8)
    canvas[:DISPLAY_HEADER_HEIGHT] = np.asarray([28, 28, 28], dtype=np.uint8)
    canvas[DISPLAY_HEADER_HEIGHT:] = display_image

    title = episode_label or "SAM2 bbox selection"
    subtitle = f"camera={camera_label} object={placeholder} mode={prompt_mode}"
    help_text = "m toggle | bbox: drag left | point: left=fg right=bg | p preview | Enter/Space/s save | r reset | q/Esc quit"
    cv2.putText(canvas, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    cv2.putText(canvas, subtitle, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (220, 220, 220), 1)
    cv2.putText(canvas, preview_status, (12, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120, 255, 180), 1)
    cv2.putText(canvas, help_text, (12, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 220, 255), 1)
    return canvas


def select_bbox_for_image(
    label: str,
    placeholder: str,
    image: np.ndarray,
    *,
    episode_label: str = "",
    display_scale: float = DEFAULT_DISPLAY_SCALE,
    preview_tracker_factory: Callable[[Sequence[str], str], object] | None = None,
) -> dict[str, object]:
    drawing = False
    start: tuple[int, int] | None = None
    current: tuple[int, int] | None = None
    bbox: tuple[int, int, int, int] | None = None
    points_xy: list[list[int]] = []
    point_labels: list[int] = []
    prompt_mode = "bbox"
    preview_mask: np.ndarray | None = None
    preview_dirty = False
    preview_error: str | None = None
    preview_status = "preview: draw a prompt; mask will overlay after SAM2 finishes"
    window = f"SAM2 bbox: {label} {placeholder}"
    display_scale = max(1.0, float(display_scale))

    def current_prompt_record() -> dict[str, object] | None:
        if prompt_mode == "bbox":
            normalized = try_normalize_bbox_xyxy(bbox, image.shape[:2])
            if normalized is None:
                return None
            return normalize_prompt_record({"prompt_type": "bbox", "bbox_xyxy": normalized}, image.shape[:2])
        if not points_xy:
            return None
        return normalize_prompt_record(
            {
                "prompt_type": "point",
                "points_xy": points_xy,
                "point_labels": point_labels,
            },
            image.shape[:2],
        )

    def on_mouse(event, x, y, _flags, _param):
        nonlocal drawing, start, current, bbox, preview_dirty
        image_point = _display_to_image_point(
            int(x),
            int(y),
            display_scale=display_scale,
            image_shape_hw=image.shape[:2],
        )
        if image_point is None:
            return
        current = image_point
        if prompt_mode == "point":
            if event == cv2.EVENT_LBUTTONDOWN:
                points_xy.append([int(image_point[0]), int(image_point[1])])
                point_labels.append(1)
                preview_dirty = True
            elif event == cv2.EVENT_RBUTTONDOWN:
                points_xy.append([int(image_point[0]), int(image_point[1])])
                point_labels.append(0)
                preview_dirty = True
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start = image_point
            bbox = None
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            current = image_point
        elif event == cv2.EVENT_LBUTTONUP and drawing and start is not None:
            drawing = False
            candidate = (start[0], start[1], image_point[0], image_point[1])
            if try_normalize_bbox_xyxy(candidate, image.shape[:2]) is None:
                bbox = None
            else:
                bbox = candidate
                preview_dirty = True

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    window_width = max(1, int(round(image.shape[1] * display_scale)))
    window_height = DISPLAY_HEADER_HEIGHT + max(1, int(round(image.shape[0] * display_scale)))
    cv2.resizeWindow(window, window_width, window_height)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        if preview_dirty:
            preview_dirty = False
            prompt = current_prompt_record()
            if prompt is not None and preview_tracker_factory is not None:
                try:
                    tracker = preview_tracker_factory([placeholder], label)
                    preview_mask = tracker.preview_prompt(image, placeholder, prompt)
                    mask_pixels = int(np.asarray(preview_mask).astype(bool).sum())
                    preview_status = f"preview: ok, mask_pixels={mask_pixels}"
                    preview_error = None
                    print(f"SAM2 preview {label} {placeholder}: mask_pixels={mask_pixels}")
                except Exception as exc:
                    preview_mask = None
                    preview_error = str(exc)
                    preview_status = f"preview: failed, {preview_error[:90]}"
                    print(f"SAM2 preview failed for {label} {placeholder}: {exc}")
            elif prompt is not None and preview_tracker_factory is None:
                preview_status = "preview: disabled; prompt will still be saved"
        vis = _render_selection_canvas(
            image,
            display_scale=display_scale,
            episode_label=episode_label,
            camera_label=label,
            placeholder=placeholder,
            prompt_mode=prompt_mode,
            start=start,
            current=current,
            bbox=bbox,
            points_xy=points_xy,
            point_labels=point_labels,
            preview_mask=preview_mask,
            preview_status=preview_status,
            drawing=drawing,
        )
        if preview_error:
            cv2.putText(
                vis,
                f"preview failed: {preview_error[:90]}",
                (12, min(vis.shape[0] - 12, DISPLAY_HEADER_HEIGHT + 24)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 80, 80),
                2,
            )
        cv2.imshow(window, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 32, ord("s")):
            prompt = current_prompt_record()
            if prompt is None:
                continue
            cv2.destroyWindow(window)
            return prompt
        if key in (27, ord("q")):
            cv2.destroyWindow(window)
            raise KeyboardInterrupt("SAM2 bbox selection aborted")
        if key == ord("r"):
            bbox = None
            start = None
            current = None
            points_xy = []
            point_labels = []
            drawing = False
            preview_mask = None
            preview_error = None
            preview_status = "preview: reset; draw a prompt"
            preview_dirty = False
        if key == ord("m"):
            prompt_mode = "point" if prompt_mode == "bbox" else "bbox"
            preview_mask = None
            preview_error = None
            preview_status = f"preview: mode={prompt_mode}; draw a prompt"
            preview_dirty = current_prompt_record() is not None
        if key == ord("p"):
            preview_dirty = current_prompt_record() is not None
            if not preview_dirty:
                preview_status = "preview: no valid prompt yet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactively select first-frame SAM2 tracking bbox/point prompts per camera/object.")
    parser.add_argument("--raw_root", default=DEFAULT_RAW_ROOT)
    parser.add_argument("--raw_episode_dir", default="")
    parser.add_argument("--output_bbox_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--camera_labels", default="global,left,right")
    parser.add_argument("--object_placeholders", default="{A},{B}")
    parser.add_argument("--frame_index", type=int, default=100)
    parser.add_argument("--all_episodes", action="store_true", default=True)
    parser.add_argument("--per_episode_subdir", action="store_true", default=False)
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--display_scale", type=float, default=DEFAULT_DISPLAY_SCALE)
    parser.add_argument("--disable_sam2_preview", action="store_true", default=False)
    parser.add_argument("--sam2_root", default=str(DEFAULT_SAM2_ROOT))
    parser.add_argument("--sam2_config", default=DEFAULT_SAM2_CONFIG)
    parser.add_argument("--sam2_checkpoint", default=str(DEFAULT_SAM2_CHECKPOINT))
    parser.add_argument("--sam2_device", default="cuda")
    parser.add_argument("--sam2_autocast_dtype", default="bfloat16")
    return parser.parse_args()


def select_and_save_episode_bboxes(
    *,
    raw_episode_dir: str | Path,
    output_bbox_root: str | Path,
    camera_labels: Sequence[str],
    placeholders: Sequence[str],
    frame_index: int,
    episode_index: int | None = None,
    episode_total: int | None = None,
    display_scale: float = DEFAULT_DISPLAY_SCALE,
    preview_tracker_factory: Callable[[Sequence[str], str], object] | None = None,
) -> Path:
    raw_episode_dir = Path(raw_episode_dir).expanduser().resolve()
    labels = [str(item) for item in camera_labels]
    manifest = read_json(raw_episode_dir / "manifest.json")
    frames = manifest.get("frames", [])
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"No frames in raw episode: {raw_episode_dir}")
    frame_idx = min(max(0, int(frame_index)), len(frames) - 1)
    frame = frames[frame_idx]
    cameras = frame.get("cameras", {})
    if episode_index is None or episode_total is None:
        episode_label = f"Episode: {raw_episode_dir.name}"
    else:
        episode_label = f"Episode {int(episode_index) + 1}/{int(episode_total)}: {raw_episode_dir.name}"
    episode_label = f"{episode_label} | frame={int(frame.get('frame_index', frame_idx))}"

    image_shapes: dict[str, tuple[int, int]] = {}
    boxes_by_camera: dict[str, dict[str, dict[str, object]]] = {}
    images: dict[str, np.ndarray] = {}
    for label in labels:
        images[label] = _load_rgb(raw_episode_dir, str(cameras[label]))
        image_shapes[label] = images[label].shape[:2]
        boxes_by_camera[label] = {}
        for placeholder in placeholders:
            boxes_by_camera[label][placeholder] = select_bbox_for_image(
                label,
                placeholder,
                images[label],
                episode_label=episode_label,
                display_scale=float(display_scale),
                preview_tracker_factory=preview_tracker_factory,
            )

        overlay = _draw_existing_boxes(images[label], boxes_by_camera[label])
        out_dir = ensure_dir(Path(output_bbox_root) / str(label))
        ok = cv2.imwrite(str(out_dir / "sam2_bbox_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError(f"Failed to save bbox overlay for {label}")

    path = save_sam2_bbox_prompts(
        output_root=output_bbox_root,
        raw_episode_dir=raw_episode_dir,
        frame_index=int(frame.get("frame_index", frame_idx)),
        image_shapes_by_camera=image_shapes,
        boxes_by_camera=boxes_by_camera,
    )
    return path


def main() -> None:
    args = parse_args()
    labels = [item.strip() for item in str(args.camera_labels).split(",") if item.strip()]
    placeholders = [item.strip() for item in str(args.object_placeholders).split(",") if item.strip()]
    if bool(args.all_episodes):
        raw_episode_dirs = find_raw_episode_dirs(args.raw_root)
    else:
        raw_episode_dirs = [_resolve_raw_episode(args.raw_root, args.raw_episode_dir)]

    preview_tracker_cache: dict[tuple[str, tuple[str, ...]], object] = {}

    def preview_tracker_factory(placeholders_for_tracker: Sequence[str], camera_label: str) -> object:
        key = (str(camera_label), tuple(str(item) for item in placeholders_for_tracker))
        if key not in preview_tracker_cache:
            factory = make_sam2_tracker_factory(
                placeholders=list(key[1]),
                sam2_root=args.sam2_root,
                config=args.sam2_config,
                checkpoint=args.sam2_checkpoint,
                device=args.sam2_device,
                autocast_dtype=args.sam2_autocast_dtype,
            )
            preview_tracker_cache[key] = factory(str(camera_label))
        return preview_tracker_cache[key]

    active_preview_factory = None if bool(args.disable_sam2_preview) else preview_tracker_factory

    last_path = None
    episode_total = len(raw_episode_dirs)
    for episode_idx, raw_episode_dir in enumerate(raw_episode_dirs):
        episode_output_root = Path(args.output_bbox_root)
        if bool(args.all_episodes) or bool(args.per_episode_subdir):
            episode_output_root = episode_output_root / raw_episode_dir.name
        prompt_path = episode_output_root / "sam2_bbox_prompts.json"
        if bool(args.skip_existing) and prompt_path.exists():
            print(f"skip existing {prompt_path}")
            last_path = prompt_path
            continue
        print(f"Selecting SAM2 bboxes for {raw_episode_dir} -> {episode_output_root}")
        last_path = select_and_save_episode_bboxes(
            raw_episode_dir=raw_episode_dir,
            output_bbox_root=episode_output_root,
            camera_labels=labels,
            placeholders=placeholders,
            frame_index=int(args.frame_index),
            episode_index=episode_idx,
            episode_total=episode_total,
            display_scale=float(args.display_scale),
            preview_tracker_factory=active_preview_factory,
        )
        print(last_path)

    if last_path is not None:
        print(last_path)


if __name__ == "__main__":
    main()
