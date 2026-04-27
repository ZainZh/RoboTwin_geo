#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np

from script.real_zed_collection.real_zed_utils import ensure_dir, read_json, write_json
from script.real_zed_collection.select_camera_workspace_masks import _load_rgb, _resolve_raw_episode


DEFAULT_RAW_ROOT = "/media/zheng/Extreme SSD/geo_mani_data/grasp_mug/real_zed_raw"
DEFAULT_OUTPUT_ROOT = "/media/zheng/Extreme SSD/geo_mani_data/grasp_mug/sam2_bbox_prompts"


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


def save_sam2_bbox_prompts(
    *,
    output_root: str | Path,
    raw_episode_dir: str | Path,
    frame_index: int,
    image_shapes_by_camera: Mapping[str, Sequence[int]],
    boxes_by_camera: Mapping[str, Mapping[str, Sequence[float]]],
) -> Path:
    output_root = ensure_dir(output_root)
    records: dict[str, dict[str, dict[str, object]]] = {}
    for camera_label, per_placeholder in boxes_by_camera.items():
        shape = tuple(int(v) for v in image_shapes_by_camera[camera_label][:2])
        records[str(camera_label)] = {}
        for placeholder, bbox in per_placeholder.items():
            records[str(camera_label)][str(placeholder)] = {
                "bbox_xyxy": normalize_bbox_xyxy(bbox, shape),
                "image_shape_hw": [int(shape[0]), int(shape[1])],
            }
    payload = {
        "raw_episode_dir": str(raw_episode_dir),
        "frame_index": int(frame_index),
        "camera_labels": sorted(records.keys()),
        "records": records,
    }
    path = output_root / "sam2_bbox_prompts.json"
    write_json(path, payload)
    return path


def load_sam2_bbox_prompts(
    path_or_root: str | Path,
    *,
    camera_labels: Sequence[str] | None = None,
    placeholders: Sequence[str] | None = None,
) -> dict[str, dict[str, list[int]]]:
    path = Path(path_or_root).expanduser().resolve()
    if path.is_dir():
        path = path / "sam2_bbox_prompts.json"
    data = read_json(path)
    records = data.get("records", {})
    if not isinstance(records, dict):
        raise ValueError(f"Invalid SAM2 bbox prompt file: {path}")
    labels = [str(item) for item in (camera_labels or records.keys())]
    placeholder_filter = None if placeholders is None else {str(item) for item in placeholders}
    out: dict[str, dict[str, list[int]]] = {}
    for label in labels:
        per_camera = records.get(label, {})
        if not isinstance(per_camera, dict):
            continue
        out[label] = {}
        for placeholder, item in per_camera.items():
            if placeholder_filter is not None and str(placeholder) not in placeholder_filter:
                continue
            bbox = item.get("bbox_xyxy") if isinstance(item, dict) else None
            if bbox is None:
                continue
            out[label][str(placeholder)] = [int(v) for v in bbox]
    return out


def find_raw_episode_dirs(raw_root: str | Path) -> list[Path]:
    root = Path(raw_root).expanduser().resolve()
    episodes = sorted(path for path in root.glob("episode_*") if (path / "manifest.json").exists())
    if not episodes:
        raise FileNotFoundError(f"No raw episodes found under: {root}")
    return episodes


def _draw_existing_boxes(image: np.ndarray, boxes: Mapping[str, Sequence[int]]) -> np.ndarray:
    vis = image.copy()
    colors = {
        "{A}": (255, 48, 48),
        "A": (255, 48, 48),
        "{B}": (48, 96, 255),
        "B": (48, 96, 255),
    }
    for placeholder, bbox in boxes.items():
        x0, y0, x1, y1 = [int(v) for v in bbox]
        color = colors.get(str(placeholder), (255, 255, 0))
        cv2.rectangle(vis, (x0, y0), (x1, y1), color=(int(color[2]), int(color[1]), int(color[0])), thickness=2)
        cv2.putText(vis, str(placeholder), (x0 + 4, max(16, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return vis


def select_bbox_for_image(label: str, placeholder: str, image: np.ndarray) -> list[int]:
    drawing = False
    start: tuple[int, int] | None = None
    current: tuple[int, int] | None = None
    bbox: tuple[int, int, int, int] | None = None
    window = f"SAM2 bbox: {label} {placeholder}"

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
            x0, y0, x1, y1 = normalize_bbox_xyxy(bbox, image.shape[:2])
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 220, 255), 2)
        cv2.putText(
            vis,
            f"Draw bbox for {placeholder} in {label}. Enter/Space/s save, r reset, q/Esc quit",
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
            return normalize_bbox_xyxy(bbox, image.shape[:2])
        if key in (27, ord("q")):
            cv2.destroyWindow(window)
            raise KeyboardInterrupt("SAM2 bbox selection aborted")
        if key == ord("r"):
            bbox = None
            start = None
            current = None
            drawing = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactively select first-frame SAM2 tracking bboxes per camera/object.")
    parser.add_argument("--raw_root", default=DEFAULT_RAW_ROOT)
    parser.add_argument("--raw_episode_dir", default="")
    parser.add_argument("--output_bbox_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--camera_labels", default="global,left,right")
    parser.add_argument("--object_placeholders", default="{A},{B}")
    parser.add_argument("--frame_index", type=int, default=0)
    parser.add_argument("--all_episodes", action="store_true", default=False)
    parser.add_argument("--per_episode_subdir", action="store_true", default=False)
    parser.add_argument("--skip_existing", action="store_true", default=False)
    return parser.parse_args()


def select_and_save_episode_bboxes(
    *,
    raw_episode_dir: str | Path,
    output_bbox_root: str | Path,
    camera_labels: Sequence[str],
    placeholders: Sequence[str],
    frame_index: int,
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

    image_shapes: dict[str, tuple[int, int]] = {}
    boxes_by_camera: dict[str, dict[str, list[int]]] = {}
    images: dict[str, np.ndarray] = {}
    for label in labels:
        images[label] = _load_rgb(raw_episode_dir, str(cameras[label]))
        image_shapes[label] = images[label].shape[:2]
        boxes_by_camera[label] = {}
        for placeholder in placeholders:
            boxes_by_camera[label][placeholder] = select_bbox_for_image(label, placeholder, images[label])

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

    last_path = None
    for raw_episode_dir in raw_episode_dirs:
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
        )
        print(last_path)

    if last_path is not None:
        print(last_path)


if __name__ == "__main__":
    main()
