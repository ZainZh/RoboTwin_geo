#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from script.real_zed_collection.real_zed_utils import ensure_dir, read_json, write_json


def rasterize_polygon_mask(image_shape_hw: tuple[int, int], points_xy: Sequence[Sequence[float]]) -> np.ndarray:
    height, width = int(image_shape_hw[0]), int(image_shape_hw[1])
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(points_xy) < 3:
        return mask.astype(bool)
    pts = np.asarray(points_xy, dtype=np.float32)
    pts[:, 0] = np.clip(pts[:, 0], 0, max(0, width - 1))
    pts[:, 1] = np.clip(pts[:, 1], 0, max(0, height - 1))
    cv2.fillPoly(mask, [np.rint(pts).astype(np.int32)], 255)
    return mask.astype(bool)


def load_camera_workspace_masks(mask_root: str | Path, camera_labels: Sequence[str] | None = None) -> dict[str, np.ndarray]:
    root = Path(mask_root).expanduser().resolve()
    labels = [str(label) for label in (camera_labels or [])]
    if not labels:
        labels = sorted(path.stem for path in root.glob("*.png"))
        labels += sorted(path.name for path in root.iterdir() if path.is_dir())
    out: dict[str, np.ndarray] = {}
    for label in labels:
        candidates = [
            root / str(label) / "workspace_mask.png",
            root / f"{label}.png",
            root / str(label) / "mask.png",
        ]
        for path in candidates:
            if not path.exists():
                continue
            raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if raw is None:
                raise RuntimeError(f"Failed to read workspace mask: {path}")
            out[str(label)] = raw > 0
            break
    return out


def _load_rgb(raw_episode_dir: Path, rel_path: str) -> np.ndarray:
    with np.load(raw_episode_dir / rel_path, allow_pickle=False) as data:
        return np.asarray(data["rgb"], dtype=np.uint8)


def _resolve_raw_episode(raw_root: str | Path = "", raw_episode_dir: str | Path = "") -> Path:
    if str(raw_episode_dir).strip():
        path = Path(raw_episode_dir).expanduser().resolve()
        if not (path / "manifest.json").exists():
            raise FileNotFoundError(f"Raw episode has no manifest.json: {path}")
        return path
    root = Path(raw_root).expanduser().resolve()
    episodes = sorted(path for path in root.glob("episode_*") if (path / "manifest.json").exists())
    if not episodes:
        raise FileNotFoundError(f"No raw episodes found under: {root}")
    return episodes[0]


def _draw_polygon(image: np.ndarray, points: list[tuple[int, int]], cursor: tuple[int, int] | None = None) -> np.ndarray:
    vis = image.copy()
    overlay = vis.copy()
    if len(points) >= 3:
        mask = rasterize_polygon_mask(vis.shape[:2], points)
        overlay[mask] = (0.55 * overlay[mask].astype(np.float32) + 0.45 * np.array([255, 120, 20])).astype(np.uint8)
        vis = overlay
    for idx, point in enumerate(points):
        cv2.circle(vis, point, 4, (0, 255, 255), -1)
        cv2.putText(vis, str(idx), (point[0] + 5, point[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    if len(points) >= 2:
        cv2.polylines(vis, [np.asarray(points, dtype=np.int32)], isClosed=False, color=(0, 255, 255), thickness=2)
    if cursor is not None and len(points) >= 1:
        cv2.line(vis, points[-1], cursor, (120, 255, 120), 1)
    if len(points) >= 3:
        cv2.line(vis, points[-1], points[0], (0, 180, 255), 1)
    help_lines = [
        "Left click: add point | u/right click: undo | r: reset",
        "Enter/Space/s: save camera | q/Esc: quit",
    ]
    y = 20
    for text in help_lines:
        cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
        cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1)
        y += 22
    return vis


def select_polygon_for_image(label: str, image: np.ndarray) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    cursor: tuple[int, int] | None = None
    window = f"workspace mask: {label}"

    def on_mouse(event, x, y, _flags, _param):
        nonlocal cursor
        cursor = (int(x), int(y))
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((int(x), int(y)))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        cv2.imshow(window, cv2.cvtColor(_draw_polygon(image, points, cursor), cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 32, ord("s")):
            if len(points) < 3:
                print(f"{label}: need at least 3 points before saving.")
                continue
            break
        if key in (27, ord("q")):
            cv2.destroyWindow(window)
            raise KeyboardInterrupt("workspace mask selection aborted")
        if key == ord("u") and points:
            points.pop()
        if key == ord("r"):
            points.clear()
    cv2.destroyWindow(window)
    return points


def save_camera_workspace_mask(
    *,
    output_root: str | Path,
    camera_label: str,
    image: np.ndarray,
    points_xy: Sequence[Sequence[float]],
) -> dict:
    output_root = ensure_dir(output_root)
    camera_dir = ensure_dir(output_root / str(camera_label))
    mask = rasterize_polygon_mask(image.shape[:2], points_xy)
    mask_path = camera_dir / "workspace_mask.png"
    overlay_path = camera_dir / "workspace_overlay.png"
    ok = cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)
    if not ok:
        raise RuntimeError(f"Failed to save workspace mask: {mask_path}")
    overlay = _draw_polygon(image, [(int(round(x)), int(round(y))) for x, y in points_xy])
    ok = cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"Failed to save workspace overlay: {overlay_path}")
    return {
        "camera_label": str(camera_label),
        "image_shape_hw": [int(image.shape[0]), int(image.shape[1])],
        "points_xy": [[float(x), float(y)] for x, y in points_xy],
        "mask_path": str(mask_path),
        "overlay_path": str(overlay_path),
        "mask_pixels": int(mask.sum()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactively click per-camera 2D workspace polygons on the first raw frame.")
    parser.add_argument("--raw_root", default="/media/zheng/Extreme SSD/geo_mani_data/grasp_mug/real_zed_raw")
    parser.add_argument("--raw_episode_dir", default="")
    parser.add_argument("--output_mask_root", default="/media/zheng/Extreme SSD/geo_mani_data/grasp_mug/camera_workspace_masks" )
    parser.add_argument("--camera_labels", default="global,left,right")
    parser.add_argument("--frame_index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_episode_dir = _resolve_raw_episode(args.raw_root, args.raw_episode_dir)
    labels = [item.strip() for item in str(args.camera_labels).split(",") if item.strip()]
    manifest = read_json(raw_episode_dir / "manifest.json")
    frames = manifest.get("frames", [])
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"No frames in raw episode: {raw_episode_dir}")
    frame_idx = min(max(0, int(args.frame_index)), len(frames) - 1)
    frame = frames[frame_idx]
    cameras = frame.get("cameras", {})

    output_root = ensure_dir(args.output_mask_root)
    records = {}
    for label in labels:
        image = _load_rgb(raw_episode_dir, str(cameras[label]))
        points = select_polygon_for_image(label, image)
        records[label] = save_camera_workspace_mask(
            output_root=output_root,
            camera_label=label,
            image=image,
            points_xy=points,
        )
        print(f"{label}: saved {records[label]['mask_path']} pixels={records[label]['mask_pixels']}")

    meta = {
        "raw_episode_dir": str(raw_episode_dir),
        "frame_index": int(frame.get("frame_index", frame_idx)),
        "camera_labels": labels,
        "records": records,
    }
    write_json(output_root / "workspace_masks_meta.json", meta)
    print(output_root)


if __name__ == "__main__":
    main()
