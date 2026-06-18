#!/usr/bin/env python3

from __future__ import annotations

import argparse
import getpass
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from script.real_zed_collection.real_zed_utils import ensure_dir, read_json, write_json

try:
    import cv2
except Exception:  # pragma: no cover - import availability depends on runtime image stack.
    cv2 = None


def default_raw_root(task_name: str, *, user: str | None = None) -> Path:
    resolved_user = user or getpass.getuser()
    return Path("/media") / resolved_user / "Extreme SSD" / "geo_mani_data" / str(task_name) / "real_zed_raw"


def parse_csv(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def _episode_dirs(raw_root: Path) -> list[Path]:
    return sorted(path for path in raw_root.glob("episode_*") if (path / "manifest.json").exists())


def resolve_raw_episode_dir(task_name: str, episode: str | int, raw_root: str | Path | None = None) -> Path:
    root = Path(raw_root).expanduser() if raw_root is not None else default_raw_root(task_name)
    episode_text = str(episode).strip()
    episode_path = Path(episode_text).expanduser()
    if episode_path.exists():
        return episode_path.resolve()

    if not root.exists():
        raise FileNotFoundError(f"Raw root does not exist: {root}")

    if episode_text.isdigit():
        episodes = _episode_dirs(root)
        episode_idx = int(episode_text)
        if episode_idx < 0 or episode_idx >= len(episodes):
            raise IndexError(f"Episode index {episode_idx} out of range. Found {len(episodes)} episodes under {root}")
        return episodes[episode_idx].resolve()

    candidates = [root / episode_text]
    if not episode_text.startswith("episode_"):
        candidates.append(root / f"episode_{episode_text}")
    for candidate in candidates:
        if (candidate / "manifest.json").exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve episode {episode_text!r} under {root}")


def _resolve_camera_labels(manifest: dict[str, Any], requested_labels: Sequence[str] | None) -> list[str]:
    labels = [str(label) for label in (requested_labels or []) if str(label)]
    if labels:
        return labels
    manifest_labels = manifest.get("camera_labels", [])
    if isinstance(manifest_labels, list) and manifest_labels:
        return [str(label) for label in manifest_labels]
    frames = manifest.get("frames", [])
    if frames:
        cameras = frames[0].get("cameras", {})
        if isinstance(cameras, dict):
            return sorted(str(label) for label in cameras)
    raise ValueError("Cannot infer camera labels from manifest; pass --camera_labels.")


def _load_rgb(raw_episode_dir: Path, rel_path: str, rgb_key: str) -> np.ndarray:
    path = raw_episode_dir / rel_path
    if not path.exists():
        raise FileNotFoundError(f"Camera frame file does not exist: {path}")
    with np.load(path, allow_pickle=False) as data:
        if rgb_key not in data.files:
            raise KeyError(f"Camera frame {path} does not contain key {rgb_key!r}. Available keys: {data.files}")
        rgb = np.asarray(data[rgb_key], dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"{rgb_key} must be HxWx3 RGB, got {rgb.shape} from {path}")
    return rgb[:, :, :3]


def _write_png(path: Path, rgb: np.ndarray) -> None:
    if cv2 is None:
        raise RuntimeError("cv2 is required to write PNG images.")
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(path), bgr)
    if not ok:
        raise RuntimeError(f"Failed to write PNG: {path}")


def export_raw_rgb_images(
    *,
    raw_episode_dir: str | Path,
    output_dir: str | Path,
    camera_labels: Sequence[str] | None = None,
    frame_stride: int = 1,
    max_frames: int | None = None,
    rgb_key: str = "rgb",
) -> dict[str, Any]:
    raw_episode = Path(raw_episode_dir).expanduser().resolve()
    manifest = read_json(raw_episode / "manifest.json")
    frames = manifest.get("frames", [])
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"No frames found in {raw_episode / 'manifest.json'}")
    labels = _resolve_camera_labels(manifest, camera_labels)
    stride = max(1, int(frame_stride))
    limit = None if max_frames is None or int(max_frames) <= 0 else int(max_frames)
    out_root = ensure_dir(output_dir)

    saved_count = 0
    saved_by_camera = {label: 0 for label in labels}
    selected_frames = frames[::stride]
    if limit is not None:
        selected_frames = selected_frames[:limit]

    for frame in selected_frames:
        frame_index = int(frame.get("frame_index", saved_count))
        cameras = frame.get("cameras", {})
        if not isinstance(cameras, dict):
            raise ValueError(f"Invalid cameras record for frame {frame_index}: {cameras!r}")
        for label in labels:
            if label not in cameras:
                raise KeyError(f"Frame {frame_index} has no camera {label!r}. Available: {sorted(cameras)}")
            rgb = _load_rgb(raw_episode, str(cameras[label]), rgb_key=rgb_key)
            _write_png(out_root / label / f"frame_{frame_index:06d}.png", rgb)
            saved_count += 1
            saved_by_camera[label] += 1

    summary = {
        "raw_episode_dir": str(raw_episode),
        "output_dir": str(out_root),
        "camera_labels": labels,
        "rgb_key": str(rgb_key),
        "frame_stride": stride,
        "max_frames": limit,
        "saved_count": saved_count,
        "saved_by_camera": saved_by_camera,
    }
    write_json(out_root / "export_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export saved raw real-ZED RGB frames for one task episode.")
    parser.add_argument("--task_name", default="view_stirring_coffee",help="Task/project name, e.g. grasp_mug.")
    parser.add_argument("--episode",default="2", help="Episode index, episode directory name, timestamp, or full episode path.")
    parser.add_argument("--raw_root", default="", help="Raw root. Defaults to /media/$USER/Extreme SSD/geo_mani_data/<task>/real_zed_raw.")
    parser.add_argument("--output_dir", default="", help="Output directory. Defaults under outputs/real_zed_collection/raw_rgb/.")
    parser.add_argument("--camera_labels", default="", help="Comma-separated camera labels. Empty means manifest camera_labels.")
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=0, help="0 means export all selected frames.")
    parser.add_argument("--rgb_key", default="rgb", choices=["rgb", "full_rgb_debug"], help="NPZ RGB key to export.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root or None
    raw_episode_dir = resolve_raw_episode_dir(args.task_name, args.episode, raw_root=raw_root)
    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else Path("outputs") / "real_zed_collection" / "raw_rgb" / args.task_name / raw_episode_dir.name
    )
    summary = export_raw_rgb_images(
        raw_episode_dir=raw_episode_dir,
        output_dir=output_dir,
        camera_labels=parse_csv(args.camera_labels),
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        rgb_key=args.rgb_key,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
