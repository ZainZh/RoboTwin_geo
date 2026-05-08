#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np
import os

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
from script.real_zed_collection.select_camera_workspace_masks import load_camera_workspace_masks
from script.real_zed_collection.select_sam2_bboxes import load_sam2_prompt_records
DEFAULT_TASK_NAME = "grasp_mug"

def default_raw_root(task_name: str = DEFAULT_TASK_NAME, *, user: str | None = None) -> Path:
    resolved_user = user or os.environ.get("USER") or os.environ.get("USERNAME") or "zheng"
    return Path("/media") / resolved_user / "Extreme SSD" / "geo_mani_data" / str(task_name) / "real_zed_raw"


def default_bbox_prompt_root(task_name: str = DEFAULT_TASK_NAME, *, user: str | None = None) -> Path:
    resolved_user = user or os.environ.get("USER") or os.environ.get("USERNAME") or "zheng"
    return Path("/media") / resolved_user / "Extreme SSD" / "geo_mani_data" / str(task_name) / "sam2_bbox_prompts"


def default_output_dir(task_name: str = DEFAULT_TASK_NAME, task_config: str | None = None, *, user: str | None = None) -> Path:
    resolved_user = user or os.environ.get("USER") or os.environ.get("USERNAME") or "zheng"
    return Path("/media") / resolved_user / "Extreme SSD" / "geo_mani_data" / str(task_name) / "robotwin_objpc" / str(task_config)


def repo_data_link_path(task_name: str, task_config: str) -> Path:
    return REPO_ROOT / "data" / str(task_name) / str(task_config)

def _load_frame_npz(raw_episode_dir: Path, rel_path: str) -> dict[str, np.ndarray]:
    with np.load(raw_episode_dir / rel_path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def apply_image_domain(image: np.ndarray, domain_mask: np.ndarray | None) -> np.ndarray:
    image_arr = np.asarray(image, dtype=np.uint8)
    if domain_mask is None:
        return image_arr
    domain = np.asarray(domain_mask).astype(bool)
    if domain.shape != image_arr.shape[:2]:
        domain = cv2.resize(domain.astype(np.uint8), (image_arr.shape[1], image_arr.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    out = image_arr.copy()
    out[~domain] = 0
    return out


def apply_mask_domain(mask: np.ndarray, domain_mask: np.ndarray | None) -> np.ndarray:
    mask_arr = np.asarray(mask).astype(bool)
    if domain_mask is None:
        return mask_arr
    domain = np.asarray(domain_mask).astype(bool)
    if domain.shape != mask_arr.shape:
        domain = cv2.resize(domain.astype(np.uint8), (mask_arr.shape[1], mask_arr.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    return mask_arr & domain


def save_mask_overlay(
    *,
    image: np.ndarray,
    masks_by_placeholder: Mapping[str, np.ndarray],
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_arr = np.asarray(image, dtype=np.uint8)
    overlay = image_arr.copy()
    colors = {
        "{A}": np.asarray([255, 48, 48], dtype=np.float32),
        "A": np.asarray([255, 48, 48], dtype=np.float32),
        "{B}": np.asarray([48, 96, 255], dtype=np.float32),
        "B": np.asarray([48, 96, 255], dtype=np.float32),
    }
    for placeholder, mask in masks_by_placeholder.items():
        mask_arr = np.asarray(mask).astype(bool)
        color = colors.get(str(placeholder), np.asarray([255, 255, 0], dtype=np.float32))
        overlay[mask_arr] = (0.55 * overlay[mask_arr].astype(np.float32) + 0.45 * color).clip(0, 255).astype(np.uint8)
    ok = cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"Failed to save SAM2 overlay: {output_path}")
    return output_path


def _empty_masks(placeholders: Sequence[str], image_shape_hw: tuple[int, int]) -> dict[str, np.ndarray]:
    return {
        str(placeholder): np.zeros((int(image_shape_hw[0]), int(image_shape_hw[1])), dtype=bool)
        for placeholder in placeholders
    }


def segment_episode_sam2(
    *,
    raw_episode_dir: str | Path,
    output_mask_root: str | Path,
    bbox_prompts_by_camera: Mapping[str, Mapping[str, object]],
    camera_labels: Sequence[str],
    placeholders: Sequence[str],
    tracker_factory: Callable[[str], Any],
    start_frame: int = 0,
    max_frames: int = -1,
    overwrite: bool = False,
    mask_domain_by_camera: Mapping[str, np.ndarray] | None = None,
    debug_overlay_root: str | Path | None = None,
    debug_frame_indices: set[int] | None = None,
    mask_input_domain: bool = True,
) -> Path:
    raw_episode_dir = Path(raw_episode_dir).expanduser().resolve()
    output_mask_root = ensure_dir(output_mask_root)
    labels = [str(label) for label in camera_labels]
    placeholder_list = [str(placeholder) for placeholder in placeholders]
    manifest = read_json(raw_episode_dir / "manifest.json")
    frames = manifest.get("frames", [])
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"No frames in raw manifest: {raw_episode_dir / 'manifest.json'}")
    frame_start = int(start_frame)
    frame_end = len(frames) if int(max_frames) <= 0 else min(len(frames), frame_start + int(max_frames))
    selected_frames = frames[frame_start:frame_end]
    debug_root = None if debug_overlay_root is None else ensure_dir(debug_overlay_root)

    meta: dict[str, Any] = {
        "tracker": "sam2_streaming",
        "raw_episode_dir": str(raw_episode_dir),
        "camera_labels": labels,
        "placeholders": placeholder_list,
        "prompts_by_camera": {
            label: {
                placeholder: dict(prompt) if isinstance(prompt, Mapping) else {"bbox_xyxy": [int(v) for v in prompt]}
                for placeholder, prompt in per_camera.items()
            }
            for label, per_camera in bbox_prompts_by_camera.items()
        },
        "mask_input_domain": bool(mask_input_domain),
        "frames": {},
    }

    for label in labels:
        camera_prompts = {
            placeholder: bbox_prompts_by_camera.get(label, {}).get(placeholder)
            for placeholder in placeholder_list
            if placeholder in bbox_prompts_by_camera.get(label, {})
        }
        if not camera_prompts:
            raise ValueError(f"No SAM2 prompts for camera {label!r}.")
        tracker = tracker_factory(label)
        initialized = False
        for local_idx, frame in enumerate(selected_frames):
            frame_idx = int(frame["frame_index"])
            cameras = frame.get("cameras", {})
            camera_frame = _load_frame_npz(raw_episode_dir, str(cameras[label]))
            image = np.asarray(camera_frame["rgb"], dtype=np.uint8)
            domain_mask = None if mask_domain_by_camera is None else mask_domain_by_camera.get(label)
            tracker_image = apply_image_domain(image, domain_mask) if bool(mask_input_domain) else image

            if not initialized:
                if hasattr(tracker, "initialize_prompts"):
                    masks = tracker.initialize_prompts(tracker_image, camera_prompts)
                else:
                    masks = tracker.initialize(tracker_image, camera_prompts)
                initialized = True
                mode = "initialized"
            else:
                masks = tracker.track(tracker_image)
                mode = "tracked"
            if not masks:
                masks = _empty_masks(placeholder_list, image.shape[:2])

            frame_meta = meta["frames"].setdefault(str(frame_idx), {})
            overlay_masks: dict[str, np.ndarray] = {}
            for placeholder in placeholder_list:
                mask = apply_mask_domain(masks.get(placeholder, np.zeros(image.shape[:2], dtype=bool)), domain_mask)
                out_dir = ensure_dir(output_mask_root / placeholder / label)
                out_path = out_dir / f"mask_{frame_idx:06d}.png"
                if not out_path.exists() or bool(overwrite):
                    ok = cv2.imwrite(str(out_path), mask.astype(np.uint8) * 255)
                    if not ok:
                        raise RuntimeError(f"Failed to save SAM2 mask: {out_path}")
                overlay_masks[placeholder] = mask
                frame_meta.setdefault(placeholder, {})[label] = {
                    "mode": mode,
                    "path": str(out_path),
                    "mask_pixels": int(mask.sum()),
                    "mask_domain": domain_mask is not None,
                }

            if debug_root is not None and (debug_frame_indices is None or int(frame_idx) in debug_frame_indices):
                save_mask_overlay(
                    image=image,
                    masks_by_placeholder=overlay_masks,
                    output_path=debug_root / label / f"overlay_{frame_idx:06d}.png",
                )

    write_json(output_mask_root / "sam2_mask_meta.json", meta)
    return output_mask_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate real-ZED per-object masks using bbox/point-initialized SAM2 streaming tracking.")
    parser.add_argument("--raw_episode_dir", required=True)
    parser.add_argument("--output_mask_root", required=True)
    parser.add_argument("--bbox_prompt_root", required=True)
    parser.add_argument("--camera_labels", default="global,left,right")
    parser.add_argument("--object_placeholders", default="{A},{B}")
    parser.add_argument("--sam2_root", default=str(DEFAULT_SAM2_ROOT))
    parser.add_argument("--sam2_config", default=DEFAULT_SAM2_CONFIG)
    parser.add_argument("--sam2_checkpoint", default=str(DEFAULT_SAM2_CHECKPOINT))
    parser.add_argument("--sam2_device", default="cuda")
    parser.add_argument("--sam2_autocast_dtype", default="bfloat16")
    parser.add_argument("--camera_workspace_mask_root", default="")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--debug_overlay_root", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = [item.strip() for item in str(args.camera_labels).split(",") if item.strip()]
    placeholders = [item.strip() for item in str(args.object_placeholders).split(",") if item.strip()]
    prompts = load_sam2_prompt_records(args.bbox_prompt_root, camera_labels=labels, placeholders=placeholders)
    domains = (
        {}
        if not str(args.camera_workspace_mask_root).strip()
        else load_camera_workspace_masks(args.camera_workspace_mask_root, labels)
    )
    tracker_factory = make_sam2_tracker_factory(
        placeholders=placeholders,
        sam2_root=args.sam2_root,
        config=args.sam2_config,
        checkpoint=args.sam2_checkpoint,
        device=args.sam2_device,
        autocast_dtype=args.sam2_autocast_dtype,
    )
    out = segment_episode_sam2(
        raw_episode_dir=args.raw_episode_dir,
        output_mask_root=args.output_mask_root,
        bbox_prompts_by_camera=prompts,
        camera_labels=labels,
        placeholders=placeholders,
        tracker_factory=tracker_factory,
        start_frame=int(args.start_frame),
        max_frames=int(args.max_frames),
        overwrite=bool(args.overwrite),
        mask_domain_by_camera=domains,
        debug_overlay_root=args.debug_overlay_root or None,
    )
    print(out)


if __name__ == "__main__":
    main()
