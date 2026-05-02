#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from script.real_zed_collection.postprocess.postprocess_raw_to_robotwin_hdf5 import (
    _resolve_calibration_path,
    parse_object_prompts,
    postprocess_episode,
)
from script.real_zed_collection.real_zed_utils import ensure_dir, read_json, write_json
from script.real_zed_collection.sam2_tracking_utils import (
    DEFAULT_SAM2_CHECKPOINT,
    DEFAULT_SAM2_CONFIG,
    DEFAULT_SAM2_ROOT,
    make_sam2_tracker_factory,
)
from script.real_zed_collection.segment_objects_sam2 import segment_episode_sam2
from script.real_zed_collection.select_camera_workspace_masks import load_camera_workspace_masks
from script.real_zed_collection.select_sam2_bboxes import load_sam2_prompt_records
from script.real_zed_collection.workspace_crop_utils import WorkspaceBounds


DEFAULT_TASK_NAME = "grasp_mug"
DEFAULT_TASK_CONFIG = "demo_real_zed_sam2_objpc"
DEFAULT_OBJECT_PROMPTS = "{A}:cup,{B}:box"
DEFAULT_CAMERA_LABELS = "global,left,right"
REPO_ROOT = Path(__file__).resolve().parents[3]
AUTO_VALUE_STRINGS = {"", "auto", "manifest"}


def default_raw_root(task_name: str = DEFAULT_TASK_NAME, *, user: str | None = None) -> Path:
    resolved_user = user or os.environ.get("USER") or os.environ.get("USERNAME") or "zheng"
    return Path("/media") / resolved_user / "Extreme SSD" / "geo_mani_data" / str(task_name) / "real_zed_raw"


def default_bbox_prompt_root(task_name: str = DEFAULT_TASK_NAME, *, user: str | None = None) -> Path:
    resolved_user = user or os.environ.get("USER") or os.environ.get("USERNAME") or "zheng"
    return Path("/media") / resolved_user / "Extreme SSD" / "geo_mani_data" / str(task_name) / "sam2_bbox_prompts"


def default_output_dir(task_name: str, task_config: str, *, user: str | None = None) -> Path:
    resolved_user = user or os.environ.get("USER") or os.environ.get("USERNAME") or "zheng"
    return Path("/media") / resolved_user / "Extreme SSD" / "geo_mani_data" / str(task_name) / "robotwin_objpc" / str(task_config)


def repo_data_link_path(task_name: str, task_config: str) -> Path:
    return REPO_ROOT / "data" / str(task_name) / str(task_config)


def ensure_repo_data_link(
    *,
    output_dir: str | Path,
    task_name: str,
    task_config: str,
    enabled: bool = True,
    overwrite: bool = False,
) -> Path | None:
    if not bool(enabled):
        return None
    target = Path(output_dir).expanduser().resolve()
    link_path = repo_data_link_path(task_name, task_config)
    if link_path.exists() and target == link_path.resolve():
        return link_path
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink():
        current = link_path.resolve()
        if current == target:
            return link_path
        if not bool(overwrite):
            raise FileExistsError(f"Repo data link already points to {current}; pass --overwrite_repo_link to replace it.")
        link_path.unlink()
    elif link_path.exists():
        if not bool(overwrite):
            raise FileExistsError(
                f"Repo data path already exists and is not a symlink: {link_path}. "
                "Move it away or pass --overwrite_repo_link."
            )
        if link_path.is_dir():
            shutil.rmtree(link_path)
        else:
            link_path.unlink()
    link_path.symlink_to(target, target_is_directory=True)
    return link_path


def find_raw_episode_dirs(
    raw_root: str | Path,
    *,
    start_episode: int = 0,
    max_episodes: int = -1,
) -> list[Path]:
    root = Path(raw_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Raw root does not exist: {root}")
    episodes = sorted(path for path in root.glob("episode_*") if (path / "manifest.json").exists())
    if not episodes:
        raise FileNotFoundError(f"No raw episode manifests found under: {root}")
    selected = episodes[max(0, int(start_episode)) :]
    if int(max_episodes) > 0:
        selected = selected[: int(max_episodes)]
    return selected


def resolve_episode_bbox_prompts(
    *,
    bbox_prompt_root: str | Path,
    raw_episode_dir: str | Path,
    episode_index: int,
    camera_labels: list[str],
    placeholders: list[str],
    require_per_episode: bool = False,
) -> tuple[dict[str, dict[str, dict[str, object]]], str]:
    root = Path(bbox_prompt_root).expanduser().resolve()
    raw_episode_dir = Path(raw_episode_dir).expanduser().resolve()
    per_episode_candidates = [
        root / raw_episode_dir.name,
        root / f"episode{int(episode_index)}",
        root / f"episode_{int(episode_index):06d}",
    ]
    for candidate in per_episode_candidates:
        prompt_file = candidate / "sam2_bbox_prompts.json"
        if prompt_file.exists():
            return (
                load_sam2_prompt_records(prompt_file, camera_labels=camera_labels, placeholders=placeholders),
                str(prompt_file),
            )

    global_prompt_file = root / "sam2_bbox_prompts.json"
    if bool(require_per_episode):
        raise FileNotFoundError(
            "Per-episode SAM2 bbox prompts are required but missing. Expected one of: "
            + ", ".join(str(path / "sam2_bbox_prompts.json") for path in per_episode_candidates)
        )
    if global_prompt_file.exists():
        return (
            load_sam2_prompt_records(global_prompt_file, camera_labels=camera_labels, placeholders=placeholders),
            str(global_prompt_file),
        )
    raise FileNotFoundError(
        "No SAM2 bbox prompts found. Expected one of: "
        + ", ".join(str(path / "sam2_bbox_prompts.json") for path in per_episode_candidates)
        + f", or fallback {global_prompt_file}"
    )


def workspace_bounds_from_calibration(calibration_path: str | Path) -> WorkspaceBounds | None:
    path = Path(calibration_path).expanduser()
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        return None
    workspace = cfg.get("workspace", cfg.get("workspace_frame", {}))
    if not isinstance(workspace, dict):
        return None
    bbox = workspace.get("bbox_m", {})
    if not isinstance(bbox, dict):
        return None
    keys = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
    if not all(key in bbox for key in keys):
        return None
    return WorkspaceBounds(
        x_min=float(bbox["x_min"]),
        x_max=float(bbox["x_max"]),
        y_min=float(bbox["y_min"]),
        y_max=float(bbox["y_max"]),
        z_min=float(bbox["z_min"]),
        z_max=float(bbox["z_max"]),
    )


def is_auto_value(value: object) -> bool:
    return str(value or "").strip().lower() in AUTO_VALUE_STRINGS


def _manifest_path(raw_episode_dir: str | Path, value: object) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = Path(raw_episode_dir).expanduser() / path
    return path.resolve()


def calibration_file_has_workspace(calibration_path: str | Path) -> bool:
    path = Path(calibration_path).expanduser()
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        return False
    return isinstance(cfg.get("workspace", cfg.get("workspace_frame", None)), dict)


def _workspace_calibration_candidates(raw_episode_dir: str | Path, manifest: dict[str, Any]) -> list[Path]:
    raw_episode_dir = Path(raw_episode_dir).expanduser().resolve()
    candidates: list[Path] = []
    keys = [
        "workspace_calibration_snapshot_path",
        "workspace_calibration_path",
        "calibration_snapshot_path",
    ]
    if not str(manifest.get("calibration_snapshot_path", "")).strip():
        keys.append("calibration_path")
    for key in keys:
        path = _manifest_path(raw_episode_dir, manifest.get(key, ""))
        if path is not None and path.exists() and calibration_file_has_workspace(path):
            candidates.append(path)
    return candidates


def resolve_episode_postprocess_settings(
    *,
    raw_episode_dir: str | Path,
    manifest: dict[str, Any],
    calibration_path: str | Path,
    frame_mode: str,
) -> dict[str, str]:
    raw_episode_dir = Path(raw_episode_dir).expanduser().resolve()
    auto_calibration = is_auto_value(calibration_path)
    requested_frame_mode = str(frame_mode or "auto").strip()
    if requested_frame_mode not in {"auto", "reference_camera", "workspace"}:
        raise ValueError("frame_mode must be one of: auto, reference_camera, workspace")

    actual_frame_mode = requested_frame_mode
    workspace_candidates = _workspace_calibration_candidates(raw_episode_dir, manifest) if auto_calibration else []
    if requested_frame_mode == "auto":
        if auto_calibration:
            actual_frame_mode = "workspace" if workspace_candidates else "reference_camera"
        else:
            actual_frame_mode = (
                "workspace"
                if calibration_file_has_workspace(calibration_path)
                else "reference_camera"
            )
    elif requested_frame_mode == "workspace" and auto_calibration and not workspace_candidates:
        # Collection-time calibration is more important than forcing a stale repo workspace file.
        actual_frame_mode = "reference_camera"

    resolved_path = _resolve_calibration_path(
        raw_episode_dir,
        manifest,
        "" if auto_calibration else calibration_path,
        frame_mode=actual_frame_mode,
    )
    return {
        "calibration_path": str(resolved_path),
        "frame_mode": str(actual_frame_mode),
    }


def workspace_bounds_from_args(
    args: argparse.Namespace,
    *,
    calibration_path: str | Path | None = None,
) -> WorkspaceBounds | None:
    values = [
        args.workspace_crop_x_min,
        args.workspace_crop_x_max,
        args.workspace_crop_y_min,
        args.workspace_crop_y_max,
        args.workspace_crop_z_min,
        args.workspace_crop_z_max,
    ]
    if any(value is not None for value in values):
        if any(value is None for value in values):
            raise ValueError("All six workspace crop bounds must be provided together.")
        return WorkspaceBounds(
            x_min=float(args.workspace_crop_x_min),
            x_max=float(args.workspace_crop_x_max),
            y_min=float(args.workspace_crop_y_min),
            y_max=float(args.workspace_crop_y_max),
            z_min=float(args.workspace_crop_z_min),
            z_max=float(args.workspace_crop_z_max),
        )
    if bool(args.disable_workspace_crop):
        return None
    selected_calibration_path = calibration_path if calibration_path is not None else args.calibration_path
    if is_auto_value(selected_calibration_path):
        return None
    bounds = workspace_bounds_from_calibration(selected_calibration_path)
    if bounds is not None:
        return bounds
    return None


def selected_debug_frame_indices(num_frames: int, *, stride: int, max_frames: int) -> list[int]:
    if int(num_frames) <= 0:
        return []
    selected = list(range(0, int(num_frames), max(1, int(stride))))
    return selected[: int(max_frames)] if int(max_frames) > 0 else selected


def _clean_debug_cloud(point_cloud: np.ndarray) -> np.ndarray:
    pc = np.asarray(point_cloud, dtype=np.float32)
    if pc.ndim != 2 or pc.shape[1] < 3:
        return np.zeros((0, 3), dtype=np.float32)
    xyz = pc[:, :3]
    mask = np.isfinite(xyz).all(axis=1) & (~np.isclose(xyz, 0.0).all(axis=1))
    return xyz[mask].astype(np.float32)


def write_colored_point_cloud_ply(
    path: str | Path,
    colored_clouds: list[tuple[np.ndarray, tuple[int, int, int]]],
) -> Path:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[float, float, float, int, int, int]] = []
    for cloud, color in colored_clouds:
        xyz = _clean_debug_cloud(cloud)
        r, g, b = [int(v) for v in color]
        rows.extend((float(x), float(y), float(z), r, g, b) for x, y, z in xyz)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(rows)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for x, y, z, r, g, b in rows:
            f.write(f"{x:.8f} {y:.8f} {z:.8f} {r} {g} {b}\n")
    return path


def save_debug_object_pointclouds(
    *,
    hdf5_path: str | Path,
    debug_root: str | Path,
    episode_index: int,
    placeholders: list[str],
    stride: int,
    max_frames: int,
) -> list[str]:
    debug_dir = Path(debug_root).expanduser().resolve() / f"episode{int(episode_index)}" / "pointclouds"
    color_by_placeholder = {
        "{A}": (255, 48, 48),
        "A": (255, 48, 48),
        "{B}": (48, 96, 255),
        "B": (48, 96, 255),
    }
    written: list[str] = []
    with h5py.File(hdf5_path, "r") as root:
        obj_group = root.get("object_pointcloud")
        if obj_group is None:
            return written
        first_key = next(iter(obj_group.keys()), None)
        if first_key is None:
            return written
        frame_count = int(obj_group[first_key].shape[0])
        for frame_idx in selected_debug_frame_indices(frame_count, stride=int(stride), max_frames=int(max_frames)):
            colored_clouds = []
            for placeholder in placeholders:
                if placeholder not in obj_group:
                    continue
                color = color_by_placeholder.get(placeholder, (220, 220, 220))
                colored_clouds.append((obj_group[placeholder][frame_idx], color))
            if not colored_clouds:
                continue
            path = write_colored_point_cloud_ply(debug_dir / f"frame_{frame_idx:06d}_objects_ab.ply", colored_clouds)
            written.append(str(path))
    return written


def _parse_csv(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def resolve_camera_workspace_mask_root(args: argparse.Namespace) -> Path | None:
    text = str(args.camera_workspace_mask_root or "").strip()
    if not text:
        return None
    return Path(text).expanduser().resolve()


def process_dataset(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = ensure_dir(
        args.output_dir
        if str(args.output_dir).strip()
        else default_output_dir(args.task_name, args.task_config)
    )
    repo_data_link = ensure_repo_data_link(
        output_dir=output_dir,
        task_name=str(args.task_name),
        task_config=str(args.task_config),
        enabled=not bool(args.no_link_repo_data),
        overwrite=bool(args.overwrite_repo_link),
    )
    raw_root = Path(args.raw_root).expanduser().resolve()
    raw_episodes = find_raw_episode_dirs(
        raw_root,
        start_episode=int(args.start_episode),
        max_episodes=int(args.max_episodes),
    )
    object_prompts = parse_object_prompts(args.object_prompts)
    placeholders = list(object_prompts.keys())
    if not placeholders:
        raise ValueError("object_prompts must not be empty.")
    camera_labels = _parse_csv(args.camera_labels)
    bbox_prompt_root = Path(args.bbox_prompt_root).expanduser().resolve()
    camera_workspace_mask_root = resolve_camera_workspace_mask_root(args)
    camera_workspace_masks = (
        {}
        if camera_workspace_mask_root is None
        else load_camera_workspace_masks(camera_workspace_mask_root, camera_labels)
    )
    if camera_workspace_mask_root is not None and len(camera_workspace_masks) == 0:
        raise RuntimeError(f"No camera workspace masks were found under: {camera_workspace_mask_root}")

    masks_root = ensure_dir(output_dir / "sam2_masks")
    debug_root = ensure_dir(output_dir / "debug") if bool(args.debug) else None
    tracker_factory = None
    if not bool(args.skip_segmentation):
        tracker_factory = make_sam2_tracker_factory(
            placeholders=placeholders,
            sam2_root=args.sam2_root,
            config=args.sam2_config,
            checkpoint=args.sam2_checkpoint,
            device=args.sam2_device,
            autocast_dtype=args.sam2_autocast_dtype,
        )

    processed: list[dict[str, Any]] = []
    for local_idx, raw_episode_dir in enumerate(raw_episodes):
        episode_index = int(args.episode_index_offset) + int(local_idx)
        hdf5_path = output_dir / "data" / f"episode{episode_index}.hdf5"
        mask_root = masks_root / f"episode{episode_index}"
        manifest = read_json(raw_episode_dir / "manifest.json")
        postprocess_settings = resolve_episode_postprocess_settings(
            raw_episode_dir=raw_episode_dir,
            manifest=manifest,
            calibration_path=args.calibration_path,
            frame_mode=args.frame_mode,
        )
        workspace_bounds = workspace_bounds_from_args(
            args,
            calibration_path=postprocess_settings["calibration_path"],
        )
        print(
            "[INFO] "
            f"{raw_episode_dir.name}: calibration={postprocess_settings['calibration_path']}, "
            f"frame_mode={postprocess_settings['frame_mode']}"
        )
        if str(args.frame_mode) != "auto" and str(args.frame_mode) != postprocess_settings["frame_mode"]:
            print(
                "[WARN] "
                f"{raw_episode_dir.name}: requested frame_mode={args.frame_mode}, "
                f"but collection-time calibration has no workspace frame; "
                f"using frame_mode={postprocess_settings['frame_mode']}"
            )

        if tracker_factory is not None:
            bbox_prompts_by_camera, bbox_prompt_path = resolve_episode_bbox_prompts(
                bbox_prompt_root=bbox_prompt_root,
                raw_episode_dir=raw_episode_dir,
                episode_index=episode_index,
                camera_labels=camera_labels,
                placeholders=placeholders,
                require_per_episode=bool(args.require_per_episode_bboxes),
            )
            debug_frame_indices = None
            if debug_root is not None:
                frames = manifest.get("frames", []) if isinstance(manifest, dict) else []
                frame_start = max(0, int(args.start_frame))
                frame_end = len(frames) if int(args.max_frames_per_episode) <= 0 else min(
                    len(frames),
                    frame_start + int(args.max_frames_per_episode),
                )
                selected_local = selected_debug_frame_indices(
                    max(0, frame_end - frame_start),
                    stride=int(args.debug_stride),
                    max_frames=int(args.debug_max_frames),
                )
                debug_frame_indices = {
                    int(frames[frame_start + idx]["frame_index"])
                    for idx in selected_local
                    if frame_start + idx < len(frames)
                }
            segment_episode_sam2(
                raw_episode_dir=raw_episode_dir,
                output_mask_root=mask_root,
                bbox_prompts_by_camera=bbox_prompts_by_camera,
                camera_labels=camera_labels,
                placeholders=placeholders,
                tracker_factory=tracker_factory,
                start_frame=int(args.start_frame),
                max_frames=int(args.max_frames_per_episode),
                overwrite=bool(args.overwrite_masks),
                mask_domain_by_camera=camera_workspace_masks,
                debug_overlay_root=None if debug_root is None else debug_root / f"episode{episode_index}" / "mask_overlays_sam2",
                debug_frame_indices=debug_frame_indices,
                mask_input_domain=not bool(args.disable_mask_input_domain),
            )
        else:
            bbox_prompt_path = ""

        skipped_existing_hdf5 = bool(hdf5_path.exists() and not bool(args.overwrite_hdf5))
        if skipped_existing_hdf5:
            print(f"skip existing hdf5: {hdf5_path}")
        else:
            hdf5_path = postprocess_episode(
                raw_episode_dir=raw_episode_dir,
                output_dir=output_dir,
                episode_index=episode_index,
                calibration_path=postprocess_settings["calibration_path"],
                camera_labels=camera_labels,
                object_prompts=object_prompts,
                mask_root=mask_root,
                scene_point_num=int(args.scene_point_num),
                object_point_num=int(args.object_point_num),
                min_depth_m=float(args.min_depth_m),
                max_depth_m=float(args.max_depth_m),
                frame_mode=postprocess_settings["frame_mode"],
                workspace_crop_bounds=workspace_bounds,
                workspace_crop_margin_px=int(args.workspace_crop_margin_px),
                intrinsics_source=args.intrinsics_source,
                serial_remap=not bool(args.disable_serial_remap),
                start_frame=int(args.start_frame),
                max_frames=int(args.max_frames_per_episode),
                store_observations=bool(args.store_observations),
                camera_workspace_mask_root=camera_workspace_mask_root,
                output_frame=args.output_frame,
                robot_camera_calibration_path=args.robot_camera_calibration_path,
            )
            print(f"wrote {hdf5_path}")

        debug_pointclouds: list[str] = []
        if debug_root is not None:
            debug_pointclouds = save_debug_object_pointclouds(
                hdf5_path=hdf5_path,
                debug_root=debug_root,
                episode_index=episode_index,
                placeholders=placeholders,
                stride=int(args.debug_stride),
                max_frames=int(args.debug_max_frames),
            )

        processed.append(
            {
                "episode_index": int(episode_index),
                "raw_episode_dir": str(raw_episode_dir),
                "hdf5_path": str(hdf5_path),
                "mask_root": str(mask_root),
                "bbox_prompt_path": str(bbox_prompt_path),
                "debug_pointclouds": debug_pointclouds,
                "calibration_path": postprocess_settings["calibration_path"],
                "frame_mode": postprocess_settings["frame_mode"],
                "workspace_crop_bounds_m": None if workspace_bounds is None else workspace_bounds.as_dict(),
                "skipped_existing_hdf5": skipped_existing_hdf5,
            }
        )

    meta = {
        "task_name": str(args.task_name),
        "task_config": str(args.task_config),
        "output_dir": str(output_dir),
        "repo_data_link": None if repo_data_link is None else str(repo_data_link),
        "raw_root": str(raw_root),
        "object_prompts": object_prompts,
        "camera_labels": camera_labels,
        "bbox_prompt_root": str(bbox_prompt_root),
        "sam2_root": str(Path(args.sam2_root).expanduser().resolve()),
        "sam2_config": str(args.sam2_config),
        "sam2_checkpoint": str(Path(args.sam2_checkpoint).expanduser().resolve()),
        "sam2_device": str(args.sam2_device),
        "sam2_autocast_dtype": str(args.sam2_autocast_dtype),
        "scene_point_num": int(args.scene_point_num),
        "object_point_num": int(args.object_point_num),
        "requested_frame_mode": str(args.frame_mode),
        "output_frame": str(args.output_frame),
        "robot_camera_calibration_path": str(args.robot_camera_calibration_path),
        "requested_calibration_path": str(args.calibration_path),
        "camera_workspace_mask_root": None if camera_workspace_mask_root is None else str(camera_workspace_mask_root),
        "camera_workspace_mask_labels": sorted(camera_workspace_masks.keys()),
        "processed": processed,
        "debug": bool(args.debug),
    }
    write_json(output_dir / "real_zed_sam2_objpc_meta.json", meta)
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate bbox/point-initialized SAM2 tracking masks and convert real three-ZED raw episodes to train_objpc.sh-compatible HDF5."
    )
    parser.add_argument("--raw_root", default=str(default_raw_root()))
    parser.add_argument("--task_name", default=DEFAULT_TASK_NAME)
    parser.add_argument("--task_config", default=DEFAULT_TASK_CONFIG)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--object_prompts", default=DEFAULT_OBJECT_PROMPTS)
    parser.add_argument("--camera_labels", default=DEFAULT_CAMERA_LABELS)
    parser.add_argument("--bbox_prompt_root", default=str(default_bbox_prompt_root()))
    parser.add_argument("--require_per_episode_bboxes", action="store_true", default=False)
    parser.add_argument("--calibration_path", default="auto")
    parser.add_argument("--frame_mode", default="auto", choices=["auto", "reference_camera", "workspace"])
    parser.add_argument("--output_frame", default="source", choices=["source", "workspace", "left_base", "right_base"])
    parser.add_argument("--robot_camera_calibration_path", default="")
    parser.add_argument("--start_episode", type=int, default=0)
    parser.add_argument("--max_episodes", type=int, default=-1)
    parser.add_argument("--episode_index_offset", type=int, default=0)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--max_frames_per_episode", type=int, default=-1)
    parser.add_argument("--sam2_root", default=str(DEFAULT_SAM2_ROOT))
    parser.add_argument("--sam2_config", default=DEFAULT_SAM2_CONFIG)
    parser.add_argument("--sam2_checkpoint", default=str(DEFAULT_SAM2_CHECKPOINT))
    parser.add_argument("--sam2_device", default="cuda")
    parser.add_argument("--sam2_autocast_dtype", default="bfloat16")
    parser.add_argument("--skip_segmentation", action="store_true", default=False)
    parser.add_argument("--overwrite_masks", action="store_true", default=False)
    parser.add_argument("--overwrite_hdf5", action="store_true", default=False)
    parser.add_argument("--no_link_repo_data", action="store_true", default=False)
    parser.add_argument("--overwrite_repo_link", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--debug_stride", type=int, default=10)
    parser.add_argument("--debug_max_frames", type=int, default=5)
    parser.add_argument("--camera_workspace_mask_root", default="")
    parser.add_argument("--disable_mask_input_domain", action="store_true", default=False)
    parser.add_argument("--scene_point_num", type=int, default=1024)
    parser.add_argument("--object_point_num", type=int, default=1024)
    parser.add_argument("--min_depth_m", type=float, default=0.05)
    parser.add_argument("--max_depth_m", type=float, default=3.0)
    parser.add_argument("--intrinsics_source", default="calibration", choices=["calibration", "frame"])
    parser.add_argument("--disable_serial_remap", action="store_true", default=False)
    parser.add_argument("--store_observations", action="store_true", default=False)
    parser.add_argument("--disable_workspace_crop", action="store_true", default=False)
    parser.add_argument("--workspace_crop_x_min", type=float, default=None)
    parser.add_argument("--workspace_crop_x_max", type=float, default=None)
    parser.add_argument("--workspace_crop_y_min", type=float, default=None)
    parser.add_argument("--workspace_crop_y_max", type=float, default=None)
    parser.add_argument("--workspace_crop_z_min", type=float, default=None)
    parser.add_argument("--workspace_crop_z_max", type=float, default=None)
    parser.add_argument("--workspace_crop_margin_px", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    meta = process_dataset(parse_args())
    print(meta["output_dir"])


if __name__ == "__main__":
    main()
