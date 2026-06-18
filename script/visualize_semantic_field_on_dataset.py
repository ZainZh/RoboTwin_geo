from __future__ import annotations

import argparse
import colorsys
import json
import sys
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DP3_SCRIPT_ROOT = REPO_ROOT / "policy" / "DP3" / "scripts"
if str(DP3_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(DP3_SCRIPT_ROOT))

DEBUG_PLACEHOLDER_COLORS_RGB = {
    "{A}": np.asarray([255.0, 48.0, 48.0], dtype=np.float32),
    "A": np.asarray([255.0, 48.0, 48.0], dtype=np.float32),
    "{B}": np.asarray([48.0, 96.0, 255.0], dtype=np.float32),
    "B": np.asarray([48.0, 96.0, 255.0], dtype=np.float32),
}

from object_pointcloud_utils import (  # noqa: E402
    default_placeholder_order,
    ensure_point_cloud_channels,
    extract_placeholder_point_cloud,
    load_hdf5,
    load_scene_info,
    parse_placeholder_list,
    resample_point_cloud,
    strip_zero_points,
)


def parse_episode_frame_specs(
    *,
    episode: int,
    frame: int,
    episode_frames: Sequence[str] | str | None = None,
) -> list[tuple[int, int]]:
    if not episode_frames:
        return [(int(episode), int(frame))]

    if isinstance(episode_frames, str):
        raw_items = [episode_frames]
    else:
        raw_items = list(episode_frames)
    items = [
        part.strip()
        for raw_item in raw_items
        for part in str(raw_item).split(",")
        if part.strip()
    ]

    specs: list[tuple[int, int]] = []
    for item in items:
        text = str(item).strip()
        if ":" not in text:
            raise ValueError(f"episode frame spec must use '<episode>:<frame>', got: {item}")
        episode_text, frame_text = text.split(":", 1)
        try:
            specs.append((int(episode_text), int(frame_text)))
        except ValueError as exc:
            raise ValueError(f"episode frame spec must use integer values, got: {item}") from exc
    return specs


def parse_feature_methods(methods: Sequence[str] | str | None = None) -> list[str]:
    if not methods:
        return ["semantic"]
    raw_items = [methods] if isinstance(methods, str) else list(methods)
    parsed = [
        part.strip()
        for raw_item in raw_items
        for part in str(raw_item).split(",")
        if part.strip()
    ]
    allowed = {"semantic", "utonia", "dinov2"}
    result: list[str] = []
    for method in parsed:
        if method not in allowed:
            raise ValueError(f"feature method must be one of {sorted(allowed)}, got: {method}")
        if method not in result:
            result.append(method)
    return result or ["semantic"]


def resolve_feature_methods(args: argparse.Namespace) -> list[str]:
    switch_methods = [
        ("semantic", bool(getattr(args, "run_semantic", False))),
        ("utonia", bool(getattr(args, "run_utonia", False))),
        ("dinov2", bool(getattr(args, "run_dinov2", False))),
    ]
    if any(enabled for _, enabled in switch_methods):
        return [method for method, enabled in switch_methods if enabled]
    return parse_feature_methods(getattr(args, "feature_methods", "semantic"))


def fit_shared_pca_projection(embedding_sets: Sequence[np.ndarray]) -> dict[str, np.ndarray | int]:
    valid_sets = [
        np.asarray(item, dtype=np.float32)
        for item in embedding_sets
        if item is not None and np.asarray(item).size > 0
    ]
    if not valid_sets:
        raise ValueError("Shared PCA requires at least one non-empty embedding set.")

    embeddings = np.concatenate(valid_sets, axis=0).astype(np.float32, copy=False)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")

    mean = embeddings.mean(axis=0, keepdims=True)
    centered = embeddings - mean
    projection_dim = min(3, centered.shape[1]) if centered.ndim == 2 else 1
    components = np.zeros((centered.shape[1], 3), dtype=np.float32)
    if centered.shape[0] >= 3 and centered.shape[1] >= 2:
        import torch

        matrix = torch.from_numpy(centered)
        _, _, basis = torch.pca_lowrank(matrix, q=projection_dim)
        basis_np = basis[:, :projection_dim].numpy().astype(np.float32, copy=False)
        components[:, :projection_dim] = basis_np
    else:
        components[:projection_dim, :projection_dim] = np.eye(projection_dim, dtype=np.float32)

    projected = centered @ components
    projected_min = projected.min(axis=0, keepdims=True)
    projected_span = projected.max(axis=0, keepdims=True) - projected_min
    projected_span[projected_span < 1e-8] = 1.0
    return {
        "mean": mean.astype(np.float32, copy=False),
        "components": components,
        "projected_min": projected_min.astype(np.float32, copy=False),
        "projected_span": projected_span.astype(np.float32, copy=False),
        "embedding_dim": int(embeddings.shape[1]),
        "num_points": int(embeddings.shape[0]),
    }


def apply_shared_pca_to_rgb(
    embeddings: np.ndarray,
    projection: Mapping[str, np.ndarray | int],
) -> np.ndarray:
    embeddings_np = np.asarray(embeddings, dtype=np.float32)
    if embeddings_np.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    centered = embeddings_np - np.asarray(projection["mean"], dtype=np.float32)
    projected = centered @ np.asarray(projection["components"], dtype=np.float32)
    projected = projected - np.asarray(projection["projected_min"], dtype=np.float32)
    projected = projected / np.asarray(projection["projected_span"], dtype=np.float32)
    return np.clip(projected * 255.0, 0.0, 255.0).astype(np.uint8)


def fit_utonia_demo_pca_projection(
    embedding_sets: Sequence[np.ndarray],
    *,
    brightness: float = 1.2,
    center: bool = True,
) -> dict[str, np.ndarray | int | float | str]:
    valid_sets = [
        np.asarray(item, dtype=np.float32)
        for item in embedding_sets
        if item is not None and np.asarray(item).size > 0
    ]
    if not valid_sets:
        raise ValueError("Utonia demo PCA requires at least one non-empty embedding set.")
    embeddings = np.concatenate(valid_sets, axis=0).astype(np.float32, copy=False)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")

    import torch

    feat = torch.from_numpy(embeddings)
    q = min(9, int(feat.shape[0]), int(feat.shape[1]))
    components = np.zeros((embeddings.shape[1], 9), dtype=np.float32)
    if q > 0:
        _, _, basis = torch.pca_lowrank(feat, center=bool(center), niter=5, q=q)
        components[:, :q] = basis[:, :q].numpy().astype(np.float32, copy=False)
    projection = embeddings @ components
    mixed = projection[:, :3] * 0.6 + projection[:, 3:6] * 0.4
    mixed_min = mixed.min(axis=0, keepdims=True)
    mixed_span = mixed.max(axis=0, keepdims=True) - mixed_min
    mixed_span[mixed_span < 1e-6] = 1.0
    return {
        "components": components,
        "mixed_min": mixed_min.astype(np.float32, copy=False),
        "mixed_span": mixed_span.astype(np.float32, copy=False),
        "brightness": float(brightness),
        "embedding_dim": int(embeddings.shape[1]),
        "num_points": int(embeddings.shape[0]),
        "projection_mode": "utonia_demo_6pc_mix",
    }


def apply_utonia_demo_pca_to_rgb(
    embeddings: np.ndarray,
    projection: Mapping[str, np.ndarray | int | float | str],
) -> np.ndarray:
    embeddings_np = np.asarray(embeddings, dtype=np.float32)
    if embeddings_np.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    projected = embeddings_np @ np.asarray(projection["components"], dtype=np.float32)
    mixed = projected[:, :3] * 0.6 + projected[:, 3:6] * 0.4
    mixed = mixed - np.asarray(projection["mixed_min"], dtype=np.float32)
    mixed = mixed / np.asarray(projection["mixed_span"], dtype=np.float32)
    mixed = mixed * float(projection.get("brightness", 1.2))
    return np.clip(mixed * 255.0, 0.0, 255.0).astype(np.uint8)


def build_label_palette(num_classes: int) -> np.ndarray:
    colors: list[list[int]] = []
    for index in range(max(int(num_classes), 1)):
        hue = (index * 0.6180339887498949) % 1.0
        saturation = 0.70 + 0.30 * ((index % 3) / 2.0)
        value = 0.85 + 0.15 * (index % 2)
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append([int(channel * 255) for channel in rgb])
    return np.asarray(colors, dtype=np.uint8)


def labels_to_rgb(
    labels: np.ndarray,
    palette: np.ndarray,
    *,
    unknown_color: Sequence[int] = (128, 128, 128),
) -> np.ndarray:
    labels_np = np.asarray(labels, dtype=np.int64).reshape(-1)
    palette_np = np.asarray(palette, dtype=np.uint8).reshape(-1, 3)
    colors = np.repeat(np.asarray(unknown_color, dtype=np.uint8)[None, :], labels_np.shape[0], axis=0)
    valid = (labels_np >= 0) & (labels_np < palette_np.shape[0])
    colors[valid] = palette_np[labels_np[valid]]
    return colors.astype(np.uint8, copy=False)


def labels_to_named_histogram(labels: np.ndarray, label_names: Sequence[str]) -> dict[str, int]:
    labels_np = np.asarray(labels, dtype=np.int64).reshape(-1)
    histogram: dict[str, int] = {}
    if labels_np.size == 0:
        return histogram
    for label_idx, count in zip(*np.unique(labels_np, return_counts=True), strict=True):
        idx = int(label_idx)
        if 0 <= idx < len(label_names):
            histogram[str(label_names[idx])] = int(count)
        else:
            histogram[f"unknown_{idx}"] = int(count)
    return histogram


def prepare_semantic_input_cloud(
    object_cloud: np.ndarray,
    *,
    placeholder: str,
    color_mode: str,
) -> np.ndarray:
    cloud = ensure_point_cloud_channels(object_cloud, channels=6).astype(np.float32, copy=True)
    mode = str(color_mode)
    if mode == "stored":
        return cloud
    if mode == "stored_scaled":
        if cloud.shape[0] > 0 and float(np.nanmax(cloud[:, 3:6])) <= 1.0:
            cloud[:, 3:6] *= 255.0
        return cloud
    if mode == "debug_placeholder":
        color = DEBUG_PLACEHOLDER_COLORS_RGB.get(str(placeholder))
        if color is None:
            color = np.asarray([180.0, 180.0, 180.0], dtype=np.float32)
        cloud[:, 3:6] = color[None, :]
        return cloud
    raise ValueError(f"Unsupported semantic input color mode: {color_mode}")


def _normalize_rgb_255(rgb: np.ndarray) -> np.ndarray:
    rgb_np = np.asarray(rgb, dtype=np.float32)
    if rgb_np.size > 0 and float(np.nanmax(rgb_np)) <= 1.0:
        rgb_np = rgb_np * 255.0
    return np.clip(rgb_np[:, :3], 0.0, 255.0).astype(np.float32)


def _scene_with_background(scene_cloud: np.ndarray, *, background_mode: str) -> np.ndarray:
    scene = ensure_point_cloud_channels(scene_cloud, channels=6)
    out = np.zeros((scene.shape[0], 6), dtype=np.float32)
    out[:, :3] = scene[:, :3]
    if background_mode == "gray":
        out[:, 3:6] = 120.0
    elif background_mode == "black":
        out[:, 3:6] = 0.0
    else:
        out[:, 3:6] = _normalize_rgb_255(scene[:, 3:6])
    return out


def overlay_semantic_colors_on_scene(
    scene_cloud: np.ndarray,
    object_xyz: np.ndarray,
    object_rgb: np.ndarray,
    *,
    max_distance: float,
    min_neighbors: int = 1,
    chunk_size: int = 4096,
    background_mode: str = "original",
) -> np.ndarray:
    overlay = _scene_with_background(scene_cloud, background_mode=background_mode)
    object_xyz_np = np.asarray(object_xyz, dtype=np.float32).reshape(-1, 3)
    object_rgb_np = np.asarray(object_rgb, dtype=np.uint8).reshape(-1, 3)
    if object_xyz_np.shape[0] == 0 or object_rgb_np.shape[0] == 0:
        return overlay
    if object_xyz_np.shape[0] != object_rgb_np.shape[0]:
        raise ValueError(
            f"object xyz/color size mismatch: {object_xyz_np.shape[0]} vs {object_rgb_np.shape[0]}"
        )

    scene_xyz = overlay[:, :3].astype(np.float32, copy=False)
    max_dist2 = float(max_distance) * float(max_distance)
    for start in range(0, scene_xyz.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), scene_xyz.shape[0])
        chunk = scene_xyz[start:stop]
        dist2 = np.sum((chunk[:, None, :] - object_xyz_np[None, :, :]) ** 2, axis=-1)
        nearest = np.argmin(dist2, axis=1)
        nearest_dist2 = dist2[np.arange(stop - start), nearest]
        valid = nearest_dist2 <= max_dist2
        if int(min_neighbors) > 1:
            valid &= np.sum(dist2 <= max_dist2, axis=1) >= int(min_neighbors)
        if np.any(valid):
            overlay[start:stop][valid, 3:6] = object_rgb_np[nearest[valid]].astype(np.float32)
    return overlay.astype(np.float32)


def compose_scene_semantic_overlay(
    scene_cloud: np.ndarray,
    object_xyz: np.ndarray,
    object_rgb: np.ndarray,
    *,
    mode: str,
    max_distance: float,
    min_neighbors: int = 1,
    cut_z_min: float | None = None,
    background_mode: str = "original",
) -> np.ndarray:
    mode_text = str(mode)
    scene = _scene_with_background(scene_cloud, background_mode=background_mode)
    object_xyz_np = np.asarray(object_xyz, dtype=np.float32).reshape(-1, 3)
    object_rgb_np = np.asarray(object_rgb, dtype=np.uint8).reshape(-1, 3).astype(np.float32)
    if object_xyz_np.shape[0] == 0:
        return scene
    colored_object = np.concatenate([object_xyz_np, object_rgb_np], axis=1).astype(np.float32)

    if mode_text == "append":
        return np.concatenate([scene, colored_object], axis=0).astype(np.float32)
    if mode_text == "cut_replace":
        max_dist2 = float(max_distance) * float(max_distance)
        keep = np.ones((scene.shape[0],), dtype=bool)
        scene_xyz = scene[:, :3].astype(np.float32, copy=False)
        z_min = None if cut_z_min is None else float(cut_z_min)
        for start in range(0, scene_xyz.shape[0], 4096):
            stop = min(start + 4096, scene_xyz.shape[0])
            chunk = scene_xyz[start:stop]
            candidate = np.ones((chunk.shape[0],), dtype=bool)
            if z_min is not None:
                candidate &= chunk[:, 2] >= z_min
            candidate_idx = np.flatnonzero(candidate)
            if candidate_idx.shape[0] == 0:
                continue
            candidate_xyz = chunk[candidate_idx]
            dist2 = np.sum((candidate_xyz[:, None, :] - object_xyz_np[None, :, :]) ** 2, axis=-1)
            remove_candidate = np.min(dist2, axis=1) <= max_dist2
            if np.any(remove_candidate):
                keep[start + candidate_idx[remove_candidate]] = False
        return np.concatenate([scene[keep], colored_object], axis=0).astype(np.float32)
    if mode_text == "replace_nearest":
        return overlay_semantic_colors_on_scene(
            scene,
            object_xyz,
            object_rgb,
            max_distance=float(max_distance),
            min_neighbors=int(min_neighbors),
            background_mode=background_mode,
        )
    raise ValueError("overlay mode must be one of: cut_replace, append, replace_nearest")


def write_colored_ply(path: str | Path, cloud: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pc = ensure_point_cloud_channels(cloud, channels=6)
    xyz = pc[:, :3].astype(np.float32, copy=False)
    rgb = _normalize_rgb_255(pc[:, 3:6]).astype(np.uint8)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(xyz, rgb, strict=True):
            f.write(
                f"{float(point[0]):.7f} {float(point[1]):.7f} {float(point[2]):.7f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def _safe_placeholder_name(placeholder: str) -> str:
    return str(placeholder).replace("{", "").replace("}", "").replace("/", "_")


def _normalize_placeholder_name(value: str) -> str:
    text = str(value).strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    return "{" + text.strip("{}") + "}"


def _parse_checkpoint_mapping(text: str | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not text:
        return mapping
    for item in str(text).split(","):
        token = item.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"semantic checkpoint mapping must use '<placeholder>=<path>', got: {item}")
        key, value = token.split("=", 1)
        mapping[_normalize_placeholder_name(key)] = value.strip()
    return mapping


def build_semantic_checkpoint_map(args: argparse.Namespace, placeholders: Sequence[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    shared = str(getattr(args, "semantic_ckpt", "") or "").strip()
    if shared and shared.lower() != "none":
        for placeholder in placeholders:
            mapping[str(placeholder)] = shared

    ckpt_a = str(getattr(args, "semantic_ckpt_A", "") or "").strip()
    ckpt_b = str(getattr(args, "semantic_ckpt_B", "") or "").strip()
    if ckpt_a and ckpt_a.lower() != "none":
        mapping["{A}"] = ckpt_a
    if ckpt_b and ckpt_b.lower() != "none":
        mapping["{B}"] = ckpt_b
    mapping.update(_parse_checkpoint_mapping(getattr(args, "semantic_ckpts", "")))
    return {placeholder: mapping[placeholder] for placeholder in placeholders if placeholder in mapping}


def resolve_dataset_dir(args: argparse.Namespace) -> Path:
    if getattr(args, "dataset_dir", None):
        return Path(args.dataset_dir).expanduser().resolve()
    if not getattr(args, "task_name", None) or not getattr(args, "task_config", None):
        raise ValueError("Either --dataset_dir or both --task_name and --task_config must be provided.")
    return (Path(args.data_root).expanduser() / args.task_name / args.task_config).resolve()


def _parse_camera_labels(text: str | Sequence[str] | None) -> list[str] | None:
    if text is None:
        return None
    if isinstance(text, (list, tuple)):
        labels = [str(item).strip() for item in text if str(item).strip()]
    else:
        labels = [item.strip() for item in str(text).split(",") if item.strip()]
    return labels or None


def _load_processed_meta(dataset_dir: Path) -> dict:
    path = Path(dataset_dir) / "real_zed_sam2_objpc_meta.json"
    if not path.exists():
        raise FileNotFoundError(f"Processed real-ZED meta not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Processed real-ZED meta must be a JSON object: {path}")
    return data


def _processed_episode_record(meta: Mapping[str, object], episode_idx: int) -> Mapping[str, object]:
    processed = meta.get("processed", [])
    if not isinstance(processed, list):
        raise ValueError("Processed real-ZED meta field 'processed' must be a list.")
    for record in processed:
        if isinstance(record, Mapping) and int(record.get("episode_index", -1)) == int(episode_idx):
            return record
    raise KeyError(f"No processed meta record for episode {episode_idx}.")


def _rgb_to_depth_shape(rgb: np.ndarray, depth_shape: tuple[int, int]) -> np.ndarray:
    rgb_arr = np.asarray(rgb, dtype=np.uint8)
    if rgb_arr.shape[:2] == tuple(depth_shape):
        return rgb_arr
    try:
        import cv2
    except Exception as exc:  # pragma: no cover - depends on runtime image stack.
        raise RuntimeError("cv2 is required when raw RGB/depth shapes differ.") from exc
    return cv2.resize(rgb_arr, (int(depth_shape[1]), int(depth_shape[0])), interpolation=cv2.INTER_LINEAR)


def load_raw_full_scene_point_cloud(
    *,
    dataset_dir: Path,
    episode_idx: int,
    frame_idx: int,
    camera_labels: Sequence[str] | None,
    point_num: int,
    min_depth_m: float,
    max_depth_m: float,
    intrinsics_source: str,
    serial_remap: bool,
) -> np.ndarray:
    from script.real_zed_collection.postprocess.postprocess_raw_to_robotwin_hdf5 import (
        _load_frame_npz,
        _output_frame_transforms,
        _select_robot_camera_calibration,
    )
    from script.real_zed_collection.real_zed_utils import (
        calibration_label_map_from_manifest,
        depth_rgb_to_point_cloud,
        deterministic_resample,
        load_three_zed_calibration,
        merge_point_clouds,
        read_json,
        transform_point_cloud,
    )

    meta = _load_processed_meta(dataset_dir)
    record = _processed_episode_record(meta, episode_idx)
    raw_episode_dir = Path(str(record["raw_episode_dir"])).expanduser().resolve()
    manifest = read_json(raw_episode_dir / "manifest.json")
    frames = manifest.get("frames", [])
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"Raw episode manifest has no frames: {raw_episode_dir / 'manifest.json'}")
    if frame_idx < 0 or frame_idx >= len(frames):
        raise IndexError(f"Raw frame {frame_idx} is out of range for episode {episode_idx}.")

    frame_mode = str(record.get("frame_mode", "reference_camera"))
    output_frame = str(meta.get("output_frame", "source"))
    calibration_path = Path(str(record["calibration_path"])).expanduser().resolve()
    calib = load_three_zed_calibration(calibration_path, frame_mode=frame_mode)
    labels = list(camera_labels or meta.get("camera_labels", []) or manifest.get("camera_labels", []) or calib.keys())
    labels = [str(label) for label in labels]
    label_to_calib = (
        calibration_label_map_from_manifest(manifest, calib, labels)
        if bool(serial_remap)
        else {label: label for label in labels}
    )
    robot_calibration = _select_robot_camera_calibration(
        raw_episode_dir=raw_episode_dir,
        manifest=manifest,
        output_frame=output_frame,
        explicit_path=record.get("robot_camera_calibration_path", meta.get("robot_camera_calibration_path", "")),
    )
    _, t_output_from_cam_by_label = _output_frame_transforms(
        calib=calib,
        labels=labels,
        label_to_calib=label_to_calib,
        output_frame=output_frame,
        robot_camera_calibration=robot_calibration,
    )

    frame = frames[int(frame_idx)]
    cameras = frame.get("cameras", {})
    chunks = []
    for label in labels:
        calib_label = label_to_calib[label]
        camera_frame = _load_frame_npz(raw_episode_dir, str(cameras[label]))
        depth = (
            np.asarray(camera_frame["depth_m"], dtype=np.float32)
            if "depth_m" in camera_frame
            else np.asarray(camera_frame["depth_mm"], dtype=np.float32) / 1000.0
        )
        rgb = _rgb_to_depth_shape(np.asarray(camera_frame["rgb"], dtype=np.uint8), depth.shape)
        camera_matrix = (
            np.asarray(camera_frame["camera_matrix"], dtype=np.float32).reshape(3, 3)
            if str(intrinsics_source) == "frame" and "camera_matrix" in camera_frame
            else calib[calib_label].camera_matrix.astype(np.float32)
        )
        pc_cam = depth_rgb_to_point_cloud(
            depth_m=depth,
            rgb=rgb,
            camera_matrix=camera_matrix,
            mask=None,
            min_depth_m=float(min_depth_m),
            max_depth_m=float(max_depth_m),
        )
        chunks.append(transform_point_cloud(pc_cam, t_output_from_cam_by_label[label]))
    merged = merge_point_clouds(chunks)
    return deterministic_resample(merged, int(point_num)) if int(point_num) > 0 else merged.astype(np.float32)


def load_scene_cloud_for_frame(
    *,
    dataset_dir: Path,
    episodes: Mapping[int, dict],
    episode_idx: int,
    frame_idx: int,
    args: argparse.Namespace,
    raw_scene_loader: Callable[..., np.ndarray] | None = None,
) -> tuple[np.ndarray, str]:
    scene_source = str(getattr(args, "scene_source", "auto"))
    if scene_source not in {"auto", "raw_full", "hdf5"}:
        raise ValueError("scene_source must be one of: auto, raw_full, hdf5")

    if scene_source in {"auto", "raw_full"}:
        loader = raw_scene_loader or load_raw_full_scene_point_cloud
        try:
            scene = loader(
                dataset_dir=Path(dataset_dir),
                episode_idx=int(episode_idx),
                frame_idx=int(frame_idx),
                camera_labels=_parse_camera_labels(getattr(args, "camera_labels", "")),
                point_num=int(getattr(args, "scene_point_num", 0)),
                min_depth_m=float(getattr(args, "min_depth_m", 0.05)),
                max_depth_m=float(getattr(args, "max_depth_m", 3.0)),
                intrinsics_source=str(getattr(args, "raw_intrinsics_source", "frame")),
                serial_remap=not bool(getattr(args, "disable_serial_remap", False)),
            )
            return ensure_point_cloud_channels(scene, channels=6), "raw_full"
        except FileNotFoundError:
            if scene_source == "raw_full":
                raise
        except KeyError:
            if scene_source == "raw_full":
                raise

    scene = ensure_point_cloud_channels(episodes[int(episode_idx)]["pointcloud"][int(frame_idx)], channels=6)
    if int(getattr(args, "scene_point_num", 0)) > 0:
        scene = resample_point_cloud(scene, int(getattr(args, "scene_point_num", 0)))
    return scene, "hdf5"


def _load_object_cloud_for_frame(
    *,
    episode: dict,
    frame_idx: int,
    placeholder: str,
    target_num_points: int,
) -> np.ndarray:
    exact = episode.get("object_pointcloud", {}).get(placeholder)
    if exact is not None:
        cloud = ensure_point_cloud_channels(exact[int(frame_idx)], channels=6)
        cloud = strip_zero_points(cloud)
        if int(target_num_points) > 0:
            cloud = resample_point_cloud(cloud, int(target_num_points))
        return cloud.astype(np.float32)

    cloud, _ = extract_placeholder_point_cloud(
        episode,
        frame_idx=int(frame_idx),
        placeholder=placeholder,
        target_num_points=int(target_num_points) if int(target_num_points) > 0 else episode["pointcloud"].shape[1],
    )
    return strip_zero_points(cloud).astype(np.float32)


def _load_semantic_backend(device: str):
    import torch
    from semantic_feature_utils import (
        compute_semantic_pointwise_cloud,
        compute_semantic_pointwise_prediction,
        load_semantic_model,
    )

    return torch.device(device), load_semantic_model, compute_semantic_pointwise_cloud, compute_semantic_pointwise_prediction


def group_records_for_pca(records: list[dict], *, shared_pca_scope: str) -> dict[str, list[dict]]:
    if shared_pca_scope == "all":
        return {"all": records}
    if shared_pca_scope == "checkpoint":
        grouped: dict[str, list[dict]] = {}
        for record in records:
            grouped.setdefault(str(record["semantic_checkpoint"]), []).append(record)
        return grouped
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(str(record["placeholder"]), []).append(record)
    return grouped


def group_feature_records_for_pca(records: list[dict], *, shared_pca_scope: str) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        method = str(record.get("feature_method", "semantic"))
        if shared_pca_scope == "all":
            key = f"{method}:all"
        elif shared_pca_scope == "checkpoint" and method == "semantic":
            key = f"{method}:{record.get('semantic_checkpoint', '')}"
        elif shared_pca_scope == "placeholder":
            key = f"{method}:{record.get('placeholder', '')}"
        else:
            key = f"{method}:all"
        grouped.setdefault(key, []).append(record)
    return grouped


def _estimate_radial_normals(xyz: np.ndarray) -> np.ndarray:
    points = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    if points.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    center = points.mean(axis=0, keepdims=True)
    normals = points - center
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    fallback = norm[:, 0] < 1e-6
    normals = normals / np.clip(norm, 1e-6, None)
    if np.any(fallback):
        normals[fallback] = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    return normals.astype(np.float32, copy=False)


def _utonia_center_shift(xyz: np.ndarray) -> np.ndarray:
    points = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    if points.shape[0] == 0:
        return np.zeros((3,), dtype=np.float32)
    xyz_min = points.min(axis=0)
    xyz_max = points.max(axis=0)
    return np.asarray(
        [
            (float(xyz_min[0]) + float(xyz_max[0])) * 0.5,
            (float(xyz_min[1]) + float(xyz_max[1])) * 0.5,
            float(xyz_min[2]),
        ],
        dtype=np.float32,
    )


def _build_utonia_transform_config(
    *,
    grid_size: float,
    scale: float = 1.0,
    apply_z_positive: bool = True,
    normalize_coord: bool = False,
) -> list[dict]:
    return [
        *([{"type": "NormalizeCoord"}] if normalize_coord else []),
        {"type": "RandomScale", "scale": [float(scale), float(scale)]},
        *([{"type": "CenterShift", "apply_z": True}] if apply_z_positive else []),
        {
            "type": "GridSample",
            "grid_size": float(grid_size),
            "hash_type": "fnv",
            "mode": "train",
            "return_grid_coord": True,
            "return_inverse": True,
        },
        {"type": "NormalizeColor"},
        {"type": "ToTensor"},
        {
            "type": "Collect",
            "keys": ("coord", "grid_coord", "color", "inverse"),
            "feat_keys": ("coord", "color", "normal"),
        },
    ]


def prepare_utonia_input_cloud(
    object_cloud: np.ndarray,
    *,
    placeholder: str,
    color_mode: str,
) -> np.ndarray:
    if str(color_mode) == "zero":
        cloud = ensure_point_cloud_channels(object_cloud, channels=6).astype(np.float32, copy=True)
        cloud[:, 3:6] = 0.0
        return cloud
    return prepare_semantic_input_cloud(object_cloud, placeholder=placeholder, color_mode=color_mode)


def _load_utonia_backend(args: argparse.Namespace):
    utonia_root = str(getattr(args, "utonia_root", "") or "").strip()
    if utonia_root and utonia_root not in sys.path:
        sys.path.insert(0, utonia_root)

    import torch
    import utonia

    try:
        import flash_attn  # noqa: F401
    except ImportError:
        flash_attn = None

    device = torch.device(str(getattr(args, "utonia_device", "") or getattr(args, "device", "cuda")))
    repo_id = str(getattr(args, "utonia_repo_id", "Pointcept/Utonia"))
    sample_seed = int(getattr(args, "utonia_sample_seed", 73))
    if sample_seed >= 0:
        utonia.utils.set_seed(sample_seed)
    if flash_attn is None:
        custom_config = {
            "enc_patch_size": [1024 for _ in range(5)],
            "enable_flash": False,
        }
        model = utonia.load("utonia", repo_id=repo_id, custom_config=custom_config).to(device)
    else:
        model = utonia.load("utonia", repo_id=repo_id).to(device)
    model.eval()
    transform = utonia.transform.Compose(
        _build_utonia_transform_config(
            grid_size=float(getattr(args, "utonia_grid_size", 0.01)),
            normalize_coord=False,
        )
    )
    input_color_mode = str(getattr(args, "utonia_input_color_mode", "stored_scaled"))

    def compute_utonia_pointwise_cloud(
        object_cloud: np.ndarray,
        *,
        placeholder: str,
        target_num_points: int,
        **_kwargs,
    ) -> np.ndarray:
        cloud = ensure_point_cloud_channels(object_cloud, channels=6)
        cloud = strip_zero_points(cloud)
        if int(target_num_points) > 0:
            cloud = resample_point_cloud(cloud, int(target_num_points))
        cloud = prepare_utonia_input_cloud(cloud, placeholder=placeholder, color_mode=input_color_mode)
        point = {
            "coord": cloud[:, :3].astype(np.float32, copy=False),
            "color": _normalize_rgb_255(cloud[:, 3:6]).astype(np.float32, copy=False),
            "normal": _estimate_radial_normals(cloud[:, :3]),
        }
        coord_shift = _utonia_center_shift(point["coord"])
        point = transform(point)
        batch = utonia.data.collate_fn([point])
        with torch.inference_mode():
            for key, value in list(batch.items()):
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device, non_blocking=True)
            output = model(batch)
            for _ in range(2):
                if "pooling_parent" not in output.keys() or "pooling_inverse" not in output.keys():
                    break
                parent = output.pop("pooling_parent")
                inverse = output.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, output.feat[inverse]], dim=-1)
                output = parent
            while "pooling_parent" in output.keys() and "pooling_inverse" in output.keys():
                parent = output.pop("pooling_parent")
                inverse = output.pop("pooling_inverse")
                parent.feat = output.feat[inverse]
                output = parent
        coords = output.coord.detach().float().cpu().numpy().astype(np.float32, copy=False)
        coords = coords + coord_shift[None, :]
        features = output.feat.detach().float().cpu().numpy().astype(np.float32, copy=False)
        return np.concatenate([coords[:, :3], features], axis=1).astype(np.float32, copy=False)

    return compute_utonia_pointwise_cloud


def _load_dinov2_camera_contexts(
    *,
    dataset_dir: Path,
    episode_idx: int,
    frame_idx: int,
    camera_labels: Sequence[str] | None,
    args: argparse.Namespace,
) -> list[dict]:
    from script.real_zed_collection.postprocess.postprocess_raw_to_robotwin_hdf5 import (
        _load_frame_npz,
        _output_frame_transforms,
        _select_robot_camera_calibration,
    )
    from script.real_zed_collection.real_zed_utils import (
        calibration_label_map_from_manifest,
        load_three_zed_calibration,
        read_json,
    )

    meta = _load_processed_meta(dataset_dir)
    record = _processed_episode_record(meta, int(episode_idx))
    raw_episode_dir = Path(str(record["raw_episode_dir"])).expanduser().resolve()
    manifest = read_json(raw_episode_dir / "manifest.json")
    frames = manifest.get("frames", [])
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"Raw episode manifest has no frames: {raw_episode_dir / 'manifest.json'}")
    if frame_idx < 0 or frame_idx >= len(frames):
        raise IndexError(f"Raw frame {frame_idx} is out of range for episode {episode_idx}.")

    frame_mode = str(record.get("frame_mode", "reference_camera"))
    output_frame = str(meta.get("output_frame", "source"))
    calibration_path = Path(str(record["calibration_path"])).expanduser().resolve()
    calib = load_three_zed_calibration(calibration_path, frame_mode=frame_mode)
    labels = list(camera_labels or meta.get("camera_labels", []) or manifest.get("camera_labels", []) or calib.keys())
    labels = [str(label) for label in labels]
    label_to_calib = (
        calibration_label_map_from_manifest(manifest, calib, labels)
        if not bool(getattr(args, "disable_serial_remap", False))
        else {label: label for label in labels}
    )
    robot_calibration = _select_robot_camera_calibration(
        raw_episode_dir=raw_episode_dir,
        manifest=manifest,
        output_frame=output_frame,
        explicit_path=record.get("robot_camera_calibration_path", meta.get("robot_camera_calibration_path", "")),
    )
    _, t_output_from_cam_by_label = _output_frame_transforms(
        calib=calib,
        labels=labels,
        label_to_calib=label_to_calib,
        output_frame=output_frame,
        robot_camera_calibration=robot_calibration,
    )

    frame = frames[int(frame_idx)]
    cameras = frame.get("cameras", {})
    contexts: list[dict] = []
    for label in labels:
        calib_label = label_to_calib[label]
        camera_frame = _load_frame_npz(raw_episode_dir, str(cameras[label]))
        depth = (
            np.asarray(camera_frame["depth_m"], dtype=np.float32)
            if "depth_m" in camera_frame
            else np.asarray(camera_frame["depth_mm"], dtype=np.float32) / 1000.0
        )
        rgb = _rgb_to_depth_shape(np.asarray(camera_frame["rgb"], dtype=np.uint8), depth.shape)
        camera_matrix = (
            np.asarray(camera_frame["camera_matrix"], dtype=np.float32).reshape(3, 3)
            if str(getattr(args, "raw_intrinsics_source", "frame")) == "frame" and "camera_matrix" in camera_frame
            else calib[calib_label].camera_matrix.astype(np.float32)
        )
        contexts.append(
            {
                "label": label,
                "rgb": rgb,
                "camera_matrix": camera_matrix,
                "t_cam_from_output": np.linalg.inv(t_output_from_cam_by_label[label]).astype(np.float32),
            }
        )
    return contexts


def _project_xyz_to_camera_pixels(
    xyz_output: np.ndarray,
    *,
    camera_matrix: np.ndarray,
    t_cam_from_output: np.ndarray,
    image_shape_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(xyz_output, dtype=np.float32).reshape(-1, 3)
    hom = np.concatenate([xyz, np.ones((xyz.shape[0], 1), dtype=np.float32)], axis=1)
    xyz_cam = (np.asarray(t_cam_from_output, dtype=np.float32).reshape(4, 4) @ hom.T).T[:, :3]
    z = xyz_cam[:, 2]
    valid = z > 1e-6
    k = np.asarray(camera_matrix, dtype=np.float32).reshape(3, 3)
    u = k[0, 0] * xyz_cam[:, 0] / np.clip(z, 1e-6, None) + k[0, 2]
    v = k[1, 1] * xyz_cam[:, 1] / np.clip(z, 1e-6, None) + k[1, 2]
    height, width = image_shape_hw
    valid &= (u >= 0.0) & (u < float(width)) & (v >= 0.0) & (v < float(height))
    pixels = np.stack([u, v], axis=1).astype(np.float32, copy=False)
    return pixels, valid


def _load_dinov2_backend(args: argparse.Namespace, dataset_dir: Path):
    import torch
    import torch.nn.functional as F

    device = torch.device(str(getattr(args, "dinov2_device", "") or getattr(args, "device", "cuda")))
    model_name = str(getattr(args, "dinov2_model_name", "dinov2_vits14"))
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.eval().to(device)
    patch_size = getattr(model, "patch_size", 14)
    if isinstance(patch_size, tuple):
        patch_size = int(patch_size[0])
    patch_size = int(patch_size)
    image_size = int(getattr(args, "dinov2_image_size", 224))
    image_size = max(patch_size, (image_size // patch_size) * patch_size)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    camera_labels = _parse_camera_labels(getattr(args, "dinov2_camera_labels", "")) or _parse_camera_labels(
        getattr(args, "camera_labels", "")
    )
    context_cache: dict[tuple[int, int], list[dict]] = {}
    feature_cache: dict[tuple[int, int, str], dict] = {}

    def image_to_patch_features(episode_idx: int, frame_idx: int, context: Mapping[str, object]) -> dict:
        cache_key = (int(episode_idx), int(frame_idx), str(context["label"]))
        if cache_key in feature_cache:
            return feature_cache[cache_key]
        rgb = np.asarray(context["rgb"], dtype=np.uint8)
        tensor = torch.from_numpy(rgb).to(device=device, dtype=torch.float32).permute(2, 0, 1)[None] / 255.0
        tensor = F.interpolate(tensor, size=(image_size, image_size), mode="bilinear", align_corners=False)
        tensor = (tensor - mean) / std
        with torch.inference_mode():
            if hasattr(model, "forward_features"):
                output = model.forward_features(tensor)
                if isinstance(output, Mapping):
                    patch_tokens = output.get("x_norm_patchtokens")
                    if patch_tokens is None:
                        patch_tokens = output.get("patch_tokens")
                else:
                    patch_tokens = output
            else:
                patch_tokens = model(tensor)
        if patch_tokens is None:
            raise RuntimeError(f"DINOv2 model {model_name} did not return patch tokens.")
        patch_tokens = patch_tokens.detach().float().cpu().numpy()
        grid_h = image_size // patch_size
        grid_w = image_size // patch_size
        patch_tokens = patch_tokens.reshape(1, grid_h, grid_w, -1)[0]
        result = {
            "features": patch_tokens.astype(np.float32, copy=False),
            "grid_h": int(grid_h),
            "grid_w": int(grid_w),
            "image_h": int(rgb.shape[0]),
            "image_w": int(rgb.shape[1]),
        }
        feature_cache[cache_key] = result
        return result

    def compute_dinov2_pointwise_cloud(
        object_cloud: np.ndarray,
        *,
        episode_idx: int,
        frame_idx: int,
        target_num_points: int,
        **_kwargs,
    ) -> np.ndarray:
        cloud = ensure_point_cloud_channels(object_cloud, channels=6)
        cloud = strip_zero_points(cloud)
        if int(target_num_points) > 0:
            cloud = resample_point_cloud(cloud, int(target_num_points))
        xyz = cloud[:, :3].astype(np.float32, copy=False)
        cache_key = (int(episode_idx), int(frame_idx))
        if cache_key not in context_cache:
            context_cache[cache_key] = _load_dinov2_camera_contexts(
                dataset_dir=dataset_dir,
                episode_idx=int(episode_idx),
                frame_idx=int(frame_idx),
                camera_labels=camera_labels,
                args=args,
            )
        contexts = context_cache[cache_key]
        if not contexts:
            raise RuntimeError("DINOv2 feature extraction requires at least one camera context.")

        feature_sum = None
        valid_count = np.zeros((xyz.shape[0], 1), dtype=np.float32)
        for context in contexts:
            patch = image_to_patch_features(int(episode_idx), int(frame_idx), context)
            pixels, valid = _project_xyz_to_camera_pixels(
                xyz,
                camera_matrix=np.asarray(context["camera_matrix"], dtype=np.float32),
                t_cam_from_output=np.asarray(context["t_cam_from_output"], dtype=np.float32),
                image_shape_hw=(int(patch["image_h"]), int(patch["image_w"])),
            )
            if feature_sum is None:
                feature_sum = np.zeros((xyz.shape[0], int(patch["features"].shape[-1])), dtype=np.float32)
            valid_idx = np.flatnonzero(valid)
            if valid_idx.shape[0] == 0:
                continue
            px = pixels[valid_idx, 0] / max(float(patch["image_w"]), 1.0)
            py = pixels[valid_idx, 1] / max(float(patch["image_h"]), 1.0)
            patch_x = np.clip((px * int(patch["grid_w"])).astype(np.int64), 0, int(patch["grid_w"]) - 1)
            patch_y = np.clip((py * int(patch["grid_h"])).astype(np.int64), 0, int(patch["grid_h"]) - 1)
            feature_sum[valid_idx] += patch["features"][patch_y, patch_x]
            valid_count[valid_idx, 0] += 1.0
        if feature_sum is None:
            raise RuntimeError("DINOv2 feature extraction did not produce any patch features.")
        valid_rows = valid_count[:, 0] > 0.0
        if np.any(valid_rows):
            feature_sum[valid_rows] /= valid_count[valid_rows]
            fallback = feature_sum[valid_rows].mean(axis=0, keepdims=True)
        else:
            fallback = np.zeros((1, feature_sum.shape[1]), dtype=np.float32)
        if np.any(~valid_rows):
            feature_sum[~valid_rows] = fallback
        return np.concatenate([xyz, feature_sum], axis=1).astype(np.float32, copy=False)

    return compute_dinov2_pointwise_cloud


def run_visualization(args: argparse.Namespace) -> dict:
    dataset_dir = resolve_dataset_dir(args)
    specs = parse_episode_frame_specs(
        episode=int(args.episode),
        frame=int(args.frame),
        episode_frames=args.episode_frames,
    )
    feature_methods = resolve_feature_methods(args)
    if str(args.color_mode) == "label" and feature_methods != ["semantic"]:
        raise ValueError("label color mode is only supported when --feature_methods is semantic.")
    output_dir = Path(args.output_dir).expanduser().resolve() / args.task_name
    output_dir.mkdir(parents=True, exist_ok=True)

    first_episode = load_hdf5(str(dataset_dir / "data" / f"episode{specs[0][0]}.hdf5"))
    scene_info_path = dataset_dir / "scene_info.json"
    scene_info = load_scene_info(str(scene_info_path)) if scene_info_path.exists() else {}
    placeholders = parse_placeholder_list(args.object_placeholders)
    if not placeholders:
        placeholders = default_placeholder_order(scene_info, first_episode)
    if not placeholders:
        raise RuntimeError("No object placeholders were provided or found in the dataset.")

    semantic_ckpts: dict[str, str] = {}
    semantic_models: dict[str, object] = {}
    compute_semantic_pointwise_cloud = None
    compute_semantic_pointwise_prediction = None
    if "semantic" in feature_methods:
        semantic_ckpts = build_semantic_checkpoint_map(args, placeholders)
        missing = [placeholder for placeholder in placeholders if placeholder not in semantic_ckpts]
        if missing:
            raise ValueError(f"Missing semantic checkpoint for placeholders: {missing}")

        backend = _load_semantic_backend(args.device)
        if len(backend) == 3:
            device, load_semantic_model, compute_semantic_pointwise_cloud = backend
            compute_semantic_pointwise_prediction = None
        else:
            device, load_semantic_model, compute_semantic_pointwise_cloud, compute_semantic_pointwise_prediction = backend
        semantic_models = {
            placeholder: load_semantic_model(semantic_ckpts[placeholder], device=device)
            for placeholder in placeholders
        }
    if str(args.semantic_forward_mode) == "reference":
        semantic_forward_kwargs = {"normal_mode": "fallback", "query_sample_mode": "random"}
    else:
        semantic_forward_kwargs = {"normal_mode": "estimated", "query_sample_mode": "fps"}
    compute_utonia_pointwise_cloud = _load_utonia_backend(args) if "utonia" in feature_methods else None
    compute_dinov2_pointwise_cloud = (
        _load_dinov2_backend(args, dataset_dir=dataset_dir) if "dinov2" in feature_methods else None
    )

    episodes: dict[int, dict] = {specs[0][0]: first_episode}
    records: list[dict] = []
    frame_records: dict[tuple[int, int], list[dict]] = {}

    for episode_idx, frame_idx in specs:
        if episode_idx not in episodes:
            episodes[episode_idx] = load_hdf5(str(dataset_dir / "data" / f"episode{episode_idx}.hdf5"))
        episode = episodes[episode_idx]
        if frame_idx < 0 or frame_idx >= int(episode["pointcloud"].shape[0]):
            raise IndexError(f"Frame {frame_idx} is out of range for episode {episode_idx}.")
        for placeholder in placeholders:
            object_cloud = _load_object_cloud_for_frame(
                episode=episode,
                frame_idx=frame_idx,
                placeholder=placeholder,
                target_num_points=int(args.object_point_num),
            )
            semantic_input_cloud = prepare_semantic_input_cloud(
                object_cloud,
                placeholder=placeholder,
                color_mode=str(args.semantic_input_color_mode),
            )
            if "semantic" in feature_methods:
                prediction = None
                if str(args.color_mode) == "label":
                    if compute_semantic_pointwise_prediction is None:
                        raise RuntimeError("Label color mode requires compute_semantic_pointwise_prediction in semantic_feature_utils.")
                    prediction = compute_semantic_pointwise_prediction(
                        semantic_models[placeholder],
                        semantic_input_cloud,
                        target_num_points=int(args.semantic_point_num),
                        placeholder=placeholder,
                        semantic_input_color_mode="stored",
                        **semantic_forward_kwargs,
                    )
                    semantic_cloud = np.asarray(prediction["point_cloud"], dtype=np.float32)
                else:
                    if compute_semantic_pointwise_cloud is None:
                        raise RuntimeError("Semantic feature method requires compute_semantic_pointwise_cloud.")
                    semantic_cloud = compute_semantic_pointwise_cloud(
                        semantic_models[placeholder],
                        semantic_input_cloud,
                        target_num_points=int(args.semantic_point_num),
                        placeholder=placeholder,
                        semantic_input_color_mode="stored",
                        **semantic_forward_kwargs,
                    )
                record = {
                    "episode": int(episode_idx),
                    "frame": int(frame_idx),
                    "placeholder": str(placeholder),
                    "feature_method": "semantic",
                    "object_cloud": object_cloud,
                    "feature_cloud": semantic_cloud,
                    "semantic_cloud": semantic_cloud,
                    "semantic_checkpoint": semantic_ckpts[placeholder],
                }
                if prediction is not None:
                    record["pred_labels"] = np.asarray(prediction["pred_labels"], dtype=np.int64)
                    record["confidence"] = np.asarray(prediction["confidence"], dtype=np.float32)
                    record["label_names"] = list(prediction.get("label_names", []))
                records.append(record)
                frame_records.setdefault((int(episode_idx), int(frame_idx)), []).append(record)
            if "utonia" in feature_methods:
                if compute_utonia_pointwise_cloud is None:
                    raise RuntimeError("Utonia feature method is not initialized.")
                utonia_cloud = compute_utonia_pointwise_cloud(
                    object_cloud,
                    target_num_points=int(getattr(args, "utonia_point_num", args.semantic_point_num)),
                    placeholder=placeholder,
                    episode_idx=int(episode_idx),
                    frame_idx=int(frame_idx),
                )
                record = {
                    "episode": int(episode_idx),
                    "frame": int(frame_idx),
                    "placeholder": str(placeholder),
                    "feature_method": "utonia",
                    "object_cloud": object_cloud,
                    "feature_cloud": np.asarray(utonia_cloud, dtype=np.float32),
                    "semantic_checkpoint": "",
                }
                records.append(record)
                frame_records.setdefault((int(episode_idx), int(frame_idx)), []).append(record)
            if "dinov2" in feature_methods:
                if compute_dinov2_pointwise_cloud is None:
                    raise RuntimeError("DINOv2 feature method is not initialized.")
                dinov2_cloud = compute_dinov2_pointwise_cloud(
                    object_cloud,
                    target_num_points=int(getattr(args, "dinov2_point_num", args.semantic_point_num)),
                    placeholder=placeholder,
                    episode_idx=int(episode_idx),
                    frame_idx=int(frame_idx),
                )
                record = {
                    "episode": int(episode_idx),
                    "frame": int(frame_idx),
                    "placeholder": str(placeholder),
                    "feature_method": "dinov2",
                    "object_cloud": object_cloud,
                    "feature_cloud": np.asarray(dinov2_cloud, dtype=np.float32),
                    "semantic_checkpoint": "",
                }
                records.append(record)
                frame_records.setdefault((int(episode_idx), int(frame_idx)), []).append(record)

    label_palette_entries: dict[str, dict] = {}
    if str(args.color_mode) == "label":
        for record in records:
            label_names = list(record.get("label_names", []))
            if not label_names:
                raise RuntimeError(
                    f"Label color mode requires canonical label names for {record['semantic_checkpoint']}."
                )
            palette = build_label_palette(len(label_names))
            record["semantic_rgb"] = labels_to_rgb(record["pred_labels"], palette)
            record["feature_rgb"] = record["semantic_rgb"]
            record["pred_label_histogram"] = labels_to_named_histogram(record["pred_labels"], label_names)
            record["mean_confidence"] = float(np.asarray(record["confidence"], dtype=np.float32).mean())
            record["pca_embedding_dim"] = None
            record["pca_num_points"] = None
            key = str(record["semantic_checkpoint"])
            label_palette_entries.setdefault(
                key,
                {
                    "semantic_checkpoint": key,
                    "label_names": label_names,
                    "colors": {
                        str(label_name): [int(value) for value in palette[index].tolist()]
                        for index, label_name in enumerate(label_names)
                    },
                },
            )
    else:
        for group_records in group_feature_records_for_pca(records, shared_pca_scope=args.shared_pca_scope).values():
            method = str(group_records[0].get("feature_method", "semantic")) if group_records else "semantic"
            use_utonia_demo_pca = (
                method == "utonia"
                and str(getattr(args, "utonia_pca_mode", "demo")) == "demo"
            )
            if use_utonia_demo_pca:
                projection = fit_utonia_demo_pca_projection(
                    [record["feature_cloud"][:, 3:] for record in group_records],
                    brightness=float(getattr(args, "utonia_pca_brightness", 1.2)),
                )
            else:
                projection = fit_shared_pca_projection([record["feature_cloud"][:, 3:] for record in group_records])
            for record in group_records:
                if use_utonia_demo_pca:
                    record["feature_rgb"] = apply_utonia_demo_pca_to_rgb(record["feature_cloud"][:, 3:], projection)
                else:
                    record["feature_rgb"] = apply_shared_pca_to_rgb(record["feature_cloud"][:, 3:], projection)
                record["semantic_rgb"] = record["feature_rgb"]
                record["pca_embedding_dim"] = int(projection["embedding_dim"])
                record["pca_num_points"] = int(projection["num_points"])
                record["pca_projection_mode"] = str(projection.get("projection_mode", "standard_3pc"))

    written_files: list[str] = []
    for record in records:
        name = _safe_placeholder_name(record["placeholder"])
        stem = f"episode{record['episode']}_frame{record['frame']}_{name}"
        method = str(record.get("feature_method", "semantic"))
        object_suffix = f"{method}_labels" if str(args.color_mode) == "label" else f"{method}_pca"
        object_ply = output_dir / f"{stem}_{object_suffix}.ply"
        colored_object = np.concatenate(
            [
                record["feature_cloud"][:, :3].astype(np.float32),
                np.asarray(record["feature_rgb"], dtype=np.float32),
            ],
            axis=1,
        )
        write_colored_ply(object_ply, colored_object)
        record["object_feature_ply"] = str(object_ply)
        if method == "semantic":
            record["object_semantic_ply"] = str(object_ply)
        written_files.append(str(object_ply))

    label_palette_path = None
    if label_palette_entries:
        label_palette_path = output_dir / "label_palette.json"
        label_palette_path.write_text(
            json.dumps(label_palette_entries, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        written_files.append(str(label_palette_path))

    scene_outputs: list[dict] = []
    for (episode_idx, frame_idx), items in frame_records.items():
        scene_cloud, scene_source = load_scene_cloud_for_frame(
            dataset_dir=dataset_dir,
            episodes=episodes,
            episode_idx=episode_idx,
            frame_idx=frame_idx,
            args=args,
        )
        original_scene = _scene_with_background(scene_cloud, background_mode="original")
        original_scene_ply = output_dir / f"episode{episode_idx}_frame{frame_idx}_scene_original_color.ply"
        write_colored_ply(original_scene_ply, original_scene)
        written_files.append(str(original_scene_ply))

        scene_entry = {
            "episode": int(episode_idx),
            "frame": int(frame_idx),
            "scene_source": str(scene_source),
            "scene_original_color_ply": str(original_scene_ply),
            "scene_point_count": int(scene_cloud.shape[0]),
            "scene_feature_overlay_plys": {},
        }
        for method in feature_methods:
            method_items = [record for record in items if str(record.get("feature_method", "semantic")) == method]
            if not method_items:
                continue
            overlay = _scene_with_background(scene_cloud, background_mode=args.background_mode)
            for record in method_items:
                overlay = compose_scene_semantic_overlay(
                    overlay,
                    record["feature_cloud"][:, :3],
                    record["feature_rgb"],
                    mode=str(args.overlay_mode),
                    max_distance=float(args.overlay_distance),
                    min_neighbors=int(args.overlay_min_neighbors),
                    cut_z_min=args.overlay_cut_z_min,
                    background_mode="original",
                )
            scene_ply = output_dir / f"episode{episode_idx}_frame{frame_idx}_scene_{method}_overlay.ply"
            write_colored_ply(scene_ply, overlay)
            written_files.append(str(scene_ply))
            scene_entry["scene_feature_overlay_plys"][method] = str(scene_ply)
            scene_entry[f"scene_{method}_overlay_ply"] = str(scene_ply)
        scene_outputs.append(scene_entry)

    summary = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "episode_frames": [{"episode": ep, "frame": fr} for ep, fr in specs],
        "object_placeholders": list(placeholders),
        "object_source": "processed_hdf5_object_pointcloud",
        "feature_methods": list(feature_methods),
        "color_mode": str(args.color_mode),
        "semantic_forward_mode": str(args.semantic_forward_mode),
        "semantic_input_color_mode": str(args.semantic_input_color_mode),
        "semantic_point_num": int(args.semantic_point_num),
        "object_point_num": int(args.object_point_num),
        "overlay_mode": str(args.overlay_mode),
        "overlay_distance": float(args.overlay_distance),
        "overlay_min_neighbors": int(args.overlay_min_neighbors),
        "overlay_cut_z_min": None if args.overlay_cut_z_min is None else float(args.overlay_cut_z_min),
        "scene_source": str(args.scene_source),
        "scene_point_num": int(args.scene_point_num),
        "camera_labels": [] if _parse_camera_labels(args.camera_labels) is None else _parse_camera_labels(args.camera_labels),
        "background_mode": str(args.background_mode),
        "shared_pca_scope": str(args.shared_pca_scope),
        "label_palette": None if label_palette_path is None else str(label_palette_path),
        "objects": [
            {
                "episode": int(record["episode"]),
                "frame": int(record["frame"]),
                "placeholder": str(record["placeholder"]),
                "feature_method": str(record.get("feature_method", "semantic")),
                "semantic_checkpoint": str(record["semantic_checkpoint"]),
                "object_point_count": int(record["object_cloud"].shape[0]),
                "feature_point_count": int(record["feature_cloud"].shape[0]),
                "semantic_point_count": int(record["feature_cloud"].shape[0]),
                "object_feature_ply": str(record["object_feature_ply"]),
                "object_semantic_ply": record.get("object_semantic_ply"),
                "pca_embedding_dim": None
                if record["pca_embedding_dim"] is None
                else int(record["pca_embedding_dim"]),
                "pca_num_points": None if record["pca_num_points"] is None else int(record["pca_num_points"]),
                "pca_projection_mode": record.get("pca_projection_mode"),
                "pred_label_histogram": record.get("pred_label_histogram"),
                "mean_confidence": record.get("mean_confidence"),
            }
            for record in records
        ],
        "scenes": scene_outputs,
        "files": written_files,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[semantic-field-viz] wrote summary: {summary_path}")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize semantic-field embeddings on processed RoboTwin object point clouds.",
    )
    parser.add_argument("--task_name", default="view_pour_water_new")
    parser.add_argument("--task_config", default="demo_real_zed_sam2_objpc_global")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--dataset_dir", default="")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument(
        "--episode_frames",
        nargs="*",
        default=["0:0", "1:0", "2:0", "3:0", "4:0", "5:0"],
        help="Optional explicit selections such as 0:10 0:20 1:5. Overrides --episode/--frame.",
    )
    parser.add_argument("--object_placeholders", default="{A}")
    parser.add_argument(
        "--feature_methods",
        nargs="*",
        default=["semantic"],
        help=(
            "Feature methods to visualize. Use one or more of: semantic, utonia, dinov2. "
            "Comma-separated values are accepted. PCA color spaces are fit independently per method."
        ),
    )
    parser.add_argument(
        "--run_semantic",
        action="store_true",
        default=False,
        help="Run only methods selected by individual --run_* switches; include the semantic feature backend.",
    )
    parser.add_argument(
        "--run_utonia",
        action="store_true",
        default=False,
        help="Run only methods selected by individual --run_* switches; include the Utonia object-feature backend.",
    )
    parser.add_argument(
        "--run_dinov2",
        action="store_true",
        default=True,
        help="Run only methods selected by individual --run_* switches; include the DINOv2 RGB-feature backend.",
    )
    parser.add_argument("--semantic_ckpt", default="")
    parser.add_argument("--semantic_ckpt_A", default="/home/zheng/model/semantic/mug.pt")
    parser.add_argument("--semantic_ckpt_B", default="/home/zheng/model/semantic/mug.pt")
    parser.add_argument(
        "--semantic_ckpts",
        default="",
        help="Comma-separated placeholder mapping, e.g. '{A}=/path/a.pt,{B}=/path/b.pt'.",
    )
    parser.add_argument(
        "--semantic_point_num",
        type=int,
        default=5000,
        help="Semantic query points per object. Default matches visualize_utonia_universal_field.py point-cloud mode.",
    )
    parser.add_argument(
        "--object_point_num",
        type=int,
        default=0,
        help="Object support points before semantic query. 0 keeps the stored object cloud count.",
    )
    parser.add_argument(
        "--color_mode",
        choices=["label", "pca"],
        default="pca",
        help="Use predicted semantic labels for discrete colors, or PCA colors for continuous embeddings.",
    )
    parser.add_argument(
        "--semantic_forward_mode",
        choices=["reference", "dp3"],
        default="reference",
        help=(
            "reference matches visualize_utonia_universal_field.py direct point-cloud mode "
            "(fallback normals + random query sampling); dp3 uses the original DP3 helper path."
        ),
    )
    parser.add_argument(
        "--semantic_input_color_mode",
        choices=["debug_placeholder", "stored_scaled", "stored"],
        default="debug_placeholder",
        help=(
            "Colors fed to Utonia. debug_placeholder matches the debug PLYs used by "
            "visualize_utonia_universal_field.py; stored_scaled uses HDF5 RGB scaled to 0-255; stored keeps HDF5 RGB as-is."
        ),
    )
    parser.add_argument("--utonia_point_num", type=int, default=20000)
    parser.add_argument("--utonia_device", default="")
    parser.add_argument("--utonia_root", default="/home/zheng/github/Utonia")
    parser.add_argument("--utonia_repo_id", default="Pointcept/Utonia")
    parser.add_argument("--utonia_sample_seed", type=int, default=73)
    parser.add_argument(
        "--utonia_grid_size",
        type=float,
        # default=0.01,
        default=0.005,
        help=(
            "Voxel grid size used by the Utonia GridSample transform. "
            "Smaller values keep denser visualized points, e.g. 0.003."
        ),
    )
    parser.add_argument(
        "--utonia_input_color_mode",
        choices=["debug_placeholder", "stored_scaled", "stored", "zero"],
        default="stored_scaled",
    )
    parser.add_argument("--utonia_pca_mode", choices=["demo", "standard"], default="demo")
    parser.add_argument("--utonia_pca_brightness", type=float, default=1.2)
    parser.add_argument("--dinov2_point_num", type=int, default=5000)
    parser.add_argument("--dinov2_device", default="")
    parser.add_argument("--dinov2_model_name", default="dinov2_vits14")
    parser.add_argument(
        "--dinov2_camera_labels",
        default="",
        help="Comma-separated raw camera labels used to project object points into RGB frames for DINOv2.",
    )
    parser.add_argument("--dinov2_image_size", type=int, default=224)
    parser.add_argument(
        "--overlay_mode",
        choices=["cut_replace", "append", "replace_nearest"],
        default="cut_replace",
        help=(
            "cut_replace removes scene points matching the segmented object and inserts colored semantic object points. "
            "append keeps the full scene unchanged and adds colored semantic object points. "
            "replace_nearest recolors nearby scene points and may color table points near the object."
        ),
    )
    parser.add_argument("--overlay_distance", type=float, default=0.02)
    parser.add_argument(
        "--overlay_cut_z_min",
        type=float,
        default=0.005,
        help=(
            "For cut_replace only: only scene points with z >= this value are eligible to be removed. "
            "Use this to preserve table points below the object. The z value is in the current scene/output frame."
        ),
    )
    parser.add_argument(
        "--overlay_min_neighbors",
        type=int,
        default=0,
        help=(
            "Minimum number of semantic object points within --overlay_distance before a scene point is recolored. "
            "Use 1 for the old nearest-neighbor behavior."
        ),
    )
    parser.add_argument(
        "--scene_source",
        choices=["auto", "raw_full", "hdf5"],
        default="auto",
        help=(
            "Background scene overlay source only. Object semantic clouds always come from processed "
            "HDF5 /object_pointcloud so workspace-constrained segmentation is preserved. "
            "auto uses raw full RGB/depth for the scene when processed real-ZED meta is available."
        ),
    )
    parser.add_argument(
        "--scene_point_num",
        type=int,
        default=800000,
        help="Scene points to save. 0 keeps all points for raw_full and keeps HDF5 point count for hdf5.",
    )
    parser.add_argument("--camera_labels", default="", help="Comma-separated raw camera labels for raw_full scene export.")
    parser.add_argument("--min_depth_m", type=float, default=0.3)
    parser.add_argument("--max_depth_m", type=float, default=3.0)
    parser.add_argument("--raw_intrinsics_source", choices=["frame", "calibration"], default="frame")
    parser.add_argument("--disable_serial_remap", action="store_true", default=False)
    parser.add_argument("--background_mode", choices=["original", "gray", "black"], default="original")
    parser.add_argument("--shared_pca_scope", choices=["checkpoint", "placeholder", "all"], default="checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="outputs/semantic_field_dataset_viz")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_visualization(args)


if __name__ == "__main__":
    main()
