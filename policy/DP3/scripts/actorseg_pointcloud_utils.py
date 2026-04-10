import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
from PIL import ImageColor

from object_pointcloud_utils import (
    ensure_point_cloud_channels,
    frame_image,
    frame_matrix,
    merge_object_point_clouds,
    resample_point_cloud,
    strip_zero_points,
)


def _read_optional(group: h5py.Group, path: str):
    if path not in group:
        return None
    return group[path][()]


def load_hdf5_with_actorseg(
    dataset_path: str,
    *,
    camera_names: Sequence[str],
    segmentation_key: str = "actor_segmentation",
) -> dict:
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset does not exist at {dataset_path}")

    with h5py.File(dataset_path, "r") as root:
        data = {
            "vector": root["/joint_action/vector"][()],
            "pointcloud": root["/pointcloud"][()],
            "cameras": {},
        }
        if "object_pointcloud" in root:
            object_group = root["/object_pointcloud"]
            data["object_pointcloud"] = {str(key): object_group[key][()] for key in object_group.keys()}
        else:
            data["object_pointcloud"] = {}

        for camera_name in camera_names:
            group_path = f"/observation/{camera_name}"
            if group_path not in root:
                continue
            camera_group = root[group_path]
            data["cameras"][camera_name] = {
                segmentation_key: _read_optional(camera_group, segmentation_key),
                "intrinsic_cv": _read_optional(camera_group, "intrinsic_cv"),
                "extrinsic_cv": _read_optional(camera_group, "extrinsic_cv"),
                "cam2world_gl": _read_optional(camera_group, "cam2world_gl"),
            }
    return data


def parse_camera_list(text: Optional[str], *, default: Iterable[str] = ("head_camera", "front_camera")) -> List[str]:
    if text is None or str(text).strip() == "":
        return [str(item) for item in default]
    return [item.strip() for item in str(text).split(",") if item.strip()]


def parse_episode_actor_id_map(
    scene_info: dict,
    episode_idx: int,
    placeholders: Sequence[str],
) -> Dict[str, List[int]]:
    episode_info = scene_info.get(f"episode_{episode_idx}", {}) if isinstance(scene_info, dict) else {}
    target_info = episode_info.get("object_pointcloud", {}).get("targets", {}) if isinstance(episode_info, dict) else {}
    result: Dict[str, List[int]] = {}
    for placeholder in placeholders:
        actor_ids = target_info.get(placeholder, {}).get("actor_ids", [])
        result[str(placeholder)] = [int(actor_id) for actor_id in actor_ids]
    return result


def _build_color_palette() -> np.ndarray:
    colormap = sorted(set(ImageColor.colormap.values()))
    return np.array([ImageColor.getrgb(color) for color in colormap], dtype=np.uint8)


SEGMENTATION_COLOR_PALETTE = _build_color_palette()


def actor_ids_to_colors(actor_ids: Sequence[int]) -> np.ndarray:
    actor_ids_arr = np.asarray(list(actor_ids), dtype=np.int64)
    if actor_ids_arr.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    if np.any(actor_ids_arr < 0) or np.any(actor_ids_arr >= len(SEGMENTATION_COLOR_PALETTE)):
        raise ValueError(
            f"Actor ids {actor_ids_arr.tolist()} exceed the supported palette range "
            f"[0, {len(SEGMENTATION_COLOR_PALETTE) - 1}] used by stored simulator segmentations."
        )
    return SEGMENTATION_COLOR_PALETTE[actor_ids_arr]


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

    pts_cam = pts_cam[valid]
    uv_float = (intrinsic_cv @ pts_cam.T).T
    uv_float = uv_float[:, :2] / np.clip(uv_float[:, 2:3], 1e-6, None)
    u = np.round(uv_float[:, 0]).astype(np.int64)
    v = np.round(uv_float[:, 1]).astype(np.int64)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    valid_idx = np.nonzero(valid)[0]
    valid[valid_idx] &= inside
    uv[valid_idx[inside], 0] = u[inside]
    uv[valid_idx[inside], 1] = v[inside]
    return uv, valid


def _select_points_from_actorseg(
    scene_point_cloud: np.ndarray,
    *,
    segmentation_image: np.ndarray,
    intrinsic_cv: np.ndarray,
    extrinsic_cv: np.ndarray,
    actor_ids: Sequence[int],
) -> Tuple[Optional[np.ndarray], Dict[str, object]]:
    scene_point_cloud = ensure_point_cloud_channels(scene_point_cloud, channels=6)
    scene_point_cloud = strip_zero_points(scene_point_cloud)
    if len(scene_point_cloud) == 0:
        return None, {"mode": "empty_scene"}

    if len(actor_ids) == 0:
        return None, {"mode": "missing_actor_ids"}

    seg_image = np.asarray(segmentation_image, dtype=np.uint8)
    if seg_image.ndim != 3 or seg_image.shape[-1] < 3:
        return None, {"mode": "invalid_segmentation_shape", "shape": list(seg_image.shape)}

    uv, valid = _project_points_with_matrix(
        scene_point_cloud[:, :3].astype(np.float32),
        intrinsic_cv=np.asarray(intrinsic_cv, dtype=np.float32),
        extrinsic_cv=np.asarray(extrinsic_cv, dtype=np.float32),
        image_hw=seg_image.shape[:2],
    )
    if not np.any(valid):
        return None, {"mode": "no_visible_points"}

    actor_colors = actor_ids_to_colors(actor_ids)
    valid_uv = uv[valid]
    visible_colors = seg_image[valid_uv[:, 1], valid_uv[:, 0], :3]
    color_match = np.any(
        np.all(visible_colors[:, None, :] == actor_colors[None, :, :], axis=-1),
        axis=1,
    )
    point_mask = np.zeros((len(scene_point_cloud),), dtype=bool)
    point_mask[np.nonzero(valid)[0]] = color_match
    selected = scene_point_cloud[point_mask]
    if len(selected) == 0:
        return None, {
            "mode": "actorseg_empty",
            "visible_points": int(valid.sum()),
            "actor_ids": [int(actor_id) for actor_id in actor_ids],
        }

    return selected.astype(np.float32), {
        "mode": "actorseg_projected",
        "visible_points": int(valid.sum()),
        "selected_point_count": int(len(selected)),
        "actor_ids": [int(actor_id) for actor_id in actor_ids],
    }


def extract_placeholder_point_cloud_actorseg(
    episode: dict,
    *,
    frame_idx: int,
    placeholder: str,
    actor_ids: Sequence[int],
    camera_names: Sequence[str],
    target_num_points: int,
    segmentation_key: str = "actor_segmentation",
) -> Tuple[np.ndarray, Dict[str, object]]:
    scene_pc = ensure_point_cloud_channels(episode["pointcloud"][frame_idx], channels=6)
    scene_pc = strip_zero_points(scene_pc)
    if len(scene_pc) == 0:
        return np.zeros((target_num_points, 6), dtype=np.float32), {"mode": "empty_scene"}

    selected_clouds: List[np.ndarray] = []
    camera_meta: Dict[str, Dict[str, object]] = {}
    for camera_name in camera_names:
        camera_info = episode["cameras"].get(camera_name)
        if camera_info is None:
            continue
        seg_all = camera_info.get(segmentation_key)
        intrinsic_cv_all = camera_info.get("intrinsic_cv")
        extrinsic_cv_all = camera_info.get("extrinsic_cv")
        if seg_all is None or intrinsic_cv_all is None or extrinsic_cv_all is None:
            camera_meta[camera_name] = {"mode": "missing_camera_data"}
            continue

        segmentation = frame_image(seg_all, frame_idx)
        intrinsic_cv = frame_matrix(intrinsic_cv_all, frame_idx)
        extrinsic_cv = frame_matrix(extrinsic_cv_all, frame_idx)
        if segmentation is None or intrinsic_cv is None or extrinsic_cv is None:
            camera_meta[camera_name] = {"mode": "missing_camera_frame"}
            continue

        selected_points, meta = _select_points_from_actorseg(
            scene_point_cloud=scene_pc,
            segmentation_image=segmentation,
            intrinsic_cv=intrinsic_cv,
            extrinsic_cv=extrinsic_cv,
            actor_ids=actor_ids,
        )
        camera_meta[camera_name] = meta
        if selected_points is not None:
            selected_clouds.append(selected_points)

    if len(selected_clouds) == 0:
        return np.zeros((target_num_points, 6), dtype=np.float32), {
            "mode": "actorseg_empty",
            "placeholder": str(placeholder),
            "actor_ids": [int(actor_id) for actor_id in actor_ids],
            "cameras": camera_meta,
        }

    merged = merge_object_point_clouds(selected_clouds, target_num_points=int(target_num_points))
    return resample_point_cloud(merged, int(target_num_points)), {
        "mode": "actorseg_projected",
        "placeholder": str(placeholder),
        "actor_ids": [int(actor_id) for actor_id in actor_ids],
        "camera_count": int(len(selected_clouds)),
        "cameras": camera_meta,
    }


def extract_placeholder_point_cloud_actorseg_online(
    observation: dict,
    *,
    placeholder: str,
    actor_ids: Sequence[int],
    camera_names: Sequence[str],
    target_num_points: int,
    segmentation_key: str = "actor_segmentation",
) -> Tuple[np.ndarray, Dict[str, object]]:
    scene_pc = ensure_point_cloud_channels(observation["pointcloud"], channels=6)
    scene_pc = strip_zero_points(scene_pc)
    if len(scene_pc) == 0:
        return np.zeros((target_num_points, 6), dtype=np.float32), {"mode": "empty_scene"}

    camera_obs = observation.get("observation", {})
    selected_clouds: List[np.ndarray] = []
    camera_meta: Dict[str, Dict[str, object]] = {}
    for camera_name in camera_names:
        camera_info = camera_obs.get(camera_name)
        if camera_info is None:
            continue
        segmentation = camera_info.get(segmentation_key)
        intrinsic_cv = camera_info.get("intrinsic_cv")
        extrinsic_cv = camera_info.get("extrinsic_cv")
        if segmentation is None or intrinsic_cv is None or extrinsic_cv is None:
            camera_meta[camera_name] = {"mode": "missing_camera_data"}
            continue

        selected_points, meta = _select_points_from_actorseg(
            scene_point_cloud=scene_pc,
            segmentation_image=np.asarray(segmentation, dtype=np.uint8),
            intrinsic_cv=np.asarray(intrinsic_cv, dtype=np.float32),
            extrinsic_cv=np.asarray(extrinsic_cv, dtype=np.float32),
            actor_ids=actor_ids,
        )
        camera_meta[camera_name] = meta
        if selected_points is not None:
            selected_clouds.append(selected_points)

    if len(selected_clouds) == 0:
        return np.zeros((target_num_points, 6), dtype=np.float32), {
            "mode": "actorseg_empty",
            "placeholder": str(placeholder),
            "actor_ids": [int(actor_id) for actor_id in actor_ids],
            "cameras": camera_meta,
        }

    merged = merge_object_point_clouds(selected_clouds, target_num_points=int(target_num_points))
    return resample_point_cloud(merged, int(target_num_points)), {
        "mode": "actorseg_projected",
        "placeholder": str(placeholder),
        "actor_ids": [int(actor_id) for actor_id in actor_ids],
        "camera_count": int(len(selected_clouds)),
        "cameras": camera_meta,
    }


__all__ = [
    "SEGMENTATION_COLOR_PALETTE",
    "actor_ids_to_colors",
    "extract_placeholder_point_cloud_actorseg",
    "extract_placeholder_point_cloud_actorseg_online",
    "load_hdf5_with_actorseg",
    "parse_camera_list",
    "parse_episode_actor_id_map",
]
