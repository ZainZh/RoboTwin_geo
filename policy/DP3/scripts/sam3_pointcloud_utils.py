import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import h5py
import numpy as np

from ndf_feature_utils import parse_asset_spec
from object_pointcloud_utils import (
    ensure_point_cloud_channels,
    frame_matrix,
    load_scene_info,
    merge_object_point_clouds,
    parse_target_extents,
    resample_point_cloud,
    strip_zero_points,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
YOLO_CONFIG_DIR = REPO_ROOT / "policy" / "DP3" / ".ultralytics"
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))
YOLO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_CLIP_ROOTS = [
    Path("/home/zheng/github/O3Afford/models"),
]
for candidate in LOCAL_CLIP_ROOTS:
    if (candidate / "clip" / "__init__.py").is_file() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


def _ensure_clip_tokenizer_callable():
    try:
        import clip.simple_tokenizer as clip_simple_tokenizer
        import torch
    except Exception:
        return

    tokenizer_cls = getattr(clip_simple_tokenizer, "SimpleTokenizer", None)
    if tokenizer_cls is None or "__call__" in tokenizer_cls.__dict__:
        return

    def _call(self, texts, context_length: int = 77, truncate: bool = False):
        if isinstance(texts, str):
            texts = [texts]

        sot_token = self.encoder["<|startoftext|>"]
        eot_token = self.encoder["<|endoftext|>"]
        result = torch.zeros(len(texts), int(context_length), dtype=torch.int)

        for row_idx, text in enumerate(texts):
            tokens = [sot_token] + self.encode(text) + [eot_token]
            if len(tokens) > int(context_length):
                if truncate:
                    tokens = tokens[: int(context_length)]
                    tokens[-1] = eot_token
                else:
                    raise RuntimeError(f"Input {text!r} is too long for context length {context_length}")
            result[row_idx, : len(tokens)] = torch.tensor(tokens, dtype=torch.int)
        return result

    tokenizer_cls.__call__ = _call


_ensure_clip_tokenizer_callable()


def decode_jpeg_frame(encoded_frame) -> np.ndarray:
    encoded = bytes(encoded_frame).rstrip(b"\0")
    if len(encoded) == 0:
        raise ValueError("Empty encoded rgb frame.")
    arr = np.frombuffer(encoded, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode rgb frame.")
    return image


def _read_optional(group: h5py.Group, path: str):
    if path not in group:
        return None
    return group[path][()]


def load_hdf5_with_cameras(dataset_path: str, camera_names: Sequence[str]) -> dict:
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset does not exist at {dataset_path}")

    with h5py.File(dataset_path, "r") as root:
        data = {
            "vector": root["/joint_action/vector"][()],
            "pointcloud": root["/pointcloud"][()],
            "cameras": {},
        }
        for camera_name in camera_names:
            group_path = f"/observation/{camera_name}"
            if group_path not in root:
                continue
            camera_group = root[group_path]
            data["cameras"][camera_name] = {
                "rgb": _read_optional(camera_group, "rgb"),
                "intrinsic_cv": _read_optional(camera_group, "intrinsic_cv"),
                "extrinsic_cv": _read_optional(camera_group, "extrinsic_cv"),
                "cam2world_gl": _read_optional(camera_group, "cam2world_gl"),
            }
    return data


def parse_camera_list(text: Optional[str], *, default: Iterable[str] = ("head_camera", "front_camera")) -> List[str]:
    if text is None or str(text).strip() == "":
        return [str(item) for item in default]
    return [item.strip() for item in str(text).split(",") if item.strip()]


def parse_prompt_map(text: Optional[str]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    if text is None:
        return result
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Expected PLACEHOLDER=TEXT prompt mapping, got {item!r}")
        placeholder, prompt = item.split("=", 1)
        placeholder = placeholder.strip()
        prompt = prompt.strip()
        if placeholder and prompt:
            result[placeholder] = prompt
    return result


def target_spec_to_prompt(target_spec) -> Optional[str]:
    if isinstance(target_spec, (list, tuple)):
        for item in target_spec:
            prompt = target_spec_to_prompt(item)
            if prompt:
                return prompt
        return None
    if target_spec is None:
        return None
    text = str(target_spec).strip()
    if not text:
        return None
    text = text.split(".", 1)[0]
    text = text.split("[", 1)[0]
    text = text.replace("_", " ").strip()
    return text or None


def asset_prompt_from_spec(asset_spec: Optional[str]) -> Optional[str]:
    model_name, _ = parse_asset_spec(asset_spec)
    if not model_name:
        return None
    name = str(model_name).split("/")[-1]
    if "_" in name:
        name = name.split("_", 1)[1]
    name = name.replace("_", " ").strip()
    return name or None


def build_placeholder_prompt_map(
    scene_info: dict,
    episode_idx: int,
    placeholders: Sequence[str],
    prompt_overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    episode_info = scene_info.get(f"episode_{episode_idx}", {}) if isinstance(scene_info, dict) else {}
    info_dict = episode_info.get("info", {}) if isinstance(episode_info, dict) else {}
    prompt_overrides = dict(prompt_overrides or {})

    result: Dict[str, str] = {}
    for placeholder in placeholders:
        if placeholder in prompt_overrides:
            result[placeholder] = str(prompt_overrides[placeholder])
            continue
        prompt = asset_prompt_from_spec(info_dict.get(placeholder))
        if prompt is None:
            prompt = placeholder.strip("{}")
        result[placeholder] = str(prompt)
    return result


def build_placeholder_prompt_map_from_targets(
    placeholders: Sequence[str],
    target_specs: Dict[str, object],
    *,
    prompt_overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    result: Dict[str, str] = {}
    prompt_overrides = dict(prompt_overrides or {})
    for placeholder in placeholders:
        if placeholder in prompt_overrides:
            result[placeholder] = str(prompt_overrides[placeholder])
            continue
        prompt = target_spec_to_prompt(target_specs.get(placeholder))
        if prompt is None:
            prompt = placeholder.strip("{}")
        result[placeholder] = str(prompt)
    return result


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


def _coerce_masks_and_boxes(masks, boxes) -> Tuple[np.ndarray, np.ndarray]:
    if masks is None or boxes is None:
        return np.zeros((0, 1, 1), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)
    if hasattr(masks, "detach"):
        masks = masks.detach().cpu().numpy()
    else:
        masks = np.asarray(masks)
    if hasattr(boxes, "detach"):
        boxes = boxes.detach().cpu().numpy()
    else:
        boxes = np.asarray(boxes)
    if masks.ndim == 2:
        masks = masks[None, ...]
    if boxes.ndim == 1:
        boxes = boxes[None, ...]
    if boxes.shape[-1] > 4:
        boxes = boxes[..., :4]
    return masks.astype(np.float32), boxes.astype(np.float32)


def _score_mask_points(
    points: np.ndarray,
    *,
    target_extents: Optional[np.ndarray],
    prev_centroid: Optional[np.ndarray],
) -> float:
    if len(points) == 0:
        return np.inf
    xyz = points[:, :3]
    score = 0.0
    if target_extents is not None:
        obs_ext = np.sort(np.maximum(xyz.max(axis=0) - xyz.min(axis=0), 1e-4))
        tgt_ext = np.sort(np.maximum(np.asarray(target_extents, dtype=np.float32), 1e-4))
        score += float(np.mean(np.abs(np.log(obs_ext / tgt_ext))))
    else:
        score -= 0.0005 * float(len(points))
    if prev_centroid is not None:
        centroid = xyz.mean(axis=0)
        score += 0.5 * float(np.linalg.norm(centroid - prev_centroid))
    else:
        score -= 0.0005 * float(len(points))
    return score


def _select_mask_candidate(
    scene_point_cloud: np.ndarray,
    *,
    masks: np.ndarray,
    boxes: np.ndarray,
    intrinsic_cv: np.ndarray,
    extrinsic_cv: np.ndarray,
    target_extents: Optional[np.ndarray],
    prev_centroid: Optional[np.ndarray],
    min_points: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, object]]:
    scene_point_cloud = ensure_point_cloud_channels(scene_point_cloud, channels=6)
    scene_point_cloud = strip_zero_points(scene_point_cloud)
    if len(scene_point_cloud) == 0:
        return None, None, {"mode": "empty_scene"}

    masks, boxes = _coerce_masks_and_boxes(masks, boxes)
    if len(masks) == 0:
        return None, None, {"mode": "no_mask"}

    image_hw = masks.shape[-2:]
    uv, valid = _project_points_with_matrix(
        scene_point_cloud[:, :3].astype(np.float32),
        intrinsic_cv=np.asarray(intrinsic_cv, dtype=np.float32),
        extrinsic_cv=np.asarray(extrinsic_cv, dtype=np.float32),
        image_hw=image_hw,
    )
    if not np.any(valid):
        return None, None, {"mode": "no_visible_points"}

    best_points = None
    best_box = None
    best_score = np.inf
    best_count = 0
    best_idx = None
    for idx in range(len(masks)):
        mask = masks[idx] > 0.5
        point_mask = np.zeros((len(scene_point_cloud),), dtype=bool)
        valid_uv = uv[valid]
        point_mask[np.nonzero(valid)[0]] = mask[valid_uv[:, 1], valid_uv[:, 0]]
        selected = scene_point_cloud[point_mask]
        if len(selected) < int(min_points):
            continue
        score = _score_mask_points(
            selected,
            target_extents=target_extents,
            prev_centroid=prev_centroid,
        )
        if score < best_score or (np.isclose(score, best_score) and len(selected) > best_count):
            best_score = score
            best_points = selected
            best_box = boxes[idx]
            best_count = int(len(selected))
            best_idx = int(idx)

    if best_points is None:
        return None, None, {
            "mode": "no_candidate_points",
            "num_masks": int(len(masks)),
        }

    return best_points.astype(np.float32), np.asarray(best_box, dtype=np.float32), {
        "mode": "mask_selected",
        "num_masks": int(len(masks)),
        "selected_mask_idx": int(best_idx),
        "selected_point_count": int(best_count),
    }


class SAM3ProjectiveTracker:

    def __init__(
        self,
        *,
        model_path: str,
        conf: float = 0.50,
        verbose: bool = False,
    ):
        try:
            from ultralytics.models.sam import SAM3SemanticPredictor
        except ImportError as exc:
            raise ImportError(
                "SAM3 preprocessing requires ultralytics with SAM3SemanticPredictor installed "
                "in the current Python environment."
            ) from exc

        overrides = dict(conf=float(conf), task="segment", mode="predict", model=str(model_path), verbose=bool(verbose), imgsz =644)
        self.feature_predictor = SAM3SemanticPredictor(overrides=overrides)
        self.query_predictor = SAM3SemanticPredictor(overrides=overrides)
        self.query_predictor.setup_model()
        self._src_shape = None

    def _set_image(self, image: np.ndarray):
        self._src_shape = image.shape[:2]
        try:
            self.feature_predictor.set_image(image)
            return
        except Exception:
            pass

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            temp_path = f.name
        try:
            cv2.imwrite(temp_path, image)
            self.feature_predictor.set_image(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def infer(self, image: np.ndarray, *, text: Optional[str] = None, bbox: Optional[Sequence[float]] = None):
        self._set_image(image)
        kwargs = {}
        if bbox is not None:
            kwargs["bboxes"] = [list(map(float, bbox))]
        elif text is not None:
            kwargs["text"] = [str(text)]
        else:
            raise ValueError("Either text or bbox prompt must be provided to SAM3.")
        masks, boxes = self.query_predictor.inference_features(
            self.feature_predictor.features,
            src_shape=self._src_shape,
            **kwargs,
        )
        return _coerce_masks_and_boxes(masks, boxes)


@dataclass
class Sam3TrackingState:
    prev_box: Optional[np.ndarray] = None
    prev_centroid: Optional[np.ndarray] = None


def extract_placeholder_point_cloud_sam3(
    episode: dict,
    *,
    frame_idx: int,
    placeholder: str,
    prompt: str,
    camera_names: Sequence[str],
    tracker: SAM3ProjectiveTracker,
    tracking_state_by_camera: Dict[str, Sam3TrackingState],
    target_num_points: int,
    target_extents: Optional[np.ndarray] = None,
    min_mask_points: int = 16,
    text_refresh_every: int = 15,
) -> Tuple[np.ndarray, Dict[str, object]]:
    scene_pc = ensure_point_cloud_channels(episode["pointcloud"][frame_idx], channels=6)
    scene_pc = strip_zero_points(scene_pc)
    if len(scene_pc) == 0:
        return np.zeros((target_num_points, 6), dtype=np.float32), {"mode": "empty_scene"}

    selected_clouds: List[np.ndarray] = []
    camera_meta: Dict[str, Dict[str, object]] = {}
    refresh_by_text = int(text_refresh_every) > 0 and (frame_idx % int(text_refresh_every) == 0)

    for camera_name in camera_names:
        camera_info = episode["cameras"].get(camera_name)
        if camera_info is None:
            continue
        rgb_all = camera_info.get("rgb")
        intrinsic_cv_all = camera_info.get("intrinsic_cv")
        extrinsic_cv_all = camera_info.get("extrinsic_cv")
        if rgb_all is None or intrinsic_cv_all is None or extrinsic_cv_all is None:
            camera_meta[camera_name] = {"mode": "missing_camera_data"}
            continue

        image = decode_jpeg_frame(rgb_all[frame_idx])
        intrinsic_cv = frame_matrix(intrinsic_cv_all, frame_idx)
        extrinsic_cv = frame_matrix(extrinsic_cv_all, frame_idx)
        if intrinsic_cv is None or extrinsic_cv is None:
            camera_meta[camera_name] = {"mode": "missing_camera_matrices"}
            continue

        tracking_state = tracking_state_by_camera.setdefault(camera_name, Sam3TrackingState())
        candidate_modes: List[Tuple[str, dict]] = []
        if tracking_state.prev_box is not None and not refresh_by_text:
            candidate_modes.append(("bbox", {"bbox": tracking_state.prev_box}))
        candidate_modes.append(("text", {"text": prompt}))

        selected_points = None
        selected_box = None
        selected_meta = None
        for prompt_mode, prompt_kwargs in candidate_modes:
            masks, boxes = tracker.infer(image, **prompt_kwargs)
            points, box, meta = _select_mask_candidate(
                scene_point_cloud=scene_pc,
                masks=masks,
                boxes=boxes,
                intrinsic_cv=intrinsic_cv,
                extrinsic_cv=extrinsic_cv,
                target_extents=target_extents,
                prev_centroid=tracking_state.prev_centroid,
                min_points=int(min_mask_points),
            )
            if points is None:
                selected_meta = {"prompt_mode": prompt_mode, **meta}
                continue
            selected_points = points
            selected_box = box
            selected_meta = {"prompt_mode": prompt_mode, **meta}
            break

        if selected_points is None:
            camera_meta[camera_name] = selected_meta or {"mode": "sam3_failed"}
            continue

        selected_clouds.append(selected_points)
        tracking_state.prev_box = selected_box
        centroid = selected_points[:, :3].mean(axis=0) if len(selected_points) > 0 else None
        if centroid is not None:
            tracking_state.prev_centroid = centroid.astype(np.float32)
        camera_meta[camera_name] = selected_meta

    if len(selected_clouds) == 0:
        return np.zeros((target_num_points, 6), dtype=np.float32), {
            "mode": "sam3_empty",
            "prompt": prompt,
            "cameras": camera_meta,
        }

    merged = merge_object_point_clouds(selected_clouds, target_num_points=int(target_num_points))
    return resample_point_cloud(merged, int(target_num_points)), {
        "mode": "sam3_projected",
        "prompt": prompt,
        "camera_count": int(len(selected_clouds)),
        "cameras": camera_meta,
    }


def extract_placeholder_point_cloud_sam3_online(
    observation: dict,
    *,
    placeholder: str,
    prompt: str,
    camera_names: Sequence[str],
    tracker: SAM3ProjectiveTracker,
    tracking_state_by_camera: Dict[str, Sam3TrackingState],
    target_num_points: int,
    target_extents: Optional[np.ndarray] = None,
    min_mask_points: int = 16,
    text_refresh_every: int = 15,
    frame_idx: int = 0,
) -> Tuple[np.ndarray, Dict[str, object]]:
    scene_pc = ensure_point_cloud_channels(observation["pointcloud"], channels=6)
    scene_pc = strip_zero_points(scene_pc)
    if len(scene_pc) == 0:
        return np.zeros((target_num_points, 6), dtype=np.float32), {"mode": "empty_scene"}

    camera_obs = observation.get("observation", {})
    selected_clouds: List[np.ndarray] = []
    camera_meta: Dict[str, Dict[str, object]] = {}
    refresh_by_text = int(text_refresh_every) > 0 and (int(frame_idx) % int(text_refresh_every) == 0)

    for camera_name in camera_names:
        camera_info = camera_obs.get(camera_name)
        if camera_info is None:
            continue
        image = camera_info.get("rgb")
        intrinsic_cv = camera_info.get("intrinsic_cv")
        extrinsic_cv = camera_info.get("extrinsic_cv")
        if image is None or intrinsic_cv is None or extrinsic_cv is None:
            camera_meta[camera_name] = {"mode": "missing_camera_data"}
            continue

        tracking_state = tracking_state_by_camera.setdefault(camera_name, Sam3TrackingState())
        candidate_modes: List[Tuple[str, dict]] = []
        if tracking_state.prev_box is not None and not refresh_by_text:
            candidate_modes.append(("bbox", {"bbox": tracking_state.prev_box}))
        candidate_modes.append(("text", {"text": prompt}))

        selected_points = None
        selected_box = None
        selected_meta = None
        for prompt_mode, prompt_kwargs in candidate_modes:
            masks, boxes = tracker.infer(image, **prompt_kwargs)
            points, box, meta = _select_mask_candidate(
                scene_point_cloud=scene_pc,
                masks=masks,
                boxes=boxes,
                intrinsic_cv=np.asarray(intrinsic_cv, dtype=np.float32),
                extrinsic_cv=np.asarray(extrinsic_cv, dtype=np.float32),
                target_extents=target_extents,
                prev_centroid=tracking_state.prev_centroid,
                min_points=int(min_mask_points),
            )
            if points is None:
                selected_meta = {"prompt_mode": prompt_mode, **meta}
                continue
            selected_points = points
            selected_box = box
            selected_meta = {"prompt_mode": prompt_mode, **meta}
            break

        if selected_points is None:
            camera_meta[camera_name] = selected_meta or {"mode": "sam3_failed"}
            continue

        selected_clouds.append(selected_points)
        tracking_state.prev_box = selected_box
        centroid = selected_points[:, :3].mean(axis=0) if len(selected_points) > 0 else None
        if centroid is not None:
            tracking_state.prev_centroid = centroid.astype(np.float32)
        camera_meta[camera_name] = selected_meta

    if len(selected_clouds) == 0:
        return np.zeros((target_num_points, 6), dtype=np.float32), {
            "mode": "sam3_empty",
            "prompt": prompt,
            "cameras": camera_meta,
        }

    merged = merge_object_point_clouds(selected_clouds, target_num_points=int(target_num_points))
    return resample_point_cloud(merged, int(target_num_points)), {
        "mode": "sam3_projected",
        "prompt": prompt,
        "camera_count": int(len(selected_clouds)),
        "cameras": camera_meta,
    }


__all__ = [
    "SAM3ProjectiveTracker",
    "Sam3TrackingState",
    "build_placeholder_prompt_map",
    "build_placeholder_prompt_map_from_targets",
    "decode_jpeg_frame",
    "extract_placeholder_point_cloud_sam3",
    "extract_placeholder_point_cloud_sam3_online",
    "load_hdf5_with_cameras",
    "load_scene_info",
    "merge_object_point_clouds",
    "parse_camera_list",
    "parse_prompt_map",
    "parse_target_extents",
]
