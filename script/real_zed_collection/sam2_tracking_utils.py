#!/usr/bin/env python3

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from typing import Callable, Mapping, Sequence

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAM2_CONFIG = "sam2.1/sam2.1_hiera_l.yaml"


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser() if value else None


def default_sam2_root_candidates() -> list[Path]:
    candidates = [
        _env_path("SAM2_STREAMING_ROOT"),
        REPO_ROOT / "include" / "SAM2_streaming",
        Path.home() / "github" / "SAM2_streaming",
        Path.home() / "SAM2_streaming",
    ]
    return [path for path in candidates if path is not None]


def default_sam2_checkpoint_candidates() -> list[Path]:
    candidates = [
        _env_path("SAM2_CHECKPOINT"),
        Path.home() / "Datasets" / "sam2" / "sam2.1_hiera_large.pt",
        Path.home() / "github" / "SAM2_streaming" / "checkpoints" / "sam2.1_hiera_large.pt",
    ]
    return [path for path in candidates if path is not None]


DEFAULT_SAM2_ROOT = default_sam2_root_candidates()[0]
DEFAULT_SAM2_CHECKPOINT = default_sam2_checkpoint_candidates()[0]


def _resolve_first_existing_path(paths: Sequence[str | Path], *, description: str) -> Path:
    checked: list[str] = []
    for path in paths:
        candidate = Path(path).expanduser()
        checked.append(str(candidate))
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError:
            continue
        if resolved.exists():
            return resolved
    raise FileNotFoundError(f"{description} does not exist. Checked: {checked}")


def resolve_existing_sam2_root(
    sam2_root: str | Path = DEFAULT_SAM2_ROOT,
    *,
    fallback_roots: Sequence[str | Path] | None = None,
) -> Path:
    candidates = [Path(sam2_root).expanduser()]
    candidates.extend(fallback_roots if fallback_roots is not None else default_sam2_root_candidates())
    return _resolve_first_existing_path(candidates, description="SAM2 streaming root")


def resolve_existing_sam2_checkpoint(
    checkpoint: str | Path = DEFAULT_SAM2_CHECKPOINT,
    *,
    fallback_paths: Sequence[str | Path] | None = None,
) -> Path:
    candidates = [Path(checkpoint).expanduser()]
    candidates.extend(fallback_paths if fallback_paths is not None else default_sam2_checkpoint_candidates())
    return _resolve_first_existing_path(candidates, description="SAM2 checkpoint")


def ensure_sam2_root_on_path(sam2_root: str | Path = DEFAULT_SAM2_ROOT) -> Path:
    root = resolve_existing_sam2_root(sam2_root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def bbox_xyxy_to_sam2_box(bbox_xyxy: Sequence[float]) -> np.ndarray:
    x0, y0, x1, y1 = [float(v) for v in bbox_xyxy]
    return np.asarray([[x0, y0], [x1, y1]], dtype=np.float32)


def prompt_spec_to_sam2_kwargs(prompt_spec) -> dict[str, np.ndarray]:
    if prompt_spec is None:
        return {}
    if isinstance(prompt_spec, Mapping):
        bbox = prompt_spec.get("bbox_xyxy")
        points = prompt_spec.get("points_xy", prompt_spec.get("points"))
        labels = prompt_spec.get("point_labels", prompt_spec.get("labels"))
    else:
        bbox = prompt_spec
        points = None
        labels = None

    kwargs: dict[str, np.ndarray] = {}
    if bbox is not None:
        bbox_values = [float(v) for v in bbox]
        if len(bbox_values) >= 4:
            kwargs["bbox"] = bbox_xyxy_to_sam2_box(bbox_values[:4])
    if points is not None:
        points_arr = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if points_arr.size > 0:
            if labels is None:
                labels_arr = np.ones((points_arr.shape[0],), dtype=np.int32)
            else:
                labels_arr = np.asarray(labels, dtype=np.int32).reshape(-1)
            if labels_arr.shape[0] != points_arr.shape[0]:
                raise ValueError("SAM2 point prompt labels must match points.")
            kwargs["points"] = points_arr
            kwargs["labels"] = labels_arr
    return kwargs


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def masks_from_sam2_logits(
    *,
    out_obj_ids: Sequence[int],
    out_mask_logits,
    obj_id_to_placeholder: Mapping[int, str],
    image_shape_hw: tuple[int, int],
) -> dict[str, np.ndarray]:
    logits = _to_numpy(out_mask_logits)
    if logits.ndim == 3:
        logits = logits[:, None, ...]
    if logits.ndim != 4:
        raise ValueError(f"Expected SAM2 mask logits with shape [N,1,H,W], got {logits.shape}")

    height, width = int(image_shape_hw[0]), int(image_shape_hw[1])
    out: dict[str, np.ndarray] = {}
    for idx, obj_id in enumerate(list(out_obj_ids)):
        placeholder = obj_id_to_placeholder.get(int(obj_id))
        if placeholder is None or idx >= logits.shape[0]:
            continue
        mask = np.asarray(logits[idx]).squeeze() > 0.0
        if mask.shape != (height, width):
            mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
        out[str(placeholder)] = mask.astype(bool)
    return out


class SAM2StreamingObjectTracker:
    def __init__(
        self,
        *,
        predictor,
        placeholders: Sequence[str],
        obj_ids: Sequence[int] | None = None,
        device: str = "cuda",
        autocast_dtype: str | None = "bfloat16",
    ):
        self.predictor = predictor
        self.placeholders = [str(item) for item in placeholders]
        self.device = str(device)
        self.autocast_dtype = None if autocast_dtype is None else str(autocast_dtype)
        ids = list(obj_ids) if obj_ids is not None else list(range(1, len(self.placeholders) + 1))
        if len(ids) != len(self.placeholders):
            raise ValueError("obj_ids must have the same length as placeholders.")
        self.placeholder_to_obj_id = {
            placeholder: int(obj_id)
            for placeholder, obj_id in zip(self.placeholders, ids)
        }
        self.obj_id_to_placeholder = {
            int(obj_id): placeholder
            for placeholder, obj_id in self.placeholder_to_obj_id.items()
        }

    def _inference_context(self):
        dtype_name = "" if self.autocast_dtype is None else self.autocast_dtype.strip().lower()
        if not self.device.startswith("cuda") or dtype_name in {"", "none", "false", "float", "float32", "fp32"}:
            return contextlib.nullcontext()
        import torch

        dtype_by_name = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
        }
        dtype = dtype_by_name.get(dtype_name)
        if dtype is None:
            raise ValueError(f"Unsupported SAM2 autocast dtype: {self.autocast_dtype}")
        return torch.autocast(device_type="cuda", dtype=dtype)

    @classmethod
    def from_checkpoint(
        cls,
        *,
        placeholders: Sequence[str],
        sam2_root: str | Path = DEFAULT_SAM2_ROOT,
        config: str = DEFAULT_SAM2_CONFIG,
        checkpoint: str | Path = DEFAULT_SAM2_CHECKPOINT,
        device: str = "cuda",
        autocast_dtype: str | None = "bfloat16",
    ) -> "SAM2StreamingObjectTracker":
        ensure_sam2_root_on_path(sam2_root)
        try:
            import torch
            from sam2.build_sam import build_sam2_camera_predictor
        except Exception as exc:
            raise ImportError("SAM2 streaming dependencies are not available in this environment.") from exc

        if not torch.cuda.is_available() or not str(device).startswith("cuda"):
            raise RuntimeError(
                "SAM2 streaming predictor currently requires CUDA because the upstream "
                "SAM2CameraPredictor stores tracking state on cuda."
            )
        checkpoint_path = resolve_existing_sam2_checkpoint(checkpoint)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)
        predictor = build_sam2_camera_predictor(str(config), str(checkpoint_path), device=str(device))
        return cls(
            predictor=predictor,
            placeholders=placeholders,
            device=device,
            autocast_dtype=autocast_dtype,
        )

    def _complete_masks(self, masks: Mapping[str, np.ndarray], image_shape_hw: tuple[int, int]) -> dict[str, np.ndarray]:
        height, width = int(image_shape_hw[0]), int(image_shape_hw[1])
        out = {
            placeholder: np.zeros((height, width), dtype=bool)
            for placeholder in self.placeholders
        }
        for placeholder, mask in masks.items():
            mask_arr = np.asarray(mask).astype(bool)
            if mask_arr.shape != (height, width):
                mask_arr = cv2.resize(mask_arr.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
            out[str(placeholder)] = mask_arr
        return out

    def initialize_prompts(
        self,
        image: np.ndarray,
        prompts_by_placeholder: Mapping[str, object],
    ) -> dict[str, np.ndarray]:
        image_arr = np.asarray(image, dtype=np.uint8)
        with self._inference_context():
            self.predictor.load_first_frame(image_arr)
        latest_masks: dict[str, np.ndarray] = {}
        for placeholder in self.placeholders:
            prompt_spec = prompts_by_placeholder.get(placeholder)
            kwargs = prompt_spec_to_sam2_kwargs(prompt_spec)
            if not kwargs:
                continue
            with self._inference_context():
                _frame_idx, out_obj_ids, out_mask_logits = self.predictor.add_new_prompt(
                    frame_idx=0,
                    obj_id=self.placeholder_to_obj_id[placeholder],
                    **kwargs,
                )
            latest_masks.update(
                masks_from_sam2_logits(
                    out_obj_ids=out_obj_ids,
                    out_mask_logits=out_mask_logits,
                    obj_id_to_placeholder=self.obj_id_to_placeholder,
                    image_shape_hw=image_arr.shape[:2],
                )
            )
        return self._complete_masks(latest_masks, image_arr.shape[:2])

    def initialize(
        self,
        image: np.ndarray,
        boxes_by_placeholder: Mapping[str, Sequence[float]],
    ) -> dict[str, np.ndarray]:
        return self.initialize_prompts(image, boxes_by_placeholder)

    def preview_prompt(self, image: np.ndarray, placeholder: str, prompt_spec) -> np.ndarray:
        image_arr = np.asarray(image, dtype=np.uint8)
        kwargs = prompt_spec_to_sam2_kwargs(prompt_spec)
        if not kwargs:
            return np.zeros(image_arr.shape[:2], dtype=bool)
        with self._inference_context():
            self.predictor.load_first_frame(image_arr)
        with self._inference_context():
            _frame_idx, out_obj_ids, out_mask_logits = self.predictor.add_new_prompt(
                frame_idx=0,
                obj_id=self.placeholder_to_obj_id[str(placeholder)],
                **kwargs,
            )
        masks = masks_from_sam2_logits(
            out_obj_ids=out_obj_ids,
            out_mask_logits=out_mask_logits,
            obj_id_to_placeholder=self.obj_id_to_placeholder,
            image_shape_hw=image_arr.shape[:2],
        )
        return self._complete_masks(masks, image_arr.shape[:2]).get(str(placeholder), np.zeros(image_arr.shape[:2], dtype=bool))

    def track(self, image: np.ndarray) -> dict[str, np.ndarray]:
        image_arr = np.asarray(image, dtype=np.uint8)
        with self._inference_context():
            out_obj_ids, out_mask_logits = self.predictor.track(image_arr)
        masks = masks_from_sam2_logits(
            out_obj_ids=out_obj_ids,
            out_mask_logits=out_mask_logits,
            obj_id_to_placeholder=self.obj_id_to_placeholder,
            image_shape_hw=image_arr.shape[:2],
        )
        return self._complete_masks(masks, image_arr.shape[:2])


def make_sam2_tracker_factory(
    *,
    placeholders: Sequence[str],
    sam2_root: str | Path = DEFAULT_SAM2_ROOT,
    config: str = DEFAULT_SAM2_CONFIG,
    checkpoint: str | Path = DEFAULT_SAM2_CHECKPOINT,
    device: str = "cuda",
    autocast_dtype: str | None = "bfloat16",
) -> Callable[[str], SAM2StreamingObjectTracker]:
    def _factory(_camera_label: str) -> SAM2StreamingObjectTracker:
        return SAM2StreamingObjectTracker.from_checkpoint(
            placeholders=placeholders,
            sam2_root=sam2_root,
            config=config,
            checkpoint=checkpoint,
            device=device,
            autocast_dtype=autocast_dtype,
        )

    return _factory
