#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from script.real_zed_collection.postprocess_raw_to_robotwin_hdf5 import parse_object_prompts
from script.real_zed_collection.real_zed_utils import ensure_dir, read_json


class GroundedSam2Segmentor:
    def __init__(
        self,
        *,
        device: str,
        gdino_model: str,
        sam2_cfg: str,
        sam2_ckpt: str,
        text_threshold: float,
        box_threshold: float,
        min_pixels: int,
    ) -> None:
        try:
            import torch
            from PIL import Image
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except Exception as e:
            raise RuntimeError(f"Missing Grounded-SAM2 dependencies: {e}") from e

        self.torch = torch
        self.image_cls = Image
        self.processor = AutoProcessor.from_pretrained(gdino_model)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(gdino_model).to(device)
        self.grounding_model.eval()
        sam_model = build_sam2(
            config_file=str(sam2_cfg),
            ckpt_path=str(sam2_ckpt),
            device=str(device),
            apply_postprocessing=True,
        )
        self.predictor = SAM2ImagePredictor(sam_model)
        self.device = str(device)
        self.text_threshold = float(text_threshold)
        self.box_threshold = float(box_threshold)
        self.min_pixels = int(min_pixels)

    def segment(self, rgb: np.ndarray, prompt: str) -> np.ndarray:
        text = " ".join(f"{item.strip().lower().rstrip('.')}." for item in prompt.split(",") if item.strip())
        if not text:
            raise ValueError("Prompt must be non-empty.")
        image = self.image_cls.fromarray(np.asarray(rgb, dtype=np.uint8))
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            outputs = self.grounding_model(**inputs)
        det = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=float(self.box_threshold),
            text_threshold=float(self.text_threshold),
            target_sizes=[image.size[::-1]],
        )[0]
        boxes = det["boxes"].detach().cpu().numpy() if "boxes" in det else np.empty((0, 4), dtype=np.float32)
        scores = det.get("scores", self.torch.tensor([])).detach().cpu().numpy()
        if boxes.size == 0:
            return np.zeros(rgb.shape[:2], dtype=bool)

        self.predictor.set_image(np.asarray(rgb, dtype=np.uint8))
        combined = np.zeros(rgb.shape[:2], dtype=bool)
        for idx, box in enumerate(boxes):
            score = float(scores[idx]) if idx < len(scores) else 1.0
            if score < float(self.box_threshold):
                continue
            masks, ious, _ = self.predictor.predict(box=box, multimask_output=True)
            mask = masks[int(np.argmax(ious))]
            if mask.dtype != np.bool_:
                mask = mask > 0
            if int(mask.sum()) >= self.min_pixels:
                combined |= mask
        return combined


def _load_rgb(raw_episode_dir: Path, rel_path: str) -> np.ndarray:
    with np.load(raw_episode_dir / rel_path, allow_pickle=False) as data:
        rgb = data["rgb"]
    return np.asarray(rgb, dtype=np.uint8)


def segment_episode(
    *,
    raw_episode_dir: str | Path,
    output_mask_root: str | Path,
    object_prompts: dict[str, str],
    camera_labels: list[str],
    segmentor: Any,
    every_n_frames: int = 1,
) -> Path:
    raw_episode_dir = Path(raw_episode_dir).expanduser().resolve()
    output_mask_root = ensure_dir(output_mask_root)
    manifest = read_json(raw_episode_dir / "manifest.json")
    labels = camera_labels or list(manifest.get("camera_labels", []))
    frames = manifest.get("frames", [])
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"No frames in raw manifest: {raw_episode_dir / 'manifest.json'}")
    if not object_prompts:
        raise ValueError("object_prompts must not be empty.")

    step = max(1, int(every_n_frames))
    for frame in frames:
        frame_idx = int(frame["frame_index"])
        if frame_idx % step != 0:
            continue
        cameras = frame.get("cameras", {})
        for placeholder, prompt in object_prompts.items():
            for label in labels:
                rgb = _load_rgb(raw_episode_dir, str(cameras[label]))
                mask = segmentor.segment(rgb, prompt)
                out_dir = ensure_dir(output_mask_root / placeholder / label)
                out_path = out_dir / f"mask_{frame_idx:06d}.png"
                ok = cv2.imwrite(str(out_path), (mask.astype(np.uint8) * 255))
                if not ok:
                    raise RuntimeError(f"Failed to save mask: {out_path}")
    return output_mask_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate per-object masks for a real ZED raw episode using Grounded-SAM2.")
    parser.add_argument("--raw_episode_dir", required=True)
    parser.add_argument("--output_mask_root", required=True)
    parser.add_argument("--object_prompts", required=True, help='Comma list like "{A}:mug,{B}:rack".')
    parser.add_argument("--camera_labels", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gdino_model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--sam2_cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--sam2_ckpt", default="/home/zheng/Datasets/sam_gdino/sam_checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--box_threshold", type=float, default=0.30)
    parser.add_argument("--min_pixels", type=int, default=500)
    parser.add_argument("--every_n_frames", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = [item.strip() for item in args.camera_labels.split(",") if item.strip()]
    prompts = parse_object_prompts(args.object_prompts)
    segmentor = GroundedSam2Segmentor(
        device=args.device,
        gdino_model=args.gdino_model,
        sam2_cfg=args.sam2_cfg,
        sam2_ckpt=args.sam2_ckpt,
        text_threshold=args.text_threshold,
        box_threshold=args.box_threshold,
        min_pixels=args.min_pixels,
    )
    out = segment_episode(
        raw_episode_dir=args.raw_episode_dir,
        output_mask_root=args.output_mask_root,
        object_prompts=prompts,
        camera_labels=labels,
        segmentor=segmentor,
        every_n_frames=args.every_n_frames,
    )
    print(out)


if __name__ == "__main__":
    main()
