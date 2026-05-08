#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAM2_ROOT = Path(os.environ.get("SAM2_STREAMING_ROOT", Path.home() / "github" / "SAM2_streaming")).expanduser()
DEFAULT_SAM2_CHECKPOINT = Path(
    os.environ.get("SAM2_CHECKPOINT", Path.home() / "DataModel" / "sam2.1" / "sam2.1_hiera_base_plus.pt")
    # os.environ.get("SAM2_CHECKPOINT", Path.home() / "DataModel" / "sam2.1" / "sam2.1_hiera_large.pt")
).expanduser()


class BBoxSelector:
    def __init__(self) -> None:
        self.drawing = False
        self.start: tuple[int, int] | None = None
        self.current: tuple[int, int] | None = None
        self.bbox_xyxy: tuple[int, int, int, int] | None = None

    def reset(self) -> None:
        self.drawing = False
        self.start = None
        self.current = None
        self.bbox_xyxy = None

    def callback(self, event, x, y, _flags, _param) -> None:
        point = (int(x), int(y))
        self.current = point
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start = point
            self.bbox_xyxy = None
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.current = point
        elif event == cv2.EVENT_LBUTTONUP and self.drawing and self.start is not None:
            self.drawing = False
            x0, y0 = self.start
            x1, y1 = point
            lo_x, hi_x = sorted((x0, x1))
            lo_y, hi_y = sorted((y0, y1))
            if hi_x > lo_x and hi_y > lo_y:
                self.bbox_xyxy = (lo_x, lo_y, hi_x, hi_y)

    def draw(self, image_bgr: np.ndarray) -> np.ndarray:
        out = image_bgr.copy()
        if self.drawing and self.start is not None and self.current is not None:
            cv2.rectangle(out, self.start, self.current, (255, 180, 0), 2)
        if self.bbox_xyxy is not None:
            x0, y0, x1, y1 = self.bbox_xyxy
            cv2.rectangle(out, (x0, y0), (x1, y1), (0, 220, 255), 2)
        return out


def parse_bbox(text: str) -> tuple[int, int, int, int] | None:
    if not text.strip():
        return None
    values = [int(round(float(item))) for item in text.replace(";", ",").split(",") if item.strip()]
    if len(values) != 4:
        raise ValueError("--bbox must be formatted as x0,y0,x1,y1")
    x0, y0, x1, y1 = values
    lo_x, hi_x = sorted((x0, x1))
    lo_y, hi_y = sorted((y0, y1))
    if hi_x <= lo_x or hi_y <= lo_y:
        raise ValueError(f"Invalid bbox: {text}")
    return lo_x, lo_y, hi_x, hi_y


def zed_resolution_enum(sl, name: str):
    key = str(name).strip().upper()
    if not hasattr(sl.RESOLUTION, key):
        allowed = [item for item in dir(sl.RESOLUTION) if item.isupper()]
        raise ValueError(f"Unknown ZED resolution {name!r}. Available: {allowed}")
    return getattr(sl.RESOLUTION, key)


def resize_keep_aspect(image: np.ndarray, width: int) -> np.ndarray:
    if int(width) <= 0 or image.shape[1] <= int(width):
        return image
    scale = float(width) / float(image.shape[1])
    height = max(1, int(round(image.shape[0] * scale)))
    return cv2.resize(image, (int(width), height), interpolation=cv2.INTER_AREA)


def open_zed(serial: int, resolution: str, fps: int):
    import pyzed.sl as sl

    zed = sl.Camera()
    init = sl.InitParameters()
    if int(serial) > 0:
        init.set_from_serial_number(int(serial))
    init.camera_resolution = zed_resolution_enum(sl, resolution)
    init.camera_fps = int(fps)
    init.coordinate_units = sl.UNIT.METER
    if hasattr(sl.DEPTH_MODE, "NONE"):
        init.depth_mode = sl.DEPTH_MODE.NONE
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open ZED serial={serial}: {status}")
    return zed, sl


def read_zed_rgb(zed, sl, image_mat) -> np.ndarray | None:
    runtime = sl.RuntimeParameters()
    if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
        return None
    zed.retrieve_image(image_mat, sl.VIEW.LEFT)
    image_raw = image_mat.get_data()
    if image_raw is None:
        return None
    bgr = image_raw[:, :, :3] if image_raw.ndim == 3 and image_raw.shape[2] >= 3 else image_raw
    return cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2RGB)


def ensure_sam2_imports(sam2_root: Path):
    root = sam2_root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"SAM2 root does not exist: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


@contextmanager
def sam2_autocast(device: str, dtype_name: str):
    if not str(device).startswith("cuda") or str(dtype_name).lower() in {"", "none", "float32", "fp32"}:
        yield
        return
    import torch

    dtype_by_name = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    dtype = dtype_by_name.get(str(dtype_name).lower())
    if dtype is None:
        raise ValueError(f"Unsupported autocast dtype: {dtype_name}")
    with torch.autocast(device_type="cuda", dtype=dtype):
        yield


def build_predictor(args: argparse.Namespace):
    import torch
    from hydra import initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from sam2.build_sam import build_sam2_camera_predictor

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for SAM2, but torch.cuda.is_available() is false.")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    checkpoint = Path(args.sam2_checkpoint).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"SAM2 checkpoint does not exist: {checkpoint}")
    config_dir = Path(args.sam2_root).expanduser().resolve() / "configs"
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.2"):
        return build_sam2_camera_predictor(
            str(args.sam2_config),
            str(checkpoint),
            device=str(args.device),
        )


def logits_to_combined_mask(out_mask_logits, image_shape_hw: tuple[int, int]) -> np.ndarray:
    height, width = image_shape_hw
    combined = np.zeros((height, width), dtype=np.uint8)
    for i in range(int(out_mask_logits.shape[0])):
        mask = (out_mask_logits[i] > 0.0).squeeze()
        if hasattr(mask, "detach"):
            mask = mask.detach().cpu().numpy()
        mask = np.asarray(mask, dtype=np.uint8)
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        combined = cv2.bitwise_or(combined, mask.astype(np.uint8))
    return combined.astype(bool)


def overlay_mask(image_rgb: np.ndarray, mask: np.ndarray, color_rgb: Sequence[int] = (0, 255, 80)) -> np.ndarray:
    out = image_rgb.copy()
    color = np.asarray(color_rgb, dtype=np.uint8)
    out[mask] = (0.45 * out[mask] + 0.55 * color).astype(np.uint8)
    return out


def draw_status(image_bgr: np.ndarray, text: str) -> np.ndarray:
    out = image_bgr.copy()
    cv2.putText(out, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 3)
    cv2.putText(out, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 220, 255), 1)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time ZED RGB + SAM2 bbox tracker speed test.")
    parser.add_argument("--serial", type=int, default=38968158, help="ZED serial number. 0 opens the default camera.")
    parser.add_argument("--zed_resolution", default="HD720")
    parser.add_argument("--zed_fps", type=int, default=30)
    parser.add_argument("--resize_width", type=int, default=640, help="Resize RGB before SAM2. Use 0 for native ZED size.")
    parser.add_argument("--sam2_root", default=str(DEFAULT_SAM2_ROOT))
    parser.add_argument("--sam2_config", default="sam2.1/sam2.1_hiera_b+.yaml")
    parser.add_argument("--sam2_checkpoint", default=str(DEFAULT_SAM2_CHECKPOINT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--autocast_dtype", default="bfloat16")
    parser.add_argument("--bbox", default="", help="Optional x0,y0,x1,y1 bbox on the resized display image.")
    parser.add_argument("--print_interval", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sam2_root = ensure_sam2_imports(Path(args.sam2_root))
    args.sam2_root = str(sam2_root)
    bbox = parse_bbox(str(args.bbox))

    print(f"[INFO] SAM2 root: {sam2_root}")
    print(f"[INFO] SAM2 config: {args.sam2_config}")
    print(f"[INFO] SAM2 checkpoint: {Path(args.sam2_checkpoint).expanduser()}")
    print(f"[INFO] Opening ZED serial={args.serial} resolution={args.zed_resolution} fps={args.zed_fps}")

    zed = None
    try:
        zed, sl = open_zed(int(args.serial), str(args.zed_resolution), int(args.zed_fps))
        image_mat = sl.Mat()
        predictor = build_predictor(args)
        selector = BBoxSelector()
        if bbox is not None:
            selector.bbox_xyxy = bbox

        window = "zed_sam2_realtime_bbox"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window, selector.callback)

        initialized = False
        frame_idx = 0
        last_report = time.perf_counter()
        last_frame = last_report
        while True:
            rgb = read_zed_rgb(zed, sl, image_mat)
            if rgb is None:
                continue
            rgb = resize_keep_aspect(rgb, int(args.resize_width))
            frame_start = time.perf_counter()
            status = "draw bbox; release mouse to start; q quit; r reset"

            if selector.bbox_xyxy is not None and not initialized:
                x0, y0, x1, y1 = selector.bbox_xyxy
                box = np.asarray([[x0, y0], [x1, y1]], dtype=np.float32)
                with sam2_autocast(str(args.device), str(args.autocast_dtype)):
                    predictor.load_first_frame(rgb)
                    _frame_idx, _out_obj_ids, _out_mask_logits = predictor.add_new_prompt(
                        frame_idx=0,
                        obj_id=1,
                        bbox=box,
                    )
                initialized = True
                status = "initialized"
            elif initialized:
                with sam2_autocast(str(args.device), str(args.autocast_dtype)):
                    out_obj_ids, out_mask_logits = predictor.track(rgb)
                mask = logits_to_combined_mask(out_mask_logits, rgb.shape[:2]) if len(out_obj_ids) else np.zeros(rgb.shape[:2], dtype=bool)
                rgb = overlay_mask(rgb, mask)
                status = f"tracking objects={len(out_obj_ids)}"

            display = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            display = selector.draw(display)
            now = time.perf_counter()
            inst_fps = 1.0 / max(now - last_frame, 1e-6)
            last_frame = now
            display = draw_status(display, f"{status} | fps={inst_fps:.1f}")
            cv2.imshow(window, display)

            frame_idx += 1
            if frame_idx % max(1, int(args.print_interval)) == 0:
                dt = max(now - last_report, 1e-6)
                print(
                    f"[FPS] avg={max(1, int(args.print_interval)) / dt:.2f} "
                    f"last_frame_ms={(now - frame_start) * 1000.0:.1f} initialized={initialized}"
                )
                last_report = now

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                selector.reset()
                initialized = False
                predictor = build_predictor(args)
    finally:
        if zed is not None:
            zed.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
