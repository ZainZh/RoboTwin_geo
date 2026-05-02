#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DP_ROOT = REPO_ROOT / "policy" / "DP"
for path in (REPO_ROOT, DP_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from script.real_zed_collection.collect_zed_robotwin_raw import (  # noqa: E402
    SharedZedFrame,
    _camera_matrix_from_zed_info,
    _resize_rgb_for_storage,
    _resolution_enum,
)
from script.real_zed_collection.real_zed_utils import load_three_zed_calibration  # noqa: E402
from script.real_zed_inference.real_dp3_inference import (  # noqa: E402
    AsyncActionController,
    build_action_diagnostic_row,
    build_robot_env,
    clamp_action,
    configure_robot_servo_params,
    current_joint_vector,
    execute_action,
    format_timings,
    prepare_action_for_execution,
    print_info,
    print_warning,
    profile_section,
    reset_robot_to_photo_pose,
    resolve_path,
    should_print_step_timing,
)


DEFAULT_CALIBRATION = REPO_ROOT / "script" / "real_zed_collection" / "calibration" / "three_camera_workspace_extrinsics.yaml"
STANDARD_DP_CAMERA_MAP = {
    "head_cam": "global",
    "left_cam": "left",
    "right_cam": "right",
}


@dataclass
class LiveRGBCameras:
    labels: list[str]
    serials: list[int]
    shared_by_label: dict[str, SharedZedFrame]
    stop_event: threading.Event
    threads: list[threading.Thread]

    def stop(self) -> None:
        self.stop_event.set()
        for thread in self.threads:
            thread.join(timeout=2.0)


def _cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_plain_mapping(obj: Any) -> Mapping[str, Any]:
    if isinstance(obj, Mapping):
        return obj
    try:
        from omegaconf import OmegaConf  # noqa: PLC0415

        return OmegaConf.to_container(obj, resolve=True)
    except Exception:
        return dict(obj)


def required_image_obs_keys(shape_meta: Any) -> list[str]:
    meta = _as_plain_mapping(shape_meta)
    obs_meta = _as_plain_mapping(meta.get("obs", {}))
    keys = [
        str(key)
        for key, value in obs_meta.items()
        if str(_as_plain_mapping(value).get("type", "")).lower() == "rgb"
    ]
    if not keys:
        raise ValueError("DP checkpoint shape_meta does not contain any rgb observation keys.")
    return keys


def image_shapes_from_shape_meta(shape_meta: Any, obs_keys: Sequence[str]) -> dict[str, tuple[int, int, int]]:
    meta = _as_plain_mapping(shape_meta)
    obs_meta = _as_plain_mapping(meta.get("obs", {}))
    shapes: dict[str, tuple[int, int, int]] = {}
    for key in obs_keys:
        value = _as_plain_mapping(obs_meta.get(key, {}))
        shape = value.get("shape", [3, -1, -1])
        if len(shape) != 3:
            raise ValueError(f"Expected image shape [C,H,W] for {key}, got {shape}")
        shapes[str(key)] = (int(shape[0]), int(shape[1]), int(shape[2]))
    return shapes


def parse_camera_labels(text: str | Sequence[str]) -> list[str]:
    if isinstance(text, str):
        return [item.strip() for item in text.split(",") if item.strip()]
    return [str(item).strip() for item in text if str(item).strip()]


def parse_serials(text: str | Sequence[int] | Sequence[str] | None) -> list[int]:
    if text is None:
        return []
    if isinstance(text, str):
        if not text.strip():
            return []
        return [int(item.strip()) for item in text.split(",") if item.strip()]
    return [int(item) for item in text]


def parse_dp_camera_map(
    text: str,
    *,
    required_keys: Sequence[str],
    live_labels: Sequence[str],
) -> dict[str, str]:
    live = {str(label) for label in live_labels}
    required = [str(key) for key in required_keys]
    if str(text or "").strip():
        mapping: dict[str, str] = {}
        for item in str(text).split(","):
            if not item.strip():
                continue
            if ":" not in item:
                raise ValueError(f"Invalid --dp_camera_map item {item!r}; expected obs_key:camera_label.")
            obs_key, label = [part.strip() for part in item.split(":", 1)]
            mapping[obs_key] = label
    elif required == ["head_cam"] and len(live_labels) == 1:
        mapping = {"head_cam": str(live_labels[0])}
    else:
        mapping = {key: STANDARD_DP_CAMERA_MAP[key] for key in required if key in STANDARD_DP_CAMERA_MAP}

    missing = [key for key in required if key not in mapping]
    if missing:
        raise ValueError(f"Missing camera map entries for DP obs keys: {missing}")
    unknown = [key for key in mapping if key not in required]
    if unknown:
        raise ValueError(f"--dp_camera_map contains obs keys not used by checkpoint: {unknown}")
    missing_labels = [label for label in mapping.values() if label not in live]
    if missing_labels:
        raise ValueError(f"--dp_camera_map references live camera labels not started: {missing_labels}; live={sorted(live)}")
    return mapping


def _resize_rgb_if_needed(rgb: np.ndarray, image_shape_chw: tuple[int, int, int]) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.uint8)
    channels, target_h, target_w = image_shape_chw
    if channels != 3:
        raise ValueError(f"Only 3-channel RGB observations are supported, got shape {image_shape_chw}")
    if target_h > 0 and target_w > 0 and arr.shape[:2] != (target_h, target_w):
        arr = cv2.resize(arr, (int(target_w), int(target_h)), interpolation=cv2.INTER_AREA)
    return arr


def build_dp_image_observation(
    *,
    frames_by_label: Mapping[str, Mapping[str, Any]],
    joint_vector: np.ndarray,
    obs_keys: Sequence[str],
    camera_map: Mapping[str, str],
    image_shapes: Mapping[str, tuple[int, int, int]],
) -> dict[str, np.ndarray]:
    obs: dict[str, np.ndarray] = {}
    for obs_key in obs_keys:
        label = str(camera_map[str(obs_key)])
        if label not in frames_by_label:
            raise KeyError(f"Live camera frame for label {label!r} is missing.")
        rgb = _resize_rgb_if_needed(np.asarray(frames_by_label[label]["rgb"], dtype=np.uint8), image_shapes[str(obs_key)])
        obs[str(obs_key)] = (np.moveaxis(rgb, -1, 0).astype(np.float32) / 255.0)
    obs["agent_pos"] = np.asarray(joint_vector, dtype=np.float32).reshape(14)
    return obs


class RealDPActionRunner:
    def __init__(self, *, n_obs_steps: int, n_action_steps: int, obs_keys: Sequence[str]):
        self.n_obs_steps = int(n_obs_steps)
        self.n_action_steps = int(n_action_steps)
        self.obs_keys = [str(key) for key in obs_keys]
        self.obs: deque[dict[str, np.ndarray]] = deque(maxlen=self.n_obs_steps)

    @staticmethod
    def stack_last_n_obs(all_obs: Sequence[np.ndarray], n_steps: int) -> np.ndarray:
        if not all_obs:
            raise RuntimeError("No observations are recorded.")
        last = np.asarray(all_obs[-1])
        result = np.zeros((int(n_steps), *last.shape), dtype=last.dtype)
        start_idx = -min(int(n_steps), len(all_obs))
        result[start_idx:] = np.asarray(all_obs[start_idx:], dtype=last.dtype)
        if int(n_steps) > len(all_obs):
            result[:start_idx] = result[start_idx]
        return result

    def reset_obs(self) -> None:
        self.obs.clear()

    def update_obs(self, current_obs: Mapping[str, np.ndarray]) -> None:
        self.obs.append({key: np.asarray(value) for key, value in current_obs.items()})

    def get_n_steps_obs(self) -> dict[str, np.ndarray]:
        if not self.obs:
            raise RuntimeError("No observations are recorded.")
        keys = [*self.obs_keys, "agent_pos"]
        return {
            key: self.stack_last_n_obs([obs[key] for obs in self.obs], self.n_obs_steps)
            for key in keys
        }

    def get_action(self, policy, observation: Mapping[str, np.ndarray] | None = None) -> np.ndarray:
        if observation is not None:
            self.update_obs(observation)
        obs = self.get_n_steps_obs()
        import torch  # noqa: PLC0415

        device = getattr(policy, "device", torch.device("cpu"))
        obs_dict = {
            key: torch.from_numpy(value).to(device=device).unsqueeze(0)
            for key, value in obs.items()
            if key in self.obs_keys or key == "agent_pos"
        }
        with torch.no_grad():
            action_dict = policy.predict_action(obs_dict)
        action = action_dict["action"].detach().to("cpu").numpy().squeeze(0)
        return np.asarray(action[: self.n_action_steps], dtype=np.float32)


def _depth_mode_enum(sl, name: str):
    upper = str(name).upper()
    if upper == "NONE" and hasattr(sl.DEPTH_MODE, "NONE"):
        return sl.DEPTH_MODE.NONE
    return {
        "PERFORMANCE": sl.DEPTH_MODE.PERFORMANCE,
        "QUALITY": sl.DEPTH_MODE.QUALITY,
        "ULTRA": sl.DEPTH_MODE.ULTRA,
        "NEURAL": getattr(sl.DEPTH_MODE, "NEURAL", sl.DEPTH_MODE.QUALITY),
        "NEURAL_PLUS": getattr(sl.DEPTH_MODE, "NEURAL_PLUS", getattr(sl.DEPTH_MODE, "NEURAL", sl.DEPTH_MODE.QUALITY)),
    }.get(upper, getattr(sl.DEPTH_MODE, "NEURAL", sl.DEPTH_MODE.QUALITY))


def zed_rgb_capture_loop(
    *,
    label: str,
    serial: int,
    resolution: str,
    fps: int,
    depth_mode: str,
    save_rgb_width: int,
    save_rgb_height: int,
    shared: SharedZedFrame,
    stop_event: threading.Event,
) -> None:
    try:
        import pyzed.sl as sl  # noqa: PLC0415
    except Exception as exc:
        shared.set_error(f"pyzed.sl import failed for camera {label}: {exc}")
        return

    zed = sl.Camera()
    init = sl.InitParameters()
    init.set_from_serial_number(int(serial))
    init.camera_resolution = _resolution_enum(sl, resolution)
    init.camera_fps = int(fps)
    init.depth_mode = _depth_mode_enum(sl, depth_mode)
    init.coordinate_units = sl.UNIT.METER
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        shared.set_error(f"Failed to open ZED label={label}, serial={serial}: {status}")
        return

    camera_matrix = _camera_matrix_from_zed_info(zed.get_camera_information())
    runtime = sl.RuntimeParameters()
    image_mat = sl.Mat()
    frame_idx = 0
    try:
        while not stop_event.is_set():
            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                continue
            zed.retrieve_image(image_mat, sl.VIEW.LEFT)
            image_raw = image_mat.get_data()
            if image_raw is None:
                continue
            bgr = image_raw[:, :, :3] if image_raw.ndim == 3 and image_raw.shape[2] >= 3 else image_raw
            rgb = cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2RGB)
            rgb = _resize_rgb_for_storage(
                rgb=rgb,
                target_width=int(save_rgb_width),
                target_height=int(save_rgb_height),
            )
            shared.update(
                {
                    "label": label,
                    "serial": int(serial),
                    "frame_index": int(frame_idx),
                    "timestamp_unix_sec": time.time(),
                    "rgb": np.asarray(rgb, dtype=np.uint8),
                    "camera_matrix": np.asarray(camera_matrix, dtype=np.float32),
                }
            )
            frame_idx += 1
    finally:
        zed.close()


def _resolve_rgb_cameras(args: argparse.Namespace) -> tuple[list[str], list[int]]:
    labels = parse_camera_labels(args.camera_labels)
    if not labels:
        raise ValueError("--camera_labels must contain at least one camera label.")
    serials = parse_serials(args.zed_serials)
    if not serials:
        calib = load_three_zed_calibration(resolve_path(args.calibration_path), frame_mode="reference_camera")
        serials = [int(calib[label].serial_number) for label in labels]
    if len(labels) != len(serials):
        raise ValueError(f"Camera label/serial count mismatch: labels={labels}, serials={serials}")
    print(f"[INFO] DP RGB camera label -> serial mapping: {dict(zip(labels, serials))}")
    return labels, serials


def start_rgb_zed_cameras(args: argparse.Namespace) -> LiveRGBCameras:
    labels, serials = _resolve_rgb_cameras(args)
    shared_by_label = {label: SharedZedFrame() for label in labels}
    stop_event = threading.Event()
    threads: list[threading.Thread] = []
    for label, serial in zip(labels, serials):
        thread = threading.Thread(
            target=zed_rgb_capture_loop,
            kwargs={
                "label": label,
                "serial": int(serial),
                "resolution": args.zed_resolution,
                "fps": int(args.zed_fps),
                "depth_mode": args.zed_depth_mode,
                "save_rgb_width": int(args.save_rgb_width),
                "save_rgb_height": int(args.save_rgb_height),
                "shared": shared_by_label[label],
                "stop_event": stop_event,
            },
            daemon=True,
        )
        thread.start()
        threads.append(thread)
    live = LiveRGBCameras(
        labels=labels,
        serials=serials,
        shared_by_label=shared_by_label,
        stop_event=stop_event,
        threads=threads,
    )
    wait_for_cameras(live, timeout_sec=float(args.camera_warmup_timeout_sec))
    return live


def wait_for_cameras(live: LiveRGBCameras, timeout_sec: float) -> None:
    deadline = time.time() + float(timeout_sec)
    ready: set[str] = set()
    last_error = ""
    while time.time() < deadline:
        for label in live.labels:
            if label in ready:
                continue
            try:
                live.shared_by_label[label].snapshot()
                ready.add(label)
            except RuntimeError as exc:
                last_error = str(exc)
        if len(ready) == len(live.labels):
            print(f"[INFO] ZED RGB cameras ready: {live.labels}")
            return
        time.sleep(0.05)
    raise TimeoutError(f"Timed out waiting for ZED RGB cameras. ready={sorted(ready)}, last_error={last_error}")


def snapshot_frames(live: LiveRGBCameras) -> dict[str, dict[str, Any]]:
    return {label: live.shared_by_label[label].snapshot() for label in live.labels}


def maybe_show_cameras(frames_by_label: Mapping[str, Mapping[str, Any]], labels: Sequence[str]) -> None:
    images = []
    for label in labels:
        if label not in frames_by_label:
            continue
        arr = np.asarray(frames_by_label[label]["rgb"], dtype=np.uint8).copy()
        cv2.putText(arr, str(label), (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        images.append(arr)
    if not images:
        return
    min_h = min(img.shape[0] for img in images)
    resized = [
        cv2.resize(img, (int(img.shape[1] * min_h / img.shape[0]), min_h), interpolation=cv2.INTER_AREA)
        if img.shape[0] != min_h
        else img
        for img in images
    ]
    canvas = np.hstack(resized)
    cv2.imshow("real_zed_dp_rgb", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)


def derive_ckpt_setting(args: argparse.Namespace) -> str:
    if str(args.ckpt_setting).strip():
        return str(args.ckpt_setting).strip()
    camera_setting = "_".join(parse_camera_labels(args.camera_labels))
    return f"{args.task_config}-dp-{camera_setting}"


def resolve_dp_checkpoint_path(args: argparse.Namespace, ckpt_setting: str) -> Path:
    if str(args.ckpt_path).strip():
        return resolve_path(args.ckpt_path)
    return (
        DP_ROOT
        / "checkpoints"
        / f"{args.task_name}-{ckpt_setting}-{args.expert_data_num}-{args.seed}"
        / f"{args.checkpoint_num}.ckpt"
    )


def load_dp_policy(args: argparse.Namespace) -> tuple[Any, Any, Path, str]:
    if args.gpu_id is not None and str(args.gpu_id).strip() != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    import dill  # noqa: PLC0415
    import hydra  # noqa: PLC0415
    import torch  # noqa: PLC0415

    ckpt_setting = derive_ckpt_setting(args)
    ckpt_path = resolve_dp_checkpoint_path(args, ckpt_setting)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"DP checkpoint does not exist: {ckpt_path}")
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill, map_location=args.device)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=None)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if bool(_cfg_get(_cfg_get(cfg, "training"), "use_ema", False)) else workspace.model
    device = torch.device(str(args.device))
    policy.to(device)
    policy.eval()
    print(f"[INFO] Loaded DP policy ckpt_setting={ckpt_setting}, checkpoint={ckpt_path}")
    return policy, cfg, ckpt_path, ckpt_setting


def _cfg_int(cfg: Any, key: str, default: int) -> int:
    value = _cfg_get(cfg, key, default)
    return int(value if value is not None else default)


def run_real_inference(args: argparse.Namespace) -> None:
    if bool(args.execute) and bool(args.no_robot):
        raise ValueError("--execute cannot be used together with --no_robot")

    print("[INFO] Starting real DP image inference.")
    print("[INFO] Loading DP policy ...")
    policy, cfg, _ckpt_path, ckpt_setting = load_dp_policy(args)
    obs_keys = required_image_obs_keys(_cfg_get(cfg, "shape_meta"))
    image_shapes = image_shapes_from_shape_meta(_cfg_get(cfg, "shape_meta"), obs_keys)
    n_obs_steps = _cfg_int(cfg, "n_obs_steps", int(args.n_obs_steps))
    n_action_steps = _cfg_int(cfg, "n_action_steps", int(args.n_action_steps))
    runner = RealDPActionRunner(n_obs_steps=n_obs_steps, n_action_steps=n_action_steps, obs_keys=obs_keys)
    print(f"[INFO] DP obs_keys={obs_keys}, n_obs_steps={n_obs_steps}, n_action_steps={n_action_steps}")

    print("[INFO] Starting ZED RGB camera threads ...")
    live = start_rgb_zed_cameras(args)
    camera_map = parse_dp_camera_map(args.dp_camera_map, required_keys=obs_keys, live_labels=live.labels)
    print(f"[INFO] DP obs camera map: {camera_map}")

    env = None
    try:
        print("[INFO] Initializing robot interface ...")
        env = build_robot_env(args)
        configure_robot_servo_params(env, args)
        if env is not None and bool(args.execute) and not bool(args.skip_robot_reset):
            last_action = reset_robot_to_photo_pose(env)
        elif env is not None:
            print("[INFO] Reading initial robot joint state ...")
            last_action = np.asarray(env.get_obs()["joint_positions"], dtype=np.float32).reshape(14)
        else:
            last_action = np.zeros(14, dtype=np.float32)
        last_action[6] = float(args.initial_left_gripper)
        last_action[13] = float(args.initial_right_gripper)

        if not bool(args.execute):
            print("[INFO] Dry-run mode: model actions are printed but not sent to the robot. Pass --execute to move.")

        if bool(args.async_control):
            _run_async_loop(
                args=args,
                live=live,
                env=env,
                policy=policy,
                runner=runner,
                obs_keys=obs_keys,
                image_shapes=image_shapes,
                camera_map=camera_map,
                initial_action=last_action,
            )
        else:
            _run_sync_loop(
                args=args,
                live=live,
                env=env,
                policy=policy,
                runner=runner,
                obs_keys=obs_keys,
                image_shapes=image_shapes,
                camera_map=camera_map,
                initial_action=last_action,
            )
    finally:
        live.stop()
        if bool(args.show_img):
            cv2.destroyAllWindows()


def _build_current_obs(
    *,
    args: argparse.Namespace,
    live: LiveRGBCameras,
    joints: np.ndarray,
    obs_keys: Sequence[str],
    image_shapes: Mapping[str, tuple[int, int, int]],
    camera_map: Mapping[str, str],
    timings: dict[str, float],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    with profile_section(timings, "snapshot_rgb", bool(args.profile_timing)):
        frames = snapshot_frames(live)
    if bool(args.show_img):
        with profile_section(timings, "show_img", bool(args.profile_timing)):
            maybe_show_cameras(frames, live.labels)
    with profile_section(timings, "encode_rgb", bool(args.profile_timing)):
        obs = build_dp_image_observation(
            frames_by_label=frames,
            joint_vector=joints,
            obs_keys=obs_keys,
            camera_map=camera_map,
            image_shapes=image_shapes,
        )
    return obs, frames


def _run_async_loop(
    *,
    args: argparse.Namespace,
    live: LiveRGBCameras,
    env,
    policy,
    runner: RealDPActionRunner,
    obs_keys: Sequence[str],
    image_shapes: Mapping[str, tuple[int, int, int]],
    camera_map: Mapping[str, str],
    initial_action: np.ndarray,
) -> None:
    controller: AsyncActionController | None = None
    try:
        while controller is None or controller.step_count < int(args.max_steps):
            if controller is not None:
                controller.raise_if_error()
                joints = controller.latest_action()
            else:
                joints = np.asarray(initial_action, dtype=np.float32).reshape(14)
            chunk_timings: dict[str, float] = {}
            obs, _frames = _build_current_obs(
                args=args,
                live=live,
                joints=joints,
                obs_keys=obs_keys,
                image_shapes=image_shapes,
                camera_map=camera_map,
                timings=chunk_timings,
            )
            with profile_section(chunk_timings, "dp_policy", bool(args.profile_timing)):
                actions = runner.get_action(policy, obs)
            if controller is None:
                controller = AsyncActionController(env=env, args=args, initial_action=joints)
                controller.start()
            controller.submit_actions(actions)
            if bool(args.profile_timing):
                chunk_timings["chunk_total"] = sum(chunk_timings.values())
                print_info(
                    f"[TIMING dp_async_chunk@step {controller.step_count + 1:04d}] "
                    f"{format_timings(chunk_timings)} actions_submitted={len(actions)} "
                    f"control_steps={controller.step_count}",
                )
        if controller is not None:
            controller.raise_if_error()
    finally:
        if controller is not None:
            controller.stop()


def _run_sync_loop(
    *,
    args: argparse.Namespace,
    live: LiveRGBCameras,
    env,
    policy,
    runner: RealDPActionRunner,
    obs_keys: Sequence[str],
    image_shapes: Mapping[str, tuple[int, int, int]],
    camera_map: Mapping[str, str],
    initial_action: np.ndarray,
) -> None:
    action_step = 0
    first_action = True
    last_action = np.asarray(initial_action, dtype=np.float32).reshape(14).copy()
    previous_command_delta = np.zeros(14, dtype=np.float32)
    rate_sec = 0.0 if float(args.control_hz) <= 0 else 1.0 / float(args.control_hz)
    diagnostic_file = None
    diagnostic_writer = None
    if str(args.action_diagnostics_csv).strip():
        import csv  # noqa: PLC0415

        diagnostics_path = resolve_path(args.action_diagnostics_csv)
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostic_file = diagnostics_path.open("w", newline="", encoding="utf-8")
        diagnostic_writer = csv.DictWriter(diagnostic_file, fieldnames=list(build_action_diagnostic_row(
            step=0,
            raw_policy_action=last_action,
            command_action=last_action,
            observed_after_step=last_action,
            action_before_step=last_action,
            previous_command_delta=previous_command_delta,
            target_period_sec=rate_sec,
            step_elapsed_sec=0.0,
        ).keys()))
        diagnostic_writer.writeheader()
        print(f"[INFO] Writing action diagnostics CSV: {diagnostics_path}")
    try:
        while action_step < int(args.max_steps):
            chunk_timings: dict[str, float] = {}
            with profile_section(chunk_timings, "robot_obs_pre", bool(args.profile_timing)):
                joints = current_joint_vector(env, last_action)
            obs, _frames = _build_current_obs(
                args=args,
                live=live,
                joints=joints,
                obs_keys=obs_keys,
                image_shapes=image_shapes,
                camera_map=camera_map,
                timings=chunk_timings,
            )
            with profile_section(chunk_timings, "dp_policy", bool(args.profile_timing)):
                actions = runner.get_action(policy, obs)
            if bool(args.profile_timing):
                chunk_timings["chunk_total"] = sum(chunk_timings.values())
                print_info(f"[TIMING dp_chunk@step {action_step + 1:04d}] {format_timings(chunk_timings)} actions={len(actions)}")

            for action in actions:
                if action_step >= int(args.max_steps):
                    break
                loop_start = time.perf_counter()
                action_before_step = last_action.copy()
                command_action = prepare_action_for_execution(
                    action,
                    action_before_step,
                    args,
                    previous_command_delta=previous_command_delta,
                )
                last_action = execute_action(
                    env=env,
                    action=action,
                    last_action=action_before_step,
                    args=args,
                    first_action=first_action,
                    previous_command_delta=previous_command_delta,
                )
                first_action = False
                action_step += 1
                elapsed = time.perf_counter() - loop_start
                row = build_action_diagnostic_row(
                    step=action_step,
                    raw_policy_action=action,
                    command_action=command_action,
                    observed_after_step=last_action,
                    action_before_step=action_before_step,
                    previous_command_delta=previous_command_delta,
                    target_period_sec=rate_sec,
                    step_elapsed_sec=elapsed,
                )
                previous_command_delta = command_action - action_before_step
                if diagnostic_writer is not None:
                    diagnostic_writer.writerow(row)
                    diagnostic_file.flush()
                print(
                    f"[DP STEP {action_step:04d}] left_gripper={last_action[6]:.3f} right_gripper={last_action[13]:.3f} "
                    f"cmd_arm_delta={row['command_arm_delta']:.4f} cmd_arm_delta_change={row['command_arm_delta_change']:.4f} "
                    f"execute={bool(args.execute)}",
                    flush=True,
                )
                if bool(args.reobserve_each_action):
                    step_timings: dict[str, float] = {}
                    obs, _frames = _build_current_obs(
                        args=args,
                        live=live,
                        joints=current_joint_vector(env, last_action),
                        obs_keys=obs_keys,
                        image_shapes=image_shapes,
                        camera_map=camera_map,
                        timings=step_timings,
                    )
                    runner.update_obs(obs)
                else:
                    obs = dict(obs)
                    obs["agent_pos"] = last_action.copy()
                    runner.update_obs(obs)
                elapsed_total = time.perf_counter() - loop_start
                if rate_sec > 0 and elapsed_total < rate_sec:
                    time.sleep(rate_sec - elapsed_total)
                if bool(args.profile_timing) and should_print_step_timing(args, action_step):
                    overrun = max(0.0, elapsed_total - rate_sec) if rate_sec > 0 else 0.0
                    print_info(
                        f"[TIMING dp_step {action_step:04d}] step_total={elapsed_total * 1000.0:.1f}ms "
                        f"target_period={rate_sec * 1000.0:.1f}ms overrun={overrun * 1000.0:.1f}ms",
                    )
    finally:
        if diagnostic_file is not None:
            diagnostic_file.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RoboTwin DP image policies on the real ZED + xtrainer robot setup.")
    parser.add_argument("--task_name", default="grasp_mug")
    parser.add_argument("--task_config", default="demo_real_zed_sam2_objpc")
    parser.add_argument("--ckpt_setting", default="")
    parser.add_argument("--ckpt_path", default="")
    parser.add_argument("--expert_data_num", default="32")
    parser.add_argument("--seed", default="0")
    parser.add_argument("--checkpoint_num", default="3000")
    parser.add_argument("--gpu_id", default="0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n_obs_steps", type=int, default=3)
    parser.add_argument("--n_action_steps", type=int, default=6)

    parser.add_argument("--calibration_path", default=str(DEFAULT_CALIBRATION))
    parser.add_argument("--camera_labels", default="left")
    parser.add_argument("--dp_camera_map", default="")
    parser.add_argument("--zed_serials", default="")
    parser.add_argument("--zed_resolution", default="HD720")
    parser.add_argument("--zed_fps", type=int, default=15)
    parser.add_argument("--zed_depth_mode", default="NONE")
    parser.add_argument("--save_rgb_width", type=int, default=640)
    parser.add_argument("--save_rgb_height", type=int, default=360)
    parser.add_argument("--camera_warmup_timeout_sec", type=float, default=30.0)

    parser.add_argument("--robot_port", type=int, default=6001)
    parser.add_argument("--hostname", default="127.0.0.1")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--no_robot", action="store_true")
    parser.add_argument("--skip_robot_reset", action="store_true")
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--control_hz", type=float, default=10.0)
    parser.add_argument("--async_control", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--async_control_hz", type=float, default=25.0)
    parser.add_argument("--async_control_max_idle_repeats", type=int, default=-1)
    parser.add_argument("--async_control_replace_buffer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--initial_left_gripper", type=float, default=1.0)
    parser.add_argument("--initial_right_gripper", type=float, default=1.0)
    parser.add_argument("--interpolate_first_action", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--first_action_interp_step", type=float, default=0.001)
    parser.add_argument("--execution_substeps", type=int, default=1)
    parser.add_argument("--execution_substep_sleep_sec", type=float, default=0.0)
    parser.add_argument("--servo_j_t", type=float, default=0.06)
    parser.add_argument("--servo_j_gain", type=int, default=300)
    parser.add_argument("--max_executed_joint_delta", type=float, default=0.02)
    parser.add_argument("--max_executed_gripper_delta", type=float, default=0.04)
    parser.add_argument("--max_executed_joint_delta_change", type=float, default=0.005)
    parser.add_argument("--max_executed_gripper_delta_change", type=float, default=0.01)
    parser.add_argument("--max_action_delta", type=float, default=0.35)
    parser.add_argument("--disable_action_delta_safety", action="store_true")
    parser.add_argument("--disable_xyz_safety", action="store_true")
    parser.add_argument("--xyz_left_x_min", type=float, default=-450.0)
    parser.add_argument("--xyz_left_x_max", type=float, default=290.0)
    parser.add_argument("--xyz_right_x_min", type=float, default=-290.0)
    parser.add_argument("--xyz_right_x_max", type=float, default=450.0)
    parser.add_argument("--xyz_y_min", type=float, default=-750.0)
    parser.add_argument("--xyz_y_max", type=float, default=-160.0)
    parser.add_argument("--xyz_left_z_min", type=float, default=44.0)
    parser.add_argument("--xyz_right_z_min", type=float, default=40.0)
    parser.add_argument("--show_img", action="store_true")
    parser.add_argument("--reobserve_each_action", action="store_true")
    parser.add_argument("--profile_timing", action="store_true")
    parser.add_argument("--profile_timing_interval", type=int, default=1)
    parser.add_argument("--action_diagnostics_csv", default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if bool(args.async_control) and bool(args.reobserve_each_action):
        print_warning("[WARN] --reobserve_each_action is ignored when --async_control is enabled.")
    run_real_inference(args)


if __name__ == "__main__":
    main()
