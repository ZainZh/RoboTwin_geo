#!/usr/bin/env python3

from __future__ import annotations

import datetime
import os
import sys
import threading
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from script.real_zed_collection.real_zed_utils import ensure_dir, load_three_zed_calibration, write_json


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "real_zed_collection.yaml"


def parse_camera_labels(value: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


@dataclass
class Args:
    robot_port: int = 6001
    hostname: str = "127.0.0.1"
    save_data_path: str = "./real_data"
    project_name: str = "robotwin_real_zed"
    calibration_path: str = ""
    camera_labels: str = "global,left,right"
    zed_serials: list[int] = field(default_factory=list)
    zed_resolution: str = "HD720"
    zed_fps: int = 15
    zed_depth_mode: str = "NEURAL"
    show_img: bool = False
    save_zed_xyzrgba: bool = False
    max_frames: int = -1


def _extract_config_path(argv: list[str]) -> tuple[Path | None, list[str]]:
    config_path = DEFAULT_CONFIG_PATH if DEFAULT_CONFIG_PATH.exists() else None
    remaining_args: list[str] = []
    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if arg == "--config":
            if idx + 1 >= len(argv):
                raise SystemExit("--config requires a path.")
            config_path = Path(argv[idx + 1]).expanduser().resolve()
            idx += 2
            continue
        if arg.startswith("--config="):
            config_path = Path(arg.split("=", 1)[1]).expanduser().resolve()
            idx += 1
            continue
        remaining_args.append(arg)
        idx += 1
    return config_path, remaining_args


def _load_args_defaults(config_path: Path | None) -> Args:
    if config_path is None:
        return Args()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")

    allowed = {item.name for item in fields(Args)}
    unknown = sorted(str(key) for key in data.keys() if str(key) not in allowed)
    if unknown:
        raise ValueError(f"Unknown config keys in {config_path}: {unknown}")

    defaults = Args(**{name: data[name] for name in allowed if name in data})
    print(f"[INFO] Loaded collection config: {config_path}")
    return defaults


# Same button-state semantics as include/xtrainer_clover/experiments/run_control.py.
what_to_do = np.array(([0, 0, 0], [0, 0, 0]))
dt_time = np.array([20240507161455])
is_falling = np.array([0])


def button_monitor_realtime(agent) -> None:
    last_keys_status = np.array(([0, 0], [0, 0]))
    start_press_status = np.array(([0, 0], [0, 0]))
    keys_press_count = np.array(([0, 0, 0], [0, 0, 0]))
    tic = time.time()

    while True:
        now_keys = agent.get_keys()
        dev_keys = now_keys - last_keys_status
        for i in range(2):
            if dev_keys[i, 0] == -1:
                tic = time.time()
                start_press_status[i, 0] = 1
            if dev_keys[i, 0] == 1 and start_press_status[i, 0]:
                start_press_status[i, 0] = 0
                toc = time.time()
                if toc - tic < 0.5:
                    keys_press_count[i, 0] += 1
                    what_to_do[i, 0] = 1 if keys_press_count[i, 0] % 2 == 1 else 0
                    print(f"ButtonA: [{i}] {'unlock' if what_to_do[i, 0] else 'lock'}", what_to_do)
                elif toc - tic > 1:
                    keys_press_count[i, 1] += 1
                    what_to_do[i, 1] = 1 if keys_press_count[i, 1] % 2 == 1 else 0
                    print(f"ButtonA: [{i}] {'servo' if what_to_do[i, 1] else 'stop servo'}")

        for i in range(2):
            if dev_keys[i, 1] == -1:
                start_press_status[i, 1] = 1
            if dev_keys[i, 1] == 1:
                start_press_status[i, 1] = 0
                if keys_press_count[0, 2] % 2 == 1:
                    if keys_press_count[0, 1] % 2 == 1 or keys_press_count[1, 1] % 2 == 1:
                        what_to_do[0, 2] = 1
                        now_time = datetime.datetime.now()
                        dt_time[0] = int(now_time.strftime("%Y%m%d%H%M%S"))
                        keys_press_count[0, 2] += 1
                else:
                    what_to_do[0, 2] = 0
                    keys_press_count[0, 2] += 1

        last_keys_status = now_keys


class SharedZedFrame:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.frame: dict[str, Any] | None = None
        self.error: str | None = None

    def update(self, frame: dict[str, Any]) -> None:
        with self.lock:
            self.frame = frame
            self.error = None

    def set_error(self, error: str) -> None:
        with self.lock:
            self.error = error

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            if self.error:
                raise RuntimeError(self.error)
            if self.frame is None:
                raise RuntimeError("ZED frame is not ready yet.")
            return {key: np.copy(value) if isinstance(value, np.ndarray) else value for key, value in self.frame.items()}


def dh_transformation_matrix(theta, d, a, alpha):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
    return np.array(
        [
            [cos_theta, -sin_theta * cos_alpha, sin_theta * sin_alpha, a * cos_theta],
            [sin_theta, cos_theta * cos_alpha, -cos_theta * sin_alpha, a * sin_theta],
            [0, sin_alpha, cos_alpha, d],
            [0, 0, 0, 1],
        ]
    )


def claw_width(coef):
    claw_servo = 2.3818 - coef * 1.5401
    cos_claw_servo = np.cos(claw_servo)
    return 0.03 * cos_claw_servo + 0.5 * np.sqrt(0.0036 * cos_claw_servo**2 + 0.0028)


def forward_kinematics(q0, q1, q2, q3, q4, q5, y):
    dh_params = [
        (q0, 0.2234, 0, np.pi / 2),
        (q1 - np.pi / 2, 0, -0.280, 0),
        (q2, 0, -0.225, 0),
        (q3 - np.pi / 2, 0.1175, 0, np.pi / 2),
        (q4, 0.120, 0, -np.pi / 2),
        (q5, 0.088, 0, 0),
    ]
    t = np.eye(4)
    for params in dh_params:
        t = np.dot(t, dh_transformation_matrix(*params))
    t_tool = np.eye(4)
    t_tool[:3, 3] = np.array([0, y, 0.2])
    return np.dot(t, t_tool)[:3, 3]


def calculate_vel_pos(action, last_action, total_time):
    total_time = max(float(total_time), 1e-6)
    claw_left = claw_width(action[6])
    claw_right = claw_width(action[13])
    positions = {}
    vel = {}
    for side in ["left", "right"]:
        for paw in ["left", "right"]:
            coef = 1 if paw == "left" else -1
            claw = (claw_left if side == "left" else claw_right) * coef
            current_fk = forward_kinematics(*(action[0:6] if side == "left" else action[7:13]), claw)
            last_fk = forward_kinematics(*(last_action[0:6] if side == "left" else last_action[7:13]), claw)
            positions[f"{side}_{paw}"] = current_fk
            vel[f"{side}_{paw}"] = (current_fk - last_fk) / total_time
    return positions, vel


def is_within_safe_position(position, x_range, y_range, z_min):
    return x_range[0] <= position[0] <= x_range[1] and y_range[0] <= position[1] <= y_range[1] and position[2] > z_min


def check_pose_protection(positions, vel, servo_state):
    protect_err = False
    positions_mm = {key: value * 1000 for key, value in positions.items()}
    x_range_left = (-450, 290)
    x_range_right = (-290, 450)
    y_range = (-750, -160)
    z_range_left = 44
    z_range_right = 42

    if servo_state[0, 1]:
        if vel["left_left"][2] < -1 or vel["left_right"][2] < -1:
            print("[Warn]:The left robot speed of the TCP is moving too fast!")
            protect_err = True
        if not all(
            is_within_safe_position(positions_mm[pos], x_range_left, y_range, z_range_left)
            for pos in ["left_left", "left_right"]
        ):
            print("[Warn]:The left arm is out of the safe zone!")
            protect_err = True

    if servo_state[1, 1]:
        if vel["right_left"][2] < -1 or vel["right_right"][2] < -1:
            print("[Warn]:The right robot speed of the TCP is moving too fast!")
            protect_err = True
        if not all(
            is_within_safe_position(positions_mm[pos], x_range_right, y_range, z_range_right)
            for pos in ["right_left", "right_right"]
        ):
            print("[Warn]:The right arm is out of the safe zone!")
            protect_err = True
    return protect_err


def check_joint_safety(action):
    protect_err = False
    if not (action[2] < 0):
        print("[Warn]:The J3 joints of the left robotic arm are out of the safe position!")
        protect_err = True
    if not (action[9] > 0):
        print("[Warn]:The J3 joints of the right robotic arm are out of the safe position!")
        protect_err = True
    if protect_err:
        print(action)
    return protect_err


def _resolution_enum(sl, name: str):
    return {
        "HD2K": sl.RESOLUTION.HD2K,
        "HD1080": sl.RESOLUTION.HD1080,
        "HD720": sl.RESOLUTION.HD720,
        "VGA": sl.RESOLUTION.VGA,
    }.get(str(name).upper(), sl.RESOLUTION.HD720)


def _depth_mode_enum(sl, name: str):
    return {
        "NONE": sl.DEPTH_MODE.NONE,
        "PERFORMANCE": sl.DEPTH_MODE.PERFORMANCE,
        "QUALITY": sl.DEPTH_MODE.QUALITY,
        "ULTRA": sl.DEPTH_MODE.ULTRA,
        "NEURAL": getattr(sl.DEPTH_MODE, "NEURAL", sl.DEPTH_MODE.QUALITY),
        "NEURAL_PLUS": getattr(sl.DEPTH_MODE, "NEURAL_PLUS", getattr(sl.DEPTH_MODE, "NEURAL", sl.DEPTH_MODE.QUALITY)),
    }.get(str(name).upper(), getattr(sl.DEPTH_MODE, "NEURAL", sl.DEPTH_MODE.QUALITY))


def zed_capture_loop(
    *,
    label: str,
    serial: int,
    resolution: str,
    fps: int,
    depth_mode: str,
    save_xyzrgba: bool,
    shared: SharedZedFrame,
    stop_event: threading.Event,
) -> None:
    try:
        import pyzed.sl as sl
    except Exception as e:
        shared.set_error(f"pyzed.sl import failed for camera {label}: {e}")
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

    runtime = sl.RuntimeParameters()
    image_mat = sl.Mat()
    depth_mat = sl.Mat()
    xyzrgba_mat = sl.Mat()
    frame_idx = 0
    try:
        while not stop_event.is_set():
            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                continue
            zed.retrieve_image(image_mat, sl.VIEW.LEFT)
            zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
            image_raw = image_mat.get_data()
            depth_raw = depth_mat.get_data()
            if image_raw is None or depth_raw is None:
                continue
            bgr = image_raw[:, :, :3] if image_raw.ndim == 3 and image_raw.shape[2] >= 3 else image_raw
            rgb = cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2RGB)
            frame = {
                "label": label,
                "serial": int(serial),
                "frame_index": int(frame_idx),
                "timestamp_unix_sec": time.time(),
                "rgb": rgb.astype(np.uint8),
                "depth_m": np.asarray(depth_raw, dtype=np.float32),
            }
            if save_xyzrgba:
                zed.retrieve_measure(xyzrgba_mat, sl.MEASURE.XYZRGBA)
                xyzrgba = xyzrgba_mat.get_data()
                if xyzrgba is not None:
                    frame["xyzrgba"] = np.asarray(xyzrgba, dtype=np.float32)
            shared.update(frame)
            frame_idx += 1
    finally:
        zed.close()


def _resolve_cameras(args: Args) -> tuple[list[str], list[int]]:
    labels = parse_camera_labels(args.camera_labels)
    serials = [int(x) for x in args.zed_serials]
    if args.calibration_path:
        calib = load_three_zed_calibration(args.calibration_path)
        if not labels:
            labels = list(calib.keys())
        if not serials:
            serials = [int(calib[label].serial_number) for label in labels]
    if len(labels) != 3:
        raise ValueError(f"First version expects exactly 3 camera labels, got {labels}")
    if len(serials) != 3:
        raise ValueError("Provide exactly 3 ZED serials or a calibration file with serial_number entries.")
    print(f"[INFO] Collection camera label -> serial mapping: {dict(zip(labels, serials))}")
    return labels, serials


def _save_raw_frame(
    *,
    episode_dir: Path,
    frame_idx: int,
    obs: dict[str, Any],
    action: np.ndarray,
    camera_frames: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    robot_path = episode_dir / f"robot_{frame_idx:06d}.npz"
    joint_positions = np.asarray(obs.get("joint_positions", action), dtype=np.float32).reshape(-1)
    if joint_positions.shape[0] != 14:
        raise ValueError(f"Expected 14-dim joint_positions, got {joint_positions.shape}")
    np.savez(
        robot_path,
        timestamp_unix_sec=np.asarray(time.time(), dtype=np.float64),
        joint_vector=joint_positions,
        joint_positions=joint_positions,
        joint_velocities=np.asarray(obs.get("joint_velocities", np.zeros_like(joint_positions)), dtype=np.float32),
        control=np.asarray(action, dtype=np.float32),
    )

    camera_rel_paths = {}
    for label, frame in camera_frames.items():
        camera_path = episode_dir / f"{label}_{frame_idx:06d}.npz"
        payload = {
            "timestamp_unix_sec": np.asarray(frame["timestamp_unix_sec"], dtype=np.float64),
            "camera_frame_index": np.asarray(frame["frame_index"], dtype=np.int64),
            "rgb": np.asarray(frame["rgb"], dtype=np.uint8),
            "depth_m": np.asarray(frame["depth_m"], dtype=np.float32),
        }
        if "xyzrgba" in frame:
            payload["xyzrgba"] = np.asarray(frame["xyzrgba"], dtype=np.float32)
        np.savez_compressed(camera_path, **payload)
        camera_rel_paths[label] = camera_path.name

    return {
        "frame_index": int(frame_idx),
        "timestamp_unix_sec": time.time(),
        "robot": robot_path.name,
        "cameras": camera_rel_paths,
    }


def _write_manifest(episode_dir: Path, manifest: dict[str, Any]) -> None:
    write_json(episode_dir / "manifest.json", manifest)


def main(args: Args) -> None:
    include_dir = Path(__file__).resolve().parents[2] / "include" / "xtrainer_clover"
    sys.path.append(str(include_dir))
    from dobot_control.agents.agent import BimanualAgent
    from dobot_control.agents.dobot_agent import DobotAgent
    from dobot_control.env import RobotEnv
    from dobot_control.robots.robot_node import ZMQClientRobot
    from scripts.manipulate_utils import (
        dynamic_approach,
        load_ini_data_hands,
        robot_pose_init,
        servo_action_check,
        set_light,
    )

    labels, serials = _resolve_cameras(args)
    save_root = ensure_dir(Path(args.save_data_path) / args.project_name / "real_zed_raw")

    stop_event = threading.Event()
    shared_by_label = {label: SharedZedFrame() for label in labels}
    camera_threads = []
    for label, serial in zip(labels, serials):
        thread = threading.Thread(
            target=zed_capture_loop,
            kwargs={
                "label": label,
                "serial": serial,
                "resolution": args.zed_resolution,
                "fps": int(args.zed_fps),
                "depth_mode": args.zed_depth_mode,
                "save_xyzrgba": bool(args.save_zed_xyzrgba),
                "shared": shared_by_label[label],
                "stop_event": stop_event,
            },
            daemon=True,
        )
        thread.start()
        camera_threads.append(thread)
    time.sleep(2.0)
    print(f"ZED camera threads initialized: {dict(zip(labels, serials))}")

    _, hands_dict = load_ini_data_hands()
    left_agent = DobotAgent(which_hand="LEFT", dobot_config=hands_dict["HAND_LEFT"])
    right_agent = DobotAgent(which_hand="RIGHT", dobot_config=hands_dict["HAND_RIGHT"])
    agent = BimanualAgent(left_agent, right_agent)

    robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
    env = RobotEnv(robot_client)
    env.set_do_status([1, 0])
    env.set_do_status([2, 0])
    env.set_do_status([3, 0])
    robot_pose_init(env)
    print("robot init success")

    last_status = np.array(([0, 0, 0], [0, 0, 0]))
    button_thread = threading.Thread(target=button_monitor_realtime, args=(agent,), daemon=True)
    button_thread.start()
    print("button thread init success")

    start_servo = False
    curr_light = "dark"
    total_time = 0.04
    safe_limit = 0
    frame_count = 0
    active_episode_dir: Path | None = None
    manifest: dict[str, Any] | None = None

    try:
        while True:
            tic = time.time()
            for thread, label in zip(camera_threads, labels):
                if not thread.is_alive():
                    raise RuntimeError(f"ZED thread stopped: {label}")
            if np.any(is_falling):
                raise RuntimeError("sensor detection")

            action = agent.act({})
            dev_what_to_do = what_to_do.copy() - last_status
            last_status = what_to_do.copy()

            for i in range(2):
                if dev_what_to_do[i, 0] != 0:
                    agent.set_torque(i, not what_to_do[i, 0])

            if dev_what_to_do[0, 1] == 1 or dev_what_to_do[1, 1] == 1:
                print("dynamic approach")
                for i in range(2):
                    if what_to_do[i, 1]:
                        agent.set_torque(i, True)
                flag_in = np.array([what_to_do[0, 1], what_to_do[1, 1]])
                last_action = dynamic_approach(env, agent, flag_in)
                for i in range(2):
                    if what_to_do[i, 0] and what_to_do[i, 1]:
                        agent.set_torque(i, False)
                start_servo = True
                obs = env.get_obs()
                if curr_light != "green":
                    curr_light = set_light(env, "yellow", 1)

            if dev_what_to_do[0, 1] == -1 or dev_what_to_do[1, 1] == -1:
                if what_to_do[0, 1] == 0 and what_to_do[1, 1] == 0:
                    set_light(env, "green", 0)

            if (what_to_do[0, 1] or what_to_do[1, 1]) and start_servo:
                action = agent.act({})
                flag_in = np.array([what_to_do[0, 1], what_to_do[1, 1]])
                err3, action = servo_action_check(action, last_action, flag_in)
                assert err3 != 0, set_light(env, "red", 1)
                if safe_limit < 1:
                    safe_limit += 1
                else:
                    positions, vel = calculate_vel_pos(action, last_action, total_time)
                    protect_err = [
                        check_pose_protection(positions, vel, what_to_do),
                        check_joint_safety(action),
                    ]
                    if any(protect_err):
                        set_light(env, "red", 1)
                        time.sleep(1)
                        raise RuntimeError("Safety protection triggered.")

                if dev_what_to_do[0, 2] == 1:
                    curr_light = set_light(env, "green", 1)
                elif dev_what_to_do[0, 2] == -1:
                    curr_light = set_light(env, "yellow", 1)
                    if active_episode_dir is not None and manifest is not None:
                        _write_manifest(active_episode_dir, manifest)
                    active_episode_dir = None
                    manifest = None

                if what_to_do[0, 2] == 1:
                    if active_episode_dir is None:
                        active_episode_dir = ensure_dir(save_root / f"episode_{int(dt_time[0])}")
                        manifest = {
                            "format": "robotwin_real_zed_raw_v1",
                            "created_at_unix_sec": time.time(),
                            "camera_labels": labels,
                            "camera_serials": dict(zip(labels, serials)),
                            "calibration_path": str(Path(args.calibration_path).expanduser().resolve()) if args.calibration_path else "",
                            "frames": [],
                        }
                    camera_frames = {label: shared_by_label[label].snapshot() for label in labels}
                    frame_record = _save_raw_frame(
                        episode_dir=active_episode_dir,
                        frame_idx=len(manifest["frames"]),
                        obs=obs,
                        action=np.asarray(action, dtype=np.float32),
                        camera_frames=camera_frames,
                    )
                    manifest["frames"].append(frame_record)
                    _write_manifest(active_episode_dir, manifest)
                    frame_count += 1
                    print(f"saved raw frame {frame_count} -> {active_episode_dir}", end="\r")
                    if args.max_frames > 0 and frame_count >= int(args.max_frames):
                        break

                obs = env.step(action, flag_in)
                obs["joint_positions"][6] = action[6]
                obs["joint_positions"][13] = action[13]
                last_action = action
            else:
                start_servo = False
                safe_limit = 0

            if args.show_img:
                panels = []
                for label in labels:
                    rgb = shared_by_label[label].snapshot()["rgb"]
                    panels.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                canvas = cv2.hconcat([cv2.resize(panel, (480, 270)) for panel in panels])
                cv2.imshow("real_zed_collection", canvas)
                cv2.waitKey(1)

            total_time = time.time() - tic
    finally:
        stop_event.set()
        if active_episode_dir is not None and manifest is not None:
            _write_manifest(active_episode_dir, manifest)
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    import tyro

    config_path, cli_args = _extract_config_path(sys.argv[1:])
    default_args = _load_args_defaults(config_path)
    main(tyro.cli(Args, args=cli_args, default=default_args))
