from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R


DEFAULT_LEFT_IP = "192.168.5.1"
DEFAULT_RIGHT_IP = "192.168.5.2"


def dh_transform(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
    return np.array(
        [
            [cos_theta, -sin_theta * cos_alpha, sin_theta * sin_alpha, a * cos_theta],
            [sin_theta, cos_theta * cos_alpha, -cos_theta * sin_alpha, a * sin_theta],
            [0.0, sin_alpha, cos_alpha, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def offline_dobot_fk_matrix(
    joint_rad: np.ndarray,
    *,
    tool_x_m: float = 0.0,
    tool_y_m: float = 0.0,
    tool_z_m: float = 0.197,
) -> np.ndarray:
    """Local DH FK for the Dobot arm, returning base_from_tcp.

    Defaults match the xtrainer controller-side tool frame:
    `SetTool(1, 0, 0, 197, 0, 0, 0)`.
    """
    q = np.asarray(joint_rad, dtype=np.float64).reshape(6)
    dh_params = [
        (q[0], 0.2234, 0.0, np.pi / 2.0),
        (q[1] - np.pi / 2.0, 0.0, -0.280, 0.0),
        (q[2], 0.0, -0.225, 0.0),
        (q[3] - np.pi / 2.0, 0.1175, 0.0, np.pi / 2.0),
        (q[4], 0.120, 0.0, -np.pi / 2.0),
        (q[5], 0.088, 0.0, 0.0),
    ]

    transform = np.eye(4, dtype=np.float64)
    for params in dh_params:
        transform = transform @ dh_transform(*params)

    tool = np.eye(4, dtype=np.float64)
    tool[:3, 3] = np.array([tool_x_m, tool_y_m, tool_z_m], dtype=np.float64)
    return transform @ tool


def transform_from_xyz_euler_deg(xyzrxryrz: Any) -> np.ndarray:
    pose = np.asarray(xyzrxryrz, dtype=np.float64).reshape(6)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = R.from_euler("xyz", pose[3:6], degrees=True).as_matrix()
    transform[:3, 3] = pose[:3] / 1000.0
    return transform


def _parse_dobot_vector(response: Any, *, name: str) -> np.ndarray:
    text = str(response)
    match = re.search(r"\{([^}]*)\}", text)
    if match is None:
        raise ValueError(f"Could not parse Dobot {name} response: {text!r}")
    values = [float(item.strip()) for item in match.group(1).split(",") if item.strip()]
    if len(values) < 6:
        raise ValueError(f"Dobot {name} response has fewer than 6 values: {text!r}")
    return np.asarray(values[:6], dtype=np.float64)


def _rotation_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    rot_a = np.asarray(a, dtype=np.float64).reshape(4, 4)[:3, :3]
    rot_b = np.asarray(b, dtype=np.float64).reshape(4, 4)[:3, :3]
    if np.allclose(rot_a, rot_b, atol=1e-12, rtol=1e-12):
        return 0.0
    rel = rot_a.T @ rot_b
    trace_value = float(np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(trace_value)))


def arm_pose_error(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a = np.asarray(a, dtype=np.float64).reshape(4, 4)
    b = np.asarray(b, dtype=np.float64).reshape(4, 4)
    translation_error = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
    rotation_error = _rotation_error_deg(a, b)
    return translation_error, rotation_error


def _matrix_to_xyz_rotvec(transform: np.ndarray) -> list[float]:
    transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    rotvec = R.from_matrix(transform[:3, :3]).as_rotvec()
    return [*transform[:3, 3].astype(float).tolist(), *rotvec.astype(float).tolist()]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_xtrainer_dobot_robot():
    include_root = _repo_root() / "include" / "xtrainer_clover"
    if str(include_root) not in sys.path:
        sys.path.insert(0, str(include_root))
    from dobot_control.robots.dobot import DobotRobot  # noqa: PLC0415

    return DobotRobot


@dataclass
class ArmSample:
    arm: str
    sample_index: int
    joint_rad: np.ndarray
    offline: np.ndarray
    controller_fk: np.ndarray
    controller_pose: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        offline_vs_controller_fk = arm_pose_error(self.offline, self.controller_fk)
        getpose_vs_controller_fk = arm_pose_error(self.controller_pose, self.controller_fk)
        offline_vs_getpose = arm_pose_error(self.offline, self.controller_pose)
        return {
            "arm": self.arm,
            "sample_index": int(self.sample_index),
            "joint_deg": np.rad2deg(self.joint_rad).astype(float).tolist(),
            "offline_xyz_rotvec_m_rad": _matrix_to_xyz_rotvec(self.offline),
            "controller_fk_xyz_rotvec_m_rad": _matrix_to_xyz_rotvec(self.controller_fk),
            "controller_getpose_xyz_rotvec_m_rad": _matrix_to_xyz_rotvec(self.controller_pose),
            "offline_vs_controller_fk": {
                "translation_error_m": float(offline_vs_controller_fk[0]),
                "rotation_error_deg": float(offline_vs_controller_fk[1]),
            },
            "getpose_vs_controller_fk": {
                "translation_error_m": float(getpose_vs_controller_fk[0]),
                "rotation_error_deg": float(getpose_vs_controller_fk[1]),
            },
            "offline_vs_getpose": {
                "translation_error_m": float(offline_vs_getpose[0]),
                "rotation_error_deg": float(offline_vs_getpose[1]),
            },
        }


def _controller_fk_matrix(robot: Any, joint_rad: np.ndarray, *, user: int, tool: int) -> np.ndarray:
    joint_deg = np.rad2deg(np.asarray(joint_rad, dtype=np.float64).reshape(6))
    response = robot.r_inter.PositiveSolution(
        float(joint_deg[0]),
        float(joint_deg[1]),
        float(joint_deg[2]),
        float(joint_deg[3]),
        float(joint_deg[4]),
        float(joint_deg[5]),
        int(user),
        int(tool),
    )
    return transform_from_xyz_euler_deg(_parse_dobot_vector(response, name="PositiveSolution"))


def _controller_getpose_matrix(robot: Any) -> np.ndarray:
    response = robot.r_inter.GetPose()
    return transform_from_xyz_euler_deg(_parse_dobot_vector(response, name="GetPose"))


def sample_arm(
    *,
    arm: str,
    robot: Any,
    sample_index: int,
    user: int,
    tool: int,
    tool_x_m: float,
    tool_y_m: float,
    tool_z_m: float,
) -> ArmSample:
    joint_state = np.asarray(robot.get_joint_state(), dtype=np.float64).reshape(-1)
    joint_rad = joint_state[:6].copy()
    return ArmSample(
        arm=arm,
        sample_index=int(sample_index),
        joint_rad=joint_rad,
        offline=offline_dobot_fk_matrix(
            joint_rad,
            tool_x_m=float(tool_x_m),
            tool_y_m=float(tool_y_m),
            tool_z_m=float(tool_z_m),
        ),
        controller_fk=_controller_fk_matrix(robot, joint_rad, user=int(user), tool=int(tool)),
        controller_pose=_controller_getpose_matrix(robot),
    )


def _print_sample(sample: dict[str, Any]) -> None:
    offline_fk = sample["offline_vs_controller_fk"]
    getpose_fk = sample["getpose_vs_controller_fk"]
    print(
        "[FK CHECK] "
        f"arm={sample['arm']} sample={sample['sample_index']} "
        f"offline_vs_PositiveSolution: trans={offline_fk['translation_error_m']:.6f}m "
        f"rot={offline_fk['rotation_error_deg']:.3f}deg | "
        f"GetPose_vs_PositiveSolution: trans={getpose_fk['translation_error_m']:.6f}m "
        f"rot={getpose_fk['rotation_error_deg']:.3f}deg",
        flush=True,
    )
    print(f"  joint_deg={np.round(np.asarray(sample['joint_deg'], dtype=np.float64), 3).tolist()}", flush=True)
    print(
        "  xyz offline/controller_fk/getpose="
        f"{np.round(np.asarray(sample['offline_xyz_rotvec_m_rad'][:3]), 4).tolist()} / "
        f"{np.round(np.asarray(sample['controller_fk_xyz_rotvec_m_rad'][:3]), 4).tolist()} / "
        f"{np.round(np.asarray(sample['controller_getpose_xyz_rotvec_m_rad'][:3]), 4).tolist()}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare local DH FK with Dobot controller FK/GetPose. "
            "This script only reads robot state and does not command motion."
        )
    )
    parser.add_argument("--arm", choices=["left", "right", "both"], default="both")
    parser.add_argument("--left_ip", default=DEFAULT_LEFT_IP)
    parser.add_argument("--right_ip", default=DEFAULT_RIGHT_IP)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--interval_sec", type=float, default=0.5)
    parser.add_argument("--user", type=int, default=0)
    parser.add_argument("--tool", type=int, default=1)
    parser.add_argument("--tool_x_m", type=float, default=0.0)
    parser.add_argument("--tool_y_m", type=float, default=0.0)
    parser.add_argument("--tool_z_m", type=float, default=0.197)
    parser.add_argument("--max_translation_error_m", type=float, default=0.02)
    parser.add_argument("--max_rotation_error_deg", type=float, default=5.0)
    parser.add_argument("--json_out", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    DobotRobot = _load_xtrainer_dobot_robot()
    arm_ips = []
    if args.arm in {"left", "both"}:
        arm_ips.append(("left", args.left_ip))
    if args.arm in {"right", "both"}:
        arm_ips.append(("right", args.right_ip))

    robots = {
        arm: DobotRobot(robot_ip=ip, no_gripper=True, robot_number=2)
        for arm, ip in arm_ips
    }

    samples: list[dict[str, Any]] = []
    print(
        "[INFO] FK consistency check started. No robot motion commands will be sent. "
        f"tool=({args.tool_x_m}, {args.tool_y_m}, {args.tool_z_m})m user={args.user} tool_index={args.tool}",
        flush=True,
    )
    for sample_index in range(max(1, int(args.samples))):
        for arm, robot in robots.items():
            sample = sample_arm(
                arm=arm,
                robot=robot,
                sample_index=sample_index,
                user=int(args.user),
                tool=int(args.tool),
                tool_x_m=float(args.tool_x_m),
                tool_y_m=float(args.tool_y_m),
                tool_z_m=float(args.tool_z_m),
            ).to_dict()
            samples.append(sample)
            _print_sample(sample)
        if sample_index + 1 < int(args.samples):
            time.sleep(max(0.0, float(args.interval_sec)))

    max_translation = max(
        sample["offline_vs_controller_fk"]["translation_error_m"]
        for sample in samples
    )
    max_rotation = max(
        sample["offline_vs_controller_fk"]["rotation_error_deg"]
        for sample in samples
    )
    payload = {
        "samples": samples,
        "summary": {
            "max_offline_vs_controller_fk_translation_error_m": float(max_translation),
            "max_offline_vs_controller_fk_rotation_error_deg": float(max_rotation),
            "max_translation_error_m": float(args.max_translation_error_m),
            "max_rotation_error_deg": float(args.max_rotation_error_deg),
            "pass": bool(
                max_translation <= float(args.max_translation_error_m)
                and max_rotation <= float(args.max_rotation_error_deg)
            ),
        },
    }
    if str(args.json_out).strip():
        out_path = Path(args.json_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[INFO] Wrote FK check JSON: {out_path}")

    print(
        "[SUMMARY] "
        f"max offline_vs_PositiveSolution trans={max_translation:.6f}m rot={max_rotation:.3f}deg "
        f"thresholds=({float(args.max_translation_error_m):.6f}m, {float(args.max_rotation_error_deg):.3f}deg) "
        f"pass={payload['summary']['pass']}",
        flush=True,
    )
    return 0 if payload["summary"]["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
