from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import sapien
import transforms3d as t3d

from .placement_metrics import (
    functional_pose_alignment_errors,
    functional_pose_alignment_success,
)
from .place_shoe_rotating_block import place_shoe_rotating_block
from .utils import Actor, ArmTag, create_actor, preprocess, rand_pose


RAMP_FUNCTIONAL_ROTATION = np.array(
    [
        [0.0, -1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)


def resolve_shoe_id_candidates(config: dict) -> tuple[int, ...]:
    """Resolve an evaluation-only object subset without exposing IDs to policy."""
    modelname = str(config.get("shoe_modelname", "041_shoe"))
    default_values = list(range(10)) if modelname == "041_shoe" else None
    values = config.get("allowed_shoe_ids", default_values)
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(
            "geometry_marker.allowed_shoe_ids must be a non-empty list; "
            "external shoe assets require an explicit frozen ID list"
        )
    result = tuple(int(value) for value in values)
    if len(set(result)) != len(result):
        raise ValueError("geometry_marker.allowed_shoe_ids must not contain duplicates")
    invalid = [
        value
        for value in result
        if value < 0 or (modelname == "041_shoe" and value >= 10)
    ]
    if invalid:
        raise ValueError(f"geometry_marker.allowed_shoe_ids contains invalid IDs: {invalid}")
    return result


def resolve_shoe_modelname(config: dict) -> str:
    modelname = str(config.get("shoe_modelname", "041_shoe"))
    if not modelname or "/" in modelname or "\\" in modelname or modelname in {".", ".."}:
        raise ValueError(f"invalid geometry_marker.shoe_modelname: {modelname!r}")
    return modelname


def loaded_shoe_footprint(modelname: str, model_id: int) -> np.ndarray:
    """Return the shoe's loaded half width/length in its local X/Z plane."""
    path = Path("assets/objects") / modelname / f"model_data{int(model_id)}.json"
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    extents = np.asarray(data["extents"], dtype=np.float64)
    scale = np.asarray(data["scale"], dtype=np.float64)
    if scale.ndim == 0:
        scale = np.full((3,), float(scale), dtype=np.float64)
    loaded = extents * scale
    return 0.5 * loaded[[0, 2]]


def oriented_rectangles_overlap(
    center_a: np.ndarray,
    axes_a: np.ndarray,
    half_extents_a: np.ndarray,
    center_b: np.ndarray,
    axes_b: np.ndarray,
    half_extents_b: np.ndarray,
    clearance: float = 0.0,
) -> bool:
    """Two-dimensional separating-axis test for task initialization."""
    center_a = np.asarray(center_a, dtype=np.float64)
    center_b = np.asarray(center_b, dtype=np.float64)
    axes_a = np.asarray(axes_a, dtype=np.float64)
    axes_b = np.asarray(axes_b, dtype=np.float64)
    half_extents_a = np.asarray(half_extents_a, dtype=np.float64)
    half_extents_b = np.asarray(half_extents_b, dtype=np.float64)
    delta = center_b - center_a
    for axis in np.concatenate([axes_a, axes_b], axis=0):
        axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
        radius_a = float(np.sum(half_extents_a * np.abs(axes_a @ axis)))
        radius_b = float(np.sum(half_extents_b * np.abs(axes_b @ axis)))
        if abs(float(delta @ axis)) > radius_a + radius_b + float(clearance):
            return False
    return True


def shoe_overlaps_ramp_footprint(
    shoe_pose: sapien.Pose,
    shoe_half_extents: np.ndarray,
    target_xy: np.ndarray,
    target_yaw: float,
    ramp_half_extents: np.ndarray,
    clearance: float = 0.005,
) -> bool:
    shoe_rotation = t3d.quaternions.quat2mat(np.asarray(shoe_pose.q, dtype=np.float64))
    shoe_axes = np.stack([shoe_rotation[:2, 0], shoe_rotation[:2, 2]], axis=0)
    ramp_rotation = t3d.euler.euler2mat(0.0, 0.0, float(target_yaw))
    ramp_axes = np.stack([ramp_rotation[:2, 0], ramp_rotation[:2, 1]], axis=0)
    return oriented_rectangles_overlap(
        np.asarray(shoe_pose.p[:2], dtype=np.float64),
        shoe_axes,
        shoe_half_extents,
        np.asarray(target_xy, dtype=np.float64),
        ramp_axes,
        ramp_half_extents,
        clearance,
    )


def geometry_marker_local_transform(
    marker_x: float, marker_y: float, marker_yaw: float
) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = t3d.euler.euler2mat(0.0, 0.0, float(marker_yaw))
    result[:3, 3] = [float(marker_x), float(marker_y), 0.0]
    return result


def geometry_marker_functional_matrix(
    marker_x: float, marker_y: float, marker_yaw: float
) -> np.ndarray:
    marker = geometry_marker_local_transform(marker_x, marker_y, marker_yaw)
    base_functional = np.eye(4, dtype=np.float64)
    base_functional[:3, :3] = RAMP_FUNCTIONAL_ROTATION
    return marker @ base_functional


def create_geometry_marker_ramp(
    scene,
    *,
    pose: sapien.Pose,
    marker_x: float,
    marker_y: float,
    marker_yaw: float,
    half_length: float,
    half_width: float,
    half_thickness: float = 0.0005,
    name: str = "geometry_marker_ramp",
) -> Actor:
    physical_scene, pose = preprocess(scene, pose)
    builder = physical_scene.create_actor_builder()
    builder.set_physx_body_type("static")

    base_half_size = [float(half_length), float(half_width), float(half_thickness)]
    base_pose = sapien.Pose([0.0, 0.0, 0.0])
    builder.add_box_collision(
        pose=base_pose,
        half_size=base_half_size,
        material=physical_scene.default_physical_material,
    )
    builder.add_box_visual(
        pose=base_pose,
        half_size=base_half_size,
        material=[0.10, 0.20, 0.75],
    )

    marker_rotation = t3d.euler.euler2mat(0.0, 0.0, float(marker_yaw))
    marker_quaternion = t3d.quaternions.mat2quat(marker_rotation)
    marker_center = np.array([marker_x, marker_y, 0.004], dtype=np.float64)
    longitudinal_half_size = np.array([0.050, 0.007, 0.0025], dtype=np.float64)
    crossbar_half_size = np.array([0.007, 0.035, 0.0025], dtype=np.float64)
    crossbar_offset = marker_rotation @ np.array([0.043, 0.0, 0.0])

    # These high-contrast marker bars are visual geometry only.  They encode the
    # desired local frame without changing the support surface contact physics.
    builder.add_box_visual(
        pose=sapien.Pose(marker_center, marker_quaternion),
        half_size=longitudinal_half_size.tolist(),
        material=[0.95, 0.80, 0.05],
    )
    builder.add_box_visual(
        pose=sapien.Pose(marker_center + crossbar_offset, marker_quaternion),
        half_size=crossbar_half_size.tolist(),
        material=[0.95, 0.80, 0.05],
    )

    builder.set_initial_pose(pose)
    entity = builder.build(name=name)
    functional = geometry_marker_functional_matrix(marker_x, marker_y, marker_yaw)
    data = {
        "center": [0.0, 0.0, 0.0],
        "extents": base_half_size,
        "scale": [1.0, 1.0, 1.0],
        "functional_matrix": [functional.tolist()],
        "geometry_marker": {
            "x": float(marker_x),
            "y": float(marker_y),
            "yaw_rad": float(marker_yaw),
        },
    }
    return Actor(entity, data)


class place_shoe_geometry_marker(place_shoe_rotating_block):
    """Place a shoe at a visible, continuously varying geometric target frame."""

    def setup_demo(self, is_test=False, **kwags):
        self.geometry_marker_config = dict(kwags.get("geometry_marker", {}))
        self.placement_success_config = dict(kwags.get("placement_success", {}))
        self._placement_phase_started = False
        self._reset_evaluation_trackers()
        super().setup_demo(is_test=is_test, **kwags)

    def _reset_evaluation_trackers(self):
        self._success_hold_count = 0
        self._max_success_hold_count = 0
        self._pose_alignment_step_count = 0
        self._raw_success_step_count = 0
        self._first_pose_alignment_step = None
        self._first_raw_success_step = None
        self._min_translation_error_norm_m = float("inf")
        self._min_rotation_error_deg = float("inf")
        self._best_combined_alignment_score = float("inf")
        self._best_combined_translation_error_norm_m = float("inf")
        self._best_combined_rotation_error_deg = float("inf")

    def _ensure_evaluation_trackers(self):
        # Some lightweight unit-test fixtures instantiate the task without the
        # full simulator setup path. Production episodes always reset in
        # setup_demo, while this guard keeps check_success independently safe.
        if not hasattr(self, "_min_translation_error_norm_m"):
            self._reset_evaluation_trackers()

    def get_object_pointcloud_targets(self):
        return {"{A}": "shoe", "{B}": "target_block"}

    def load_actors(self):
        config = self.geometry_marker_config
        target_yaw = np.random.uniform(-np.pi, np.pi)
        ramp_pitch = np.deg2rad(float(config.get("ramp_pitch_deg", 10.0)))
        half_length = float(config.get("ramp_half_length", 0.18))
        half_width = float(config.get("ramp_half_width", 0.13))
        marker_x_limit = float(config.get("marker_x_limit", 0.06))
        marker_y_limit = float(config.get("marker_y_limit", 0.035))
        marker_yaw_limit = np.deg2rad(float(config.get("marker_yaw_limit_deg", 60.0)))

        self.marker_x = float(np.random.uniform(-marker_x_limit, marker_x_limit))
        self.marker_y = float(np.random.uniform(-marker_y_limit, marker_y_limit))
        self.marker_yaw = float(np.random.uniform(-marker_yaw_limit, marker_yaw_limit))

        target_center_z = 0.74 + half_length * np.sin(abs(ramp_pitch))
        yaw_mat = t3d.euler.euler2mat(0.0, 0.0, target_yaw)
        pitch_mat = t3d.euler.euler2mat(0.0, ramp_pitch, 0.0)
        target_quat = t3d.quaternions.mat2quat(yaw_mat @ pitch_mat)
        target_pose = sapien.Pose([0.0, -0.08, target_center_z], target_quat)
        self.target_block = create_geometry_marker_ramp(
            self,
            pose=target_pose,
            marker_x=self.marker_x,
            marker_y=self.marker_y,
            marker_yaw=self.marker_yaw,
            half_length=half_length,
            half_width=half_width,
        )

        def sample_shoe_pose():
            return rand_pose(
                xlim=[-0.25, 0.25],
                ylim=[-0.1, 0.05],
                ylim_prop=True,
                rotate_rand=True,
                rotate_lim=[0, 3.14, 0],
                qpos=[0.707, 0.707, 0, 0],
            )

        shoe_pose = sample_shoe_pose()
        target_xy = self.target_block.get_pose().p[:2]
        too_close_to_origin = np.sum(shoe_pose.get_p()[:2] ** 2) < 0.0225
        too_close_to_target = np.sum((shoe_pose.get_p()[:2] - target_xy) ** 2) < 0.0225
        while too_close_to_origin or too_close_to_target:
            shoe_pose = sample_shoe_pose()
            too_close_to_origin = np.sum(shoe_pose.get_p()[:2] ** 2) < 0.0225
            too_close_to_target = (
                np.sum((shoe_pose.get_p()[:2] - target_xy) ** 2) < 0.0225
            )

        self.shoe_modelname = resolve_shoe_modelname(config)
        self.shoe_id = int(np.random.choice(resolve_shoe_id_candidates(config)))
        shoe_half_extents = loaded_shoe_footprint(self.shoe_modelname, self.shoe_id)
        ramp_half_extents = np.asarray([half_length, half_width], dtype=np.float64)
        for _ in range(1000):
            overlap = shoe_overlaps_ramp_footprint(
                shoe_pose,
                shoe_half_extents,
                target_xy,
                target_yaw,
                ramp_half_extents,
            )
            too_close_to_origin = np.sum(shoe_pose.get_p()[:2] ** 2) < 0.0225
            if not overlap and not too_close_to_origin:
                break
            shoe_pose = sample_shoe_pose()
        else:
            raise RuntimeError("could not sample a collision-free shoe pose in 1000 attempts")
        self.shoe = create_actor(
            scene=self,
            pose=shoe_pose,
            modelname=self.shoe_modelname,
            convex=True,
            model_id=self.shoe_id,
        )
        if self.shoe is None:
            raise FileNotFoundError(
                f"failed to create shoe asset {self.shoe_modelname}/base{self.shoe_id}"
            )
        self.initial_shoe_z = float(self.shoe.get_pose().p[2])
        self.shoe_arm_name = "left" if float(self.shoe.get_pose().p[0]) < 0.0 else "right"
        self.prohibited_area.append([-0.2, -0.15, 0.2, -0.01])
        self.add_prohibit_area(self.shoe, padding=0.1)

    def _relation_phase(self):
        gripper_closed = (
            self.is_left_gripper_close()
            if self.shoe_arm_name == "left"
            else self.is_right_gripper_close()
        )
        lifted = float(self.shoe.get_pose().p[2]) - float(self.initial_shoe_z) >= 0.03
        if bool(gripper_closed) and lifted:
            self._placement_phase_started = True
        return float(self._placement_phase_started)

    def _grasp_and_lift(self):
        arm_tag = super()._grasp_and_lift()
        self._placement_phase_started = True
        return arm_tag

    def get_obs(self):
        observation = super().get_obs()
        observation["task_state"].update(
            {
                "anchor_geometry_params": np.asarray(
                    [self.marker_x, self.marker_y, self.marker_yaw], dtype=np.float32
                ),
            }
        )
        return observation

    def play_once(self):
        arm_tag = self._grasp_and_lift()
        target_pose = self.target_block.get_functional_point(0)
        self.move(
            self.place_actor(
                self.shoe,
                arm_tag=arm_tag,
                target_pose=target_pose,
                functional_point_id=0,
                pre_dis=0.12,
                constrain="align",
            )
        )
        self.move(self.open_gripper(arm_tag=arm_tag))
        self._settle_for_success_hold()
        self.info["info"] = {
            "{A}": "shoe",
            "{B}": "geometry marker ramp",
            "{a}": str(arm_tag),
        }
        return self.info

    def recover_policy_placement_phase(self, arm_tag: ArmTag):
        """Plan a placement recovery from the policy-induced current state."""
        self._placement_phase_started = True
        target_pose = self.target_block.get_functional_point(0)
        moved = self.move(
            self.place_actor(
                self.shoe,
                arm_tag=arm_tag,
                target_pose=target_pose,
                functional_point_id=0,
                pre_dis=0.12,
                constrain="align",
            )
        )
        if not moved or not self.plan_success:
            return False
        self.move(self.open_gripper(arm_tag=arm_tag))
        return bool(self._settle_for_success_hold())

    def _settle_for_success_hold(self):
        """Advance expert execution long enough to evaluate stable placement."""

        required = max(1, int(self.placement_success_config.get("stable_steps", 1)))
        # Expert motion primitives do not call check_success at every physics
        # step, while policy take_action does.  Make the expert admission check
        # use the same physical hold criterion instead of counting one call.
        for _ in range(required):
            self.scene.step()
            self._update_render()
            if self.check_success():
                return True
        return False

    def check_success(self):
        self._ensure_evaluation_trackers()
        current_policy_step = int(getattr(self, "take_action_cnt", 0))
        shoe_pose = self.shoe.get_functional_point(0, "pose")
        target_pose = self.target_block.get_functional_point(0, "pose")
        position_tolerance = np.asarray((0.02, 0.02, 0.025), dtype=np.float64)
        min_quaternion_alignment = 0.995
        pose_errors = functional_pose_alignment_errors(
            shoe_pose.p,
            shoe_pose.q,
            target_pose.p,
            target_pose.q,
        )
        translation_norm = float(pose_errors["translation_error_norm_m"])
        rotation_error_deg = float(pose_errors["rotation_error_deg"])
        self._min_translation_error_norm_m = min(
            self._min_translation_error_norm_m, translation_norm
        )
        self._min_rotation_error_deg = min(
            self._min_rotation_error_deg, rotation_error_deg
        )
        translation_ratio = float(
            np.max(
                np.asarray(pose_errors["translation_error_abs_xyz_m"], dtype=np.float64)
                / position_tolerance
            )
        )
        rotation_tolerance_deg = float(
            np.degrees(2.0 * np.arccos(min_quaternion_alignment))
        )
        combined_score = max(
            translation_ratio,
            rotation_error_deg / rotation_tolerance_deg,
        )
        if combined_score < self._best_combined_alignment_score:
            self._best_combined_alignment_score = combined_score
            self._best_combined_translation_error_norm_m = translation_norm
            self._best_combined_rotation_error_deg = rotation_error_deg
        gripper_open = (
            self.is_left_gripper_open()
            if self.shoe_arm_name == "left"
            else self.is_right_gripper_open()
        )
        aligned = functional_pose_alignment_success(
            shoe_pose.p,
            shoe_pose.q,
            target_pose.p,
            target_pose.q,
            position_tolerance=position_tolerance,
            min_quaternion_alignment=min_quaternion_alignment,
        )
        if aligned:
            self._pose_alignment_step_count += 1
            if self._first_pose_alignment_step is None:
                self._first_pose_alignment_step = current_policy_step
        config = self.placement_success_config
        contact = bool(
            self.check_actors_contact(self.shoe.get_name(), self.target_block.get_name())
        )
        linear_speed, angular_speed = self._shoe_velocity_norms()
        require_contact = bool(config.get("require_contact", False))
        max_linear_speed = float(config.get("max_linear_speed_mps", float("inf")))
        max_angular_speed = float(config.get("max_angular_speed_radps", float("inf")))
        raw_success = bool(
            gripper_open
            and aligned
            and (contact or not require_contact)
            and linear_speed <= max_linear_speed
            and angular_speed <= max_angular_speed
        )
        if raw_success:
            self._raw_success_step_count += 1
            if self._first_raw_success_step is None:
                self._first_raw_success_step = current_policy_step
        self._success_hold_count = self._success_hold_count + 1 if raw_success else 0
        self._max_success_hold_count = max(
            self._max_success_hold_count, self._success_hold_count
        )
        stable_steps = max(1, int(config.get("stable_steps", 1)))
        return bool(self._success_hold_count >= stable_steps)

    def _shoe_velocity_norms(self):
        for component in self.shoe.actor.get_components():
            if isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
                return (
                    float(np.linalg.norm(component.get_linear_velocity())),
                    float(np.linalg.norm(component.get_angular_velocity())),
                )
        return float("nan"), float("nan")

    def get_evaluation_metrics(self):
        self._ensure_evaluation_trackers()
        shoe_pose = self.shoe.get_functional_point(0, "pose")
        target_pose = self.target_block.get_functional_point(0, "pose")
        metrics = functional_pose_alignment_errors(
            shoe_pose.p,
            shoe_pose.q,
            target_pose.p,
            target_pose.q,
        )
        linear_speed, angular_speed = self._shoe_velocity_norms()
        metrics.update(
            {
                "shoe_id": int(self.shoe_id),
                "shoe_modelname": str(getattr(self, "shoe_modelname", "041_shoe")),
                "marker_x_m": float(self.marker_x),
                "marker_y_m": float(self.marker_y),
                "marker_yaw_deg": float(np.degrees(self.marker_yaw)),
                "gripper_open": bool(
                    self.is_left_gripper_open()
                    if self.shoe_arm_name == "left"
                    else self.is_right_gripper_open()
                ),
                "shoe_ramp_contact": bool(
                    self.check_actors_contact(self.shoe.get_name(), self.target_block.get_name())
                ),
                "shoe_linear_speed_mps": linear_speed,
                "shoe_angular_speed_radps": angular_speed,
                "success_hold_count": int(self._success_hold_count),
                "max_success_hold_count": int(self._max_success_hold_count),
                "pose_alignment_step_count": int(self._pose_alignment_step_count),
                "raw_success_step_count": int(self._raw_success_step_count),
                "first_pose_alignment_step": self._first_pose_alignment_step,
                "first_raw_success_step": self._first_raw_success_step,
                "min_translation_error_norm_m": float(
                    self._min_translation_error_norm_m
                ),
                "min_rotation_error_deg": float(self._min_rotation_error_deg),
                "best_combined_alignment_score": float(
                    self._best_combined_alignment_score
                ),
                "best_combined_translation_error_norm_m": float(
                    self._best_combined_translation_error_norm_m
                ),
                "best_combined_rotation_error_deg": float(
                    self._best_combined_rotation_error_deg
                ),
                "success_required_hold_steps": max(
                    1, int(self.placement_success_config.get("stable_steps", 1))
                ),
                "policy_steps": int(self.take_action_cnt),
            }
        )
        return metrics
