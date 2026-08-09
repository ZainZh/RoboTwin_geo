"""Short-horizon, handle-aware mug alignment task.

The task deliberately makes the mug's in-plane orientation observable and
necessary: a visible T marker specifies where the mug handle must point.  The
source functional frame is defined by two category-stable local axes rather
than by the simulator object's arbitrary raw frame.
"""

from __future__ import annotations

import numpy as np
import sapien
import transforms3d as t3d

from .handled_mug_geometry import (
    HANDLED_MUG_IDS,
    mug_functional_matrix,
    handle_marker_functional_matrix,
)
from .placement_metrics import functional_pose_alignment_errors
from .place_container_plate import PLACEMENT_CLEARANCE_M, place_container_plate
from .place_shoe_geometry_marker import create_geometry_marker_ramp
from .utils import ArmTag, create_actor


class place_handled_mug_geometry_marker(place_container_plate):
    """Align a held mug to a visible position-and-handle target."""

    def setup_demo(self, **kwags):
        self.handle_marker_config = dict(kwags.get("handle_marker", {}))
        super().setup_demo(**kwags)

    def get_object_pointcloud_targets(self):
        return {"{A}": "container", "{B}": "target_block"}

    def _allowed_container_ids(self) -> list[int]:
        values = [
            int(value)
            for value in self.container_geometry_config.get(
                "allowed_container_ids", HANDLED_MUG_IDS
            )
        ]
        if not values or not set(values).issubset(HANDLED_MUG_IDS):
            raise ValueError(
                "place_handled_mug_geometry_marker requires 039_mug IDs 0-9"
            )
        return values

    @staticmethod
    def _upright_pose_with_world_yaw(position, yaw: float) -> sapien.Pose:
        base = np.asarray([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
        world_yaw = t3d.euler.euler2quat(0.0, 0.0, float(yaw))
        return sapien.Pose(position, t3d.quaternions.qmult(world_yaw, base))

    def load_actors(self):
        allowed_ids = self._allowed_container_ids()
        self.actor_name = "039_mug"
        self.container_id = int(np.random.choice(allowed_ids))

        fixed_arm = str(self.handle_marker_config.get("fixed_arm", "left"))
        if fixed_arm not in {"left", "right", "random"}:
            raise ValueError("handle_marker.fixed_arm must be left, right, or random")
        side = (
            float(np.random.choice([-1.0, 1.0]))
            if fixed_arm == "random"
            else (-1.0 if fixed_arm == "left" else 1.0)
        )
        cup_x = float(np.random.uniform(0.21, 0.28) * side)
        cup_y = float(np.random.uniform(-0.08, 0.04))
        initial_yaw_limit = np.deg2rad(
            float(self.handle_marker_config.get("initial_yaw_limit_deg", 90.0))
        )
        cup_yaw = float(np.random.uniform(-initial_yaw_limit, initial_yaw_limit))
        container_pose = self._upright_pose_with_world_yaw(
            [cup_x, cup_y, 0.741], cup_yaw
        )
        self.container = create_actor(
            self,
            pose=container_pose,
            modelname=self.actor_name,
            model_id=self.container_id,
            convex=True,
        )
        if self.container is None:
            raise FileNotFoundError(
                f"failed to create handled mug 039_mug/base{self.container_id}"
            )
        self.container.config["functional_matrix"] = [
            mug_functional_matrix().tolist()
        ]

        config = self.handle_marker_config
        self.marker_x = float(
            np.random.uniform(
                -float(config.get("marker_x_limit", 0.035)),
                float(config.get("marker_x_limit", 0.035)),
            )
        )
        self.marker_y = float(
            np.random.uniform(
                -float(config.get("marker_y_limit", 0.025)),
                float(config.get("marker_y_limit", 0.025)),
            )
        )
        yaw_limit = np.deg2rad(float(config.get("marker_yaw_limit_deg", 180.0)))
        self.marker_yaw = float(np.random.uniform(-yaw_limit, yaw_limit))
        half_length = float(config.get("support_half_length", 0.12))
        half_width = float(config.get("support_half_width", 0.10))
        self.target_block = create_geometry_marker_ramp(
            self,
            pose=sapien.Pose([0.0, -0.12, 0.741]),
            marker_x=self.marker_x,
            marker_y=self.marker_y,
            marker_yaw=self.marker_yaw,
            half_length=half_length,
            half_width=half_width,
            name="handled_mug_geometry_marker",
        )
        self.target_block.config["functional_matrix"] = [
            handle_marker_functional_matrix(
                self.marker_x, self.marker_y, self.marker_yaw
            ).tolist()
        ]

        # Compatibility aliases used by the short-horizon collector/runtime.
        self.plate = self.target_block
        self.plate_id = 0
        self.shoe = self.container
        self.shoe_id = int(self.container_id)
        self.shoe_arm_name = "right" if cup_x > 0.0 else "left"
        self.initial_container_z = float(self.container.get_pose().p[2])
        self.initial_cup_yaw = cup_yaw
        self.add_prohibit_area(self.container, padding=0.08)
        self.add_prohibit_area(self.target_block, padding=0.08)

    def prepare_policy_placement_phase(self):
        arm_tag = ArmTag(self.shoe_arm_name)
        # The cup assets expose two side contacts.  Pick the reachable side;
        # unlike the legacy task, never request the nonexistent contact ID 2.
        contact_id = 0 if arm_tag == "right" else 1
        self.move(
            self.grasp_actor(
                self.container,
                arm_tag=arm_tag,
                contact_point_id=contact_id,
                pre_grasp_dis=0.1,
            )
        )
        self.move(self.move_by_displacement(arm_tag, z=0.1, move_axis="arm"))
        return arm_tag

    def get_pose_correction_spec(self):
        return {
            "target_pose": self.target_block.get_functional_point(0),
            "functional_point_id": 0,
            "pre_dis": 0.12,
            "final_dis": PLACEMENT_CLEARANCE_M,
            "pre_dis_axis": "fp",
            "constrain": "align",
        }

    def _placement_target_matrix(self):
        target = np.asarray(
            self.target_block.get_functional_point(0, "matrix"), dtype=np.float64
        ).copy()
        target[:3, 3] -= PLACEMENT_CLEARANCE_M * target[:3, 2]
        return target

    def pose_correction_alignment(self):
        source = self.container.get_functional_point(0, "pose")
        target_matrix = self._placement_target_matrix()
        target = sapien.Pose(
            target_matrix[:3, 3],
            t3d.quaternions.mat2quat(target_matrix[:3, :3]),
        )
        errors = functional_pose_alignment_errors(
            source.p, source.q, target.p, target.q
        )
        return {
            "translation_m": float(errors["translation_error_norm_m"]),
            "rotation_deg": float(errors["rotation_error_deg"]),
        }

    def get_obs(self):
        observation = super().get_obs()
        source = np.asarray(
            self.container.get_functional_point(0, "matrix"), dtype=np.float32
        )
        target = np.asarray(self._placement_target_matrix(), dtype=np.float32)
        observation["task_state"].update(
            {
                "source_handle_axis3_label_only": source[:3, 0].copy(),
                # The placement frame points its third axis downward so that
                # the legacy approach-distance convention moves above the
                # support.  Expose the semantically natural upward body axis
                # to representation supervision by negating that column.
                "source_vertical_axis3_label_only": -source[:3, 2].copy(),
                "target_handle_axis3_label_only": target[:3, 0].copy(),
                "target_vertical_axis3_label_only": -target[:3, 2].copy(),
                "anchor_geometry_params": np.asarray(
                    [self.marker_x, self.marker_y, self.marker_yaw], dtype=np.float32
                ),
            }
        )
        return observation

    def check_success(self):
        alignment = self.pose_correction_alignment()
        return bool(
            alignment["translation_m"] <= 0.025
            and alignment["rotation_deg"] <= 15.0
            and self.is_left_gripper_open()
            and self.is_right_gripper_open()
        )

    def play_once(self):
        arm_tag = self.prepare_policy_placement_phase()
        self.move(
            self.place_actor(
                self.container,
                target_pose=self.target_block.get_functional_point(0),
                arm_tag=arm_tag,
                functional_point_id=0,
                pre_dis=0.12,
                dis=PLACEMENT_CLEARANCE_M,
                pre_dis_axis="fp",
                constrain="align",
            )
        )
        self.info["info"] = {
            "{A}": f"039_mug/base{self.container_id}",
            "{B}": "visible handle-direction marker",
            "{a}": str(arm_tag),
        }
        return self.info
