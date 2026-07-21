from ._base_task import Base_Task
from .placement_metrics import functional_pose_alignment_success
from .utils import *
import sapien
import numpy as np
import transforms3d as t3d


class place_shoe_rotating_block(Base_Task):

    def setup_demo(self, is_test=False, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        target_yaw = np.random.uniform(-np.pi, np.pi)
        ramp_pitch = np.deg2rad(10.0)
        ramp_half_length = 0.13
        target_center_z = 0.74 + ramp_half_length * np.sin(abs(ramp_pitch))
        yaw_mat = t3d.euler.euler2mat(0, 0, target_yaw)
        pitch_mat = t3d.euler.euler2mat(0, ramp_pitch, 0)
        target_quat = t3d.quaternions.mat2quat(yaw_mat @ pitch_mat)
        target_pose = sapien.Pose([0, -0.08, target_center_z], target_quat)
        self.target_block = create_box(
            scene=self,
            pose=target_pose,
            half_size=(0.13, 0.05, 0.0005),
            color=(0, 0, 1),
            is_static=True,
            name="box",
        )
        self.target_block.config["functional_matrix"] = [[
            [0.0, -1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0],
            [0.0, 0.0, 0.0, 1.0],
        ], [
            [0.0, -1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0],
            [0.0, 0.0, 0.0, 1.0],
        ]]

        def sample_shoe_pose():
            return rand_pose(
                xlim=[-0.25, 0.25],
                ylim=[-0.1, 0.05],
                ylim_prop=True,
                rotate_rand=True,
                rotate_lim=[0, 3.14, 0],
                qpos=[0.707, 0.707, 0, 0],
            )

        shoes_pose = sample_shoe_pose()
        target_xy = self.target_block.get_pose().p[:2]
        too_close_to_origin = np.sum((shoes_pose.get_p()[:2] - np.zeros(2)) ** 2) < 0.0225
        too_close_to_target = np.sum((shoes_pose.get_p()[:2] - target_xy) ** 2) < 0.0225
        while too_close_to_origin or too_close_to_target:
            shoes_pose = sample_shoe_pose()
            too_close_to_origin = np.sum((shoes_pose.get_p()[:2] - np.zeros(2)) ** 2) < 0.0225
            too_close_to_target = np.sum((shoes_pose.get_p()[:2] - self.target_block.get_pose().p[:2]) ** 2) < 0.0225
        self.shoe_id = np.random.choice([i for i in range(10)])
        self.shoe = create_actor(
            scene=self,
            pose=shoes_pose,
            modelname="041_shoe",
            convex=True,
            model_id=self.shoe_id,
        )

        self.initial_shoe_z = float(self.shoe.get_pose().p[2])
        self.shoe_arm_name = "left" if float(self.shoe.get_pose().p[0]) < 0.0 else "right"
        self.prohibited_area.append([-0.2, -0.15, 0.2, -0.01])
        self.add_prohibit_area(self.shoe, padding=0.1)

    @staticmethod
    def _pose7(actor):
        pose = actor.get_pose()
        return np.concatenate(
            [
                np.asarray(pose.p, dtype=np.float32),
                np.asarray(pose.q, dtype=np.float32),
            ]
        )

    @staticmethod
    def _scaled_functional_matrix(actor, functional_point_id=0):
        matrix = np.asarray(
            actor.config["functional_matrix"][functional_point_id],
            dtype=np.float64,
        ).copy()
        scale = np.asarray(actor.config.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
        if scale.ndim == 0:
            scale = np.full((3,), float(scale), dtype=np.float64)
        matrix[:3, 3] *= scale.reshape(3)
        return matrix

    def _relation_phase(self):
        gripper_closed = (
            self.is_left_gripper_close()
            if self.shoe_arm_name == "left"
            else self.is_right_gripper_close()
        )
        lifted = float(self.shoe.get_pose().p[2]) - float(self.initial_shoe_z) >= 0.03
        return float(bool(gripper_closed) and lifted)

    def get_obs(self):
        pkl_dic = super().get_obs()
        shoe_functional = self._scaled_functional_matrix(self.shoe)
        ramp_functional = self._scaled_functional_matrix(self.target_block)
        goal_a_from_b = shoe_functional @ np.linalg.inv(ramp_functional)
        pkl_dic["task_state"] = {
            "object_pose_A": self._pose7(self.shoe),
            "object_pose_B": self._pose7(self.target_block),
            "goal_T_A_from_B_oracle": goal_a_from_b.reshape(-1).astype(np.float32),
            "shoe_id": np.asarray([self.shoe_id], dtype=np.int64),
            "relation_phase": np.asarray([self._relation_phase()], dtype=np.float32),
        }
        return pkl_dic

    def _grasp_and_lift(self):
        shoe_pose = self.shoe.get_pose().p
        arm_tag = ArmTag("left" if shoe_pose[0] < 0 else "right")
        self.move(self.grasp_actor(self.shoe, arm_tag=arm_tag, pre_grasp_dis=0.1, gripper_pos=0))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07))
        return arm_tag

    def prepare_policy_placement_phase(self):
        return self._grasp_and_lift()

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
            ))
        self.move(self.open_gripper(arm_tag=arm_tag))

        self.info["info"] = {
            "{A}": "shoe",
            "{B}": "target block",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        shoe_pose = self.shoe.get_functional_point(0, "pose")
        target_pose = self.target_block.get_functional_point(0, "pose")

        # This benchmark evaluates geometric placement alignment. The learned
        # placement-only policy may accurately move the shoe to the ramp while
        # keeping the grasp closed, so gripper release is deliberately not part
        # of this metric. Checking XYZ (rather than the previous XY-only test)
        # prevents a shoe passing above the target from being counted as success.
        return functional_pose_alignment_success(
            shoe_pose.p,
            shoe_pose.q,
            target_pose.p,
            target_pose.q,
            position_tolerance=(0.05, 0.03, 0.04),
            min_quaternion_alignment=0.98,
        )
