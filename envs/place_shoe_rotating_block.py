from ._base_task import Base_Task
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

        self.prohibited_area.append([-0.2, -0.15, 0.2, -0.01])
        self.add_prohibit_area(self.shoe, padding=0.1)

    def play_once(self):
        shoe_pose = self.shoe.get_pose().p
        arm_tag = ArmTag("left" if shoe_pose[0] < 0 else "right")

        self.move(self.grasp_actor(self.shoe, arm_tag=arm_tag, pre_grasp_dis=0.1, gripper_pos=0))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07))

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
            "{A}": f"041_shoe/base{self.shoe_id}",
            "{B}": "target block",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        shoe_pose = self.shoe.get_functional_point(0, "pose")
        target_pose = self.target_block.get_functional_point(0, "pose")
        shoe_pose_p = np.array(shoe_pose.p)
        target_pose_p = np.array(target_pose.p)
        shoe_pose_q = np.array(shoe_pose.q, dtype=np.float64)
        target_pose_q = np.array(target_pose.q, dtype=np.float64)
        shoe_pose_q /= max(np.linalg.norm(shoe_pose_q), 1e-8)
        target_pose_q /= max(np.linalg.norm(target_pose_q), 1e-8)
        quat_alignment = abs(float(np.dot(shoe_pose_q, target_pose_q)))

        return (np.all(abs(shoe_pose_p[:2] - target_pose_p[:2]) < np.array([0.05, 0.03]))
                and quat_alignment > 0.98
                and self.is_left_gripper_open()
                and self.is_right_gripper_open())
