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
        target_pose = rand_pose(
            xlim=[-0.18, 0.18],
            ylim=[-0.14, -0.02],
            zlim=[0.74],
            rotate_rand=False,
            qpos=t3d.euler.euler2quat(0, 0, target_yaw),
        )
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

        shoes_pose = rand_pose(
            xlim=[-0.25, 0.25],
            ylim=[-0.1, 0.08],
            ylim_prop=True,
            rotate_rand=True,
            rotate_lim=[0, 3.14, 0],
            qpos=[0.707, 0.707, 0, 0],
        )
        while np.sum((shoes_pose.get_p()[:2] - self.target_block.get_pose().p[:2]) ** 2) < 0.0225:
            shoes_pose = rand_pose(
                xlim=[-0.25, 0.25],
                ylim=[-0.1, 0.08],
                ylim_prop=True,
                rotate_rand=True,
                rotate_lim=[0, 3.14, 0],
                qpos=[0.707, 0.707, 0, 0],
            )
        self.shoe_id = np.random.choice([i for i in range(10)])
        self.shoe = create_actor(
            scene=self,
            pose=shoes_pose,
            modelname="041_shoe",
            convex=True,
            model_id=self.shoe_id,
        )

        target_p = self.target_block.get_pose().p
        self.prohibited_area.append([target_p[0] - 0.18, target_p[1] - 0.09, target_p[0] + 0.18, target_p[1] + 0.09])
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
        shoe_pose_p = np.array(self.shoe.get_pose().p)
        shoe_pose_q = np.array(self.shoe.get_pose().q)
        if shoe_pose_q[0] < 0:
            shoe_pose_q *= -1

        target_pose = self.target_block.get_functional_point(0, "pose")
        target_pose_p = np.array(target_pose.p[:2])
        target_pose_q = np.array(target_pose.q)
        if target_pose_q[0] < 0:
            target_pose_q *= -1

        eps = np.array([0.05, 0.03, 0.08, 0.08, 0.08, 0.08])
        return (np.all(abs(shoe_pose_p[:2] - target_pose_p) < eps[:2])
                and np.all(abs(shoe_pose_q - target_pose_q) < eps[-4:])
                and self.is_left_gripper_open()
                and self.is_right_gripper_open())
