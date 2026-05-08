from ._base_task import Base_Task
from .utils import *
import sapien


class adjust_kettle(Base_Task):

    KETTLE_QPOS = [-0.6018, -0.6026, -0.3631, -0.3779]
    TARGET_BLOCK_HALF_SIZE = (0.065, 0.06, 0.05)
    TARGET_BLOCK_XY = (-0.08, -0.12)
    XY_SUCCESS_EPS = 0.06
    Z_SUCCESS_EPS = 0.03
    INITIAL_TARGET_CLEARANCE = 0.20

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.qpose_tag = 0  # 只用左臂
        xlims = [[-0.40, -0.10], [0.10, 0.40]]
        ylims = [[-0.18, 0.18], [-0.18, 0.18]]
        self.seed_diagnostics = {}

        self.model_id = 13
        self.target_block = create_box(
            scene=self,
            pose=sapien.Pose(
                [self.TARGET_BLOCK_XY[0], self.TARGET_BLOCK_XY[1], 0.741 + self.TARGET_BLOCK_HALF_SIZE[2]],
                [1, 0, 0, 0],
            ),
            half_size=self.TARGET_BLOCK_HALF_SIZE,
            color=(1, 0, 0),
            is_static=True,
            name="target_block",
        )
        self.add_prohibit_area(self.target_block, padding=0.12)

        self.kettle = rand_create_actor(
            self,
            xlim=xlims[self.qpose_tag],
            ylim=ylims[self.qpose_tag],
            zlim=[0.78],
            rotate_rand=True,
            qpos=self.KETTLE_QPOS,
            modelname="092_teapot",
            convex=True,
            rotate_lim=(0, 1.0, 0),
            model_id=self.model_id,
        )
        self.place_functional_point_id = self._append_kettle_bottom_functional_point()
        self.delay(4)
        self._validate_initial_seed()
        self.add_prohibit_area(self.kettle, padding=0.15)

        self.target_block_top_z = self.target_block.get_pose().p[2] + self.TARGET_BLOCK_HALF_SIZE[2]
        target_q = self.kettle.get_functional_point(self.place_functional_point_id)[3:]
        self.target_pose = [
            self.TARGET_BLOCK_XY[0],
            self.TARGET_BLOCK_XY[1],
            self.target_block_top_z,
            *target_q,
        ]

    def _append_kettle_bottom_functional_point(self):
        config = self.kettle.config
        center = np.array(config["center"], dtype=np.float64)
        extents = np.array(config["extents"], dtype=np.float64)
        bottom_center = center.copy()
        bottom_center[1] -= extents[1] / 2.0

        bottom_matrix = np.array(config["functional_matrix"][0], dtype=np.float64)
        bottom_matrix[:3, 3] = bottom_center
        config["functional_matrix"].append(bottom_matrix.tolist())
        return len(config["functional_matrix"]) - 1

    def _validate_initial_seed(self):
        target_xy = np.array(self.target_block.get_pose().p[:2])
        actor_xy = np.array(self.kettle.get_pose().p[:2])
        functional_xy = np.array(self.kettle.get_functional_point(0)[:2])
        place_point_xy = np.array(self.kettle.get_functional_point(self.place_functional_point_id)[:2])
        distances = [
            np.linalg.norm(actor_xy - target_xy),
            np.linalg.norm(functional_xy - target_xy),
            np.linalg.norm(place_point_xy - target_xy),
        ]
        contact_block = self.check_actors_contact(self.kettle.get_name(), self.target_block.get_name())
        self.seed_diagnostics = {
            "initial_actor_xy": actor_xy.round(4).tolist(),
            "initial_functional_xy": functional_xy.round(4).tolist(),
            "initial_place_point_xy": place_point_xy.round(4).tolist(),
            "target_xy": target_xy.round(4).tolist(),
            "initial_min_target_distance": round(float(min(distances)), 4),
            "initial_contact_block": contact_block,
        }
        if contact_block or min(distances) < self.INITIAL_TARGET_CLEARANCE:
            raise UnStableError(
                "invalid adjust_kettle seed: teapot starts too close to or touching the red target block; "
                f"diagnostics={self.seed_diagnostics}")

    def play_once(self):
        arm_tag = ArmTag("right" if self.qpose_tag == 1 else "left")

        self.move(self.grasp_actor(self.kettle, arm_tag=arm_tag, pre_grasp_dis=0.1))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.12))
        self.move(
            self.place_actor(
                self.kettle,
                target_pose=self.target_pose,
                arm_tag=arm_tag,
                functional_point_id=self.place_functional_point_id,
                pre_dis=0.12,
                dis=0.0,
                pre_dis_axis=[0.0, 0.0, 1.0],
                constrain="free",
                is_open=False,
            ))
        self.delay(2)
        self.move(self.open_gripper(arm_tag=arm_tag))
        self.delay(2)

        self.info["info"] = {
            "{A}": f"092_teapot/base{self.model_id}",
            "{B}": "red block",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        kettle_pose = np.array(self.kettle.get_pose().p[:3])
        place_point_pose = np.array(self.kettle.get_functional_point(self.place_functional_point_id)[:3])
        target_xy = np.array(self.target_block.get_pose().p[:2])
        xy_ok = np.linalg.norm(place_point_pose[:2] - target_xy) < self.XY_SUCCESS_EPS
        z_ok = abs(place_point_pose[2] - self.target_pose[2]) < self.Z_SUCCESS_EPS
        contact_block = self.check_actors_contact(self.kettle.get_name(), self.target_block.get_name())
        contact_table = self.check_actors_contact(self.kettle.get_name(), "table")
        if not hasattr(self, "seed_diagnostics"):
            self.seed_diagnostics = {}
        self.seed_diagnostics.update({
            "final_actor_xyz": kettle_pose.round(4).tolist(),
            "final_place_point_xyz": place_point_pose.round(4).tolist(),
            "target_xyz": np.array(self.target_pose[:3]).round(4).tolist(),
            "final_xy_error": round(float(np.linalg.norm(place_point_pose[:2] - target_xy)), 4),
            "final_z_error": round(float(abs(place_point_pose[2] - self.target_pose[2])), 4),
            "final_contact_block": contact_block,
            "final_contact_table": contact_table,
            "left_gripper_open": self.is_left_gripper_open(),
            "right_gripper_open": self.is_right_gripper_open(),
            "plan_success": self.plan_success,
        })
        return (xy_ok and z_ok and contact_block and not contact_table and self.is_left_gripper_open()
                and self.is_right_gripper_open())

    def get_seed_diagnostics(self):
        seed_diagnostics = getattr(self, "seed_diagnostics", {})
        if not seed_diagnostics:
            return ""
        return "adjust_kettle diagnostics: " + ", ".join(
            f"{key}={value}" for key, value in seed_diagnostics.items())
