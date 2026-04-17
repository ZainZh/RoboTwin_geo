from ._base_task import Base_Task
from .utils import *
import numpy as np
import transforms3d as t3d


POUR_HEIGHT_PRE = 0.08
POUR_HEIGHT_FINAL = 0.05
POUR_TILT_DEG = 55.0
MUG_UPRIGHT_DOT_MIN = 0.9
MUG_HEIGHT_EPS = 0.03
SPOUT_XY_EPS = 0.04
SPOUT_Z_RANGE = (0.02, 0.10)
KETTLE_TILT_DOT_MAX = float(np.cos(np.deg2rad(45.0)))


def pose_to_matrix(pose):
    pose = np.asarray(pose, dtype=np.float64)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = t3d.quaternions.quat2mat(pose[3:])
    matrix[:3, 3] = pose[:3]
    return matrix


def normalize_quat(quat):
    quat = np.asarray(quat, dtype=np.float64)
    return (quat / np.linalg.norm(quat)).tolist()


class pour_kettle_mug(Base_Task):

    def setup_demo(self, is_test=False, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.arm_tag = ArmTag("left")

        self.kettle_name = "009_kettle"
        self.kettle_id = int(np.random.randint(0, 3))
        self.kettle = rand_create_sapien_urdf_obj(
            scene=self,
            modelname="009_kettle",
            modelid=self.kettle_id,
            xlim=[-0.26, -0.16],
            ylim=[-0.10, 0.04],
            rotate_rand=True,
            rotate_lim=[0, 0, np.pi / 10],
            qpos=[1, 0, 0, 0],
            fix_root_link=False,
        )
        self.kettle.set_mass(0.05)
        self.kettle.set_properties(0.0, 0.0)

        self.mug_name = "039_mug"
        self.mug_id = int(np.random.randint(0, 13))
        self.mug = rand_create_actor(
            scene=self,
            modelname="039_mug",
            model_id=self.mug_id,
            xlim=[0.02, 0.10],
            ylim=[-0.02, 0.08],
            rotate_rand=True,
            rotate_lim=[0, 0, np.pi / 12],
            qpos=[1, 0, 0, 0],
            convex=True,
            is_static=True,
        )

        self.mug_start_height = float(self.mug.get_pose().p[2])

        self.add_prohibit_area(self.kettle, padding=0.10)
        self.add_prohibit_area(self.mug, padding=0.08)

    def _get_scale_vec(self, scale):
        scale = np.asarray(scale, dtype=np.float64).reshape(-1)
        if scale.size == 1:
            scale = np.repeat(scale, 3)
        return scale

    def _get_mug_opening_center(self):
        mug_bottom = np.asarray(self.mug.get_functional_point(1)[:3], dtype=np.float64)
        extents = np.asarray(self.mug.config.get("extents", [0.0, 0.0, 0.12]), dtype=np.float64)
        scale = self._get_scale_vec(self.mug.config.get("scale", [1.0, 1.0, 1.0]))
        mug_height = max(float(extents[2] * scale[2]), 0.10)
        return mug_bottom + np.array([0.0, 0.0, mug_height * 0.8], dtype=np.float64)

    def _get_spout_in_ee(self):
        ee_matrix = pose_to_matrix(self.get_arm_pose(self.arm_tag))
        spout_matrix = np.asarray(self.kettle.get_functional_point(0, "matrix"), dtype=np.float64)
        return np.linalg.inv(ee_matrix) @ spout_matrix

    def _build_ee_pose_for_spout_target(self, spout_target, ee_quat, spout_in_ee):
        ee_rot = t3d.quaternions.quat2mat(np.asarray(ee_quat, dtype=np.float64))
        ee_pos = np.asarray(spout_target, dtype=np.float64) - ee_rot @ spout_in_ee[:3, 3]
        return ee_pos.tolist() + normalize_quat(ee_quat)

    def _rotate_ee_quat_about_world_axis(self, base_quat, axis_world, angle_deg):
        axis_world = np.asarray(axis_world, dtype=np.float64)
        if np.linalg.norm(axis_world) < 1e-6:
            axis_world = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis_world = axis_world / np.linalg.norm(axis_world)
        delta_quat = t3d.quaternions.axangle2quat(axis_world, np.deg2rad(angle_deg))
        return normalize_quat(t3d.quaternions.qmult(delta_quat, np.asarray(base_quat, dtype=np.float64)))

    def play_once(self):
        self.move(self.grasp_actor(self.kettle, arm_tag=self.arm_tag, pre_grasp_dis=0.08, contact_point_id=0))
        self.move(self.move_by_displacement(arm_tag=self.arm_tag, z=0.12))

        spout_in_ee = self._get_spout_in_ee()
        mug_opening_center = self._get_mug_opening_center()

        pre_pose = self._build_ee_pose_for_spout_target(
            mug_opening_center + np.array([0.0, 0.0, POUR_HEIGHT_PRE], dtype=np.float64),
            self.get_arm_pose(self.arm_tag)[3:],
            spout_in_ee,
        )
        self.move(self.move_to_pose(arm_tag=self.arm_tag, target_pose=pre_pose))

        pre_pose_np = np.asarray(pre_pose, dtype=np.float64)
        spout_offset_world = t3d.quaternions.quat2mat(pre_pose_np[3:]) @ spout_in_ee[:3, 3]
        tilt_axis_world = np.cross(spout_offset_world, np.array([0.0, 0.0, -1.0], dtype=np.float64))
        final_quat = self._rotate_ee_quat_about_world_axis(pre_pose_np[3:], tilt_axis_world, POUR_TILT_DEG)

        final_pose = self._build_ee_pose_for_spout_target(
            mug_opening_center + np.array([0.0, 0.0, POUR_HEIGHT_FINAL], dtype=np.float64),
            final_quat,
            spout_in_ee,
        )
        self.move(self.move_to_pose(arm_tag=self.arm_tag, target_pose=final_pose))

        self.info["info"] = {
            "{A}": f"{self.kettle_name}/base{self.kettle_id}",
            "{B}": f"{self.mug_name}/base{self.mug_id}",
            "{a}": str(self.arm_tag),
        }
        return self.info

    def check_success(self):
        if not self.plan_success:
            return False

        mug_pose_matrix = self.mug.get_pose().to_transformation_matrix()
        mug_up = mug_pose_matrix[:3, 2]
        mug_is_upright = float(np.dot(mug_up, np.array([0.0, 0.0, 1.0], dtype=np.float64))) > MUG_UPRIGHT_DOT_MIN
        mug_height_ok = abs(float(self.mug.get_pose().p[2]) - self.mug_start_height) < MUG_HEIGHT_EPS

        mug_opening_center = self._get_mug_opening_center()
        spout = np.asarray(self.kettle.get_functional_point(0)[:3], dtype=np.float64)
        spout_xy_error = float(np.linalg.norm((spout - mug_opening_center)[:2]))
        spout_z_offset = float(spout[2] - mug_opening_center[2])

        kettle_pose_matrix = self.kettle.get_pose().to_transformation_matrix()
        kettle_up = kettle_pose_matrix[:3, 2]
        kettle_is_tilted = float(np.dot(kettle_up, np.array([0.0, 0.0, 1.0], dtype=np.float64))) < KETTLE_TILT_DOT_MAX

        return (
            mug_is_upright
            and mug_height_ok
            and spout_xy_error < SPOUT_XY_EPS
            and SPOUT_Z_RANGE[0] < spout_z_offset < SPOUT_Z_RANGE[1]
            and kettle_is_tilted
            and (not self.is_left_gripper_open())
        )
