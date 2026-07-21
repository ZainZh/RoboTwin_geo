import json
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_PATH = REPO_ROOT / "envs" / "place_shoe_rotating_block.py"
TARGETS_PATH = REPO_ROOT / "envs" / "object_pointcloud_targets.py"
INSTRUCTION_PATH = REPO_ROOT / "description" / "task_instruction" / "place_shoe_rotating_block.json"


class TestPlaceShoeRotatingBlockTaskIntegration(unittest.TestCase):
    def test_task_source_exists_with_matching_class(self):
        source = TASK_PATH.read_text(encoding="utf-8")
        self.assertIn("class place_shoe_rotating_block", source)
        self.assertIn("class place_shoe(Base_Task)", (REPO_ROOT / "envs" / "place_shoe.py").read_text(encoding="utf-8"))

    def test_target_block_yaw_is_randomized_but_position_stays_expert_feasible(self):
        source = TASK_PATH.read_text(encoding="utf-8")
        self.assertIn("target_yaw", source)
        self.assertIn("target_center_z", source)
        self.assertIn("[0, -0.08, target_center_z]", source)
        self.assertNotIn("target_pose = rand_pose", source)


    def test_shoe_spawn_preserves_original_center_clearance_and_target_clearance(self):
        source = TASK_PATH.read_text(encoding="utf-8")
        self.assertIn("too_close_to_origin", source)
        self.assertIn("too_close_to_target", source)
        self.assertIn("target_block.get_pose().p[:2]", source)


    def test_target_block_is_a_yaw_randomized_ramp_not_flat_symmetric_block(self):
        source = TASK_PATH.read_text(encoding="utf-8")
        self.assertIn("ramp_pitch", source)
        self.assertIn("yaw_mat @ pitch_mat", source)
        self.assertIn("target_quat", source)
        self.assertNotIn("t3d.euler.euler2quat(0, 0, target_yaw)", source)

    def test_success_uses_target_block_functional_pose_not_fixed_pose(self):
        source = TASK_PATH.read_text(encoding="utf-8")
        self.assertIn("target_pose = self.target_block.get_functional_point(0)", source)
        self.assertNotIn("target_pose_p = np.array([0, -0.08])", source)
        self.assertNotIn("target_pose_q = np.array([0.5, 0.5, -0.5, -0.5])", source)


    def test_success_uses_functional_point_pose_and_quaternion_dot_similarity(self):
        source = TASK_PATH.read_text(encoding="utf-8")
        self.assertIn('self.shoe.get_functional_point(0, "pose")', source)
        self.assertIn("functional_pose_alignment_success", source)

    def test_success_measures_full_pose_alignment_without_requiring_release(self):
        source = TASK_PATH.read_text(encoding="utf-8")
        self.assertIn("position_tolerance=(0.05, 0.03, 0.04)", source)
        self.assertNotIn("and self.is_left_gripper_open()", source)
        self.assertNotIn("and self.is_right_gripper_open()", source)

    def test_object_pointcloud_targets_include_shoe_and_rotating_block(self):
        source = TARGETS_PATH.read_text(encoding="utf-8")
        self.assertIn('"place_shoe_rotating_block"', source)
        self.assertIn('"{A}": "shoe"', source)
        self.assertIn('"{B}": "target_block"', source)


    def test_episode_info_uses_plain_language_names_not_asset_paths(self):
        source = TASK_PATH.read_text(encoding="utf-8")
        self.assertIn('"{A}": "shoe"', source)
        self.assertIn('"{B}": "target block"', source)
        self.assertNotIn('"{A}": f"041_shoe/base{self.shoe_id}"', source)

    def test_task_records_pose_and_phase_state_for_se3_relation_comparison(self):
        source = TASK_PATH.read_text(encoding="utf-8")
        self.assertIn('pkl_dic["task_state"]', source)
        self.assertIn('"object_pose_A"', source)
        self.assertIn('"object_pose_B"', source)
        self.assertIn('"goal_T_A_from_B_oracle"', source)
        self.assertIn('"shoe_id"', source)
        self.assertIn('"relation_phase"', source)

    def test_task_exposes_expert_grasp_lift_prefix_for_placement_only_eval(self):
        source = TASK_PATH.read_text(encoding="utf-8")
        self.assertIn("def prepare_policy_placement_phase", source)
        self.assertIn("self._grasp_and_lift()", source)

    def test_task_instruction_json_exists(self):
        data = json.loads(INSTRUCTION_PATH.read_text(encoding="utf-8"))
        self.assertTrue(data)
        self.assertTrue(any("shoe" in item.lower() for values in data.values() for item in values))


if __name__ == "__main__":
    unittest.main()
