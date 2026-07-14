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

    def test_target_block_pose_is_randomized(self):
        source = TASK_PATH.read_text(encoding="utf-8")
        self.assertIn("target_pose = rand_pose", source)
        self.assertIn("target_yaw", source)
        self.assertIn("euler2quat", source)

    def test_success_uses_target_block_functional_pose_not_fixed_pose(self):
        source = TASK_PATH.read_text(encoding="utf-8")
        self.assertIn("target_pose = self.target_block.get_functional_point(0)", source)
        self.assertNotIn("target_pose_p = np.array([0, -0.08])", source)
        self.assertNotIn("target_pose_q = np.array([0.5, 0.5, -0.5, -0.5])", source)

    def test_object_pointcloud_targets_include_shoe_and_rotating_block(self):
        source = TARGETS_PATH.read_text(encoding="utf-8")
        self.assertIn('"place_shoe_rotating_block"', source)
        self.assertIn('"{A}": "shoe"', source)
        self.assertIn('"{B}": "target_block"', source)

    def test_task_instruction_json_exists(self):
        data = json.loads(INSTRUCTION_PATH.read_text(encoding="utf-8"))
        self.assertTrue(data)
        self.assertTrue(any("shoe" in item.lower() for values in data.values() for item in values))


if __name__ == "__main__":
    unittest.main()
