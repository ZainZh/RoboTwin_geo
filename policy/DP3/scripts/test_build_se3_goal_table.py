import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_se3_goal_table_from_validation import build_goal_table  # noqa: E402


class TestBuildSe3GoalTable(unittest.TestCase):
    def test_selects_lowest_energy_trial_per_shoe_and_route(self):
        identity = np.eye(4).tolist()
        shifted = np.eye(4)
        shifted[0, 3] = 0.1
        validation = {
            "trials": [
                {"query_shoe_id": 2, "direction_weight": 0.0, "total_energy": 2.0,
                 "predicted_transform_a_from_b": shifted.tolist()},
                {"query_shoe_id": 2, "direction_weight": 0.0, "total_energy": 1.0,
                 "predicted_transform_a_from_b": identity},
                {"query_shoe_id": 2, "direction_weight": 5.0, "total_energy": 0.5,
                 "predicted_transform_a_from_b": shifted.tolist()},
            ]
        }
        table = build_goal_table(validation, no_direction_weight=0.0, direction_weight=5.0)
        no_direction = table["routes"]["ndf_no_direction"]["goals"]["2"]
        directional = table["routes"]["ndf_direction"]["goals"]["2"]
        np.testing.assert_allclose(no_direction["goal_T_A_from_B"], np.eye(4))
        np.testing.assert_allclose(directional["goal_T_A_from_B"], shifted)
        self.assertEqual(no_direction["solver_energy"], 1.0)
        self.assertEqual(no_direction["confidence"], 1.0)

    def test_missing_transform_is_rejected(self):
        validation = {
            "trials": [
                {"query_shoe_id": 0, "direction_weight": 0.0, "total_energy": 1.0}
            ]
        }
        with self.assertRaisesRegex(ValueError, "predicted_transform"):
            build_goal_table(validation, no_direction_weight=0.0, direction_weight=5.0)


if __name__ == "__main__":
    unittest.main()
