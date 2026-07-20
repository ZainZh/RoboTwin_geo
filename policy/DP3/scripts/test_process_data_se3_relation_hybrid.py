import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from process_data_se3_relation_hybrid import (  # noqa: E402
    extract_task_state_frame,
    relation_token_for_frame,
    should_keep_frame,
)
from object_pointcloud_utils import load_hdf5  # noqa: E402
from se3_relation_token_utils import RELATION_TOKEN_DIM  # noqa: E402


class TestProcessDataSe3RelationHybrid(unittest.TestCase):
    def setUp(self):
        identity_pose = np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        self.episode = {
            "task_state": {
                "object_pose_A": np.stack([identity_pose, identity_pose]),
                "object_pose_B": np.stack([identity_pose, identity_pose]),
                "goal_T_A_from_B_oracle": np.stack([np.eye(4).reshape(-1)] * 2),
                "shoe_id": np.asarray([[2], [2]], dtype=np.int64),
                "relation_phase": np.asarray([[0.0], [1.0]], dtype=np.float32),
            }
        }

    def test_extract_task_state_selects_one_frame_without_dropping_scalar_arrays(self):
        state = extract_task_state_frame(self.episode, 1)
        self.assertEqual(np.asarray(state["object_pose_A"]).shape, (7,))
        self.assertEqual(np.asarray(state["shoe_id"]).shape, (1,))
        self.assertEqual(float(np.asarray(state["relation_phase"])[0]), 1.0)

    def test_oracle_token_is_gated_before_lift_and_valid_after_lift(self):
        before = relation_token_for_frame(
            route="oracle", task_state=extract_task_state_frame(self.episode, 0), goal_table=None
        )
        after = relation_token_for_frame(
            route="oracle", task_state=extract_task_state_frame(self.episode, 1), goal_table=None
        )
        self.assertEqual(before.shape, (RELATION_TOKEN_DIM,))
        np.testing.assert_allclose(before, 0.0)
        self.assertEqual(float(after[-1]), 1.0)

    def test_missing_task_state_has_actionable_error(self):
        with self.assertRaisesRegex(RuntimeError, "recollect"):
            extract_task_state_frame({}, 0)

    def test_placement_only_keeps_only_active_relation_phase(self):
        inactive = extract_task_state_frame(self.episode, 0)
        active = extract_task_state_frame(self.episode, 1)
        self.assertFalse(should_keep_frame(inactive, placement_only=True))
        self.assertTrue(should_keep_frame(active, placement_only=True))
        self.assertTrue(should_keep_frame(inactive, placement_only=False))

    def test_hdf5_loader_reads_collected_task_state_group(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "episode0.hdf5"
            with h5py.File(path, "w") as root:
                root.create_dataset("joint_action/vector", data=np.zeros((2, 14), dtype=np.float32))
                root.create_dataset("pointcloud", data=np.zeros((2, 8, 6), dtype=np.float32))
                task_state = root.create_group("task_state")
                for key, value in self.episode["task_state"].items():
                    task_state.create_dataset(key, data=value)

            loaded = load_hdf5(str(path))

        self.assertIn("task_state", loaded)
        self.assertEqual(set(loaded["task_state"]), set(self.episode["task_state"]))
        np.testing.assert_array_equal(
            loaded["task_state"]["relation_phase"],
            self.episode["task_state"]["relation_phase"],
        )


if __name__ == "__main__":
    unittest.main()
