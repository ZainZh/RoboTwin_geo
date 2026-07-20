import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from se3_relation_token_utils import (  # noqa: E402
    RELATION_TOKEN_DIM,
    build_relation_token_from_task_state,
    build_se3_relation_token,
    functional_goal_a_from_b,
    infer_placement_phase,
    pose7_to_matrix,
    resolve_relation_goal,
)


def pose7(xyz, quat_wxyz=(1.0, 0.0, 0.0, 0.0)):
    return np.asarray([*xyz, *quat_wxyz], dtype=np.float64)


class TestSe3RelationTokenUtils(unittest.TestCase):
    def test_pose7_uses_sapien_wxyz_quaternion_order(self):
        transform = pose7_to_matrix(
            pose7([1.0, 2.0, 3.0], [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
        )
        np.testing.assert_allclose(transform[:3, 3], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(
            transform[:3, :3],
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            atol=1e-7,
        )

    def test_goal_relation_produces_zero_translation_and_identity_rotation(self):
        pose_b = pose7([0.2, -0.1, 0.7])
        goal_a_from_b = np.eye(4, dtype=np.float64)
        goal_a_from_b[0, 3] = -0.15
        goal_b_from_a = np.linalg.inv(goal_a_from_b)
        world_from_a = pose7_to_matrix(pose_b) @ goal_b_from_a

        token = build_se3_relation_token(
            object_pose_a=world_from_a,
            object_pose_b=pose_b,
            goal_a_from_b=goal_a_from_b,
            phase_gate=1.0,
            solver_energy=0.25,
            confidence=1.0,
        )

        self.assertEqual(token.shape, (RELATION_TOKEN_DIM,))
        np.testing.assert_allclose(token[:3], 0.0, atol=1e-7)
        np.testing.assert_allclose(token[3:9], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0], atol=1e-7)
        self.assertAlmostEqual(float(token[9]), 0.25)
        self.assertAlmostEqual(float(token[10]), 1.0)

    def test_phase_gate_suppresses_relation_during_grasp(self):
        goal = np.eye(4, dtype=np.float64)
        token = build_se3_relation_token(
            object_pose_a=pose7([0.3, 0.0, 0.7]),
            object_pose_b=pose7([0.0, 0.0, 0.7]),
            goal_a_from_b=goal,
            phase_gate=0.0,
            solver_energy=4.0,
            confidence=1.0,
        )
        np.testing.assert_allclose(token, np.zeros(RELATION_TOKEN_DIM), atol=1e-7)

    def test_goal_source_changes_only_the_goal_not_current_poses(self):
        oracle = np.eye(4, dtype=np.float64)
        ndf = np.eye(4, dtype=np.float64)
        ndf[1, 3] = 0.04
        task_state = {
            "goal_T_A_from_B_oracle": oracle.reshape(-1),
            "shoe_id": np.asarray([3], dtype=np.int64),
        }
        goal_table = {
            "3": {
                "goal_T_A_from_B": ndf.reshape(-1).tolist(),
                "solver_energy": 0.7,
                "confidence": 0.8,
            }
        }

        oracle_goal, oracle_energy, oracle_confidence = resolve_relation_goal(
            route="oracle", task_state=task_state, goal_table=None
        )
        ndf_goal, ndf_energy, ndf_confidence = resolve_relation_goal(
            route="ndf_direction", task_state=task_state, goal_table=goal_table
        )

        np.testing.assert_allclose(oracle_goal, oracle)
        np.testing.assert_allclose(ndf_goal, ndf)
        self.assertEqual(oracle_energy, 0.0)
        self.assertEqual(oracle_confidence, 1.0)
        self.assertEqual(ndf_energy, 0.7)
        self.assertEqual(ndf_confidence, 0.8)

    def test_missing_ndf_shoe_id_returns_invalid_goal(self):
        task_state = {"shoe_id": np.asarray([9], dtype=np.int64)}
        goal, energy, confidence = resolve_relation_goal(
            route="ndf_no_direction",
            task_state=task_state,
            goal_table={"0": {"goal_T_A_from_B": np.eye(4).reshape(-1).tolist()}},
        )
        self.assertIsNone(goal)
        self.assertTrue(np.isinf(energy))
        self.assertEqual(confidence, 0.0)

    def test_shared_task_state_entrypoint_matches_direct_token_builder(self):
        task_state = {
            "object_pose_A": pose7([0.1, 0.0, 0.0]),
            "object_pose_B": pose7([0.0, 0.0, 0.0]),
            "goal_T_A_from_B_oracle": np.eye(4).reshape(-1),
            "shoe_id": np.asarray([0]),
            "relation_phase": np.asarray([1.0]),
        }
        shared = build_relation_token_from_task_state(
            route="oracle", task_state=task_state, goal_table=None
        )
        direct = build_se3_relation_token(
            object_pose_a=task_state["object_pose_A"],
            object_pose_b=task_state["object_pose_B"],
            goal_a_from_b=np.eye(4),
            phase_gate=1.0,
            solver_energy=0.0,
            confidence=1.0,
        )
        np.testing.assert_allclose(shared, direct)

    def test_functional_frames_define_goal_a_from_b(self):
        functional_a = np.eye(4, dtype=np.float64)
        functional_a[2, 3] = 0.12
        functional_b = np.eye(4, dtype=np.float64)
        functional_b[0, 3] = -0.04
        goal = functional_goal_a_from_b(functional_a, functional_b)
        np.testing.assert_allclose(goal, functional_a @ np.linalg.inv(functional_b))

    def test_placement_phase_requires_both_closed_gripper_and_lift(self):
        self.assertEqual(infer_placement_phase(0.70, 0.75, gripper_closed=True), 1.0)
        self.assertEqual(infer_placement_phase(0.70, 0.71, gripper_closed=True), 0.0)
        self.assertEqual(infer_placement_phase(0.70, 0.75, gripper_closed=False), 0.0)


if __name__ == "__main__":
    unittest.main()
