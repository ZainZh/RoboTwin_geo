import unittest
from types import SimpleNamespace

import numpy as np

from .deploy_fulltask_tagrt_v2 import execute_predicted_actions


class _Task:
    def __init__(self):
        self.step_lim = 100
        self.take_action_cnt = 0
        self.eval_success = False
        self.position = None
        self.velocity = None

    def take_low_level_joint_action_chunk(self, position, velocity):
        self.position = position.copy()
        self.velocity = velocity.copy()


class FullTaskDeployExecuteTest(unittest.TestCase):
    def test_executes_precomputed_dense_chunk(self):
        runtime = SimpleNamespace(
            action_representation="dense_joint26",
            last_gripper_open_probability=np.full((2, 2), 0.25, dtype=np.float32),
        )
        model = SimpleNamespace(
            runtime=runtime,
            execute_steps=2,
            max_policy_steps=100,
            action_chunks=0,
            executed_actions=0,
            min_gripper_open_probability=np.ones(2, dtype=np.float32),
            last_gripper_command=np.ones(2, dtype=np.float32),
            gripper_closed_count=np.zeros(2, dtype=np.int64),
            first_gripper_close_step=[None, None],
        )
        task = _Task()
        actions = np.arange(52, dtype=np.float32).reshape(2, 26)
        actions[:, [12, 25]] = 0.25
        execute_predicted_actions(task, model, {}, actions)
        self.assertEqual(model.action_chunks, 1)
        self.assertEqual(model.executed_actions, 2)
        self.assertEqual(task.position.shape, (2, 14))
        self.assertEqual(task.velocity.shape, (2, 12))
        np.testing.assert_allclose(task.position[:, 6], 0.25)
        np.testing.assert_allclose(task.position[:, 13], 0.25)


if __name__ == "__main__":
    unittest.main()
