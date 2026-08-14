import unittest

import numpy as np

from policy.GeometryFlow.relabel_dense_gripper_commands import (
    causal_command_sequence,
    relabel_payload,
)


class RelabelDenseGripperCommandsTest(unittest.TestCase):
    def test_causal_sequence_holds_command_through_feedback_plateau(self):
        current = np.asarray([1.0, 1.0, 0.8, 0.2, 0.0, 0.0, 0.3, 0.9])
        observed = np.asarray([1.0, 0.8, 0.2, 0.0, 0.0, 0.3, 0.9, 1.0])
        command, transitions = causal_command_sequence(
            current, observed, motion_threshold=1e-3
        )
        np.testing.assert_array_equal(command, [1, 0, 0, 0, 0, 1, 1, 1])
        self.assertEqual(
            transitions,
            [{"row": 1, "command": "close"}, {"row": 5, "command": "open"}],
        )

    def test_relabel_changes_only_gripper_action_channels(self):
        samples, horizon = 5, 3
        action = np.arange(samples * horizon * 14, dtype=np.float32).reshape(
            samples, horizon, 14
        )
        state = np.zeros((samples, 3, 20), dtype=np.float32)
        state[:, -1, 9] = [1.0, 1.0, 0.5, 0.0, 0.0]
        state[:, -1, 19] = 1.0
        action[:, 0, 6] = [1.0, 0.5, 0.0, 0.0, 0.5]
        action[:, 0, 13] = 1.0
        payload = {
            "action": action,
            "state": state,
            "episode_id": np.zeros(samples, dtype=np.int64),
            "frame_index": np.arange(samples),
        }
        output, diagnostics = relabel_payload(payload, motion_threshold=1e-3)
        mask = np.ones(14, dtype=bool)
        mask[[6, 13]] = False
        np.testing.assert_array_equal(output["action"][:, :, mask], action[:, :, mask])
        np.testing.assert_array_equal(output["action"][:, 0, 6], [1, 0, 0, 0, 1])
        np.testing.assert_array_equal(output["action"][:, :, 13], 1.0)
        self.assertEqual(len(diagnostics["0"]["left"]["transitions"]), 2)


if __name__ == "__main__":
    unittest.main()
