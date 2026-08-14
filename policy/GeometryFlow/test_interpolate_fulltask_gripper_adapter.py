import unittest

import torch

from .interpolate_fulltask_gripper_adapter import (
    graft_exact_trunk,
    interpolate_adapter_state,
)


class FullTaskGripperAdapterInterpolationTest(unittest.TestCase):
    def test_only_gripper_adapter_is_interpolated(self):
        low = {
            "gripper_geometry_adapter.0.weight": torch.tensor([0.0, 2.0]),
            "decoder.weight": torch.tensor([4.0]),
        }
        high = {
            "gripper_geometry_adapter.0.weight": torch.tensor([10.0, 12.0]),
            "decoder.weight": torch.tensor([4.0]),
        }
        state, changed = interpolate_adapter_state(low, high, 0.8)
        torch.testing.assert_close(
            state["gripper_geometry_adapter.0.weight"], torch.tensor([8.0, 10.0])
        )
        self.assertEqual(changed, ["gripper_geometry_adapter.0.weight"])

    def test_non_adapter_change_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-adapter"):
            interpolate_adapter_state(
                {"decoder.weight": torch.tensor([0.0])},
                {"decoder.weight": torch.tensor([1.0])},
                0.5,
            )

    def test_exact_trunk_replaces_only_non_adapter_tensors(self):
        adapted = {
            "gripper_geometry_adapter.0.weight": torch.tensor([8.0]),
            "observation_gripper_position": torch.tensor([3.0]),
            "observation_gripper_head.1.weight": torch.tensor([2.0]),
            "decoder.weight": torch.tensor([4.000001]),
        }
        result = graft_exact_trunk(
            adapted, {"decoder.weight": torch.tensor([4.0])}
        )
        torch.testing.assert_close(
            result["gripper_geometry_adapter.0.weight"], torch.tensor([8.0])
        )
        torch.testing.assert_close(
            result["observation_gripper_position"], torch.tensor([3.0])
        )
        torch.testing.assert_close(
            result["observation_gripper_head.1.weight"], torch.tensor([2.0])
        )
        self.assertTrue(torch.equal(result["decoder.weight"], torch.tensor([4.0])))


if __name__ == "__main__":
    unittest.main()
