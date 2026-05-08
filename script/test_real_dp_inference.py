import argparse
import unittest

import numpy as np


class RealDPInferenceEncodingTest(unittest.TestCase):
    def test_required_image_obs_keys_are_read_from_shape_meta(self):
        from script.real_zed_inference.real_dp_inference import required_image_obs_keys

        shape_meta = {
            "obs": {
                "head_cam": {"type": "rgb", "shape": [3, 64, 80]},
                "left_cam": {"type": "rgb", "shape": [3, 64, 80]},
                "agent_pos": {"type": "low_dim", "shape": [14]},
            }
        }

        self.assertEqual(required_image_obs_keys(shape_meta), ["head_cam", "left_cam"])

    def test_parse_dp_camera_map_supports_explicit_single_camera_mapping(self):
        from script.real_zed_inference.real_dp_inference import parse_dp_camera_map

        mapping = parse_dp_camera_map("head_cam:left", required_keys=["head_cam"], live_labels=["left"])

        self.assertEqual(mapping, {"head_cam": "left"})

    def test_default_dp_camera_map_uses_standard_multicam_labels(self):
        from script.real_zed_inference.real_dp_inference import parse_dp_camera_map

        mapping = parse_dp_camera_map(
            "",
            required_keys=["head_cam", "left_cam", "right_cam"],
            live_labels=["global", "left", "right"],
        )

        self.assertEqual(mapping, {"head_cam": "global", "left_cam": "left", "right_cam": "right"})

    def test_build_dp_image_observation_resizes_and_normalizes_required_cameras(self):
        from script.real_zed_inference.real_dp_inference import build_dp_image_observation

        frames = {
            "left": {"rgb": np.full((4, 6, 3), 255, dtype=np.uint8)},
        }
        joint_vector = np.arange(14, dtype=np.float32)

        obs = build_dp_image_observation(
            frames_by_label=frames,
            joint_vector=joint_vector,
            obs_keys=["head_cam"],
            camera_map={"head_cam": "left"},
            image_shapes={"head_cam": (3, 2, 3)},
        )

        self.assertEqual(set(obs.keys()), {"head_cam", "agent_pos"})
        self.assertEqual(obs["head_cam"].shape, (3, 2, 3))
        self.assertEqual(obs["head_cam"].dtype, np.float32)
        self.assertAlmostEqual(float(obs["head_cam"].max()), 1.0, places=6)
        np.testing.assert_allclose(obs["agent_pos"], joint_vector)


class RealDPInferenceRunnerTest(unittest.TestCase):
    def test_dp_action_runner_uses_only_checkpoint_required_obs_keys(self):
        from script.real_zed_inference.real_dp_inference import RealDPActionRunner

        class FakePolicy:
            device = "cpu"
            dtype = "float32"

            def __init__(self):
                self.received = None

            def predict_action(self, obs_dict):
                self.received = obs_dict
                import torch

                return {"action": torch.ones((1, 6, 14), dtype=torch.float32)}

        runner = RealDPActionRunner(n_obs_steps=3, n_action_steps=2, obs_keys=["head_cam"])
        obs = {
            "head_cam": np.ones((3, 2, 2), dtype=np.float32),
            "left_cam": np.zeros((3, 2, 2), dtype=np.float32),
            "agent_pos": np.zeros(14, dtype=np.float32),
        }
        policy = FakePolicy()

        actions = runner.get_action(policy, obs)

        self.assertEqual(actions.shape, (2, 14))
        self.assertIn("head_cam", policy.received)
        self.assertNotIn("left_cam", policy.received)
        self.assertIn("agent_pos", policy.received)
        self.assertEqual(tuple(policy.received["head_cam"].shape), (1, 3, 3, 2, 2))


class RealDPInferenceParserTest(unittest.TestCase):
    def test_parser_defaults_to_async_control_and_dry_run(self):
        from script.real_zed_inference.real_dp_inference import build_arg_parser

        args = build_arg_parser().parse_args([])

        self.assertFalse(args.execute)
        self.assertTrue(args.async_control)
        self.assertAlmostEqual(args.async_control_hz, 25.0)
        self.assertAlmostEqual(args.max_executed_joint_delta, 0.02)
        self.assertAlmostEqual(args.max_executed_joint_delta_change, 0.005)


if __name__ == "__main__":
    unittest.main()
