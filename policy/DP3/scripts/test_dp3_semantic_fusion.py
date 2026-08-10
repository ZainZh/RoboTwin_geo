import sys
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[3]
DP3_ROOT = REPO_ROOT / "policy" / "DP3" / "3D-Diffusion-Policy"
if str(DP3_ROOT) not in sys.path:
    sys.path.insert(0, str(DP3_ROOT))

from diffusion_policy_3d.model.vision.pointnet_extractor import DP3Encoder
from diffusion_policy_3d.model.vision.semantic_fusion_encoder import (
    SemanticFusionDP3Encoder,
)


def pointnet_cfg(out_channels: int = 16):
    return OmegaConf.create(
        {
            "in_channels": 3,
            "out_channels": out_channels,
            "use_layernorm": True,
            "final_norm": "layernorm",
            "normal_channel": False,
        }
    )


def observations(batch_size: int = 3):
    return {
        "point_cloud": torch.randn(batch_size, 32, 6),
        "semantic_point_cloud_A": torch.randn(batch_size, 12, 5),
        "agent_pos": torch.randn(batch_size, 4),
    }


class TestSemanticFusionDP3Encoder(unittest.TestCase):
    out_channels = 16
    state_size = (8,)

    def build_vanilla(self, seed: int = 7):
        torch.manual_seed(seed)
        return DP3Encoder(
            observation_space={
                "point_cloud": [32, 6],
                "agent_pos": [4],
            },
            out_channel=self.out_channels,
            state_mlp_size=self.state_size,
            pointcloud_encoder_cfg=pointnet_cfg(self.out_channels),
            use_pc_color=False,
        )

    def build_fusion(
        self,
        mode: str,
        *,
        seed: int = 7,
        trainable: bool = True,
        gate_scale: float = 1.0,
    ):
        torch.manual_seed(seed)
        return SemanticFusionDP3Encoder(
            observation_space={
                "point_cloud": [32, 6],
                "semantic_point_cloud_A": [12, 5],
                "agent_pos": [4],
            },
            out_channel=self.out_channels,
            state_mlp_size=self.state_size,
            pointcloud_encoder_cfg=pointnet_cfg(self.out_channels),
            use_pc_color=False,
            pointcloud_fusion_mode=mode,
            semantic_gate_init=0.0,
            semantic_gate_trainable=trainable,
            semantic_gate_scale=gate_scale,
            semantic_attention_heads=4,
        )

    def assert_base_parameters_equal(self, vanilla, fusion):
        expected = vanilla.state_dict()
        actual = fusion.base_encoder.state_dict()
        self.assertEqual(set(expected), set(actual))
        for key in expected:
            self.assertTrue(
                torch.equal(expected[key], actual[key]),
                f"base parameter differs: {key}",
            )

    def test_semantic_modules_preserve_outer_rng_stream(self):
        torch.manual_seed(19)
        self.build_vanilla(seed=19)
        expected_next = torch.randn(32)

        torch.manual_seed(19)
        self.build_fusion("state_cross_attention", seed=19)
        actual_next = torch.randn(32)

        self.assertTrue(torch.equal(expected_next, actual_next))

    def test_zero_gate_is_bitwise_vanilla_for_all_modes(self):
        obs = observations()
        vanilla_obs = {
            "point_cloud": obs["point_cloud"],
            "agent_pos": obs["agent_pos"],
        }
        vanilla = self.build_vanilla()
        expected = vanilla(vanilla_obs)

        for mode in (
            "residual_pointnet",
            "state_cross_attention",
            "part_soft_pool",
        ):
            with self.subTest(mode=mode):
                fusion = self.build_fusion(mode)
                self.assert_base_parameters_equal(vanilla, fusion)
                actual = fusion(obs)
                self.assertEqual(vanilla.output_shape(), fusion.output_shape())
                self.assertEqual(24, fusion.output_shape())
                self.assertTrue(torch.equal(expected, actual))

    def test_trainable_gate_receives_gradient_at_zero(self):
        model = self.build_fusion("residual_pointnet")
        output = model(observations())
        weights = torch.linspace(-1.0, 1.0, output.shape[-1])
        loss = (output * weights).sum()
        loss.backward()

        gate = model.semantic_gates["semantic_point_cloud_A"]
        self.assertIsNotNone(gate.grad)
        self.assertGreater(float(gate.grad.abs().sum()), 0.0)

    def test_runtime_zero_gate_scale_is_bitwise_vanilla(self):
        obs = observations()
        vanilla = self.build_vanilla()
        fusion = self.build_fusion("part_soft_pool", gate_scale=0.0)
        with torch.no_grad():
            fusion.semantic_gates["semantic_point_cloud_A"].fill_(0.75)
        expected = vanilla({
            "point_cloud": obs["point_cloud"],
            "agent_pos": obs["agent_pos"],
        })
        actual = fusion(obs)
        self.assertTrue(torch.equal(expected, actual))

    def test_frozen_zero_gate_is_a_capacity_matched_vanilla_control(self):
        model = self.build_fusion(
            "residual_pointnet",
            trainable=False,
        )
        gate = model.semantic_gates["semantic_point_cloud_A"]
        self.assertFalse(gate.requires_grad)
        self.assertTrue(torch.equal(gate, torch.zeros_like(gate)))

    def test_part_soft_pool_is_permutation_invariant_after_gate_opens(self):
        model = self.build_fusion("part_soft_pool")
        with torch.no_grad():
            model.semantic_gates["semantic_point_cloud_A"].fill_(0.5)
        obs = observations()
        permuted = dict(obs)
        permutation = torch.randperm(obs["semantic_point_cloud_A"].shape[1])
        permuted["semantic_point_cloud_A"] = obs[
            "semantic_point_cloud_A"
        ][:, permutation]

        first = model(obs)
        second = model(permuted)
        self.assertTrue(torch.allclose(first, second, atol=1e-6, rtol=1e-6))

    def test_state_cross_attention_is_finite_after_gate_opens(self):
        model = self.build_fusion("state_cross_attention")
        with torch.no_grad():
            model.semantic_gates["semantic_point_cloud_A"].fill_(0.5)
        output = model(observations())
        self.assertEqual((3, 24), tuple(output.shape))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
