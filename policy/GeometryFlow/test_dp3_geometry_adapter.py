from __future__ import annotations

import sys
from pathlib import Path
import unittest

import torch


DP3_ROOT = Path(__file__).resolve().parents[1] / "DP3" / "3D-Diffusion-Policy"
sys.path.insert(0, str(DP3_ROOT))

from diffusion_policy_3d.policy.dp3 import (  # noqa: E402
    BinaryGripperHead,
    FutureGeometryHead,
    ZeroInitGlobalGeometryAdapter,
    ZeroInitGeometryAdapter,
)


class BinaryGripperHeadTest(unittest.TestCase):
    def test_output_shape_and_shared_context_gradient(self):
        head = BinaryGripperHead(24, 16, 12)
        context = torch.randn(5, 24, requires_grad=True)
        output = head(context)
        self.assertEqual(tuple(output.shape), (5, 12))
        output.square().mean().backward()
        self.assertGreater(float(context.grad.abs().sum()), 0.0)


class ZeroInitGeometryAdapterTest(unittest.TestCase):
    def test_initial_output_is_exactly_zero_and_projection_learns(self):
        torch.manual_seed(4)
        adapter = ZeroInitGeometryAdapter(6, 16, 24)
        points = torch.randn(5, 32, 6)
        output = adapter(points)
        torch.testing.assert_close(output, torch.zeros_like(output))
        loss = torch.square(output - 1.0).mean()
        loss.backward()
        self.assertGreater(float(adapter.projection.weight.grad.abs().sum()), 0.0)

    def test_force_zero_disables_adapter_even_after_projection_changes(self):
        adapter = ZeroInitGeometryAdapter(6, 8, 12, force_zero=True)
        with torch.no_grad():
            adapter.projection.weight.fill_(1.0)
            adapter.projection.bias.fill_(1.0)
        output = adapter(torch.randn(3, 10, 6))
        torch.testing.assert_close(output, torch.zeros_like(output))

    def test_confidence_zero_disables_residual(self):
        adapter = ZeroInitGeometryAdapter(6, 8, 12)
        with torch.no_grad():
            adapter.projection.weight.fill_(0.1)
        output = adapter(
            torch.randn(3, 10, 6), confidence=torch.zeros(3, 1)
        )
        torch.testing.assert_close(output, torch.zeros_like(output))


class FutureGeometryHeadTest(unittest.TestCase):
    def test_output_shape_and_shared_context_gradient(self):
        head = FutureGeometryHead(24, 16, 96)
        context = torch.randn(5, 24, requires_grad=True)
        output = head(context)
        self.assertEqual(tuple(output.shape), (5, 96))
        output.square().mean().backward()
        self.assertGreater(float(context.grad.abs().sum()), 0.0)


class ZeroInitGlobalGeometryAdapterTest(unittest.TestCase):
    def test_initial_output_is_exactly_zero_and_projection_learns(self):
        adapter = ZeroInitGlobalGeometryAdapter(9, 16, 24)
        relation = torch.randn(5, 9)
        output = adapter(relation)
        torch.testing.assert_close(output, torch.zeros_like(output))
        torch.square(output - 1.0).mean().backward()
        self.assertGreater(float(adapter.projection.weight.grad.abs().sum()), 0.0)

    def test_zero_confidence_disables_global_relation(self):
        adapter = ZeroInitGlobalGeometryAdapter(9, 16, 24)
        with torch.no_grad():
            adapter.projection.weight.fill_(0.1)
        output = adapter(torch.randn(3, 9), confidence=torch.zeros(3, 1))
        torch.testing.assert_close(output, torch.zeros_like(output))


if __name__ == "__main__":
    unittest.main()
