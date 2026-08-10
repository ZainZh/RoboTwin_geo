import contextlib
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


DP3_TRAIN_ROOT = Path(__file__).resolve().parents[1] / "3D-Diffusion-Policy"
sys.path.insert(0, str(DP3_TRAIN_ROOT))

from train_dp3 import (  # noqa: E402
    configure_training_tf32,
    resolve_training_precision,
    training_autocast_context,
)


class TestDp3TrainingPrecision(unittest.TestCase):
    def test_missing_precision_preserves_legacy_fp32(self):
        self.assertEqual("fp32", resolve_training_precision({}))
        self.assertEqual("fp32", resolve_training_precision({"precision": None}))

    def test_bf16_and_invalid_precision(self):
        self.assertEqual("bf16", resolve_training_precision({"precision": "BF16"}))
        with self.assertRaisesRegex(ValueError, "training.precision"):
            resolve_training_precision({"precision": "fp16"})

    def test_fp32_context_is_cpu_safe_noop(self):
        with training_autocast_context(torch.device("cpu"), "fp32"):
            value = torch.ones(2) + 1
        torch.testing.assert_close(value, torch.full((2,), 2.0))

    def test_bf16_autocast_is_cuda_bfloat16(self):
        fake_context = contextlib.nullcontext()
        with mock.patch("train_dp3.torch.autocast", return_value=fake_context) as autocast:
            with training_autocast_context(torch.device("cuda:0"), "bf16"):
                pass
        autocast.assert_called_once_with(device_type="cuda", dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "requires a CUDA device"):
            training_autocast_context(torch.device("cpu"), "bf16")

    def test_tf32_absent_preserves_defaults_and_explicit_values_apply(self):
        original_matmul = torch.backends.cuda.matmul.allow_tf32
        original_cudnn = torch.backends.cudnn.allow_tf32
        try:
            self.assertIsNone(configure_training_tf32({}))
            self.assertEqual(original_matmul, torch.backends.cuda.matmul.allow_tf32)
            self.assertEqual(original_cudnn, torch.backends.cudnn.allow_tf32)

            self.assertTrue(configure_training_tf32({"allow_tf32": True}))
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
            self.assertTrue(torch.backends.cudnn.allow_tf32)

            self.assertFalse(configure_training_tf32({"allow_tf32": False}))
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertFalse(torch.backends.cudnn.allow_tf32)

            with self.assertRaisesRegex(ValueError, "training.allow_tf32"):
                configure_training_tf32({"allow_tf32": "true"})
        finally:
            torch.backends.cuda.matmul.allow_tf32 = original_matmul
            torch.backends.cudnn.allow_tf32 = original_cudnn


if __name__ == "__main__":
    unittest.main()
