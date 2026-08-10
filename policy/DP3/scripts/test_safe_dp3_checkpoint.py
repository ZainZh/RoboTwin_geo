import tempfile
import unittest
from pathlib import Path

from unittest import mock
import torch

from safe_dp3_checkpoint import (
    FORMAT_NAME,
    load_weights_only_checkpoint,
    save_weights_only_checkpoint,
    validate_weights_only_checkpoint,
)


class TestSafeDp3Checkpoint(unittest.TestCase):
    def test_round_trip_with_raw_and_ema_weights(self):
        raw = torch.nn.Linear(3, 2)
        ema = torch.nn.Linear(3, 2)
        with torch.no_grad():
            ema.weight.fill_(0.25)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.weights.pt"
            save_weights_only_checkpoint(
                path,
                model=raw,
                ema_model=ema,
                metadata={"seed": 7, "shape_meta": {"action": {"shape": [20]}}},
            )
            payload = load_weights_only_checkpoint(path)
            temporary_files = list(Path(temp_dir).glob(".model.weights.pt.*.tmp"))

        self.assertEqual(FORMAT_NAME, payload["format"])
        self.assertEqual(7, payload["metadata"]["seed"])
        torch.testing.assert_close(raw.weight, payload["model"]["weight"])
        torch.testing.assert_close(ema.weight, payload["ema_model"]["weight"])
        self.assertEqual([], temporary_files)

    def test_failed_validation_preserves_existing_destination_and_cleans_temp(self):
        model = torch.nn.Linear(2, 1)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.weights.pt"
            original = b"previous-complete-checkpoint"
            path.write_bytes(original)
            with mock.patch(
                "safe_dp3_checkpoint.load_weights_only_checkpoint",
                side_effect=ValueError("injected validation failure"),
            ):
                with self.assertRaisesRegex(ValueError, "injected validation failure"):
                    save_weights_only_checkpoint(
                        path,
                        model=model,
                        ema_model=None,
                        metadata={"seed": 9},
                    )
            self.assertEqual(original, path.read_bytes())
            self.assertEqual([], list(Path(temp_dir).glob(".model.weights.pt.*.tmp")))

    def test_metadata_validation_is_strict(self):
        model = torch.nn.Linear(2, 1)
        metadata = {
            "epoch": 3,
            "seed": 11,
            "precision": "bf16",
            "allow_tf32": True,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.weights.pt"
            save_weights_only_checkpoint(path, model=model, ema_model=None, metadata=metadata)
            validate_weights_only_checkpoint(path, expected_metadata=metadata)
            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                validate_weights_only_checkpoint(
                    path,
                    expected_metadata={"precision": "fp32"},
                )



if __name__ == "__main__":
    unittest.main()
