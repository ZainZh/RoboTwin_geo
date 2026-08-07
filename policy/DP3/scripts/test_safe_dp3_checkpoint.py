import tempfile
import unittest
from pathlib import Path

import torch

from safe_dp3_checkpoint import FORMAT_NAME, load_weights_only_checkpoint, save_weights_only_checkpoint


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

        self.assertEqual(FORMAT_NAME, payload["format"])
        self.assertEqual(7, payload["metadata"]["seed"])
        torch.testing.assert_close(raw.weight, payload["model"]["weight"])
        torch.testing.assert_close(ema.weight, payload["ema_model"]["weight"])


if __name__ == "__main__":
    unittest.main()
