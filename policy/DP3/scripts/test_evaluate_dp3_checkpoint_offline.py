import unittest

from omegaconf import OmegaConf

from evaluate_dp3_checkpoint_offline import summarize_losses, validate_checkpoint_metadata


class TestOfflineDp3Evaluation(unittest.TestCase):
    def test_loss_summary_uses_sample_standard_deviation(self):
        summary = summarize_losses([1.0, 2.0, 3.0])
        self.assertEqual(3, summary["n"])
        self.assertAlmostEqual(2.0, summary["mean"])
        self.assertAlmostEqual(1.0, summary["std"])

    def test_metadata_requires_matching_shape_and_zarr(self):
        cfg = OmegaConf.create(
            {
                "task": {
                    "shape_meta": {"action": {"shape": [20]}},
                    "dataset": {"zarr_path": "/tmp/matched.zarr"},
                }
            }
        )
        metadata = {
            "shape_meta": {"action": {"shape": [20]}},
            "zarr_path": "/tmp/matched.zarr",
        }
        validate_checkpoint_metadata(metadata, cfg)

        metadata["shape_meta"] = {"action": {"shape": [14]}}
        with self.assertRaisesRegex(ValueError, "shape_meta"):
            validate_checkpoint_metadata(metadata, cfg)

    def test_empty_loss_summary_fails(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            summarize_losses([])


if __name__ == "__main__":
    unittest.main()
