import unittest

import torch

from .train_fulltask_tagrt_v2 import (
    initial_model_config_matches,
    learning_rate_factor,
    update_ema_model,
)


class InitialModelConfigTest(unittest.TestCase):
    def setUp(self):
        self.regression = {
            "history": 3,
            "horizon": 15,
            "feature_dim": 192,
            "decoder_type": "regression",
            "motion_dim": 24,
        }
        self.flow = {**self.regression, "decoder_type": "flow"}

    def test_registered_regression_to_flow_conversion(self):
        self.assertTrue(
            initial_model_config_matches(
                self.regression,
                self.flow,
                convert_regression_checkpoint_to_flow=True,
            )
        )

    def test_conversion_does_not_hide_architecture_mismatch(self):
        mismatched = {**self.flow, "horizon": 6}
        self.assertFalse(
            initial_model_config_matches(
                self.regression,
                mismatched,
                convert_regression_checkpoint_to_flow=True,
            )
        )

    def test_cosine_schedule_warms_up_and_decays(self):
        self.assertAlmostEqual(
            learning_rate_factor(
                0, total_steps=100, warmup_steps=10, schedule="cosine"
            ),
            0.1,
        )
        self.assertAlmostEqual(
            learning_rate_factor(
                10, total_steps=100, warmup_steps=10, schedule="cosine"
            ),
            1.0,
        )
        self.assertAlmostEqual(
            learning_rate_factor(
                100, total_steps=100, warmup_steps=10, schedule="cosine"
            ),
            0.0,
        )

    def test_ema_update(self):
        source = torch.nn.Linear(1, 1, bias=False)
        target = torch.nn.Linear(1, 1, bias=False)
        source.weight.data.fill_(3.0)
        target.weight.data.fill_(1.0)
        update_ema_model(target, source, 0.5)
        self.assertAlmostEqual(float(target.weight), 2.0)

    def test_conversion_must_be_explicit(self):
        self.assertFalse(
            initial_model_config_matches(
                self.regression,
                self.flow,
                convert_regression_checkpoint_to_flow=False,
            )
        )


if __name__ == "__main__":
    unittest.main()
