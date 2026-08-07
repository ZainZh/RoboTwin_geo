import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_ROOT = REPO_ROOT / "policy" / "DP3" / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import matched_utonia_feature_utils as utils


class TestMatchedUtoniaFeatureUtils(unittest.TestCase):
    def setUp(self):
        self.artifacts = {
            "feature_dim": 4,
            "coord_scale": 1.0,
            "center_shift_z": True,
            "device": torch.device("cpu"),
        }
        self.support = np.asarray(
            [
                [0.0, 0.0, 0.1, 0.1, 0.2, 0.3],
                [0.1, 0.0, 0.1, 0.2, 0.3, 0.4],
                [0.0, 0.1, 0.1, 0.3, 0.4, 0.5],
                [0.0, 0.0, 0.2, 0.4, 0.5, 0.6],
            ],
            dtype=np.float32,
        )

    def test_debug_color_matches_semantic_preprocessing(self):
        prepared = utils.prepare_utonia_input_point_cloud(
            self.support,
            placeholder="{A}",
            color_mode="debug_placeholder",
        )
        expected = np.asarray([255.0, 48.0, 48.0], dtype=np.float32)
        self.assertTrue(np.array_equal(prepared[:, 3:6], np.tile(expected, (4, 1))))

    def test_exact_queries_are_bitwise_preserved(self):
        queries = np.asarray(
            [[0.025, 0.025, 0.11], [0.075, 0.025, 0.15]], dtype=np.float32
        )
        support_features = np.arange(16, dtype=np.float32).reshape(4, 4)
        with patch.object(
            utils,
            "_extract_support_features",
            return_value=support_features,
        ):
            output = utils.compute_utonia_features_at_queries(
                self.artifacts,
                self.support,
                queries,
                placeholder="{A}",
            )
        self.assertEqual(output.shape, (2, 7))
        self.assertTrue(np.array_equal(output[:, :3], queries))

    def test_empty_support_keeps_query_coordinates(self):
        queries = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
        output = utils.compute_utonia_features_at_queries(
            self.artifacts,
            np.zeros((0, 6), dtype=np.float32),
            queries,
        )
        self.assertTrue(np.array_equal(output[:, :3], queries))
        self.assertTrue(np.array_equal(output[:, 3:], np.zeros((1, 4), dtype=np.float32)))

    def test_nearest_context_support_selects_query_neighborhood(self):
        context = np.asarray(
            [
                [0.0, 0.0, 0.1, 0, 0, 0],
                [0.01, 0.0, 0.1, 0, 0, 0],
                [1.0, 1.0, 1.0, 0, 0, 0],
            ],
            dtype=np.float32,
        )
        query = np.asarray([[0.0, 0.0, 0.1]], dtype=np.float32)
        support = utils.nearest_context_support(
            context,
            query,
            support_num_points=2,
        )
        self.assertEqual(support.shape, (2, 6))
        self.assertLess(float(np.max(support[:, 0])), 0.1)


if __name__ == "__main__":
    unittest.main()
