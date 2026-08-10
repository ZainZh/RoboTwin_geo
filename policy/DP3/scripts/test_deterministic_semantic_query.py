import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from object_pointcloud_utils import resample_point_cloud


def make_cloud(size: int) -> np.ndarray:
    index = np.arange(size, dtype=np.float32)
    return np.stack(
        [
            index,
            np.square(index + 1.0),
            np.sin(index),
            np.zeros_like(index),
            np.ones_like(index),
            np.full_like(index, 2.0),
        ],
        axis=1,
    ).astype(np.float32)


class TestDeterministicSemanticQuery(unittest.TestCase):
    def test_deterministic_fps_repeats_across_global_seeds(self):
        cloud = make_cloud(17)
        outputs = []
        for seed in (1, 7, 20260810):
            np.random.seed(seed)
            outputs.append(
                resample_point_cloud(cloud, 8, deterministic=True)
            )
        np.testing.assert_array_equal(outputs[0], outputs[1])
        np.testing.assert_array_equal(outputs[0], outputs[2])

    def test_deterministic_fps_does_not_advance_global_numpy_rng(self):
        cloud = make_cloud(17)
        np.random.seed(123)
        expected = np.random.random(12)

        np.random.seed(123)
        resample_point_cloud(cloud, 8, deterministic=True)
        actual = np.random.random(12)

        np.testing.assert_array_equal(expected, actual)

    def test_deterministic_padding_does_not_advance_global_numpy_rng(self):
        cloud = make_cloud(3)
        np.random.seed(456)
        expected = np.random.random(12)

        np.random.seed(456)
        output = resample_point_cloud(cloud, 8, deterministic=True)
        actual = np.random.random(12)

        self.assertEqual((8, 6), output.shape)
        np.testing.assert_array_equal(expected, actual)
        np.testing.assert_array_equal(output[:3], cloud)
        np.testing.assert_array_equal(output[3:], cloud[[0, 1, 2, 0, 1]])

    def test_legacy_default_still_uses_global_rng(self):
        cloud = make_cloud(17)
        np.random.seed(5)
        before = np.random.get_state()
