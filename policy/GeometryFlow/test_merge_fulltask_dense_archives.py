import unittest

import numpy as np

from .merge_fulltask_dense_archives import merge


class MergeFullTaskDenseArchivesTest(unittest.TestCase):
    @staticmethod
    def payload(episodes):
        episode_id = np.asarray(episodes, dtype=np.int64)
        count = len(episode_id)
        return {
            "episode_id": episode_id,
            "shoe_id": np.arange(count, dtype=np.int64),
            "state": np.arange(count * 2, dtype=np.float32).reshape(count, 2),
        }

    def test_reindexes_and_marks_corrections(self):
        nominal = self.payload([0, 0, 1])
        correction = self.payload([0, 1, 1])
        result = merge(nominal, correction)
        np.testing.assert_array_equal(result["episode_id"], [0, 0, 1, 2, 3, 3])
        np.testing.assert_array_equal(
            result["is_correction_sample"], [0, 0, 0, 1, 1, 1]
        )

    def test_rejects_key_mismatch(self):
        nominal = self.payload([0])
        correction = self.payload([0])
        correction["extra"] = np.zeros((1,), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "identical keys"):
            merge(nominal, correction)


if __name__ == "__main__":
    unittest.main()
