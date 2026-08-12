import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from .train_fulltask_tagrt_v2 import Statistics, select_statistics, split_from_file


class FullTaskCorrectionSplitTest(unittest.TestCase):
    def write(self, root: Path, payload: dict) -> Path:
        path = root / "split.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_correction_episodes_can_be_train_only(self):
        episode_id = np.asarray([0, 0, 1, 2, 3, 3], dtype=np.int64)
        with tempfile.TemporaryDirectory() as directory:
            split, definition = split_from_file(
                episode_id,
                self.write(
                    Path(directory),
                    {"train": [0, 3], "validation": [1], "test": [2]},
                ),
            )
        self.assertEqual(definition["train"], [0, 3])
        np.testing.assert_array_equal(split["train"], [0, 1, 4, 5])

    def test_split_must_cover_every_episode(self):
        episode_id = np.asarray([0, 1, 2], dtype=np.int64)
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(
                Path(directory),
                {"train": [0], "validation": [1], "test": [3]},
            )
            with self.assertRaisesRegex(ValueError, "cover every"):
                split_from_file(episode_id, path)

    def test_continuation_can_reuse_initial_statistics(self):
        expected = Statistics(
            point_mean=[1.0],
            point_std=[2.0],
            local_mean=[3.0],
            local_std=[4.0],
            global_mean=[5.0],
            global_std=[6.0],
            state_mean=[7.0],
            state_std=[8.0],
            motion_mean=[9.0],
            motion_std=[10.0],
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "initial.pt"
            torch.save({"statistics": expected.__dict__}, checkpoint)
            actual = select_statistics(
                {},
                np.asarray([], dtype=np.int64),
                initialize_from=checkpoint,
                reuse_initial_statistics=True,
            )
        self.assertEqual(actual, expected)

    def test_reuse_statistics_requires_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "requires --initialize-from"):
            select_statistics(
                {},
                np.asarray([], dtype=np.int64),
                initialize_from=None,
                reuse_initial_statistics=True,
            )


if __name__ == "__main__":
    unittest.main()
