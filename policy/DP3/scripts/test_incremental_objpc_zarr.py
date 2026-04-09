import tempfile
import unittest
from pathlib import Path

import numpy as np
import zarr

from incremental_objpc_zarr import (
    append_episode_to_buffer,
    open_or_reset_replay_buffer,
)


def make_episode(length: int, *, point_dim: int = 6):
    point_cloud = np.arange(length * 4 * point_dim, dtype=np.float32).reshape(length, 4, point_dim)
    state = np.arange(length * 3, dtype=np.float32).reshape(length, 3)
    action = (np.arange(length * 3, dtype=np.float32).reshape(length, 3) + 100.0)
    object_pc_a = point_cloud + 1.0
    object_pc_b = point_cloud + 2.0
    return {
        "point_cloud": point_cloud,
        "state": state,
        "action": action,
        "object_point_cloud_A": object_pc_a,
        "object_point_cloud_B": object_pc_b,
    }


class IncrementalObjPCZarrTest(unittest.TestCase):
    def test_append_persists_across_reopen(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "sample.zarr"

            buffer, start_episode = open_or_reset_replay_buffer(str(zarr_path))
            self.assertEqual(start_episode, 0)

            ep0 = make_episode(2)
            append_episode_to_buffer(buffer, ep0)

            reopened, resumed_episode = open_or_reset_replay_buffer(str(zarr_path))
            self.assertEqual(resumed_episode, 1)
            self.assertEqual(reopened.n_episodes, 1)
            np.testing.assert_array_equal(reopened.episode_ends[:], np.asarray([2], dtype=np.int64))
            self.assertEqual(reopened["state"].shape[0], 2)
            self.assertEqual(reopened["object_point_cloud_A"].shape[0], 2)

            ep1 = make_episode(1)
            append_episode_to_buffer(reopened, ep1)

            resumed, resumed_episode_again = open_or_reset_replay_buffer(str(zarr_path))
            self.assertEqual(resumed_episode_again, 2)
            np.testing.assert_array_equal(resumed.episode_ends[:], np.asarray([2, 3], dtype=np.int64))
            self.assertEqual(resumed["point_cloud"].shape[0], 3)
            self.assertEqual(resumed["action"].shape[0], 3)

    def test_incomplete_store_is_reset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "broken.zarr"
            root = zarr.group(str(zarr_path))
            root.create_group("data")
            root.create_group("meta")

            buffer, start_episode = open_or_reset_replay_buffer(str(zarr_path))
            self.assertEqual(start_episode, 0)
            self.assertEqual(buffer.n_episodes, 0)
            np.testing.assert_array_equal(buffer.episode_ends[:], np.asarray([], dtype=np.int64))


if __name__ == "__main__":
    unittest.main()
