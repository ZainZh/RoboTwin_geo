import unittest

import numpy as np

from actorseg_pointcloud_utils import (
    actor_ids_to_colors,
    extract_placeholder_point_cloud_actorseg,
    extract_placeholder_point_cloud_actorseg_online,
)


class TestActorSegPointCloudUtils(unittest.TestCase):
    def setUp(self):
        self.intrinsic = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        self.extrinsic = np.asarray(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        self.scene_pc = np.asarray(
            [
                [1.0, 1.0, 1.0, 0.1, 0.2, 0.3],
                [2.0, 2.0, 1.0, 0.4, 0.5, 0.6],
                [4.0, 4.0, 1.0, 0.7, 0.8, 0.9],
            ],
            dtype=np.float32,
        )
        self.actor_id = 5
        self.seg = np.zeros((6, 6, 3), dtype=np.uint8)
        color = actor_ids_to_colors([self.actor_id])[0]
        self.seg[1, 1] = color
        self.seg[2, 2] = color

    def test_extract_online_selects_expected_points(self):
        observation = {
            "pointcloud": self.scene_pc,
            "observation": {
                "head_camera": {
                    "actor_segmentation": self.seg,
                    "intrinsic_cv": self.intrinsic,
                    "extrinsic_cv": self.extrinsic,
                }
            },
        }

        cloud, meta = extract_placeholder_point_cloud_actorseg_online(
            observation,
            placeholder="{A}",
            actor_ids=[self.actor_id],
            camera_names=["head_camera"],
            target_num_points=2,
        )

        self.assertEqual(meta["mode"], "actorseg_projected")
        self.assertEqual(cloud.shape, (2, 6))
        np.testing.assert_allclose(cloud[:, :3], self.scene_pc[:2, :3], atol=1e-6)

    def test_extract_offline_returns_zero_cloud_when_actor_ids_missing(self):
        episode = {
            "pointcloud": np.asarray([self.scene_pc], dtype=np.float32),
            "vector": np.zeros((1, 14), dtype=np.float32),
            "cameras": {
                "head_camera": {
                    "actor_segmentation": np.asarray([self.seg], dtype=np.uint8),
                    "intrinsic_cv": np.asarray([self.intrinsic], dtype=np.float32),
                    "extrinsic_cv": np.asarray([self.extrinsic], dtype=np.float32),
                }
            },
        }

        cloud, meta = extract_placeholder_point_cloud_actorseg(
            episode,
            frame_idx=0,
            placeholder="{A}",
            actor_ids=[],
            camera_names=["head_camera"],
            target_num_points=4,
        )

        self.assertEqual(meta["mode"], "actorseg_empty")
        self.assertEqual(cloud.shape, (4, 6))
        self.assertTrue(np.allclose(cloud, 0.0))


if __name__ == "__main__":
    unittest.main()
