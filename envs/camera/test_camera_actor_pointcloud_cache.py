import numpy as np
import torch
import unittest
from unittest.mock import patch

from envs.camera import camera as camera_module


class _FakeCamera:
    def __init__(self):
        self.calls = {"Color": 0, "Position": 0, "Segmentation": 0}
        self.color = np.zeros((1, 4, 4), dtype=np.float32)
        self.color[0, :, :3] = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        self.position = np.zeros((1, 4, 4), dtype=np.float32)
        self.position[0, :, 0] = np.arange(4, dtype=np.float32)
        self.segmentation = np.zeros((1, 4, 4), dtype=np.uint32)
        self.segmentation[0, :, 1] = np.asarray([11, 22, 11, 22])

    def get_picture(self, name):
        self.calls[name] += 1
        return {
            "Color": self.color,
            "Position": self.position,
            "Segmentation": self.segmentation,
        }[name]

    @staticmethod
    def get_model_matrix():
        return np.eye(4, dtype=np.float32)


class ActorPointCloudCacheTest(unittest.TestCase):
    def test_actor_clouds_share_one_camera_read_per_render(self):
        fake = _FakeCamera()
        cameras = camera_module.Camera.__new__(camera_module.Camera)
        cameras.pcd_crop = False
        cameras.pcd_down_sample_num = 1
        cameras.collect_wrist_camera = False
        cameras.collect_head_camera = True
        cameras.head_camera_id = 0
        cameras.static_camera_list = [fake]

        def first_point_fps(points, num_points=1024, use_cuda=True):
            del use_cuda
            count = min(int(num_points), len(points))
            indices = torch.arange(count, dtype=torch.int64).reshape(1, -1)
            return np.asarray(points[:count]), indices

        with patch.object(camera_module, "fps", first_point_fps):
            result = cameras.get_actor_point_clouds(
                {"{A}": [11], "{B}": [22]}, point_num=1
            )

        self.assertEqual(
            fake.calls, {"Color": 1, "Position": 1, "Segmentation": 1}
        )
        np.testing.assert_allclose(result["{A}"][0, :3], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(result["{B}"][0, :3], [1.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
