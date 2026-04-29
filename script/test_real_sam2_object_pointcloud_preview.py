import unittest

import numpy as np


class RealSam2ObjectPointcloudPreviewTest(unittest.TestCase):
    def test_prepare_placeholder_clouds_filters_empty_rows_and_applies_placeholder_colors(self):
        from script.real_zed_inference.preview_sam2_object_pointcloud import prepare_placeholder_clouds

        object_pointcloud = {
            "{A}": np.array(
                [
                    [0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                ],
                dtype=np.float32,
            ),
            "{B}": np.array([[0.4, 0.5, 0.6, 0.2, 0.3, 0.4]], dtype=np.float32),
        }

        clouds = prepare_placeholder_clouds(
            object_pointcloud,
            placeholders=["{A}", "{B}"],
            color_mode="placeholder",
        )

        self.assertEqual(set(clouds), {"{A}", "{B}"})
        self.assertEqual(clouds["{A}"].shape, (1, 6))
        np.testing.assert_allclose(clouds["{A}"][0, :3], [0.1, 0.2, 0.3])
        np.testing.assert_allclose(clouds["{A}"][0, 3:6], [0.95, 0.25, 0.2])
        np.testing.assert_allclose(clouds["{B}"][0, 3:6], [0.2, 0.45, 1.0])

    def test_prepare_placeholder_clouds_can_keep_rgb_colors(self):
        from script.real_zed_inference.preview_sam2_object_pointcloud import prepare_placeholder_clouds

        object_pointcloud = {
            "{A}": np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], dtype=np.float32),
        }

        clouds = prepare_placeholder_clouds(
            object_pointcloud,
            placeholders=["{A}"],
            color_mode="rgb",
        )

        np.testing.assert_allclose(clouds["{A}"][0, 3:6], [0.4, 0.5, 0.6])


if __name__ == "__main__":
    unittest.main()
