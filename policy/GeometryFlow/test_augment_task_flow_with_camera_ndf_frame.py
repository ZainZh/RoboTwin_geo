from __future__ import annotations

import unittest

import numpy as np

from .augment_task_flow_with_camera_ndf_frame import error_summary


class CameraNdfFrameAugmentTest(unittest.TestCase):
    def test_empty_split_summary_is_explicit_and_json_safe(self):
        summary = error_summary(
            np.empty((0, 3), dtype=np.float32),
            np.asarray([], dtype=np.float32),
        )
        self.assertEqual(summary["samples"], 0)
        self.assertIsNone(summary["translation_cm_p90"])
        self.assertIsNone(summary["rotation_deg_p90"])


if __name__ == "__main__":
    unittest.main()
