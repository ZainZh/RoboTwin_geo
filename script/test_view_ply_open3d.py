import unittest

import numpy as np

import view_ply_open3d


class FakeViewControl:
    def __init__(self):
        self.front = None
        self.up = None
        self.lookat = None
        self.zoom = None

    def set_front(self, value):
        self.front = list(value)

    def set_up(self, value):
        self.up = list(value)

    def set_lookat(self, value):
        self.lookat = list(value)

    def set_zoom(self, value):
        self.zoom = float(value)


class FakeVisualizer:
    def __init__(self, status="{\"class_name\":\"ViewTrajectory\"}"):
        self.status = status
        self.received_status = None
        self.registered = {}

    def get_view_status(self):
        return self.status

    def set_view_status(self, status):
        self.received_status = status

    def register_key_callback(self, key, callback):
        self.registered[int(key)] = callback


class TestViewPlyOpen3d(unittest.TestCase):
    def test_compute_scene_center_uses_all_clouds(self):
        clouds = [
            np.asarray([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]], dtype=np.float32),
            np.asarray([[4.0, -2.0, 1.0]], dtype=np.float32),
        ]

        center = view_ply_open3d.compute_scene_center_from_arrays(clouds)

        self.assertTrue(np.allclose(center, np.asarray([2.0, 0.0, 1.0], dtype=np.float32)))

    def test_global_view_sets_stable_front_up_lookat_and_zoom(self):
        control = FakeViewControl()
        lookat = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)

        view_ply_open3d.apply_view_preset(
            control,
            preset="global",
            lookat=lookat,
            zoom=0.42,
        )

        self.assertEqual(control.front, [0.0, -1.0, 0.0])
        self.assertEqual(control.up, [0.0, 0.0, 1.0])
        self.assertEqual(control.lookat, [1.0, 2.0, 3.0])
        self.assertEqual(control.zoom, 0.42)

    def test_view_status_report_contains_python_assignment(self):
        report = view_ply_open3d.format_view_status_report('{"zoom": 0.5}')

        self.assertIn("DEFAULT_VIEW_STATUS = r'''", report)
        self.assertIn('{"zoom": 0.5}', report)
        self.assertIn("--view_status_file", report)

    def test_apply_view_status_prefers_explicit_status(self):
        visualizer = FakeVisualizer()

        applied = view_ply_open3d.apply_view_status(visualizer, view_status='{"zoom": 0.5}')

        self.assertTrue(applied)
        self.assertEqual(visualizer.received_status, '{"zoom": 0.5}')

    def test_register_print_view_status_key_uses_uppercase_key(self):
        visualizer = FakeVisualizer()

        view_ply_open3d.register_print_view_status_key(visualizer, key="p")

        self.assertIn(ord("P"), visualizer.registered)


if __name__ == "__main__":
    unittest.main()
