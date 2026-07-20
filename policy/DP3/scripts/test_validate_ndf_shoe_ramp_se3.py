import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from validate_ndf_shoe_ramp_se3 import (  # noqa: E402
    axis_alignment_cosine,
    frame_alignment_loss,
    functional_matrix_with_scale,
    goal_shoe_from_ramp,
    rotation_about_point,
    _ramp_probe_grid,
    configure_descriptor_acts,
)


class TestShoeRampGeometry(unittest.TestCase):
    def setUp(self):
        self.shoe_functional = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.15],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.ramp_functional = np.array(
            [
                [0.0, -1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def test_functional_translation_uses_actor_scale(self):
        scaled = functional_matrix_with_scale(self.shoe_functional, [0.1, 0.2, 0.3])
        np.testing.assert_allclose(scaled[:3, 3], [0.0, 0.03, 0.0])
        np.testing.assert_allclose(scaled[:3, :3], self.shoe_functional[:3, :3])

    def test_task_goal_maps_ramp_uphill_axis_to_shoe_toe_axis(self):
        shoe_functional = functional_matrix_with_scale(self.shoe_functional, [0.11] * 3)
        goal = goal_shoe_from_ramp(shoe_functional, self.ramp_functional)
        cosine = axis_alignment_cosine(
            goal,
            source_axis=np.array([-1.0, 0.0, 0.0]),
            target_axis=np.array([0.0, 0.0, 1.0]),
        )
        self.assertAlmostEqual(cosine, 1.0, places=7)

    def test_half_turn_about_sole_center_reverses_toe_alignment(self):
        shoe_functional = functional_matrix_with_scale(self.shoe_functional, [0.11] * 3)
        goal = goal_shoe_from_ramp(shoe_functional, self.ramp_functional)
        half_turn = rotation_about_point(
            axis=np.array([0.0, 1.0, 0.0]),
            angle_rad=np.pi,
            point=shoe_functional[:3, 3],
        )
        flipped = half_turn @ goal
        cosine = axis_alignment_cosine(
            flipped,
            source_axis=np.array([-1.0, 0.0, 0.0]),
            target_axis=np.array([0.0, 0.0, 1.0]),
        )
        self.assertAlmostEqual(cosine, -1.0, places=7)

    def test_two_axes_reject_roll_that_toe_axis_alone_cannot_detect(self):
        shoe_functional = functional_matrix_with_scale(self.shoe_functional, [0.11] * 3)
        goal = goal_shoe_from_ramp(shoe_functional, self.ramp_functional)
        roll_about_toe = rotation_about_point(
            axis=np.array([0.0, 0.0, 1.0]),
            angle_rad=np.pi,
            point=shoe_functional[:3, 3],
        )
        rolled = roll_about_toe @ goal

        self.assertAlmostEqual(
            axis_alignment_cosine(
                rolled,
                source_axis=np.array([-1.0, 0.0, 0.0]),
                target_axis=np.array([0.0, 0.0, 1.0]),
            ),
            1.0,
            places=7,
        )
        self.assertAlmostEqual(frame_alignment_loss(goal), 0.0, places=7)
        self.assertAlmostEqual(frame_alignment_loss(rolled), 2.0, places=7)

    def test_contact_probe_grid_respects_requested_footprint(self):
        probes = _ramp_probe_grid(5, 3, half_length=0.08, half_width=0.03)
        self.assertEqual(probes.shape, (15, 3))
        self.assertAlmostEqual(float(np.max(np.abs(probes[:, 0]))), 0.08, places=7)
        self.assertAlmostEqual(float(np.max(np.abs(probes[:, 1]))), 0.03, places=7)
        np.testing.assert_allclose(probes[:, 2], 0.0005)

    def test_descriptor_layer_selection_matches_dp3_last_layer_mode(self):
        model = SimpleNamespace(decoder=SimpleNamespace(acts="all"))
        configure_descriptor_acts(model, "last")
        self.assertEqual(model.decoder.acts, "last")
        with self.assertRaisesRegex(ValueError, "descriptor_acts"):
            configure_descriptor_acts(model, "unsupported")


if __name__ == "__main__":
    unittest.main()
