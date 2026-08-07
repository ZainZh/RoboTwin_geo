import json
import unittest
from pathlib import Path

import numpy as np
import trimesh

from script.prepare_gso_shoe_eval_assets import (
    GSO_TO_ROBOTWIN,
    asset_gate,
    build_collision_scene,
    build_model_data,
)


class PrepareGsoShoeEvalAssetsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        reference_path = Path(__file__).resolve().parents[1] / "assets/objects/041_shoe/model_data0.json"
        cls.reference = json.loads(reference_path.read_text(encoding="utf-8"))

    def test_coordinate_rotation_maps_gso_axes(self) -> None:
        self.assertTrue(np.allclose(GSO_TO_ROBOTWIN[:3, :3] @ [1, 0, 0], [1, 0, 0]))
        self.assertTrue(np.allclose(GSO_TO_ROBOTWIN[:3, :3] @ [0, 0, 1], [0, 1, 0]))
        self.assertTrue(np.allclose(GSO_TO_ROBOTWIN[:3, :3] @ [0, -1, 0], [0, 0, 1]))
        self.assertAlmostEqual(np.linalg.det(GSO_TO_ROBOTWIN[:3, :3]), 1.0)

    def test_model_data_matches_reference_frame_and_target_length(self) -> None:
        mesh = trimesh.creation.box(extents=[0.09, 0.08, 0.27])
        mesh.apply_translation([0.0, 0.04, 0.0])
        data = build_model_data(mesh, self.reference, target_length_m=0.216)
        loaded = np.asarray(data["extents"]) * np.asarray(data["scale"])
        self.assertAlmostEqual(loaded[2], 0.216)
        self.assertTrue(
            np.allclose(
                np.asarray(data["functional_matrix"])[0, :3, :3],
                np.asarray(self.reference["functional_matrix"])[0, :3, :3],
            )
        )
        self.assertGreater(np.asarray(data["orientation_point"])[2, 3], 0.0)

    def test_convex_collision_passes_gate(self) -> None:
        mesh = trimesh.creation.box(extents=[0.09, 0.08, 0.27])
        data = build_model_data(mesh, self.reference, target_length_m=0.216)
        collision_scene, collision_parts = build_collision_scene(mesh)
        gate = asset_gate(mesh, collision_parts, data)
        # The synthetic box has too few visual faces, but its collision and
        # metric extent checks must independently pass.
        self.assertIn("visual_mesh_too_small", gate["reasons"])
        self.assertNotIn("collision_not_watertight", gate["reasons"])
        self.assertNotIn("loaded_length_out_of_range", gate["reasons"])
        self.assertGreaterEqual(gate["collision_parts"], 2)
        self.assertLessEqual(gate["collision_max_part_vertices"], 64)
        self.assertTrue(gate["flat_sole_support"])
        self.assertGreaterEqual(len(collision_scene.geometry), 2)
        self.assertGreaterEqual(gate["bottom_width_coverage"], 0.5)
        self.assertGreaterEqual(gate["bottom_length_coverage"], 0.6)


if __name__ == "__main__":
    unittest.main()
