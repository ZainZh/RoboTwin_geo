import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "script"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from partnext_mug_eval_utils import build_partnext_mug_asset


class TestPartNextMugEvalUtils(unittest.TestCase):
    def test_build_partnext_mug_asset_honors_requested_glb(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            partnext_dir = root / "partnext"
            partnext_dir.mkdir()

            scene_a = trimesh.Scene()
            scene_a.add_geometry(trimesh.creation.cylinder(radius=0.4, height=1.0), geom_name="a.glb_0")
            scene_a.add_geometry(trimesh.creation.box(extents=(0.2, 0.5, 0.1)), geom_name="a.glb_1")
            scene_a.export(partnext_dir / "a.glb")

            scene_b = trimesh.Scene()
            scene_b.add_geometry(trimesh.creation.cylinder(radius=0.5, height=1.2), geom_name="b.glb_0")
            scene_b.add_geometry(trimesh.creation.box(extents=(0.25, 0.55, 0.1)), geom_name="b.glb_1")
            scene_b.export(partnext_dir / "b.glb")

            annotation_path = root / "annotation.jsonl"
            rows = []
            for name in ("a.glb", "b.glb"):
                rows.append(
                    {
                        "glb_dst": name,
                        "model_id": Path(name).stem,
                        "object_name": "Mug",
                        "hierarchyList": json.dumps(
                            [
                                {
                                    "name": "Mug",
                                    "children": [
                                        {"name": "Body", "children": [{"name": "Main Body", "maskId": 0}]},
                                        {"name": "Handle", "maskId": 1},
                                    ],
                                }
                            ]
                        ),
                        "masks": json.dumps({"0": {"0": list(range(128))}, "1": {"1": list(range(12))}}),
                    }
                )
            annotation_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            reference_model_data_path = root / "model_data0.json"
            reference_model_data_path.write_text(
                json.dumps(
                    {
                        "extents": [2.0, 1.4, 2.2],
                        "scale": [0.08, 0.08, 0.08],
                        "stable": False,
                    }
                ),
                encoding="utf-8",
            )

            prepared_asset = build_partnext_mug_asset(
                partnext_dir=partnext_dir,
                annotation_path=annotation_path,
                output_modelname="partnext_mug_eval",
                reference_model_data_path=reference_model_data_path,
                requested_glb_name="b.glb",
            )

            self.assertEqual(prepared_asset.visual_glb_path, partnext_dir / "b.glb")
            self.assertEqual(prepared_asset.source_meta["glb_dst"], "b.glb")

    def test_build_partnext_mug_asset_generates_contacts_functions_and_split_collision(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            partnext_dir = root / "partnext"
            partnext_dir.mkdir()

            body = trimesh.creation.cylinder(radius=0.45, height=1.2, sections=32)
            body.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2.0, [1, 0, 0]))
            handle = trimesh.creation.box(extents=(0.15, 0.45, 0.5))
            handle.apply_translation([0.55, 0.0, 0.0])
            scene = trimesh.Scene()
            scene.add_geometry(body, geom_name="candidate.glb_0")
            scene.add_geometry(handle, geom_name="candidate.glb_1")
            candidate_glb = partnext_dir / "candidate.glb"
            scene.export(candidate_glb)

            annotation_path = root / "annotation.jsonl"
            annotation_path.write_text(
                json.dumps(
                    {
                        "glb_dst": "candidate.glb",
                        "model_id": "candidate",
                        "object_name": "Mug",
                        "hierarchyList": json.dumps(
                            [
                                {
                                    "name": "Mug",
                                    "children": [
                                        {"name": "Body", "children": [{"name": "Main Body", "maskId": 0}]},
                                        {"name": "Handle", "maskId": 1},
                                    ],
                                }
                            ]
                        ),
                        "masks": json.dumps({"0": {"0": list(range(len(body.faces)))}, "1": {"1": list(range(len(handle.faces)))}}),
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            reference_model_data_path = root / "model_data0.json"
            reference_model_data_path.write_text(
                json.dumps(
                    {
                        "extents": [2.0, 1.4, 2.2],
                        "scale": [0.08, 0.08, 0.08],
                        "stable": False,
                    }
                ),
                encoding="utf-8",
            )

            prepared_asset = build_partnext_mug_asset(
                partnext_dir=partnext_dir,
                annotation_path=annotation_path,
                output_modelname="partnext_mug_eval",
                reference_model_data_path=reference_model_data_path,
                requested_glb_name="candidate.glb",
            )

            self.assertGreaterEqual(len(prepared_asset.model_data["contact_points_pose"]), 4)
            self.assertEqual(len(prepared_asset.model_data["functional_matrix"]), 2)
            self.assertNotEqual(prepared_asset.collision_glb_path, prepared_asset.visual_glb_path)

            collision_scene = trimesh.load(prepared_asset.collision_glb_path, force="scene")
            self.assertGreaterEqual(len(collision_scene.geometry), 2)

            hanging_fp = np.asarray(prepared_asset.model_data["functional_matrix"][0], dtype=np.float64)
            bottom_fp = np.asarray(prepared_asset.model_data["functional_matrix"][1], dtype=np.float64)
            self.assertGreater(np.linalg.norm(hanging_fp[:3, 3] - bottom_fp[:3, 3]), 1e-3)


if __name__ == "__main__":
    unittest.main()
