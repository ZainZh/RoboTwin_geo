import json
import sys
import tempfile
import unittest
from pathlib import Path

import trimesh

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "script"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import prepare_partnext_mug_eval_asset as prepare


class TestPreparePartNextMugEvalAsset(unittest.TestCase):
    def test_prepare_asset_packages_all_writes_single_robotwin_asset_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            partnext_dir = root / "partnext"
            partnext_dir.mkdir()
            for name in ("b.glb", "a.glb"):
                scene = trimesh.Scene()
                scene.add_geometry(trimesh.creation.cylinder(radius=0.4, height=1.0), geom_name=f"{name}_0")
                scene.add_geometry(trimesh.creation.box(extents=(0.2, 0.5, 0.1)), geom_name=f"{name}_1")
                scene.export(partnext_dir / name)

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
                        "masks": json.dumps({"0": {"0": list(range(64))}, "1": {"1": list(range(12))}}),
                    }
                )
            annotation_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            reference_model_data = root / "model_data0.json"
            reference_model_data.write_text(
                json.dumps(
                    {
                        "extents": [2.0, 1.4, 2.2],
                        "scale": [0.08, 0.08, 0.08],
                        "stable": False,
                    }
                ),
                encoding="utf-8",
            )
            output_root = root / "assets" / "objects"
            description_root = root / "description" / "objects_description"

            summaries = prepare.prepare_asset_packages(
                partnext_dir=partnext_dir,
                annotation_path=annotation_path,
                output_modelname="partnext_mug_eval",
                output_root=output_root,
                reference_model_data=reference_model_data,
                glb_name=None,
                prepare_all=True,
                description_root=description_root,
            )

            asset_dir = output_root / "partnext_mug_eval"
            self.assertTrue((asset_dir / "visual" / "base0.glb").is_file())
            self.assertTrue((asset_dir / "visual" / "base1.glb").is_file())
            self.assertTrue((asset_dir / "collision" / "base0.glb").is_file())
            self.assertTrue((asset_dir / "collision" / "base1.glb").is_file())
            self.assertTrue((asset_dir / "model_data0.json").is_file())
            self.assertTrue((asset_dir / "model_data1.json").is_file())
            self.assertTrue((asset_dir / "points_info.json").is_file())
            self.assertTrue((asset_dir / "preview" / "overview0.ply").is_file())
            self.assertTrue((asset_dir / "preview" / "overview1.ply").is_file())
            self.assertTrue((description_root / "partnext_mug_eval" / "base0.json").is_file())
            self.assertTrue((description_root / "partnext_mug_eval" / "base1.json").is_file())
            self.assertEqual([item["selected_glb"] for item in summaries], ["a.glb", "b.glb"])
            self.assertEqual([item["model_id"] for item in summaries], [0, 1])

    def test_prepare_asset_package_writes_object_description_for_single_asset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            partnext_dir = root / "partnext"
            partnext_dir.mkdir()
            scene = trimesh.Scene()
            scene.add_geometry(trimesh.creation.cylinder(radius=0.4, height=1.0), geom_name="a.glb_0")
            scene.add_geometry(trimesh.creation.box(extents=(0.2, 0.5, 0.1)), geom_name="a.glb_1")
            scene.export(partnext_dir / "a.glb")

            annotation_path = root / "annotation.jsonl"
            annotation_path.write_text(
                json.dumps(
                    {
                        "glb_dst": "a.glb",
                        "model_id": "a",
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
                        "masks": json.dumps({"0": {"0": list(range(64))}, "1": {"1": list(range(12))}}),
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            reference_model_data = root / "model_data0.json"
            reference_model_data.write_text(
                json.dumps(
                    {
                        "extents": [2.0, 1.4, 2.2],
                        "scale": [0.08, 0.08, 0.08],
                        "stable": False,
                    }
                ),
                encoding="utf-8",
            )
            output_root = root / "assets" / "objects"
            description_root = root / "description" / "objects_description"

            prepare.prepare_asset_packages(
                partnext_dir=partnext_dir,
                annotation_path=annotation_path,
                output_modelname="partnext_mug_eval",
                output_root=output_root,
                reference_model_data=reference_model_data,
                glb_name=None,
                prepare_all=False,
                description_root=description_root,
            )

            description_path = description_root / "partnext_mug_eval" / "base0.json"
            self.assertTrue(description_path.is_file())
            description_data = json.loads(description_path.read_text(encoding="utf-8"))
            self.assertEqual(description_data["raw_description"], "mug")
            self.assertTrue(description_data["seen"])
            self.assertTrue(description_data["unseen"])


if __name__ == "__main__":
    unittest.main()
