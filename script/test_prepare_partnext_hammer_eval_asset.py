import json
import sys
import tempfile
import unittest
from pathlib import Path

import trimesh

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "script"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import prepare_partnext_hammer_eval_asset as prepare


class TestPreparePartNextHammerEvalAsset(unittest.TestCase):
    def test_prepare_asset_packages_all_writes_single_robotwin_asset_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            partnext_dir = root / "partnext"
            partnext_dir.mkdir()
            for name in ("b.glb", "a.glb"):
                trimesh.creation.box(extents=(0.1, 0.4, 0.12)).export(partnext_dir / name)

            annotation_path = root / "annotation.jsonl"
            rows = []
            for name in ("a.glb", "b.glb"):
                rows.append(
                    {
                        "glb_dst": name,
                        "model_id": Path(name).stem,
                        "hierarchyList": json.dumps(
                            [
                                {
                                    "name": "Hammer",
                                    "children": [
                                        {
                                            "name": "Head",
                                            "children": [{"name": "Hammer Head", "maskId": 0}],
                                        },
                                        {
                                            "name": "Handle",
                                            "children": [{"name": "Grip", "maskId": 3}],
                                        },
                                    ],
                                }
                            ]
                        ),
                        "masks": json.dumps({"0": {"0": [0, 1, 2, 3]}, "3": {"0": [4, 5, 6, 7]}}),
                    }
                )
            annotation_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            reference_model_data = root / "model_data0.json"
            reference_model_data.write_text(
                json.dumps(
                    {
                        "extents": [0.1, 0.4, 0.12],
                        "scale": [1.0, 1.0, 1.0],
                        "stable": False,
                    }
                ),
                encoding="utf-8",
            )
            output_root = root / "assets" / "objects"

            summaries = prepare.prepare_asset_packages(
                partnext_dir=partnext_dir,
                annotation_path=annotation_path,
                output_modelname="partnext_hammer_eval",
                output_root=output_root,
                reference_model_data=reference_model_data,
                glb_name=None,
                prepare_all=True,
            )

            asset_dir = output_root / "partnext_hammer_eval"
            self.assertTrue((asset_dir / "visual" / "base0.glb").is_file())
            self.assertTrue((asset_dir / "visual" / "base1.glb").is_file())
            self.assertTrue((asset_dir / "collision" / "base0.glb").is_file())
            self.assertTrue((asset_dir / "collision" / "base1.glb").is_file())
            self.assertTrue((asset_dir / "model_data0.json").is_file())
            self.assertTrue((asset_dir / "model_data1.json").is_file())
            self.assertTrue((asset_dir / "points_info.json").is_file())
            self.assertFalse((output_root / "partnext_hammer_eval_a").exists())
            self.assertEqual([item["selected_glb"] for item in summaries], ["a.glb", "b.glb"])
            self.assertEqual([item["model_id"] for item in summaries], [0, 1])

    def test_prepare_asset_packages_all_rejects_existing_output_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            partnext_dir = root / "partnext"
            partnext_dir.mkdir()
            (partnext_dir / "a.glb").write_bytes(b"")
            output_root = root / "assets" / "objects"
            (output_root / "partnext_hammer_eval").mkdir(parents=True)

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                prepare.prepare_asset_packages(
                    partnext_dir=partnext_dir,
                    annotation_path=root / "annotation.jsonl",
                    output_modelname="partnext_hammer_eval",
                    output_root=output_root,
                    reference_model_data=None,
                    glb_name=None,
                    prepare_all=True,
                )


if __name__ == "__main__":
    unittest.main()
