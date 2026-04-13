import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "script"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import prepare_partnext_hammer_eval_asset as prepare


class TestPreparePartNextHammerEvalAsset(unittest.TestCase):
    def test_make_batch_output_modelname_appends_sanitized_glb_stem(self):
        self.assertEqual(
            prepare.make_batch_output_modelname("partnext_hammer_eval", "Foo-Bar.glb"),
            "partnext_hammer_eval_foo_bar",
        )

    def test_prepare_asset_packages_all_iterates_all_glbs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            partnext_dir = root / "partnext"
            partnext_dir.mkdir()
            (partnext_dir / "b.glb").write_bytes(b"")
            (partnext_dir / "a.glb").write_bytes(b"")
            called = []

            def fake_prepare_asset_package(**kwargs):
                called.append((kwargs["glb_name"], kwargs["output_modelname"]))
                return {
                    "asset_dir": str(root / kwargs["output_modelname"]),
                    "selected_glb": kwargs["glb_name"],
                    "model_id": Path(kwargs["glb_name"]).stem,
                    "scale": [1.0, 1.0, 1.0],
                }

            summaries = prepare.prepare_asset_packages(
                partnext_dir=partnext_dir,
                annotation_path=root / "annotation.jsonl",
                output_modelname="partnext_hammer_eval",
                output_root=root / "assets",
                reference_model_data=None,
                glb_name=None,
                prepare_all=True,
                single_prepare_fn=fake_prepare_asset_package,
            )

        self.assertEqual(
            called,
            [
                ("a.glb", "partnext_hammer_eval_a"),
                ("b.glb", "partnext_hammer_eval_b"),
            ],
        )
        self.assertEqual([item["selected_glb"] for item in summaries], ["a.glb", "b.glb"])


if __name__ == "__main__":
    unittest.main()
