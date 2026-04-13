from ast import FunctionDef, Module, fix_missing_locations, parse
from pathlib import Path
import tempfile
import unittest

import yaml


ENV_SOURCE_PATH = Path("envs/beat_block_hammer.py")
BASE_CONFIG_PATH = Path("task_config/demo_clean_3d_object_pc.yml")
OVERRIDE_CONFIG_PATH = Path("task_config/demo_clean_3d_partnext_hammer_eval.yml")


def load_helpers(*function_names):
    source = ENV_SOURCE_PATH.read_text(encoding="utf-8")
    tree = parse(source)
    func_nodes = [
        node for node in tree.body if isinstance(node, FunctionDef) and node.name in set(function_names)
    ]
    module = Module(body=func_nodes, type_ignores=[])
    fix_missing_locations(module)
    namespace = {"Path": Path}
    exec(compile(module, filename=str(ENV_SOURCE_PATH), mode="exec"), namespace)
    return [namespace[name] for name in function_names]


class TestBeatBlockHammerEvalOverride(unittest.TestCase):
    def test_resolve_hammer_asset_config_defaults(self):
        resolve_hammer_asset_config, = load_helpers("resolve_hammer_asset_config")

        self.assertEqual(
            resolve_hammer_asset_config(None),
            {
                "modelname": "020_hammer",
                "model_id": 0,
                "info_asset_path": "020_hammer/base0",
            },
        )

    def test_resolve_hammer_asset_config_custom_override(self):
        resolve_hammer_asset_config, = load_helpers("resolve_hammer_asset_config")

        self.assertEqual(
            resolve_hammer_asset_config(
                {
                    "enabled": True,
                    "modelname": "partnext_hammer_eval",
                    "model_id": 0,
                }
            ),
            {
                "modelname": "partnext_hammer_eval",
                "model_id": 0,
                "info_asset_path": "partnext_hammer_eval/base0",
            },
        )

    def test_resolve_hammer_asset_config_round_robins_model_id_list(self):
        resolve_hammer_asset_config, = load_helpers("resolve_hammer_asset_config")

        self.assertEqual(
            resolve_hammer_asset_config(
                {
                    "enabled": True,
                    "modelname": "partnext_hammer_eval",
                    "model_id": [1, 3, 7],
                },
                episode_index=0,
            ),
            {
                "modelname": "partnext_hammer_eval",
                "model_id": 1,
                "info_asset_path": "partnext_hammer_eval/base1",
            },
        )
        self.assertEqual(
            resolve_hammer_asset_config(
                {
                    "enabled": True,
                    "modelname": "partnext_hammer_eval",
                    "model_id": [1, 3, 7],
                },
                episode_index=4,
            )["model_id"],
            3,
        )

    def test_validate_hammer_asset_config_accepts_prepared_asset_layout(self):
        validate_hammer_asset_config, = load_helpers("validate_hammer_asset_config")

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            model_dir = repo_root / "assets" / "objects" / "partnext_hammer_eval"
            (model_dir / "visual").mkdir(parents=True)
            (model_dir / "collision").mkdir(parents=True)
            (model_dir / "visual" / "base0.glb").write_bytes(b"visual")
            (model_dir / "collision" / "base0.glb").write_bytes(b"collision")
            (model_dir / "model_data0.json").write_text("{}", encoding="utf-8")

            validate_hammer_asset_config(
                {
                    "modelname": "partnext_hammer_eval",
                    "model_id": 0,
                    "info_asset_path": "partnext_hammer_eval/base0",
                },
                repo_root=repo_root,
            )

    def test_validate_hammer_asset_config_raises_actionable_error_for_missing_asset(self):
        validate_hammer_asset_config, = load_helpers("validate_hammer_asset_config")

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(FileNotFoundError, "prepare_partnext_hammer_eval_asset.py"):
                validate_hammer_asset_config(
                    {
                        "modelname": "partnext_hammer_eval",
                        "model_id": 0,
                        "info_asset_path": "partnext_hammer_eval/base0",
                    },
                    repo_root=Path(tmpdir),
                )

    def test_env_source_wires_override_into_setup_load_info_and_validation(self):
        source = ENV_SOURCE_PATH.read_text(encoding="utf-8")

        self.assertIn('self.custom_hammer_eval = kwags.get("custom_hammer_eval")', source)
        self.assertIn(
            'self.hammer_asset_config = resolve_hammer_asset_config(',
            source,
        )
        self.assertIn('episode_index=kwags.get("now_ep_num", 0)', source)
        self.assertIn('validate_hammer_asset_config(self.hammer_asset_config)', source)
        self.assertIn('modelname=self.hammer_asset_config["modelname"]', source)
        self.assertIn('model_id=self.hammer_asset_config["model_id"]', source)
        self.assertIn('self.hammer_asset_config["info_asset_path"]', source)

    def test_override_config_matches_baseline_plus_custom_hammer_block(self):
        baseline_config = yaml.safe_load(BASE_CONFIG_PATH.read_text(encoding="utf-8"))
        override_config = yaml.safe_load(OVERRIDE_CONFIG_PATH.read_text(encoding="utf-8"))

        custom_hammer_eval = override_config.pop("custom_hammer_eval")

        self.assertEqual(
            custom_hammer_eval,
            {
                "enabled": True,
                "modelname": "partnext_hammer_eval",
                "model_id": 0,
            },
        )
        self.assertEqual(override_config, baseline_config)


if __name__ == "__main__":
    unittest.main()
