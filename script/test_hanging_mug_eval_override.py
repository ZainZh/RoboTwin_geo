from ast import FunctionDef, Module, fix_missing_locations, parse
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import transforms3d as t3d
import yaml


ENV_SOURCE_PATH = Path("envs/hanging_mug.py")
BASE_CONFIG_PATH = Path("task_config/demo_clean_3d_object_pc.yml")
OVERRIDE_CONFIG_PATH = Path("task_config/demo_clean_3d_partnext_objpc_mug_eval.yml")


def load_helpers(*function_names):
    source = ENV_SOURCE_PATH.read_text(encoding="utf-8")
    tree = parse(source)
    func_nodes = [
        node for node in tree.body if isinstance(node, FunctionDef) and node.name in set(function_names)
    ]
    module = Module(body=func_nodes, type_ignores=[])
    fix_missing_locations(module)
    namespace = {
        "Path": Path,
        "json": json,
        "np": np,
        "t3d": t3d,
        "DEFAULT_MUG_SPAWN_QPOS": [0.707, 0.707, 0.0, 0.0],
    }
    exec(compile(module, filename=str(ENV_SOURCE_PATH), mode="exec"), namespace)
    return [namespace[name] for name in function_names]


class TestHangingMugEvalOverride(unittest.TestCase):
    def test_resolve_mug_asset_config_defaults(self):
        resolve_mug_asset_config, = load_helpers("resolve_mug_asset_config")

        self.assertEqual(
            resolve_mug_asset_config(None),
            {
                "modelname": "039_mug",
                "model_id": 0,
                "info_asset_path": "039_mug/base0",
            },
        )

    def test_resolve_mug_asset_config_custom_override(self):
        resolve_mug_asset_config, = load_helpers("resolve_mug_asset_config")

        self.assertEqual(
            resolve_mug_asset_config(
                {
                    "enabled": True,
                    "modelname": "partnext_mug_eval",
                    "model_id": 3,
                }
            ),
            {
                "modelname": "partnext_mug_eval",
                "model_id": 3,
                "info_asset_path": "partnext_mug_eval/base3",
            },
        )

    def test_resolve_mug_asset_config_round_robins_model_id_list(self):
        resolve_mug_asset_config, = load_helpers("resolve_mug_asset_config")

        self.assertEqual(
            resolve_mug_asset_config(
                {
                    "enabled": True,
                    "modelname": "partnext_mug_eval",
                    "model_id": [1, 4, 7],
                },
                episode_index=4,
            )["model_id"],
            4,
        )

    def test_validate_mug_asset_config_accepts_prepared_asset_layout(self):
        validate_mug_asset_config, = load_helpers("validate_mug_asset_config")

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            model_dir = repo_root / "assets" / "objects" / "partnext_mug_eval"
            (model_dir / "visual").mkdir(parents=True)
            (model_dir / "collision").mkdir(parents=True)
            (model_dir / "visual" / "base0.glb").write_bytes(b"visual")
            (model_dir / "collision" / "base0.glb").write_bytes(b"collision")
            (model_dir / "model_data0.json").write_text("{}", encoding="utf-8")

            validate_mug_asset_config(
                {
                    "modelname": "partnext_mug_eval",
                    "model_id": 0,
                    "info_asset_path": "partnext_mug_eval/base0",
                },
                repo_root=repo_root,
            )

    def test_resolve_mug_spawn_qpos_aligns_custom_bottom_frame_to_reference(self):
        (
            resolve_mug_spawn_qpos,
            load_scaled_local_pose_matrix,
            _,
        ) = load_helpers(
            "resolve_mug_spawn_qpos",
            "load_scaled_local_pose_matrix",
            "quat_multiply",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            ref_dir = repo_root / "assets" / "objects" / "039_mug"
            cur_dir = repo_root / "assets" / "objects" / "partnext_mug_eval"
            ref_dir.mkdir(parents=True)
            cur_dir.mkdir(parents=True)

            (ref_dir / "model_data0.json").write_text(
                json.dumps(
                    {
                        "scale": [1.0, 1.0, 1.0],
                        "contact_points_pose": [
                            [
                                [1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        ],
                        "functional_matrix": [
                            [
                                [1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ],
                            [
                                [1.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0],
                                [0.0, -1.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ],
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (cur_dir / "model_data0.json").write_text(
                json.dumps(
                    {
                        "scale": [1.0, 1.0, 1.0],
                        "contact_points_pose": [
                            [
                                [0.0, -1.0, 0.0, 0.1],
                                [1.0, 0.0, 0.0, 0.2],
                                [0.0, 0.0, 1.0, 0.3],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        ],
                        "functional_matrix": [
                            [
                                [1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ],
                            [
                                [0.0, -1.0, 0.0, 0.1],
                                [1.0, 0.0, 0.0, 0.2],
                                [0.0, 0.0, 1.0, 0.3],
                                [0.0, 0.0, 0.0, 1.0],
                            ],
                        ],
                    }
                ),
                encoding="utf-8",
            )

            spawn_qpos = resolve_mug_spawn_qpos(
                {
                    "modelname": "partnext_mug_eval",
                    "model_id": 0,
                    "info_asset_path": "partnext_mug_eval/base0",
                },
                custom_mug_eval={"enabled": True},
                repo_root=repo_root,
            )

            ref_root = t3d.quaternions.quat2mat(np.asarray([0.707, 0.707, 0.0, 0.0], dtype=np.float64))
            ref_local = load_scaled_local_pose_matrix("039_mug", 0, "functional_matrix", repo_root=repo_root, point_index=1)
            cur_local = load_scaled_local_pose_matrix("partnext_mug_eval", 0, "functional_matrix", repo_root=repo_root, point_index=1)
            cur_root = t3d.quaternions.quat2mat(np.asarray(spawn_qpos, dtype=np.float64))

            self.assertTrue(np.allclose(ref_root @ ref_local[:3, :3], cur_root @ cur_local[:3, :3], atol=1e-6))

    def test_env_source_wires_override_into_setup_load_info_and_validation(self):
        source = ENV_SOURCE_PATH.read_text(encoding="utf-8")

        self.assertIn('self.custom_mug_eval = kwags.get("custom_mug_eval")', source)
        self.assertIn('self.mug_asset_config = resolve_mug_asset_config(', source)
        self.assertIn('self.mug_spawn_qpos = resolve_mug_spawn_qpos(', source)
        self.assertIn('validate_mug_asset_config(self.mug_asset_config)', source)
        self.assertIn('modelname=self.mug_asset_config["modelname"]', source)
        self.assertIn('model_id=self.mug_asset_config["model_id"]', source)
        self.assertIn('self.mug_asset_config["info_asset_path"]', source)

    def test_override_config_matches_baseline_plus_custom_mug_block(self):
        baseline_config = yaml.safe_load(BASE_CONFIG_PATH.read_text(encoding="utf-8"))
        override_config = yaml.safe_load(OVERRIDE_CONFIG_PATH.read_text(encoding="utf-8"))

        custom_mug_eval = override_config.pop("custom_mug_eval")

        self.assertTrue(custom_mug_eval["enabled"])
        self.assertTrue(str(custom_mug_eval["modelname"]).startswith("partnext_mug_eval"))
        self.assertIn("model_id", custom_mug_eval)
        for key, value in baseline_config.items():
            if key == "render_freq":
                continue
            self.assertEqual(override_config[key], value)


if __name__ == "__main__":
    unittest.main()
