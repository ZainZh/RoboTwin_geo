from ast import FunctionDef, Module, fix_missing_locations, parse
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import transforms3d as t3d
import yaml


ENV_SOURCE_PATH = Path("envs/beat_block_hammer.py")
BASE_CONFIG_PATH = Path("task_config/demo_clean_3d_object_pc.yml")
OVERRIDE_CONFIG_PATH = Path("task_config/demo_clean_3d_partnext_objpc_hammer_eval.yml")


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
        "DEFAULT_HAMMER_SPAWN_POSITION": [0.0, -0.06, 0.783],
        "DEFAULT_HAMMER_SPAWN_QUATERNION": [0.0, 0.0, 0.995, 0.105],
    }
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

    def test_resolve_hammer_spawn_pose_defaults_to_original_pose(self):
        resolve_hammer_spawn_pose, = load_helpers("resolve_hammer_spawn_pose")

        spawn_pose = resolve_hammer_spawn_pose(
            {
                "modelname": "020_hammer",
                "model_id": 0,
                "info_asset_path": "020_hammer/base0",
            },
            custom_hammer_eval=None,
        )

        self.assertEqual(spawn_pose["position"], [0.0, -0.06, 0.783])
        self.assertEqual(spawn_pose["quaternion"], [0.0, 0.0, 0.995, 0.105])

    def test_resolve_hammer_spawn_pose_aligns_custom_contact_frame_to_reference(self):
        (
            _,
            resolve_hammer_spawn_pose,
            _,
            load_scaled_local_pose_matrix,
            _,
        ) = load_helpers(
            "resolve_hammer_asset_config",
            "resolve_hammer_spawn_pose",
            "validate_hammer_asset_config",
            "load_scaled_local_pose_matrix",
            "pose_to_matrix",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            ref_dir = repo_root / "assets" / "objects" / "020_hammer"
            cur_dir = repo_root / "assets" / "objects" / "partnext_hammer_eval"
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
                    }
                ),
                encoding="utf-8",
            )

            spawn_pose = resolve_hammer_spawn_pose(
                {
                    "modelname": "partnext_hammer_eval",
                    "model_id": 0,
                    "info_asset_path": "partnext_hammer_eval/base0",
                },
                custom_hammer_eval={"enabled": True},
                repo_root=repo_root,
            )

            world_root = np.eye(4, dtype=np.float64)
            world_root[:3, :3] = t3d.quaternions.quat2mat(np.asarray(spawn_pose["quaternion"], dtype=np.float64))
            world_root[:3, 3] = np.asarray(spawn_pose["position"], dtype=np.float64)

            ref_spawn = np.eye(4, dtype=np.float64)
            ref_spawn[:3, :3] = t3d.quaternions.quat2mat(np.asarray([0.0, 0.0, 0.995, 0.105], dtype=np.float64))
            ref_spawn[:3, 3] = np.asarray([0.0, -0.06, 0.783], dtype=np.float64)

            ref_local = load_scaled_local_pose_matrix("020_hammer", 0, "contact_points_pose", repo_root=repo_root)
            cur_local = load_scaled_local_pose_matrix(
                "partnext_hammer_eval", 0, "contact_points_pose", repo_root=repo_root
            )

            self.assertTrue(np.allclose(world_root @ cur_local, ref_spawn @ ref_local, atol=1e-6))

    def test_env_source_wires_override_into_setup_load_info_and_validation(self):
        source = ENV_SOURCE_PATH.read_text(encoding="utf-8")

        self.assertIn('self.custom_hammer_eval = kwags.get("custom_hammer_eval")', source)
        self.assertIn(
            'self.hammer_asset_config = resolve_hammer_asset_config(',
            source,
        )
        self.assertIn('self.hammer_spawn_pose = resolve_hammer_spawn_pose(', source)
        self.assertIn('episode_index=kwags.get("now_ep_num", 0)', source)
        self.assertIn('validate_hammer_asset_config(self.hammer_asset_config)', source)
        self.assertIn('modelname=self.hammer_asset_config["modelname"]', source)
        self.assertIn('model_id=self.hammer_asset_config["model_id"]', source)
        self.assertIn('self.hammer_asset_config["info_asset_path"]', source)

    def test_override_config_matches_baseline_plus_custom_hammer_block(self):
        baseline_config = yaml.safe_load(BASE_CONFIG_PATH.read_text(encoding="utf-8"))
        override_config = yaml.safe_load(OVERRIDE_CONFIG_PATH.read_text(encoding="utf-8"))

        custom_hammer_eval = override_config.pop("custom_hammer_eval")

        self.assertTrue(custom_hammer_eval["enabled"])
        self.assertTrue(str(custom_hammer_eval["modelname"]).startswith("partnext_hammer_eval"))
        self.assertIn("model_id", custom_hammer_eval)
        for key, value in baseline_config.items():
            if key == "render_freq":
                continue
            self.assertEqual(override_config[key], value)


if __name__ == "__main__":
    unittest.main()
