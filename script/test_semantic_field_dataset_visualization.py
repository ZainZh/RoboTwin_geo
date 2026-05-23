import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import torch

import script.visualize_semantic_field_on_dataset as viz
from script.visualize_semantic_field_on_dataset import (
    apply_shared_pca_to_rgb,
    build_label_palette,
    compose_scene_semantic_overlay,
    fit_shared_pca_projection,
    group_records_for_pca,
    labels_to_rgb,
    load_scene_cloud_for_frame,
    overlay_semantic_colors_on_scene,
    parse_episode_frame_specs,
    prepare_semantic_input_cloud,
    run_visualization,
    write_colored_ply,
)


def _read_ascii_ply_colors(path: Path) -> np.ndarray:
    lines = path.read_text(encoding="utf-8").splitlines()
    start = lines.index("end_header") + 1
    colors = []
    for line in lines[start:]:
        parts = line.split()
        if len(parts) >= 6:
            colors.append([int(parts[3]), int(parts[4]), int(parts[5])])
    return np.asarray(colors, dtype=np.uint8)


class SemanticFieldDatasetVisualizationTest(unittest.TestCase):
    def test_parse_episode_frame_specs_uses_explicit_pairs(self):
        specs = parse_episode_frame_specs(
            episode=0,
            frame=0,
            episode_frames=["0:10", "2:5"],
        )

        self.assertEqual(specs, [(0, 10), (2, 5)])

    def test_parse_episode_frame_specs_accepts_comma_separated_values(self):
        specs = parse_episode_frame_specs(
            episode=0,
            frame=0,
            episode_frames=["0:10,0:20", "1:5"],
        )

        self.assertEqual(specs, [(0, 10), (0, 20), (1, 5)])

    def test_parse_episode_frame_specs_accepts_string_default(self):
        specs = parse_episode_frame_specs(
            episode=0,
            frame=0,
            episode_frames="0:10,0:20,1:5",
        )

        self.assertEqual(specs, [(0, 10), (0, 20), (1, 5)])

    def test_parse_episode_frame_specs_rejects_bad_pair(self):
        with self.assertRaises(ValueError):
            parse_episode_frame_specs(
                episode=0,
                frame=0,
                episode_frames=["bad"],
            )

    def test_shared_pca_colors_multiple_sets_in_one_space(self):
        embeddings_a = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        embeddings_b = np.asarray(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        projection = fit_shared_pca_projection([embeddings_a, embeddings_b])
        colors_a = apply_shared_pca_to_rgb(embeddings_a, projection)
        colors_b = apply_shared_pca_to_rgb(embeddings_b, projection)

        self.assertEqual(colors_a.shape, (2, 3))
        self.assertEqual(colors_b.shape, (2, 3))
        self.assertEqual(colors_a.dtype, np.uint8)
        self.assertTrue(np.any(colors_a[0] != colors_b[0]))

    def test_shared_pca_matches_utonia_universal_field_joint_pca(self):
        embeddings = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [2.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )

        def reference_projection(embedding_sets):
            all_embeddings = np.concatenate(embedding_sets, axis=0).astype(np.float32, copy=False)
            mean = all_embeddings.mean(axis=0, keepdims=True)
            centered = all_embeddings - mean
            projection_dim = min(3, centered.shape[1]) if centered.ndim == 2 else 1
            components = np.zeros((centered.shape[1], 3), dtype=np.float32)
            if centered.shape[0] >= 3 and centered.shape[1] >= 2:
                matrix = torch.from_numpy(centered)
                _, _, basis = torch.pca_lowrank(matrix, q=projection_dim)
                basis_np = basis[:, :projection_dim].numpy().astype(np.float32, copy=False)
                components[:, :projection_dim] = basis_np
            else:
                components[:projection_dim, :projection_dim] = np.eye(projection_dim, dtype=np.float32)
            projected = centered @ components
            projected_min = projected.min(axis=0, keepdims=True)
            projected_max = projected.max(axis=0, keepdims=True)
            span = projected_max - projected_min
            span[span < 1e-8] = 1.0
            return {
                "mean": mean.astype(np.float32, copy=False),
                "components": components.astype(np.float32, copy=False),
                "projected_min": projected_min.astype(np.float32, copy=False),
                "projected_span": span.astype(np.float32, copy=False),
                "embedding_dim": int(all_embeddings.shape[1]),
                "num_points": int(all_embeddings.shape[0]),
            }

        torch.manual_seed(0)
        expected = apply_shared_pca_to_rgb(embeddings, reference_projection([embeddings]))
        torch.manual_seed(0)
        actual = apply_shared_pca_to_rgb(embeddings, fit_shared_pca_projection([embeddings]))

        np.testing.assert_array_equal(actual, expected)

    def test_group_records_for_pca_defaults_to_checkpoint_identity(self):
        records = [
            {"placeholder": "{A}", "semantic_checkpoint": "/models/mug.pt"},
            {"placeholder": "{C}", "semantic_checkpoint": "/models/mug.pt"},
            {"placeholder": "{B}", "semantic_checkpoint": "/models/box.pt"},
        ]

        grouped = group_records_for_pca(records, shared_pca_scope="checkpoint")

        self.assertEqual(sorted(len(items) for items in grouped.values()), [1, 2])

    def test_labels_to_rgb_uses_shared_palette(self):
        palette = build_label_palette(3)
        labels = np.asarray([0, 2, -1, 3], dtype=np.int64)

        colors = labels_to_rgb(labels, palette)

        np.testing.assert_array_equal(colors[0], palette[0])
        np.testing.assert_array_equal(colors[1], palette[2])
        np.testing.assert_array_equal(colors[2], np.asarray([128, 128, 128], dtype=np.uint8))
        np.testing.assert_array_equal(colors[3], np.asarray([128, 128, 128], dtype=np.uint8))

    def test_prepare_semantic_input_cloud_can_match_debug_placeholder_colors(self):
        cloud = np.asarray(
            [
                [0.0, 0.0, 0.0, 0.6, 0.2, 0.1],
                [1.0, 0.0, 0.0, 0.7, 0.3, 0.2],
            ],
            dtype=np.float32,
        )

        prepared = prepare_semantic_input_cloud(cloud, placeholder="{A}", color_mode="debug_placeholder")

        np.testing.assert_array_equal(prepared[:, 3:6], np.asarray([[255.0, 48.0, 48.0], [255.0, 48.0, 48.0]], dtype=np.float32))

    def test_prepare_semantic_input_cloud_scales_stored_rgb(self):
        cloud = np.asarray([[0.0, 0.0, 0.0, 0.5, 0.25, 0.0]], dtype=np.float32)

        prepared = prepare_semantic_input_cloud(cloud, placeholder="{A}", color_mode="stored_scaled")

        np.testing.assert_allclose(prepared[:, 3:6], np.asarray([[127.5, 63.75, 0.0]], dtype=np.float32))

    def test_overlay_semantic_colors_replaces_only_near_scene_points(self):
        scene = np.asarray(
            [
                [0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
                [1.0, 0.0, 0.0, 20.0, 20.0, 20.0],
                [5.0, 0.0, 0.0, 30.0, 30.0, 30.0],
            ],
            dtype=np.float32,
        )
        object_xyz = np.asarray(
            [
                [0.01, 0.0, 0.0],
                [1.01, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        object_rgb = np.asarray(
            [
                [255, 0, 0],
                [0, 255, 0],
            ],
            dtype=np.uint8,
        )

        overlay = overlay_semantic_colors_on_scene(
            scene,
            object_xyz,
            object_rgb,
            max_distance=0.05,
        )

        np.testing.assert_array_equal(overlay[:2, 3:6].astype(np.uint8), object_rgb)
        np.testing.assert_array_equal(overlay[2, 3:6].astype(np.uint8), np.asarray([30, 30, 30], dtype=np.uint8))

    def test_compose_scene_semantic_overlay_append_preserves_scene_colors(self):
        scene = np.asarray([[0.0, 0.0, 0.0, 10.0, 10.0, 10.0]], dtype=np.float32)
        object_xyz = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
        object_rgb = np.asarray([[255, 0, 0]], dtype=np.uint8)

        overlay = compose_scene_semantic_overlay(
            scene,
            object_xyz,
            object_rgb,
            mode="append",
            max_distance=0.01,
        )

        self.assertEqual(overlay.shape, (2, 6))
        np.testing.assert_array_equal(overlay[0, 3:6].astype(np.uint8), np.asarray([10, 10, 10], dtype=np.uint8))
        np.testing.assert_array_equal(overlay[1, 3:6].astype(np.uint8), np.asarray([255, 0, 0], dtype=np.uint8))

    def test_compose_scene_semantic_overlay_cut_replace_removes_matching_scene_points(self):
        scene = np.asarray(
            [
                [0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
                [1.0, 0.0, 0.0, 20.0, 20.0, 20.0],
                [5.0, 0.0, 0.0, 30.0, 30.0, 30.0],
            ],
            dtype=np.float32,
        )
        object_xyz = np.asarray([[0.01, 0.0, 0.0], [1.01, 0.0, 0.0]], dtype=np.float32)
        object_rgb = np.asarray([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)

        overlay = compose_scene_semantic_overlay(
            scene,
            object_xyz,
            object_rgb,
            mode="cut_replace",
            max_distance=0.05,
        )

        self.assertEqual(overlay.shape, (3, 6))
        np.testing.assert_array_equal(overlay[0, 3:6].astype(np.uint8), np.asarray([30, 30, 30], dtype=np.uint8))
        np.testing.assert_array_equal(overlay[1:, 3:6].astype(np.uint8), object_rgb)

    def test_compose_scene_semantic_overlay_cut_replace_respects_z_min(self):
        scene = np.asarray(
            [
                [0.0, 0.0, 0.000, 10.0, 10.0, 10.0],
                [0.0, 0.0, 0.030, 20.0, 20.0, 20.0],
                [1.0, 0.0, 0.100, 30.0, 30.0, 30.0],
            ],
            dtype=np.float32,
        )
        object_xyz = np.asarray([[0.0, 0.0, 0.020]], dtype=np.float32)
        object_rgb = np.asarray([[255, 0, 0]], dtype=np.uint8)

        overlay = compose_scene_semantic_overlay(
            scene,
            object_xyz,
            object_rgb,
            mode="cut_replace",
            max_distance=0.05,
            cut_z_min=0.015,
        )

        self.assertEqual(overlay.shape, (3, 6))
        np.testing.assert_array_equal(overlay[0, :3], scene[0, :3])
        np.testing.assert_array_equal(overlay[0, 3:6].astype(np.uint8), np.asarray([10, 10, 10], dtype=np.uint8))
        np.testing.assert_array_equal(overlay[1, :3], scene[2, :3])
        np.testing.assert_array_equal(overlay[2, 3:6].astype(np.uint8), object_rgb[0])

    def test_overlay_semantic_colors_can_require_local_object_density(self):
        scene = np.asarray(
            [
                [0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
                [0.018, 0.0, 0.0, 20.0, 20.0, 20.0],
            ],
            dtype=np.float32,
        )
        object_xyz = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        object_rgb = np.asarray([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)

        overlay = overlay_semantic_colors_on_scene(
            scene,
            object_xyz,
            object_rgb,
            max_distance=0.02,
            min_neighbors=2,
        )

        np.testing.assert_array_equal(overlay[:, 3:6].astype(np.uint8), scene[:, 3:6].astype(np.uint8))

    def test_load_scene_cloud_auto_prefers_raw_full_when_available(self):
        hdf5_episode = {
            "pointcloud": np.asarray(
                [
                    [
                        [0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
                    ]
                ],
                dtype=np.float32,
            )
        }
        raw_scene = np.asarray(
            [
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        args = Namespace(
            scene_source="auto",
            scene_point_num=0,
            camera_labels="",
            min_depth_m=0.05,
            max_depth_m=3.0,
            raw_intrinsics_source="frame",
            disable_serial_remap=False,
        )

        scene, source = load_scene_cloud_for_frame(
            dataset_dir=Path("/tmp/nonexistent"),
            episodes={0: hdf5_episode},
            episode_idx=0,
            frame_idx=0,
            args=args,
            raw_scene_loader=lambda **_kwargs: raw_scene,
        )

        self.assertEqual(source, "raw_full")
        self.assertEqual(scene.shape[0], 2)

    def test_write_colored_ply_writes_ascii_color_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cloud.ply"
            cloud = np.asarray([[0.0, 0.0, 0.0, 255.0, 1.0, 2.0]], dtype=np.float32)

            write_colored_ply(path, cloud)

            text = path.read_text(encoding="utf-8")
            self.assertIn("element vertex 1", text)
            self.assertIn("property uchar red", text)
            self.assertTrue(text.strip().endswith("0.0000000 0.0000000 0.0000000 255 1 2"))

    def test_run_visualization_reads_processed_hdf5_and_writes_overlay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "data" / "task" / "config"
            data_dir = dataset_dir / "data"
            data_dir.mkdir(parents=True)
            with h5py.File(data_dir / "episode0.hdf5", "w") as f:
                f.create_dataset("joint_action/vector", data=np.zeros((2, 14), dtype=np.float32))
                f.create_dataset(
                    "pointcloud",
                    data=np.asarray(
                        [
                            [
                                [0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
                                [1.0, 0.0, 0.0, 20.0, 20.0, 20.0],
                                [5.0, 0.0, 0.0, 30.0, 30.0, 30.0],
                            ],
                            [
                                [0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
                                [1.0, 0.0, 0.0, 20.0, 20.0, 20.0],
                                [5.0, 0.0, 0.0, 30.0, 30.0, 30.0],
                            ],
                        ],
                        dtype=np.float32,
                    ),
                )
                f.create_dataset(
                    "object_pointcloud/{A}",
                    data=np.asarray(
                        [
                            [
                                [0.0, 0.0, 0.0, 255.0, 0.0, 0.0],
                                [1.0, 0.0, 0.0, 0.0, 255.0, 0.0],
                            ],
                            [
                                [0.0, 0.0, 0.0, 255.0, 0.0, 0.0],
                                [1.0, 0.0, 0.0, 0.0, 255.0, 0.0],
                            ],
                        ],
                        dtype=np.float32,
                    ),
                )

            def fake_load(_checkpoint, *, device):
                return {"device": device}

            def fake_compute(_model, object_cloud, *, target_num_points, **_kwargs):
                points = np.asarray(object_cloud[:target_num_points, :3], dtype=np.float32)
                features = np.concatenate([points, points[:, :1]], axis=1)
                return np.concatenate([points, features], axis=1).astype(np.float32)

            args = Namespace(
                dataset_dir=str(dataset_dir),
                task_name="",
                task_config="",
                data_root="data",
                episode=0,
                frame=0,
                episode_frames=None,
                object_placeholders="{A}",
                semantic_ckpt="",
                semantic_ckpt_A="fake_mug.pt",
                semantic_ckpt_B="",
                semantic_ckpts="",
                semantic_point_num=2,
                object_point_num=2,
                color_mode="pca",
                semantic_forward_mode="reference",
                semantic_input_color_mode="stored",
                overlay_mode="cut_replace",
                overlay_distance=0.05,
                overlay_min_neighbors=1,
                overlay_cut_z_min=None,
                scene_source="hdf5",
                scene_point_num=0,
                camera_labels="",
                min_depth_m=0.05,
                max_depth_m=3.0,
                raw_intrinsics_source="frame",
                disable_serial_remap=False,
                background_mode="original",
                shared_pca_scope="checkpoint",
                device="cpu",
                output_dir=str(root / "out"),
            )

            with mock.patch.object(viz, "_load_semantic_backend", return_value=("cpu", fake_load, fake_compute)):
                summary = run_visualization(args)

            self.assertEqual(len(summary["objects"]), 1)
            self.assertEqual(summary["object_source"], "processed_hdf5_object_pointcloud")
            self.assertTrue((root / "out" / "episode0_frame0_A_semantic_pca.ply").is_file())
            self.assertTrue((root / "out" / "episode0_frame0_scene_semantic_overlay.ply").is_file())
            self.assertTrue((root / "out" / "summary.json").is_file())

    def test_run_visualization_label_mode_uses_predicted_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "data" / "task" / "config"
            data_dir = dataset_dir / "data"
            data_dir.mkdir(parents=True)
            with h5py.File(data_dir / "episode0.hdf5", "w") as f:
                f.create_dataset("joint_action/vector", data=np.zeros((1, 14), dtype=np.float32))
                f.create_dataset(
                    "pointcloud",
                    data=np.asarray(
                        [[[0.0, 0.0, 0.0, 10.0, 10.0, 10.0], [1.0, 0.0, 0.0, 20.0, 20.0, 20.0]]],
                        dtype=np.float32,
                    ),
                )
                f.create_dataset(
                    "object_pointcloud/{A}",
                    data=np.asarray(
                        [[[0.0, 0.0, 0.0, 255.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 255.0, 0.0]]],
                        dtype=np.float32,
                    ),
                )

            def fake_load(_checkpoint, *, device):
                return {"device": device, "canonical_label_names": ["handle", "body"]}

            def fake_compute(_model, object_cloud, *, target_num_points, **_kwargs):
                points = np.asarray(object_cloud[:target_num_points, :3], dtype=np.float32)
                features = np.ones((points.shape[0], 4), dtype=np.float32)
                return np.concatenate([points, features], axis=1).astype(np.float32)

            def fake_predict(_model, object_cloud, *, target_num_points, **_kwargs):
                pointwise = fake_compute(_model, object_cloud, target_num_points=target_num_points)
                return {
                    "point_cloud": pointwise,
                    "pred_labels": np.asarray([0, 1], dtype=np.int64),
                    "confidence": np.asarray([0.9, 0.8], dtype=np.float32),
                    "label_names": ["handle", "body"],
                }

            args = Namespace(
                dataset_dir=str(dataset_dir),
                task_name="",
                task_config="",
                data_root="data",
                episode=0,
                frame=0,
                episode_frames=None,
                object_placeholders="{A}",
                semantic_ckpt="",
                semantic_ckpt_A="fake_mug.pt",
                semantic_ckpt_B="",
                semantic_ckpts="",
                semantic_point_num=2,
                object_point_num=2,
                color_mode="label",
                semantic_forward_mode="reference",
                semantic_input_color_mode="stored",
                overlay_mode="cut_replace",
                overlay_distance=0.05,
                overlay_min_neighbors=1,
                overlay_cut_z_min=None,
                scene_source="hdf5",
                scene_point_num=0,
                camera_labels="",
                min_depth_m=0.05,
                max_depth_m=3.0,
                raw_intrinsics_source="frame",
                disable_serial_remap=False,
                background_mode="original",
                shared_pca_scope="checkpoint",
                device="cpu",
                output_dir=str(root / "out"),
            )

            with mock.patch.object(
                viz,
                "_load_semantic_backend",
                return_value=("cpu", fake_load, fake_compute, fake_predict),
            ):
                summary = run_visualization(args)

            self.assertEqual(summary["color_mode"], "label")
            self.assertEqual(summary["objects"][0]["pred_label_histogram"], {"handle": 1, "body": 1})
            self.assertTrue((root / "out" / "episode0_frame0_A_semantic_labels.ply").is_file())
            self.assertTrue((root / "out" / "label_palette.json").is_file())
            scene_colors = _read_ascii_ply_colors(root / "out" / "episode0_frame0_scene_semantic_overlay.ply")
            palette = build_label_palette(2)
            self.assertTrue(np.any(np.all(scene_colors == palette[0], axis=1)))
            self.assertTrue(np.any(np.all(scene_colors == palette[1], axis=1)))


if __name__ == "__main__":
    unittest.main()
