from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from semantic_field_eval_utils import (
    apply_support_normalization,
    compute_support_normalization,
    confusion_matrix_from_predictions,
    embedding_stability_metrics,
    merge_confusions,
    metrics_from_confusion,
    resolve_legacy_config_path,
    sha256_file,
)
from semantic_support_perturbations import (
    build_fixed_query_support_pair,
    normalize_condition_weights,
    parse_support_condition,
    select_observed_support,
)


class SemanticFieldEvalUtilsTest(unittest.TestCase):
    def test_confusion_and_miou(self) -> None:
        confusion, valid_count = confusion_matrix_from_predictions(
            predictions=np.asarray([0, 1, 1, 1]),
            targets=np.asarray([0, 0, 1, 1]),
            num_classes=2,
        )
        np.testing.assert_array_equal(confusion, [[1, 1], [0, 2]])
        self.assertEqual(valid_count, 4)

        metrics = metrics_from_confusion(confusion, ["Handle", "Head"])
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["per_class"][0]["iou"], 0.5)
        self.assertAlmostEqual(metrics["per_class"][1]["iou"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["mean_iou"], (0.5 + 2.0 / 3.0) / 2.0)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.75)

    def test_ignore_label_and_merge(self) -> None:
        first, valid_count = confusion_matrix_from_predictions(
            predictions=np.asarray([0, 1, 0]),
            targets=np.asarray([0, -1, 1]),
            num_classes=2,
        )
        second, _ = confusion_matrix_from_predictions(
            predictions=np.asarray([1, 1]),
            targets=np.asarray([1, 1]),
            num_classes=2,
        )
        self.assertEqual(valid_count, 2)
        merged = merge_confusions([first, second], num_classes=2)
        np.testing.assert_array_equal(merged, [[1, 0], [1, 2]])

    def test_invalid_prediction_is_excluded(self) -> None:
        confusion, valid_count = confusion_matrix_from_predictions(
            predictions=np.asarray([0, 2]),
            targets=np.asarray([0, 1]),
            num_classes=2,
        )
        self.assertEqual(valid_count, 1)
        np.testing.assert_array_equal(confusion, [[1, 0], [0, 0]])

    def test_hash_and_legacy_config_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            legacy_root = root / "legacy"
            local_root = root / "local"
            legacy_root.mkdir()
            local_root.mkdir()
            legacy = legacy_root / "labels.json"
            local = local_root / "labels.json"
            legacy.write_text('{"Handle": "Handle"}', encoding="utf-8")
            local.write_text('{"Handle": "Handle"}', encoding="utf-8")

            resolved, manifest = resolve_legacy_config_path(legacy, local_root)
            self.assertEqual(resolved, local.resolve())
            self.assertEqual(manifest["sha256"], sha256_file(local))

            local.write_text('{"Handle": "Head"}', encoding="utf-8")
            with self.assertRaises(RuntimeError):
                resolve_legacy_config_path(legacy, local_root)

    def test_support_normalization_matches_expected_frame(self) -> None:
        support = np.asarray(
            [[-1.0, -1.0, 0.0], [1.0, 1.0, 2.0]],
            dtype=np.float32,
        )
        parameters = compute_support_normalization(
            support,
            coord_scale=2.0,
            center_shift_z=True,
        )
        normalized = apply_support_normalization(support, parameters)
        self.assertAlmostEqual(float(normalized[:, 0].mean()), 0.0, places=6)
        self.assertAlmostEqual(float(normalized[:, 1].mean()), 0.0, places=6)
        self.assertAlmostEqual(float(normalized[:, 2].min()), 0.0, places=6)
        self.assertAlmostEqual(float(np.linalg.norm((support - support.mean(0)), axis=1).max()), parameters["radius"])

    def test_embedding_stability(self) -> None:
        reference = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        identical = embedding_stability_metrics(reference, reference.copy())
        self.assertAlmostEqual(identical["cosine_mean"], 1.0)
        self.assertAlmostEqual(identical["embedding_l2_mean"], 0.0)

        swapped = embedding_stability_metrics(
            reference,
            np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        )
        self.assertAlmostEqual(swapped["cosine_mean"], 0.0)
        self.assertAlmostEqual(swapped["embedding_l2_mean"], np.sqrt(2.0))

    def test_support_condition_validation(self) -> None:
        self.assertEqual(parse_support_condition("count_512").value, 512.0)
        self.assertEqual(parse_support_condition("crop_keep_0.5").kind, "view_crop")
        conditions, probabilities = normalize_condition_weights(
            ["count_512", "crop_keep_0.5"],
            [1.0, 3.0],
        )
        self.assertEqual(len(conditions), 2)
        np.testing.assert_allclose(probabilities, [0.25, 0.75])
        with self.assertRaises(ValueError):
            parse_support_condition("dropout_0.5")

    def test_view_crop_selects_one_side(self) -> None:
        points = np.stack(
            [np.arange(20, dtype=np.float32), np.zeros(20), np.zeros(20)],
            axis=1,
        )
        normals = np.ones_like(points)
        colors = np.zeros_like(points)
        observed = select_observed_support(
            points,
            normals,
            colors,
            parse_support_condition("crop_keep_0.5"),
            np.random.default_rng(7),
        )
        self.assertEqual(observed["observed_source_points"], 10)
        direction = observed["direction"]
        kept_projection = observed["points"] @ direction
        all_projection = points @ direction
        self.assertGreaterEqual(
            float(kept_projection.min()),
            float(np.partition(all_projection, -10)[-10]),
        )

    def test_fixed_query_pair_has_fixed_network_shape(self) -> None:
        rng = np.random.default_rng(11)
        points = rng.normal(size=(40, 3)).astype(np.float32)
        normals = rng.normal(size=(40, 3)).astype(np.float32)
        colors = rng.uniform(0.0, 255.0, size=(40, 3)).astype(np.float32)
        queries = rng.normal(size=(13, 3)).astype(np.float32)
        pair = build_fixed_query_support_pair(
            reference_points=points,
            reference_normals=normals,
            reference_colors=colors,
            raw_query_points=queries,
            condition=parse_support_condition("count_8"),
            output_support_count=32,
            coord_scale=1.0,
            center_shift_z=True,
            rng=rng,
        )
        self.assertEqual(pair["clean_support_points"].shape, (32, 3))
        self.assertEqual(pair["perturbed_support_points"].shape, (32, 3))
        self.assertEqual(pair["clean_query_points"].shape, (13, 3))
        self.assertEqual(pair["perturbed_query_points"].shape, (13, 3))
        self.assertEqual(pair["observed_source_points"], 8)
        self.assertTrue(np.isfinite(pair["perturbed_query_points"]).all())


if __name__ == "__main__":
    unittest.main()
