from __future__ import annotations

from pathlib import Path
import re
import unittest

import numpy as np

from .generate_functional_frame_confidence_variants import (
    VARIANTS,
    build_variant_payload,
    confidence_variants,
)
from .train_task_flow_benchmark import FOLDS


def payload(disagreement: np.ndarray) -> dict[str, np.ndarray]:
    count = len(disagreement)
    return {
        "shoe_id": np.arange(count, dtype=np.int64),
        "target_frame9": np.zeros((count, 9), dtype=np.float32),
        "source_frame_disagreement_deg": np.asarray(disagreement, dtype=np.float32),
        "goal_frame_confidence": np.ones(count, dtype=np.float32),
    }


class FunctionalFrameConfidenceVariantsTest(unittest.TestCase):
    def test_variants_have_expected_fields_and_protocol(self) -> None:
        source = payload(np.arange(1.0, 11.0))
        variants, diagnostic = confidence_variants(source, fold=0)
        self.assertEqual(set(variants), set(VARIANTS))
        train = source["source_frame_disagreement_deg"][
            np.isin(source["shoe_id"], FOLDS[0]["train"])
        ]
        self.assertAlmostEqual(
            diagnostic["hard_threshold_deg"],
            float(np.percentile(train, 99.0)),
        )
        np.testing.assert_array_equal(variants["always_on"], np.ones(10))
        np.testing.assert_array_equal(
            variants["train_p99_hard"],
            source["source_frame_disagreement_deg"]
            <= diagnostic["hard_threshold_deg"],
        )
        soft = variants["soft_continuous"]
        self.assertTrue(np.all((soft >= 0.0) & (soft <= 1.0)))
        self.assertTrue(np.all(np.diff(soft) <= 0.0))
        self.assertGreater(len(np.unique(soft)), 2)
        self.assertEqual(diagnostic["hard_threshold_percentile"], 99.0)
        self.assertTrue(diagnostic["goal_confidence_preserved"])

    def test_calibration_is_invariant_to_test_object_values(self) -> None:
        first = payload(np.arange(1.0, 11.0))
        second = payload(np.arange(1.0, 11.0))
        second["source_frame_disagreement_deg"][[0, 5]] = [1000.0, 2000.0]
        _, first_diagnostic = confidence_variants(first, fold=0)
        _, second_diagnostic = confidence_variants(second, fold=0)
        self.assertEqual(
            first_diagnostic["hard_threshold_deg"],
            second_diagnostic["hard_threshold_deg"],
        )
        self.assertEqual(
            first_diagnostic["soft_scale_deg"],
            second_diagnostic["soft_scale_deg"],
        )

    def test_goal_reliability_multiplies_every_source_variant(self) -> None:
        source = payload(np.arange(1.0, 11.0))
        source["goal_frame_confidence"][[1, 3]] = 0.0
        variants, _ = confidence_variants(source, fold=0)
        for variant in VARIANTS:
            np.testing.assert_array_equal(variants[variant][[1, 3]], 0.0)

    def test_policy_interface_fields_are_explicit_and_original_is_preserved(self) -> None:
        source = payload(np.arange(1.0, 11.0))
        source["target_frame_confidence"] = np.full(10, 0.25, dtype=np.float32)
        output = build_variant_payload(
            source,
            variant="always_on",
            target_confidence=np.ones(10, dtype=np.float32),
            source_confidence=np.ones(10, dtype=np.float32),
            hard_threshold_deg=9.0,
            soft_scale_deg=0.5,
        )
        np.testing.assert_array_equal(
            output["target_frame_confidence_original"], 0.25
        )
        np.testing.assert_array_equal(
            output["source_frame_confidence_variant"], 1.0
        )
        np.testing.assert_array_equal(output["target_frame_confidence"], 1.0)
        self.assertAlmostEqual(float(output["confidence_hard_threshold_deg"]), 9.0)
        self.assertAlmostEqual(float(output["confidence_soft_scale_deg"]), 0.5)

    def test_missing_disagreement_is_rejected(self) -> None:
        source = payload(np.arange(1.0, 11.0))
        del source["source_frame_disagreement_deg"]
        with self.assertRaisesRegex(KeyError, "source_frame_disagreement_deg"):
            confidence_variants(source, fold=0)

    def test_training_queue_matches_fold0_clean_protocol(self) -> None:
        directory = Path(__file__).resolve().parent
        reference = (directory / "run_dense_functional_frame_ablation.sh").read_text(
            encoding="utf-8"
        )
        candidate = (
            directory / "queue_fold0_functional_frame_confidence_variants.sh"
        ).read_text(encoding="utf-8")

        def command(script: str) -> str:
            match = re.search(
                r'"\$\{python_bin\}" -m policy\.GeometryFlow\.'
                r'train_interaction_flow_tokens \\\n(?P<body>.*?)'
                r'--episode-balanced-sampling',
                script,
                flags=re.DOTALL,
            )
            self.assertIsNotNone(match)
            body = match.group("body")
            body = re.sub(r'--dataset\s+"[^\n]+"', "--dataset DATASET", body)
            body = re.sub(r'--output-dir\s+"[^\n]+"', "--output-dir OUTPUT", body)
            body = body.replace(
                '--target-frame-mode "${mode}"', "--target-frame-mode normal"
            )
            return " ".join(body.replace("\\", "").split())

        self.assertEqual(command(candidate), command(reference))


if __name__ == "__main__":
    unittest.main()
