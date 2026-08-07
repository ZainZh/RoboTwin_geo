from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from .ensemble_ndf_functional_frames import geodesic_deg
from .select_ndf_functional_frame_ensemble import (
    build_selected_ensemble,
    write_selected_ensemble,
)
from .train_task_flow_benchmark import FOLDS


SHOES = np.asarray([2, 3, 4, 7, 8, 9, 1, 6, 0, 5], dtype=np.int64)
EPISODES = np.arange(len(SHOES), dtype=np.int64)


def identity_pose9(count: int) -> np.ndarray:
    rotation6 = np.eye(3, dtype=np.float32)[:, :2].reshape(6)
    pose9 = np.zeros((count, 9), dtype=np.float32)
    pose9[:, 3:] = rotation6
    return pose9


def create_fixture(root: Path) -> tuple[Path, list[tuple[Path, Path]]]:
    dataset = root / "dataset.npz"
    np.savez_compressed(
        dataset,
        shoe_id=SHOES,
        episode_id=EPISODES,
        current_object_pose9=identity_pose9(len(SHOES)),
    )
    split = FOLDS[0]
    train = np.isin(SHOES, split["train"])
    validation = np.isin(SHOES, split["validation"])
    test = np.isin(SHOES, split["test"])
    # Seed 0 is best on validation but intentionally catastrophic on test.
    # Seed 2 is good on test but bad on validation and must be excluded.
    angle_specs = {
        0: {"train": 0.0, "validation": 0.0, "test": 170.0},
        1: {"train": 2.0, "validation": 10.0, "test": 0.0},
        2: {"train": 50.0, "validation": 80.0, "test": 0.0},
    }
    pairs = []
    target = np.repeat(np.eye(3)[None], len(SHOES), axis=0)
    for seed, spec in angle_specs.items():
        angles = np.zeros(len(SHOES), dtype=np.float64)
        angles[train] = spec["train"]
        angles[validation] = spec["validation"]
        angles[test] = spec["test"]
        frames = Rotation.from_euler("z", angles, degrees=True).as_matrix().astype(
            np.float32
        )
        validation_p90 = float(
            np.percentile(geodesic_deg(frames[validation], target[validation]), 90.0)
        )
        frame_path = root / f"seed{seed}_frames.npz"
        result_path = root / f"seed{seed}.json"
        np.savez_compressed(
            frame_path,
            frames=frames,
            shoe_id=SHOES,
            episode_id=EPISODES,
        )
        result = {
            "condition": "correct_frame",
            "fold": 0,
            "seed": seed,
            "trainable_scope": "full",
            "initialization": "pretrained",
            "trainable_parameters": 1234,
            "validation": {
                "samples": int(validation.sum()),
                "p90_deg": validation_p90,
            },
            # Deliberately invalid and contradictory. Selection must not parse it.
            "test": {"p90_deg": "intentionally-invalid-and-unused" if seed == 0 else 0.0},
            "config": {
                "dataset": str(dataset.resolve()),
                "checkpoint": "/synthetic/shoe.pth",
                "output_dir": str((root / f"run{seed}").resolve()),
                "folds": [0],
                "seeds": [seed],
                "conditions": ["correct_frame"],
                "epochs": 200,
                "patience": 30,
                "batch_size": 8,
                "learning_rate": 3e-4,
                "weight_decay": 1e-4,
                "minimum_improvement": 1e-5,
                "axis_loss_weight": 0.5,
                "orthogonality_loss_weight": 0.05,
                "trainable_scope": "full",
                "promote_alpha_to_beta": False,
                "initialization": "pretrained",
                "train_episodes_per_object": None,
                "device": "cuda:0",
                "cuda_memory_fraction": 0.48,
            },
            "frame_semantics": {
                "alpha": "signed object Z",
                "beta": "signed object Y",
                "rotation": "columns [X,Y,Z], X=Y cross Z",
            },
        }
        result_path.write_text(json.dumps(result), encoding="utf-8")
        pairs.append((result_path, frame_path))
    return dataset, pairs


class SelectNdfFunctionalFrameEnsembleTest(unittest.TestCase):
    def test_bad_test_seed_does_not_affect_selection_and_bad_validation_is_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, pairs = create_fixture(Path(directory))
            # Even invalid test supervision must be irrelevant: only validation
            # pose labels are decoded by the selector.
            with np.load(dataset, allow_pickle=False) as archive:
                dataset_payload = {
                    key: np.asarray(archive[key]) for key in archive.files
                }
            test_mask = np.isin(SHOES, FOLDS[0]["test"])
            dataset_payload["current_object_pose9"] = dataset_payload[
                "current_object_pose9"
            ].copy()
            dataset_payload["current_object_pose9"][test_mask] = np.nan
            np.savez_compressed(dataset, **dataset_payload)
            payload, metadata = build_selected_ensemble(
                dataset=dataset,
                candidate_pairs=[pairs[2], pairs[0], pairs[1]],
                fold=0,
            )

        self.assertEqual(metadata["selected_seeds"], [0, 1])
        self.assertFalse(metadata["test_metrics_used_for_selection"])
        candidates = metadata["candidate_validation_p90_deg"]
        self.assertEqual([item["seed"] for item in candidates], [0, 1, 2])
        self.assertEqual([item["selected"] for item in candidates], [True, True, False])
        self.assertAlmostEqual(candidates[0]["validation_p90_deg"], 0.0, places=4)
        self.assertAlmostEqual(candidates[1]["validation_p90_deg"], 10.0, places=3)
        self.assertAlmostEqual(candidates[2]["validation_p90_deg"], 80.0, places=3)
        self.assertEqual(
            set(payload),
            {
                "frames",
                "frame_confidence",
                "frame_disagreement_deg",
                "confidence_threshold_deg",
                "shoe_id",
                "episode_id",
            },
        )
        self.assertAlmostEqual(float(payload["confidence_threshold_deg"]), 2.0, places=3)

    def test_rejects_fold_and_seed_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, pairs = create_fixture(root)
            result = json.loads(pairs[0][0].read_text(encoding="utf-8"))
            result["fold"] = 1
            pairs[0][0].write_text(json.dumps(result), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "result fold"):
                build_selected_ensemble(dataset=dataset, candidate_pairs=pairs, fold=0)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, pairs = create_fixture(root)
            result = json.loads(pairs[2][0].read_text(encoding="utf-8"))
            result["seed"] = 1
            result["config"]["seeds"] = [1]
            pairs[2][0].write_text(json.dumps(result), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "seeds must be unique"):
                build_selected_ensemble(dataset=dataset, candidate_pairs=pairs, fold=0)

    def test_rejects_alignment_and_hyperparameter_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, pairs = create_fixture(root)
            with np.load(pairs[2][1], allow_pickle=False) as archive:
                frame_payload = {key: np.asarray(archive[key]) for key in archive.files}
            frame_payload["episode_id"] = frame_payload["episode_id"].copy()
            frame_payload["episode_id"][0] = 999
            np.savez_compressed(pairs[2][1], **frame_payload)
            with self.assertRaisesRegex(ValueError, "episode_id alignment mismatch"):
                build_selected_ensemble(dataset=dataset, candidate_pairs=pairs, fold=0)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, pairs = create_fixture(root)
            result = json.loads(pairs[2][0].read_text(encoding="utf-8"))
            result["config"]["learning_rate"] = 1e-2
            pairs[2][0].write_text(json.dumps(result), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hyperparameters differ"):
                build_selected_ensemble(dataset=dataset, candidate_pairs=pairs, fold=0)

    def test_writer_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, pairs = create_fixture(root)
            payload, metadata = build_selected_ensemble(
                dataset=dataset, candidate_pairs=pairs, fold=0
            )
            output = root / "selected.npz"
            write_selected_ensemble(output, payload, metadata)
            self.assertTrue(output.is_file())
            self.assertTrue(output.with_suffix(".json").is_file())
            with self.assertRaisesRegex(FileExistsError, "refuse to overwrite"):
                write_selected_ensemble(output, payload, metadata)

    def test_legacy_runs_without_initialization_are_pretrained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, pairs = create_fixture(root)
            for result_path, _ in pairs:
                result = json.loads(result_path.read_text(encoding="utf-8"))
                result.pop("initialization")
                result["config"].pop("initialization")
                result_path.write_text(json.dumps(result), encoding="utf-8")
            _payload, metadata = build_selected_ensemble(
                dataset=dataset, candidate_pairs=pairs, fold=0
            )
        self.assertEqual(metadata["training_hyperparameters"]["initialization"], "pretrained")

    def test_confidence_percentile_is_train_only_and_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, pairs = create_fixture(root)
            _payload, metadata = build_selected_ensemble(
                dataset=dataset,
                candidate_pairs=pairs,
                fold=0,
                confidence_percentile=99.0,
            )
            self.assertEqual(metadata["confidence_percentile"], 99.0)
            with self.assertRaisesRegex(ValueError, "strictly between"):
                build_selected_ensemble(
                    dataset=dataset,
                    candidate_pairs=pairs,
                    fold=0,
                    confidence_percentile=100.0,
                )


if __name__ == "__main__":
    unittest.main()
