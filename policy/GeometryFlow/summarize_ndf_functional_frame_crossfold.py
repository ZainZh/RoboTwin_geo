#!/usr/bin/env python3
"""Validate and summarize dense NDF functional frames across object folds.

Fold 0 uses the original split seed directories. Folds 1--4 prefer the
versioned cross-fold queue and fall back, per seed, to the earlier combined
``ndf_functional_frame_dense_fold*_seeds012`` directories. Missing folds are an
error unless ``--allow-incomplete`` is explicit; invalid present artifacts are
always an error.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .ensemble_ndf_functional_frames import (
    geodesic_deg,
    maximum_pairwise_disagreement,
    project_rotation_mean,
)
from .functional_action_frame import pose9_rotation
from .train_task_flow_benchmark import FOLDS


SEEDS = (0, 1, 2)
EXPECTED_TRAINING = {
    "conditions": ["correct_frame"],
    "epochs": 200,
    "patience": 30,
    "batch_size": 8,
    "learning_rate": 0.0003,
    "weight_decay": 0.0001,
    "axis_loss_weight": 0.5,
    "orthogonality_loss_weight": 0.05,
    "trainable_scope": "full",
    "promote_alpha_to_beta": False,
}
SUMMARY_METRICS = (
    "mean_deg",
    "median_deg",
    "p90_deg",
    "flip_over_90deg",
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "outputs/geometry_flow/remote4090/task_flow_remote50_xyz128_stride1.npz"
        ),
    )
    result.add_argument(
        "--crossfold-root",
        type=Path,
        default=Path("outputs/geometry_flow/remote4090/dense_ndf_crossfold_v1"),
    )
    result.add_argument(
        "--legacy-root",
        type=Path,
        default=Path("outputs/geometry_flow/remote4090"),
    )
    result.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--allow-incomplete", action="store_true")
    return result


def error_summary(error_deg: np.ndarray) -> dict:
    value = np.asarray(error_deg, dtype=np.float64).reshape(-1)
    if not len(value):
        return {
            "samples": 0,
            "mean_deg": None,
            "median_deg": None,
            "p90_deg": None,
            "max_deg": None,
            "within_15deg": None,
            "within_30deg": None,
            "flip_over_90deg": None,
            "flip_count": 0,
        }
    return {
        "samples": int(len(value)),
        "mean_deg": float(value.mean()),
        "median_deg": float(np.median(value)),
        "p90_deg": float(np.percentile(value, 90)),
        "max_deg": float(value.max()),
        "within_15deg": float(np.mean(value <= 15.0)),
        "within_30deg": float(np.mean(value <= 30.0)),
        "flip_over_90deg": float(np.mean(value >= 90.0)),
        "flip_count": int(np.sum(value >= 90.0)),
    }


def macro_summary(records: list[dict]) -> dict:
    result = {"folds_contributing": int(len(records))}
    for metric in SUMMARY_METRICS:
        values = [record[metric] for record in records if record.get(metric) is not None]
        result[metric] = float(np.mean(values)) if len(values) == len(records) and values else None
    return result


def validate_fold_definitions(folds: list[int], *, require_complete: bool) -> None:
    seen_test: set[int] = set()
    for fold in folds:
        split = {name: set(values) for name, values in FOLDS[int(fold)].items()}
        for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
            overlap = split[first] & split[second]
            if overlap:
                raise ValueError(f"fold {fold} {first}/{second} object leakage: {sorted(overlap)}")
        repeated = seen_test & split["test"]
        if repeated:
            raise ValueError(f"held-out objects repeat across folds: {sorted(repeated)}")
        seen_test |= split["test"]
    if require_complete:
        all_objects = set().union(*(set(split) for fold in FOLDS for split in fold.values()))
        if len(folds) != len(FOLDS) or seen_test != all_objects:
            raise ValueError(
                "complete cross-fold summary must test every object exactly once; "
                f"folds={folds}, test_objects={sorted(seen_test)}"
            )


def _preferred_seed_paths(
    fold: int, seed: int, crossfold_root: Path, legacy_root: Path
) -> tuple[Path, Path]:
    if int(fold) == 0:
        directory = legacy_root / (
            "ndf_functional_frame_dense_fold0"
            if int(seed) == 0
            else "ndf_functional_frame_dense_fold0_seeds12"
        )
        stem = f"fold0_correct_frame_seed{seed}"
        return directory / f"{stem}.json", directory / f"{stem}_frames.npz"
    stem = f"fold{fold}_correct_frame_seed{seed}"
    queued = crossfold_root / f"fold{fold}" / f"ndf_seed{seed}"
    queued_pair = (queued / f"{stem}.json", queued / f"{stem}_frames.npz")
    if all(path.is_file() for path in queued_pair):
        return queued_pair
    legacy = legacy_root / f"ndf_functional_frame_dense_fold{fold}_seeds012"
    return legacy / f"{stem}.json", legacy / f"{stem}_frames.npz"


def fold_paths(
    fold: int, crossfold_root: Path, legacy_root: Path
) -> tuple[list[tuple[Path, Path]], Path, Path]:
    seeds = [
        _preferred_seed_paths(fold, seed, crossfold_root, legacy_root)
        for seed in SEEDS
    ]
    if int(fold) == 0:
        ensemble = legacy_root / "ndf_functional_frame_dense_fold0_ensemble.npz"
    else:
        ensemble = (
            crossfold_root
            / f"fold{fold}"
            / f"fold{fold}_correct_frame_ensemble_p95.npz"
        )
    return seeds, ensemble, ensemble.with_suffix(".json")


def validate_training_result(result: dict, *, fold: int, seed: int) -> int:
    if result.get("condition") != "correct_frame":
        raise ValueError(f"fold {fold} seed {seed}: condition is not correct_frame")
    if int(result.get("fold", -1)) != int(fold):
        raise ValueError(f"fold {fold} seed {seed}: result fold mismatch")
    if int(result.get("seed", -1)) != int(seed):
        raise ValueError(f"fold {fold} seed {seed}: result seed mismatch")
    config = result.get("config", {})
    for key, expected in EXPECTED_TRAINING.items():
        if config.get(key) != expected:
            raise ValueError(
                f"fold {fold} seed {seed}: config {key}={config.get(key)!r}, "
                f"expected {expected!r}"
            )
    if int(fold) not in [int(value) for value in config.get("folds", [])]:
        raise ValueError(f"fold {fold} seed {seed}: config does not include fold")
    if int(seed) not in [int(value) for value in config.get("seeds", [])]:
        raise ValueError(f"fold {fold} seed {seed}: config does not include seed")
    semantics = result.get("frame_semantics", {})
    if semantics.get("alpha") != "signed object Z" or semantics.get("beta") != "signed object Y":
        raise ValueError(f"fold {fold} seed {seed}: unexpected frame semantics")
    return int(result.get("trainable_parameters", -1))


def load_frame_predictions(
    path: Path, *, shoe_id: np.ndarray, episode_id: np.ndarray
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        frames = np.asarray(archive["frames"], dtype=np.float32)
        if "shoe_id" not in archive or not np.array_equal(shoe_id, archive["shoe_id"]):
            raise ValueError(f"shoe_id alignment mismatch in {path}")
        if "episode_id" not in archive or not np.array_equal(
            episode_id, archive["episode_id"]
        ):
            raise ValueError(f"episode_id alignment mismatch in {path}")
    if frames.shape != (len(shoe_id), 3, 3) or not np.all(np.isfinite(frames)):
        raise ValueError(f"invalid frame array in {path}: {frames.shape}")
    orthogonality = np.max(
        np.abs(np.swapaxes(frames, -1, -2) @ frames - np.eye(3)), axis=(-2, -1)
    )
    determinant = np.linalg.det(frames)
    if float(orthogonality.max()) > 2e-3 or float(np.max(np.abs(determinant - 1.0))) > 2e-3:
        raise ValueError(f"non-SO(3) predictions in {path}")
    return frames


def validate_json_test_metrics(
    result: dict, computed: dict, *, fold: int, seed: int
) -> None:
    reported = result.get("test", {})
    for key in ("samples", "mean_deg", "median_deg", "p90_deg", "flip_over_90deg"):
        if key not in reported:
            raise ValueError(f"fold {fold} seed {seed}: test metric {key} is missing")
        if key == "samples":
            agrees = int(reported[key]) == int(computed[key])
        else:
            agrees = bool(np.isclose(reported[key], computed[key], atol=2e-3, rtol=2e-5))
        if not agrees:
            raise ValueError(
                f"fold {fold} seed {seed}: JSON/frame mismatch for {key}: "
                f"{reported[key]} != {computed[key]}"
            )


def validate_ensemble(
    *,
    ensemble_path: Path,
    metadata_path: Path,
    frame_paths: list[Path],
    model_frames: np.ndarray,
    shoe_id: np.ndarray,
    episode_id: np.ndarray,
    fold: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict[str, float]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata.get("fold", -1)) != int(fold):
        raise ValueError(f"fold {fold}: ensemble metadata fold mismatch")
    if metadata.get("method") != "projected_rotation_mean_with_max_pairwise_disagreement":
        raise ValueError(f"fold {fold}: unexpected ensemble method")
    if metadata.get("confidence_calibration") != "training objects only":
        raise ValueError(f"fold {fold}: confidence was not calibrated on training objects only")
    if float(metadata.get("confidence_percentile", -1.0)) != 95.0:
        raise ValueError(f"fold {fold}: confidence percentile is not p95")
    expected_names = {path.name for path in frame_paths}
    metadata_names = {Path(path).name for path in metadata.get("frame_predictions", [])}
    if metadata_names != expected_names or len(metadata.get("frame_predictions", [])) != 3:
        raise ValueError(f"fold {fold}: ensemble does not cite exactly the three seed frames")

    with np.load(ensemble_path, allow_pickle=False) as archive:
        required = {
            "frames",
            "frame_confidence",
            "frame_disagreement_deg",
            "confidence_threshold_deg",
            "shoe_id",
            "episode_id",
        }
        missing = required - set(archive.files)
        if missing:
            raise KeyError(f"fold {fold}: ensemble is missing {sorted(missing)}")
        if not np.array_equal(shoe_id, archive["shoe_id"]) or not np.array_equal(
            episode_id, archive["episode_id"]
        ):
            raise ValueError(f"fold {fold}: ensemble row alignment mismatch")
        frames = np.asarray(archive["frames"], dtype=np.float32)
        confidence = np.asarray(archive["frame_confidence"], dtype=np.float32)
        disagreement = np.asarray(archive["frame_disagreement_deg"], dtype=np.float32)
        threshold = float(np.asarray(archive["confidence_threshold_deg"]))

    expected_frames = project_rotation_mean(model_frames)
    expected_disagreement = maximum_pairwise_disagreement(model_frames)
    train_mask = np.isin(shoe_id, np.asarray(FOLDS[int(fold)]["train"], dtype=np.int64))
    expected_threshold = float(np.percentile(expected_disagreement[train_mask], 95.0))
    expected_confidence = (expected_disagreement <= expected_threshold).astype(np.float32)
    if not np.allclose(frames, expected_frames, atol=2e-5):
        raise ValueError(f"fold {fold}: saved ensemble is not the projected three-seed mean")
    if not np.allclose(disagreement, expected_disagreement, atol=2e-4):
        raise ValueError(f"fold {fold}: saved ensemble disagreement is inconsistent")
    if not np.isclose(threshold, expected_threshold, atol=2e-4):
        raise ValueError(f"fold {fold}: threshold was not fitted at train-only p95")
    if not np.array_equal(confidence, expected_confidence):
        raise ValueError(f"fold {fold}: confidence mask is inconsistent with train-only p95")
    if not np.isclose(float(metadata.get("confidence_threshold_deg")), expected_threshold, atol=2e-4):
        raise ValueError(f"fold {fold}: metadata threshold mismatch")

    acceptance = {}
    for split, ids in FOLDS[int(fold)].items():
        mask = np.isin(shoe_id, np.asarray(ids, dtype=np.int64))
        acceptance[split] = float(confidence[mask].mean())
        reported = float(metadata.get("acceptance", {}).get(split, -1.0))
        if not np.isclose(reported, acceptance[split], atol=2e-6):
            raise ValueError(f"fold {fold}: metadata {split} acceptance mismatch")
    return frames, confidence, disagreement, expected_threshold, acceptance


def main() -> None:
    args = parser().parse_args()
    folds = [int(value) for value in args.folds]
    if len(set(folds)) != len(folds) or any(fold < 0 or fold >= len(FOLDS) for fold in folds):
        raise ValueError(f"folds must be unique and lie in [0,{len(FOLDS) - 1}]")
    if not args.dataset.is_file():
        raise FileNotFoundError(args.dataset)
    validate_fold_definitions(folds, require_complete=not args.allow_incomplete)

    with np.load(args.dataset, allow_pickle=False) as archive:
        required = {"current_object_pose9", "shoe_id", "episode_id"}
        missing = required - set(archive.files)
        if missing:
            raise KeyError(f"dataset is missing {sorted(missing)}")
        current_rotation = pose9_rotation(archive["current_object_pose9"])
        shoe_id = np.asarray(archive["shoe_id"], dtype=np.int64)
        episode_id = np.asarray(archive["episode_id"], dtype=np.int64)
    if len(current_rotation) != len(shoe_id) or len(episode_id) != len(shoe_id):
        raise ValueError("dataset rows are not aligned")

    output = {
        "schema_version": 1,
        "status": None,
        "dataset": str(args.dataset.resolve()),
        "requested_folds": folds,
        "available_folds": [],
        "incomplete_folds": {},
        "expected_seeds": list(SEEDS),
        "confidence_calibration": "strictly training-object p95 per fold",
        "per_fold": {},
        "macro_available_folds": {},
        "pooled_available_heldout": {},
        "held_out_objects": {},
        "held_out_object_macro": {},
    }
    trainable_parameters = None
    all_single_errors = []
    all_ensemble_errors = []
    all_confidence = []
    all_object_records: dict[int, dict] = {}

    for fold in folds:
        seed_pairs, ensemble_path, ensemble_metadata = fold_paths(
            fold, args.crossfold_root, args.legacy_root
        )
        required_paths = [path for pair in seed_pairs for path in pair] + [
            ensemble_path,
            ensemble_metadata,
        ]
        missing_paths = [str(path) for path in required_paths if not path.is_file()]
        if missing_paths:
            if not args.allow_incomplete:
                raise FileNotFoundError(
                    f"fold {fold} is incomplete: {json.dumps(missing_paths)}"
                )
            output["incomplete_folds"][str(fold)] = {
                "status": "missing_artifacts",
                "missing": missing_paths,
            }
            continue

        test_ids = np.asarray(FOLDS[fold]["test"], dtype=np.int64)
        test_mask = np.isin(shoe_id, test_ids)
        actual_test = set(np.unique(shoe_id[test_mask]).tolist())
        if actual_test != set(test_ids.tolist()):
            raise ValueError(
                f"fold {fold}: dataset test objects {sorted(actual_test)} do not match "
                f"protocol {sorted(test_ids.tolist())}"
            )

        model_frames = []
        per_seed = {}
        frame_paths = []
        seed_errors = []
        for seed, (json_path, frame_path) in zip(SEEDS, seed_pairs):
            result = json.loads(json_path.read_text(encoding="utf-8"))
            current_parameters = validate_training_result(result, fold=fold, seed=seed)
            if current_parameters <= 0:
                raise ValueError(f"fold {fold} seed {seed}: invalid trainable parameter count")
            if trainable_parameters is None:
                trainable_parameters = current_parameters
            elif current_parameters != trainable_parameters:
                raise ValueError(
                    f"fold {fold} seed {seed}: trainable parameter mismatch "
                    f"{current_parameters} != {trainable_parameters}"
                )
            frames = load_frame_predictions(
                frame_path, shoe_id=shoe_id, episode_id=episode_id
            )
            error = geodesic_deg(frames[test_mask], current_rotation[test_mask])
            summary = error_summary(error)
            validate_json_test_metrics(result, summary, fold=fold, seed=seed)
            per_seed[str(seed)] = {
                "json": str(json_path.resolve()),
                "frames": str(frame_path.resolve()),
                **summary,
            }
            model_frames.append(frames)
            frame_paths.append(frame_path)
            seed_errors.append(error)

        stacked_frames = np.stack(model_frames, axis=0)
        seed_error_array = np.stack(seed_errors, axis=0)
        single_seed_macro = {
            metric: float(np.mean([per_seed[str(seed)][metric] for seed in SEEDS]))
            for metric in SUMMARY_METRICS
        }
        single_seed_macro["seeds"] = 3
        ensemble_frames, confidence, disagreement, threshold, acceptance = validate_ensemble(
            ensemble_path=ensemble_path,
            metadata_path=ensemble_metadata,
            frame_paths=frame_paths,
            model_frames=stacked_frames,
            shoe_id=shoe_id,
            episode_id=episode_id,
            fold=fold,
        )
        ensemble_error = geodesic_deg(
            ensemble_frames[test_mask], current_rotation[test_mask]
        )
        test_confidence = confidence[test_mask] >= 0.5
        fold_result = {
            "status": "complete",
            "train_objects": list(FOLDS[fold]["train"]),
            "validation_objects": list(FOLDS[fold]["validation"]),
            "test_objects": list(FOLDS[fold]["test"]),
            "three_seed_validation": True,
            "single_model": {
                "per_seed": per_seed,
                "seed_macro": single_seed_macro,
                "pooled_seed_samples": error_summary(seed_error_array),
            },
            "ensemble": {
                "archive": str(ensemble_path.resolve()),
                "train_only_p95_threshold_deg": threshold,
                "acceptance_by_split": acceptance,
                "heldout_all": error_summary(ensemble_error),
                "heldout_accepted": error_summary(ensemble_error[test_confidence]),
                "heldout_rejected": error_summary(ensemble_error[~test_confidence]),
                "heldout_coverage": float(test_confidence.mean()),
                "heldout_disagreement_deg": error_summary(disagreement[test_mask]),
            },
            "held_out_objects": {},
        }

        test_shoes = shoe_id[test_mask]
        for object_id in test_ids:
            object_mask = test_shoes == int(object_id)
            object_confidence = test_confidence[object_mask]
            record = {
                "fold": int(fold),
                "single_model_pooled_seed_samples": error_summary(
                    seed_error_array[:, object_mask]
                ),
                "ensemble_all": error_summary(ensemble_error[object_mask]),
                "ensemble_accepted": error_summary(
                    ensemble_error[object_mask][object_confidence]
                ),
                "coverage": float(object_confidence.mean()),
            }
            fold_result["held_out_objects"][str(int(object_id))] = record
            if int(object_id) in all_object_records:
                raise ValueError(f"object {object_id} appears held out more than once")
            all_object_records[int(object_id)] = record

        output["per_fold"][str(fold)] = fold_result
        output["available_folds"].append(int(fold))
        all_single_errors.append(seed_error_array)
        all_ensemble_errors.append(ensemble_error)
        all_confidence.append(test_confidence)

    available = output["available_folds"]
    output["status"] = (
        "complete_five_fold"
        if available == list(range(5)) and not output["incomplete_folds"]
        else "incomplete_explicitly_allowed"
    )
    output["complete_five_fold_macro_available"] = output["status"] == "complete_five_fold"
    if not available:
        raise ValueError("no complete fold is available to summarize")

    fold_results = [output["per_fold"][str(fold)] for fold in available]
    output["macro_available_folds"] = {
        "label": "descriptive macro over available complete folds only",
        "single_model_seed_macro": macro_summary(
            [item["single_model"]["seed_macro"] for item in fold_results]
        ),
        "ensemble_all": macro_summary(
            [item["ensemble"]["heldout_all"] for item in fold_results]
        ),
        "ensemble_accepted": macro_summary(
            [item["ensemble"]["heldout_accepted"] for item in fold_results]
        ),
        "coverage": float(
            np.mean([item["ensemble"]["heldout_coverage"] for item in fold_results])
        ),
    }
    pooled_single = np.concatenate(all_single_errors, axis=1)
    pooled_ensemble = np.concatenate(all_ensemble_errors)
    pooled_confidence = np.concatenate(all_confidence)
    output["pooled_available_heldout"] = {
        "label": "frame-weighted across objects held out in available folds",
        "single_model": error_summary(pooled_single),
        "ensemble_all": error_summary(pooled_ensemble),
        "ensemble_accepted": error_summary(pooled_ensemble[pooled_confidence]),
        "ensemble_rejected": error_summary(pooled_ensemble[~pooled_confidence]),
        "coverage": float(pooled_confidence.mean()),
    }
    output["held_out_objects"] = {
        str(object_id): all_object_records[object_id]
        for object_id in sorted(all_object_records)
    }
    object_values = list(all_object_records.values())
    output["held_out_object_macro"] = {
        "label": "equal weight per available independently held-out object",
        "objects_contributing": len(object_values),
        "object_ids": sorted(all_object_records),
        "single_model": macro_summary(
            [item["single_model_pooled_seed_samples"] for item in object_values]
        ),
        "ensemble_all": macro_summary(
            [item["ensemble_all"] for item in object_values]
        ),
        "ensemble_accepted": macro_summary(
            [item["ensemble_accepted"] for item in object_values]
        ),
        "coverage": float(np.mean([item["coverage"] for item in object_values])),
    }
    output["trainable_parameters"] = trainable_parameters

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output))


if __name__ == "__main__":
    main()
