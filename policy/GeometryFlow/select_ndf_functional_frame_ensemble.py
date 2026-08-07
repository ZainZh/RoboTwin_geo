#!/usr/bin/env python3
"""Select two NDF frame seeds on validation p90 and build a safe ensemble.

Exactly three paired training-result JSON/frame archives are required.  The
top two are selected solely by held-out validation-object frame p90; test
metrics are deliberately never read.  Ensemble confidence is then calibrated
from selected-model disagreement on training objects only.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .ensemble_ndf_functional_frames import (
    geodesic_deg,
    maximum_pairwise_disagreement,
    project_rotation_mean,
)
from .functional_action_frame import pose9_rotation
from .train_task_flow_benchmark import FOLDS


CANDIDATE_COUNT = 3
SELECTED_COUNT = 2
CONFIDENCE_PERCENTILE = 95.0
SELECTION_CRITERION = (
    "lowest validation.p90_deg; deterministic tie-break by numeric seed; "
    "test metrics are not read"
)
REQUIRED_HYPERPARAMETERS = (
    "checkpoint",
    "conditions",
    "epochs",
    "patience",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "minimum_improvement",
    "axis_loss_weight",
    "orthogonality_loss_weight",
    "trainable_scope",
    "promote_alpha_to_beta",
)
EXPECTED_FRAME_SEMANTICS = {
    "alpha": "signed object Z",
    "beta": "signed object Y",
    "rotation": "columns [X,Y,Z], X=Y cross Z",
}


@dataclass(frozen=True)
class Candidate:
    seed: int
    validation_p90_deg: float
    result_json: Path
    frame_predictions: Path
    frames: np.ndarray
    hyperparameters: dict[str, Any]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", type=Path, required=True)
    result.add_argument(
        "--candidate",
        action="append",
        nargs=2,
        type=Path,
        metavar=("RESULT_JSON", "FRAMES_NPZ"),
        required=True,
        help="Repeat exactly three times; JSON and frames must be from one seed.",
    )
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--fold", type=int, required=True)
    result.add_argument("--confidence-percentile", type=float, default=95.0)
    return result


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def _as_int_list(value: Any, *, name: str, path: Path) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path}: config {name} must be a nonempty list")
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path}: config {name} must contain integers") from error


def _validate_so3(frames: np.ndarray, *, path: Path) -> None:
    if not np.all(np.isfinite(frames)):
        raise ValueError(f"{path}: frame predictions contain non-finite values")
    identity_error = np.max(
        np.abs(np.swapaxes(frames, -1, -2) @ frames - np.eye(3)),
        axis=(-2, -1),
    )
    determinant_error = np.abs(np.linalg.det(frames) - 1.0)
    if float(identity_error.max(initial=0.0)) > 2e-3 or float(
        determinant_error.max(initial=0.0)
    ) > 2e-3:
        raise ValueError(f"{path}: frame predictions are not proper SO(3) rotations")


def _load_dataset(
    path: Path, *, fold: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    if fold < 0 or fold >= len(FOLDS):
        raise ValueError(f"fold must lie in [0,{len(FOLDS) - 1}]")
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        required = {"shoe_id", "episode_id", "current_object_pose9"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise KeyError(f"{path}: dataset is missing {missing}")
        shoe_id = np.asarray(archive["shoe_id"], dtype=np.int64)
        episode_id = np.asarray(archive["episode_id"], dtype=np.int64)
        pose9 = np.asarray(archive["current_object_pose9"], dtype=np.float32)
    samples = len(shoe_id)
    if shoe_id.shape != (samples,) or episode_id.shape != (samples,):
        raise ValueError(f"{path}: shoe_id and episode_id must be one-dimensional")
    if pose9.shape != (samples, 9):
        raise ValueError(f"{path}: current_object_pose9 must have shape [samples,9]")
    split_masks = {
        name: np.isin(shoe_id, np.asarray(ids, dtype=np.int64))
        for name, ids in FOLDS[fold].items()
    }
    for name, mask in split_masks.items():
        if not np.any(mask):
            raise ValueError(f"{path}: fold {fold} {name} split has no samples")
    covered = np.logical_or.reduce(tuple(split_masks.values()))
    if not np.all(covered):
        unknown = sorted(int(value) for value in np.unique(shoe_id[~covered]))
        raise ValueError(f"{path}: fold {fold} has unknown shoe ids {unknown}")
    # Decode supervision only for validation rows. Test pose labels do not
    # participate in candidate validation, selection, aggregation or confidence.
    validation_pose9 = pose9[split_masks["validation"]]
    if not np.all(np.isfinite(validation_pose9)):
        raise ValueError(f"{path}: validation current_object_pose9 is non-finite")
    validation_rotations = pose9_rotation(validation_pose9)
    _validate_so3(validation_rotations, path=path)
    return shoe_id, episode_id, validation_rotations, split_masks


def _candidate_hyperparameters(
    result: dict[str, Any], *, result_path: Path, dataset: Path, fold: int, seed: int
) -> dict[str, Any]:
    config = result.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"{result_path}: result config is missing")
    missing = [key for key in REQUIRED_HYPERPARAMETERS if key not in config]
    if missing:
        raise ValueError(f"{result_path}: config is missing hyperparameters {missing}")
    configured_dataset = Path(str(config.get("dataset", "")))
    if _resolved(configured_dataset) != _resolved(dataset):
        raise ValueError(f"{result_path}: config dataset does not match --dataset")
    folds = _as_int_list(config.get("folds"), name="folds", path=result_path)
    seeds = _as_int_list(config.get("seeds"), name="seeds", path=result_path)
    if fold not in folds:
        raise ValueError(f"{result_path}: config folds do not contain fold {fold}")
    if seed not in seeds:
        raise ValueError(f"{result_path}: config seeds do not contain seed {seed}")
    conditions = config.get("conditions")
    if not isinstance(conditions, list) or "correct_frame" not in conditions:
        raise ValueError(f"{result_path}: config conditions lack correct_frame")
    scope = str(config["trainable_scope"])
    # Runs produced before the capacity-matched random-control option existed
    # have no explicit field; those runs necessarily loaded pretrained NDF.
    initialization = str(config.get("initialization", "pretrained"))
    if str(result.get("trainable_scope")) != scope:
        raise ValueError(f"{result_path}: result/config trainable_scope mismatch")
    if str(result.get("initialization", "pretrained")) != initialization:
        raise ValueError(f"{result_path}: result/config initialization mismatch")
    if initialization == "random" and scope != "full":
        raise ValueError(f"{result_path}: random initialization requires full scope")
    parameter_count = result.get("trainable_parameters")
    if not isinstance(parameter_count, int) or parameter_count <= 0:
        raise ValueError(f"{result_path}: invalid trainable parameter count")
    episodes_per_object = config.get("train_episodes_per_object")
    if episodes_per_object is not None:
        if not isinstance(episodes_per_object, int) or episodes_per_object <= 0:
            raise ValueError(
                f"{result_path}: train_episodes_per_object must be positive or null"
            )
    signature = {key: config[key] for key in REQUIRED_HYPERPARAMETERS}
    signature["initialization"] = initialization
    signature["train_episodes_per_object"] = episodes_per_object
    signature["trainable_parameters"] = parameter_count
    return signature


def _load_candidate(
    result_path: Path,
    frame_path: Path,
    *,
    dataset: Path,
    fold: int,
    shoe_id: np.ndarray,
    episode_id: np.ndarray,
    validation_target_rotations: np.ndarray,
    validation_mask: np.ndarray,
) -> Candidate:
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    if not frame_path.is_file():
        raise FileNotFoundError(frame_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"{result_path}: result JSON must contain an object")
    if result.get("condition") != "correct_frame":
        raise ValueError(f"{result_path}: condition must be correct_frame")
    try:
        result_fold = int(result.get("fold"))
        seed = int(result.get("seed"))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{result_path}: fold and seed must be integers") from error
    if result_fold != fold:
        raise ValueError(f"{result_path}: result fold {result_fold} != requested {fold}")
    if seed < 0:
        raise ValueError(f"{result_path}: seed must be non-negative")
    semantics = result.get("frame_semantics")
    if semantics != EXPECTED_FRAME_SEMANTICS:
        raise ValueError(f"{result_path}: unexpected frame semantics")
    hyperparameters = _candidate_hyperparameters(
        result,
        result_path=result_path,
        dataset=dataset,
        fold=fold,
        seed=seed,
    )

    with np.load(frame_path, allow_pickle=False) as archive:
        missing = sorted({"frames", "shoe_id", "episode_id"} - set(archive.files))
        if missing:
            raise KeyError(f"{frame_path}: frame archive is missing {missing}")
        if not np.array_equal(shoe_id, archive["shoe_id"]):
            raise ValueError(f"{frame_path}: shoe_id alignment mismatch")
        if not np.array_equal(episode_id, archive["episode_id"]):
            raise ValueError(f"{frame_path}: episode_id alignment mismatch")
        frames = np.asarray(archive["frames"], dtype=np.float32)
    if frames.shape != (len(shoe_id), 3, 3):
        raise ValueError(f"{frame_path}: frames must have shape {(len(shoe_id), 3, 3)}")
    _validate_so3(frames, path=frame_path)

    validation = result.get("validation")
    if not isinstance(validation, dict):
        raise ValueError(f"{result_path}: validation metrics are missing")
    validation_samples = int(np.sum(validation_mask))
    try:
        reported_samples = int(validation.get("samples"))
        reported_p90 = float(validation.get("p90_deg"))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{result_path}: invalid validation samples/p90_deg") from error
    if reported_samples != validation_samples:
        raise ValueError(
            f"{result_path}: validation samples {reported_samples} != {validation_samples}"
        )
    if not np.isfinite(reported_p90) or not 0.0 <= reported_p90 <= 180.0 + 1e-3:
        raise ValueError(f"{result_path}: validation p90_deg is invalid")
    computed_error = geodesic_deg(
        frames[validation_mask], validation_target_rotations
    )
    computed_p90 = float(np.percentile(computed_error, 90.0))
    if not np.isclose(reported_p90, computed_p90, atol=2e-3, rtol=2e-5):
        raise ValueError(
            f"{result_path}: validation p90/frame mismatch: "
            f"{reported_p90} != {computed_p90}"
        )
    return Candidate(
        seed=seed,
        validation_p90_deg=reported_p90,
        result_json=_resolved(result_path),
        frame_predictions=_resolved(frame_path),
        frames=frames,
        hyperparameters=hyperparameters,
    )


def _validate_shared_hyperparameters(candidates: Sequence[Candidate]) -> dict[str, Any]:
    reference = candidates[0].hyperparameters
    for candidate in candidates[1:]:
        if candidate.hyperparameters == reference:
            continue
        differing = sorted(
            key
            for key in set(reference) | set(candidate.hyperparameters)
            if reference.get(key) != candidate.hyperparameters.get(key)
        )
        raise ValueError(
            f"seed {candidate.seed}: candidate hyperparameters differ in {differing}"
        )
    return dict(reference)


def build_selected_ensemble(
    *,
    dataset: Path,
    candidate_pairs: Sequence[tuple[Path, Path]],
    fold: int,
    confidence_percentile: float = CONFIDENCE_PERCENTILE,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load, validate, select and aggregate without writing an output file."""
    if len(candidate_pairs) != CANDIDATE_COUNT:
        raise ValueError(f"exactly {CANDIDATE_COUNT} candidate pairs are required")
    if not 0.0 < float(confidence_percentile) < 100.0:
        raise ValueError("confidence percentile must lie strictly between 0 and 100")
    resolved_results = [_resolved(pair[0]) for pair in candidate_pairs]
    resolved_frames = [_resolved(pair[1]) for pair in candidate_pairs]
    if len(set(resolved_results)) != CANDIDATE_COUNT:
        raise ValueError("candidate result JSON paths must be unique")
    if len(set(resolved_frames)) != CANDIDATE_COUNT:
        raise ValueError("candidate frame paths must be unique")
    shoe_id, episode_id, validation_rotations, masks = _load_dataset(
        dataset, fold=fold
    )
    candidates = [
        _load_candidate(
            result_path,
            frame_path,
            dataset=dataset,
            fold=fold,
            shoe_id=shoe_id,
            episode_id=episode_id,
            validation_target_rotations=validation_rotations,
            validation_mask=masks["validation"],
        )
        for result_path, frame_path in candidate_pairs
    ]
    seeds = [candidate.seed for candidate in candidates]
    if len(set(seeds)) != CANDIDATE_COUNT:
        raise ValueError(f"candidate seeds must be unique, got {seeds}")
    hyperparameters = _validate_shared_hyperparameters(candidates)

    ranked = sorted(
        candidates,
        key=lambda candidate: (candidate.validation_p90_deg, candidate.seed),
    )
    selected = ranked[:SELECTED_COUNT]
    stacked = np.stack([candidate.frames for candidate in selected], axis=0)
    ensemble = project_rotation_mean(stacked)
    disagreement = maximum_pairwise_disagreement(stacked)
    train_disagreement = disagreement[masks["train"]]
    threshold = float(np.percentile(train_disagreement, confidence_percentile))
    confidence = (disagreement <= threshold).astype(np.float32)

    payload = {
        "frames": ensemble.astype(np.float32),
        "frame_confidence": confidence,
        "frame_disagreement_deg": disagreement.astype(np.float32),
        "confidence_threshold_deg": np.asarray(threshold, dtype=np.float32),
        "shoe_id": shoe_id,
        "episode_id": episode_id,
    }
    ranks = {candidate.seed: rank for rank, candidate in enumerate(ranked, start=1)}
    selected_seeds = [candidate.seed for candidate in selected]
    candidate_records = [
        {
            "rank": int(ranks[candidate.seed]),
            "seed": int(candidate.seed),
            "validation_p90_deg": float(candidate.validation_p90_deg),
            "selected": bool(candidate.seed in selected_seeds),
            "result_json": str(candidate.result_json),
            "frame_predictions": str(candidate.frame_predictions),
        }
        for candidate in ranked
    ]
    metadata = {
        "schema_version": 1,
        "method": (
            "validation_p90_top2_projected_rotation_mean_"
            "with_max_pairwise_disagreement"
        ),
        "aggregation": "projected_mean",
        "fold": int(fold),
        "selection_split": "validation",
        "selection_metric": "p90_deg",
        "selection_top_k": SELECTED_COUNT,
        "selection_criterion": SELECTION_CRITERION,
        "test_metrics_used_for_selection": False,
        "candidate_validation_p90_deg": candidate_records,
        "selected_seeds": selected_seeds,
        "selected_result_jsons": [str(candidate.result_json) for candidate in selected],
        "frame_predictions": [
            str(candidate.frame_predictions) for candidate in selected
        ],
        "training_hyperparameters": hyperparameters,
        "confidence_calibration": "training objects only",
        "confidence_percentile": float(confidence_percentile),
        "confidence_threshold_deg": threshold,
        "acceptance": {
            name: float(confidence[mask].mean()) for name, mask in masks.items()
        },
    }
    return payload, metadata


def write_selected_ensemble(
    output: Path,
    payload: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> None:
    if output.suffix != ".npz":
        raise ValueError("output must end in .npz")
    metadata_path = output.with_suffix(".json")
    existing = [path for path in (output, metadata_path) if path.exists()]
    if existing:
        raise FileExistsError(f"refuse to overwrite existing outputs: {existing}")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parser().parse_args()
    metadata_path = args.output.with_suffix(".json")
    existing = [path for path in (args.output, metadata_path) if path.exists()]
    if existing:
        raise FileExistsError(f"refuse to overwrite existing outputs: {existing}")
    pairs = [(Path(pair[0]), Path(pair[1])) for pair in args.candidate]
    payload, metadata = build_selected_ensemble(
        dataset=args.dataset,
        candidate_pairs=pairs,
        fold=int(args.fold),
        confidence_percentile=float(args.confidence_percentile),
    )
    write_selected_ensemble(args.output, payload, metadata)
    print(json.dumps({"output": str(args.output), **metadata}))


if __name__ == "__main__":
    main()
