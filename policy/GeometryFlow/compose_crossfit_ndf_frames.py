#!/usr/bin/env python3
"""Compose leakage-free policy-training NDF frames.

Training-object rows use leave-one-object-out NDF predictions, while
validation and blind-test rows use the final NDF ensemble trained on every
training object.  This gives the downstream policy the same out-of-object
representation error during training that it will encounter at deployment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .ensemble_ndf_functional_frames import (
    maximum_pairwise_disagreement,
    project_rotation_mean,
    rotation_medoid,
)
from .functional_action_frame import pose9_rotation
from .train_ndf_functional_frame_head import resolve_object_split
from .augment_handled_mug_ndf_relation import rotation_error_deg


def compose_crossfit_frames(
    shoe_id: np.ndarray,
    full_members: np.ndarray,
    crossfit_members: dict[int, np.ndarray],
    train_ids: tuple[int, ...],
    *,
    confidence_percentile: float,
    aggregation: str = "projected_mean",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return composite frames, confidence, disagreement, and threshold."""

    identities = np.asarray(shoe_id, dtype=np.int64)
    full = np.asarray(full_members, dtype=np.float32)
    if full.ndim != 4 or full.shape[1:] != (len(identities), 3, 3):
        raise ValueError(
            "full_members must have shape [models,samples,3,3], got "
            f"{full.shape}"
        )
    if not 0.0 < float(confidence_percentile) < 100.0:
        raise ValueError("confidence_percentile must lie strictly between 0 and 100")
    if aggregation not in {"projected_mean", "medoid"}:
        raise ValueError("aggregation must be projected_mean or medoid")
    aggregate = project_rotation_mean if aggregation == "projected_mean" else rotation_medoid
    frames = aggregate(full)
    disagreement = maximum_pairwise_disagreement(full)
    for object_id in train_ids:
        if int(object_id) not in crossfit_members:
            raise KeyError(f"missing cross-fit predictions for object {object_id}")
        members = np.asarray(crossfit_members[int(object_id)], dtype=np.float32)
        if members.shape != full.shape:
            raise ValueError(
                f"cross-fit members for object {object_id} have {members.shape}, "
                f"expected {full.shape}"
            )
        keep = identities == int(object_id)
        if not np.any(keep):
            raise ValueError(f"training object {object_id} has no dataset rows")
        frames[keep] = aggregate(members)[keep]
        disagreement[keep] = maximum_pairwise_disagreement(members)[keep]
    train_mask = np.isin(identities, np.asarray(train_ids, dtype=np.int64))
    threshold = float(
        np.percentile(disagreement[train_mask], float(confidence_percentile))
    )
    confidence = (disagreement <= threshold).astype(np.float32)
    return frames, confidence, disagreement, threshold


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", type=Path, required=True)
    result.add_argument(
        "--full-frame-predictions", action="append", type=Path, required=True
    )
    result.add_argument("--crossfit-root", type=Path)
    result.add_argument(
        "--crossfit-directory-template",
        default="ndf_two_axis_crossfit_holdout{object_id}_v1",
        help="Directory name below crossfit-root; must contain {object_id}.",
    )
    result.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    result.add_argument(
        "--crossfit-object-predictions",
        action="append",
        nargs="+",
        default=[],
        metavar="VALUE",
        help=(
            "Explicit OBJECT_ID FRAME_PATH FRAME_PATH [...] mapping. Repeat "
            "once per training object when held-out predictions come from "
            "different folds."
        ),
    )
    result.add_argument("--fold", type=int, default=0)
    result.add_argument("--train-ids", nargs="+", type=int, required=True)
    result.add_argument("--validation-ids", nargs="+", type=int, required=True)
    result.add_argument("--test-ids", nargs="+", type=int, required=True)
    result.add_argument("--confidence-percentile", type=float, default=99.0)
    result.add_argument(
        "--aggregation",
        choices=("projected_mean", "medoid"),
        default="medoid",
        help="Use medoid to prevent one flipped ensemble member corrupting a frame.",
    )
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--overwrite", action="store_true")
    return result


def _load_members(
    paths: list[Path], shoe_id: np.ndarray, episode_id: np.ndarray
) -> np.ndarray:
    values = []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            if "shoe_id" in archive and not np.array_equal(
                shoe_id, np.asarray(archive["shoe_id"], dtype=np.int64)
            ):
                raise ValueError(f"shoe_id alignment mismatch in {path}")
            if "episode_id" in archive and not np.array_equal(
                episode_id, np.asarray(archive["episode_id"], dtype=np.int64)
            ):
                raise ValueError(f"episode_id alignment mismatch in {path}")
            value = np.asarray(archive["frames"], dtype=np.float32)
        if value.shape != (len(shoe_id), 3, 3):
            raise ValueError(f"unexpected frame shape {value.shape} in {path}")
        values.append(value)
    if len(values) < 2:
        raise ValueError("at least two ensemble members are required")
    return np.stack(values, axis=0)


def parse_object_prediction_groups(
    groups: list[list[str]], train_ids: tuple[int, ...]
) -> dict[int, list[Path]]:
    """Validate explicit per-object cross-fold frame member paths."""

    parsed: dict[int, list[Path]] = {}
    for group in groups:
        if len(group) < 3:
            raise ValueError(
                "each crossfit-object-predictions group needs an object ID "
                "and at least two frame archives"
            )
        object_id = int(group[0])
        if object_id in parsed:
            raise ValueError(f"duplicate explicit predictions for object {object_id}")
        parsed[object_id] = [Path(value) for value in group[1:]]
    expected = {int(value) for value in train_ids}
    actual = set(parsed)
    if actual != expected:
        raise ValueError(
            "explicit crossfit object IDs must exactly match training IDs; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return parsed


def _summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90.0)),
        "max": float(array.max()),
    }


def main() -> None:
    args = parser().parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    split = resolve_object_split(
        int(args.fold),
        train_ids=args.train_ids,
        validation_ids=args.validation_ids,
        test_ids=args.test_ids,
    )
    with np.load(args.dataset, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    shoe_id = np.asarray(payload["shoe_id"], dtype=np.int64)
    episode_id = np.asarray(payload["episode_id"], dtype=np.int64)
    full_members = _load_members(
        list(args.full_frame_predictions), shoe_id, episode_id
    )
    crossfit_members = {}
    crossfit_paths = {}
    if args.crossfit_object_predictions:
        explicit_paths = parse_object_prediction_groups(
            args.crossfit_object_predictions, split["train"]
        )
    else:
        if args.crossfit_root is None:
            raise ValueError(
                "provide crossfit-root or explicit crossfit-object-predictions"
            )
        if "{object_id}" not in args.crossfit_directory_template:
            raise ValueError(
                "crossfit-directory-template must contain the {object_id} placeholder"
            )
        explicit_paths = {
            int(object_id): [
                args.crossfit_root
                / args.crossfit_directory_template.format(object_id=object_id)
                / f"fold{args.fold}_correct_frame_seed{seed}_frames.npz"
                for seed in args.seeds
            ]
            for object_id in split["train"]
        }
    for object_id in split["train"]:
        paths = explicit_paths[int(object_id)]
        crossfit_members[int(object_id)] = _load_members(
            paths, shoe_id, episode_id
        )
        crossfit_paths[str(object_id)] = [str(path.resolve()) for path in paths]
    frames, confidence, disagreement, threshold = compose_crossfit_frames(
        shoe_id,
        full_members,
        crossfit_members,
        split["train"],
        confidence_percentile=args.confidence_percentile,
        aggregation=args.aggregation,
    )
    true_rotation = pose9_rotation(payload["current_object_pose9"])
    frame_error = rotation_error_deg(frames, true_rotation)
    diagnostics = {}
    for name, ids in split.items():
        keep = np.isin(shoe_id, np.asarray(ids, dtype=np.int64))
        diagnostics[name] = {
            "samples": int(keep.sum()),
            "rotation_error_deg": _summary(frame_error[keep]),
            "ensemble_disagreement_deg": _summary(disagreement[keep]),
            "confidence_acceptance": float(confidence[keep].mean()),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        frames=frames.astype(np.float32),
        frame_confidence=confidence,
        frame_disagreement_deg=disagreement.astype(np.float32),
        confidence_threshold_deg=np.asarray(threshold, dtype=np.float32),
        shoe_id=shoe_id,
        episode_id=episode_id,
    )
    metadata = {
        "schema_version": 1,
        "method": (
            "leave-one-object-out cross-fitted NDF training tokens with "
            f"{args.aggregation} aggregation"
        ),
        "aggregation": str(args.aggregation),
        "policy_train_representation": "NDF ensemble excluding the row's object",
        "validation_test_representation": "final NDF ensemble trained on all training objects",
        "confidence_calibration": "cross-fitted training-object disagreement only",
        "confidence_percentile": float(args.confidence_percentile),
        "confidence_threshold_deg": threshold,
        "object_split": {name: list(ids) for name, ids in split.items()},
        "full_frame_predictions": [
            str(path.resolve()) for path in args.full_frame_predictions
        ],
        "crossfit_frame_predictions": crossfit_paths,
        "diagnostics_label_only": diagnostics,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "diagnostics": diagnostics}))


if __name__ == "__main__":
    main()
