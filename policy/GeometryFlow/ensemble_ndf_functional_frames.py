#!/usr/bin/env python3
"""Ensemble task-aligned NDF frames and calibrate epistemic confidence.

All thresholds are fitted on training-object predictions only.  Test pose
labels are never used to form either the ensemble frame or its confidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .train_ndf_functional_frame_head import resolve_object_split


_PROPER_AXIS_SIGN_BRANCHES = np.asarray(
    (
        np.diag((1.0, 1.0, 1.0)),
        np.diag((1.0, -1.0, -1.0)),
        np.diag((-1.0, 1.0, -1.0)),
        np.diag((-1.0, -1.0, 1.0)),
    ),
    dtype=np.float32,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", type=Path, required=True)
    result.add_argument(
        "--frame-predictions", action="append", type=Path, required=True
    )
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--fold", type=int, required=True)
    result.add_argument("--train-ids", nargs="+", type=int)
    result.add_argument("--validation-ids", nargs="+", type=int)
    result.add_argument("--test-ids", nargs="+", type=int)
    result.add_argument("--confidence-percentile", type=float, default=95.0)
    result.add_argument(
        "--aggregation",
        choices=("projected_mean", "medoid"),
        default="projected_mean",
        help=(
            "SO(3) projected arithmetic mean, or the member minimizing summed "
            "pairwise geodesic distance (robust to one flipped member)."
        ),
    )
    result.add_argument("--overwrite", action="store_true")
    return result


def geodesic_deg(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    relative = np.asarray(first) @ np.swapaxes(np.asarray(second), -1, -2)
    return np.rad2deg(
        Rotation.from_matrix(relative.reshape(-1, 3, 3)).magnitude()
    ).reshape(relative.shape[:-2])


def project_rotation_mean(frames: np.ndarray) -> np.ndarray:
    value = np.asarray(frames, dtype=np.float64)
    if value.ndim != 4 or value.shape[-2:] != (3, 3):
        raise ValueError(f"frames must be [models,samples,3,3], got {value.shape}")
    left, _, right = np.linalg.svd(value.mean(axis=0))
    rotation = left @ right
    reflection = np.linalg.det(rotation) < 0.0
    left[reflection, :, -1] *= -1.0
    return (left @ right).astype(np.float32)


def maximum_pairwise_disagreement(frames: np.ndarray) -> np.ndarray:
    value = np.asarray(frames)
    if len(value) < 2:
        raise ValueError("functional-frame confidence requires at least two models")
    pairwise = [
        geodesic_deg(value[first], value[second])
        for first in range(len(value))
        for second in range(first + 1, len(value))
    ]
    return np.max(np.stack(pairwise, axis=0), axis=0).astype(np.float32)


def rotation_medoid(frames: np.ndarray) -> np.ndarray:
    """Select the ensemble member closest to all other members on SO(3)."""

    value = np.asarray(frames, dtype=np.float32)
    if value.ndim != 4 or value.shape[-2:] != (3, 3):
        raise ValueError(f"frames must be [models,samples,3,3], got {value.shape}")
    if len(value) < 2:
        raise ValueError("rotation medoid requires at least two models")
    score = np.zeros(value.shape[:2], dtype=np.float64)
    for first in range(len(value)):
        for second in range(first + 1, len(value)):
            distance = geodesic_deg(value[first], value[second])
            score[first] += distance
            score[second] += distance
    winner = np.argmin(score, axis=0)
    return value[winner, np.arange(value.shape[1])].astype(np.float32)


def stabilize_rotation_against_previous(
    rotation: np.ndarray,
    previous: np.ndarray | None,
    *,
    jump_trigger_deg: float = 120.0,
    alternative_accept_deg: float = 45.0,
) -> tuple[np.ndarray, bool, bool, float | None]:
    """Track a signed SO(3) branch using only the preceding camera estimate.

    The four candidates are the proper rotations obtained by flipping zero or
    two local axes.  A jump larger than ``jump_trigger_deg`` is always marked
    ambiguous.  When a sign-equivalent branch is within
    ``alternative_accept_deg`` of the previous estimate, that continuous
    branch is returned but remains confidence-masked for the current frame.
    No simulator pose or future frame is consulted.
    """

    current = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
    if previous is None:
        return current.copy(), False, False, None
    reference = np.asarray(previous, dtype=np.float32).reshape(3, 3)
    trigger = float(jump_trigger_deg)
    acceptance = float(alternative_accept_deg)
    if not 0.0 < acceptance < trigger < 180.0:
        raise ValueError(
            "temporal thresholds must satisfy 0 < alternative < trigger < 180"
        )
    raw_jump = float(geodesic_deg(current[None], reference[None])[0])
    if raw_jump <= trigger:
        return current.copy(), False, False, raw_jump
    candidates = np.einsum(
        "ij,kjl->kil", current, _PROPER_AXIS_SIGN_BRANCHES
    ).astype(np.float32)
    distances = geodesic_deg(candidates, np.broadcast_to(reference, candidates.shape))
    winner = int(np.argmin(distances))
    corrected = bool(winner != 0 and float(distances[winner]) < acceptance)
    selected = candidates[winner] if corrected else current
    return selected.astype(np.float32), True, corrected, raw_jump


def stabilize_rotation_sequences(
    rotations: np.ndarray,
    episode_id: np.ndarray,
    frame_index: np.ndarray,
    *,
    jump_trigger_deg: float = 120.0,
    alternative_accept_deg: float = 45.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply causal branch tracking independently inside every episode."""

    values = np.asarray(rotations, dtype=np.float32)
    episodes = np.asarray(episode_id, dtype=np.int64).reshape(-1)
    frames = np.asarray(frame_index, dtype=np.int64).reshape(-1)
    if values.shape != (len(episodes), 3, 3) or len(frames) != len(episodes):
        raise ValueError("rotations, episode_id, and frame_index are misaligned")
    output = values.copy()
    ambiguous = np.zeros(len(values), dtype=bool)
    corrected = np.zeros(len(values), dtype=bool)
    for episode in np.unique(episodes):
        rows = np.flatnonzero(episodes == int(episode))
        rows = rows[np.argsort(frames[rows], kind="stable")]
        previous = None
        for row in rows:
            selected, is_ambiguous, was_corrected, _jump = (
                stabilize_rotation_against_previous(
                    values[row],
                    previous,
                    jump_trigger_deg=jump_trigger_deg,
                    alternative_accept_deg=alternative_accept_deg,
                )
            )
            output[row] = selected
            ambiguous[row] = is_ambiguous
            corrected[row] = was_corrected
            # A corrected signed branch is continuous and can safely anchor
            # the next observation.  An unexplained large jump cannot.
            if not is_ambiguous or was_corrected:
                previous = selected
    return output, ambiguous, corrected


def main() -> None:
    args = parser().parse_args()
    object_split = resolve_object_split(
        int(args.fold),
        train_ids=args.train_ids,
        validation_ids=args.validation_ids,
        test_ids=args.test_ids,
    )
    if not 0.0 < args.confidence_percentile < 100.0:
        raise ValueError("confidence percentile must lie strictly between 0 and 100")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    with np.load(args.dataset, allow_pickle=False) as archive:
        shoe_id = np.asarray(archive["shoe_id"], dtype=np.int64)
        episode_id = np.asarray(archive["episode_id"], dtype=np.int64)
    frames = []
    for path in args.frame_predictions:
        with np.load(path, allow_pickle=False) as archive:
            if "shoe_id" in archive and not np.array_equal(shoe_id, archive["shoe_id"]):
                raise ValueError(f"shoe_id alignment mismatch in {path}")
            if "episode_id" in archive and not np.array_equal(
                episode_id, archive["episode_id"]
            ):
                raise ValueError(f"episode_id alignment mismatch in {path}")
            value = np.asarray(archive["frames"], dtype=np.float32)
            if value.shape != (len(shoe_id), 3, 3):
                raise ValueError(f"unexpected frame shape {value.shape} in {path}")
            frames.append(value)
    stacked = np.stack(frames, axis=0)
    ensemble = (
        project_rotation_mean(stacked)
        if args.aggregation == "projected_mean"
        else rotation_medoid(stacked)
    )
    disagreement = maximum_pairwise_disagreement(stacked)
    train_mask = np.isin(
        shoe_id,
        np.asarray(object_split["train"], dtype=np.int64),
    )
    threshold = float(
        np.percentile(disagreement[train_mask], args.confidence_percentile)
    )
    confidence = (disagreement <= threshold).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        frames=ensemble,
        frame_confidence=confidence,
        frame_disagreement_deg=disagreement,
        confidence_threshold_deg=np.asarray(threshold, dtype=np.float32),
        shoe_id=shoe_id,
        episode_id=episode_id,
    )
    metadata = {
        "schema_version": 1,
        "method": f"{args.aggregation}_with_max_pairwise_disagreement",
        "aggregation": str(args.aggregation),
        "frame_predictions": [str(path.resolve()) for path in args.frame_predictions],
        "fold": int(args.fold),
        "confidence_calibration": "training objects only",
        "confidence_percentile": float(args.confidence_percentile),
        "confidence_threshold_deg": threshold,
        "acceptance": {
            split: float(
                confidence[
                    np.isin(shoe_id, np.asarray(ids, dtype=np.int64))
                ].mean()
            )
            for split, ids in object_split.items()
        },
        "object_split": {name: list(ids) for name, ids in object_split.items()},
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), **metadata}))


if __name__ == "__main__":
    main()
