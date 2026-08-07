#!/usr/bin/env python3
"""Create train-calibrated confidence variants without changing frame tokens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .train_task_flow_benchmark import FOLDS


VARIANTS = ("always_on", "train_p99_hard", "soft_continuous")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--fold", type=int, default=0)
    return result


def output_path(output_dir: Path, variant: str) -> Path:
    return output_dir / f"camera_confidence_{variant}.npz"


def sigmoid_confidence(
    disagreement_deg: np.ndarray,
    *,
    threshold_deg: float,
    scale_deg: float,
) -> np.ndarray:
    value = (np.asarray(disagreement_deg, dtype=np.float64) - threshold_deg) / scale_deg
    return (1.0 / (1.0 + np.exp(np.clip(value, -60.0, 60.0)))).astype(
        np.float32
    )


def build_variant_payload(
    payload: dict[str, np.ndarray],
    *,
    variant: str,
    target_confidence: np.ndarray,
    source_confidence: np.ndarray,
    hard_threshold_deg: float,
    soft_scale_deg: float,
) -> dict[str, np.ndarray]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown confidence variant {variant!r}")
    output = dict(payload)
    output["target_frame_confidence_original"] = np.asarray(
        payload.get(
            "target_frame_confidence",
            payload["goal_frame_confidence"],
        ),
        dtype=np.float32,
    )
    output["source_frame_confidence_variant"] = np.asarray(
        source_confidence, dtype=np.float32
    )
    output["target_frame_confidence"] = np.asarray(
        target_confidence, dtype=np.float32
    )
    output["confidence_hard_threshold_deg"] = np.asarray(
        hard_threshold_deg, dtype=np.float32
    )
    output["confidence_soft_scale_deg"] = np.asarray(
        soft_scale_deg, dtype=np.float32
    )
    return output


def confidence_variants(
    payload: dict[str, np.ndarray], fold: int
) -> tuple[dict[str, np.ndarray], dict]:
    if fold < 0 or fold >= len(FOLDS):
        raise ValueError(f"fold must lie in [0,{len(FOLDS) - 1}]")
    required = {
        "shoe_id",
        "target_frame9",
        "source_frame_disagreement_deg",
        "goal_frame_confidence",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise KeyError(f"camera-confidence dataset is missing fields: {missing}")
    samples = int(len(payload["shoe_id"]))
    shoe_id = np.asarray(payload["shoe_id"], dtype=np.int64)
    disagreement = np.asarray(
        payload["source_frame_disagreement_deg"], dtype=np.float64
    )
    goal_confidence = np.asarray(payload["goal_frame_confidence"], dtype=np.float64)
    target_frame = np.asarray(payload["target_frame9"])
    if disagreement.shape != (samples,):
        raise ValueError(
            f"source_frame_disagreement_deg must be {(samples,)}, got {disagreement.shape}"
        )
    if goal_confidence.shape != (samples,):
        raise ValueError(
            f"goal_frame_confidence must be {(samples,)}, got {goal_confidence.shape}"
        )
    if target_frame.shape != (samples, 9):
        raise ValueError(f"target_frame9 must be {(samples, 9)}, got {target_frame.shape}")
    if not np.all(np.isfinite(disagreement)) or np.any(disagreement < 0.0):
        raise ValueError("source-frame disagreement must be finite and non-negative")
    if not np.all(np.isfinite(goal_confidence)) or np.any(
        (goal_confidence < 0.0) | (goal_confidence > 1.0)
    ):
        raise ValueError("goal-frame confidence must be finite and lie in [0,1]")

    train_mask = np.isin(
        shoe_id, np.asarray(FOLDS[int(fold)]["train"], dtype=np.int64)
    )
    if not np.any(train_mask):
        raise ValueError("fold has no training-object samples for confidence calibration")
    train = disagreement[train_mask]
    percentiles = {
        name: float(np.percentile(train, percentile))
        for name, percentile in (("p50", 50.0), ("p90", 90.0), ("p95", 95.0), ("p99", 99.0))
    }
    threshold = percentiles["p99"]
    # The soft gate has the same 0.5 point as the hard p99 gate. Its scale is
    # train-calibrated so that train-p95 maps to 0.9 confidence. Degenerate
    # distributions fall back to a train-derived robust spread.
    scale = (percentiles["p99"] - percentiles["p95"]) / np.log(9.0)
    scale_fallback = False
    if not np.isfinite(scale) or scale <= 1e-6:
        q25, q75 = np.percentile(train, (25.0, 75.0))
        scale = max(float(q75 - q25) / 1.349, float(np.std(train)), 0.1)
        scale_fallback = True

    source_confidence = {
        "always_on": np.ones(samples, dtype=np.float32),
        "train_p99_hard": (disagreement <= threshold).astype(np.float32),
        "soft_continuous": sigmoid_confidence(
            disagreement,
            threshold_deg=threshold,
            scale_deg=scale,
        ),
    }
    variants = {
        name: (
            np.asarray(source_confidence[name], dtype=np.float32)
            * goal_confidence.astype(np.float32)
        )
        for name in VARIANTS
    }
    diagnostics = {
        "fold": int(fold),
        "calibration": "source disagreement from fold training objects only",
        "train_object_ids": [int(value) for value in FOLDS[int(fold)]["train"]],
        "train_samples": int(train_mask.sum()),
        "train_disagreement_percentiles_deg": percentiles,
        "hard_threshold_percentile": 99.0,
        "hard_threshold_deg": float(threshold),
        "soft_midpoint_deg": float(threshold),
        "soft_scale_deg": float(scale),
        "soft_scale_rule": "(train_p99-train_p95)/ln(9); robust train fallback",
        "soft_scale_fallback": bool(scale_fallback),
        "goal_confidence_preserved": True,
        "coverage": {
            name: {
                split: float(
                    variants[name][
                        np.isin(shoe_id, np.asarray(ids, dtype=np.int64))
                    ].mean()
                )
                for split, ids in FOLDS[int(fold)].items()
            }
            for name in VARIANTS
        },
    }
    return variants, diagnostics


def main() -> None:
    args = parser().parse_args()
    paths = {name: output_path(args.output_dir, name) for name in VARIANTS}
    metadata_paths = {name: path.with_suffix(".json") for name, path in paths.items()}
    existing = [
        path
        for path in (*paths.values(), *metadata_paths.values())
        if path.exists()
    ]
    if existing:
        raise FileExistsError(f"refuse to overwrite confidence variants: {existing}")
    with np.load(args.dataset, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    variants, diagnostics = confidence_variants(payload, int(args.fold))
    source_metadata_path = args.dataset.with_suffix(".json")
    source_metadata = (
        json.loads(source_metadata_path.read_text(encoding="utf-8"))
        if source_metadata_path.is_file()
        else {}
    )
    metadata_fold = source_metadata.get("fold")
    if metadata_fold is not None and int(metadata_fold) != int(args.fold):
        raise ValueError(
            f"source metadata fold mismatch: requested {args.fold}, found {metadata_fold}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    disagreement = np.asarray(
        payload["source_frame_disagreement_deg"], dtype=np.float32
    )
    source_confidence = {
        "always_on": np.ones(len(disagreement), dtype=np.float32),
        "train_p99_hard": (
            disagreement <= float(diagnostics["hard_threshold_deg"])
        ).astype(np.float32),
        "soft_continuous": sigmoid_confidence(
            disagreement,
            threshold_deg=float(diagnostics["soft_midpoint_deg"]),
            scale_deg=float(diagnostics["soft_scale_deg"]),
        ),
    }
    for name in VARIANTS:
        output = build_variant_payload(
            payload,
            variant=name,
            target_confidence=variants[name],
            source_confidence=source_confidence[name],
            hard_threshold_deg=float(diagnostics["hard_threshold_deg"]),
            soft_scale_deg=float(diagnostics["soft_scale_deg"]),
        )
        np.savez_compressed(paths[name], **output)
        metadata = dict(source_metadata)
        metadata.update(
            {
                "schema_version": max(int(source_metadata.get("schema_version", 1)), 6),
                "output": str(paths[name].resolve()),
                "confidence_variant_source_dataset": str(args.dataset.resolve()),
                "confidence_variant": name,
                "target_frame_confidence": (
                    "goal-frame reliability multiplied by the declared "
                    "train-calibrated source-frame confidence variant"
                ),
                **diagnostics,
            }
        )
        metadata_paths[name].write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "output": str(paths[name]),
                    "variant": name,
                    "hard_threshold_deg": diagnostics["hard_threshold_deg"],
                    "soft_scale_deg": diagnostics["soft_scale_deg"],
                    "coverage": diagnostics["coverage"][name],
                }
            )
        )


if __name__ == "__main__":
    main()
