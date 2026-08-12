#!/usr/bin/env python3
"""Merge nominal and correction full-task dense archives safely.

Correction episodes are reindexed after the nominal episode range and marked
for train-only splitting/weighting.  Observation/action tensors are otherwise
copied unchanged so Raw and TAGRT receive exactly the same additional states.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def merge(
    nominal: dict[str, np.ndarray], correction: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    if set(nominal) != set(correction):
        raise ValueError(
            "nominal/correction archives must have identical keys; "
            f"nominal-only={sorted(set(nominal) - set(correction))}, "
            f"correction-only={sorted(set(correction) - set(nominal))}"
        )
    nominal_samples = len(nominal["episode_id"])
    correction_samples = len(correction["episode_id"])
    result = {}
    correction_episode = np.asarray(correction["episode_id"], dtype=np.int64)
    nominal_episode = np.asarray(nominal["episode_id"], dtype=np.int64)
    episode_offset = int(nominal_episode.max()) + 1
    correction_reindexed = (
        correction_episode - int(correction_episode.min()) + episode_offset
    )
    for key in nominal:
        first = np.asarray(nominal[key])
        second = np.asarray(correction[key])
        if len(first) != nominal_samples or len(second) != correction_samples:
            raise ValueError(f"field {key} is not sample-aligned")
        if first.shape[1:] != second.shape[1:]:
            raise ValueError(
                f"field {key} shape mismatch: {first.shape} != {second.shape}"
            )
        if key == "episode_id":
            second = correction_reindexed.astype(first.dtype, copy=False)
        result[key] = np.concatenate(
            (first, second.astype(first.dtype, copy=False)), axis=0
        )
    result["is_correction_sample"] = np.concatenate(
        (
            np.zeros(nominal_samples, dtype=np.float32),
            np.ones(correction_samples, dtype=np.float32),
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nominal", type=Path, required=True)
    parser.add_argument("--correction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    nominal = load(args.nominal)
    correction = load(args.correction)
    output = merge(nominal, correction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)
    correction_episode_ids = np.unique(
        output["episode_id"][output["is_correction_sample"] > 0.5]
    )
    metadata = {
        "schema_version": 1,
        "nominal": str(args.nominal.resolve()),
        "correction": str(args.correction.resolve()),
        "output": str(args.output.resolve()),
        "nominal_samples": int(len(nominal["episode_id"])),
        "correction_samples": int(len(correction["episode_id"])),
        "samples": int(len(output["episode_id"])),
        "nominal_episodes": int(len(np.unique(nominal["episode_id"]))),
        "correction_episodes": int(len(correction_episode_ids)),
        "correction_episode_ids": correction_episode_ids.astype(int).tolist(),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
