#!/usr/bin/env python3
"""Attach the label-only intended relative SE(3) token to a task-flow dataset.

This produces an explicitly privileged oracle ceiling.  Simulator-derived goal
errors are training/evaluation labels and the resulting archive must never be
described as deployable or camera-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--overwrite", action="store_true")
    return result


def oracle_relative_frame(payload: dict[str, np.ndarray]) -> np.ndarray:
    required = {
        "goal_translation_error_xyz_m",
        "goal_rotation_error_rotvec",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise KeyError(f"source dataset is missing fields: {missing}")
    translation = np.asarray(
        payload["goal_translation_error_xyz_m"], dtype=np.float64
    )
    rotvec = np.asarray(payload["goal_rotation_error_rotvec"], dtype=np.float64)
    if translation.ndim != 2 or translation.shape[1] != 3:
        raise ValueError(
            "goal_translation_error_xyz_m must have shape [samples,3]"
        )
    if rotvec.shape != translation.shape:
        raise ValueError("goal_rotation_error_rotvec must match translation shape")
    rotation = Rotation.from_rotvec(rotvec).as_matrix()
    return np.concatenate(
        (translation, rotation[..., :, 0], rotation[..., :, 1]), axis=-1
    ).astype(np.float32)


def main() -> None:
    args = parser().parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    with np.load(args.dataset, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    payload["target_frame9"] = oracle_relative_frame(payload)
    payload["target_frame_confidence"] = np.ones(
        len(payload["target_frame9"]), dtype=np.float32
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)

    source_metadata = args.dataset.with_suffix(".json")
    metadata = (
        json.loads(source_metadata.read_text(encoding="utf-8"))
        if source_metadata.is_file()
        else {}
    )
    metadata.update(
        {
            "schema_version": max(int(metadata.get("schema_version", 1)), 7),
            "functional_frame_encoding": "oracle_intended_relative_se3_columns_v1",
            "functional_frame_source": "simulator goal-error labels",
            "deployable": False,
            "oracle_ceiling_only": True,
            "source_dataset": str(args.dataset.resolve()),
        }
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "samples": int(len(payload["target_frame9"])),
                "deployable": False,
            }
        )
    )


if __name__ == "__main__":
    main()
