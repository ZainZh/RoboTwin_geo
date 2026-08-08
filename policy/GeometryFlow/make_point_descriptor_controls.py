#!/usr/bin/env python3
"""Create capacity-matched controls for appended point descriptors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", type=Path, required=True)
    result.add_argument("--output-prefix", type=Path, required=True)
    result.add_argument("--seed", type=int, default=73)
    result.add_argument("--overwrite", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    with np.load(args.dataset, allow_pickle=False) as archive:
        source = {key: archive[key] for key in archive.files}
    for key in ("points_a", "points_b"):
        if key not in source or source[key].ndim != 3 or source[key].shape[-1] <= 3:
            raise ValueError(f"{key} must contain [XYZ, descriptor]")

    outputs = {
        "zero": args.output_prefix.parent / f"{args.output_prefix.name}_zero.npz",
        "shuffled": args.output_prefix.parent
        / f"{args.output_prefix.name}_shuffled.npz",
    }
    if not args.overwrite:
        existing = [str(path) for path in outputs.values() if path.exists()]
        if existing:
            raise FileExistsError(existing)

    rng = np.random.default_rng(int(args.seed))
    payloads = {
        name: {key: value.copy() for key, value in source.items()}
        for name in outputs
    }
    for key in ("points_a", "points_b"):
        payloads["zero"][key][..., 3:] = 0.0
        shuffled = payloads["shuffled"][key]
        for sample in range(shuffled.shape[0]):
            shuffled[sample, :, 3:] = shuffled[
                sample, rng.permutation(shuffled.shape[1]), 3:
            ]

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for control, path in outputs.items():
        np.savez_compressed(path, **payloads[control])
        metadata = {
            "schema_version": 1,
            "source": str(args.dataset.resolve()),
            "output": str(path.resolve()),
            "control": control,
            "seed": int(args.seed),
            "descriptor_dim": int(source["points_a"].shape[-1] - 3),
            "shuffle_scope": "within_each_object_cloud" if control == "shuffled" else None,
        }
        path.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
