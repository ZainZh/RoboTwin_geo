#!/usr/bin/env python3
"""Create a validated, zero-copy episode view for a focused policy dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from .build_task_flow_dataset import identify_active_arm


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-data-dir", type=Path, required=True)
    result.add_argument("--output-data-dir", type=Path, required=True)
    result.add_argument("--episode-indices", type=int, nargs="+", required=True)
    result.add_argument("--require-active-arm", choices=("left", "right"))
    return result


def inspect_episode(path: Path) -> tuple[int, str]:
    with h5py.File(path, "r") as archive:
        shoe_id = int(np.asarray(archive["task_state/shoe_id"][0]).reshape(-1)[0])
        active_arm = identify_active_arm(
            np.asarray(archive["endpose/left_gripper"]),
            np.asarray(archive["endpose/right_gripper"]),
        )
    return shoe_id, active_arm


def main() -> None:
    args = parser().parse_args()
    indices = [int(value) for value in args.episode_indices]
    if len(set(indices)) != len(indices):
        raise ValueError("episode indices must be unique")
    if any(value < 0 for value in indices):
        raise ValueError("episode indices must be non-negative")
    source_dir = args.source_data_dir.expanduser().resolve()
    output_dir = args.output_data_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"output data directory is not empty: {output_dir}")

    rows = []
    for canonical_index, source_index in enumerate(indices):
        source = source_dir / f"episode{source_index}.hdf5"
        if not source.is_file():
            raise FileNotFoundError(source)
        shoe_id, active_arm = inspect_episode(source)
        if args.require_active_arm and active_arm != args.require_active_arm:
            raise ValueError(
                f"episode {source_index} uses {active_arm}, expected {args.require_active_arm}"
            )
        target = output_dir / f"episode{canonical_index}.hdf5"
        target.symlink_to(source)
        rows.append(
            {
                "canonical_episode": canonical_index,
                "source_episode": source_index,
                "shoe_id": shoe_id,
                "active_arm": active_arm,
            }
        )

    counts = Counter((row["shoe_id"], row["active_arm"]) for row in rows)
    metadata = {
        "schema_version": 1,
        "source_data_dir": str(source_dir),
        "output_data_dir": str(output_dir),
        "episodes": rows,
        "counts": {
            f"shoe{shoe_id}_{arm}": count
            for (shoe_id, arm), count in sorted(counts.items())
        },
    }
    manifest = output_dir.parent / "episode_view.json"
    manifest.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
