#!/usr/bin/env python3
"""Create a task-flow archive with a shorter action-prediction horizon."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--horizon", type=int, required=True)
    result.add_argument("--overwrite", action="store_true")
    return result


def truncate_payload(payload: dict[str, np.ndarray], horizon: int) -> dict[str, np.ndarray]:
    value = int(horizon)
    if value < 1:
        raise ValueError("horizon must be positive")
    if "action" not in payload or payload["action"].ndim != 3:
        raise ValueError("dataset action must have shape [samples,horizon,channels]")
    source_horizon = int(payload["action"].shape[1])
    if value > source_horizon:
        raise ValueError(
            f"requested horizon {value} exceeds source horizon {source_horizon}"
        )
    output = {key: np.asarray(item).copy() for key, item in payload.items()}
    output["action"] = output["action"][:, :value].copy()
    if "future_eef_pose7" in output:
        future = output["future_eef_pose7"]
        if future.ndim < 2 or future.shape[1] != source_horizon:
            raise ValueError("future_eef_pose7 horizon is inconsistent with action")
        output["future_eef_pose7"] = future[:, :value].copy()
    return output


def main() -> None:
    args = parser().parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    with np.load(args.dataset, allow_pickle=False) as archive:
        source = {key: np.asarray(archive[key]) for key in archive.files}
    output = truncate_payload(source, args.horizon)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)
    metadata = {
        "schema_version": 1,
        "method": "task_flow_action_horizon_truncation",
        "source": str(args.dataset.resolve()),
        "source_horizon": int(source["action"].shape[1]),
        "output_horizon": int(args.horizon),
        "samples": int(len(source["action"])),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), **metadata}, indent=2))


if __name__ == "__main__":
    main()
