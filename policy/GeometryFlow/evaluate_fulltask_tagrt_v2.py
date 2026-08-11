#!/usr/bin/env python3
"""Re-evaluate a full-task checkpoint under causal geometry interventions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .fulltask_tagrt_v2 import FullTaskTAGRTPolicy
from .train_fulltask_tagrt_v2 import FOLD0, FullTaskDataset, Statistics, evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--flow-steps", type=int, default=12)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    with np.load(args.dataset, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    statistics = Statistics(**checkpoint["statistics"])
    test_indices = np.flatnonzero(
        np.isin(payload["shoe_id"], np.asarray(FOLD0["test"]))
    )
    dataset = FullTaskDataset(payload, test_indices, statistics)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    device = torch.device(args.device)
    model = FullTaskTAGRTPolicy(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    condition = str(checkpoint["condition"])
    interventions = (
        ("correct", "zero", "shuffled")
        if condition.startswith("tagrt")
        else ("correct",)
    )
    result = {
        intervention: evaluate(
            model,
            loader,
            condition,
            intervention,
            device,
            statistics,
            args.flow_steps,
        )
        for intervention in interventions
    }
    payload_out = {
        "dataset": str(args.dataset.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "condition": condition,
        "test_shoe_ids": list(FOLD0["test"]),
        "cross_episode_mismatch": True,
        "result": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload_out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload_out), flush=True)


if __name__ == "__main__":
    main()
