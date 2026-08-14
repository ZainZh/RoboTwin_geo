#!/usr/bin/env python3
"""Evaluate first-step gripper probabilities along stored policy histories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .fulltask_tagrt_v2 import FullTaskTAGRTPolicy
from .train_fulltask_tagrt_v2 import (
    FullTaskDataset,
    Statistics,
    condition_batch,
    device_batch,
)


def binary_transitions(values: np.ndarray) -> list[dict[str, float | int]]:
    value = np.asarray(values, dtype=np.float64).reshape(-1)
    state = value >= 0.5
    rows = [0] + [int(row) for row in np.flatnonzero(state[1:] != state[:-1]) + 1]
    return [{"position": row, "value": float(value[row])} for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--corrections-only", action="store_true")
    parser.add_argument("--episodes", nargs="+", type=int)
    parser.add_argument(
        "--intervention", choices=("correct", "zero", "shuffled"), default="correct"
    )
    args = parser.parse_args()
    with np.load(args.dataset, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    indices = np.arange(len(payload["episode_id"]), dtype=np.int64)
    if args.corrections_only:
        marker = np.asarray(payload.get("is_correction_sample", []), dtype=np.float32)
        if marker.shape != (len(indices),):
            raise ValueError("corrections-only requires sample-aligned is_correction_sample")
        indices = indices[marker > 0.5]
    if args.episodes:
        indices = indices[
            np.isin(payload["episode_id"][indices], np.asarray(args.episodes))
        ]
    if not len(indices):
        raise ValueError("selection contains no samples")
    statistics = Statistics(**checkpoint["statistics"])
    dataset = FullTaskDataset(payload, indices, statistics)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    device = torch.device(args.device)
    model = FullTaskTAGRTPolicy(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    probabilities = []
    labels = []
    sample_indices = []
    with torch.no_grad():
        for batch in loader:
            sample_indices.append(batch["sample_index"].numpy())
            labels.append(batch["gripper"][:, 0, 0].numpy())
            conditioned = condition_batch(
                device_batch(batch, device), checkpoint["condition"], args.intervention
            )
            probabilities.append(model(conditioned).gripper_logits[:, 0, 0].sigmoid().cpu().numpy())
    sample_indices = np.concatenate(sample_indices)
    probabilities = np.concatenate(probabilities)
    labels = np.concatenate(labels)
    episodes = np.asarray(payload["episode_id"])[sample_indices]
    frames = np.asarray(payload.get("frame_index", sample_indices))[sample_indices]
    summary = {}
    for episode in np.unique(episodes):
        selected = np.flatnonzero(episodes == episode)
        order = selected[np.argsort(frames[selected])]
        summary[str(int(episode))] = {
            "samples": int(len(order)),
            "prediction_transitions": binary_transitions(probabilities[order]),
            "label_transitions": binary_transitions(labels[order]),
            "binary_accuracy": float(
                np.mean((probabilities[order] >= 0.5) == (labels[order] >= 0.5))
            ),
            "mae": float(np.mean(np.abs(probabilities[order] - labels[order]))),
        }
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset": str(args.dataset.resolve()),
        "condition": checkpoint["condition"],
        "intervention": args.intervention,
        "samples": int(len(sample_indices)),
        "binary_accuracy": float(np.mean((probabilities >= 0.5) == (labels >= 0.5))),
        "mae": float(np.mean(np.abs(probabilities - labels))),
        "episodes": summary,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
