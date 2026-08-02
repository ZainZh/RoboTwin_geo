#!/usr/bin/env python3
"""Evaluate one frozen flow-policy checkpoint under causal flow interventions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .evaluate_predicted_flow_actions import load_prediction, summarize_phases
from .train_task_flow_benchmark import (
    FOLDS,
    Normalization,
    TaskFlowDataset,
    build_action_predictor,
    predict,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", type=Path, required=True)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--fold", type=int, required=True)
    result.add_argument(
        "--prediction", action="append", nargs=2, metavar=("NAME", "PATH")
    )
    result.add_argument(
        "--sample-phase", choices=("all", "pre_relation", "relation"), default="all"
    )
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    result.add_argument("--batch-size", type=int, default=512)
    result.add_argument("--num-workers", type=int, default=0)
    return result


def prediction_with_optional_suffix(
    path: Path, episode_flow: np.ndarray
) -> np.ndarray:
    """Accept nominal-only predictions when appended episodes are train-only."""
    try:
        return load_prediction(path, len(episode_flow))
    except ValueError:
        with np.load(path, allow_pickle=False) as archive:
            indices = np.asarray(archive["episode_indices"], dtype=np.int64)
            values = np.asarray(archive["predicted_flow"], dtype=np.float32)
        if len(indices) == 0 or int(indices.min()) < 0 or int(indices.max()) >= len(episode_flow):
            raise
        result = np.asarray(episode_flow, dtype=np.float32).copy()
        result[indices] = values
        return result


def main() -> None:
    args = parser().parse_args()
    if not 0 <= int(args.fold) < len(FOLDS):
        raise ValueError(f"fold must lie in [0,{len(FOLDS) - 1}]")
    with np.load(args.dataset, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    recovery = np.asarray(
        payload.get(
            "is_recovery_sample",
            payload.get("is_recovery_augmented", np.zeros(len(payload["shoe_id"]))),
        )
    )
    indices = np.flatnonzero(
        np.isin(payload["shoe_id"], FOLDS[int(args.fold)]["test"])
        & (recovery < 0.5)
    )
    if args.sample_phase != "all":
        phase = 1.0 if args.sample_phase == "relation" else 0.0
        indices = indices[np.isclose(payload["relation_phase"][indices], phase)]

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    normalization = Normalization(**checkpoint["normalization"])
    model = build_action_predictor(
        str(checkpoint["condition"]),
        int(checkpoint["flow_steps"]),
        int(checkpoint["horizon"]),
        str(checkpoint.get("architecture", "mlp")),
        bool(checkpoint.get("explicit_flow_progress", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    routes = {"oracle": np.asarray(payload["episode_flow"], dtype=np.float32)}
    for name, path in args.prediction or []:
        routes[str(name)] = prediction_with_optional_suffix(
            Path(path), payload["episode_flow"]
        )
    results = {}
    for name, episode_flow in routes.items():
        route_payload = dict(payload)
        route_payload["episode_flow"] = episode_flow
        dataset = TaskFlowDataset(
            route_payload,
            indices,
            normalization,
            flow_context_mode=str(checkpoint.get("flow_context_mode", "full")),
        )
        loader = DataLoader(
            dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=device.type == "cuda",
        )
        sample_indices, actions = predict(model, loader, device, normalization)
        order = np.argsort(sample_indices)
        results[name] = summarize_phases(
            payload, sample_indices[order], actions[order]
        )
        print(json.dumps({"route": name, "summary": results[name]}), flush=True)
    output = {
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset": str(args.dataset.resolve()),
        "fold": int(args.fold),
        "sample_phase": str(args.sample_phase),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "routes": list(results)}, indent=2))


if __name__ == "__main__":
    main()
