#!/usr/bin/env python3
"""Inject held-out predicted flows into frozen flow-conditioned action models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .train_task_flow_benchmark import (
    FOLDS,
    Normalization,
    TaskFlowDataset,
    build_action_predictor,
    predict,
    summarize_errors,
)


def load_prediction(path: Path, episode_count: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        indices = archive["episode_indices"].astype(np.int64)
        values = archive["predicted_flow"].astype(np.float32)
    if len(np.unique(indices)) != episode_count or set(indices.tolist()) != set(range(episode_count)):
        raise ValueError(f"{path} does not contain exactly one prediction per episode")
    result = np.empty_like(values)
    result[indices] = values
    return result


def summarize_phases(
    payload: dict[str, np.ndarray], indices: np.ndarray, prediction: np.ndarray
) -> dict[str, dict[str, float]]:
    result = {"all": summarize_errors(payload, indices, prediction)}
    for name, value in (("pre_relation", 0.0), ("relation", 1.0)):
        keep = np.isclose(payload["relation_phase"][indices], value)
        if np.any(keep):
            result[name] = summarize_errors(payload, indices[keep], prediction[keep])
    recovery_key = (
        "is_recovery_sample"
        if "is_recovery_sample" in payload
        else "is_recovery_augmented"
        if "is_recovery_augmented" in payload
        else None
    )
    if recovery_key is not None:
        for name, value in (("nominal", 0.0), ("recovery", 1.0)):
            keep = np.isclose(payload[recovery_key][indices], value)
            if np.any(keep):
                result[name] = summarize_errors(payload, indices[keep], prediction[keep])
    return result


def pooled_summary(runs: list[dict]) -> dict:
    phases = sorted({phase for run in runs for phase in run["summary"]})
    result = {}
    for phase in phases:
        rows = [run["summary"][phase] for run in runs if phase in run["summary"]]
        count = np.asarray([row["samples"] for row in rows], dtype=np.float64)
        metrics = {"samples_per_seed_total": int(count.sum())}
        for key in rows[0]:
            if key == "samples":
                continue
            metrics[key] = float(
                np.sum(np.asarray([row[key] for row in rows]) * count) / count.sum()
            )
        result[phase] = metrics
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--oracle-dataset", type=Path, required=True)
    result.add_argument("--prediction", action="append", nargs=2, metavar=("NAME", "PATH"))
    result.add_argument("--checkpoint-root", type=Path, required=True)
    result.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    result.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    result.add_argument(
        "--base-condition", choices=("base", "zero_flow"), default="base"
    )
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    result.add_argument("--batch-size", type=int, default=512)
    result.add_argument("--num-workers", type=int, default=0)
    result.add_argument(
        "--sample-phase",
        choices=("all", "pre_relation", "relation"),
        default="all",
    )
    return result


def main() -> None:
    args = parser().parse_args()
    with np.load(args.oracle_dataset, allow_pickle=False) as archive:
        oracle_payload = {key: archive[key] for key in archive.files}
    episode_count = len(oracle_payload["episode_flow"])
    predicted = {
        name: load_prediction(Path(path), episode_count)
        for name, path in (args.prediction or [])
    }
    payloads = {"oracle_flow": oracle_payload}
    for name, flow in predicted.items():
        value = dict(oracle_payload)
        value["episode_flow"] = flow
        payloads[name] = value
    device = torch.device(args.device)
    all_runs: dict[str, list[dict]] = {"base": [], **{key: [] for key in payloads}}
    for seed in args.seeds:
        checkpoint_dir = args.checkpoint_root.parent / f"{args.checkpoint_root.name}_seed{seed}"
        for fold_index in args.folds:
            fold = FOLDS[int(fold_index)]
            test_indices = np.flatnonzero(
                np.isin(oracle_payload["shoe_id"], fold["test"])
            )
            if args.sample_phase != "all":
                phase_value = 1.0 if args.sample_phase == "relation" else 0.0
                test_indices = test_indices[
                    np.isclose(
                        oracle_payload["relation_phase"][test_indices], phase_value
                    )
                ]
            for route, checkpoint_condition, payload_name in (
                ("base", str(args.base_condition), "oracle_flow"),
                ("oracle_flow", "flow", "oracle_flow"),
                *[(name, "flow", name) for name in predicted],
            ):
                checkpoint_path = checkpoint_dir / (
                    f"fold{fold_index}_{checkpoint_condition}_seed{seed}.pt"
                )
                checkpoint = torch.load(
                    checkpoint_path, map_location=device, weights_only=False
                )
                normalization = Normalization(**checkpoint["normalization"])
                payload = payloads[payload_name]
                dataset = TaskFlowDataset(payload, test_indices, normalization)
                loader = DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=device.type == "cuda",
                )
                model = build_action_predictor(
                    checkpoint["condition"],
                    int(checkpoint["flow_steps"]),
                    int(checkpoint["horizon"]),
                    str(checkpoint.get("architecture", "mlp")),
                    bool(checkpoint.get("explicit_flow_progress", False)),
                ).to(device)
                model.load_state_dict(checkpoint["model"])
                indices, action_prediction = predict(
                    model, loader, device, normalization
                )
                order = np.argsort(indices)
                indices = indices[order]
                action_prediction = action_prediction[order]
                run = {
                    "route": route,
                    "seed": int(seed),
                    "fold": int(fold_index),
                    "test_shoes": list(fold["test"]),
                    "summary": summarize_phases(
                        oracle_payload, indices, action_prediction
                    ),
                    "per_shoe": {
                        str(int(shoe)): summarize_errors(
                            oracle_payload,
                            indices[oracle_payload["shoe_id"][indices] == shoe],
                            action_prediction[oracle_payload["shoe_id"][indices] == shoe],
                        )
                        for shoe in sorted(
                            np.unique(oracle_payload["shoe_id"][indices]).tolist()
                        )
                    },
                }
                all_runs[route].append(run)
                print(
                    json.dumps(
                        {
                            "route": route,
                            "seed": seed,
                            "fold": fold_index,
                            "all": run["summary"]["all"],
                        }
                    ),
                    flush=True,
                )
    pooled = {route: pooled_summary(runs) for route, runs in all_runs.items()}
    retention = {}
    lower_better = (
        "active_delta_translation_l2_mm",
        "active_delta_rotation_geodesic_deg",
        "active_endpoint_translation_l2_cm",
        "active_endpoint_rotation_geodesic_deg",
    )
    for route in predicted:
        retention[route] = {}
        for phase in pooled["base"]:
            retention[route][phase] = {}
            for metric in lower_better:
                base = pooled["base"][phase][metric]
                oracle = pooled["oracle_flow"][phase][metric]
                value = pooled[route][phase][metric]
                denominator = base - oracle
                retention[route][phase][metric] = (
                    float((base - value) / denominator)
                    if abs(denominator) > 1e-12
                    else None
                )
    result = {"pooled": pooled, "oracle_gain_retention": retention, "runs": all_runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "pooled": pooled, "retention": retention}, indent=2))


if __name__ == "__main__":
    main()
