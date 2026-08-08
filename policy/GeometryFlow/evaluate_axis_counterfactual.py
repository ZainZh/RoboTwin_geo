#!/usr/bin/env python3
"""Evaluate one frozen policy under counterfactual functional-axis inputs.

Unlike separately trained corruption conditions, this is an intervention on a
single learned policy: model weights, test rows, actions, point clouds and
normalization remain fixed while only ``source_axis3`` changes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .train_interaction_flow_tokens import (
    InteractionFlowDataset,
    build_model,
    policy_phase_keep_mask,
    predict,
    summarize_phases,
)
from .train_task_flow_benchmark import FOLDS, Normalization


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--datasets", nargs="+", type=Path, required=True)
    result.add_argument("--names", nargs="+", required=True)
    result.add_argument("--fold", type=int, default=0)
    result.add_argument("--batch-size", type=int, default=64)
    result.add_argument("--num-workers", type=int, default=2)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--output", type=Path, required=True)
    return result


def normalization_from_checkpoint(payload: dict) -> Normalization:
    values = payload["normalization"]
    expected = {field.name for field in fields(Normalization)}
    missing = expected - set(values)
    if missing:
        raise KeyError(f"checkpoint normalization misses {sorted(missing)}")
    return Normalization(**{key: values[key] for key in expected})


def load_dataset(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    if "source_axis3" not in payload:
        raise KeyError(f"{path} has no source_axis3")
    return payload


def assert_counterfactual_alignment(
    reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray]
) -> None:
    """Reject controls that accidentally change labels, state, or XYZ geometry."""
    for key in (
        "episode_index",
        "frame_index",
        "shoe_id",
        "state",
        "action",
        "relation_phase",
    ):
        if not np.array_equal(reference[key], candidate[key]):
            raise ValueError(f"counterfactual dataset changes protected field {key}")
    for key in ("points_a", "points_b"):
        if not np.array_equal(reference[key][..., :3], candidate[key][..., :3]):
            raise ValueError(f"counterfactual dataset changes protected XYZ field {key}")


def main() -> None:
    args = parser().parse_args()
    if len(args.names) != len(args.datasets):
        raise ValueError("--names and --datasets must have the same length")
    if not 0 <= int(args.fold) < len(FOLDS):
        raise ValueError(f"fold must lie in [0,{len(FOLDS) - 1}]")

    device = torch.device(args.device)
    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=False
    )
    normalization = normalization_from_checkpoint(checkpoint)
    condition = str(checkpoint["condition"])
    model = build_model(
        condition,
        horizon=int(checkpoint["horizon"]),
        flow_steps=int(checkpoint["flow_steps"]),
        num_anchors=int(checkpoint["num_anchors"]),
        feature_dim=int(checkpoint["feature_dim"]),
        layers=int(checkpoint["layers"]),
        flow_parameterization=str(checkpoint["flow_parameterization"]),
        xyz_std=normalization.xyz_std,
        action_std=normalization.action_std,
        predict_rotation=bool(checkpoint["predict_rotation"]),
        point_channels=int(checkpoint["point_channels"]),
        target_color_adapter=bool(checkpoint["target_color_adapter"]),
        source_geometry_adapter=bool(checkpoint["source_geometry_adapter"]),
        shared_geometry_adapter=bool(checkpoint["shared_geometry_adapter"]),
        source_axis_relation_adapter=bool(
            checkpoint["source_axis_relation_adapter"]
        ),
        source_axis_gate_initial_value=float(
            checkpoint.get("source_axis_gate_initial_value", 0.0)
        ),
        target_axis_relation_adapter=bool(
            checkpoint["target_axis_relation_adapter"]
        ),
        diffusion_steps=int(checkpoint["diffusion_steps"]),
        diffusion_inference_steps=int(checkpoint["diffusion_inference_steps"]),
        diffusion_prediction_type=str(checkpoint["diffusion_prediction_type"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.experiment_condition = condition

    payloads = [load_dataset(path) for path in args.datasets]
    for candidate in payloads[1:]:
        assert_counterfactual_alignment(payloads[0], candidate)
    test_ids = FOLDS[int(args.fold)]["test"]
    keep = policy_phase_keep_mask(payloads[0], str(checkpoint["policy_phase"]))
    indices = np.flatnonzero(np.isin(payloads[0]["shoe_id"], test_ids) & keep)

    results = {}
    for name, path, payload in zip(args.names, args.datasets, payloads):
        dataset = InteractionFlowDataset(
            payload,
            indices,
            normalization,
            flow_steps=int(checkpoint["flow_steps"]),
            target_horizon_mode=str(checkpoint["target_horizon_mode"]),
            target_frame_mode="normal",
            functional_frame_translation_mode=str(
                checkpoint["functional_frame_translation_mode"]
            ),
        )
        loader = DataLoader(
            dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=device.type == "cuda",
        )
        prediction_indices, prediction, diagnostic = predict(
            model,
            loader,
            device,
            normalization,
            action_frame=str(checkpoint["action_frame"]),
        )
        order = np.argsort(prediction_indices)
        prediction_indices = prediction_indices[order]
        prediction = prediction[order]
        results[str(name)] = {
            "dataset": str(path),
            "summary": summarize_phases(
                payload, prediction_indices, prediction
            ),
            "geometry_diagnostic": diagnostic,
        }
        print(json.dumps({"name": name, **results[str(name)]["summary"]["all"]}))

    output = {
        "protocol": "single_frozen_checkpoint_axis_intervention_v1",
        "checkpoint": str(args.checkpoint),
        "fold": int(args.fold),
        "test_ids": list(test_ids),
        "samples": int(len(indices)),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output)}))


if __name__ == "__main__":
    main()
