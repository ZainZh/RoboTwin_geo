#!/usr/bin/env python3
"""Evaluate one learned policy under correct/zero/shuffled geometry inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .functional_action_frame import action14_from_target_frame, action14_to_target_frame
from .train_interaction_flow_tokens import InteractionFlowDataset, build_model, predict
from .train_task_flow_benchmark import Normalization, summarize_errors


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", type=Path, required=True)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--test-ids", nargs="+", type=int, required=True)
    result.add_argument("--modes", nargs="+", default=["normal", "zero", "shuffled"])
    result.add_argument("--batch-size", type=int, default=64)
    result.add_argument("--device", default="cuda:0")
    return result


def evaluate(args: argparse.Namespace) -> dict:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    with np.load(args.dataset, allow_pickle=False) as archive:
        world_payload = {key: np.asarray(archive[key]) for key in archive.files}
    payload = dict(world_payload)
    if checkpoint["policy_frame"] == "goal_action":
        payload["action"] = action14_to_target_frame(
            payload["action"], payload["goal_frame9"]
        )
    elif checkpoint["policy_frame"] != "world":
        raise ValueError(
            "intervention evaluator currently supports world and goal_action policies"
        )
    normalization = Normalization(
        **{
            key: None if value is None else np.asarray(value)
            for key, value in checkpoint["normalization"].items()
        }
    )
    model = build_model(
        checkpoint["condition"],
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
        functional_coordinate_adapter=bool(checkpoint["functional_coordinate_adapter"]),
        source_axis_relation_adapter=bool(checkpoint["source_axis_relation_adapter"]),
        source_axis_gate_initial_value=float(checkpoint["source_axis_gate_initial_value"]),
        target_axis_relation_adapter=bool(checkpoint["target_axis_relation_adapter"]),
        hand_object_frame_token=bool(checkpoint["hand_object_frame_token"]),
        diffusion_steps=int(checkpoint["diffusion_steps"]),
        diffusion_inference_steps=int(checkpoint["diffusion_inference_steps"]),
        diffusion_prediction_type=str(checkpoint["diffusion_prediction_type"]),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.experiment_condition = str(checkpoint["condition"])
    device = torch.device(args.device)
    model.to(device)
    indices = np.flatnonzero(
        np.isin(payload["shoe_id"], np.asarray(args.test_ids, dtype=np.int64))
    )
    result = {
        "dataset": str(args.dataset.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "test_ids": [int(value) for value in args.test_ids],
        "modes": {},
    }
    for mode in args.modes:
        dataset = InteractionFlowDataset(
            payload,
            indices,
            normalization,
            flow_steps=int(checkpoint["flow_steps"]),
            target_horizon_mode=str(checkpoint["target_horizon_mode"]),
            target_frame_mode=str(mode),
            functional_frame_translation_mode=str(
                checkpoint["functional_frame_translation_mode"]
            ),
            functional_point_coordinates=bool(
                checkpoint["functional_point_coordinates"]
            ),
            zero_global_functional_frame_token=bool(
                checkpoint["zero_global_functional_frame_token"]
            ),
        )
        loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False)
        predicted_indices, prediction, diagnostic = predict(
            model, loader, device, normalization, action_frame=str(checkpoint["action_frame"])
        )
        order = np.argsort(predicted_indices)
        predicted_indices = predicted_indices[order]
        prediction = prediction[order]
        if checkpoint["policy_frame"] == "goal_action":
            prediction = action14_from_target_frame(
                prediction, world_payload["goal_frame9"][predicted_indices]
            )
        result["modes"][str(mode)] = {
            "summary": summarize_errors(
                world_payload, predicted_indices, prediction
            ),
            "geometry_diagnostic": diagnostic,
        }
    return result


def main() -> None:
    args = parser().parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
