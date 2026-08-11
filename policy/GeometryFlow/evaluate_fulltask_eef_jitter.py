#!/usr/bin/env python3
"""Evaluate first-action robustness under labelled EEF translation jitter."""

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
    augment_open_approach_eef_translation,
    condition_batch,
    device_batch,
)


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: Path,
    payload: dict[str, np.ndarray],
    indices: np.ndarray,
    jitter_m: float,
    device: torch.device,
) -> dict[str, float | str]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    statistics = Statistics(**checkpoint["statistics"])
    loader = DataLoader(
        FullTaskDataset(payload, indices, statistics), batch_size=64, shuffle=False
    )
    model = FullTaskTAGRTPolicy(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    errors = []
    for batch_index, batch in enumerate(loader):
        batch = device_batch(batch, device)
        torch.manual_seed(1_000_003 + batch_index)
        batch = augment_open_approach_eef_translation(batch, statistics, jitter_m)
        conditioned = condition_batch(batch, str(checkpoint["condition"]), "correct")
        prediction = model.sample(conditioned)
        error = prediction.motion[:, 0] - batch["motion"][:, 0]
        open_mask = (batch["current_gripper"] >= 0.5).all(dim=-1)
        errors.append(error[open_mask].cpu())
    error = torch.cat(errors)
    motion_std = torch.tensor(statistics.motion_std)
    metric = error * motion_std
    translation = torch.cat((metric[:, :3], metric[:, 6:9]), dim=-1)
    rotation = torch.cat((metric[:, 3:6], metric[:, 9:12]), dim=-1)
    return {
        "checkpoint": str(checkpoint_path),
        "open_samples": int(len(error)),
        "jitter_m": float(jitter_m),
        "first_action_translation_mae_cm": float(100.0 * translation.abs().mean()),
        "first_action_translation_rmse_cm": float(
            100.0 * translation.square().mean().sqrt()
        ),
        "first_action_rotation_mae_deg": float(
            (180.0 / np.pi) * rotation.abs().mean()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--jitter-m", type=float, default=0.001)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    with np.load(args.dataset, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    indices = np.flatnonzero(payload["episode_id"] == args.episode)
    results = [
        evaluate_checkpoint(
            checkpoint, payload, indices, args.jitter_m, torch.device(args.device)
        )
        for checkpoint in args.checkpoints
    ]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
