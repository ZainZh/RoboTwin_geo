#!/usr/bin/env python3
"""Audit TAGRT-DP3 learned recovery gates without running the simulator."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import dill
import hydra
import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
DP3_ROOT = REPO_ROOT / "policy" / "DP3" / "3D-Diffusion-Policy"
if str(DP3_ROOT) not in sys.path:
    sys.path.insert(0, str(DP3_ROOT))

from diffusion_policy_3d.common.pytorch_util import dict_apply  # noqa: E402
from train_dp3 import TrainDP3Workspace  # noqa: E402


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--checkpoint", type=Path, action="append", required=True)
    result.add_argument("--dataset", type=Path, default=None)
    result.add_argument("--device", default="cpu")
    result.add_argument("--batch-size", type=int, default=256)
    result.add_argument("--thresholds", type=float, nargs="+", default=(0.5, 0.7, 0.8, 0.9))
    result.add_argument("--output", type=Path, default=None)
    return result


def binary_metrics(probability: np.ndarray, target: np.ndarray, threshold: float) -> dict:
    predicted = probability >= float(threshold)
    positive = target >= 0.5
    negative = ~positive
    return {
        "threshold": float(threshold),
        "positive_recall": (
            None if not positive.any() else float(predicted[positive].mean())
        ),
        "negative_false_positive_rate": (
            None if not negative.any() else float(predicted[negative].mean())
        ),
        "predicted_positive_fraction": float(predicted.mean()),
    }


@torch.no_grad()
def evaluate(policy, dataset, *, device: torch.device, batch_size: int) -> dict:
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=2)
    probabilities = []
    gate_targets = []
    failure_targets = []
    translation_mse = []
    translation_endpoint_error = []
    translation_endpoint_cosine = []
    for batch in loader:
        observations = dict_apply(
            batch["obs"], lambda value: value.to(device, non_blocking=True)
        )
        normalized = policy._prepare_nobs(policy.normalizer.normalize(observations))
        count = int(next(iter(normalized.values())).shape[0])
        current = dict_apply(
            normalized,
            lambda value: value[:, : policy.n_obs_steps].reshape(
                -1, *value.shape[2:]
            ),
        )
        encoded = policy._encode_observations(current)
        logits = policy._rotation_action_gate_logits(encoded, normalized, count)
        if logits is None:
            raise RuntimeError("checkpoint has no learned recovery gate")
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        translation_prediction = policy._rotation_action_translation_prediction(
            encoded, normalized, count
        )
        if translation_prediction is None:
            raise RuntimeError("checkpoint has no recovery translation head")
        start = int(policy.n_obs_steps) - 1
        target_action = batch["action"][
            :, start : start + int(policy.n_action_steps)
        ].to(device=translation_prediction.device, dtype=translation_prediction.dtype)
        target_translation = torch.stack(
            (target_action[..., :3], target_action[..., 7:10]), dim=2
        )
        active_right = batch["active_arm_right"].to(
            device=translation_prediction.device, dtype=torch.bool
        )
        active_index = active_right[:, None, None, None].expand(
            -1, int(policy.n_action_steps), 1, 3
        )
        active_prediction = torch.gather(
            translation_prediction, dim=2, index=active_index.long()
        ).squeeze(2)
        active_target = torch.gather(
            target_translation, dim=2, index=active_index.long()
        ).squeeze(2)
        translation_mse.append(
            torch.mean((active_prediction - active_target) ** 2, dim=(1, 2))
            .cpu()
            .numpy()
        )
        predicted_endpoint = active_prediction.sum(dim=1)
        target_endpoint = active_target.sum(dim=1)
        endpoint_error = torch.linalg.vector_norm(
            predicted_endpoint - target_endpoint, dim=-1
        )
        endpoint_cosine = torch.sum(
            predicted_endpoint * target_endpoint, dim=-1
        ) / (
            torch.linalg.vector_norm(predicted_endpoint, dim=-1)
            * torch.linalg.vector_norm(target_endpoint, dim=-1)
        ).clamp_min(1.0e-9)
        translation_endpoint_error.append(endpoint_error.cpu().numpy())
        translation_endpoint_cosine.append(endpoint_cosine.cpu().numpy())
        gate_targets.append(batch["is_recovery_gate_positive"].numpy())
        failure_targets.append(batch["is_failure_matched_recovery"].numpy())
    probability = np.concatenate(probabilities).astype(np.float64)
    gate_target = np.concatenate(gate_targets).astype(np.float64)
    failure_target = np.concatenate(failure_targets).astype(np.float64)
    translation_error = np.concatenate(translation_mse).astype(np.float64)
    endpoint_error = np.concatenate(translation_endpoint_error).astype(np.float64)
    endpoint_cosine = np.concatenate(translation_endpoint_cosine).astype(np.float64)
    translation_target = (gate_target >= 0.5) & (failure_target < 0.5)
    return {
        "samples": int(len(probability)),
        "gate_positives": int(np.sum(gate_target >= 0.5)),
        "failure_matched_rotation_positives": int(np.sum(failure_target >= 0.5)),
        "generic_translation_positives": int(np.sum(translation_target)),
        "probability": probability,
        "gate_target": gate_target,
        "failure_target": failure_target,
        "translation_target": translation_target,
        "translation_mse": translation_error,
        "translation_endpoint_error": endpoint_error,
        "translation_endpoint_cosine": endpoint_cosine,
    }


def main() -> None:
    args = parser().parse_args()
    device = torch.device(args.device)
    report = {"thresholds": list(map(float, args.thresholds)), "checkpoints": []}
    for checkpoint_path in args.checkpoint:
        checkpoint_path = checkpoint_path.expanduser().resolve()
        payload = torch.load(
            checkpoint_path.open("rb"), pickle_module=dill, map_location="cpu"
        )
        cfg = copy.deepcopy(payload["cfg"])
        if args.dataset is not None:
            cfg.task.dataset.npz_path = str(args.dataset.expanduser().resolve())
        workspace = TrainDP3Workspace(cfg, output_dir=str(checkpoint_path.parent))
        workspace.load_payload(payload, exclude_keys=("optimizer",))
        policy = workspace.ema_model if workspace.ema_model is not None else workspace.model
        policy.to(device).eval()
        train = hydra.utils.instantiate(cfg.task.dataset)
        splits = {
            "train": train,
            "validation": train.get_validation_dataset(),
            "test": train.get_test_dataset(),
        }
        checkpoint_report = {"checkpoint": str(checkpoint_path), "splits": {}}
        for name, dataset in splits.items():
            values = evaluate(
                policy, dataset, device=device, batch_size=int(args.batch_size)
            )
            probability = values.pop("probability")
            gate_target = values.pop("gate_target")
            failure_target = values.pop("failure_target")
            translation_target = values.pop("translation_target")
            translation_mse = values.pop("translation_mse")
            endpoint_error = values.pop("translation_endpoint_error")
            endpoint_cosine = values.pop("translation_endpoint_cosine")
            subset_probability = {
                "negative": probability[gate_target < 0.5],
                "failure_matched_rotation": probability[failure_target >= 0.5],
                "generic_translation": probability[translation_target],
            }
            values["probability_mean_by_subset"] = {
                key: None if not len(value) else float(value.mean())
                for key, value in subset_probability.items()
            }
            values["translation_mse_by_subset"] = {
                "failure_matched_rotation": (
                    None
                    if not np.any(failure_target >= 0.5)
                    else float(translation_mse[failure_target >= 0.5].mean())
                ),
                "generic_translation": (
                    None
                    if not np.any(translation_target)
                    else float(translation_mse[translation_target].mean())
                ),
            }
            values["translation_endpoint_by_subset"] = {}
            for key, mask in {
                "failure_matched_rotation": failure_target >= 0.5,
                "generic_translation": translation_target,
            }.items():
                if not np.any(mask):
                    values["translation_endpoint_by_subset"][key] = None
                    continue
                values["translation_endpoint_by_subset"][key] = {
                    "error_cm_mean": float(100.0 * endpoint_error[mask].mean()),
                    "error_cm_p95": float(
                        100.0 * np.quantile(endpoint_error[mask], 0.95)
                    ),
                    "cosine_mean": float(endpoint_cosine[mask].mean()),
                    "cosine_p05": float(
                        np.quantile(endpoint_cosine[mask], 0.05)
                    ),
                }
            values["operating_points"] = [
                binary_metrics(probability, gate_target, threshold)
                for threshold in args.thresholds
            ]
            checkpoint_report["splits"][name] = values
        report["checkpoints"].append(checkpoint_report)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
