#!/usr/bin/env python3
"""Train paired raw/TAGRT full-task policies with regression or flow decoding."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .fulltask_tagrt_v2 import FullTaskTAGRTPolicy, split_policy_action


FOLD0 = {"train": (2, 3, 4, 7, 8, 9), "validation": (1, 6), "test": (0, 5)}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument(
        "--conditions",
        nargs="+",
        choices=("raw_reg", "tagrt_reg", "raw_flow", "tagrt_flow"),
        default=("raw_reg", "tagrt_reg", "raw_flow", "tagrt_flow"),
    )
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--epochs", type=int, default=25)
    result.add_argument("--patience", type=int, default=7)
    result.add_argument("--batch-size", type=int, default=32)
    result.add_argument("--learning-rate", type=float, default=3e-4)
    result.add_argument("--weight-decay", type=float, default=1e-4)
    result.add_argument("--feature-dim", type=int, default=192)
    result.add_argument("--heads", type=int, default=6)
    result.add_argument("--memory-layers", type=int, default=4)
    result.add_argument("--decoder-layers", type=int, default=4)
    result.add_argument("--dropout", type=float, default=0.05)
    result.add_argument("--flow-steps", type=int, default=12)
    result.add_argument("--gripper-weight", type=float, default=0.2)
    result.add_argument("--gripper-transition-weight", type=float, default=1.0)
    result.add_argument("--initialize-from", type=Path, default=None)
    result.add_argument("--train-gripper-head-only", action="store_true")
    result.add_argument(
        "--gripper-selection-delay",
        type=float,
        default=0.0,
        help=(
            "For gripper-head-only calibration, select the checkpoint by the "
            "mean clean/delayed transition MAE at this deterministic delay."
        ),
    )
    result.add_argument(
        "--gripper-state-delay-jitter",
        type=float,
        default=0.0,
        help=(
            "Training-only normalized gripper-state delay.  The observed "
            "gripper is shifted opposite to the supervised future transition "
            "while the action label remains unchanged."
        ),
    )
    result.add_argument(
        "--eef-translation-jitter-m",
        type=float,
        default=0.0,
        help=(
            "Training-only Cartesian EEF perturbation for open-gripper approach "
            "states. The first action is relabelled to return to the demonstrated "
            "next waypoint; deployment remains fully learned."
        ),
    )
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--num-workers", type=int, default=0)
    result.add_argument(
        "--episode-split",
        action="store_true",
        help=(
            "Use a deterministic trajectory-level split for a train-object-only "
            "replication batch instead of the strict object-generalization split."
        ),
    )
    result.add_argument("--episode-split-seed", type=int, default=20260811)
    result.add_argument("--validation-episodes", type=int, default=3)
    result.add_argument("--test-episodes", type=int, default=3)
    result.add_argument(
        "--overfit-episodes",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Use the listed demonstration episode ids for train/validation/test. "
            "This is an online-interface admission test, not a generalization result."
        ),
    )
    return result


def split_by_episode(
    episode_id: np.ndarray,
    *,
    seed: int,
    validation_episodes: int,
    test_episodes: int,
) -> tuple[dict[str, np.ndarray], dict[str, list[int]]]:
    """Make leakage-free sample splits from complete demonstration episodes."""

    episodes = np.unique(np.asarray(episode_id, dtype=np.int64))
    validation_count = int(validation_episodes)
    test_count = int(test_episodes)
    if validation_count < 1 or test_count < 1:
        raise ValueError("episode validation/test counts must be positive")
    if len(episodes) <= validation_count + test_count:
        raise ValueError("episode split leaves no training trajectory")
    generator = np.random.default_rng(int(seed))
    shuffled = generator.permutation(episodes)
    selected = {
        "test": np.sort(shuffled[:test_count]),
        "validation": np.sort(shuffled[test_count : test_count + validation_count]),
        "train": np.sort(shuffled[test_count + validation_count :]),
    }
    split = {
        name: np.flatnonzero(np.isin(episode_id, values))
        for name, values in selected.items()
    }
    definition = {
        name: [int(value) for value in values]
        for name, values in selected.items()
    }
    return split, definition


@dataclass
class Statistics:
    point_mean: list[float]
    point_std: list[float]
    local_mean: list[float]
    local_std: list[float]
    global_mean: list[float]
    global_std: list[float]
    state_mean: list[float]
    state_std: list[float]
    motion_mean: list[float]
    motion_std: list[float]


def safe_stats(value: np.ndarray, axes: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(value, dtype=np.float64).mean(axis=axes)
    std = np.asarray(value, dtype=np.float64).std(axis=axes)
    std = np.maximum(std, 1e-4)
    return mean.astype(np.float32), std.astype(np.float32)


def make_statistics(payload: dict[str, np.ndarray], train: np.ndarray) -> Statistics:
    point_values = np.concatenate(
        (
            payload["scene"][train].reshape(-1, 6),
            payload["points_a"][train].reshape(-1, 6),
            payload["points_b"][train].reshape(-1, 6),
        ),
        axis=0,
    )
    point_mean, point_std = safe_stats(point_values, (0,))
    local_mean, local_std = safe_stats(payload["local"][train], (0, 1, 2))
    global_mean, global_std = safe_stats(payload["global"][train], (0, 1))
    state_mean, state_std = safe_stats(payload["state"][train], (0, 1))
    action = torch.from_numpy(np.asarray(payload["action"][train], dtype=np.float32))
    motion = split_policy_action(action)[0].numpy()
    motion_mean, motion_std = safe_stats(motion, (0, 1))
    return Statistics(
        point_mean=point_mean.tolist(),
        point_std=point_std.tolist(),
        local_mean=local_mean.tolist(),
        local_std=local_std.tolist(),
        global_mean=global_mean.tolist(),
        global_std=global_std.tolist(),
        state_mean=state_mean.tolist(),
        state_std=state_std.tolist(),
        motion_mean=motion_mean.tolist(),
        motion_std=motion_std.tolist(),
    )


class FullTaskDataset(Dataset):
    def __init__(
        self, payload: dict[str, np.ndarray], indices: np.ndarray, statistics: Statistics
    ) -> None:
        self.payload = payload
        self.indices = np.asarray(indices, dtype=np.int64)
        self.point_mean = np.asarray(statistics.point_mean, dtype=np.float32)
        self.point_std = np.asarray(statistics.point_std, dtype=np.float32)
        self.local_mean = np.asarray(statistics.local_mean, dtype=np.float32)
        self.local_std = np.asarray(statistics.local_std, dtype=np.float32)
        self.global_mean = np.asarray(statistics.global_mean, dtype=np.float32)
        self.global_std = np.asarray(statistics.global_std, dtype=np.float32)
        self.state_mean = np.asarray(statistics.state_mean, dtype=np.float32)
        self.state_std = np.asarray(statistics.state_std, dtype=np.float32)
        self.motion_mean = np.asarray(statistics.motion_mean, dtype=np.float32)
        self.motion_std = np.asarray(statistics.motion_std, dtype=np.float32)
        # Build a deterministic wrong-goal control that is guaranteed to come
        # from another episode.  Rolling an ordered validation batch can leave
        # most samples paired with an adjacent frame of the same trajectory,
        # which is not a meaningful geometry counterfactual.
        episodes = np.asarray(payload["episode_id"][self.indices], dtype=np.int64)
        positions = np.arange(len(self.indices), dtype=np.int64)
        mismatch = (positions + max(len(positions) // 2, 1)) % len(positions)
        if len(np.unique(episodes)) >= 2:
            for position in positions:
                attempts = 0
                while episodes[mismatch[position]] == episodes[position]:
                    mismatch[position] = (mismatch[position] + 1) % len(positions)
                    attempts += 1
                    if attempts >= len(positions):
                        raise ValueError(
                            "cross-episode shuffled control needs at least two episodes"
                        )
        else:
            # A one-trajectory overfit run is useful only as a deployment-interface
            # admission test.  It has no meaningful shuffled-geometry control.
            mismatch = positions.copy()
        self.mismatch_indices = self.indices[mismatch]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = int(self.indices[item])
        mismatch = int(self.mismatch_indices[item])
        action = torch.from_numpy(np.asarray(self.payload["action"][index], dtype=np.float32))
        motion, gripper = split_policy_action(action)
        state_raw = np.asarray(self.payload["state"][index], dtype=np.float32)
        return {
            "scene": torch.from_numpy(
                (self.payload["scene"][index] - self.point_mean) / self.point_std
            ).float(),
            "points_a": torch.from_numpy(
                (self.payload["points_a"][index] - self.point_mean) / self.point_std
            ).float(),
            "points_b": torch.from_numpy(
                (self.payload["points_b"][index] - self.point_mean) / self.point_std
            ).float(),
            "local": torch.from_numpy(
                (self.payload["local"][index] - self.local_mean) / self.local_std
            ).float(),
            "global": torch.from_numpy(
                (self.payload["global"][index] - self.global_mean) / self.global_std
            ).float(),
            "local_mismatched": torch.from_numpy(
                (self.payload["local"][mismatch] - self.local_mean) / self.local_std
            ).float(),
            "global_mismatched": torch.from_numpy(
                (self.payload["global"][mismatch] - self.global_mean) / self.global_std
            ).float(),
            "state": torch.from_numpy(
                (state_raw - self.state_mean) / self.state_std
            ).float(),
            "motion": (motion - torch.from_numpy(self.motion_mean))
            / torch.from_numpy(self.motion_std),
            "gripper": gripper.float().clamp(0.0, 1.0),
            "current_gripper": torch.from_numpy(
                state_raw[-1, [9, 19]].copy()
            ).float(),
            "sample_index": torch.tensor(index, dtype=torch.long),
        }


def device_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def condition_batch(batch: dict[str, torch.Tensor], condition: str, intervention: str) -> dict[str, torch.Tensor]:
    result = dict(batch)
    uses_geometry = condition.startswith("tagrt")
    if not uses_geometry or intervention == "zero":
        result["local"] = torch.zeros_like(batch["local"])
        result["global"] = torch.zeros_like(batch["global"])
    elif intervention == "shuffled":
        result["local"] = batch["local_mismatched"]
        result["global"] = batch["global_mismatched"]
    return result


def training_prediction(
    model: FullTaskTAGRTPolicy, batch: dict[str, torch.Tensor], condition: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = condition_batch(batch, condition, "correct")
    clean = batch["motion"]
    if condition.endswith("flow"):
        noise = torch.randn_like(clean)
        time_value = torch.rand(len(clean), device=clean.device)
        noisy = (1.0 - time_value[:, None, None]) * noise + time_value[:, None, None] * clean
        target = clean - noise
    else:
        time_value = torch.ones(len(clean), device=clean.device)
        noisy = torch.zeros_like(clean)
        target = clean
    output = model(batch, noisy_motion=noisy, time=time_value)
    return output.motion, output.gripper_logits, target


def augment_open_approach_eef_translation(
    batch: dict[str, torch.Tensor],
    statistics: Statistics,
    maximum_m: float,
) -> dict[str, torch.Tensor]:
    """Add a physically labelled EEF perturbation before the object is grasped.

    Object point clouds remain unchanged because only the robot EEF is moved.
    The demonstrated next absolute waypoint is kept fixed, so subtracting the
    perturbation from the first delta action gives exact corrective supervision.
    """
    if float(maximum_m) <= 0.0:
        return batch
    if batch["motion"].shape[-1] != 12:
        raise ValueError(
            "EEF translation jitter is only defined for Cartesian delta actions"
        )
    result = dict(batch)
    result["state"] = batch["state"].clone()
    result["local"] = batch["local"].clone()
    result["motion"] = batch["motion"].clone()
    open_mask = (batch["current_gripper"] >= 0.5).all(dim=-1).float()
    jitter = (
        2.0 * torch.rand(len(open_mask), 2, 3, device=open_mask.device) - 1.0
    ) * float(maximum_m)
    jitter = jitter * open_mask[:, None, None]
    state_std = torch.as_tensor(
        statistics.state_std, device=jitter.device, dtype=jitter.dtype
    )
    local_std = torch.as_tensor(
        statistics.local_std, device=jitter.device, dtype=jitter.dtype
    )
    motion_std = torch.as_tensor(
        statistics.motion_std, device=jitter.device, dtype=jitter.dtype
    )
    result["state"][:, -1, :3] += jitter[:, 0] / state_std[:3]
    result["state"][:, -1, 10:13] += jitter[:, 1] / state_std[10:13]
    # Local token layout: object offset, anchor-left EEF, anchor-right EEF, flow.
    result["local"][:, -1, :, 3:6] -= jitter[:, 0, None] / local_std[3:6]
    result["local"][:, -1, :, 6:9] -= jitter[:, 1, None] / local_std[6:9]
    result["motion"][:, 0, :3] -= jitter[:, 0] / motion_std[:3]
    result["motion"][:, 0, 6:9] -= jitter[:, 1] / motion_std[6:9]
    return result


def augment_gripper_state_delay(
    batch: dict[str, torch.Tensor],
    statistics: Statistics,
    maximum_shift: float,
) -> dict[str, torch.Tensor]:
    """Teach recovery when physical gripper feedback lags its command.

    For a future closing transition, the observed state is made more open; for
    an opening transition it is made more closed.  Geometry, EEF state, and the
    supervised action remain untouched, so this is a continuous observation
    perturbation rather than a stage label or action gate.
    """

    maximum_shift = float(maximum_shift)
    if maximum_shift <= 0.0:
        return batch
    if maximum_shift > 1.0:
        raise ValueError("gripper-state delay jitter must lie in [0,1]")
    result = dict(batch)
    state = batch["state"].clone()
    state_mean = torch.as_tensor(
        statistics.state_mean, device=state.device, dtype=state.dtype
    )
    state_std = torch.as_tensor(
        statistics.state_std, device=state.device, dtype=state.dtype
    )
    raw_state = state * state_std + state_mean
    current = batch["current_gripper"].to(state.dtype)
    future = batch["gripper"][:, -1].to(state.dtype)
    # Positive direction means the measured state lags a closing command and
    # remains too open.  Negative direction is the symmetric release case.
    direction = torch.sign(current - future)
    active = (current - future).abs() > 1e-3
    magnitude = torch.rand_like(current) * maximum_shift
    shifted = (current + direction * magnitude * active).clamp(0.0, 1.0)
    for arm, channel in enumerate((9, 19)):
        delta = shifted[:, arm] - current[:, arm]
        raw_state[:, :, channel] = (
            raw_state[:, :, channel] + delta[:, None]
        ).clamp(0.0, 1.0)
    result["state"] = (raw_state - state_mean) / state_std
    result["current_gripper"] = shifted
    return result


def apply_deterministic_gripper_state_delay(
    batch: dict[str, torch.Tensor],
    statistics: Statistics,
    shift: float,
) -> dict[str, torch.Tensor]:
    """Apply a fixed counterfactual feedback delay for validation."""

    shift = float(shift)
    if shift <= 0.0:
        return batch
    if shift > 1.0:
        raise ValueError("gripper-state delay must lie in [0,1]")
    result = dict(batch)
    state_mean = torch.as_tensor(
        statistics.state_mean, device=batch["state"].device, dtype=batch["state"].dtype
    )
    state_std = torch.as_tensor(
        statistics.state_std, device=batch["state"].device, dtype=batch["state"].dtype
    )
    raw_state = batch["state"] * state_std + state_mean
    current = batch["current_gripper"].to(raw_state.dtype)
    future = batch["gripper"][:, -1].to(raw_state.dtype)
    direction = torch.sign(current - future)
    active = (current - future).abs() > 1e-3
    shifted = (current + direction * shift * active).clamp(0.0, 1.0)
    for arm, channel in enumerate((9, 19)):
        delta = shifted[:, arm] - current[:, arm]
        raw_state[:, :, channel] = (
            raw_state[:, :, channel] + delta[:, None]
        ).clamp(0.0, 1.0)
    result["state"] = (raw_state - state_mean) / state_std
    result["current_gripper"] = shifted
    return result


def metric_totals() -> dict[str, float]:
    return {
        "count": 0.0,
        "motion_sq": 0.0,
        "motion_abs": 0.0,
        "position_abs": 0.0,
        "position_count": 0.0,
        "velocity_abs": 0.0,
        "velocity_count": 0.0,
        "translation_abs": 0.0,
        "rotation_abs": 0.0,
        "gripper_correct": 0.0,
        "gripper_count": 0.0,
        "gripper_abs": 0.0,
        "gripper_transition_abs": 0.0,
        "gripper_transition_count": 0.0,
    }


@torch.no_grad()
def evaluate(
    model: FullTaskTAGRTPolicy,
    loader: DataLoader,
    condition: str,
    intervention: str,
    device: torch.device,
    statistics: Statistics,
    flow_steps: int,
    gripper_state_delay: float = 0.0,
) -> dict[str, float]:
    model.eval()
    totals = metric_totals()
    motion_std = torch.tensor(statistics.motion_std, device=device)
    for batch in loader:
        batch = device_batch(batch, device)
        batch = apply_deterministic_gripper_state_delay(
            batch, statistics, gripper_state_delay
        )
        conditioned = condition_batch(batch, condition, intervention)
        generator = torch.Generator(device=device)
        generator.manual_seed(91_337 + int(batch["sample_index"][0]))
        initial = torch.randn(
            batch["motion"].shape,
            generator=generator,
            device=device,
            dtype=batch["motion"].dtype,
        )
        prediction = model.sample(
            conditioned,
            integration_steps=flow_steps,
            initial_noise=initial if condition.endswith("flow") else None,
        )
        error = prediction.motion - batch["motion"]
        metric_error = error * motion_std
        totals["count"] += float(error.numel())
        totals["motion_sq"] += float(error.square().sum())
        totals["motion_abs"] += float(metric_error.abs().sum())
        if metric_error.shape[-1] == 12:
            translation = torch.cat(
                (metric_error[..., :3], metric_error[..., 6:9]), dim=-1
            )
            rotation = torch.cat(
                (metric_error[..., 3:6], metric_error[..., 9:12]), dim=-1
            )
            totals["translation_abs"] += float(translation.abs().sum())
            totals["rotation_abs"] += float(rotation.abs().sum())
        elif metric_error.shape[-1] == 24:
            position = torch.cat(
                (metric_error[..., :6], metric_error[..., 12:18]), dim=-1
            )
            velocity = torch.cat(
                (metric_error[..., 6:12], metric_error[..., 18:24]), dim=-1
            )
            totals["position_abs"] += float(position.abs().sum())
            totals["position_count"] += float(position.numel())
            totals["velocity_abs"] += float(velocity.abs().sum())
            totals["velocity_count"] += float(velocity.numel())
        else:
            raise ValueError(f"unsupported motion width {metric_error.shape[-1]}")
        gripper_probability = prediction.gripper_logits.sigmoid()
        predicted_gripper = gripper_probability >= 0.5
        target_gripper = batch["gripper"] >= 0.5
        totals["gripper_correct"] += float((predicted_gripper == target_gripper).sum())
        totals["gripper_count"] += float(target_gripper.numel())
        gripper_abs = (gripper_probability - batch["gripper"]).abs()
        totals["gripper_abs"] += float(gripper_abs.sum())
        transition = (
            batch["gripper"] - batch["current_gripper"][:, None]
        ).abs() > 1e-3
        totals["gripper_transition_abs"] += float(gripper_abs[transition].sum())
        totals["gripper_transition_count"] += float(transition.sum())
    result = {
        "normalized_motion_mse": totals["motion_sq"] / max(totals["count"], 1.0),
        "motion_mae": totals["motion_abs"] / max(totals["count"], 1.0),
        "gripper_accuracy": totals["gripper_correct"] / max(totals["gripper_count"], 1.0),
        "gripper_mae": totals["gripper_abs"] / max(totals["gripper_count"], 1.0),
        "gripper_transition_mae": totals["gripper_transition_abs"]
        / max(totals["gripper_transition_count"], 1.0),
    }
    if totals["position_count"]:
        result.update(
            {
                "joint_position_mae_rad": totals["position_abs"]
                / totals["position_count"],
                "joint_velocity_mae_radps": totals["velocity_abs"]
                / totals["velocity_count"],
            }
        )
    else:
        result.update(
            {
                "translation_mae_cm": 100.0
                * totals["translation_abs"]
                / max(totals["count"] / 2.0, 1.0),
                "rotation_mae_deg": (180.0 / np.pi)
                * totals["rotation_abs"]
                / max(totals["count"] / 2.0, 1.0),
            }
        )
    return result


def train_one(
    args,
    condition: str,
    payload: dict[str, np.ndarray],
    statistics: Statistics,
    split: dict[str, np.ndarray],
) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    datasets = {
        name: FullTaskDataset(payload, indices, statistics)
        for name, indices in split.items()
    }
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(name == "train"),
            num_workers=args.num_workers,
            pin_memory=True,
        )
        for name, dataset in datasets.items()
    }
    model = FullTaskTAGRTPolicy(
        history=payload["scene"].shape[1],
        horizon=payload["action"].shape[1],
        feature_dim=args.feature_dim,
        heads=args.heads,
        memory_layers=args.memory_layers,
        decoder_layers=args.decoder_layers,
        dropout=args.dropout,
        decoder_type="flow" if condition.endswith("flow") else "regression",
        motion_dim=len(statistics.motion_mean),
    ).to(device)
    if args.initialize_from is not None:
        initialized = torch.load(args.initialize_from, map_location="cpu", weights_only=False)
        if initialized["model_config"] != {
            "history": int(payload["scene"].shape[1]),
            "horizon": int(payload["action"].shape[1]),
            "feature_dim": args.feature_dim,
            "heads": args.heads,
            "memory_layers": args.memory_layers,
            "decoder_layers": args.decoder_layers,
            "dropout": args.dropout,
            "decoder_type": "flow" if condition.endswith("flow") else "regression",
            "motion_dim": len(statistics.motion_mean),
        }:
            raise ValueError("initial checkpoint model configuration does not match")
        model.load_state_dict(initialized["model"])
    if args.train_gripper_head_only:
        if args.initialize_from is None:
            raise ValueError("--train-gripper-head-only requires --initialize-from")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.decoder.gripper_head.parameters():
            parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    bce = nn.BCEWithLogitsLoss(reduction="none")
    best_loss = float("inf")
    best_state = None
    patience = 0
    history = []
    start = time.time()
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        batches = 0
        for batch in loaders["train"]:
            batch = device_batch(batch, device)
            batch = augment_gripper_state_delay(
                batch, statistics, args.gripper_state_delay_jitter
            )
            batch = augment_open_approach_eef_translation(
                batch, statistics, args.eef_translation_jitter_m
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                motion, gripper, target = training_prediction(model, batch, condition)
                motion_loss = nn.functional.mse_loss(motion, target)
                gripper_element_loss = bce(gripper, batch["gripper"])
                transition = (
                    batch["gripper"] - batch["current_gripper"][:, None]
                ).abs() > 1e-3
                gripper_weights = 1.0 + (
                    float(args.gripper_transition_weight) - 1.0
                ) * transition.to(gripper_element_loss.dtype)
                gripper_loss = (
                    gripper_element_loss * gripper_weights
                ).sum() / gripper_weights.sum().clamp_min(1.0)
                loss = motion_loss + args.gripper_weight * gripper_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach())
            batches += 1
        validation = evaluate(
            model,
            loaders["validation"],
            condition,
            "correct",
            device,
            statistics,
            args.flow_steps,
        )
        delayed_validation = None
        if args.train_gripper_head_only and args.gripper_selection_delay > 0.0:
            delayed_validation = evaluate(
                model,
                loaders["validation"],
                condition,
                "correct",
                device,
                statistics,
                args.flow_steps,
                gripper_state_delay=args.gripper_selection_delay,
            )
        record = {
            "epoch": epoch + 1,
            "train_loss": loss_sum / max(batches, 1),
            **{f"validation_{key}": value for key, value in validation.items()},
        }
        history.append(record)
        if delayed_validation is not None:
            record["validation_delayed_gripper_transition_mae"] = delayed_validation[
                "gripper_transition_mae"
            ]
        print(json.dumps({"condition": condition, **record}), flush=True)
        if delayed_validation is None:
            value = validation["normalized_motion_mse"]
        else:
            value = 0.5 * (
                validation["gripper_transition_mae"]
                + delayed_validation["gripper_transition_mae"]
            )
        if value < best_loss:
            best_loss = value
            best_state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    test = {
        intervention: evaluate(
            model,
            loaders["test"],
            condition,
            intervention,
            device,
            statistics,
            args.flow_steps,
        )
        for intervention in (
            ("correct", "zero", "shuffled") if condition.startswith("tagrt") else ("correct",)
        )
    }
    result = {
        "condition": condition,
        "seed": args.seed,
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "epochs_run": len(history),
        "elapsed_minutes": (time.time() - start) / 60.0,
        "best_validation_normalized_motion_mse": best_loss,
        "test": test,
        "history": history,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": best_state,
            "condition": condition,
            "statistics": asdict(statistics),
            "model_config": {
                "history": int(payload["scene"].shape[1]),
                "horizon": int(payload["action"].shape[1]),
                "feature_dim": args.feature_dim,
                "heads": args.heads,
                "memory_layers": args.memory_layers,
                "decoder_layers": args.decoder_layers,
                "dropout": args.dropout,
                "decoder_type": "flow" if condition.endswith("flow") else "regression",
                "motion_dim": len(statistics.motion_mean),
            },
            "action_representation": (
                "dense_joint26" if payload["action"].shape[-1] == 26 else "eef_delta14"
            ),
            "training_augmentation": {
                "open_approach_eef_translation_jitter_m": float(
                    args.eef_translation_jitter_m
                ),
                "gripper_state_delay_jitter": float(
                    args.gripper_state_delay_jitter
                ),
                "gripper_transition_weight": float(args.gripper_transition_weight),
                "gripper_selection_delay": float(args.gripper_selection_delay),
            },
            "initialized_from": (
                str(args.initialize_from.resolve())
                if args.initialize_from is not None
                else None
            ),
            "trained_parameters": [
                name for name, parameter in model.named_parameters() if parameter.requires_grad
            ],
            "result": result,
        },
        args.output_dir / f"{condition}_seed{args.seed}.pt",
    )
    (args.output_dir / f"{condition}_seed{args.seed}.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    args = parser().parse_args()
    with np.load(args.dataset, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    split_definition = None
    if args.overfit_episodes:
        overfit = np.flatnonzero(
            np.isin(payload["episode_id"], np.asarray(args.overfit_episodes))
        )
        split = {name: overfit.copy() for name in ("train", "validation", "test")}
        split_definition = {
            name: [int(value) for value in args.overfit_episodes]
            for name in split
        }
    elif args.episode_split:
        split, split_definition = split_by_episode(
            payload["episode_id"],
            seed=args.episode_split_seed,
            validation_episodes=args.validation_episodes,
            test_episodes=args.test_episodes,
        )
    else:
        split = {
            name: np.flatnonzero(np.isin(payload["shoe_id"], np.asarray(ids)))
            for name, ids in FOLD0.items()
        }
    if any(len(indices) == 0 for indices in split.values()):
        raise ValueError(f"empty object split: { {key: len(value) for key, value in split.items()} }")
    statistics = make_statistics(payload, split["train"])
    print(
        json.dumps(
            {
                "dataset": str(args.dataset),
                "splits": {key: len(value) for key, value in split.items()},
                "conditions": args.conditions,
                "device": args.device,
            }
        ),
        flush=True,
    )
    results = [
        train_one(args, condition, payload, statistics, split)
        for condition in args.conditions
    ]
    summary = {
        "dataset": str(args.dataset.resolve()),
        "fold": FOLD0 if not args.episode_split else None,
        "episode_split": split_definition,
        "statistics": asdict(statistics),
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"completed": [item["condition"] for item in results]}), flush=True)


if __name__ == "__main__":
    main()
