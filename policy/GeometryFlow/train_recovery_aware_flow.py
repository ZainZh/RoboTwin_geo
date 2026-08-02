#!/usr/bin/env python3
"""Train a flow-bottleneck policy on nominal and on-policy recovery states.

The action decoder predicts only the demonstrated active arm (xyz, rotvec,
gripper).  Two auxiliary heads learn whether the policy should continue,
settle, or release and the remaining number of expert steps before release.
Simulator object poses are used only to construct these supervision labels;
the model input remains proprioception plus camera-derived rigid flow tokens.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .train_task_flow_benchmark import (
    FOLDS,
    FlowBottleneckTransformer,
    Normalization,
    TaskFlowDataset,
    fit_normalization,
    load_predicted_episode_flow,
    move_batch,
    seed_everything,
    summarize_errors,
)


@dataclass
class ActiveActionNormalization:
    mean: np.ndarray
    std: np.ndarray
    steps_log_mean: float
    steps_log_std: float


def active_action7(payload: dict[str, np.ndarray], indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    actions = np.asarray(payload["action"][indices], dtype=np.float32)
    active_right = np.asarray(payload["active_arm_right"][indices]) >= 0.5
    result = np.empty(actions.shape[:-1] + (7,), dtype=np.float32)
    result[~active_right] = actions[~active_right, :, :7]
    result[active_right] = actions[active_right, :, 7:14]
    return result


def fit_active_normalization(
    payload: dict[str, np.ndarray], indices: np.ndarray
) -> ActiveActionNormalization:
    action = active_action7(payload, indices).reshape(-1, 7)
    steps_log = np.log1p(
        np.asarray(payload["steps_to_release"][indices], dtype=np.float32)
    )
    return ActiveActionNormalization(
        mean=action.mean(axis=0).astype(np.float32),
        std=np.maximum(action.std(axis=0), 1e-4).astype(np.float32),
        steps_log_mean=float(steps_log.mean()),
        steps_log_std=float(max(steps_log.std(), 1e-4)),
    )


class RecoveryAwareTaskFlowDataset(TaskFlowDataset):
    def __init__(self, *args, active_normalization: ActiveActionNormalization, **kwargs):
        super().__init__(*args, **kwargs)
        self.active_normalization = active_normalization
        required = {"stop_phase", "steps_to_release"}
        missing = required.difference(self.payload)
        if missing:
            raise ValueError(f"recovery-aware labels are missing: {sorted(missing)}")

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        result = super().__getitem__(item)
        index = int(result["index"])
        offset = 7 if float(self.payload["active_arm_right"][index]) >= 0.5 else 0
        action = self.payload["action"][index, :, offset : offset + 7]
        action = (
            action - self.active_normalization.mean[None]
        ) / self.active_normalization.std[None]
        steps_log = np.log1p(float(self.payload["steps_to_release"][index]))
        result.update(
            {
                "active_action": torch.from_numpy(action.astype(np.float32)),
                "stop_phase": torch.tensor(
                    int(self.payload["stop_phase"][index]), dtype=torch.long
                ),
                "steps_to_release_log": torch.tensor(
                    (
                        steps_log - self.active_normalization.steps_log_mean
                    )
                    / self.active_normalization.steps_log_std,
                    dtype=torch.float32,
                ),
            }
        )
        return result


class RecoveryAwareFlowPolicy(nn.Module):
    """Strict SE(3)-flow bottleneck with active-arm and phase heads."""

    def __init__(
        self,
        condition: str,
        flow_steps: int,
        horizon: int,
        hidden_dim: int = 128,
        explicit_flow_progress: bool = True,
    ) -> None:
        super().__init__()
        self.condition = str(condition)
        self.horizon = int(horizon)
        self.backbone = FlowBottleneckTransformer(
            condition=condition,
            flow_steps=flow_steps,
            horizon=horizon,
            hidden_dim=hidden_dim,
            explicit_flow_progress=explicit_flow_progress,
            include_se3_tokens=True,
        )
        self.backbone.action_head = nn.Identity()
        self.active_action_head = nn.Linear(hidden_dim, 7)
        self.phase_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 3)
        )
        self.steps_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        memory = self.backbone.encode_memory(batch)
        action_tokens = self.backbone.decode_action_tokens(memory)
        pooled = memory.mean(dim=1)
        return {
            "active_action": self.active_action_head(action_tokens),
            "stop_logits": self.phase_head(pooled),
            "steps_to_release_log": self.steps_head(pooled).squeeze(-1),
        }


def weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (value * weight).sum() / weight.sum().clamp_min(1e-8)


def loss_terms(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    recovery_loss_weight: float,
    phase_class_weight: torch.Tensor,
    phase_loss_weight: float,
    steps_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    sample_weight = torch.where(
        batch["is_recovery_sample"] >= 0.5,
        torch.full_like(batch["is_recovery_sample"], float(recovery_loss_weight)),
        torch.ones_like(batch["is_recovery_sample"]),
    )
    action = (output["active_action"] - batch["active_action"]).square().mean(dim=(1, 2))
    phase = nn.functional.cross_entropy(
        output["stop_logits"],
        batch["stop_phase"],
        weight=phase_class_weight,
        reduction="none",
    )
    steps = nn.functional.smooth_l1_loss(
        output["steps_to_release_log"],
        batch["steps_to_release_log"],
        reduction="none",
    )
    terms = {
        "action": weighted_mean(action, sample_weight),
        "phase": weighted_mean(phase, sample_weight),
        "steps": weighted_mean(steps, sample_weight),
    }
    total = (
        terms["action"]
        + float(phase_loss_weight) * terms["phase"]
        + float(steps_loss_weight) * terms["steps"]
    )
    return total, terms


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    phase_class_weight: torch.Tensor,
    phase_loss_weight: float,
    steps_loss_weight: float,
) -> dict[str, float]:
    model.eval()
    totals = {key: 0.0 for key in ("total", "action", "phase", "steps")}
    count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        output = model(batch)
        total, terms = loss_terms(
            output,
            batch,
            recovery_loss_weight=1.0,
            phase_class_weight=phase_class_weight,
            phase_loss_weight=phase_loss_weight,
            steps_loss_weight=steps_loss_weight,
        )
        batch_count = len(batch["index"])
        totals["total"] += float(total) * batch_count
        for key in terms:
            totals[key] += float(terms[key]) * batch_count
        count += batch_count
    return {key: value / max(count, 1) for key, value in totals.items()}


def phase_class_weights(payload: dict[str, np.ndarray], indices: np.ndarray) -> np.ndarray:
    labels = np.asarray(payload["stop_phase"][indices], dtype=np.int64)
    counts = np.bincount(labels, minlength=3).astype(np.float64)
    # Square-root inverse frequency avoids allowing the rare settle class to
    # dominate the action objective while still making it learnable.
    weights = np.sqrt(counts.sum() / np.maximum(counts, 1.0))
    return (weights / weights.mean()).astype(np.float32)


def reconstruct_action14(
    active_action: np.ndarray, active_arm_right: np.ndarray
) -> np.ndarray:
    active_action = np.asarray(active_action, dtype=np.float32)
    active_right = np.asarray(active_arm_right) >= 0.5
    result = np.zeros(active_action.shape[:-1] + (14,), dtype=np.float32)
    result[~active_right, :, :7] = active_action[~active_right]
    result[active_right, :, 7:14] = active_action[active_right]
    return result


@torch.no_grad()
def predict_and_summarize(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    payload: dict[str, np.ndarray],
    normalization: ActiveActionNormalization,
) -> dict:
    model.eval()
    indices_parts = []
    action_parts = []
    phase_parts = []
    steps_parts = []
    for batch in loader:
        batch = move_batch(batch, device)
        output = model(batch)
        active = output["active_action"].cpu().numpy()
        active = active * normalization.std[None, None] + normalization.mean[None, None]
        indices_parts.append(batch["index"].cpu().numpy())
        action_parts.append(active)
        phase_parts.append(output["stop_logits"].argmax(dim=-1).cpu().numpy())
        steps_log = (
            output["steps_to_release_log"].cpu().numpy()
            * normalization.steps_log_std
            + normalization.steps_log_mean
        )
        steps_parts.append(np.maximum(np.expm1(steps_log), 0.0))
    indices = np.concatenate(indices_parts)
    active = np.concatenate(action_parts)
    phase_prediction = np.concatenate(phase_parts)
    steps_prediction = np.concatenate(steps_parts)
    action14 = reconstruct_action14(active, payload["active_arm_right"][indices])
    summary = summarize_errors(payload, indices, action14)
    phase_target = payload["stop_phase"][indices].astype(np.int64)
    confusion = np.zeros((3, 3), dtype=np.int64)
    np.add.at(confusion, (phase_target, phase_prediction), 1)
    recall = np.diag(confusion) / np.maximum(confusion.sum(axis=1), 1)
    summary.update(
        {
            "stop_phase_accuracy": float(np.mean(phase_prediction == phase_target)),
            "stop_phase_balanced_accuracy": float(recall.mean()),
            "stop_phase_recall": recall.tolist(),
            "stop_phase_confusion": confusion.tolist(),
            "steps_to_release_mae": float(
                np.mean(np.abs(steps_prediction - payload["steps_to_release"][indices]))
            ),
        }
    )
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", type=Path, required=True)
    result.add_argument("--test-flow-prediction", type=Path)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--conditions", nargs="+", default=("flow", "zero_flow"))
    result.add_argument("--fold", type=int, default=0)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--epochs", type=int, default=120)
    result.add_argument("--patience", type=int, default=18)
    result.add_argument("--batch-size", type=int, default=256)
    result.add_argument("--learning-rate", type=float, default=1e-3)
    result.add_argument("--weight-decay", type=float, default=1e-5)
    result.add_argument("--num-workers", type=int, default=2)
    result.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    result.add_argument("--recovery-target-fraction", type=float, default=0.3)
    result.add_argument("--phase-loss-weight", type=float, default=0.2)
    result.add_argument("--steps-loss-weight", type=float, default=0.05)
    result.add_argument("--sample-phase", choices=("all", "relation"), default="relation")
    result.add_argument("--flow-context-mode", choices=("full", "remaining"), default="full")
    return result


def main() -> None:
    args = parser().parse_args()
    if not 0.0 < args.recovery_target_fraction < 1.0:
        raise ValueError("recovery target fraction must lie in (0,1)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.dataset, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    test_payload = dict(payload)
    if args.test_flow_prediction is not None:
        test_payload["episode_flow"] = load_predicted_episode_flow(
            args.test_flow_prediction, len(payload["episode_flow"])
        )
    fold = FOLDS[int(args.fold)]
    masks = {
        split: np.flatnonzero(np.isin(payload["shoe_id"], shoes))
        for split, shoes in fold.items()
    }
    recovery = np.asarray(payload["is_recovery_sample"], dtype=np.float32)
    masks["validation"] = masks["validation"][recovery[masks["validation"]] < 0.5]
    masks["test"] = masks["test"][recovery[masks["test"]] < 0.5]
    if args.sample_phase == "relation":
        masks = {
            key: value[payload["relation_phase"][value] >= 0.5]
            for key, value in masks.items()
        }
    nominal_train = masks["train"][recovery[masks["train"]] < 0.5]
    recovery_train = masks["train"][recovery[masks["train"]] >= 0.5]
    if not len(nominal_train) or not len(recovery_train):
        raise RuntimeError("training requires both nominal and on-policy recovery rows")
    recovery_loss_weight = (
        args.recovery_target_fraction
        * len(nominal_train)
        / ((1.0 - args.recovery_target_fraction) * len(recovery_train))
    )
    base_normalization = fit_normalization(payload, nominal_train)
    active_normalization = fit_active_normalization(payload, nominal_train)
    phase_weights = torch.from_numpy(phase_class_weights(payload, nominal_train))
    device = torch.device(args.device)
    phase_weights = phase_weights.to(device)
    datasets = {
        "train": RecoveryAwareTaskFlowDataset(
            payload,
            masks["train"],
            base_normalization,
            active_normalization=active_normalization,
            flow_context_mode=args.flow_context_mode,
        ),
        "validation": RecoveryAwareTaskFlowDataset(
            payload,
            masks["validation"],
            base_normalization,
            active_normalization=active_normalization,
            flow_context_mode=args.flow_context_mode,
        ),
        "test": RecoveryAwareTaskFlowDataset(
            test_payload,
            masks["test"],
            base_normalization,
            active_normalization=active_normalization,
            flow_context_mode=args.flow_context_mode,
        ),
    }
    loaders = {
        key: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=key == "train",
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
        )
        for key, dataset in datasets.items()
    }
    results = []
    flow_steps = int(payload["episode_flow"].shape[2])
    horizon = int(payload["action"].shape[1])
    for condition in args.conditions:
        seed_everything(args.seed)
        model = RecoveryAwareFlowPolicy(condition, flow_steps, horizon).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        best_state = None
        best_validation = float("inf")
        best_epoch = -1
        stale = 0
        history = []
        for epoch in range(args.epochs):
            model.train()
            total = 0.0
            count = 0
            for batch in loaders["train"]:
                batch = move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                loss, _ = loss_terms(
                    model(batch),
                    batch,
                    recovery_loss_weight=recovery_loss_weight,
                    phase_class_weight=phase_weights,
                    phase_loss_weight=args.phase_loss_weight,
                    steps_loss_weight=args.steps_loss_weight,
                )
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += float(loss) * len(batch["index"])
                count += len(batch["index"])
            validation = evaluate_loss(
                model,
                loaders["validation"],
                device,
                phase_weights,
                args.phase_loss_weight,
                args.steps_loss_weight,
            )
            history.append({"epoch": epoch, "train": total / count, "validation": validation})
            if validation["total"] < best_validation:
                best_validation = validation["total"]
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
                print(json.dumps({"condition": condition, "epoch": epoch, "validation": validation}), flush=True)
            else:
                stale += 1
                if stale >= args.patience:
                    break
        if best_state is None:
            raise RuntimeError("training did not produce a checkpoint")
        model.load_state_dict(best_state)
        summary = predict_and_summarize(
            model, loaders["test"], device, test_payload, active_normalization
        )
        checkpoint = {
            "model_state": best_state,
            "model_config": {
                "condition": condition,
                "flow_steps": flow_steps,
                "horizon": horizon,
                "hidden_dim": 128,
                "explicit_flow_progress": True,
            },
            "normalization": asdict(base_normalization),
            "active_normalization": asdict(active_normalization),
            "phase_labels": {"continue": 0, "settle": 1, "release": 2},
            "training_args": vars(args),
        }
        checkpoint_path = args.output_dir / f"fold{args.fold}_{condition}_seed{args.seed}.pt"
        torch.save(checkpoint, checkpoint_path)
        result = {
            "fold": args.fold,
            "condition": condition,
            "seed": args.seed,
            "best_epoch": best_epoch,
            "best_validation": best_validation,
            "summary": summary,
            "history": history,
            "recovery_samples": int(len(recovery_train)),
            "recovery_loss_weight": float(recovery_loss_weight),
            "phase_class_weights": phase_weights.cpu().numpy().tolist(),
            "checkpoint": str(checkpoint_path.resolve()),
        }
        (args.output_dir / f"fold{args.fold}_{condition}_seed{args.seed}.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        results.append(result)
        print(json.dumps(result["summary"], indent=2), flush=True)
    (args.output_dir / f"summary_seed{args.seed}.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
