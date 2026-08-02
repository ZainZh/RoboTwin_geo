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
from scipy.spatial.transform import Rotation
from torch import nn
from torch.utils.data import DataLoader

from .eef_frame_action import payload_with_eef_frame_actions
from .train_task_flow_benchmark import (
    FOLDS,
    FlowBottleneckTransformer,
    Normalization,
    TaskFlowDataset,
    fit_normalization,
    geometric_flow_progress,
    load_predicted_episode_flow,
    move_batch,
    remaining_flow_from_current,
    rigid_flow_se3_tokens,
    seed_everything,
    summarize_errors,
)
from .train_flow_generator import rigid_rotation


@dataclass
class ActiveActionNormalization:
    mean: np.ndarray
    std: np.ndarray
    steps_log_mean: float
    steps_log_std: float


def flow_goal_rotvec(
    flow: np.ndarray, current_anchors: np.ndarray
) -> np.ndarray:
    """Observable current-to-flow-endpoint rotation in world coordinates."""
    rotation = rigid_rotation(
        np.asarray(current_anchors, dtype=np.float64),
        np.asarray(flow, dtype=np.float64)[:, -1],
    )
    return Rotation.from_matrix(rotation).as_rotvec().astype(np.float32)


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
    def __init__(
        self,
        *args,
        active_normalization: ActiveActionNormalization,
        deployment_episode_flow: np.ndarray | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.active_normalization = active_normalization
        self.deployment_episode_flow = (
            None
            if deployment_episode_flow is None
            else np.asarray(deployment_episode_flow, dtype=np.float32)
        )
        if (
            self.deployment_episode_flow is not None
            and self.deployment_episode_flow.shape != self.payload["episode_flow"].shape
        ):
            raise ValueError(
                "deployment flow shape does not match oracle episode flow: "
                f"{self.deployment_episode_flow.shape} != "
                f"{self.payload['episode_flow'].shape}"
            )
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
                "flow_goal_rotvec": torch.from_numpy(
                    flow_goal_rotvec(
                        self.payload["episode_flow"][
                            int(self.payload["episode_index"][index])
                        ],
                        self.payload["current_anchors"][index],
                    )
                ),
            }
        )
        if self.deployment_episode_flow is not None:
            episode = int(self.payload["episode_index"][index])
            current_anchors = self.payload["current_anchors"][index]
            flow = self.deployment_episode_flow[episode]
            progress = geometric_flow_progress(flow, current_anchors)
            if self.flow_context_mode == "remaining":
                flow = remaining_flow_from_current(flow, current_anchors)
            flow_track = np.concatenate(
                (
                    (current_anchors - self.norm.xyz_mean) / self.norm.xyz_std,
                    ((flow - current_anchors[:, None, :]) / self.norm.xyz_std).reshape(
                        len(current_anchors), -1
                    ),
                ),
                axis=-1,
            ).astype(np.float32)
            result.update(
                {
                    "deployment_flow": torch.from_numpy(flow_track),
                    "deployment_flow_se3": torch.from_numpy(
                        rigid_flow_se3_tokens(flow, current_anchors, self.norm.xyz_std)
                    ),
                    "deployment_flow_progress": torch.tensor(
                        [progress], dtype=torch.float32
                    ),
                    "deployment_flow_goal_rotvec": torch.from_numpy(
                        flow_goal_rotvec(flow, current_anchors)
                    ),
                }
            )
        return result


def deployment_flow_batch(
    batch: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Route a batch through its camera/deployment flow while retaining labels."""
    required = {
        "deployment_flow",
        "deployment_flow_se3",
        "deployment_flow_progress",
        "deployment_flow_goal_rotvec",
    }
    missing = required.difference(batch)
    if missing:
        raise KeyError(f"deployment-flow tensors are missing: {sorted(missing)}")
    return {
        **batch,
        "flow": batch["deployment_flow"],
        "flow_se3": batch["deployment_flow_se3"],
        "flow_progress": batch["deployment_flow_progress"],
        "flow_goal_rotvec": batch["deployment_flow_goal_rotvec"],
    }


class RecoveryAwareFlowPolicy(nn.Module):
    """Strict SE(3)-flow bottleneck with active-arm and phase heads."""

    def __init__(
        self,
        condition: str,
        flow_steps: int,
        horizon: int,
        hidden_dim: int = 128,
        explicit_flow_progress: bool = True,
        action_decoder: str = "coupled",
        rotation_parameterization: str = "direct",
        active_action_mean: np.ndarray | None = None,
        active_action_std: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.condition = str(condition)
        self.horizon = int(horizon)
        self.flow_steps = int(flow_steps)
        self.action_decoder = str(action_decoder)
        self.rotation_parameterization = str(rotation_parameterization)
        if self.action_decoder not in {"coupled", "factorized"}:
            raise ValueError(f"unknown action decoder {self.action_decoder!r}")
        if self.rotation_parameterization not in {"direct", "goal_aligned"}:
            raise ValueError(
                f"unknown rotation parameterization {self.rotation_parameterization!r}"
            )
        if (
            self.rotation_parameterization == "goal_aligned"
            and self.action_decoder != "factorized"
        ):
            raise ValueError("goal-aligned rotation requires the factorized decoder")
        backbone_kwargs = {
            "condition": condition,
            "flow_steps": flow_steps,
            "horizon": horizon,
            "hidden_dim": hidden_dim,
            "explicit_flow_progress": explicit_flow_progress,
            "include_se3_tokens": True,
        }
        if self.action_decoder == "coupled":
            self.backbone = FlowBottleneckTransformer(**backbone_kwargs)
            self.backbone.action_head = nn.Identity()
            phase_dim = hidden_dim
            self.active_action_head = nn.Linear(hidden_dim, 7)
        else:
            self.translation_backbone = FlowBottleneckTransformer(**backbone_kwargs)
            self.rotation_backbone = FlowBottleneckTransformer(**backbone_kwargs)
            self.translation_backbone.action_head = nn.Identity()
            self.rotation_backbone.action_head = nn.Identity()
            self.translation_action_head = nn.Linear(hidden_dim, 3)
            self.rotation_action_head = nn.Linear(hidden_dim, 3)
            self.gripper_action_head = nn.Linear(hidden_dim * 2, 1)
            phase_dim = hidden_dim * 2
            if self.rotation_parameterization == "goal_aligned":
                if active_action_mean is None or active_action_std is None:
                    raise ValueError(
                        "goal-aligned rotation requires active action normalization"
                    )
                self.register_buffer(
                    "rotation_action_mean",
                    torch.as_tensor(active_action_mean[3:6], dtype=torch.float32),
                )
                self.register_buffer(
                    "rotation_action_std",
                    torch.as_tensor(active_action_std[3:6], dtype=torch.float32),
                )
        self.phase_head = nn.Sequential(
            nn.LayerNorm(phase_dim),
            nn.Linear(phase_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        self.steps_head = nn.Sequential(
            nn.LayerNorm(phase_dim),
            nn.Linear(phase_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _factorized_batches(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Build causally separated translation and rotation flow inputs."""
        flow = batch["flow"]
        anchors = flow[..., :3]
        displacement = flow[..., 3:].reshape(
            *flow.shape[:-1], self.flow_steps, 3
        )
        centroid_displacement = displacement.mean(dim=1, keepdim=True)
        translation_flow = torch.cat(
            (
                anchors,
                centroid_displacement.expand_as(displacement).flatten(start_dim=-2),
            ),
            dim=-1,
        )
        rotation_flow = torch.cat(
            (
                anchors,
                (displacement - centroid_displacement).flatten(start_dim=-2),
            ),
            dim=-1,
        )

        se3 = batch["flow_se3"]
        translation_se3 = torch.zeros_like(se3)
        translation_se3[..., :3] = se3[..., :3]
        identity_sixd = se3.new_tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        translation_se3[..., 3:] = identity_sixd
        rotation_se3 = torch.zeros_like(se3)
        rotation_se3[..., 3:] = se3[..., 3:]

        # The original nearest-flow progress mixes translation and rotation.
        # Zeroing it prevents that scalar from reopening a cross-branch path;
        # each branch already observes the full residual trajectory it needs.
        zero_progress = torch.zeros_like(batch["flow_progress"])
        shared = {
            key: value
            for key, value in batch.items()
            if key not in {"flow", "flow_se3", "flow_progress"}
        }
        translation_batch = {
            **shared,
            "flow": translation_flow,
            "flow_se3": translation_se3,
            "flow_progress": zero_progress,
        }
        rotation_batch = {
            **shared,
            "flow": rotation_flow,
            "flow_se3": rotation_se3,
            "flow_progress": zero_progress,
        }
        return translation_batch, rotation_batch

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.action_decoder == "coupled":
            memory = self.backbone.encode_memory(batch)
            action_tokens = self.backbone.decode_action_tokens(memory)
            active_action = self.active_action_head(action_tokens)
            pooled = memory.mean(dim=1)
        else:
            translation_batch, rotation_batch = self._factorized_batches(batch)
            translation_memory = self.translation_backbone.encode_memory(
                translation_batch
            )
            rotation_memory = self.rotation_backbone.encode_memory(rotation_batch)
            translation_tokens = self.translation_backbone.decode_action_tokens(
                translation_memory
            )
            rotation_tokens = self.rotation_backbone.decode_action_tokens(
                rotation_memory
            )
            if self.rotation_parameterization == "goal_aligned":
                raw_rotation = self.rotation_action_head(rotation_tokens)
                goal = batch["flow_goal_rotvec"]
                if self.condition == "zero_flow":
                    goal = torch.zeros_like(goal)
                angle = torch.linalg.vector_norm(goal, dim=-1, keepdim=True)
                direction = goal / angle.clamp_min(1e-8)
                world_z = torch.zeros_like(direction)
                world_z[..., 2] = 1.0
                world_x = torch.zeros_like(direction)
                world_x[..., 0] = 1.0
                reference = torch.where(
                    (torch.abs(direction[..., 2:3]) < 0.9), world_z, world_x
                )
                tangent_first = nn.functional.normalize(
                    torch.cross(direction, reference, dim=-1), dim=-1
                )
                tangent_second = torch.cross(
                    direction, tangent_first, dim=-1
                )
                strength = torch.tanh(angle / 0.10)[:, None, :]
                parallel = (
                    torch.sigmoid(raw_rotation[..., :1]) * 0.15 * strength
                )
                orthogonal = (
                    torch.tanh(raw_rotation[..., 1:]) * 0.35 * parallel
                )
                physical_rotation = (
                    parallel * direction[:, None, :]
                    + orthogonal[..., :1] * tangent_first[:, None, :]
                    + orthogonal[..., 1:] * tangent_second[:, None, :]
                )
                normalized_rotation = (
                    physical_rotation - self.rotation_action_mean[None, None]
                ) / self.rotation_action_std[None, None]
            else:
                normalized_rotation = self.rotation_action_head(rotation_tokens)
            active_action = torch.cat(
                (
                    self.translation_action_head(translation_tokens),
                    normalized_rotation,
                    self.gripper_action_head(
                        torch.cat((translation_tokens, rotation_tokens), dim=-1)
                    ),
                ),
                dim=-1,
            )
            pooled = torch.cat(
                (
                    translation_memory.mean(dim=1),
                    rotation_memory.mean(dim=1),
                ),
                dim=-1,
            )
        return {
            "active_action": active_action,
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
    use_deployment_flow: bool = False,
) -> dict[str, float]:
    model.eval()
    totals = {key: 0.0 for key in ("total", "action", "phase", "steps")}
    count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        model_batch = deployment_flow_batch(batch) if use_deployment_flow else batch
        output = model(model_batch)
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
    use_deployment_flow: bool = False,
) -> dict:
    model.eval()
    indices_parts = []
    action_parts = []
    phase_parts = []
    steps_parts = []
    for batch in loader:
        batch = move_batch(batch, device)
        model_batch = deployment_flow_batch(batch) if use_deployment_flow else batch
        output = model(model_batch)
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
    result.add_argument(
        "--train-flow-prediction",
        type=Path,
        help=(
            "Camera/deployment flow for every episode. When supplied, the student "
            "is trained and evaluated on these flows while the base dataset retains "
            "oracle flow for an optional teacher."
        ),
    )
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
    result.add_argument("--action-frame", choices=("world", "eef"), default="world")
    result.add_argument(
        "--action-decoder", choices=("coupled", "factorized"), default="coupled"
    )
    result.add_argument(
        "--rotation-parameterization",
        choices=("direct", "goal_aligned"),
        default="direct",
    )
    result.add_argument("--init-checkpoint", type=Path)
    result.add_argument("--teacher-checkpoint", type=Path)
    result.add_argument("--distillation-weight", type=float, default=0.0)
    return result


def main() -> None:
    args = parser().parse_args()
    if not 0.0 < args.recovery_target_fraction < 1.0:
        raise ValueError("recovery target fraction must lie in (0,1)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.dataset, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    if args.action_frame == "eef":
        payload = payload_with_eef_frame_actions(payload)
    test_payload = dict(payload)
    if args.test_flow_prediction is not None:
        test_payload["episode_flow"] = load_predicted_episode_flow(
            args.test_flow_prediction, len(payload["episode_flow"])
        )
    deployment_episode_flow = None
    if args.train_flow_prediction is not None:
        deployment_episode_flow = load_predicted_episode_flow(
            args.train_flow_prediction, len(payload["episode_flow"])
        )
    if args.distillation_weight < 0.0:
        raise ValueError("distillation weight must be non-negative")
    if args.distillation_weight > 0.0 and args.teacher_checkpoint is None:
        raise ValueError("positive distillation weight requires --teacher-checkpoint")
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
            deployment_episode_flow=deployment_episode_flow,
        ),
        "validation": RecoveryAwareTaskFlowDataset(
            payload,
            masks["validation"],
            base_normalization,
            active_normalization=active_normalization,
            flow_context_mode=args.flow_context_mode,
            deployment_episode_flow=deployment_episode_flow,
        ),
        "test": RecoveryAwareTaskFlowDataset(
            test_payload,
            masks["test"],
            base_normalization,
            active_normalization=active_normalization,
            flow_context_mode=args.flow_context_mode,
            deployment_episode_flow=deployment_episode_flow,
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
        model = RecoveryAwareFlowPolicy(
            condition,
            flow_steps,
            horizon,
            action_decoder=args.action_decoder,
            rotation_parameterization=args.rotation_parameterization,
            active_action_mean=active_normalization.mean,
            active_action_std=active_normalization.std,
        ).to(device)
        if args.init_checkpoint is not None:
            initial = torch.load(
                args.init_checkpoint, map_location=device, weights_only=False
            )
            model.load_state_dict(initial["model_state"])
        teacher = None
        if args.teacher_checkpoint is not None:
            teacher_payload = torch.load(
                args.teacher_checkpoint, map_location=device, weights_only=False
            )
            teacher_config = dict(teacher_payload["model_config"])
            teacher = RecoveryAwareFlowPolicy(
                condition=str(teacher_config["condition"]),
                flow_steps=int(teacher_config["flow_steps"]),
                horizon=int(teacher_config["horizon"]),
                hidden_dim=int(teacher_config.get("hidden_dim", 128)),
                explicit_flow_progress=bool(
                    teacher_config.get("explicit_flow_progress", True)
                ),
                action_decoder=str(
                    teacher_config.get("action_decoder", "coupled")
                ),
                rotation_parameterization=str(
                    teacher_config.get("rotation_parameterization", "direct")
                ),
                active_action_mean=np.asarray(
                    teacher_payload["active_normalization"]["mean"]
                ),
                active_action_std=np.asarray(
                    teacher_payload["active_normalization"]["std"]
                ),
            ).to(device)
            teacher.load_state_dict(teacher_payload["model_state"])
            teacher.eval()
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
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
                student_batch = (
                    deployment_flow_batch(batch)
                    if deployment_episode_flow is not None
                    else batch
                )
                student_output = model(student_batch)
                loss, _ = loss_terms(
                    student_output,
                    batch,
                    recovery_loss_weight=recovery_loss_weight,
                    phase_class_weight=phase_weights,
                    phase_loss_weight=args.phase_loss_weight,
                    steps_loss_weight=args.steps_loss_weight,
                )
                if teacher is not None and args.distillation_weight > 0.0:
                    with torch.no_grad():
                        teacher_output = teacher(batch)
                    distillation = nn.functional.smooth_l1_loss(
                        student_output["active_action"],
                        teacher_output["active_action"],
                    )
                    loss = loss + float(args.distillation_weight) * distillation
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
                use_deployment_flow=deployment_episode_flow is not None,
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
            model,
            loaders["test"],
            device,
            test_payload,
            active_normalization,
            use_deployment_flow=deployment_episode_flow is not None,
        )
        checkpoint = {
            "model_state": best_state,
            "model_config": {
                "condition": condition,
                "flow_steps": flow_steps,
                "horizon": horizon,
                "hidden_dim": 128,
                "explicit_flow_progress": True,
                "action_frame": args.action_frame,
                "flow_context_mode": args.flow_context_mode,
                "action_decoder": args.action_decoder,
                "rotation_parameterization": args.rotation_parameterization,
                "trained_with_deployment_flow": deployment_episode_flow is not None,
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
