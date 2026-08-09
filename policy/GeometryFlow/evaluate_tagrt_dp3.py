#!/usr/bin/env python3
"""Same-checkpoint causal evaluation for the TAGRT-conditioned DP3 backbone."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch.utils.data import DataLoader


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DP3_ROOT = WORKSPACE_ROOT / "policy" / "DP3" / "3D-Diffusion-Policy"
sys.path.insert(0, str(DP3_ROOT))
sys.path.insert(0, str(DP3_ROOT.parent))

from diffusion_policy_3d.common.pytorch_util import dict_apply  # noqa: E402
from diffusion_policy_3d.dataset.tagrt_dataset import (  # noqa: E402
    TaskAlignedGeometryDataset,
)
from train_dp3 import TrainDP3Workspace  # noqa: E402


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--npz", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--batch-size", type=int, default=256)
    result.add_argument("--sample-repeats", type=int, default=3)
    result.add_argument(
        "--conditions", nargs="+", default=("correct", "zero", "shuffled")
    )
    return result


def _endpoint_error(
    predicted: np.ndarray, target: np.ndarray, active_right: bool
) -> tuple[float, float]:
    offset = 7 if bool(active_right) else 0
    pred = np.asarray(predicted, dtype=np.float64)[:, offset : offset + 6]
    truth = np.asarray(target, dtype=np.float64)[:, offset : offset + 6]
    translation_cm = float(
        np.linalg.norm(pred[:, :3].sum(axis=0) - truth[:, :3].sum(axis=0))
        * 100.0
    )
    pred_rotation = Rotation.identity()
    truth_rotation = Rotation.identity()
    for pred_step, truth_step in zip(pred[:, 3:6], truth[:, 3:6]):
        pred_rotation = pred_rotation * Rotation.from_rotvec(pred_step)
        truth_rotation = truth_rotation * Rotation.from_rotvec(truth_step)
    rotation_deg = float(
        np.rad2deg((pred_rotation.inv() * truth_rotation).magnitude())
    )
    return translation_cm, rotation_deg


def _condition_dataset(cfg, path: Path, condition: str):
    dataset = TaskAlignedGeometryDataset(
        str(path.resolve()),
        horizon=int(cfg.horizon),
        n_obs_steps=int(cfg.n_obs_steps),
        n_action_steps=int(cfg.n_action_steps),
        train_object_ids=list(cfg.task.dataset.train_object_ids),
        val_object_ids=list(cfg.task.dataset.val_object_ids),
        test_object_ids=list(cfg.task.dataset.test_object_ids),
        split="test",
        condition_mode=str(condition),
        use_color=bool(cfg.task.dataset.get("use_color", False)),
    )
    return dataset


def main() -> None:
    args = parser().parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if set(args.conditions) - {"correct", "zero", "shuffled"}:
        raise ValueError("conditions must be correct, zero, and/or shuffled")
    payload = torch.load(
        args.checkpoint.resolve().open("rb"),
        pickle_module=__import__("dill"),
        map_location="cpu",
    )
    workspace = TrainDP3Workspace(payload["cfg"], output_dir=str(args.output.parent))
    workspace.load_payload(payload, exclude_keys=("optimizer",))
    policy = workspace.ema_model if workspace.ema_model is not None else workspace.model
    device = torch.device(args.device)
    policy.to(device).eval()
    cfg = payload["cfg"]
    start = int(cfg.n_obs_steps) - 1
    steps = int(cfg.n_action_steps)
    all_records = {}

    with torch.no_grad():
        for condition in args.conditions:
            dataset = _condition_dataset(cfg, args.npz, str(condition))
            loader = DataLoader(
                dataset,
                batch_size=int(args.batch_size),
                shuffle=False,
                num_workers=1,
                pin_memory=True,
            )
            rows = np.asarray(dataset.active_indices, dtype=np.int64)
            condition_records = []
            cursor = 0
            for repeat in range(int(args.sample_repeats)):
                cursor = 0
                for batch_index, batch in enumerate(loader):
                    batch = dict_apply(
                        batch, lambda value: value.to(device, non_blocking=True)
                    )
                    # Identical diffusion noise for every causal condition.
                    torch.manual_seed(420_000 + repeat * 10_000 + batch_index)
                    prediction = policy.predict_action(batch["obs"])["action"]
                    target = batch["action"][:, start : start + steps]
                    pred_np = prediction.cpu().numpy()
                    target_np = target.cpu().numpy()
                    raw_abs = np.abs(pred_np - target_np)
                    batch_rows = rows[cursor : cursor + len(pred_np)]
                    cursor += len(pred_np)
                    for local_index, row in enumerate(batch_rows):
                        active_right = bool(
                            np.asarray(
                                dataset.payload.get(
                                    "active_arm_right",
                                    np.ones(len(dataset.payload["action"])),
                                )
                            )[row]
                        )
                        endpoint_cm, endpoint_deg = _endpoint_error(
                            pred_np[local_index], target_np[local_index], active_right
                        )
                        active_slice = slice(7, 14) if active_right else slice(0, 7)
                        gripper_index = 13 if active_right else 6
                        predicted_closed = (
                            pred_np[local_index, :, gripper_index] < 0.5
                        )
                        target_closed = (
                            target_np[local_index, :, gripper_index] < 0.5
                        )
                        condition_records.append(
                            {
                                "repeat": int(repeat),
                                "row": int(row),
                                "object_id": int(dataset.payload["shoe_id"][row]),
                                "relation_phase": float(
                                    np.asarray(
                                        dataset.payload.get(
                                            "relation_phase",
                                            np.ones(len(dataset.payload["action"])),
                                        )
                                    )[row]
                                ),
                                "source_sample": int(
                                    dataset.payload.get(
                                        "multigoal_source_sample",
                                        np.arange(len(dataset.payload["action"])),
                                    )[row]
                                ),
                                "raw_action_mae": float(raw_abs[local_index].mean()),
                                "active_arm_mae": float(
                                    raw_abs[local_index, :, active_slice].mean()
                                ),
                                "endpoint_translation_cm": endpoint_cm,
                                "endpoint_rotation_deg": endpoint_deg,
                                "active_gripper_correct": int(
                                    np.sum(predicted_closed == target_closed)
                                ),
                                "active_gripper_predicted_closed": int(
                                    np.sum(predicted_closed)
                                ),
                                "active_gripper_target_closed": int(
                                    np.sum(target_closed)
                                ),
                                "active_gripper_true_positive": int(
                                    np.sum(predicted_closed & target_closed)
                                ),
                            }
                        )
            all_records[str(condition)] = condition_records

    metrics = (
        "raw_action_mae",
        "active_arm_mae",
        "endpoint_translation_cm",
        "endpoint_rotation_deg",
    )
    summaries = {
        condition: {
            metric: float(np.mean([record[metric] for record in records]))
            for metric in metrics
        }
        for condition, records in all_records.items()
    }
    gripper_summaries = {}
    for condition, records in all_records.items():
        total = max(len(records) * steps, 1)
        target_closed = sum(
            record["active_gripper_target_closed"] for record in records
        )
        predicted_closed = sum(
            record["active_gripper_predicted_closed"] for record in records
        )
        true_positive = sum(
            record["active_gripper_true_positive"] for record in records
        )
        gripper_summaries[condition] = {
            "active_gripper_accuracy": float(
                sum(record["active_gripper_correct"] for record in records)
                / total
            ),
            "predicted_closed_fraction": float(predicted_closed / total),
            "target_closed_fraction": float(target_closed / total),
            "closed_recall": float(true_positive / max(target_closed, 1)),
            "closed_precision": float(true_positive / max(predicted_closed, 1)),
        }

    def summarize_records(records):
        return {
            metric: float(np.mean([record[metric] for record in records]))
            for metric in metrics
        }

    summaries_by_phase = {}
    for condition, records in all_records.items():
        summaries_by_phase[condition] = {}
        for phase_name, predicate in (
            ("pre_grasp", lambda value: value < 0.5),
            ("post_grasp", lambda value: value >= 0.5),
        ):
            selected = [
                record for record in records if predicate(record["relation_phase"])
            ]
            if selected:
                summaries_by_phase[condition][phase_name] = {
                    "rows": int(len(selected)),
                    **summarize_records(selected),
                }
    paired = {}
    paired_by_phase = {}
    if "correct" in all_records:
        correct = {
            (record["repeat"], record["row"]): record
            for record in all_records["correct"]
        }
        for condition in args.conditions:
            if condition == "correct":
                continue
            candidate = {
                (record["repeat"], record["row"]): record
                for record in all_records[str(condition)]
            }
            common = sorted(set(correct) & set(candidate))
            paired[str(condition)] = {
                metric: float(
                    np.mean(
                        [
                            correct[key][metric] - candidate[key][metric]
                            for key in common
                        ]
                    )
                )
                for metric in metrics
            }
            paired[str(condition)]["sign"] = (
                "negative means correct geometry is better"
            )
            paired_by_phase[str(condition)] = {}
            for phase_name, predicate in (
                ("pre_grasp", lambda value: value < 0.5),
                ("post_grasp", lambda value: value >= 0.5),
            ):
                phase_keys = [
                    key
                    for key in common
                    if predicate(correct[key]["relation_phase"])
                ]
                if not phase_keys:
                    continue
                paired_by_phase[str(condition)][phase_name] = {
                    "rows": int(len(phase_keys)),
                    **{
                        metric: float(
                            np.mean(
                                [
                                    correct[key][metric] - candidate[key][metric]
                                    for key in phase_keys
                                ]
                            )
                        )
                        for metric in metrics
                    },
                    "sign": "negative means correct geometry is better",
                }
    result = {
        "protocol": "tagrt_dp3_same_checkpoint_offline_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "npz": str(args.npz.resolve()),
        "test_objects": list(cfg.task.dataset.test_object_ids),
        "test_rows": int(len(_condition_dataset(cfg, args.npz, "correct"))),
        "sample_repeats": int(args.sample_repeats),
        "summaries": summaries,
        "gripper_summaries": gripper_summaries,
        "summaries_by_phase": summaries_by_phase,
        "paired_correct_minus_intervention": paired,
        "paired_correct_minus_intervention_by_phase": paired_by_phase,
        "records": all_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "summaries": summaries,
                "gripper_summaries": gripper_summaries,
                "summaries_by_phase": summaries_by_phase,
                "paired": paired,
                "paired_by_phase": paired_by_phase,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
