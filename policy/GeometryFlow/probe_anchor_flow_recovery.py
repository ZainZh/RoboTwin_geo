#!/usr/bin/env python3
"""Test whether camera-derived anchor flow identifies recovery translations.

This is an offline representation probe, not a deployment policy. Hyperparameters
are selected by leave-one-training-shoe-out validation; the requested evaluation
archive remains fully held out.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def rotation_from_columns(columns6: np.ndarray) -> np.ndarray:
    value = np.asarray(columns6, dtype=np.float64).reshape(6)
    first = value[:3]
    first /= max(float(np.linalg.norm(first)), 1.0e-8)
    second = value[3:] - first * float(np.dot(first, value[3:]))
    second /= max(float(np.linalg.norm(second)), 1.0e-8)
    return np.column_stack((first, second, np.cross(first, second)))


def active_endpoint_translation(payload: dict[str, np.ndarray], mask: np.ndarray) -> np.ndarray:
    action = np.asarray(payload["action"])[mask]
    active_right = np.asarray(payload["active_arm_right"])[mask] >= 0.5
    return np.where(
        active_right[:, None],
        action[:, :, 7:10].sum(axis=1),
        action[:, :, :3].sum(axis=1),
    ).astype(np.float64)


def summary_blocks(value: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (
            value.mean(axis=0),
            value.std(axis=0),
            value.min(axis=0),
            value.max(axis=0),
        )
    )


def extract_features(
    payload: dict[str, np.ndarray], mask: np.ndarray
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    indices = np.flatnonzero(mask)
    features = {
        "relation": [],
        "eef_flow": [],
        "anchor_flow": [],
        "dual_eef_anchor_flow": [],
    }
    analytic_eef_flow = []
    relation_translation = []
    active_right_all = np.asarray(payload["active_arm_right"])[indices] >= 0.5
    states = np.asarray(payload["state"])[indices]
    eef_positions = np.where(
        active_right_all[:, None], states[:, 10:13], states[:, :3]
    )
    for row, active_right, eef_position in zip(indices, active_right_all, eef_positions):
        goal = np.asarray(payload["goal_frame9"][row], dtype=np.float64)
        relation = np.asarray(payload["target_frame9"][row], dtype=np.float64)
        goal_rotation = rotation_from_columns(goal[3:])
        relative_rotation = rotation_from_columns(relation[3:])
        source_rotation = relative_rotation.T @ goal_rotation
        source_position = goal[:3] - relation[:3]
        eef_local = (eef_position - source_position) @ source_rotation
        desired_eef = eef_local @ goal_rotation.T + goal[:3]
        eef_flow = desired_eef - eef_position

        anchors = np.asarray(payload["points_a"][row, :, :3], dtype=np.float64)
        anchor_local = (anchors - source_position) @ source_rotation
        desired_anchors = anchor_local @ goal_rotation.T + goal[:3]
        anchor_flow = desired_anchors - anchors
        anchor_offset = anchors - eef_position
        left_offset = anchors - np.asarray(payload["state"][row, :3])
        right_offset = anchors - np.asarray(payload["state"][row, 10:13])
        arm = np.asarray([not active_right, active_right], dtype=np.float64)

        relation_feature = np.concatenate((relation, eef_local, arm))
        eef_feature = np.concatenate((relation_feature, eef_flow))
        anchor_feature = np.concatenate(
            (
                eef_feature,
                summary_blocks(anchor_local),
                summary_blocks(anchor_offset),
                summary_blocks(anchor_flow),
            )
        )
        features["relation"].append(relation_feature)
        features["eef_flow"].append(eef_feature)
        features["anchor_flow"].append(anchor_feature)
        features["dual_eef_anchor_flow"].append(
            np.concatenate(
                (
                    relation,
                    arm,
                    summary_blocks(
                        np.concatenate((left_offset, right_offset, anchor_flow), axis=-1)
                    ),
                )
            )
        )
        analytic_eef_flow.append(eef_flow)
        relation_translation.append(relation[:3])
    return (
        {key: np.asarray(value) for key, value in features.items()},
        np.asarray(analytic_eef_flow),
        np.asarray(relation_translation),
        np.asarray(payload["shoe_id"])[indices].astype(np.int64),
    )


def prediction_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = np.linalg.norm(prediction - target, axis=-1)
    pred_norm = np.linalg.norm(prediction, axis=-1)
    target_norm = np.linalg.norm(target, axis=-1)
    cosine = np.sum(prediction * target, axis=-1) / np.maximum(
        pred_norm * target_norm, 1.0e-9
    )
    return {
        "component_mse": float(np.mean(np.square(prediction - target))),
        "endpoint_error_cm_mean": float(100.0 * np.mean(error)),
        "endpoint_error_cm_median": float(100.0 * np.median(error)),
        "endpoint_error_cm_p95": float(100.0 * np.quantile(error, 0.95)),
        "cosine_mean": float(np.mean(cosine)),
        "cosine_p05": float(np.quantile(cosine, 0.05)),
    }


def choose_alpha(
    features: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    alphas: list[float],
) -> tuple[float, dict[str, float]]:
    group_values = sorted(np.unique(groups).astype(int))
    scores = {}
    for alpha in alphas:
        fold_errors = []
        for group in group_values:
            validation = groups == group
            training = ~validation
            if not np.any(training) or not np.any(validation):
                continue
            model = make_pipeline(StandardScaler(), Ridge(alpha=float(alpha)))
            model.fit(features[training], target[training])
            prediction = model.predict(features[validation])
            fold_errors.append(
                float(np.linalg.norm(prediction - target[validation], axis=-1).mean())
            )
        scores[str(alpha)] = float(np.mean(fold_errors))
    best = min(alphas, key=lambda value: scores[str(value)])
    return float(best), scores


def load_positive(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    marker = np.asarray(payload["is_translation_recovery_positive"]) >= 0.5
    if not np.any(marker):
        raise ValueError(f"no translation recovery positives in {path}")
    return payload, marker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0, 100.0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train_payload, train_mask = load_positive(args.train)
    eval_payload, eval_mask = load_positive(args.evaluation)
    train_features, train_eef_flow, train_relation, train_groups = extract_features(
        train_payload, train_mask
    )
    eval_features, eval_eef_flow, eval_relation, eval_groups = extract_features(
        eval_payload, eval_mask
    )
    train_target = active_endpoint_translation(train_payload, train_mask)
    eval_target = active_endpoint_translation(eval_payload, eval_mask)

    report = {
        "train": str(args.train.resolve()),
        "evaluation": str(args.evaluation.resolve()),
        "train_samples": int(len(train_target)),
        "evaluation_samples": int(len(eval_target)),
        "train_shoes": sorted(np.unique(train_groups).astype(int).tolist()),
        "evaluation_shoes": sorted(np.unique(eval_groups).astype(int).tolist()),
        "baselines": {
            "relation_translation": prediction_metrics(eval_relation, eval_target),
            "analytic_eef_flow": prediction_metrics(eval_eef_flow, eval_target),
        },
        "ridge": {},
    }
    for mode in ("relation", "eef_flow", "anchor_flow", "dual_eef_anchor_flow"):
        alpha, cv_scores = choose_alpha(
            train_features[mode], train_target, train_groups, list(args.alphas)
        )
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(train_features[mode], train_target)
        report["ridge"][mode] = {
            "selected_alpha": alpha,
            "leave_one_training_shoe_out_error_m": cv_scores,
            "train": prediction_metrics(model.predict(train_features[mode]), train_target),
            "evaluation": prediction_metrics(model.predict(eval_features[mode]), eval_target),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
