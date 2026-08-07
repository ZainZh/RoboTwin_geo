#!/usr/bin/env python3
"""Cross-fold, object-clustered summary for dense functional-frame policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .evaluate_dense_functional_frame_ablation import (
    METRICS,
    MODES,
    load_policy,
    per_sample_metrics,
)
from .train_interaction_flow_tokens import InteractionFlowDataset, predict
from .train_task_flow_benchmark import FOLDS


EXPECTED_PROTOCOL = {
    "epochs": 200,
    "patience": 30,
    "batch_size": 32,
    "learning_rate": 0.0003,
    "weight_decay": 0.00001,
    "feature_dim": 128,
    "layers": 2,
    "num_anchors": 16,
    "flow_steps": 4,
    "geometry_loss_weight": 0.1,
    "rigidity_loss_weight": 0.02,
    "rotation_loss_weight": 0.0,
    "flow_loss_mode": "unordered",
    "motion_loss_scale": 10.0,
    "flow_parameterization": "free",
    "target_horizon_mode": "terminal",
    "geometry_phase": "all",
    "active_arm_loss_weight": 1.0,
    "relation_loss_weight": 1.0,
    "rotation_action_loss_weight": 1.0,
    "action_frame": "world",
    "functional_frame_source": "dataset_relative",
    "sample_mode": "all",
    "episode_balanced_sampling": True,
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--dataset-template",
        default=(
            "outputs/geometry_flow/remote4090/"
            "task_flow_remote50_ndf_ensemble_camera_conf_fold{fold}.npz"
        ),
    )
    result.add_argument(
        "--checkpoint-template",
        default=(
            "outputs/geometry_flow/remote4090/dense_functional_frame_crossfold/"
            "fold{fold}/seed{seed}_{mode}/"
            "fold{fold}_interaction_functional_direct_flow_seed{seed}.pt"
        ),
    )
    result.add_argument(
        "--dataset-override",
        action="append",
        default=[],
        metavar="FOLD=PATH",
        help="Override the dataset template for one fold; repeat as needed.",
    )
    result.add_argument(
        "--checkpoint-template-override",
        action="append",
        default=[],
        metavar="FOLD=TEMPLATE",
        help=(
            "Override the checkpoint template for one fold. The value may use "
            "{fold}, {seed}, and {mode}."
        ),
    )
    result.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    result.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--batch-size", type=int, default=256)
    result.add_argument("--num-workers", type=int, default=0)
    result.add_argument("--bootstrap-samples", type=int, default=100_000)
    result.add_argument("--bootstrap-seed", type=int, default=20260804)
    result.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    return result


def parse_fold_overrides(values: list[str], label: str) -> dict[int, str]:
    result: dict[int, str] = {}
    for value in values:
        try:
            raw_fold, path = value.split("=", 1)
            fold = int(raw_fold)
        except (ValueError, TypeError) as error:
            raise ValueError(f"invalid {label} override {value!r}; expected FOLD=VALUE") from error
        if fold in result:
            raise ValueError(f"duplicate {label} override for fold {fold}")
        if not path:
            raise ValueError(f"empty {label} override for fold {fold}")
        result[fold] = path
    return result


def expand_template(template: str, *, fold: int, seed: int = 0, mode: str = "normal") -> Path:
    try:
        value = template.format(fold=int(fold), seed=int(seed), mode=str(mode))
    except (KeyError, ValueError) as error:
        raise ValueError(f"invalid path template {template!r}: {error}") from error
    return Path(value)


def paired_cluster_interval(
    cluster_values: np.ndarray,
    samples: int,
    generator: np.random.Generator,
) -> list[float]:
    """Percentile interval with the independent object/episode as the unit."""
    values = np.asarray(cluster_values, dtype=np.float64).reshape(-1)
    if not len(values):
        raise ValueError("paired cluster bootstrap requires at least one cluster")
    draws = generator.integers(0, len(values), size=(int(samples), len(values)))
    means = values[draws].mean(axis=1)
    return [float(value) for value in np.quantile(means, (0.025, 0.975))]


def adjacent_result(checkpoint_path: Path) -> dict:
    result_path = checkpoint_path.with_suffix(".json")
    if not result_path.is_file():
        raise FileNotFoundError(f"missing result JSON beside checkpoint: {result_path}")
    return json.loads(result_path.read_text(encoding="utf-8"))


def validate_result(
    result: dict,
    *,
    fold: int,
    seed: int,
    mode: str,
) -> tuple:
    if int(result.get("fold", -1)) != int(fold):
        raise ValueError(f"result fold mismatch: expected {fold}, got {result.get('fold')}")
    if int(result.get("base_seed", -1)) != int(seed):
        raise ValueError(
            f"result base seed mismatch: expected {seed}, got {result.get('base_seed')}"
        )
    if result.get("condition") != "interaction_functional_direct_flow":
        raise ValueError(f"unexpected condition: {result.get('condition')!r}")
    config = result.get("config", {})
    if config.get("target_frame_mode") != mode:
        raise ValueError(
            f"target-frame training mode mismatch: expected {mode}, "
            f"got {config.get('target_frame_mode')!r}"
        )
    actual = {key: config.get(key) for key in EXPECTED_PROTOCOL}
    if actual != EXPECTED_PROTOCOL:
        mismatch = {
            key: {"expected": expected, "actual": actual[key]}
            for key, expected in EXPECTED_PROTOCOL.items()
            if actual[key] != expected
        }
        raise ValueError(f"fold-0 protocol mismatch: {mismatch}")
    return tuple(actual[key] for key in EXPECTED_PROTOCOL)


def main() -> None:
    args = parser().parse_args()
    folds = [int(value) for value in args.folds]
    seeds = [int(value) for value in args.seeds]
    if len(set(folds)) != len(folds) or any(fold < 0 or fold >= len(FOLDS) for fold in folds):
        raise ValueError(f"folds must be unique and lie in [0,{len(FOLDS) - 1}]")
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("seeds must be a non-empty unique list")
    if int(args.bootstrap_samples) <= 0:
        raise ValueError("bootstrap-samples must be positive")
    dataset_overrides = parse_fold_overrides(args.dataset_override, "dataset")
    checkpoint_overrides = parse_fold_overrides(
        args.checkpoint_template_override, "checkpoint-template"
    )
    unknown_overrides = (set(dataset_overrides) | set(checkpoint_overrides)) - set(folds)
    if unknown_overrides:
        raise ValueError(f"overrides supplied for unrequested folds: {sorted(unknown_overrides)}")

    device = torch.device(args.device)
    fold_episodes: dict[int, np.ndarray] = {}
    fold_episode_shoes: dict[int, np.ndarray] = {}
    fold_values: dict[int, dict[str, np.ndarray]] = {}
    frame_weighted: dict[str, list[dict]] = {mode: [] for mode in MODES}
    checkpoints: dict[str, str] = {}
    datasets: dict[str, str] = {}
    reference_protocol = None
    parameter_count = None
    seen_test_shoes: set[int] = set()

    for fold in folds:
        dataset_template = dataset_overrides.get(fold, args.dataset_template)
        dataset_path = expand_template(dataset_template, fold=fold)
        if not dataset_path.is_file():
            raise FileNotFoundError(dataset_path)
        datasets[str(fold)] = str(dataset_path)
        metadata_path = dataset_path.with_suffix(".json")
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata_fold = metadata.get("fold")
            if metadata_fold is not None and int(metadata_fold) != fold:
                raise ValueError(
                    f"dataset metadata fold mismatch for {dataset_path}: "
                    f"expected {fold}, found {metadata_fold}"
                )
        with np.load(dataset_path, allow_pickle=False) as archive:
            payload = {key: np.asarray(archive[key]) for key in archive.files}
        test_indices = np.flatnonzero(
            np.isin(payload["shoe_id"], np.asarray(FOLDS[fold]["test"]))
        )
        if not len(test_indices):
            raise ValueError(f"fold {fold} has no test samples")
        episodes = np.unique(payload["episode_id"][test_indices])
        episode_shoes = []
        for episode in episodes:
            shoes = np.unique(payload["shoe_id"][test_indices][
                payload["episode_id"][test_indices] == episode
            ])
            if len(shoes) != 1:
                raise ValueError(f"fold {fold} episode {episode} spans multiple shoes")
            episode_shoes.append(int(shoes[0]))
        episode_shoes_array = np.asarray(episode_shoes, dtype=np.int64)
        actual_test_shoes = set(episode_shoes_array.tolist())
        expected_test_shoes = set(FOLDS[fold]["test"])
        if actual_test_shoes != expected_test_shoes:
            raise ValueError(
                f"fold {fold} test shoes mismatch: expected {sorted(expected_test_shoes)}, "
                f"found {sorted(actual_test_shoes)}"
            )
        overlap = seen_test_shoes & actual_test_shoes
        if overlap:
            raise ValueError(f"test shoes repeated across folds: {sorted(overlap)}")
        seen_test_shoes |= actual_test_shoes
        fold_episodes[fold] = episodes
        fold_episode_shoes[fold] = episode_shoes_array
        fold_values[fold] = {
            mode: np.empty((len(seeds), len(episodes), len(METRICS)), dtype=np.float64)
            for mode in MODES
        }

        checkpoint_template = checkpoint_overrides.get(fold, args.checkpoint_template)
        for mode in MODES:
            for seed_index, seed in enumerate(seeds):
                checkpoint_path = expand_template(
                    checkpoint_template, fold=fold, seed=seed, mode=mode
                )
                if not checkpoint_path.is_file():
                    raise FileNotFoundError(checkpoint_path)
                checkpoints[f"fold{fold}_seed{seed}_{mode}"] = str(checkpoint_path)
                result = adjacent_result(checkpoint_path)
                protocol = validate_result(result, fold=fold, seed=seed, mode=mode)
                if reference_protocol is None:
                    reference_protocol = protocol
                elif protocol != reference_protocol:
                    raise ValueError(
                        f"training protocol mismatch in {checkpoint_path}: "
                        f"{protocol!r} != {reference_protocol!r}"
                    )
                current_parameters = int(result["parameters"])
                if parameter_count is None:
                    parameter_count = current_parameters
                elif current_parameters != parameter_count:
                    raise ValueError(
                        f"parameter-count mismatch in {checkpoint_path}: "
                        f"{current_parameters} != {parameter_count}"
                    )

                checkpoint, normalization, model = load_policy(checkpoint_path, device)
                dataset = InteractionFlowDataset(
                    payload,
                    test_indices,
                    normalization,
                    flow_steps=int(checkpoint["flow_steps"]),
                    target_horizon_mode=str(
                        checkpoint.get("target_horizon_mode", "terminal")
                    ),
                    target_frame_mode=mode,
                    functional_frame_translation_mode=str(
                        checkpoint.get("functional_frame_translation_mode", "absolute")
                    ),
                )
                loader = DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                )
                indices, prediction, _ = predict(
                    model,
                    loader,
                    device,
                    normalization,
                    action_frame=str(checkpoint.get("action_frame", "world")),
                )
                order = np.argsort(indices)
                indices = indices[order]
                metrics = per_sample_metrics(payload, indices, prediction[order])
                frame_weighted[mode].append(
                    {
                        "fold": fold,
                        "seed": seed,
                        **{metric: float(metrics[metric].mean()) for metric in METRICS},
                    }
                )
                sample_episode = payload["episode_id"][indices]
                for episode_index, episode in enumerate(episodes):
                    keep = sample_episode == episode
                    for metric_index, metric in enumerate(METRICS):
                        fold_values[fold][mode][seed_index, episode_index, metric_index] = float(
                            metrics[metric][keep].mean()
                        )

    generator = np.random.default_rng(int(args.bootstrap_seed))
    output: dict = {
        "folds": folds,
        "seeds": seeds,
        "modes": list(MODES),
        "metrics": list(METRICS),
        "datasets": datasets,
        "checkpoints": checkpoints,
        "parameter_count": parameter_count,
        "independent_test_objects": len(seen_test_shoes),
        "independent_test_episodes": int(sum(len(value) for value in fold_episodes.values())),
        "frame_weighted_per_fold_seed": frame_weighted,
        "episode_balanced_mean": {},
        "object_balanced_mean": {},
        "fold_episode_balanced_mean": {},
        "comparisons": {},
        "statistical_unit": {
            "frame_aggregation": "mean within independent episode",
            "primary_interval": "paired percentile bootstrap over held-out object means",
            "secondary_interval": "paired percentile bootstrap over episode means",
            "optimization_repeats": "policy seeds are averaged within each cluster",
        },
    }

    for mode in MODES:
        concatenated = np.concatenate([fold_values[fold][mode] for fold in folds], axis=1)
        output["episode_balanced_mean"][mode] = {
            metric: float(concatenated[..., index].mean())
            for index, metric in enumerate(METRICS)
        }
        object_means = []
        for fold in folds:
            for shoe in sorted(set(fold_episode_shoes[fold].tolist())):
                keep = fold_episode_shoes[fold] == shoe
                object_means.append(fold_values[fold][mode][:, keep].mean(axis=(0, 1)))
        object_means_array = np.asarray(object_means)
        output["object_balanced_mean"][mode] = {
            metric: float(object_means_array[:, index].mean())
            for index, metric in enumerate(METRICS)
        }
        output["fold_episode_balanced_mean"][mode] = {
            str(fold): {
                metric: float(fold_values[fold][mode][..., index].mean())
                for index, metric in enumerate(METRICS)
            }
            for fold in folds
        }

    for control in ("zero", "shuffled"):
        comparison_name = f"normal_minus_{control}"
        output["comparisons"][comparison_name] = {}
        for metric_index, metric in enumerate(METRICS):
            fold_delta = {
                fold: fold_values[fold]["normal"][..., metric_index]
                - fold_values[fold][control][..., metric_index]
                for fold in folds
            }
            episode_means = np.concatenate(
                [fold_delta[fold].mean(axis=0) for fold in folds]
            )
            object_means = []
            per_object = {}
            fold_seed_means = []
            fold_means = []
            for fold in folds:
                fold_seed_means.extend(fold_delta[fold].mean(axis=1).tolist())
                fold_means.append(float(fold_delta[fold].mean()))
                for shoe in sorted(set(fold_episode_shoes[fold].tolist())):
                    keep = fold_episode_shoes[fold] == shoe
                    value = float(fold_delta[fold][:, keep].mean())
                    object_means.append(value)
                    per_object[str(shoe)] = value
            object_means_array = np.asarray(object_means, dtype=np.float64)
            fold_seed_means_array = np.asarray(fold_seed_means, dtype=np.float64)
            output["comparisons"][comparison_name][metric] = {
                "mean": float(episode_means.mean()),
                "object_cluster_bootstrap_95": paired_cluster_interval(
                    object_means_array,
                    int(args.bootstrap_samples),
                    generator,
                ),
                "episode_cluster_bootstrap_95": paired_cluster_interval(
                    episode_means,
                    int(args.bootstrap_samples),
                    generator,
                ),
                "fold_wins": int(np.sum(np.asarray(fold_means) < 0.0)),
                "folds": len(folds),
                "fold_seed_wins": int(np.sum(fold_seed_means_array < 0.0)),
                "fold_seed_pairs": int(len(fold_seed_means_array)),
                "object_wins": int(np.sum(object_means_array < 0.0)),
                "objects": int(len(object_means_array)),
                "episode_wins": int(np.sum(episode_means < 0.0)),
                "episodes": int(len(episode_means)),
                "per_object_mean": per_object,
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output))


if __name__ == "__main__":
    main()
