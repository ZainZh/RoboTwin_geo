#!/usr/bin/env python3
"""Train an ID-free geometry-to-goal head on collected demonstrations.

Inputs are actual ``object_pointcloud/{A}`` and ``object_pointcloud/{B}``.
Simulator functional metadata supplies only the offline target
``goal_T_A_from_B`` and optional object IDs are used only for data splitting
and reporting.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from data_path_utils import raw_task_data_dir
from ndf_goal_regressor import (
    FrozenNdfFeatureConfig,
    FrozenNdfGeometryEncoder,
    GoalRelationRegressor,
    goal_matrix_to_target,
    rotation_error_deg,
    rotation_loss,
)
from object_pointcloud_utils import load_hdf5, load_scene_info, parse_asset_spec


REPO_ROOT = Path(__file__).resolve().parents[3]
RAMP_FUNCTIONAL_MATRIX = np.asarray(
    [
        [0.0, -1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class GeometrySample:
    episode: int
    frame: int
    evaluator_object_id: int
    pointcloud_a: np.ndarray
    pointcloud_b: np.ndarray
    goal_t_a_from_b: np.ndarray
    supervision_source: str


def _parse_ids(value: str) -> set[int]:
    return {int(item.strip()) for item in str(value).split(",") if item.strip()}


def _episode_index(path: Path) -> int:
    match = re.fullmatch(r"episode(\d+)\.hdf5", path.name)
    if match is None:
        raise ValueError(f"unexpected episode filename {path.name!r}")
    return int(match.group(1))


def _uniform_indices(indices: np.ndarray, count: int) -> list[int]:
    values = np.asarray(indices, dtype=np.int64).reshape(-1)
    if len(values) <= int(count):
        return [int(value) for value in values]
    selected = np.linspace(0, len(values) - 1, int(count))
    return [int(values[int(round(value))]) for value in selected]


def _select_frames(episode: dict, frames_per_episode: int) -> list[int]:
    length = int(np.asarray(episode["object_pointcloud"]["{A}"]).shape[0])
    task_state = episode.get("task_state", {})
    phase = task_state.get("relation_phase") if isinstance(task_state, dict) else None
    if phase is not None:
        active = np.flatnonzero(np.asarray(phase).reshape(-1) > 0.5)
        if len(active) > 0:
            return _uniform_indices(active, frames_per_episode)
    return _uniform_indices(np.arange(length, dtype=np.int64), frames_per_episode)


def _scaled_functional_matrix(model_data: dict) -> np.ndarray:
    functional = model_data.get("functional_matrix")
    if not isinstance(functional, list) or not functional:
        raise ValueError("asset model_data has no functional_matrix")
    matrix = np.asarray(functional[0], dtype=np.float64).reshape(4, 4).copy()
    scale = np.asarray(model_data.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
    if scale.ndim == 0:
        scale = np.full((3,), float(scale), dtype=np.float64)
    matrix[:3, 3] *= scale.reshape(3)
    return matrix


def _legacy_oracle_from_scene(
    scene_info: dict,
    episode_index: int,
) -> tuple[np.ndarray, int]:
    episode_info = scene_info.get(f"episode_{episode_index}", {})
    asset_spec = episode_info.get("info", {}).get("{A}")
    model_name, model_id = parse_asset_spec(asset_spec)
    if model_name is None or model_id is None:
        raise ValueError(
            f"episode {episode_index} has no parseable {{A}} asset in scene_info"
        )
    model_data_path = (
        REPO_ROOT / "assets" / "objects" / model_name / f"model_data{model_id}.json"
    )
    model_data = json.loads(model_data_path.read_text(encoding="utf-8"))
    shoe_functional = _scaled_functional_matrix(model_data)
    return shoe_functional @ np.linalg.inv(RAMP_FUNCTIONAL_MATRIX), int(model_id)


def _supervision_for_frame(
    episode: dict,
    *,
    frame_index: int,
    episode_index: int,
    scene_info: dict,
    allow_legacy_asset_supervision: bool = False,
) -> tuple[np.ndarray, int, str]:
    task_state = episode.get("task_state")
    if isinstance(task_state, dict) and "goal_T_A_from_B_oracle" in task_state:
        goals = np.asarray(task_state["goal_T_A_from_B_oracle"])
        goal = np.asarray(goals[int(frame_index)], dtype=np.float64).reshape(4, 4)
        object_id = -1
        if "shoe_id" in task_state:
            object_id = int(
                np.asarray(task_state["shoe_id"][int(frame_index)]).reshape(-1)[0]
            )
        return goal, object_id, "task_state_oracle"
    if not allow_legacy_asset_supervision:
        raise RuntimeError(
            "dataset is missing task_state/goal_T_A_from_B_oracle; recollect with "
            "the SE(3)-relation task config or explicitly pass "
            "--allow_legacy_asset_supervision for an offline compatibility run"
        )

    goal, object_id = _legacy_oracle_from_scene(scene_info, episode_index)
    return goal, object_id, "asset_functional_oracle"


def load_samples(args) -> list[GeometrySample]:
    load_dir = raw_task_data_dir(args.task_name, args.task_config)
    scene_info = load_scene_info(str(load_dir / "scene_info.json"))
    episode_paths = sorted(
        (load_dir / "data").glob("episode*.hdf5"), key=_episode_index
    )
    episode_paths = [
        path for path in episode_paths if _episode_index(path) < int(args.expert_data_num)
    ]
    if not episode_paths:
        raise FileNotFoundError(f"no HDF5 episodes found under {load_dir / 'data'}")

    samples: list[GeometrySample] = []
    for path in episode_paths:
        episode_index = _episode_index(path)
        episode = load_hdf5(str(path))
        object_pointcloud = episode.get("object_pointcloud", {})
        if "{A}" not in object_pointcloud or "{B}" not in object_pointcloud:
            raise RuntimeError(
                f"{path} is missing separated object_pointcloud/{{A}} or {{B}}"
            )
        for frame_index in _select_frames(episode, int(args.frames_per_episode)):
            goal, object_id, source = _supervision_for_frame(
                episode,
                frame_index=frame_index,
                episode_index=episode_index,
                scene_info=scene_info,
                allow_legacy_asset_supervision=bool(args.allow_legacy_asset_supervision),
            )
            samples.append(
                GeometrySample(
                    episode=episode_index,
                    frame=frame_index,
                    evaluator_object_id=object_id,
                    pointcloud_a=np.asarray(
                        object_pointcloud["{A}"][frame_index], dtype=np.float32
                    ),
                    pointcloud_b=np.asarray(
                        object_pointcloud["{B}"][frame_index], dtype=np.float32
                    ),
                    goal_t_a_from_b=goal,
                    supervision_source=source,
                )
            )
    return samples


def split_samples(
    samples: list[GeometrySample],
    *,
    validation_object_ids: set[int],
    validation_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    if validation_object_ids:
        validation = [
            index
            for index, sample in enumerate(samples)
            if sample.evaluator_object_id in validation_object_ids
        ]
        training = [
            index
            for index, sample in enumerate(samples)
            if sample.evaluator_object_id not in validation_object_ids
        ]
    else:
        episodes = sorted({sample.episode for sample in samples})
        rng = np.random.default_rng(int(seed))
        rng.shuffle(episodes)
        validation_count = max(1, int(round(len(episodes) * validation_fraction)))
        validation_episodes = set(episodes[:validation_count])
        validation = [
            index
            for index, sample in enumerate(samples)
            if sample.episode in validation_episodes
        ]
        training = [
            index
            for index, sample in enumerate(samples)
            if sample.episode not in validation_episodes
        ]
    if not training or not validation:
        raise RuntimeError(
            f"empty train/validation split: train={len(training)}, val={len(validation)}"
        )
    return training, validation


def precompute_features(
    encoder: FrozenNdfGeometryEncoder,
    samples: list[GeometrySample],
    *,
    batch_size: int,
) -> torch.Tensor:
    chunks = []
    for start in range(0, len(samples), int(batch_size)):
        batch = samples[start : start + int(batch_size)]
        chunks.append(
            encoder.encode_batch(
                [sample.pointcloud_a for sample in batch],
                [sample.pointcloud_b for sample in batch],
            ).cpu()
        )
        print(
            f"extracting frozen NDF features: {min(start + len(batch), len(samples))}"
            f" / {len(samples)}",
            end="\r",
        )
    print()
    return torch.cat(chunks, dim=0).float()


def _metrics(
    model: GoalRelationRegressor,
    features: torch.Tensor,
    target: torch.Tensor,
    target_rotation: torch.Tensor,
) -> dict:
    model.eval()
    with torch.no_grad():
        prediction = model(features)
        translation = torch.linalg.vector_norm(
            prediction[:, :3] - target[:, :3], dim=-1
        )
        rotation = rotation_error_deg(prediction[:, 3:9], target_rotation)
    return {
        "translation_mean_m": float(translation.mean().cpu()),
        "translation_median_m": float(translation.median().cpu()),
        "translation_p90_m": float(
            torch.quantile(translation, 0.9).cpu()
        ),
        "rotation_mean_deg": float(rotation.mean().cpu()),
        "rotation_median_deg": float(rotation.median().cpu()),
        "rotation_p90_deg": float(torch.quantile(rotation, 0.9).cpu()),
        "geometry_success_rate": float(
            ((translation <= 0.05) & (rotation <= 20.0)).float().mean().cpu()
        ),
    }


def _constant_goal_metrics(
    training_target: torch.Tensor,
    target: torch.Tensor,
    target_rotation: torch.Tensor,
) -> dict:
    """Point-cloud-free mean-target ablation for the same validation split."""

    with torch.no_grad():
        constant = training_target.mean(dim=0, keepdim=True)
        prediction = constant.expand(len(target), -1)
        translation = torch.linalg.vector_norm(
            prediction[:, :3] - target[:, :3], dim=-1
        )
        rotation = rotation_error_deg(prediction[:, 3:9], target_rotation)
    return {
        "translation_mean_m": float(translation.mean().cpu()),
        "translation_median_m": float(translation.median().cpu()),
        "translation_p90_m": float(torch.quantile(translation, 0.9).cpu()),
        "rotation_mean_deg": float(rotation.mean().cpu()),
        "rotation_median_deg": float(rotation.median().cpu()),
        "rotation_p90_deg": float(torch.quantile(rotation, 0.9).cpu()),
        "geometry_success_rate": float(
            ((translation <= 0.05) & (rotation <= 20.0)).float().mean().cpu()
        ),
    }


def _portable_path(path: Path, relative_to: Path) -> str:
    try:
        return str(path.resolve().relative_to(relative_to.resolve()))
    except ValueError:
        return str(path.resolve())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train frozen-NDF point-cloud to T_A_from_B regression."
    )
    parser.add_argument("task_name")
    parser.add_argument("task_config")
    parser.add_argument("expert_data_num", type=int)
    parser.add_argument("--ndf_checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--spec_output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames_per_episode", type=int, default=8)
    parser.add_argument("--target_num_points", type=int, default=1024)
    parser.add_argument("--feature_batch_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--translation_weight", type=float, default=500.0)
    parser.add_argument("--hidden_dims", default="256,128")
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--validation_object_ids", default="")
    parser.add_argument("--validation_fraction", type=float, default=0.2)
    parser.add_argument("--allow_legacy_asset_supervision", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    requested = torch.device(args.device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        requested = torch.device("cpu")

    samples = load_samples(args)
    validation_object_ids = _parse_ids(args.validation_object_ids)
    training_indices, validation_indices = split_samples(
        samples,
        validation_object_ids=validation_object_ids,
        validation_fraction=float(args.validation_fraction),
        seed=int(args.seed),
    )
    feature_config = FrozenNdfFeatureConfig(
        latent_dim=256,
        target_num_points=int(args.target_num_points),
    )
    encoder = FrozenNdfGeometryEncoder(
        checkpoint=str(Path(args.ndf_checkpoint).expanduser().resolve()),
        device=requested,
        config=feature_config,
    )
    features = precompute_features(
        encoder,
        samples,
        batch_size=int(args.feature_batch_size),
    )
    target_numpy = np.stack(
        [goal_matrix_to_target(sample.goal_t_a_from_b) for sample in samples]
    )
    targets = torch.from_numpy(target_numpy).float()
    rotations = torch.from_numpy(
        np.stack(
            [
                np.asarray(sample.goal_t_a_from_b, dtype=np.float32)[:3, :3]
                for sample in samples
            ]
        )
    )

    train_index_t = torch.as_tensor(training_indices, dtype=torch.long)
    val_index_t = torch.as_tensor(validation_indices, dtype=torch.long)
    feature_mean = features[train_index_t].mean(dim=0)
    feature_std = features[train_index_t].std(dim=0).clamp_min(1e-5)
    features = (features - feature_mean) / feature_std
    features = features.to(requested)
    targets = targets.to(requested)
    rotations = rotations.to(requested)
    train_index_t = train_index_t.to(requested)
    val_index_t = val_index_t.to(requested)

    hidden_dims = tuple(
        int(value.strip()) for value in str(args.hidden_dims).split(",") if value.strip()
    )
    model = GoalRelationRegressor(
        input_dim=feature_config.output_dim,
        hidden_dims=hidden_dims,
        dropout=float(args.dropout),
    ).to(requested)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    train_dataset = TensorDataset(
        features[train_index_t],
        targets[train_index_t],
        rotations[train_index_t],
    )
    generator = torch.Generator()
    generator.manual_seed(int(args.seed))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        generator=generator,
    )

    best_state = None
    best_epoch = -1
    best_score = float("inf")
    stale_epochs = 0
    for epoch in range(int(args.epochs)):
        model.train()
        for feature_batch, target_batch, rotation_batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(feature_batch)
            translation_term = torch.mean(
                (prediction[:, :3] - target_batch[:, :3]) ** 2
            )
            rotation_term = rotation_loss(prediction[:, 3:9], rotation_batch)
            loss = float(args.translation_weight) * translation_term + rotation_term
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_prediction = model(features[val_index_t])
            validation_translation = torch.linalg.vector_norm(
                validation_prediction[:, :3] - targets[val_index_t, :3], dim=-1
            ).mean()
            validation_rotation = rotation_error_deg(
                validation_prediction[:, 3:9], rotations[val_index_t]
            ).mean()
            score = float(
                validation_translation.cpu() / 0.05
                + validation_rotation.cpu() / 20.0
            )
        if score < best_score - 1e-6:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1
        if epoch == 0 or (epoch + 1) % 25 == 0:
            print(
                f"epoch={epoch + 1} val_t={float(validation_translation):.4f}m "
                f"val_r={float(validation_rotation):.2f}deg"
            )
        if stale_epochs >= int(args.patience):
            break

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    train_metrics = _metrics(
        model,
        features[train_index_t],
        targets[train_index_t],
        rotations[train_index_t],
    )
    validation_metrics = _metrics(
        model,
        features[val_index_t],
        targets[val_index_t],
        rotations[val_index_t],
    )

    constant_goal_validation_metrics = _constant_goal_metrics(
        targets[train_index_t],
        targets[val_index_t],
        rotations[val_index_t],
    )

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": 1,
        "model_type": "ndf_goal_regressor",
        "model_state_dict": best_state,
        "input_dim": feature_config.output_dim,
        "latent_dim": feature_config.latent_dim,
        "target_num_points": feature_config.target_num_points,
        "hidden_dims": list(hidden_dims),
        "dropout": float(args.dropout),
        "feature_mean": feature_mean.cpu(),
        "feature_std": feature_std.cpu(),
        "best_epoch": int(best_epoch),
        "training_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "constant_goal_validation_metrics": constant_goal_validation_metrics,
        "training_episodes": sorted({samples[index].episode for index in training_indices}),
        "validation_episodes": sorted(
            {samples[index].episode for index in validation_indices}
        ),
        "validation_object_ids_evaluator_only": sorted(validation_object_ids),
        "supervision_sources": sorted(
            {sample.supervision_source for sample in samples}
        ),
    }
    torch.save(checkpoint, output)

    spec_output = Path(args.spec_output).expanduser().resolve()
    spec_output.parent.mkdir(parents=True, exist_ok=True)
    spec = {
        "schema_version": 1,
        "type": "ndf_goal_regressor",
        "regressor_checkpoint": _portable_path(output, spec_output.parent),
        "ndf_checkpoint": _portable_path(
            Path(args.ndf_checkpoint).expanduser().resolve(), spec_output.parent
        ),
        "device": str(args.device),
    }
    spec_output.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    report = {
        "sample_count": len(samples),
        "training_sample_count": len(training_indices),
        "validation_sample_count": len(validation_indices),
        "best_epoch": best_epoch,
        "training": train_metrics,
        "validation": validation_metrics,
        "constant_goal_validation": constant_goal_validation_metrics,
        "checkpoint": str(output),
        "spec": str(spec_output),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
