#!/usr/bin/env python3
"""Train one task-aligned two-vector NDF head for a complete object frame.

The legacy alpha head predicts object Z.  A beta coefficient head shares the
same equivariant VN basis and predicts object Y.  Gram--Schmidt yields the
right-handed frame ``R=[X,Y,Z]``.  Simulator pose is a supervised label only;
the model input is the current A camera point cloud.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .functional_action_frame import pose9_rotation
from .interaction_flow_tokens import rotation_geodesic_loss
from .ndf_adapter import NdfFunctionalFrameAdapter, NdfPointwiseAdapter
from .train_ndf_functional_vector_head import derangement, normalize_ndf_clouds
from .train_representation_token_grasp import seed_everything
from .train_task_flow_benchmark import FOLDS


CONDITIONS = ("correct_frame", "shuffled_frame")


def select_training_episode_indices(
    train_indices: np.ndarray,
    shoe_ids: np.ndarray,
    episode_ids: np.ndarray,
    train_object_ids: tuple[int, ...] | list[int] | np.ndarray,
    *,
    episodes_per_object: int | None,
    fold: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """Select complete training episodes without changing validation or test.

    Episode identities are first validated globally to prevent an episode from
    spanning object identities. The supplied training split must contain every
    frame of every episode it references; subsampling then keeps every frame of
    each selected episode.
    """
    indices = np.asarray(train_indices, dtype=np.int64).reshape(-1)
    shoes = np.asarray(shoe_ids, dtype=np.int64).reshape(-1)
    episodes = np.asarray(episode_ids, dtype=np.int64).reshape(-1)
    if len(shoes) != len(episodes):
        raise ValueError("shoe_ids and episode_ids must have identical row counts")
    if len(indices) == 0 or len(np.unique(indices)) != len(indices):
        raise ValueError("train_indices must be nonempty and unique")
    if int(indices.min()) < 0 or int(indices.max()) >= len(shoes):
        raise ValueError("train_indices contain an out-of-range row")
    expected_objects = {int(value) for value in train_object_ids}
    actual_objects = {int(value) for value in np.unique(shoes[indices])}
    if actual_objects != expected_objects:
        raise ValueError(
            "training object coverage mismatch: "
            f"expected {sorted(expected_objects)}, found {sorted(actual_objects)}"
        )
    if episodes_per_object is not None and int(episodes_per_object) <= 0:
        raise ValueError("train-episodes-per-object must be positive")

    episode_to_object: dict[int, int] = {}
    for episode in np.unique(episodes):
        members = np.unique(shoes[episodes == episode])
        if len(members) != 1:
            raise ValueError(
                f"episode {int(episode)} spans object identities {members.tolist()}"
            )
        episode_to_object[int(episode)] = int(members[0])

    train_episodes = np.unique(episodes[indices])
    complete_train_rows = np.flatnonzero(np.isin(episodes, train_episodes))
    if not np.array_equal(np.sort(indices), complete_train_rows):
        raise ValueError(
            "training split contains a partial episode; complete episodes are required"
        )
    available: dict[int, list[int]] = {}
    for object_id in sorted(expected_objects):
        available[object_id] = sorted(
            int(value)
            for value in train_episodes
            if episode_to_object[int(value)] == object_id
        )
        if not available[object_id]:
            raise ValueError(f"training object {object_id} has no complete episode")

    selected: dict[int, list[int]] = {}
    selection_seeds: dict[int, int | None] = {}
    for object_id, candidates in available.items():
        if episodes_per_object is None:
            selected[object_id] = list(candidates)
            selection_seeds[object_id] = None
            continue
        count = int(episodes_per_object)
        if len(candidates) < count:
            raise ValueError(
                f"training object {object_id} has {len(candidates)} episodes, "
                f"cannot select {count}"
            )
        selection_seed = (
            157_001 + int(fold) * 10_007 + int(seed) * 997 + int(object_id) * 53
        )
        generator = np.random.default_rng(selection_seed)
        selected[object_id] = sorted(
            int(value)
            for value in generator.choice(candidates, size=count, replace=False)
        )
        selection_seeds[object_id] = int(selection_seed)

    selected_episodes = sorted(
        episode for values in selected.values() for episode in values
    )
    keep = np.isin(episodes[indices], np.asarray(selected_episodes, dtype=np.int64))
    selected_indices = indices[keep]
    complete_selected_rows = np.flatnonzero(
        np.isin(episodes, np.asarray(selected_episodes, dtype=np.int64))
    )
    if not np.array_equal(np.sort(selected_indices), complete_selected_rows):
        raise RuntimeError("episode selection dropped or leaked one or more frames")
    selected_objects = {int(value) for value in np.unique(shoes[selected_indices])}
    if selected_objects != expected_objects:
        raise RuntimeError("episode selection dropped a training object")

    diagnostic = {
        "mode": (
            "all_complete_episodes"
            if episodes_per_object is None
            else "deterministic_k_complete_episodes_per_object"
        ),
        "fold": int(fold),
        "seed": int(seed),
        "requested_episodes_per_object": (
            None if episodes_per_object is None else int(episodes_per_object)
        ),
        "train_samples_before": int(len(indices)),
        "train_samples_after": int(len(selected_indices)),
        "train_episodes_before": int(len(train_episodes)),
        "train_episodes_after": int(len(selected_episodes)),
        "available_episodes_per_object": {
            str(key): value for key, value in available.items()
        },
        "selected_episodes_per_object": {
            str(key): value for key, value in selected.items()
        },
        "selection_seed_per_object": {
            str(key): value for key, value in selection_seeds.items()
        },
        "selected_samples_per_object": {
            str(object_id): int(np.sum(shoes[selected_indices] == object_id))
            for object_id in sorted(expected_objects)
        },
    }
    return selected_indices, diagnostic


def frame_loss(
    vectors: torch.Tensor,
    target_rotation: torch.Tensor,
    *,
    axis_weight: float,
    orthogonality_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    predicted_rotation = NdfFunctionalFrameAdapter.frame_from_vectors(vectors)
    geodesic = rotation_geodesic_loss(predicted_rotation, target_rotation)
    target_z = target_rotation[..., :, 2]
    target_y = target_rotation[..., :, 1]
    predicted = F.normalize(vectors, dim=-1, eps=1e-6)
    z_axis = (1.0 - torch.sum(predicted[..., 0, :] * target_z[:, None], dim=-1)).mean()
    y_axis = (1.0 - torch.sum(predicted[..., 1, :] * target_y[:, None], dim=-1)).mean()
    mean_z = F.normalize(vectors[..., 0, :].mean(dim=1), dim=-1, eps=1e-6)
    mean_y = F.normalize(vectors[..., 1, :].mean(dim=1), dim=-1, eps=1e-6)
    orthogonality = torch.abs(torch.sum(mean_z * mean_y, dim=-1)).mean()
    total = (
        geodesic
        + float(axis_weight) * 0.5 * (z_axis + y_axis)
        + float(orthogonality_weight) * orthogonality
    )
    return total, {
        "geodesic_rad": geodesic,
        "z_axis": z_axis,
        "y_axis": y_axis,
        "orthogonality": orthogonality,
    }


def load_frame_model(
    checkpoint: Path,
    device: torch.device,
    *,
    trainable_scope: str,
    promote_alpha_to_beta: bool = False,
    initialization: str = "pretrained",
) -> tuple[nn.Module, list[nn.Parameter]]:
    model = NdfPointwiseAdapter(
        checkpoint,
        device=str(device),
        include_vector_direction=True,
    ).model
    if model.decoder.fc_vec_beta is None:
        model.decoder.fc_vec_beta = nn.Linear(
            model.decoder.fc_vec_alpha.in_features,
            model.decoder.fc_vec_alpha.out_features,
        ).to(device)
        if promote_alpha_to_beta:
            # Reverse curriculum: the input checkpoint's alpha head has first
            # learned the harder signed Y/yaw axis.  Preserve it as beta (our
            # deployed Y head), then relearn alpha as the easier object Z axis.
            model.decoder.fc_vec_beta.load_state_dict(
                model.decoder.fc_vec_alpha.state_dict()
            )
            nn.init.xavier_uniform_(model.decoder.fc_vec_alpha.weight)
            nn.init.zeros_(model.decoder.fc_vec_alpha.bias)
        else:
            nn.init.xavier_uniform_(model.decoder.fc_vec_beta.weight)
            nn.init.zeros_(model.decoder.fc_vec_beta.bias)
    elif promote_alpha_to_beta:
        raise ValueError(
            "--promote-alpha-to-beta expects a legacy one-axis checkpoint"
        )
    if initialization not in {"pretrained", "pretrained_reinit_heads", "random"}:
        raise ValueError(f"unknown initialization {initialization!r}")
    if initialization == "pretrained_reinit_heads":
        if promote_alpha_to_beta:
            raise ValueError(
                "pretrained_reinit_heads is incompatible with --promote-alpha-to-beta"
            )
        # Preserve the equivariant point encoder and vector basis, but remove
        # the legacy single-axis task bias before learning a complete SO(3)
        # functional frame. Reset both deployed axes to give them an identical
        # initialization contract.
        for head in (model.decoder.fc_vec_alpha, model.decoder.fc_vec_beta):
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)
    if initialization == "random":
        if trainable_scope != "full":
            raise ValueError("random initialization requires --trainable-scope full")
        if promote_alpha_to_beta:
            raise ValueError(
                "random initialization is incompatible with --promote-alpha-to-beta"
            )
        # Load the checkpoint only to recover the exact published NDF
        # architecture, then remove its learned weights.  Calling each
        # parameter-owning module's standard initializer gives a capacity-
        # matched control without introducing another model implementation.
        reset_modules = 0
        for module in model.modules():
            reset = getattr(module, "reset_parameters", None)
            if callable(reset):
                reset()
                reset_modules += 1
        if reset_modules == 0:
            raise RuntimeError("NDF model exposes no reset_parameters modules")
    if trainable_scope not in {"alpha", "beta", "heads", "decoder", "full"}:
        raise ValueError(f"unknown trainable scope {trainable_scope!r}")
    for parameter in model.parameters():
        parameter.requires_grad_(trainable_scope == "full")
    if trainable_scope == "decoder":
        for parameter in model.decoder.parameters():
            parameter.requires_grad_(True)
    elif trainable_scope == "alpha":
        for parameter in model.decoder.fc_vec_alpha.parameters():
            parameter.requires_grad_(True)
    elif trainable_scope == "beta":
        for parameter in model.decoder.fc_vec_beta.parameters():
            parameter.requires_grad_(True)
    elif trainable_scope == "heads":
        for module in (
            model.decoder.vector_basis,
            model.decoder.fc_vec_alpha,
            model.decoder.fc_vec_beta,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return model, trainable


def predict_frames(
    model: nn.Module, clouds: torch.Tensor, batch_size: int
) -> torch.Tensor:
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(clouds), int(batch_size)):
            points = clouds[start : start + int(batch_size)]
            latent = model.extract_latent({"point_cloud": points})
            _, vectors = model.forward_latent(
                latent, points, return_vector_features=True
            )
            output.append(NdfFunctionalFrameAdapter.frame_from_vectors(vectors))
    return torch.cat(output, dim=0)


def frame_metrics(
    model: nn.Module,
    clouds: torch.Tensor,
    target_rotation: torch.Tensor,
    indices: np.ndarray,
    batch_size: int,
) -> dict:
    subset = torch.as_tensor(indices, dtype=torch.long, device=clouds.device)
    predicted = predict_frames(model, clouds[subset], batch_size)
    target = target_rotation[subset]
    relative = predicted @ target.transpose(-1, -2)
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
    angle = torch.rad2deg(torch.acos(cosine)).detach().cpu().numpy()
    return {
        "samples": int(len(indices)),
        "mean_deg": float(angle.mean()),
        "median_deg": float(np.median(angle)),
        "p90_deg": float(np.percentile(angle, 90)),
        "max_deg": float(angle.max()),
        "within_15deg": float(np.mean(angle <= 15.0)),
        "within_30deg": float(np.mean(angle <= 30.0)),
        "flip_over_90deg": float(np.mean(angle >= 90.0)),
    }


@torch.no_grad()
def batched_frame_loss(
    model: nn.Module,
    clouds: torch.Tensor,
    rotations: torch.Tensor,
    indices: np.ndarray,
    batch_size: int,
    *,
    axis_weight: float,
    orthogonality_weight: float,
) -> tuple[float, dict[str, float]]:
    """Evaluate validation loss without materializing a whole split in NDF."""
    model.eval()
    totals = {
        key: 0.0
        for key in ("loss", "geodesic_rad", "z_axis", "y_axis", "orthogonality")
    }
    samples = 0
    for start in range(0, len(indices), int(batch_size)):
        batch_indices = torch.as_tensor(
            indices[start : start + int(batch_size)],
            dtype=torch.long,
            device=clouds.device,
        )
        points = clouds[batch_indices]
        latent = model.extract_latent({"point_cloud": points})
        _, vectors = model.forward_latent(
            latent, points, return_vector_features=True
        )
        loss, components = frame_loss(
            vectors,
            rotations[batch_indices],
            axis_weight=axis_weight,
            orthogonality_weight=orthogonality_weight,
        )
        count = len(batch_indices)
        totals["loss"] += float(loss) * count
        for key, value in components.items():
            totals[key] += float(value) * count
        samples += count
    denominator = max(samples, 1)
    return (
        totals["loss"] / denominator,
        {
            key: totals[key] / denominator
            for key in ("geodesic_rad", "z_axis", "y_axis", "orthogonality")
        },
    )


def run_condition(
    *,
    checkpoint: Path,
    clouds: torch.Tensor,
    rotations: torch.Tensor,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
    condition: str,
    fold: int,
    seed: int,
    args,
    device: torch.device,
) -> tuple[dict, nn.Module]:
    seed_everything(131_003 + int(fold) * 10_007 + int(seed) * 997)
    model, trainable = load_frame_model(
        checkpoint,
        device,
        trainable_scope=args.trainable_scope,
        promote_alpha_to_beta=args.promote_alpha_to_beta,
        initialization=args.initialization,
    )
    train_targets = rotations[
        torch.as_tensor(train_indices, dtype=torch.long, device=device)
    ].clone()
    if condition == "shuffled_frame":
        permutation = derangement(
            len(train_indices), 137_011 + int(fold) * 101 + int(seed)
        )
        train_targets = train_targets[
            torch.as_tensor(permutation, dtype=torch.long, device=device)
        ]
    dataset = TensorDataset(
        clouds[torch.as_tensor(train_indices, dtype=torch.long, device=device)],
        train_targets,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(139_021 + int(seed)),
    )
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_validation = float("inf")
    best_epoch = -1
    best_state = None
    stale = 0
    history = []
    for epoch in range(args.epochs):
        model.eval()
        totals = {key: 0.0 for key in ("loss", "geodesic_rad", "z_axis", "y_axis", "orthogonality")}
        samples = 0
        for points, target in loader:
            latent = model.extract_latent({"point_cloud": points})
            _, vectors = model.forward_latent(
                latent, points, return_vector_features=True
            )
            loss, components = frame_loss(
                vectors,
                target,
                axis_weight=args.axis_loss_weight,
                orthogonality_weight=args.orthogonality_loss_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
            count = len(points)
            totals["loss"] += float(loss.detach()) * count
            for key, value in components.items():
                totals[key] += float(value.detach()) * count
            samples += count
        validation_loss, validation_components = batched_frame_loss(
            model,
            clouds,
            rotations,
            validation_indices,
            args.batch_size,
            axis_weight=args.axis_loss_weight,
            orthogonality_weight=args.orthogonality_loss_weight,
        )
        record = {
            "epoch": int(epoch),
            "train": {key: value / max(samples, 1) for key, value in totals.items()},
            "validation_loss": validation_loss,
            "validation": validation_components,
        }
        history.append(record)
        if validation_loss < best_validation - args.minimum_improvement:
            best_validation = validation_loss
            best_epoch = int(epoch)
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch == 0 or epoch % 25 == 0:
            print(json.dumps({"condition": condition, **record}), flush=True)
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("full-frame training produced no checkpoint")
    model.load_state_dict(best_state)
    return {
        "condition": condition,
        "fold": int(fold),
        "seed": int(seed),
        "trainable_scope": str(args.trainable_scope),
        "initialization": str(args.initialization),
        "trainable_parameters": int(sum(value.numel() for value in trainable)),
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_validation),
        "train": frame_metrics(
            model, clouds, rotations, train_indices, args.batch_size
        ),
        "validation": frame_metrics(
            model, clouds, rotations, validation_indices, args.batch_size
        ),
        "test": frame_metrics(
            model, clouds, rotations, test_indices, args.batch_size
        ),
        "history": history,
    }, model


def build_parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", type=Path, required=True)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--folds", nargs="+", type=int, default=[0])
    result.add_argument("--seeds", nargs="+", type=int, default=[0])
    result.add_argument(
        "--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS)
    )
    result.add_argument("--epochs", type=int, default=300)
    result.add_argument("--patience", type=int, default=50)
    result.add_argument("--batch-size", type=int, default=8)
    result.add_argument("--learning-rate", type=float, default=3e-4)
    result.add_argument("--weight-decay", type=float, default=1e-4)
    result.add_argument("--minimum-improvement", type=float, default=1e-5)
    result.add_argument("--axis-loss-weight", type=float, default=0.5)
    result.add_argument("--orthogonality-loss-weight", type=float, default=0.05)
    result.add_argument(
        "--trainable-scope",
        choices=("alpha", "beta", "heads", "decoder", "full"),
        default="full",
    )
    result.add_argument(
        "--promote-alpha-to-beta",
        action="store_true",
        help=(
            "Treat a legacy checkpoint alpha as the pretrained object-Y head: "
            "copy it into beta, reinitialize alpha, and learn object Z."
        ),
    )
    result.add_argument(
        "--initialization",
        choices=("pretrained", "pretrained_reinit_heads", "random"),
        default="pretrained",
        help=(
            "Use all checkpoint weights; retain its equivariant backbone while "
            "resetting task-specific vector heads; or reset the same architecture "
            "for a capacity-matched ablation. Random requires full scope."
        ),
    )
    result.add_argument(
        "--train-episodes-per-object",
        type=int,
        default=None,
        metavar="K",
        help=(
            "Deterministically retain K complete episodes for every training "
            "shoe, independently for each fold/seed. Default: use all episodes."
        ),
    )
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--cuda-memory-fraction", type=float, default=1.0)
    return result


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.dataset, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    points = np.asarray(payload["points_a"][..., :3], dtype=np.float32)
    rotations_numpy = pose9_rotation(payload["current_object_pose9"])
    shoe_ids = np.asarray(payload["shoe_id"], dtype=np.int64)
    episode_ids = np.asarray(payload["episode_id"], dtype=np.int64)
    if args.train_episodes_per_object is not None and args.train_episodes_per_object <= 0:
        raise ValueError("train-episodes-per-object must be positive")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            args.cuda_memory_fraction, device
        )
    clouds = torch.from_numpy(normalize_ndf_clouds(points)).to(device)
    rotations = torch.from_numpy(rotations_numpy).to(device)
    public_results = []
    for fold in args.folds:
        split = {
            name: np.flatnonzero(np.isin(shoe_ids, ids))
            for name, ids in FOLDS[int(fold)].items()
        }
        for seed in args.seeds:
            selected_train_indices, episode_selection = (
                select_training_episode_indices(
                    split["train"],
                    shoe_ids,
                    episode_ids,
                    FOLDS[int(fold)]["train"],
                    episodes_per_object=args.train_episodes_per_object,
                    fold=int(fold),
                    seed=int(seed),
                )
            )
            for condition in args.conditions:
                result, model = run_condition(
                    checkpoint=args.checkpoint,
                    clouds=clouds,
                    rotations=rotations,
                    train_indices=selected_train_indices,
                    validation_indices=split["validation"],
                    test_indices=split["test"],
                    condition=condition,
                    fold=int(fold),
                    seed=int(seed),
                    args=args,
                    device=device,
                )
                stem = f"fold{fold}_{condition}_seed{seed}"
                torch.save(model.state_dict(), args.output_dir / f"{stem}_checkpoint.pth")
                frames = predict_frames(model, clouds, args.batch_size)
                np.savez_compressed(
                    args.output_dir / f"{stem}_frames.npz",
                    frames=frames.detach().cpu().numpy().astype(np.float32),
                    episode_id=np.asarray(payload["episode_id"], dtype=np.int64),
                    shoe_id=shoe_ids,
                )
                result["config"] = {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                }
                result["training_episode_selection"] = episode_selection
                result["frame_semantics"] = {
                    "alpha": "signed object Z",
                    "beta": "signed object Y",
                    "rotation": "columns [X,Y,Z], X=Y cross Z",
                }
                (args.output_dir / f"{stem}.json").write_text(
                    json.dumps(result, indent=2) + "\n", encoding="utf-8"
                )
                public_results.append(result)
                print(
                    json.dumps(
                        {
                            "completed": stem,
                            "test": result["test"],
                        }
                    ),
                    flush=True,
                )
    (args.output_dir / "runs.json").write_text(
        json.dumps({"schema_version": 1, "runs": public_results}, indent=2)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
