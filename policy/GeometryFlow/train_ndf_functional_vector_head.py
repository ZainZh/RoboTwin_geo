#!/usr/bin/env python3
"""Task-align only the equivariant vector head of a frozen NDF model.

The occupancy encoder/decoder remains frozen.  The existing vector basis and
query-dependent scalar coefficient head are supervised with the demonstrated
gripper approach axis.  Because the output is still a scalar combination of
equivariant latent vectors, this changes task semantics without discarding the
SO(3) inductive bias.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .ndf_adapter import NdfPointwiseAdapter
from .functional_action_frame import pose9_rotation
from .train_representation_token_grasp import seed_everything
from .train_task_flow_benchmark import FOLDS


CONDITIONS = ("correct_axis", "shuffled_axis")


def axis_dataset(
    payload: dict[str, np.ndarray], axis_source: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select camera clouds and label-only axes for grasp or task alignment."""
    if axis_source == "grasp_approach":
        return (
            np.asarray(payload["points_world"], dtype=np.float32),
            np.asarray(payload["target_transform_world"][:, :3, 0], dtype=np.float32),
            np.asarray(payload["shoe_id_label_only"], dtype=np.int64),
            np.asarray(payload["episode_id"], dtype=np.int64),
        )
    if not axis_source.startswith("current_object_"):
        raise ValueError(f"unknown axis source {axis_source!r}")
    axis_index = {"x": 0, "y": 1, "z": 2}[axis_source.rsplit("_", 1)[-1]]
    rotation = pose9_rotation(payload["current_object_pose9"])
    return (
        np.asarray(payload["points_a"][..., :3], dtype=np.float32),
        np.asarray(rotation[..., :, axis_index], dtype=np.float32),
        np.asarray(payload["shoe_id"], dtype=np.int64),
        np.asarray(payload["episode_id"], dtype=np.int64),
    )


def normalize_ndf_clouds(points: np.ndarray) -> np.ndarray:
    """Match the mean/maximum-extent normalization used by NdfPointwiseAdapter."""
    value = np.asarray(points, dtype=np.float32)
    center = value.mean(axis=1, keepdims=True)
    centered = value - center
    extents = centered.max(axis=1) - centered.min(axis=1)
    scale = np.maximum(extents.max(axis=1, keepdims=True), 1e-6)
    return (centered / scale[:, None, :]).astype(np.float32)


def derangement(count: int, seed: int) -> np.ndarray:
    if int(count) < 2:
        raise ValueError("a shuffled-axis control requires at least two samples")
    generator = np.random.default_rng(int(seed))
    base = np.arange(int(count), dtype=np.int64)
    for _ in range(1_000):
        candidate = generator.permutation(base)
        if np.all(candidate != base):
            return candidate
    return np.roll(base, 1)


def functional_axis_loss(
    vectors: torch.Tensor, target_axis: torch.Tensor
) -> torch.Tensor:
    target = F.normalize(target_axis, dim=-1, eps=1e-6)
    predicted = F.normalize(vectors, dim=-1, eps=1e-6)
    cosine = torch.sum(predicted * target[:, None, :], dim=-1)
    return (1.0 - cosine).mean()


@torch.no_grad()
def predict_vectors(model, clouds: torch.Tensor, batch_size: int) -> torch.Tensor:
    model.eval()
    output = []
    for start in range(0, len(clouds), int(batch_size)):
        points = clouds[start : start + int(batch_size)]
        latent = model.extract_latent({"point_cloud": points})
        _, vectors = model.forward_latent(
            latent, points, return_vector_features=True
        )
        output.append(vectors)
    return torch.cat(output, dim=0)


@torch.no_grad()
def axis_metrics(
    model,
    clouds: torch.Tensor,
    target_axes: torch.Tensor,
    indices: np.ndarray,
    batch_size: int,
) -> dict:
    subset = torch.as_tensor(indices, dtype=torch.long, device=clouds.device)
    vectors = predict_vectors(model, clouds[subset], batch_size)
    mean_vector = F.normalize(vectors.mean(dim=1), dim=-1, eps=1e-6)
    target = F.normalize(target_axes[subset], dim=-1, eps=1e-6)
    cosine = torch.clamp(torch.sum(mean_vector * target, dim=-1), -1.0, 1.0)
    angles = torch.rad2deg(torch.acos(cosine))
    point_cosine = torch.sum(
        F.normalize(vectors, dim=-1, eps=1e-6) * target[:, None, :],
        dim=-1,
    ).mean(dim=1)
    return {
        "episodes": int(len(indices)),
        "angle_deg_mean": float(angles.mean().cpu()),
        "angle_deg_median": float(angles.median().cpu()),
        "angle_deg_max": float(angles.max().cpu()),
        "within_15deg": float((angles <= 15.0).float().mean().cpu()),
        "within_30deg": float((angles <= 30.0).float().mean().cpu()),
        "point_cosine_mean": float(point_cosine.mean().cpu()),
    }


def constant_axis_metrics(
    target_axes: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> dict:
    axis = np.asarray(target_axes[train_indices], dtype=np.float64).mean(axis=0)
    axis /= max(float(np.linalg.norm(axis)), 1e-8)
    targets = np.asarray(target_axes[test_indices], dtype=np.float64)
    targets /= np.maximum(np.linalg.norm(targets, axis=-1, keepdims=True), 1e-8)
    cosine = np.clip(targets @ axis, -1.0, 1.0)
    angles = np.degrees(np.arccos(cosine))
    return {
        "episodes": int(len(test_indices)),
        "angle_deg_mean": float(angles.mean()),
        "angle_deg_median": float(np.median(angles)),
        "angle_deg_max": float(angles.max()),
        "within_15deg": float(np.mean(angles <= 15.0)),
        "within_30deg": float(np.mean(angles <= 30.0)),
        "axis": axis.tolist(),
    }


def load_model(
    checkpoint: Path, device: torch.device, trainable_scope: str = "vector_head"
):
    model = NdfPointwiseAdapter(
        checkpoint,
        device=str(device),
        include_vector_direction=True,
    ).model
    if not model.return_vector_features:
        raise RuntimeError("checkpoint does not contain an NDF vector head")
    if trainable_scope not in {"vector_head", "decoder", "full"}:
        raise ValueError(f"unknown trainable scope {trainable_scope!r}")
    for parameter in model.parameters():
        parameter.requires_grad_(trainable_scope == "full")
    modules = (model.decoder.vector_basis, model.decoder.fc_vec_alpha)
    if any(module is None for module in modules):
        raise RuntimeError("NDF vector modules are unavailable")
    if trainable_scope == "decoder":
        for parameter in model.decoder.parameters():
            parameter.requires_grad_(True)
    else:
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
    return model, [parameter for parameter in model.parameters() if parameter.requires_grad]


def run_condition(
    *,
    checkpoint: Path,
    clouds: torch.Tensor,
    axes: torch.Tensor,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
    condition: str,
    fold: int,
    seed: int,
    args,
    device: torch.device,
) -> tuple[dict, torch.nn.Module]:
    seed_everything(93_001 + int(fold) * 10_007 + int(seed) * 997)
    model, trainable = load_model(
        checkpoint, device, trainable_scope=args.trainable_scope
    )
    original_test = axis_metrics(
        model, clouds, axes, test_indices, args.batch_size
    )
    train_targets = axes[torch.as_tensor(train_indices, device=device)].clone()
    if condition == "shuffled_axis":
        permutation = derangement(
            len(train_indices), 71_011 + int(fold) * 101 + int(seed)
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
        generator=torch.Generator().manual_seed(81_001 + int(seed)),
    )
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_validation = float("inf")
    best_epoch = -1
    best_state = None
    stale = 0
    history = []
    validation_tensor = torch.as_tensor(
        validation_indices, dtype=torch.long, device=device
    )
    for epoch in range(args.epochs):
        model.eval()
        total = samples = 0.0
        for points, target in loader:
            latent = model.extract_latent({"point_cloud": points})
            _, vectors = model.forward_latent(
                latent, points, return_vector_features=True
            )
            loss = functional_axis_loss(vectors, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
            total += float(loss.item()) * len(points)
            samples += len(points)
        with torch.no_grad():
            validation_vectors = predict_vectors(
                model, clouds[validation_tensor], args.batch_size
            )
            validation_loss = float(
                functional_axis_loss(
                    validation_vectors, axes[validation_tensor]
                ).cpu()
            )
        record = {
            "epoch": int(epoch),
            "train_loss": total / max(samples, 1),
            "validation_loss": validation_loss,
        }
        history.append(record)
        if validation_loss < best_validation - args.minimum_improvement:
            best_validation = validation_loss
            best_epoch = epoch
            # Save every trainable module, not just the vector head.  The old
            # behavior silently mixed the best vector head with the final-epoch
            # encoder/decoder when trainable_scope was ``decoder`` or ``full``.
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch == 0 or epoch % 25 == 0:
            print(
                json.dumps({"condition": condition, **record, "best": best_validation}),
                flush=True,
            )
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("functional vector training produced no checkpoint")
    model.load_state_dict(best_state)
    head_state = {
        "vector_basis": copy.deepcopy(model.decoder.vector_basis.state_dict()),
        "fc_vec_alpha": copy.deepcopy(model.decoder.fc_vec_alpha.state_dict()),
    }
    result = {
        "condition": condition,
        "fold": int(fold),
        "seed": int(seed),
        "trainable_scope": str(args.trainable_scope),
        "trainable_parameters": int(sum(value.numel() for value in trainable)),
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_validation),
        "original_test": original_test,
        "adapted_test": axis_metrics(
            model, clouds, axes, test_indices, args.batch_size
        ),
        "history": history,
        "head_state": head_state,
    }
    return result, model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", nargs="+", type=int, default=[0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument(
        "--axis-source",
        choices=(
            "grasp_approach",
            "current_object_x",
            "current_object_y",
            "current_object_z",
        ),
        default="grasp_approach",
        help=(
            "Task-dataset object axes are simulator-derived training labels only; "
            "the NDF input remains the current camera point cloud."
        ),
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--minimum-improvement", type=float, default=1e-5)
    parser.add_argument(
        "--trainable-scope",
        choices=("vector_head", "decoder", "full"),
        default="vector_head",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cuda-memory-fraction", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.dataset, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    points, axes_numpy, shoe_ids, episode_ids = axis_dataset(
        payload, args.axis_source
    )
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            args.cuda_memory_fraction, device
        )
    clouds = torch.from_numpy(normalize_ndf_clouds(points)).to(device)
    axes = torch.from_numpy(axes_numpy).to(device)
    public_results = []
    for fold in args.folds:
        split = {
            name: np.flatnonzero(np.isin(shoe_ids, ids))
            for name, ids in FOLDS[int(fold)].items()
        }
        constant = constant_axis_metrics(
            axes_numpy, split["train"], split["test"]
        )
        for seed in args.seeds:
            for condition in args.conditions:
                result, model = run_condition(
                    checkpoint=args.checkpoint,
                    clouds=clouds,
                    axes=axes,
                    train_indices=split["train"],
                    validation_indices=split["validation"],
                    test_indices=split["test"],
                    condition=condition,
                    fold=int(fold),
                    seed=int(seed),
                    args=args,
                    device=device,
                )
                head_state = result.pop("head_state")
                result["constant_train_axis_test"] = constant
                result["axis_source"] = str(args.axis_source)
                stem = f"fold{fold}_{condition}_seed{seed}"
                torch.save(head_state, args.output_dir / f"{stem}_head.pt")
                torch.save(model.state_dict(), args.output_dir / f"{stem}_checkpoint.pth")
                vectors = predict_vectors(model, clouds, args.batch_size)
                np.savez_compressed(
                    args.output_dir / f"{stem}_vectors.npz",
                    vectors=vectors.detach().cpu().numpy().astype(np.float32),
                    episode_id=episode_ids,
                    shoe_id=shoe_ids,
                )
                (args.output_dir / f"{stem}.json").write_text(
                    json.dumps(result, indent=2) + "\n", encoding="utf-8"
                )
                public_results.append(result)
                print(
                    json.dumps(
                        {
                            "completed": stem,
                            "original_angle_deg": result["original_test"]["angle_deg_mean"],
                            "adapted_angle_deg": result["adapted_test"]["angle_deg_mean"],
                            "constant_angle_deg": constant["angle_deg_mean"],
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
