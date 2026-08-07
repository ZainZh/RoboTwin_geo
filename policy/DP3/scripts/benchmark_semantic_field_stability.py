#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
SEMANTIC_RELEASE_ROOT = (
    REPO_ROOT / "include" / "3d_semantic_train" / "semantic_field_release"
)
DEFAULT_UTONIA_ROOT = REPO_ROOT / "include" / "3d_semantic_train" / "include" / "Utonia"
if str(SEMANTIC_RELEASE_ROOT) not in sys.path:
    sys.path.insert(0, str(SEMANTIC_RELEASE_ROOT))
UTONIA_ROOT = Path(
    os.environ.get("UTONIA_ROOT", str(DEFAULT_UTONIA_ROOT))
).expanduser().resolve()
if not UTONIA_ROOT.is_dir():
    raise ImportError(f"Utonia checkout is unavailable: {UTONIA_ROOT}; set UTONIA_ROOT")
sys.path.insert(0, str(UTONIA_ROOT))

from my_datasets.partnext_canonical_field import (  # noqa: E402
    UTONIA,
    _build_global_face_labels,
    _concatenate_meshes,
    _load_geometry_meshes,
    _sample_surface_points,
)
from myutils import move_to_device, set_seed  # noqa: E402

from evaluate_semantic_field import (  # noqa: E402
    _build_dataset,
    _evaluation_status,
    _git_metadata,
    _jsonable,
    _load_model,
    _resolve_checkpoint_args,
    _runtime_metadata,
    _safe_load_checkpoint,
    _sample_seed,
    _write_csv,
)
from semantic_field_eval_utils import (  # noqa: E402
    apply_support_normalization,
    compute_support_normalization,
    confusion_matrix_from_predictions,
    embedding_stability_metrics,
    merge_confusions,
    metrics_from_confusion,
    sha256_file,
)
from semantic_observation_perturbations import (  # noqa: E402
    replace_with_aabb_shell_outliers,
    select_normal_facing_view,
)


def _load_annotated_mesh(dataset, record):
    meshes = _load_geometry_meshes(record.mesh_path)
    mesh = _concatenate_meshes(meshes)
    face_labels = _build_global_face_labels(
        record=record,
        num_meshes=len(meshes),
        mesh_face_counts=[len(item.faces) for item in meshes],
        total_faces=len(mesh.faces),
        canonical_label_to_id=dataset.canonical_label_to_id,
        label_level=dataset.label_level,
        alias_lookup=dataset.alias_lookup,
        ignore_names=dataset.ignore_names,
    )
    return mesh, face_labels


def _sample_support(mesh, face_labels, count: int, balanced_ratio: float):
    points, normals, colors, _ = _sample_surface_points(
        mesh=mesh,
        face_labels=face_labels,
        num_points=int(count),
        balanced_sampling_ratio=float(balanced_ratio),
    )
    return (
        points.astype(np.float32, copy=False),
        normals.astype(np.float32, copy=False),
        colors.astype(np.float32, copy=False),
    )


def _sample_fixed_queries(mesh, face_labels, count: int, balanced_ratio: float):
    points, _, _, labels = _sample_surface_points(
        mesh=mesh,
        face_labels=face_labels,
        num_points=int(count),
        balanced_sampling_ratio=float(balanced_ratio),
    )
    return points.astype(np.float32, copy=False), labels.astype(np.int64, copy=False)


def _build_conditions(args: argparse.Namespace) -> list[dict]:
    conditions: list[dict] = []
    for count in dict.fromkeys(args.support_counts):
        if count <= 0:
            raise ValueError(f"support count must be positive, got {count}")
        conditions.append(
            {"kind": "resample_count", "value": float(count), "label": f"count_{count}"}
        )
    for ratio in dict.fromkeys(args.dropout_ratios):
        if not 0.0 < ratio < 1.0:
            raise ValueError(f"dropout ratio must be in (0, 1), got {ratio}")
        conditions.append(
            {"kind": "dropout", "value": float(ratio), "label": f"dropout_{ratio:g}"}
        )
    for fraction in dict.fromkeys(args.noise_std_fractions):
        if fraction <= 0.0:
            raise ValueError(f"noise fraction must be positive, got {fraction}")
        conditions.append(
            {
                "kind": "gaussian_noise",
                "value": float(fraction),
                "label": f"noise_{fraction:g}",
            }
        )
    for keep_ratio in dict.fromkeys(args.crop_keep_ratios):
        if not 0.0 < keep_ratio < 1.0:
            raise ValueError(f"crop keep ratio must be in (0, 1), got {keep_ratio}")
        conditions.append(
            {
                "kind": "view_crop",
                "value": float(keep_ratio),
                "label": f"crop_keep_{keep_ratio:g}",
            }
        )
    for ratio in dict.fromkeys(args.outlier_ratios):
        if not 0.0 < ratio < 1.0:
            raise ValueError(f"outlier ratio must be in (0, 1), got {ratio}")
        conditions.append(
            {
                "kind": "aabb_shell_outliers",
                "value": float(ratio),
                "label": f"outlier_replace_{ratio:g}",
            }
        )
    for keep_ratio in dict.fromkeys(args.normal_view_keep_ratios):
        if not 0.0 < keep_ratio < 1.0:
            raise ValueError(
                f"normal-view keep ratio must be in (0, 1), got {keep_ratio}"
            )
        conditions.append(
            {
                "kind": "normal_facing_view",
                "value": float(keep_ratio),
                "label": f"normal_view_keep_{keep_ratio:g}",
            }
        )
    if not conditions:
        raise ValueError("At least one perturbation condition is required")
    return conditions


def _variant_support(
    *,
    condition: dict,
    mesh,
    face_labels: np.ndarray,
    reference_points: np.ndarray,
    reference_normals: np.ndarray,
    reference_colors: np.ndarray,
    reference_radius: float,
    balanced_ratio: float,
    rng: np.random.Generator,
):
    kind = condition["kind"]
    value = condition["value"]
    if kind == "resample_count":
        return _sample_support(mesh, face_labels, int(value), balanced_ratio)

    if kind == "dropout":
        keep_count = max(8, int(round(len(reference_points) * (1.0 - value))))
        indices = rng.choice(len(reference_points), size=keep_count, replace=False)
        return (
            reference_points[indices].copy(),
            reference_normals[indices].copy(),
            reference_colors[indices].copy(),
        )

    if kind == "gaussian_noise":
        noise = rng.normal(
            loc=0.0,
            scale=float(value) * float(reference_radius),
            size=reference_points.shape,
        ).astype(np.float32)
        return (
            (reference_points + noise).astype(np.float32),
            reference_normals.copy(),
            reference_colors.copy(),
        )

    if kind == "view_crop":
        direction = rng.normal(size=3)
        direction = direction / max(float(np.linalg.norm(direction)), 1e-8)
        projection = reference_points @ direction.astype(np.float32)
        keep_count = max(8, int(round(len(reference_points) * float(value))))
        indices = np.argpartition(projection, -keep_count)[-keep_count:]
        return (
            reference_points[indices].copy(),
            reference_normals[indices].copy(),
            reference_colors[indices].copy(),
        )

    if kind == "aabb_shell_outliers":
        return replace_with_aabb_shell_outliers(
            reference_points,
            reference_normals,
            reference_colors,
            ratio=float(value),
            rng=rng,
        )

    if kind == "normal_facing_view":
        return select_normal_facing_view(
            reference_points,
            reference_normals,
            reference_colors,
            keep_ratio=float(value),
            rng=rng,
        )


    raise ValueError(f"Unsupported condition kind: {kind}")


def _run_semantic_readout(
    *,
    model,
    dataset,
    device: torch.device,
    support_points: np.ndarray,
    support_normals: np.ndarray,
    support_colors: np.ndarray,
    query_points: np.ndarray,
    amp: bool,
) -> tuple[np.ndarray, np.ndarray]:
    utonia_item = dataset.utonia_transform(
        {
            "coord": support_points.copy(),
            "color": support_colors.copy(),
            "normal": support_normals.copy(),
        }
    )
    utonia_input = UTONIA.data.collate_fn([utonia_item])
    support_tensor = torch.from_numpy(support_points).float().unsqueeze(0)
    query_tensor = torch.from_numpy(query_points).float().unsqueeze(0)
    payload = move_to_device(
        {
            "utonia_input": utonia_input,
            "support_points": support_tensor,
            "query_points": query_tensor,
        },
        device,
    )
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=bool(amp and device.type == "cuda"),
    ):
        cache = model.encode_support(
            payload["utonia_input"],
            payload["support_points"],
        )
        output = model.query_semantic(cache, payload["query_points"])
    embeddings = output["embedding"][0].detach().float().cpu().numpy()
    predictions = output["logits"][0].argmax(dim=-1).detach().cpu().numpy()
    return embeddings.astype(np.float32, copy=False), predictions.astype(np.int64)


def _mean_numeric(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def benchmark(args: argparse.Namespace) -> dict:
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to mix results into non-empty output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_state = _safe_load_checkpoint(checkpoint_path)
    label_names = [str(item) for item in checkpoint_state["canonical_label_names"]]
    checkpoint_args, alias_manifest = _resolve_checkpoint_args(
        checkpoint_state.get("args", {}), args
    )
    device = torch.device(args.device)
    dataset = _build_dataset(checkpoint_args, label_names, args)
    model = _load_model(checkpoint_state, checkpoint_args, label_names, device)
    utonia_path = Path(checkpoint_args["utonia_checkpoint"]).expanduser().resolve()

    sample_count = len(dataset)
    if args.max_samples is not None:
        sample_count = min(sample_count, int(args.max_samples))
    conditions = _build_conditions(args)
    reference_support_count = int(
        args.reference_support_points
        or checkpoint_args.get("num_support_points", 5000)
    )
    query_count = int(
        args.num_query_surface_points
        or checkpoint_args.get("num_query_surface_points", 2048)
    )
    coord_scale = float(checkpoint_args.get("coord_scale", 1.0))
    center_shift_z = not bool(checkpoint_args.get("disable_z_shift", False))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()

    trial_rows: list[dict] = []
    reference_rows: list[dict] = []
    reference_confusions: list[np.ndarray] = []
    group_confusions: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    group_rows: dict[tuple[str, str], list[dict]] = defaultdict(list)
    split_records = []

    for sample_index in range(sample_count):
        record = dataset.records[sample_index]
        split_records.append(
            {
                "index": sample_index,
                "category": record.category,
                "model_id": record.model_id,
                "mesh_path": str(record.mesh_path),
            }
        )
        object_seed = _sample_seed(args.seed, 0, sample_index)
        set_seed(object_seed)
        random.seed(object_seed)
        np.random.seed(object_seed)
        mesh, face_labels = _load_annotated_mesh(dataset, record)
        query_raw, query_labels = _sample_fixed_queries(
            mesh,
            face_labels,
            query_count,
            float(checkpoint_args.get("query_sampling_ratio", 0.75)),
        )
        reference_points, reference_normals, reference_colors = _sample_support(
            mesh,
            face_labels,
            reference_support_count,
            float(checkpoint_args.get("balanced_sampling_ratio", 0.5)),
        )
        reference_parameters = compute_support_normalization(
            reference_points,
            coord_scale,
            center_shift_z,
        )
        reference_support_normalized = apply_support_normalization(
            reference_points, reference_parameters
        )
        reference_query_normalized = apply_support_normalization(
            query_raw, reference_parameters
        )
        reference_embeddings, reference_predictions = _run_semantic_readout(
            model=model,
            dataset=dataset,
            device=device,
            support_points=reference_support_normalized,
            support_normals=reference_normals,
            support_colors=reference_colors,
            query_points=reference_query_normalized,
            amp=args.amp,
        )
        reference_confusion, valid_count = confusion_matrix_from_predictions(
            reference_predictions,
            query_labels,
            len(label_names),
        )
        if valid_count == 0:
            raise RuntimeError(f"No valid fixed queries for model {record.model_id}")
        reference_metrics = metrics_from_confusion(reference_confusion, label_names)
        reference_confusions.append(reference_confusion)
        reference_rows.append(
            {
                "sample_index": sample_index,
                "category": record.category,
                "model_id": record.model_id,
                "seed": object_seed,
                "support_points": len(reference_points),
                "query_points": len(query_raw),
                "valid_points": valid_count,
                "accuracy": reference_metrics["accuracy"],
                "mean_iou": reference_metrics["mean_iou"],
            }
        )

        for condition_index, condition in enumerate(conditions):
            for trial in range(args.trials):
                trial_seed = int(
                    object_seed + (condition_index + 1) * 100_003 + trial * 1_009
                )
                set_seed(trial_seed)
                random.seed(trial_seed)
                np.random.seed(trial_seed)
                rng = np.random.default_rng(trial_seed)
                variant_points, variant_normals, variant_colors = _variant_support(
                    condition=condition,
                    mesh=mesh,
                    face_labels=face_labels,
                    reference_points=reference_points,
                    reference_normals=reference_normals,
                    reference_colors=reference_colors,
                    reference_radius=float(reference_parameters["radius"]),
                    balanced_ratio=float(
                        checkpoint_args.get("balanced_sampling_ratio", 0.5)
                    ),
                    rng=rng,
                )

                for normalization_mode in args.normalization_modes:
                    if normalization_mode == "support":
                        parameters = compute_support_normalization(
                            variant_points,
                            coord_scale,
                            center_shift_z,
                        )
                    elif normalization_mode == "reference":
                        parameters = reference_parameters
                    else:
                        raise ValueError(
                            f"Unsupported normalization mode: {normalization_mode}"
                        )
                    support_normalized = apply_support_normalization(
                        variant_points, parameters
                    )
                    query_normalized = apply_support_normalization(query_raw, parameters)
                    embeddings, predictions = _run_semantic_readout(
                        model=model,
                        dataset=dataset,
                        device=device,
                        support_points=support_normalized,
                        support_normals=variant_normals,
                        support_colors=variant_colors,
                        query_points=query_normalized,
                        amp=args.amp,
                    )
                    confusion, _ = confusion_matrix_from_predictions(
                        predictions,
                        query_labels,
                        len(label_names),
                    )
                    metrics = metrics_from_confusion(confusion, label_names)
                    stability = embedding_stability_metrics(
                        reference_embeddings, embeddings
                    )
                    agreement = float(np.mean(predictions == reference_predictions))
                    query_drift = np.linalg.norm(
                        query_normalized - reference_query_normalized,
                        axis=1,
                    )
                    row = {
                        "sample_index": sample_index,
                        "category": record.category,
                        "model_id": record.model_id,
                        "condition": condition["label"],
                        "condition_kind": condition["kind"],
                        "condition_value": condition["value"],
                        "normalization_mode": normalization_mode,
                        "trial": trial,
                        "seed": trial_seed,
                        "support_points": len(variant_points),
                        "query_points": len(query_raw),
                        "accuracy": metrics["accuracy"],
                        "mean_iou": metrics["mean_iou"],
                        "delta_miou_vs_object_reference": (
                            metrics["mean_iou"] - reference_metrics["mean_iou"]
                        ),
                        "label_agreement_with_reference": agreement,
                        "query_coord_drift_mean": float(query_drift.mean()),
                        "query_coord_drift_p95": float(np.quantile(query_drift, 0.95)),
                        **stability,
                    }
                    trial_rows.append(row)
                    group_key = (normalization_mode, condition["label"])
                    group_confusions[group_key].append(confusion)
                    group_rows[group_key].append(row)

                print(
                    f"[R2] sample={sample_index + 1}/{sample_count} "
                    f"condition={condition['label']} trial={trial + 1}/{args.trials}",
                    flush=True,
                )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - started
    aggregate_reference = metrics_from_confusion(
        merge_confusions(reference_confusions, len(label_names)),
        label_names,
    )
    aggregate_conditions = []
    for group_key, confusions in sorted(group_confusions.items()):
        normalization_mode, condition_label = group_key
        rows = group_rows[group_key]
        metrics = metrics_from_confusion(
            merge_confusions(confusions, len(label_names)),
            label_names,
        )
        aggregate_conditions.append(
            {
                "normalization_mode": normalization_mode,
                "condition": condition_label,
                "condition_kind": rows[0]["condition_kind"],
                "condition_value": rows[0]["condition_value"],
                "num_object_trials": len(rows),
                "support_points_mean": _mean_numeric(rows, "support_points"),
                "accuracy": metrics["accuracy"],
                "mean_iou": metrics["mean_iou"],
                "delta_miou_vs_aggregate_reference": (
                    metrics["mean_iou"] - aggregate_reference["mean_iou"]
                ),
                "label_agreement_with_reference": _mean_numeric(
                    rows, "label_agreement_with_reference"
                ),
                "cosine_mean": _mean_numeric(rows, "cosine_mean"),
                "cosine_p05_mean": _mean_numeric(rows, "cosine_p05"),
                "embedding_l2_mean": _mean_numeric(rows, "embedding_l2_mean"),
                "query_coord_drift_mean": _mean_numeric(
                    rows, "query_coord_drift_mean"
                ),
                "confusion_matrix": metrics["confusion_matrix"],
                "per_class": metrics["per_class"],
            }
        )

    peak_allocated_mib = None
    peak_reserved_mib = None
    if device.type == "cuda":
        peak_allocated_mib = torch.cuda.max_memory_allocated(device) / (1024**2)
        peak_reserved_mib = torch.cuda.max_memory_reserved(device) / (1024**2)

    result = {
        "schema_version": 1,
        "experiment": "R2_fixed_query_support_perturbation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_status": _evaluation_status(args.split, checkpoint_args),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "epoch": checkpoint_state.get("epoch"),
        },
        "utonia_checkpoint": {
            "path": str(utonia_path),
            "sha256": sha256_file(utonia_path),
        },
        "alias_config": alias_manifest,
        "labels": label_names,
        "evaluation_args": _jsonable(vars(args)),
        "checkpoint_args_resolved": _jsonable(checkpoint_args),
        "fixed_query_protocol": {
            "same_raw_mesh_surface_query_coordinates_across_all_conditions": True,
            "query_count": query_count,
            "query_sampling_ratio": float(
                checkpoint_args.get("query_sampling_ratio", 0.75)
            ),
            "reference_support_count": reference_support_count,
            "normalization_modes": args.normalization_modes,
            "projector_bounds_remain_support_dependent": True,
        },
        "split": {
            "name": args.split,
            "seed": int(checkpoint_args.get("split_seed", 42)),
            "val_ratio": float(checkpoint_args.get("val_ratio", 0.1)),
            "test_ratio": float(checkpoint_args.get("test_ratio", 0.0)),
            "evaluated_records": sample_count,
            "records": split_records,
        },
        "reference": aggregate_reference,
        "aggregate_conditions": aggregate_conditions,
        "timing": {
            "wall_seconds": wall_seconds,
            "num_condition_forwards": len(trial_rows),
            "seconds_per_condition_forward": wall_seconds / max(len(trial_rows), 1),
        },
        "memory": {
            "peak_allocated_mib": peak_allocated_mib,
            "peak_reserved_mib": peak_reserved_mib,
        },
        "runtime": _runtime_metadata(device),
        "git": _git_metadata(),
    }

    (output_dir / "results.json").write_text(
        json.dumps(_jsonable(result), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_csv(output_dir / "reference_per_model.csv", reference_rows)
    _write_csv(output_dir / "trials.csv", trial_rows)
    _write_csv(
        output_dir / "aggregate_conditions.csv",
        [
            {key: value for key, value in row.items() if key not in {"confusion_matrix", "per_class"}}
            for row in aggregate_conditions
        ],
    )
    print(
        f"[R2] complete: models={sample_count} conditions={len(conditions)} "
        f"trials={args.trials} modes={args.normalization_modes} output={output_dir}",
        flush=True,
    )
    return result


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark semantic-field stability at fixed annotated query points "
            "while perturbing only the support observation."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--alias-config", type=Path)
    parser.add_argument("--utonia-checkpoint", type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--reference-support-points", type=int)
    parser.add_argument("--num-support-points", type=int)
    parser.add_argument("--num-query-surface-points", type=int)
    parser.add_argument(
        "--support-counts",
        nargs="*",
        type=int,
        default=[128, 256, 512, 1024, 5000],
    )
    parser.add_argument(
        "--dropout-ratios", nargs="*", type=float, default=[0.25, 0.5, 0.75]
    )
    parser.add_argument(
        "--noise-std-fractions",
        nargs="*",
        type=float,
        default=[0.0025, 0.005, 0.01],
    )
    parser.add_argument(
        "--crop-keep-ratios", nargs="*", type=float, default=[0.75, 0.5, 0.25]
    )
    parser.add_argument(
        "--outlier-ratios", nargs="*", type=float, default=[0.05, 0.1, 0.2]
    )
    parser.add_argument(
        "--normal-view-keep-ratios",
        nargs="*",
        type=float,
        default=[0.75, 0.5, 0.25],
        help=(
            "Approximate single-view support by keeping points with the most "
            "camera-facing normals; this is not a z-buffer visibility test."
        ),
    )
    parser.add_argument(
        "--normalization-modes",
        nargs="+",
        choices=("support", "reference"),
        default=["support", "reference"],
    )
    parser.add_argument("--amp", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    benchmark(args)


if __name__ == "__main__":
    main()
