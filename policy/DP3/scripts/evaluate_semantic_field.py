#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import socket
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
SEMANTIC_PROJECT_ROOT = REPO_ROOT / "include" / "3d_semantic_train"
SEMANTIC_RELEASE_ROOT = SEMANTIC_PROJECT_ROOT / "semantic_field_release"
DEFAULT_UTONIA_ROOT = SEMANTIC_PROJECT_ROOT / "include" / "Utonia"
if not SEMANTIC_RELEASE_ROOT.is_dir():
    raise ImportError(
        "Semantic field release tree is unavailable at "
        f"{SEMANTIC_RELEASE_ROOT}. Check include/3d_semantic_train."
    )
if str(SEMANTIC_RELEASE_ROOT) not in sys.path:
    sys.path.insert(0, str(SEMANTIC_RELEASE_ROOT))
UTONIA_ROOT = Path(
    os.environ.get("UTONIA_ROOT", str(DEFAULT_UTONIA_ROOT))
).expanduser().resolve()
if not UTONIA_ROOT.is_dir():
    raise ImportError(f"Utonia checkout is unavailable: {UTONIA_ROOT}; set UTONIA_ROOT")
sys.path.insert(0, str(UTONIA_ROOT))

from my_datasets import (  # noqa: E402
    PartNextUniversalFieldDataset,
    partnext_universal_field_collate_fn,
)
from models import UtoniaUniversalFieldNet  # noqa: E402
from myutils import move_to_device, set_seed  # noqa: E402
from semantic_field_eval_utils import (  # noqa: E402
    confusion_matrix_from_predictions,
    merge_confusions,
    metrics_from_confusion,
    resolve_legacy_config_path,
    sha256_file,
)


def _safe_load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as error:
        raise RuntimeError(
            "This evaluator requires a PyTorch version supporting "
            "torch.load(..., weights_only=True)."
        ) from error
    except Exception as error:
        raise RuntimeError(
            f"Safe weights-only checkpoint loading failed for {path}. "
            "The evaluator will not fall back to pickle-based loading."
        ) from error


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _git_metadata() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status_lines = subprocess.run(
            ["git", "status", "--short"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        return {
            "commit": commit,
            "dirty": bool(status_lines),
            "status_short": status_lines,
        }
    except (OSError, subprocess.CalledProcessError) as error:
        return {"error": f"{type(error).__name__}: {error}"}


def _runtime_metadata(device: torch.device) -> dict:
    metadata = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
    }
    if device.type == "cuda":
        metadata.update(
            {
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_capability": list(torch.cuda.get_device_capability(device)),
            }
        )
    return metadata


def _resolve_checkpoint_args(
    checkpoint_args: dict,
    cli_args: argparse.Namespace,
) -> tuple[dict, dict | None]:
    resolved = dict(checkpoint_args)
    if cli_args.dataset_root is not None:
        resolved["dataset_root"] = str(cli_args.dataset_root.expanduser().resolve())

    configured_alias = cli_args.alias_config or resolved.get("alias_config")
    alias_path, alias_manifest = resolve_legacy_config_path(
        configured_alias,
        SEMANTIC_PROJECT_ROOT / "configs",
    )
    resolved["alias_config"] = str(alias_path) if alias_path is not None else None

    if cli_args.utonia_checkpoint is not None:
        resolved["utonia_checkpoint"] = str(
            cli_args.utonia_checkpoint.expanduser().resolve()
        )
    return resolved, alias_manifest


def _load_model(
    checkpoint_state: dict,
    checkpoint_args: dict,
    label_names: list[str],
    device: torch.device,
) -> UtoniaUniversalFieldNet:
    model = UtoniaUniversalFieldNet(
        num_classes=len(label_names),
        utonia_checkpoint=checkpoint_args.get("utonia_checkpoint", "auto"),
        utonia_repo_id=checkpoint_args.get("utonia_repo_id", "Pointcept/Utonia"),
        utonia_upcast_levels=int(checkpoint_args.get("utonia_upcast_levels", 0)),
        freeze_utonia=True,
        adapter_input_dim=checkpoint_args.get("adapter_input_dim"),
        adapter_hidden_dim=int(checkpoint_args.get("adapter_hidden_dim", 256)),
        branch_dim=int(checkpoint_args.get("branch_dim", 256)),
        sem_embedding_dim=int(checkpoint_args.get("sem_embedding_dim", 128)),
        geo_embedding_dim=int(checkpoint_args.get("geo_embedding_dim", 128)),
        num_anchors=int(checkpoint_args.get("num_anchors", 256)),
        rbf_sigma=float(checkpoint_args.get("rbf_sigma", 0.12)),
        triplane_resolution=int(checkpoint_args.get("triplane_resolution", 64)),
        triplane_padding=float(checkpoint_args.get("triplane_padding", 0.05)),
    )
    projector = checkpoint_state["projector_state_dict"]
    model.semantic_adapter.load_state_dict(projector["semantic_adapter"])
    if "semantic_local_fusion" in projector:
        model.semantic_local_fusion.load_state_dict(projector["semantic_local_fusion"])
    model.semantic_decoder.load_state_dict(projector["semantic_decoder"])
    if "encoder_state_dict" in checkpoint_state:
        model.feature_extractor.encoder.load_state_dict(
            checkpoint_state["encoder_state_dict"]
        )
    return model.to(device).eval()


def _dataset_value(cli_value, checkpoint_args: dict, key: str, default):
    return cli_value if cli_value is not None else checkpoint_args.get(key, default)


def _build_dataset(
    checkpoint_args: dict,
    label_names: list[str],
    cli_args: argparse.Namespace,
) -> PartNextUniversalFieldDataset:
    dataset_root = Path(checkpoint_args["dataset_root"]).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"PartNext dataset root does not exist: {dataset_root}")

    dataset = PartNextUniversalFieldDataset(
        dataset_root=dataset_root,
        split=cli_args.split,
        categories=checkpoint_args.get("categories"),
        num_support_points=int(
            _dataset_value(
                cli_args.num_support_points,
                checkpoint_args,
                "num_support_points",
                5000,
            )
        ),
        num_query_surface_points=int(
            _dataset_value(
                cli_args.num_query_surface_points,
                checkpoint_args,
                "num_query_surface_points",
                2048,
            )
        ),
        num_query_occ_points=int(checkpoint_args.get("num_query_occ_points", 1536)),
        num_query_probe_points=int(checkpoint_args.get("num_query_probe_points", 1024)),
        balanced_sampling_ratio=float(
            checkpoint_args.get("balanced_sampling_ratio", 0.5)
        ),
        query_sampling_ratio=float(checkpoint_args.get("query_sampling_ratio", 0.75)),
        near_surface_ratio=float(checkpoint_args.get("near_surface_ratio", 0.75)),
        near_surface_noise=float(checkpoint_args.get("near_surface_noise", 0.02)),
        uniform_padding=float(checkpoint_args.get("uniform_padding", 0.15)),
        coord_scale=float(checkpoint_args.get("coord_scale", 1.0)),
        center_shift_z=not bool(checkpoint_args.get("disable_z_shift", False)),
        grid_size=float(checkpoint_args.get("grid_size", 0.01)),
        val_ratio=float(checkpoint_args.get("val_ratio", 0.1)),
        test_ratio=float(checkpoint_args.get("test_ratio", 0.0)),
        split_seed=int(checkpoint_args.get("split_seed", 42)),
        limit_samples=None,
        max_retries=1,
        label_level=str(checkpoint_args.get("label_level", "leaf")),
        alias_config_path=checkpoint_args.get("alias_config"),
        canonical_label_names=label_names,
        ignore_labels=checkpoint_args.get("ignore_labels", ["Other"]),
        rotation_mode=str(checkpoint_args.get("rotation_mode", "so3")),
        jitter_std=float(checkpoint_args.get("jitter_std", 0.005)),
        second_view=False,
        probe_label_radius=float(checkpoint_args.get("probe_label_radius", 0.1)),
    )
    if len(dataset) == 0:
        raise RuntimeError(
            f"Requested split={cli_args.split!r} is empty. "
            f"Checkpoint val_ratio={checkpoint_args.get('val_ratio', 0.1)} and "
            f"test_ratio={checkpoint_args.get('test_ratio', 0.0)}."
        )
    return dataset


def _sample_seed(base_seed: int, repeat: int, sample_index: int) -> int:
    return int(base_seed + repeat * 1_000_003 + sample_index * 10_007)


def _metric_row(prefix: dict, metrics: dict) -> dict:
    return {
        **prefix,
        "valid_points": metrics["valid_points"],
        "accuracy": metrics["accuracy"],
        "mean_iou": metrics["mean_iou"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "frequency_weighted_iou": metrics["frequency_weighted_iou"],
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _evaluation_status(split: str, checkpoint_args: dict) -> str:
    if split == "val" and float(checkpoint_args.get("test_ratio", 0.0)) == 0.0:
        return "diagnostic_validation_checkpoint_selected_on_same_split"
    if split == "test":
        return "held_out_test"
    return f"diagnostic_{split}"


def evaluate(args: argparse.Namespace) -> dict:
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to mix results into non-empty output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_state = _safe_load_checkpoint(checkpoint_path)
    label_names = [str(item) for item in checkpoint_state["canonical_label_names"]]
    checkpoint_args, alias_manifest = _resolve_checkpoint_args(
        checkpoint_state.get("args", {}),
        args,
    )
    utonia_path = Path(checkpoint_args["utonia_checkpoint"]).expanduser().resolve()
    if not utonia_path.is_file():
        raise FileNotFoundError(f"Utonia checkpoint does not exist: {utonia_path}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    dataset = _build_dataset(checkpoint_args, label_names, args)
    sample_count = len(dataset)
    if args.max_samples is not None:
        sample_count = min(sample_count, int(args.max_samples))
    if sample_count <= 0:
        raise ValueError("Evaluation requires at least one sample")

    split_records = [
        {
            "index": index,
            "category": record.category,
            "model_id": record.model_id,
            "mesh_path": str(record.mesh_path),
            "selected": index < sample_count,
        }
        for index, record in enumerate(dataset.records)
    ]
    model = _load_model(checkpoint_state, checkpoint_args, label_names, device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()

    all_confusions: list[np.ndarray] = []
    repeat_confusions: dict[int, list[np.ndarray]] = defaultdict(list)
    model_confusions: dict[str, list[np.ndarray]] = defaultdict(list)
    per_sample_rows: list[dict] = []
    sampling_seconds = 0.0
    inference_seconds = 0.0

    for repeat in range(args.repeats):
        for sample_index in range(sample_count):
            seed = _sample_seed(args.seed, repeat, sample_index)
            set_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

            sample_started = time.perf_counter()
            sample = dataset[sample_index]
            sampling_seconds += time.perf_counter() - sample_started

            expected_record = dataset.records[sample_index]
            if sample["model_id"] != expected_record.model_id:
                raise RuntimeError(
                    "Dataset retry changed the evaluated object despite max_retries=1: "
                    f"expected {expected_record.model_id}, got {sample['model_id']}"
                )
            batch = partnext_universal_field_collate_fn([sample])
            batch = move_to_device(batch, device)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_started = time.perf_counter()
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=bool(args.amp and device.type == "cuda"),
            ):
                output = model(batch, train_mode="semantic")["view1"]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds += time.perf_counter() - inference_started

            predictions = output["sem_logits"][0].argmax(dim=-1).detach().cpu().numpy()
            targets = sample["query_surface_labels"].cpu().numpy()
            confusion, valid_count = confusion_matrix_from_predictions(
                predictions,
                targets,
                len(label_names),
            )
            if valid_count == 0:
                raise RuntimeError(
                    f"No valid semantic query labels for model {sample['model_id']}"
                )
            metrics = metrics_from_confusion(confusion, label_names)
            key = f"{sample['category']}::{sample['model_id']}"
            all_confusions.append(confusion)
            repeat_confusions[repeat].append(confusion)
            model_confusions[key].append(confusion)
            per_sample_rows.append(
                _metric_row(
                    {
                        "repeat": repeat,
                        "sample_index": sample_index,
                        "seed": seed,
                        "category": sample["category"],
                        "model_id": sample["model_id"],
                        "mesh_path": sample["mesh_path"],
                    },
                    metrics,
                )
            )
            print(
                f"[R1] repeat={repeat} sample={sample_index + 1}/{sample_count} "
                f"model={sample['model_id']} acc={metrics['accuracy']:.4f} "
                f"mIoU={metrics['mean_iou']:.4f}",
                flush=True,
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - started
    aggregate_confusion = merge_confusions(all_confusions, len(label_names))
    aggregate_metrics = metrics_from_confusion(aggregate_confusion, label_names)

    per_repeat = []
    for repeat, confusions in sorted(repeat_confusions.items()):
        metrics = metrics_from_confusion(
            merge_confusions(confusions, len(label_names)),
            label_names,
        )
        per_repeat.append(_metric_row({"repeat": repeat}, metrics))

    per_model = []
    for key, confusions in sorted(model_confusions.items()):
        category, model_id = key.split("::", maxsplit=1)
        metrics = metrics_from_confusion(
            merge_confusions(confusions, len(label_names)),
            label_names,
        )
        per_model.append(
            _metric_row(
                {"category": category, "model_id": model_id},
                metrics,
            )
        )

    peak_allocated_mib = None
    peak_reserved_mib = None
    if device.type == "cuda":
        peak_allocated_mib = torch.cuda.max_memory_allocated(device) / (1024**2)
        peak_reserved_mib = torch.cuda.max_memory_reserved(device) / (1024**2)

    result = {
        "schema_version": 1,
        "experiment": "R1_part_semantic_metrics",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_status": _evaluation_status(args.split, checkpoint_args),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "epoch": checkpoint_state.get("epoch"),
            "best_sem_score": checkpoint_state.get(
                "best_sem_score", checkpoint_state.get("best_val_score")
            ),
        },
        "utonia_checkpoint": {
            "path": str(utonia_path),
            "sha256": sha256_file(utonia_path),
        },
        "alias_config": alias_manifest,
        "checkpoint_args_resolved": _jsonable(checkpoint_args),
        "evaluation_args": _jsonable(vars(args)),
        "labels": label_names,
        "split": {
            "name": args.split,
            "seed": int(checkpoint_args.get("split_seed", 42)),
            "val_ratio": float(checkpoint_args.get("val_ratio", 0.1)),
            "test_ratio": float(checkpoint_args.get("test_ratio", 0.0)),
            "total_records": len(dataset),
            "evaluated_records": sample_count,
            "records": split_records,
        },
        "aggregate": aggregate_metrics,
        "per_repeat": per_repeat,
        "per_model": per_model,
        "timing": {
            "wall_seconds": wall_seconds,
            "sampling_seconds": sampling_seconds,
            "inference_seconds": inference_seconds,
            "seconds_per_model_repeat": wall_seconds / len(per_sample_rows),
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
    (output_dir / "split_manifest.json").write_text(
        json.dumps(_jsonable(result["split"]), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "confusion_matrix.json").write_text(
        json.dumps(
            {
                "labels": label_names,
                "matrix_rows_ground_truth_columns_prediction": aggregate_confusion.tolist(),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _write_csv(output_dir / "per_sample.csv", per_sample_rows)
    _write_csv(output_dir / "per_model.csv", per_model)
    _write_csv(output_dir / "per_repeat.csv", per_repeat)
    _write_csv(output_dir / "per_part.csv", aggregate_metrics["per_class"])

    print(
        f"[R1] complete: split={args.split} models={sample_count} "
        f"repeats={args.repeats} acc={aggregate_metrics['accuracy']:.4f} "
        f"mIoU={aggregate_metrics['mean_iou']:.4f} output={output_dir}",
        flush=True,
    )
    return result


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate PartNext semantic logits with instance-level aggregate, "
            "per-part IoU, and reproducibility manifests."
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
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--num-support-points", type=int)
    parser.add_argument("--num-query-surface-points", type=int)
    parser.add_argument("--amp", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    evaluate(args)


if __name__ == "__main__":
    main()
