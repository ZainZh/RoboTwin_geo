#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[3]
SEMANTIC_RELEASE_ROOT = (
    REPO_ROOT / "include" / "3d_semantic_train" / "semantic_field_release"
)
if str(SEMANTIC_RELEASE_ROOT) not in sys.path:
    sys.path.insert(0, str(SEMANTIC_RELEASE_ROOT))

from my_datasets import PartNextUniversalFieldDataset  # noqa: E402
from my_datasets.partnext_canonical_field import (  # noqa: E402
    UTONIA,
    _build_global_face_labels,
    _build_rotation_matrix,
    _concatenate_meshes,
    _load_geometry_meshes,
    _sample_surface_points,
)
from models.universal_field.losses import (  # noqa: E402
    embedding_consistency_loss,
    semantic_cross_entropy_loss,
)
from myutils import move_to_device, set_seed  # noqa: E402

from evaluate_semantic_field import (  # noqa: E402
    _git_metadata,
    _jsonable,
    _load_model,
    _resolve_checkpoint_args,
    _runtime_metadata,
    _safe_load_checkpoint,
    _write_csv,
)
from semantic_field_eval_utils import (  # noqa: E402
    confusion_matrix_from_predictions,
    merge_confusions,
    metrics_from_confusion,
    sha256_file,
)
from semantic_support_perturbations import (  # noqa: E402
    SupportCondition,
    build_fixed_query_support_pair,
    choose_support_condition,
    normalize_condition_weights,
)


@contextmanager
def _temporary_numpy_seed(seed: int):
    state = np.random.get_state()
    np.random.seed(int(seed) % (2**32))
    try:
        yield np.random.default_rng(int(seed))
    finally:
        np.random.set_state(state)


def _worker_seed(_worker_id: int) -> None:
    seed = int(torch.initial_seed() % (2**32))
    np.random.seed(seed)
    random.seed(seed)


class FixedQuerySupportCalibrationDataset(Dataset):
    def __init__(
        self,
        *,
        base_dataset: PartNextUniversalFieldDataset,
        conditions: list[SupportCondition],
        probabilities: np.ndarray,
        reference_support_count: int,
        query_count: int,
        network_support_count: int,
        coord_scale: float,
        center_shift_z: bool,
        balanced_sampling_ratio: float,
        query_sampling_ratio: float,
        rotation_mode: str,
        train: bool,
        seed: int,
    ) -> None:
        self.base_dataset = base_dataset
        self.records = base_dataset.records
        self.conditions = conditions
        self.probabilities = probabilities
        self.reference_support_count = int(reference_support_count)
        self.query_count = int(query_count)
        self.network_support_count = int(network_support_count)
        self.coord_scale = float(coord_scale)
        self.center_shift_z = bool(center_shift_z)
        self.balanced_sampling_ratio = float(balanced_sampling_ratio)
        self.query_sampling_ratio = float(query_sampling_ratio)
        self.rotation_mode = str(rotation_mode)
        self.train = bool(train)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.records)

    def _load_mesh(self, index: int):
        record = self.records[index]
        meshes = _load_geometry_meshes(record.mesh_path)
        mesh = _concatenate_meshes(meshes)
        face_labels = _build_global_face_labels(
            record=record,
            num_meshes=len(meshes),
            mesh_face_counts=[len(item.faces) for item in meshes],
            total_faces=len(mesh.faces),
            canonical_label_to_id=self.base_dataset.canonical_label_to_id,
            label_level=self.base_dataset.label_level,
            alias_lookup=self.base_dataset.alias_lookup,
            ignore_names=self.base_dataset.ignore_names,
        )
        return record, mesh, face_labels

    def load_condition(
        self,
        index: int,
        condition: SupportCondition,
        seed: int,
    ) -> dict:
        with _temporary_numpy_seed(seed) as rng:
            record, mesh, face_labels = self._load_mesh(index)
            support_points, support_normals, support_colors, _ = _sample_surface_points(
                mesh=mesh,
                face_labels=face_labels,
                num_points=self.reference_support_count,
                balanced_sampling_ratio=self.balanced_sampling_ratio,
            )
            query_points, _, _, query_labels = _sample_surface_points(
                mesh=mesh,
                face_labels=face_labels,
                num_points=self.query_count,
                balanced_sampling_ratio=self.query_sampling_ratio,
            )
            rotation = (
                _build_rotation_matrix(self.rotation_mode)
                if self.train
                else np.eye(3, dtype=np.float32)
            )
            pair = build_fixed_query_support_pair(
                reference_points=support_points,
                reference_normals=support_normals,
                reference_colors=support_colors,
                raw_query_points=query_points,
                condition=condition,
                output_support_count=self.network_support_count,
                coord_scale=self.coord_scale,
                center_shift_z=self.center_shift_z,
                rng=rng,
                rotation=rotation,
            )
            clean_utonia = self.base_dataset.utonia_transform(
                {
                    "coord": pair["clean_support_points"].copy(),
                    "color": pair["clean_support_colors"].copy(),
                    "normal": pair["clean_support_normals"].copy(),
                }
            )
            perturbed_utonia = self.base_dataset.utonia_transform(
                {
                    "coord": pair["perturbed_support_points"].copy(),
                    "color": pair["perturbed_support_colors"].copy(),
                    "normal": pair["perturbed_support_normals"].copy(),
                }
            )

        return {
            "utonia_input_clean": clean_utonia,
            "utonia_input_perturbed": perturbed_utonia,
            "support_points_clean": torch.from_numpy(
                pair["clean_support_points"]
            ).float(),
            "support_points_perturbed": torch.from_numpy(
                pair["perturbed_support_points"]
            ).float(),
            "query_points_clean": torch.from_numpy(pair["clean_query_points"]).float(),
            "query_points_perturbed": torch.from_numpy(
                pair["perturbed_query_points"]
            ).float(),
            "query_labels": torch.from_numpy(query_labels.astype(np.int64)).long(),
            "condition": pair["condition"],
            "observed_source_points": pair["observed_source_points"],
            "crop_direction": pair["crop_direction"],
            "category": record.category,
            "model_id": record.model_id,
            "mesh_path": str(record.mesh_path),
            "sample_seed": int(seed),
        }

    def __getitem__(self, index: int) -> dict:
        sample_seed = int(np.random.randint(0, 2**31 - 1))
        rng = np.random.default_rng(sample_seed)
        condition = choose_support_condition(
            self.conditions,
            self.probabilities,
            rng,
        )
        return self.load_condition(index, condition, sample_seed)


def calibration_collate_fn(samples: list[dict]) -> dict:
    return {
        "utonia_input_clean": UTONIA.data.collate_fn(
            [sample["utonia_input_clean"] for sample in samples]
        ),
        "utonia_input_perturbed": UTONIA.data.collate_fn(
            [sample["utonia_input_perturbed"] for sample in samples]
        ),
        "support_points_clean": torch.stack(
            [sample["support_points_clean"] for sample in samples]
        ),
        "support_points_perturbed": torch.stack(
            [sample["support_points_perturbed"] for sample in samples]
        ),
        "query_points_clean": torch.stack(
            [sample["query_points_clean"] for sample in samples]
        ),
        "query_points_perturbed": torch.stack(
            [sample["query_points_perturbed"] for sample in samples]
        ),
        "query_labels": torch.stack([sample["query_labels"] for sample in samples]),
        "condition": [sample["condition"] for sample in samples],
        "observed_source_points": [
            sample["observed_source_points"] for sample in samples
        ],
        "crop_direction": [sample["crop_direction"] for sample in samples],
        "category": [sample["category"] for sample in samples],
        "model_id": [sample["model_id"] for sample in samples],
        "mesh_path": [sample["mesh_path"] for sample in samples],
        "sample_seed": [sample["sample_seed"] for sample in samples],
    }


class FrozenSemanticTeacher(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.adapter = copy.deepcopy(model.semantic_adapter)
        self.projector = copy.deepcopy(model.semantic_projector)
        self.local_fusion = copy.deepcopy(model.semantic_local_fusion)
        self.decoder = copy.deepcopy(model.semantic_decoder)
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(
        self,
        support_features: torch.Tensor,
        support_points: torch.Tensor,
        query_points: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        semantic_support = self.adapter(support_features)
        cache = self.projector.build(support_points, semantic_support)
        local_features, global_features = self.projector.sample(cache, query_points)
        local_features = self.local_fusion(local_features)
        return self.decoder(query_points, local_features, global_features)


def _student_semantic_forward(
    model,
    utonia_input: dict,
    support_points: torch.Tensor,
    query_points: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    support_features = model.feature_extractor(
        utonia_input,
        num_support_points=int(support_points.shape[1]),
    )
    semantic_support = model.semantic_adapter(support_features)
    cache = model.semantic_projector.build(support_points, semantic_support)
    local_features, global_features = model.semantic_projector.sample(cache, query_points)
    local_features = model.semantic_local_fusion(local_features)
    output = model.semantic_decoder(query_points, local_features, global_features)
    return output, support_features


def _teacher_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    valid = labels.reshape(-1) != -1
    student = student_logits.reshape(-1, student_logits.shape[-1])[valid]
    teacher = teacher_logits.reshape(-1, teacher_logits.shape[-1])[valid]
    if student.numel() == 0:
        return student_logits.new_zeros(())
    temperature = max(float(temperature), 1e-6)
    return (
        F.kl_div(
            F.log_softmax(student / temperature, dim=-1),
            F.softmax(teacher / temperature, dim=-1),
            reduction="batchmean",
        )
        * temperature**2
    )


def _prediction_confusion(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> np.ndarray:
    predictions = logits.argmax(dim=-1).detach().cpu().numpy()
    targets = labels.detach().cpu().numpy()
    confusion, _ = confusion_matrix_from_predictions(
        predictions,
        targets,
        num_classes=num_classes,
    )
    return confusion


def _build_base_dataset(
    checkpoint_args: dict,
    label_names: list[str],
    split: str,
    limit_samples: int | None,
) -> PartNextUniversalFieldDataset:
    return PartNextUniversalFieldDataset(
        dataset_root=checkpoint_args["dataset_root"],
        split=split,
        categories=checkpoint_args.get("categories"),
        num_support_points=int(checkpoint_args.get("num_support_points", 5000)),
        num_query_surface_points=int(
            checkpoint_args.get("num_query_surface_points", 2048)
        ),
        num_query_occ_points=int(checkpoint_args.get("num_query_occ_points", 1536)),
        num_query_probe_points=int(
            checkpoint_args.get("num_query_probe_points", 1024)
        ),
        balanced_sampling_ratio=float(
            checkpoint_args.get("balanced_sampling_ratio", 0.0)
        ),
        query_sampling_ratio=float(
            checkpoint_args.get("query_sampling_ratio", 0.75)
        ),
        near_surface_ratio=float(checkpoint_args.get("near_surface_ratio", 0.75)),
        near_surface_noise=float(checkpoint_args.get("near_surface_noise", 0.02)),
        uniform_padding=float(checkpoint_args.get("uniform_padding", 0.15)),
        coord_scale=float(checkpoint_args.get("coord_scale", 1.0)),
        center_shift_z=not bool(checkpoint_args.get("disable_z_shift", False)),
        grid_size=float(checkpoint_args.get("grid_size", 0.01)),
        val_ratio=float(checkpoint_args.get("val_ratio", 0.1)),
        test_ratio=float(checkpoint_args.get("test_ratio", 0.0)),
        split_seed=int(checkpoint_args.get("split_seed", 42)),
        limit_samples=limit_samples,
        max_retries=1,
        label_level=str(checkpoint_args.get("label_level", "leaf")),
        alias_config_path=checkpoint_args.get("alias_config"),
        canonical_label_names=label_names,
        ignore_labels=checkpoint_args.get("ignore_labels", ["Other"]),
        rotation_mode="none",
        jitter_std=0.0,
        second_view=False,
        probe_label_radius=float(checkpoint_args.get("probe_label_radius", 0.1)),
    )


def _build_calibration_dataset(
    base_dataset: PartNextUniversalFieldDataset,
    checkpoint_args: dict,
    conditions: list[SupportCondition],
    probabilities: np.ndarray,
    args: argparse.Namespace,
    train: bool,
) -> FixedQuerySupportCalibrationDataset:
    return FixedQuerySupportCalibrationDataset(
        base_dataset=base_dataset,
        conditions=conditions,
        probabilities=probabilities,
        reference_support_count=int(
            args.reference_support_points
            or checkpoint_args.get("num_support_points", 5000)
        ),
        query_count=int(
            args.num_query_surface_points
            or checkpoint_args.get("num_query_surface_points", 2048)
        ),
        network_support_count=int(
            args.network_support_points
            or checkpoint_args.get("num_support_points", 5000)
        ),
        coord_scale=float(checkpoint_args.get("coord_scale", 1.0)),
        center_shift_z=not bool(checkpoint_args.get("disable_z_shift", False)),
        balanced_sampling_ratio=float(
            checkpoint_args.get("balanced_sampling_ratio", 0.0)
        ),
        query_sampling_ratio=float(
            checkpoint_args.get("query_sampling_ratio", 0.75)
        ),
        rotation_mode=args.rotation_mode if train else "none",
        train=train,
        seed=args.seed,
    )


def _configure_student(model) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    modules = [
        model.semantic_adapter,
        model.semantic_local_fusion,
        model.semantic_decoder,
    ]
    parameters: list[torch.nn.Parameter] = []
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True
            parameters.append(parameter)
    return parameters


def _run_train_epoch(
    *,
    model,
    teacher: FrozenSemanticTeacher,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    args: argparse.Namespace,
) -> dict:
    model.train()
    teacher.eval()
    sums: defaultdict[str, float] = defaultdict(float)
    condition_counts: defaultdict[str, int] = defaultdict(int)
    batches = 0
    examples = 0
    amp_enabled = bool(args.amp and device.type == "cuda")

    for batch_index, batch in enumerate(loader):
        if args.max_train_batches is not None and batch_index >= args.max_train_batches:
            break
        batch = move_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            clean_output, clean_support_features = _student_semantic_forward(
                model,
                batch["utonia_input_clean"],
                batch["support_points_clean"],
                batch["query_points_clean"],
            )
            perturbed_output, _ = _student_semantic_forward(
                model,
                batch["utonia_input_perturbed"],
                batch["support_points_perturbed"],
                batch["query_points_perturbed"],
            )
            with torch.no_grad():
                teacher_output = teacher(
                    clean_support_features.detach(),
                    batch["support_points_clean"],
                    batch["query_points_clean"],
                )

            clean_ce = semantic_cross_entropy_loss(
                clean_output["logits"],
                batch["query_labels"],
                label_smoothing=args.label_smoothing,
            )
            perturbed_ce = semantic_cross_entropy_loss(
                perturbed_output["logits"],
                batch["query_labels"],
                label_smoothing=args.label_smoothing,
            )
            support_consistency = embedding_consistency_loss(
                clean_output["embedding"],
                perturbed_output["embedding"],
            )
            teacher_embedding = embedding_consistency_loss(
                clean_output["embedding"],
                teacher_output["embedding"],
            )
            teacher_kl = _teacher_kl_loss(
                clean_output["logits"],
                teacher_output["logits"],
                batch["query_labels"],
                args.teacher_temperature,
            )
            loss = (
                args.ce_weight * 0.5 * (clean_ce + perturbed_ce)
                + args.support_consistency_weight * support_consistency
                + args.teacher_embedding_weight * teacher_embedding
                + args.teacher_kl_weight * teacher_kl
            )

        scaler.scale(loss).backward()
        if args.max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [parameter for group in optimizer.param_groups for parameter in group["params"]],
                args.max_grad_norm,
            )
        scaler.step(optimizer)
        scaler.update()

        labels = batch["query_labels"]
        valid = labels != -1
        clean_accuracy = (
            (clean_output["logits"].argmax(dim=-1)[valid] == labels[valid])
            .float()
            .mean()
        )
        perturbed_accuracy = (
            (perturbed_output["logits"].argmax(dim=-1)[valid] == labels[valid])
            .float()
            .mean()
        )
        values = {
            "loss": loss,
            "clean_ce": clean_ce,
            "perturbed_ce": perturbed_ce,
            "support_consistency": support_consistency,
            "teacher_embedding": teacher_embedding,
            "teacher_kl": teacher_kl,
            "clean_accuracy": clean_accuracy,
            "perturbed_accuracy": perturbed_accuracy,
        }
        batch_size = int(labels.shape[0])
        for key, value in values.items():
            sums[key] += float(value.detach().item()) * batch_size
        for condition in batch["condition"]:
            condition_counts[str(condition)] += 1
        examples += batch_size
        batches += 1

    if batches == 0:
        raise RuntimeError("training epoch processed zero batches")
    return {
        **{key: value / examples for key, value in sums.items()},
        "examples": examples,
        "batches": batches,
        "condition_counts": dict(condition_counts),
    }


@torch.inference_mode()
def _run_validation(
    *,
    model,
    dataset: FixedQuerySupportCalibrationDataset,
    conditions: list[SupportCondition],
    device: torch.device,
    args: argparse.Namespace,
) -> dict:
    model.eval()
    num_classes = len(dataset.base_dataset.canonical_label_names)
    clean_confusions = []
    robust_confusions = []
    condition_confusions: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    amp_enabled = bool(args.amp and device.type == "cuda")
    validation_rows = []

    for trial in range(args.validation_trials):
        for sample_index in range(len(dataset)):
            for condition_index, condition in enumerate(conditions):
                seed = int(
                    args.seed
                    + 10_000_019
                    + trial * 1_000_003
                    + sample_index * 10_007
                    + condition_index * 101
                )
                sample = dataset.load_condition(sample_index, condition, seed)
                batch = move_to_device(calibration_collate_fn([sample]), device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    clean_output, _ = _student_semantic_forward(
                        model,
                        batch["utonia_input_clean"],
                        batch["support_points_clean"],
                        batch["query_points_clean"],
                    )
                    perturbed_output, _ = _student_semantic_forward(
                        model,
                        batch["utonia_input_perturbed"],
                        batch["support_points_perturbed"],
                        batch["query_points_perturbed"],
                    )
                clean_confusion = _prediction_confusion(
                    clean_output["logits"],
                    batch["query_labels"],
                    num_classes,
                )
                perturbed_confusion = _prediction_confusion(
                    perturbed_output["logits"],
                    batch["query_labels"],
                    num_classes,
                )
                clean_confusions.append(clean_confusion)
                robust_confusions.append(perturbed_confusion)
                condition_confusions[condition.label].append(perturbed_confusion)
                condition_metrics = metrics_from_confusion(
                    perturbed_confusion,
                    dataset.base_dataset.canonical_label_names,
                )
                validation_rows.append(
                    {
                        "trial": trial,
                        "sample_index": sample_index,
                        "model_id": sample["model_id"],
                        "condition": condition.label,
                        "seed": seed,
                        "observed_source_points": sample["observed_source_points"],
                        "accuracy": condition_metrics["accuracy"],
                        "mean_iou": condition_metrics["mean_iou"],
                    }
                )

    labels = dataset.base_dataset.canonical_label_names
    clean = metrics_from_confusion(
        merge_confusions(clean_confusions, num_classes),
        labels,
    )
    robust = metrics_from_confusion(
        merge_confusions(robust_confusions, num_classes),
        labels,
    )
    per_condition = {
        condition.label: metrics_from_confusion(
            merge_confusions(condition_confusions[condition.label], num_classes),
            labels,
        )
        for condition in conditions
    }
    return {
        "clean": clean,
        "robust": robust,
        "per_condition": per_condition,
        "rows": validation_rows,
    }


def _checkpoint_with_calibration(
    source_state: dict,
    model,
    *,
    calibration_epoch: int,
    global_step: int,
    args: argparse.Namespace,
    metrics: dict,
) -> dict:
    output = {
        key: value
        for key, value in source_state.items()
        if key not in {"optimizer_state_dict", "scaler_state_dict"}
    }
    projector = dict(source_state["projector_state_dict"])
    projector["semantic_adapter"] = model.semantic_adapter.state_dict()
    projector["semantic_local_fusion"] = model.semantic_local_fusion.state_dict()
    projector["semantic_decoder"] = model.semantic_decoder.state_dict()
    output["projector_state_dict"] = projector
    output["epoch"] = int(source_state.get("epoch", 0)) + int(calibration_epoch)
    output["global_step"] = int(global_step)
    output["calibration"] = {
        "experiment": "fixed_query_support_consistency",
        "calibration_epoch": int(calibration_epoch),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "args": _jsonable(vars(args)),
        "validation_clean_miou": metrics["clean"]["mean_iou"],
        "validation_robust_miou": metrics["robust"]["mean_iou"],
        "validation_per_condition_miou": {
            key: value["mean_iou"] for key, value in metrics["per_condition"].items()
        },
    }
    return output


def calibrate(args: argparse.Namespace) -> dict:
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to mix calibration outputs into non-empty directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    source_state = _safe_load_checkpoint(checkpoint_path)
    label_names = [str(value) for value in source_state["canonical_label_names"]]
    checkpoint_args, alias_manifest = _resolve_checkpoint_args(
        source_state.get("args", {}),
        args,
    )
    conditions, probabilities = normalize_condition_weights(
        args.conditions,
        args.condition_weights,
    )
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    train_base = _build_base_dataset(
        checkpoint_args,
        label_names,
        split="train",
        limit_samples=args.max_train_samples,
    )
    val_base = _build_base_dataset(
        checkpoint_args,
        label_names,
        split="val",
        limit_samples=args.max_val_samples,
    )
    if not train_base.records or not val_base.records:
        raise RuntimeError("calibration requires non-empty train and validation splits")
    train_dataset = _build_calibration_dataset(
        train_base,
        checkpoint_args,
        conditions,
        probabilities,
        args,
        train=True,
    )
    val_dataset = _build_calibration_dataset(
        val_base,
        checkpoint_args,
        conditions,
        probabilities,
        args,
        train=False,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=calibration_collate_fn,
        worker_init_fn=_worker_seed,
        generator=generator,
    )

    model = _load_model(source_state, checkpoint_args, label_names, device)
    teacher = FrozenSemanticTeacher(model).to(device)
    trainable_parameters = _configure_student(model)
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=bool(args.amp and device.type == "cuda"),
    )

    config = {
        "experiment": "fixed_query_support_consistency_calibration",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "epoch": source_state.get("epoch"),
            "global_step": source_state.get("global_step"),
        },
        "alias_config": alias_manifest,
        "canonical_label_names": label_names,
        "conditions": [
            {
                "label": condition.label,
                "kind": condition.kind,
                "value": condition.value,
                "probability": float(probability),
            }
            for condition, probability in zip(conditions, probabilities)
        ],
        "args": _jsonable(vars(args)),
        "checkpoint_args_resolved": _jsonable(checkpoint_args),
        "train_records": [
            {"category": item.category, "model_id": item.model_id}
            for item in train_base.records
        ],
        "validation_records": [
            {"category": item.category, "model_id": item.model_id}
            for item in val_base.records
        ],
        "git": _git_metadata(),
        "runtime": _runtime_metadata(device),
    }
    (output_dir / "config.json").write_text(
        json.dumps(_jsonable(config), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    baseline = _run_validation(
        model=model,
        dataset=val_dataset,
        conditions=conditions,
        device=device,
        args=args,
    )
    print(
        f"[calibration] baseline clean_mIoU={baseline['clean']['mean_iou']:.4f} "
        f"robust_mIoU={baseline['robust']['mean_iou']:.4f}",
        flush=True,
    )

    history = []
    best_score = float("-inf")
    best_metrics = None
    global_step = int(source_state.get("global_step", 0))
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        train_metrics = _run_train_epoch(
            model=model,
            teacher=teacher,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            args=args,
        )
        global_step += int(train_metrics["batches"])
        validation = _run_validation(
            model=model,
            dataset=val_dataset,
            conditions=conditions,
            device=device,
            args=args,
        )
        clean_miou = float(validation["clean"]["mean_iou"])
        robust_miou = float(validation["robust"]["mean_iou"])
        clean_drop = max(
            0.0,
            float(baseline["clean"]["mean_iou"]) - clean_miou - args.clean_drop_tolerance,
        )
        selection_score = (
            robust_miou
            + args.clean_selection_weight * clean_miou
            - args.clean_drop_penalty * clean_drop
        )
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "seconds": time.perf_counter() - epoch_started,
            "selection_score": selection_score,
            "train_loss": train_metrics["loss"],
            "train_clean_ce": train_metrics["clean_ce"],
            "train_perturbed_ce": train_metrics["perturbed_ce"],
            "train_support_consistency": train_metrics["support_consistency"],
            "train_teacher_embedding": train_metrics["teacher_embedding"],
            "train_teacher_kl": train_metrics["teacher_kl"],
            "train_clean_accuracy": train_metrics["clean_accuracy"],
            "train_perturbed_accuracy": train_metrics["perturbed_accuracy"],
            "validation_clean_accuracy": validation["clean"]["accuracy"],
            "validation_clean_miou": clean_miou,
            "validation_robust_accuracy": validation["robust"]["accuracy"],
            "validation_robust_miou": robust_miou,
            "train_condition_counts": json.dumps(
                train_metrics["condition_counts"], sort_keys=True
            ),
        }
        for condition in conditions:
            row[f"validation_{condition.label}_miou"] = validation["per_condition"][
                condition.label
            ]["mean_iou"]
        history.append(row)
        _write_csv(output_dir / "history.csv", history)
        _write_csv(
            output_dir / f"validation_epoch_{epoch:03d}.csv",
            validation["rows"],
        )

        checkpoint = _checkpoint_with_calibration(
            source_state,
            model,
            calibration_epoch=epoch,
            global_step=global_step,
            args=args,
            metrics=validation,
        )
        torch.save(checkpoint, output_dir / "last.pt")
        if selection_score > best_score:
            best_score = selection_score
            best_metrics = validation
            torch.save(checkpoint, output_dir / "best.pt")
        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save(checkpoint, output_dir / f"epoch_{epoch:03d}.pt")
        print(
            f"[calibration] epoch={epoch:03d} loss={train_metrics['loss']:.4f} "
            f"clean_mIoU={clean_miou:.4f} robust_mIoU={robust_miou:.4f} "
            f"score={selection_score:.4f}",
            flush=True,
        )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - started
    result = {
        **config,
        "baseline_validation": baseline,
        "best_validation": best_metrics,
        "best_selection_score": best_score,
        "history": history,
        "timing": {
            "wall_seconds": wall_seconds,
            "epochs": args.epochs,
            "seconds_per_epoch": wall_seconds / max(args.epochs, 1),
        },
        "memory": {
            "peak_allocated_mib": (
                torch.cuda.max_memory_allocated(device) / (1024**2)
                if device.type == "cuda"
                else None
            ),
            "peak_reserved_mib": (
                torch.cuda.max_memory_reserved(device) / (1024**2)
                if device.type == "cuda"
                else None
            ),
        },
        "artifacts": {
            "best_checkpoint": str((output_dir / "best.pt").resolve()),
            "best_checkpoint_sha256": sha256_file(output_dir / "best.pt"),
            "last_checkpoint": str((output_dir / "last.pt").resolve()),
            "last_checkpoint_sha256": sha256_file(output_dir / "last.pt"),
        },
    }
    (output_dir / "results.json").write_text(
        json.dumps(_jsonable(result), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate a semantic field with fixed-query clean/perturbed support "
            "pairs without modifying the source checkpoint."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--alias-config", type=Path, default=None)
    parser.add_argument("--utonia-checkpoint", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--rotation-mode", choices=["none", "z", "so3"], default="so3")
    parser.add_argument("--reference-support-points", type=int, default=None)
    parser.add_argument("--network-support-points", type=int, default=None)
    parser.add_argument("--num-query-surface-points", type=int, default=None)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=[
            "count_512",
            "count_1024",
            "crop_keep_0.75",
            "crop_keep_0.5",
            "crop_keep_0.25",
        ],
    )
    parser.add_argument(
        "--condition-weights",
        nargs="+",
        type=float,
        default=[0.2, 0.2, 0.15, 0.25, 0.2],
    )
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--support-consistency-weight", type=float, default=0.1)
    parser.add_argument("--teacher-embedding-weight", type=float, default=0.05)
    parser.add_argument("--teacher-kl-weight", type=float, default=0.2)
    parser.add_argument("--teacher-temperature", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--clean-selection-weight", type=float, default=0.25)
    parser.add_argument("--clean-drop-tolerance", type=float, default=0.01)
    parser.add_argument("--clean-drop-penalty", type=float, default=5.0)
    parser.add_argument("--validation-trials", type=int, default=1)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=5)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    result = calibrate(args)
    print(
        f"[calibration] complete best_clean_mIoU="
        f"{result['best_validation']['clean']['mean_iou']:.4f} "
        f"best_robust_mIoU={result['best_validation']['robust']['mean_iou']:.4f} "
        f"output={args.output_dir.expanduser().resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
