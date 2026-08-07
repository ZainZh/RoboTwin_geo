"""Strict, atomic training-state checkpoints for semantic-field ablations.

Evaluation artifacts stay in the existing weights-only v1 format.  A separate
``resume.pt`` stores optimizer-continuous state and is also loadable with
``weights_only=True``; legacy partial runs therefore cannot be mistaken for
exactly resumable runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


EVALUATION_FORMAT = "semantic_field_weights_only_v1"
TRAINING_STATE_FORMAT = "semantic_field_training_state_v1"
RESUME_FILENAME = "resume.pt"
_CONTROL_KEYS = {
    "resume",
    "resume_additional_epochs",
    "resume_from",
    "auto_resume",
}
# Epochs is a stopping budget only for this trainer (AdamW has no
# epoch-dependent scheduler). Resume validation below permits only shortening.
_OPERATIONAL_KEYS = {"num_workers", "epochs", "save_every"}


def primitive(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, argparse.Namespace):
        return primitive(vars(value))
    if isinstance(value, dict):
        return {str(key): primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [primitive(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"checkpoint metadata contains unsupported type {type(value)!r}")


def normalized_run_config(args: Any) -> dict[str, Any]:
    values = primitive(args)
    if not isinstance(values, dict):
        raise TypeError("training args must normalize to a dictionary")
    return {key: value for key, value in values.items() if key not in _CONTROL_KEYS}


def _config_sha256(config: dict[str, Any]) -> str:
    encoded = json.dumps(
        config, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _scientific_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in config.items() if key not in _OPERATIONAL_KEYS
    }


def run_identity(args: Any) -> dict[str, Any]:
    config = normalized_run_config(args)
    return {
        "run_name": str(config.get("run_name")),
        "seed": int(config["seed"]),
        "config_sha256": _config_sha256(config),
        "config": config,
    }


def capture_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": primitive(random.getstate()),
        "numpy": {
            "bit_generator": str(numpy_state[0]),
            "keys": torch.from_numpy(numpy_state[1].astype(np.int64, copy=True)),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": (
            [state.clone().cpu() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else []
        ),
    }


def _nested_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    return value


def restore_rng_state(state: dict[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    missing = required.difference(state)
    if missing:
        raise RuntimeError(f"resume RNG state is incomplete: missing {sorted(missing)}")
    random.setstate(_nested_tuple(state["python"]))
    numpy_state = state["numpy"]
    keys = numpy_state["keys"]
    if not isinstance(keys, torch.Tensor):
        raise RuntimeError("resume NumPy RNG keys are not a tensor")
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            keys.cpu().numpy().astype(np.uint32, copy=True),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = state["torch_cuda"]
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError("resume checkpoint contains CUDA RNG state but CUDA is unavailable")
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError(
                "resume CUDA device count mismatch: "
                f"checkpoint={len(cuda_states)} current={torch.cuda.device_count()}"
            )
        torch.cuda.set_rng_state_all([item.cpu() for item in cuda_states])


def projector_state_dict(model: Any) -> dict[str, Any]:
    return {
        "semantic_adapter": model.semantic_adapter.state_dict(),
        "pose_adapter": model.pose_adapter.state_dict(),
        "geometric_adapter": model.pose_adapter.state_dict(),
        "semantic_local_fusion": model.semantic_local_fusion.state_dict(),
        "pose_local_fusion": model.pose_local_fusion.state_dict(),
        "geometric_local_fusion": model.pose_local_fusion.state_dict(),
        "occupancy_local_fusion": model.occupancy_local_fusion.state_dict(),
        "semantic_decoder": model.semantic_decoder.state_dict(),
        "pose_decoder": model.pose_decoder.state_dict(),
        "geometric_decoder": model.pose_decoder.state_dict(),
        "occupancy_decoder": model.occupancy_decoder.state_dict(),
    }


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        checked = torch.load(temporary, map_location="cpu", weights_only=True)
        if not isinstance(checked, dict):
            raise RuntimeError(f"checkpoint round-trip is not a dictionary: {temporary}")
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def make_evaluation_payload(
    *,
    model: Any,
    epoch: int,
    global_step: int,
    best_sem_score: float,
    best_pose_score: float,
    best_joint_score: float,
    args: Any,
    canonical_label_names: list[str],
) -> dict[str, Any]:
    payload = {
        "format": EVALUATION_FORMAT,
        "projector_state_dict": projector_state_dict(model),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_val_score": float(best_sem_score),
        "best_sem_score": float(best_sem_score),
        "best_pose_score": float(best_pose_score),
        "best_geo_score": float(best_pose_score),
        "best_joint_score": float(best_joint_score),
        "args": primitive(args),
        "canonical_label_names": [str(item) for item in canonical_label_names],
        "training_resume_supported": False,
    }
    if not model.feature_extractor.freeze_encoder:
        payload["encoder_state_dict"] = model.feature_extractor.encoder.state_dict()
    return payload


def make_training_state_payload(
    evaluation_payload: dict[str, Any],
    *,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    args: Any,
) -> dict[str, Any]:
    return {
        **evaluation_payload,
        "format": TRAINING_STATE_FORMAT,
        "training_resume_supported": True,
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "rng_state_dict": capture_rng_state(),
        "run_identity": run_identity(args),
        "resume_epoch_boundary": True,
    }


def load_training_state(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise RuntimeError(f"resume checkpoint is unavailable: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise RuntimeError(f"resume checkpoint is not a dictionary: {checkpoint_path}")
    if payload.get("format") != TRAINING_STATE_FORMAT:
        raise RuntimeError(
            f"checkpoint is not an exact training-state artifact: {checkpoint_path}; "
            f"expected {TRAINING_STATE_FORMAT!r}, got {payload.get('format')!r}"
        )
    if payload.get("training_resume_supported") is not True:
        raise RuntimeError(f"checkpoint is not marked resumable: {checkpoint_path}")
    required = {
        "projector_state_dict",
        "optimizer_state_dict",
        "scaler_state_dict",
        "rng_state_dict",
        "epoch",
        "global_step",
        "best_sem_score",
        "best_pose_score",
        "best_joint_score",
        "args",
        "canonical_label_names",
        "run_identity",
    }
    missing = required.difference(payload)
    if missing:
        raise RuntimeError(
            f"resume checkpoint is incomplete ({checkpoint_path}): missing {sorted(missing)}"
        )
    if payload.get("resume_epoch_boundary") is not True:
        raise RuntimeError(f"resume checkpoint is not at an epoch boundary: {checkpoint_path}")
    return payload


def validate_training_state(path: str | Path, expected_args: Any) -> dict[str, Any]:
    checkpoint_path = Path(path).expanduser().resolve()
    payload = load_training_state(checkpoint_path)
    actual = payload.get("run_identity")
    if not isinstance(actual, dict) or not isinstance(actual.get("config"), dict):
        raise RuntimeError(f"stored run identity is malformed: {checkpoint_path}")

    stored_config = primitive(actual["config"])
    stored_sha = str(actual.get("config_sha256", ""))
    if stored_sha != _config_sha256(stored_config):
        raise RuntimeError(
            f"stored run identity SHA mismatch in resume checkpoint: {checkpoint_path}"
        )
    if str(actual.get("run_name")) != str(stored_config.get("run_name")):
        raise RuntimeError(f"stored run_name is not self-consistent: {checkpoint_path}")
    if int(actual.get("seed", -1)) != int(stored_config.get("seed", -2)):
        raise RuntimeError(f"stored seed is not self-consistent: {checkpoint_path}")
    checkpoint_args = payload.get("args")
    if normalized_run_config(checkpoint_args) != stored_config:
        raise RuntimeError(f"stored args and run identity disagree: {checkpoint_path}")

    expected = run_identity(expected_args)
    expected_config = expected["config"]
    stored_target = int(stored_config["epochs"])
    expected_target = int(expected_config["epochs"])
    if expected_target > stored_target:
        raise RuntimeError(
            "resume target epoch may only stay fixed or decrease: "
            f"checkpoint_target={stored_target} requested_target={expected_target}"
        )
    if _scientific_config(stored_config) != _scientific_config(expected_config):
        raise RuntimeError(
            "resume run identity mismatch in scientific config: "
            f"checkpoint={checkpoint_path}; expected={expected_config}; "
            f"actual={stored_config}"
        )
    if int(payload["epoch"]) >= expected_target:
        raise RuntimeError(
            f"resume checkpoint already reached target epoch {payload['epoch']}"
        )
    return payload


def make_restore_checkpoint(original: Callable[..., Any]) -> Callable[..., Any]:
    def restore(checkpoint: dict[str, Any], **kwargs: Any) -> Any:
        if checkpoint.get("format") != TRAINING_STATE_FORMAT:
            raise RuntimeError(
                "safe resume requires the strict full-state checkpoint format"
            )
        result = original(checkpoint, **kwargs)
        restore_rng_state(checkpoint["rng_state_dict"])
        return result

    return restore


def install_resume_cli_and_loader(trainer: Any) -> None:
    original_build_argparser = trainer.build_argparser

    def build_argparser():
        parser = original_build_argparser()
        parser.add_argument(
            "--resume-from",
            type=Path,
            default=None,
            help=(
                "Strictly resume this same run from a full-state resume.pt "
                "artifact"
            ),
        )
        parser.add_argument(
            "--auto-resume",
            action="store_true",
            help=(
                "Resume run_dir/resume.pt when the exact run directory is "
                "incomplete"
            ),
        )
        return parser

    trainer.build_argparser = build_argparser
    trainer._load_checkpoint = load_training_state
    trainer.restore_checkpoint = make_restore_checkpoint(
        trainer.restore_checkpoint
    )
    trainer.SEMANTIC_RESUME_SUPPORT_ENABLED = True
