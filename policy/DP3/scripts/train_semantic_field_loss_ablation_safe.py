#!/usr/bin/env python3
"""Run the released semantic-field trainer with weights-only checkpoints.

The release trainer serializes optimizer/scaler state and has a legacy pickle
fallback for resume.  Paper ablations are fresh runs, so this wrapper rejects
resume and replaces only the checkpoint writer.  The actual dataset, model,
loss computation, optimizer, validation, and checkpoint selection remain the
released implementation.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Mapping

from semantic_field_training_state import (
    EVALUATION_FORMAT,
    RESUME_FILENAME,
    atomic_torch_save,
    install_resume_cli_and_loader,
    make_evaluation_payload,
    make_training_state_payload,
)

from semantic_field_run_lock import run_fresh_locked


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RELEASE_ROOT = REPO_ROOT / "include" / "3d_semantic_train" / "semantic_field_release"
DEFAULT_UTONIA_ROOT = REPO_ROOT / "include" / "3d_semantic_train" / "include" / "Utonia"
FORMAT_NAME = EVALUATION_FORMAT


def _argument_value(argv: list[str], name: str) -> str | None:
    value = None
    for index, token in enumerate(argv):
        if token.startswith(f"{name}="):
            value = token.split("=", 1)[1]
        elif token == name:
            if index + 1 >= len(argv):
                raise SystemExit(f"{name} requires a value")
            value = argv[index + 1]
    return value


def _semantic_fast_path_requested(
    argv: list[str], environ: Mapping[str, str] = os.environ
) -> bool:
    if _argument_value(argv, "--train-mode") != "semantic":
        return False
    value = environ.get("SEMANTIC_FAST_PATH", "1").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise SystemExit(
        "SEMANTIC_FAST_PATH must be one of 1/0, true/false, yes/no, on/off"
    )


def _install_default_semantic_fast_path(
    trainer: Any,
    argv: list[str],
    environ: Mapping[str, str] = os.environ,
) -> bool:
    """Install the semantic fast path for future safe-entry ablations."""
    if not _semantic_fast_path_requested(argv, environ):
        return False

    # Keep this import lazy: the dedicated fast entry imports safe checkpoint
    # helpers, so a top-level import here would create a cycle.
    from semantic_field_fast_path import (  # noqa: PLC0415
        install_semantic_only_fast_path,
        semantic_only_encode_support,
        semantic_only_forward,
    )

    if getattr(trainer, "SEMANTIC_FAST_PATH_ENABLED", False) is not True:
        install_semantic_only_fast_path(trainer)
    installed = (
        getattr(trainer, "SEMANTIC_FAST_PATH_ENABLED", False) is True
        and trainer.UtoniaUniversalFieldNet.encode_support
        is semantic_only_encode_support
        and trainer.UtoniaUniversalFieldNet.forward is semantic_only_forward
    )
    if not installed:
        raise RuntimeError("safe entry could not verify semantic fast-path identities")
    print("[semantic-fast-path] safe-entry marker and identities verified", flush=True)
    return True




def save_weights_only_checkpoint(
    path: Path,
    *,
    model,
    optimizer,
    scaler,
    epoch: int,
    global_step: int,
    best_sem_score: float,
    best_pose_score: float,
    best_joint_score: float,
    args: argparse.Namespace,
    canonical_label_names: list[str],
) -> None:
    """Write a v1 evaluator checkpoint and full resume state beside ``last.pt``."""
    payload = make_evaluation_payload(
        model=model, epoch=epoch, global_step=global_step,
        best_sem_score=best_sem_score, best_pose_score=best_pose_score,
        best_joint_score=best_joint_score, args=args,
        canonical_label_names=canonical_label_names,
    )
    path = Path(path)
    atomic_torch_save(payload, path)
    if path.name == "last.pt":
        resume_payload = make_training_state_payload(
            payload, optimizer=optimizer, scaler=scaler, args=args
        )
        resume_path = path.with_name(RESUME_FILENAME)
        atomic_torch_save(resume_payload, resume_path)
        print(
            f"[checkpoint] Atomic optimizer-continuous state: {resume_path}",
            flush=True,
        )


def _reject_resume(argv: list[str]) -> None:
    for token in argv:
        if token == "--resume" or token.startswith("--resume="):
            raise SystemExit(
                "Direct release --resume is disabled; use strict --resume-from "
                "or --auto-resume with a full-state resume.pt artifact."
            )


def main() -> None:
    _reject_resume(sys.argv[1:])
    release_root = Path(
        os.environ.get("SEMANTIC_FIELD_RELEASE_ROOT", str(DEFAULT_RELEASE_ROOT))
    ).expanduser().resolve()
    trainer_path = release_root / "train" / "train_utonia_universal_field.py"
    if not trainer_path.is_file():
        raise FileNotFoundError(f"release trainer is unavailable: {trainer_path}")
    utonia_root = Path(
        os.environ.get("UTONIA_ROOT", str(DEFAULT_UTONIA_ROOT))
    ).expanduser().resolve()
    if not utonia_root.is_dir():
        raise FileNotFoundError(
            f"Utonia checkout is unavailable: {utonia_root}; set UTONIA_ROOT"
        )
    sys.path.insert(0, str(utonia_root))
    sys.path.insert(0, str(release_root))
    from train import train_utonia_universal_field as trainer  # noqa: PLC0415

    _install_default_semantic_fast_path(trainer, sys.argv[1:])
    install_resume_cli_and_loader(trainer)
    trainer.save_checkpoint = save_weights_only_checkpoint
    run_fresh_locked(trainer)


if __name__ == "__main__":
    main()
