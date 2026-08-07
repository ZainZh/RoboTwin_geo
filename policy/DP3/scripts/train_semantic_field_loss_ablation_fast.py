#!/usr/bin/env python3
"""Fresh semantic-field ablation training with the isolated semantic fast path."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from semantic_field_fast_path import install_semantic_only_fast_path
from semantic_field_run_lock import run_fresh_locked
from semantic_field_training_state import install_resume_cli_and_loader
from train_semantic_field_loss_ablation_safe import (
    DEFAULT_RELEASE_ROOT,
    DEFAULT_UTONIA_ROOT,
    _reject_resume,
    save_weights_only_checkpoint,
)


def _require_explicit_semantic_mode(argv: list[str]) -> None:
    value = None
    for index, token in enumerate(argv):
        if token.startswith("--train-mode="):
            value = token.split("=", 1)[1]
        elif token == "--train-mode":
            if index + 1 >= len(argv):
                raise SystemExit("--train-mode requires a value")
            value = argv[index + 1]
    if value != "semantic":
        raise SystemExit(
            "The fast entry is semantic-only; pass --train-mode semantic explicitly."
        )


def main() -> None:
    _reject_resume(sys.argv[1:])
    _require_explicit_semantic_mode(sys.argv[1:])
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

    install_semantic_only_fast_path(trainer)
    install_resume_cli_and_loader(trainer)
    trainer.save_checkpoint = save_weights_only_checkpoint
    if getattr(trainer, "SEMANTIC_FAST_PATH_ENABLED", False) is not True:
        raise RuntimeError("semantic fast path marker was not installed")
    run_fresh_locked(trainer)


if __name__ == "__main__":
    main()
