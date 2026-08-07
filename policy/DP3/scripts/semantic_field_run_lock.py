"""Fail-closed locking and completed-run reuse for semantic-field ablations."""

from __future__ import annotations

import fcntl
import hashlib
import os
import json
import sys
from pathlib import Path
from typing import Any, Callable

from semantic_field_training_state import RESUME_FILENAME, validate_training_state

import torch


_CHECKPOINT_NAMES = ("last.pt", "best.pt", "best_sem.pt")
_IGNORED_CONFIG_KEYS = {
    "resume", "resume_additional_epochs", "resume_from", "auto_resume",
    "num_workers", "epochs", "save_every",
}


def _primitive(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_primitive(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _assert_config_matches(expected_args: Any, actual: dict, source: Path) -> None:
    expected = _primitive(vars(expected_args))
    mismatches = []
    for key, expected_value in expected.items():
        if key in _IGNORED_CONFIG_KEYS:
            continue
        if key not in actual:
            mismatches.append(f"{key}: missing")
            continue
        actual_value = _primitive(actual[key])
        if actual_value != expected_value:
            mismatches.append(
                f"{key}: expected={expected_value!r} actual={actual_value!r}"
            )
    if mismatches:
        raise RuntimeError(
            f"existing run config mismatch in {source}: " + "; ".join(mismatches)
        )


def validate_completed_run(run_dir: Path, expected_args: Any) -> dict:
    """Validate a completed deterministic run without unsafe deserialization."""

    run_dir = Path(run_dir)
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        raise RuntimeError(f"existing run is incomplete: missing {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    _assert_config_matches(expected_args, config, config_path)

    payloads = {}
    for name in _CHECKPOINT_NAMES:
        path = run_dir / name
        if not path.is_file():
            raise RuntimeError(f"existing run is incomplete: missing {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format") != "semantic_field_weights_only_v1":
            raise RuntimeError(f"unexpected safe checkpoint format: {path}")
        if payload.get("training_resume_supported") is not False:
            raise RuntimeError(f"checkpoint is not marked weights-only: {path}")
        if name == "last.pt" and int(payload.get("epoch", -1)) < int(
            expected_args.epochs
        ):
            raise RuntimeError(
                f"existing run is incomplete: last epoch={payload.get('epoch')} "
                f"target={expected_args.epochs}"
            )
        checkpoint_args = payload.get("args")
        if not isinstance(checkpoint_args, dict):
            raise RuntimeError(f"checkpoint args missing from {path}")
        _assert_config_matches(expected_args, checkpoint_args, path)
        payloads[name] = payload
    marker_path = run_dir / "completion.json"
    if marker_path.is_file():
        with marker_path.open("r", encoding="utf-8") as handle:
            marker = json.load(handle)
        if int(marker.get("target_epoch", -1)) != int(expected_args.epochs):
            raise RuntimeError(f"completion target mismatch: {marker_path}")
        expected_hashes = marker.get("sha256")
        if not isinstance(expected_hashes, dict):
            raise RuntimeError(f"completion hashes missing: {marker_path}")
        allowed_hashes = ("config.json",) + _CHECKPOINT_NAMES + (RESUME_FILENAME,)
        unknown = set(expected_hashes).difference(allowed_hashes)
        if unknown:
            raise RuntimeError(f"completion hashes contain unknown files: {sorted(unknown)}")
        for name in expected_hashes:
            actual_hash = _sha256(run_dir / name)
            if expected_hashes.get(name) != actual_hash:
                raise RuntimeError(
                    f"completion SHA256 mismatch for {run_dir / name}"
                )
    return payloads


def write_completion_marker(run_dir: Path, expected_args: Any) -> Path:
    """Atomically write hashes used by future skip validation."""

    run_dir = Path(run_dir)
    files = ("config.json",) + _CHECKPOINT_NAMES
    if (run_dir / RESUME_FILENAME).is_file():
        files += (RESUME_FILENAME,)
    marker = {
        "format": "semantic_field_completion_v1",
        "run_name": str(expected_args.run_name),
        "target_epoch": int(expected_args.epochs),
        "compute_elided_zero_weight_losses": bool(
            getattr(expected_args, "compute_elided_zero_weight_losses", False)
        ),
        "sha256": {name: _sha256(run_dir / name) for name in files},
    }
    path = run_dir / "completion.json"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(marker, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return path


def _validated_resume_path(run_dir: Path, args: Any) -> Path | None:
    explicit = getattr(args, "resume_from", None)
    automatic = bool(getattr(args, "auto_resume", False))
    if explicit is not None and automatic:
        raise SystemExit("choose only one of --resume-from and --auto-resume")
    candidate = (
        Path(explicit).expanduser().resolve()
        if explicit is not None
        else (run_dir / RESUME_FILENAME if automatic else None)
    )
    if candidate is None:
        return None
    if candidate.parent.resolve() != run_dir.resolve():
        raise RuntimeError(
            f"resume checkpoint must belong to exact target run_dir: {candidate}"
        )
    validate_training_state(candidate, args)
    return candidate


def _inject_release_resume_argv(resume_path: Path) -> list[str]:
    filtered: list[str] = []
    index = 0
    argv = sys.argv[1:]
    while index < len(argv):
        token = argv[index]
        if token in {"--auto-resume"}:
            index += 1
            continue
        if token in {"--resume", "--resume-from"}:
            index += 2
            continue
        if token.startswith("--resume=") or token.startswith("--resume-from="):
            index += 1
            continue
        filtered.append(token)
        index += 1
    return [sys.argv[0], *filtered, "--resume", str(resume_path)]


def run_fresh_locked(trainer: Any, run_callable: Callable[[], None] | None = None) -> None:
    """Serialize a run; reuse completion or strictly resume full training state."""

    args = trainer.build_argparser().parse_args()
    if args.resume is not None:
        raise SystemExit(
            "direct --resume is disabled; use --resume-from or --auto-resume"
        )
    if not args.run_name:
        raise SystemExit("locked ablation runs require an explicit --run-name")

    output_dir = Path(args.output_dir).expanduser().resolve()
    run_dir = output_dir / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / f".{args.run_name}.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        print(f"[run-lock] waiting for {lock_path}", flush=True)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        print(f"[run-lock] acquired {lock_path}", flush=True)

        resume_path = None
        if run_dir.exists():
            try:
                validate_completed_run(run_dir, args)
            except RuntimeError:
                if (run_dir / "completion.json").is_file():
                    raise
                resume_path = _validated_resume_path(run_dir, args)
                if resume_path is None:
                    raise RuntimeError(
                        f"existing run is incomplete and resume was not authorized: {run_dir}"
                    )
                payload = validate_training_state(resume_path, args)
                resume_epoch = int(payload["epoch"])
                print(
                    f"[run-lock] verified resumable epoch={resume_epoch}: "
                    f"{resume_path}",
                    flush=True,
                )
            else:
                if not (run_dir / "completion.json").is_file():
                    write_completion_marker(run_dir, args)
                validate_completed_run(run_dir, args)
                print(f"[run-lock] verified complete; skipping {run_dir}", flush=True)
                return
        elif getattr(args, "resume_from", None) is not None:
            raise RuntimeError(
                f"explicit resume target run directory is unavailable: {run_dir}"
            )

        callback = trainer.main if run_callable is None else run_callable
        original_argv = sys.argv
        try:
            if resume_path is not None:
                sys.argv = _inject_release_resume_argv(resume_path)
            callback()
        finally:
            sys.argv = original_argv
        validate_completed_run(run_dir, args)
        write_completion_marker(run_dir, args)
        validate_completed_run(run_dir, args)
        print(f"[run-lock] verified newly completed {run_dir}", flush=True)
