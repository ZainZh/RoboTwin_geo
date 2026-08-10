#!/usr/bin/env python3
"""Frozen protocol and fail-closed artifact audit for Hammer stage42.

This module never launches training.  It centralizes the independent-test
matrix identity so the local queue, remote supervisor queue, launcher tests,
and final artifact audit cannot silently disagree about a run.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
EPOCHS = 1750
ARTIFACT_EPOCH_TAG = 4000
SPLIT_SEED = 424242
VAL_RATIO = 0.1
TEST_RATIO = 0.2
BATCH_SIZE = 6
LOCAL_SEEDS = ("20260805", "20260807")
REMOTE_SEEDS = ("20260806",)
VARIANT_LOSSES = {
    "full_fixed": (1.0, 0.2, 0.1),
    "no_ce": (0.0, 0.2, 0.1),
    "no_supcon": (1.0, 0.0, 0.1),
    "no_consistency": (1.0, 0.2, 0.0),
    "ce_only": (1.0, 0.0, 0.0),
}
VARIANTS = tuple(VARIANT_LOSSES)
DEFAULT_LOCAL_TEMPLATE = str(
    REPO_ROOT
    / "outputs/paper_revision/hammer_stage42_independent_local4090_"
    "seed{seed}_e4000_split424242_20260806"
)
DEFAULT_REMOTE_TEMPLATE = (
    "/workspace/RoboTwin_geo/outputs/paper_revision/"
    "hammer_stage42_independent_remote6000ada_"
    "seed{seed}_e4000_split424242_20260806"
)


# Every scientific/default optimizer field in the current formal resolved
# config is frozen here.  Host paths, device, output directory, training seed,
# and num_workers are operational fields audited separately.
FROZEN_CONFIG: dict[str, Any] = {
    "categories": ["Hammer"],
    "split_seed": SPLIT_SEED,
    "val_ratio": VAL_RATIO,
    "test_ratio": TEST_RATIO,
    "limit_train_samples": None,
    "limit_val_samples": None,
    "num_support_points": 5000,
    "num_query_surface_points": 2048,
    "num_query_occ_points": 1536,
    "balanced_sampling_ratio": 0.0,
    "query_sampling_ratio": 0.75,
    "near_surface_ratio": 0.75,
    "near_surface_noise": 0.02,
    "uniform_padding": 0.15,
    "coord_scale": 1.0,
    "disable_z_shift": False,
    "grid_size": 0.01,
    "label_level": "config",
    "ignore_labels": ["Other"],
    "rotation_mode": "so3",
    "jitter_std": 0.005,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "train_mode": "semantic",
    "lr": 0.001,
    "weight_decay": 0.0001,
    "adapter_hidden_dim": 256,
    "branch_dim": 256,
    "sem_embedding_dim": 128,
    "geo_embedding_dim": 128,
    "num_anchors": 256,
    "rbf_sigma": 0.12,
    "triplane_resolution": 64,
    "triplane_padding": 0.05,
    "sem_contrastive_temperature": 0.1,
    "sem_contrastive_max_points": 2048,
    "sem_label_smoothing": 0.0,
    "geo_dense_correspondence_weight": 0.0,
    "geo_dense_correspondence_temperature": 0.1,
    "geo_dense_correspondence_max_points": 1024,
    "geo_spatial_tolerance_radius": 0.03,
    "geo_metric_weight": 0.0,
    "geo_k": 16,
    "geo_same_part_scale": 0.15,
    "geo_negative_margin": 0.2,
    "geo_consistency_weight": 0.0,
    "occ_weight": 0.0,
    "best_sem_occ_weight": 0.1,
    "best_geo_metric_weight": 0.1,
    "best_joint_geo_dense_weight": 0.5,
    "best_joint_geo_metric_weight": 0.1,
    "best_joint_geo_weight": 0.5,
    "utonia_repo_id": "Pointcept/Utonia",
    "utonia_upcast_levels": 0,
    "finetune_utonia": False,
    "amp": True,
    "save_every": EPOCHS,
    "max_train_batches": None,
    "max_val_batches": None,
    "wandb_project": "semantic-field",
    "wandb_entity": None,
    "wandb_mode": "disabled",
    "wandb_tags": [],
    "resume": None,
    "resume_additional_epochs": 0,
    "resume_from": None,
    "auto_resume": True,
    "compute_elided_zero_weight_losses": True,
    "canonical_label_names": ["Handle", "Head"],
}


@dataclasses.dataclass(frozen=True)
class Stage42Task:
    role: str
    seed: str
    variant: str
    output_root: Path
    num_workers: int
    program: str = ""

    @property
    def run_name(self) -> str:
        return f"hammer_{self.variant}_e{ARTIFACT_EPOCH_TAG}_split{SPLIT_SEED}_seed{self.seed}"

    @property
    def run_dir(self) -> Path:
        return self.output_root / self.run_name

    @property
    def lock_path(self) -> Path:
        return self.output_root / f".{self.run_name}.lock"

    @property
    def completion(self) -> Path:
        return self.run_dir / "completion.json"

    @property
    def outer_completion(self) -> Path:
        return self.output_root / "completed" / f"{self.run_name}.complete"

    @property
    def outer_manifest(self) -> Path:
        return self.output_root / "manifests" / f"{self.run_name}.sha256"

    @property
    def key(self) -> str:
        return f"{self.role}/{self.seed}/{self.variant}"


def seeds_for_role(role: str) -> tuple[str, ...]:
    if role == "local":
        return LOCAL_SEEDS
    if role == "remote":
        return REMOTE_SEEDS
    raise ValueError(f"unknown stage42 role: {role!r}")


def default_template(role: str) -> str:
    if role == "local":
        return DEFAULT_LOCAL_TEMPLATE
    if role == "remote":
        return DEFAULT_REMOTE_TEMPLATE
    raise ValueError(f"unknown stage42 role: {role!r}")


def build_tasks(
    role: str,
    *,
    output_template: str | None = None,
    seeds: Iterable[str] | None = None,
    variants: Iterable[str] | None = None,
    num_workers: int | None = None,
) -> list[Stage42Task]:
    allowed_seeds = seeds_for_role(role)
    selected_seeds = tuple(allowed_seeds if seeds is None else seeds)
    selected_variants = tuple(VARIANTS if variants is None else variants)
    unknown_seeds = set(selected_seeds).difference(allowed_seeds)
    unknown_variants = set(selected_variants).difference(VARIANTS)
    if unknown_seeds:
        raise ValueError(f"seeds are assigned to another machine: {sorted(unknown_seeds)}")
    if unknown_variants:
        raise ValueError(f"unknown variants: {sorted(unknown_variants)}")
    if len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError("duplicate training seeds are not allowed")
    if len(set(selected_variants)) != len(selected_variants):
        raise ValueError("duplicate variants are not allowed")
    workers = (4 if role == "local" else 8) if num_workers is None else num_workers
    if workers < 0:
        raise ValueError("num_workers must be non-negative")
    template = default_template(role) if output_template is None else output_template
    tasks = []
    for variant in selected_variants:
        for seed in selected_seeds:
            program = (
                f"hammer-stage42-local-{variant}-seed{seed}.service"
                if role == "local"
                else f"hammer_stage42_remote_{variant}_seed{seed}"
            )
            tasks.append(
                Stage42Task(
                    role=role,
                    seed=seed,
                    variant=variant,
                    output_root=Path(template.format(seed=seed)),
                    num_workers=workers,
                    program=program,
                )
            )
    roots = [task.output_root for task in tasks]
    if len({str(root) for root in roots}) != len(set(selected_seeds)):
        raise ValueError("output template must resolve to one unique root per seed")
    return tasks


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_values(values: dict[str, Any], expected: dict[str, Any], source: Path) -> None:
    mismatches = []
    for key, expected_value in expected.items():
        if key not in values:
            mismatches.append(f"{key}=<missing>")
        elif values[key] != expected_value:
            mismatches.append(f"{key}={values[key]!r} expected={expected_value!r}")
    if mismatches:
        raise RuntimeError(f"stage42 identity mismatch in {source}: " + "; ".join(mismatches))


def expected_config(task: Stage42Task) -> dict[str, Any]:
    ce, supcon, consistency = VARIANT_LOSSES[task.variant]
    return {
        **FROZEN_CONFIG,
        "num_workers": task.num_workers,
        "sem_ce_weight": ce,
        "sem_contrastive_weight": supcon,
        "sem_consistency_weight": consistency,
        "seed": int(task.seed),
        "run_name": task.run_name,
        "output_dir": str(task.output_root),
    }

def expected_checkpoint_args(task: Stage42Task) -> dict[str, Any]:
    """Checkpoint args omit labels stored in the checkpoint top level."""
    return {
        key: value for key, value in expected_config(task).items()
        if key not in {"canonical_label_names", "resume", "epochs", "save_every"}
    }


def expected_audited_config(task: Stage42Task) -> dict[str, Any]:
    """Scientific config plus fixed controls, excluding the resolved resume source."""
    return {
        key: value for key, value in expected_config(task).items() if key != "resume"
    }


def _require_same_run_resume(values: dict[str, Any], task: Stage42Task, source: Path) -> None:
    if "resume" not in values:
        raise RuntimeError(f"stage42 identity mismatch in {source}: resume=<missing>")
    resolved = values["resume"]
    if resolved is None:
        return
    expected = (task.run_dir / "resume.pt").resolve()
    if Path(resolved).expanduser().resolve() != expected:
        raise RuntimeError(
            f"stage42 identity mismatch in {source}: resume={resolved!r} expected None or {str(expected)!r}"
        )


def _require_checkpoint_budget(
    values: dict[str, Any], source: Path, *, allow_historical_target: bool
) -> None:
    try:
        epochs = int(values["epochs"])
        save_every = int(values["save_every"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"checkpoint budget metadata missing from {source}") from error
    allowed = {EPOCHS, ARTIFACT_EPOCH_TAG} if allow_historical_target else {EPOCHS}
    if epochs not in allowed or save_every != epochs:
        raise RuntimeError(
            f"stage42 checkpoint budget mismatch in {source}: "
            f"epochs={epochs}, save_every={save_every}, allowed={sorted(allowed)}"
        )

def _load_weights_only(path: Path) -> dict[str, Any]:
    import torch  # Imported only by explicit artifact audits, never by queues.

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise RuntimeError(f"checkpoint is not a dictionary: {path}")
    return payload


def audit_completed_run(task: Stage42Task, *, require_outer: bool = False) -> dict[str, Any]:
    """Verify one completed run without unsafe deserialization."""

    if not task.lock_path.is_file():
        raise RuntimeError(f"missing per-run lock evidence: {task.lock_path}")
    config_path = task.run_dir / "config.json"
    if not config_path.is_file() or not task.completion.is_file():
        raise RuntimeError(f"incomplete stage42 run: {task.run_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _require_values(config, expected_audited_config(task), config_path)
    _require_same_run_resume(config, task, config_path)

    marker = json.loads(task.completion.read_text(encoding="utf-8"))
    _require_values(
        marker,
        {
            "format": "semantic_field_completion_v1",
            "run_name": task.run_name,
            "target_epoch": EPOCHS,
            "compute_elided_zero_weight_losses": True,
        },
        task.completion,
    )
    required_files = ("config.json", "last.pt", "best.pt", "best_sem.pt", "resume.pt")
    hashes = marker.get("sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(required_files):
        raise RuntimeError(
            f"completion hash set mismatch in {task.completion}: "
            f"expected={sorted(required_files)} actual={sorted(hashes) if isinstance(hashes, dict) else hashes!r}"
        )
    for name in required_files:
        path = task.run_dir / name
        if not path.is_file() or hashes[name] != _sha256(path):
            raise RuntimeError(f"missing or SHA-mismatched stage42 artifact: {path}")

    checkpoint_epochs: dict[str, int] = {}
    for name in ("last.pt", "best.pt", "best_sem.pt"):
        path = task.run_dir / name
        payload = _load_weights_only(path)
        _require_values(
            payload,
            {
                "format": "semantic_field_weights_only_v1",
                "training_resume_supported": False,
                "canonical_label_names": ["Handle", "Head"],
            },
            path,
        )
        args = payload.get("args")
        if not isinstance(args, dict):
            raise RuntimeError(f"checkpoint args missing from {path}")
        _require_values(args, expected_checkpoint_args(task), path)
        _require_same_run_resume(args, task, path)
        checkpoint_epochs[name] = int(payload.get("epoch", -1))
        _require_checkpoint_budget(args, path, allow_historical_target=name != "last.pt")
    if checkpoint_epochs["last.pt"] != EPOCHS:
        raise RuntimeError(f"last checkpoint is not exactly e{EPOCHS}: {task.run_dir}")
    for name in ("best.pt", "best_sem.pt"):
        if not 1 <= checkpoint_epochs[name] <= EPOCHS:
            raise RuntimeError(f"invalid {name} epoch for fixed e{EPOCHS} budget: {task.run_dir}")

    resume_path = task.run_dir / "resume.pt"
    resume = _load_weights_only(resume_path)
    _require_values(
        resume,
        {
            "format": "semantic_field_training_state_v1",
            "training_resume_supported": True,
            "resume_epoch_boundary": True,
            "canonical_label_names": ["Handle", "Head"],
        },
        resume_path,
    )
    resume_args = resume.get("args")
    identity = resume.get("run_identity")
    if not isinstance(resume_args, dict) or not isinstance(identity, dict):
        raise RuntimeError(f"resume identity missing from {resume_path}")
    _require_values(resume_args, expected_checkpoint_args(task), resume_path)
    _require_same_run_resume(resume_args, task, resume_path)
    _require_checkpoint_budget(
        resume_args, resume_path, allow_historical_target=False
    )
    _require_values(
        identity,
        {"run_name": task.run_name, "seed": int(task.seed)},
        resume_path,
    )
    identity_config = identity.get("config")
    if not isinstance(identity_config, dict):
        raise RuntimeError(f"resume config identity missing from {resume_path}")
    identity_expected = {
        key: value
        for key, value in expected_checkpoint_args(task).items()
        if key not in {"resume", "resume_additional_epochs", "resume_from", "auto_resume"}
    }
    _require_values(identity_config, identity_expected, resume_path)
    _require_checkpoint_budget(identity_config, resume_path, allow_historical_target=False)
    if int(resume.get("epoch", -1)) != EPOCHS:
        raise RuntimeError(f"resume state is not exactly e{EPOCHS}: {resume_path}")

    if require_outer:
        if not task.outer_completion.is_file() or not task.outer_manifest.is_file():
            raise RuntimeError(f"remote outer completion is missing for {task.run_name}")

    return {
        "status": "verified_complete",
        "task": task.key,
        "run_dir": str(task.run_dir),
        "checkpoint_epochs": checkpoint_epochs,
        "resume_epoch": int(resume["epoch"]),
        "completion_sha256": _sha256(task.completion),
        "paper_claim_split": "independent_test",
    }


def task_status(task: Stage42Task) -> str:
    if task.completion.is_file():
        try:
            audit_completed_run(task)
        except Exception:
            return "blocked_invalid_completion"
        return "verified_complete"
    if task.run_dir.exists():
        return "incomplete_resumable_only_after_identity_audit"
    return "pending"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("local", "remote", "all"), default="all")
    parser.add_argument("--seed", action="append")
    parser.add_argument("--variant", action="append", choices=VARIANTS)
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Audit the realized worker count; defaults to local=0, remote=8.",
    )
    parser.add_argument("--mode", choices=("plan", "audit"), default="plan")
    parser.add_argument("--require-outer", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roles = ("local", "remote") if args.role == "all" else (args.role,)
    tasks: list[Stage42Task] = []
    for role in roles:
        allowed = set(seeds_for_role(role))
        selected = None if args.seed is None else [seed for seed in args.seed if seed in allowed]
        if args.seed is not None and not selected:
            continue
        workers = args.num_workers
        if workers is None:
            workers = 0 if role == "local" else 8
        tasks.extend(build_tasks(role, seeds=selected, variants=args.variant, num_workers=workers))
    if args.seed is not None:
        known = {task.seed for task in tasks}
        unknown = set(args.seed).difference(known)
        if unknown:
            raise SystemExit(f"seeds do not match selected role: {sorted(unknown)}")
    reports = []
    for task in tasks:
        if args.mode == "audit":
            reports.append(audit_completed_run(task, require_outer=args.require_outer))
        else:
            reports.append(
                {
                    "task": task.key,
                    "run_name": task.run_name,
                    "run_dir": str(task.run_dir),
                    "num_workers": task.num_workers,
                    "status": task_status(task),
                }
            )
    print(json.dumps({"format": "hammer_stage42_protocol_v1", "tasks": reports}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
