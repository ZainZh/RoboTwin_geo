#!/usr/bin/env python3
"""Audit, wait for, and plan/run the Hammer loss-ablation evaluation matrix.

The default mode is audit-only.  GPU evaluators can run only with both
``--mode run`` and ``--confirm-run-evaluators``.  Waiting for checkpoints never
implicitly changes the mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_ROOT = REPO_ROOT / "policy" / "DP3" / "scripts"
VARIANTS = ("full_fixed", "no_ce", "no_supcon", "no_consistency", "ce_only")
EXPECTED_WEIGHTS = {
    "full_fixed": (1.0, 0.2, 0.1),
    "no_ce": (0.0, 0.2, 0.1),
    "no_supcon": (1.0, 0.0, 0.1),
    "no_consistency": (1.0, 0.2, 0.0),
    "ce_only": (1.0, 0.0, 0.0),
}
FORMAT_NAME = "semantic_field_weights_only_v1"
EVALUATION_PROTOCOLS = {
    "legacy_val": {
        "split": "val",
        "status": "diagnostic_validation_checkpoint_selected_on_same_split",
        "default_val_ratio": 0.1,
        "default_test_ratio": 0.0,
        "paper_claim_eligible": False,
    },
    "independent_test": {
        "split": "test",
        "status": "held_out_test",
        "default_val_ratio": 0.1,
        "default_test_ratio": 0.1,
        "paper_claim_eligible": True,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_seed_roots(values: list[str]) -> dict[int, Path]:
    roots: dict[int, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--seed-root must be SEED=PATH, got {value!r}")
        seed_text, path_text = value.split("=", 1)
        seed = int(seed_text)
        if seed in roots:
            raise ValueError(f"duplicate seed root: {seed}")
        roots[seed] = Path(path_text).expanduser().resolve()
    if len(roots) != 3:
        raise ValueError(f"exactly three --seed-root entries are required, got {len(roots)}")
    return roots


def run_dir(root: Path, seed: int, variant: str, epochs: int, split_seed: int) -> Path:
    return root / f"hammer_{variant}_e{epochs}_split{split_seed}_seed{seed}"


def expected_paths(
    seed_roots: dict[int, Path], epochs: int, split_seed: int
) -> dict[tuple[int, str], dict[str, Path]]:
    paths = {}
    for seed, root in sorted(seed_roots.items()):
        for variant in VARIANTS:
            directory = run_dir(root, seed, variant, epochs, split_seed)
            paths[(seed, variant)] = {
                "run_dir": directory,
                "config": directory / "config.json",
                "best_sem": directory / "best_sem.pt",
                "last": directory / "last.pt",
            }
    return paths


def missing_paths(matrix_paths: dict[tuple[int, str], dict[str, Path]]) -> list[Path]:
    missing = []
    for entry in matrix_paths.values():
        for key in ("config", "best_sem", "last"):
            if not entry[key].is_file():
                missing.append(entry[key])
    return missing


def wait_until_complete(
    matrix_paths: dict[tuple[int, str], dict[str, Path]],
    *,
    interval_seconds: float,
    timeout_seconds: float | None,
) -> None:
    started = time.monotonic()
    while True:
        missing = missing_paths(matrix_paths)
        if not missing:
            return
        elapsed = time.monotonic() - started
        print(
            json.dumps(
                {
                    "status": "waiting",
                    "elapsed_seconds": round(elapsed, 1),
                    "missing_files": len(missing),
                    "first_missing": str(missing[0]),
                }
            ),
            flush=True,
        )
        if timeout_seconds is not None and elapsed >= timeout_seconds:
            raise TimeoutError(f"timed out with {len(missing)} required files missing")
        time.sleep(min(max(interval_seconds, 0.1), 60.0))


def _safe_load(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise RuntimeError(f"strict weights-only load failed: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint must contain a dictionary: {path}")
    return payload


def _expect_equal(actual: Any, expected: Any, label: str) -> None:
    if isinstance(expected, float):
        if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{label}: expected {expected!r}, got {actual!r}")
    elif actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, got {actual!r}")


def audit_matrix(
    seed_roots: dict[int, Path],
    *,
    epochs: int = 4000,
    artifact_epoch_tag: int | None = None,
    split_seed: int = 42,
    evaluation_protocol: str = "legacy_val",
    expected_val_ratio: float | None = None,
    expected_test_ratio: float | None = None,
) -> dict[str, Any]:
    if evaluation_protocol not in EVALUATION_PROTOCOLS:
        raise ValueError(f"unknown evaluation protocol: {evaluation_protocol}")
    protocol_spec = EVALUATION_PROTOCOLS[evaluation_protocol]
    val_ratio = float(
        protocol_spec["default_val_ratio"]
        if expected_val_ratio is None
        else expected_val_ratio
    )
    test_ratio = float(
        protocol_spec["default_test_ratio"]
        if expected_test_ratio is None
        else expected_test_ratio
    )
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"expected val ratio must be in (0, 1), got {val_ratio}")
    if not 0.0 <= test_ratio < 1.0 or val_ratio + test_ratio >= 1.0:
        raise ValueError(
            f"invalid expected val/test ratios: val={val_ratio}, test={test_ratio}"
        )
    if evaluation_protocol == "legacy_val" and test_ratio != 0.0:
        raise ValueError("legacy_val protocol requires expected test ratio 0")
    if evaluation_protocol == "independent_test" and test_ratio <= 0.0:
        raise ValueError("independent_test protocol requires a positive test ratio")

    artifact_tag = epochs if artifact_epoch_tag is None else int(artifact_epoch_tag)
    if artifact_tag <= 0:
        raise ValueError(f"artifact epoch tag must be positive, got {artifact_tag}")

    matrix_paths = expected_paths(seed_roots, artifact_tag, split_seed)
    missing = missing_paths(matrix_paths)
    if missing:
        preview = "\n".join(str(path) for path in missing[:8])
        raise FileNotFoundError(f"matrix is incomplete ({len(missing)} files missing):\n{preview}")

    entries = []
    reference_protocol = None
    for (seed, variant), paths in sorted(matrix_paths.items()):
        config = json.loads(paths["config"].read_text(encoding="utf-8"))
        best = _safe_load(paths["best_sem"])
        last = _safe_load(paths["last"])
        expected_ce, expected_supcon, expected_consistency = EXPECTED_WEIGHTS[variant]

        checks = {
            "categories": ["Hammer"],
            "canonical_label_names": ["Handle", "Head"],
            "train_mode": "semantic",
            "split_seed": split_seed,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "seed": seed,
            "epochs": epochs,
            "save_every": epochs,
            "batch_size": 6,
            "num_support_points": 5000,
            "num_query_surface_points": 2048,
            "balanced_sampling_ratio": 0.0,
            "query_sampling_ratio": 0.75,
            "rotation_mode": "so3",
            "jitter_std": 0.005,
            "sem_ce_weight": expected_ce,
            "sem_contrastive_weight": expected_supcon,
            "sem_consistency_weight": expected_consistency,
            "geo_dense_correspondence_weight": 0.0,
            "geo_metric_weight": 0.0,
            "geo_consistency_weight": 0.0,
            "occ_weight": 0.0,
            "finetune_utonia": False,
        }
        for key, expected in checks.items():
            _expect_equal(config.get(key), expected, f"seed={seed} variant={variant} config.{key}")
        for name, checkpoint in (("best_sem", best), ("last", last)):
            _expect_equal(checkpoint.get("format"), FORMAT_NAME, f"{name}.format")
            _expect_equal(checkpoint.get("canonical_label_names"), ["Handle", "Head"], f"{name}.labels")
            if checkpoint.get("training_resume_supported") is not False:
                raise ValueError(f"{name} unexpectedly supports resume: {paths[name]}")
            if "optimizer_state_dict" in checkpoint or "scaler_state_dict" in checkpoint:
                raise ValueError(f"unsafe optimizer/scaler state found in {paths[name]}")
            checkpoint_args = checkpoint.get("args", {})
            for key, expected in checks.items():
                if key in {"canonical_label_names", "epochs", "save_every"}:
                    continue
                _expect_equal(
                    checkpoint_args.get(key), expected, f"seed={seed} variant={variant} {name}.args.{key}"
                )
            try:
                checkpoint_epochs = int(checkpoint_args["epochs"])
                checkpoint_save_every = int(checkpoint_args["save_every"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{name} budget metadata missing: {paths[name]}") from error
            allowed_targets = (
                {epochs, artifact_tag} if name == "best_sem" else {epochs}
            )
            if checkpoint_epochs not in allowed_targets or checkpoint_save_every != checkpoint_epochs:
                raise ValueError(
                    f"{name} budget mismatch: epochs={checkpoint_epochs}, "
                    f"save_every={checkpoint_save_every}, allowed={sorted(allowed_targets)}: "
                    f"{paths[name]}"
                )

        if int(last.get("epoch", -1)) != epochs:
            raise ValueError(f"run is not complete: {paths['last']} epoch={last.get('epoch')}")
        best_epoch = int(best.get("epoch", -1))
        if not 1 <= best_epoch <= epochs:
            raise ValueError(f"invalid best_sem epoch {best_epoch}: {paths['best_sem']}")

        protocol = {
            key: config[key]
            for key in checks
            if key not in {"seed", "sem_ce_weight", "sem_contrastive_weight", "sem_consistency_weight"}
        }
        if reference_protocol is None:
            reference_protocol = protocol
        elif protocol != reference_protocol:
            raise ValueError(f"non-loss protocol mismatch at seed={seed} variant={variant}")

        entries.append(
            {
                "seed": seed,
                "variant": variant,
                "run_dir": str(paths["run_dir"]),
                "config": str(paths["config"]),
                "checkpoint": str(paths["best_sem"]),
                "checkpoint_sha256": sha256_file(paths["best_sem"]),
                "best_epoch": best_epoch,
                "last_epoch": int(last["epoch"]),
                "loss_weights": {
                    "ce": expected_ce,
                    "supcon": expected_supcon,
                    "consistency": expected_consistency,
                },
            }
        )

    return {
        "schema_version": 1,
        "experiment": "hammer_semantic_field_three_loss_ablation",
        "audit_created_at_utc": datetime.now(timezone.utc).isoformat(),
        "matrix_complete": True,
        "strict_weights_only": True,
        "training_seeds": sorted(seed_roots),
        "variants": list(VARIANTS),
        "expected_runs": len(seed_roots) * len(VARIANTS),
        "training_epochs": epochs,
        "artifact_epoch_tag": artifact_tag,
        "split_seed": split_seed,
        "evaluation_protocol": evaluation_protocol,
        "evaluation_split": protocol_spec["split"],
        "expected_evaluation_status": protocol_spec["status"],
        "paper_claim_eligible": protocol_spec["paper_claim_eligible"],
        "eligibility_note": (
            "Legacy validation is also used for checkpoint selection; this matrix is diagnostic only."
            if evaluation_protocol == "legacy_val"
            else "Independent test is disjoint from train/validation by the checkpoint split config; "
            "claim eligibility additionally assumes the test protocol was frozen before inspecting test results."
        ),
        "non_loss_protocol": reference_protocol,
        "entries": entries,
    }


def build_evaluation_commands(
    manifest: dict[str, Any],
    *,
    python_bin: Path,
    dataset_root: Path,
    alias_config: Path,
    utonia_checkpoint: Path,
    output_root: Path,
    device: str,
    evaluation_seed: int,
) -> list[dict[str, Any]]:
    evaluation_split = str(manifest.get("evaluation_split", "val"))
    expected_status = str(
        manifest.get(
            "expected_evaluation_status",
            "diagnostic_validation_checkpoint_selected_on_same_split",
        )
    )
    split_seed = int(manifest.get("split_seed", 42))
    training_protocol = manifest.get("non_loss_protocol", {})
    val_ratio = float(training_protocol.get("val_ratio", 0.1))
    test_ratio = float(training_protocol.get("test_ratio", 0.0))
    commands = []
    for entry in manifest["entries"]:
        label = f"seed{entry['seed']}_{entry['variant']}"
        common = [
            "--checkpoint", entry["checkpoint"],
            "--dataset-root", str(dataset_root),
            "--alias-config", str(alias_config),
            "--utonia-checkpoint", str(utonia_checkpoint),
            "--split", evaluation_split,
            "--device", device,
            "--seed", str(evaluation_seed),
        ]
        r1_output = output_root / label / f"r1_{evaluation_split}_r5_fp32"
        r1 = [
            str(python_bin), str(SCRIPT_ROOT / "evaluate_semantic_field.py"),
            *common, "--output-dir", str(r1_output), "--repeats", "5",
        ]
        r2_output = output_root / label / f"r2_{evaluation_split}_t5_fp32"
        r2 = [
            str(python_bin), str(SCRIPT_ROOT / "benchmark_semantic_field_stability.py"),
            *common, "--output-dir", str(r2_output), "--trials", "5",
            "--support-counts", "128", "256", "512", "1024", "5000",
            "--dropout-ratios", "0.25", "0.5", "0.75",
            "--noise-std-fractions", "0.0025", "0.005", "0.01",
            "--crop-keep-ratios", "0.75", "0.5", "0.25",
            "--outlier-ratios", "0.05", "0.1", "0.2",
            "--normal-view-keep-ratios", "0.75", "0.5", "0.25",
            "--normalization-modes", "support", "reference",
        ]
        commands.extend(
            [
                {
                    "label": label,
                    "kind": "r1",
                    "experiment": "R1_part_semantic_metrics",
                    "checkpoint_sha256": entry["checkpoint_sha256"],
                    "output_dir": str(r1_output),
                    "expected_result": {
                        "evaluation_status": expected_status,
                        "split": evaluation_split,
                        "split_seed": split_seed,
                        "val_ratio": val_ratio,
                        "test_ratio": test_ratio,
                        "evaluation_seed": int(evaluation_seed),
                        "repeats": 5,
                        "amp": False,
                    },
                    "argv": r1,
                },
                {
                    "label": label,
                    "kind": "r2",
                    "experiment": "R2_fixed_query_support_perturbation",
                    "checkpoint_sha256": entry["checkpoint_sha256"],
                    "output_dir": str(r2_output),
                    "expected_result": {
                        "evaluation_status": expected_status,
                        "split": evaluation_split,
                        "split_seed": split_seed,
                        "val_ratio": val_ratio,
                        "test_ratio": test_ratio,
                        "evaluation_seed": int(evaluation_seed),
                        "trials": 5,
                        "amp": False,
                    },
                    "argv": r2,
                },
            ]
        )
    return commands


def _write_text_fail_closed(path: Path, content: str) -> None:
    if path.exists():
        if not path.is_file() or path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"refusing to overwrite mismatched file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {temporary}")
    temporary.write_text(content, encoding="utf-8")
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _manifest_comparable(payload: dict[str, Any]) -> dict[str, Any]:
    comparable = dict(payload)
    comparable.pop("audit_created_at_utc", None)
    return comparable


def write_manifest_fail_closed(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if _manifest_comparable(existing) != _manifest_comparable(manifest):
            raise FileExistsError(f"refusing to overwrite mismatched manifest: {path}")
        return existing
    _write_text_fail_closed(path, json.dumps(manifest, indent=2) + "\n")
    return manifest


def write_plan(commands: list[dict[str, Any]], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    _write_text_fail_closed(
        output_root / "evaluation_plan.json", json.dumps(commands, indent=2) + "\n"
    )
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    lines.extend(shlex.join(command["argv"]) for command in commands)
    _write_text_fail_closed(
        output_root / "evaluation_commands.sh", "\n".join(lines) + "\n"
    )


def validate_completed_result(command: dict[str, Any], results_path: Path) -> None:
    result = json.loads(results_path.read_text(encoding="utf-8"))
    if result.get("experiment") != command["experiment"]:
        raise ValueError(f"experiment mismatch: {results_path}")
    if result.get("checkpoint", {}).get("sha256") != command["checkpoint_sha256"]:
        raise ValueError(f"checkpoint SHA mismatch: {results_path}")
    expected = command["expected_result"]
    if result.get("evaluation_status") != expected["evaluation_status"]:
        raise ValueError(f"evaluation status mismatch: {results_path}")
    split = result.get("split", {})
    if split.get("name") != expected["split"] or int(split.get("seed", -1)) != int(
        expected["split_seed"]
    ):
        raise ValueError(f"split mismatch: {results_path}")
    for ratio_key in ("val_ratio", "test_ratio"):
        if not math.isclose(
            float(split.get(ratio_key, -1.0)),
            float(expected[ratio_key]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{ratio_key} mismatch: {results_path}")
    args = result.get("evaluation_args", {})
    if int(args.get("seed", -1)) != int(expected["evaluation_seed"]):
        raise ValueError(f"evaluation seed mismatch: {results_path}")
    count_key = "repeats" if command["kind"] == "r1" else "trials"
    if int(args.get(count_key, -1)) != int(expected[count_key]):
        raise ValueError(f"{count_key} mismatch: {results_path}")
    if bool(args.get("amp", False)) != bool(expected["amp"]):
        raise ValueError(f"precision mode mismatch: {results_path}")


def validate_commands(commands: list[dict[str, Any]]) -> dict[str, int]:
    missing = []
    for command in commands:
        results_path = Path(command["output_dir"]) / "results.json"
        if not results_path.is_file():
            missing.append(results_path)
            continue
        validate_completed_result(command, results_path)
    if missing:
        preview = "\n".join(str(path) for path in missing[:8])
        raise FileNotFoundError(
            f"evaluation is incomplete ({len(missing)} results missing):\n{preview}"
        )
    return {"expected": len(commands), "validated": len(commands)}


def run_commands(commands: list[dict[str, Any]]) -> None:
    for command in commands:
        output_dir = Path(command["output_dir"])
        results_path = output_dir / "results.json"
        if results_path.is_file():
            validate_completed_result(command, results_path)
            print(f"[skip-complete] {command['label']} {command['kind']}", flush=True)
            continue
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"refusing non-empty incomplete evaluation directory: {output_dir}")
        print(f"[run] {shlex.join(command['argv'])}", flush=True)
        subprocess.run(command["argv"], check=True, cwd=REPO_ROOT)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-root", action="append", required=True, metavar="SEED=PATH")
    parser.add_argument(
        "--mode", choices=("audit", "plan", "validate", "run"), default="audit"
    )
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--wait-interval-seconds", type=float, default=30.0)
    parser.add_argument("--wait-timeout-seconds", type=float)
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--artifact-epoch-tag", type=int)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--evaluation-protocol",
        choices=tuple(EVALUATION_PROTOCOLS),
        default="legacy_val",
    )
    parser.add_argument("--expected-val-ratio", type=float)
    parser.add_argument("--expected-test-ratio", type=float)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--python-bin", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--alias-config", type=Path)
    parser.add_argument("--utonia-checkpoint", type=Path)
    parser.add_argument("--evaluation-output-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--evaluation-seed", type=int, default=20260805)
    parser.add_argument("--confirm-run-evaluators", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    roots = parse_seed_roots(args.seed_root)
    artifact_tag = (
        args.epochs if args.artifact_epoch_tag is None else args.artifact_epoch_tag
    )
    paths = expected_paths(roots, artifact_tag, args.split_seed)
    if args.wait:
        wait_until_complete(
            paths,
            interval_seconds=args.wait_interval_seconds,
            timeout_seconds=args.wait_timeout_seconds,
        )
    manifest = audit_matrix(
        roots,
        epochs=args.epochs,
        split_seed=args.split_seed,
        artifact_epoch_tag=artifact_tag,
        evaluation_protocol=args.evaluation_protocol,
        expected_val_ratio=args.expected_val_ratio,
        expected_test_ratio=args.expected_test_ratio,
    )
    manifest_path = args.manifest.expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = write_manifest_fail_closed(manifest_path, manifest)
    print(json.dumps({"audit": "passed", "runs": len(manifest["entries"]), "manifest": str(manifest_path)}))
    if args.mode == "audit":
        return

    required = {
        "python_bin": args.python_bin,
        "dataset_root": args.dataset_root,
        "alias_config": args.alias_config,
        "utonia_checkpoint": args.utonia_checkpoint,
        "evaluation_output_root": args.evaluation_output_root,
    }
    missing_args = [name for name, value in required.items() if value is None]
    if missing_args:
        raise ValueError(f"mode={args.mode} requires: {', '.join(missing_args)}")
    for name, value in required.items():
        if name != "evaluation_output_root" and not value.expanduser().resolve().exists():
            raise FileNotFoundError(f"{name} does not exist: {value}")
    output_root = args.evaluation_output_root.expanduser().resolve()
    commands = build_evaluation_commands(
        manifest,
        python_bin=args.python_bin.expanduser().resolve(),
        dataset_root=args.dataset_root.expanduser().resolve(),
        alias_config=args.alias_config.expanduser().resolve(),
        utonia_checkpoint=args.utonia_checkpoint.expanduser().resolve(),
        output_root=output_root,
        device=args.device,
        evaluation_seed=args.evaluation_seed,
    )
    write_plan(commands, output_root)
    print(json.dumps({"plan": str(output_root / 'evaluation_plan.json'), "commands": len(commands)}))
    if args.mode == "plan":
        return
    if args.mode == "validate":
        validated = validate_commands(commands)
        print(json.dumps({"validation": "passed", **validated}))
        return
    if not args.confirm_run_evaluators:
        raise PermissionError("mode=run additionally requires --confirm-run-evaluators")
    run_commands(commands)


if __name__ == "__main__":
    main()
