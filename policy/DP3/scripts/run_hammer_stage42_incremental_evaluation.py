#!/usr/bin/env python3
"""Run claim-compatible R1/R2 for a verified subset of Stage42 runs.

This incremental runner never writes a complete-matrix manifest.  It only
reuses the frozen evaluator command builder after each selected training run
passes the Stage42 completion/SHA audit.  The final paper claim must still pass
the separate 15-run audit and validation gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hammer_stage42_protocol import (
    SPLIT_SEED,
    TEST_RATIO,
    VAL_RATIO,
    VARIANTS,
    audit_completed_run,
    build_tasks,
)
from prepare_hammer_loss_ablation_evaluation import (
    build_evaluation_commands,
    run_commands,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOTS = {
    "local": REPO_ROOT
    / "outputs/paper_revision/hammer_stage42_independent_test_evaluation_e1750_v1",
    "remote": Path(
        "/workspace/RoboTwin_geo/outputs/paper_revision/"
        "hammer_stage42_independent_test_evaluation_e1750_v1"
    ),
}
ROLE_RUNTIME = {
    "local": {
        "python_bin": Path("/home/zheng/miniforge3/envs/RoboTwin/bin/python"),
        "dataset_root": Path("/home/zheng/Datasets/PartNext_mesh"),
        "alias_config": REPO_ROOT
        / "include/3d_semantic_train/semantic_field_release/configs/hammer.json",
        "utonia_checkpoint": Path("/home/zheng/.cache/utonia/ckpt/utonia.pth"),
    },
    "remote": {
        "python_bin": Path("/workspace/venvs/geo-utonia/bin/python"),
        "dataset_root": Path("/workspace/data/PartNext_mesh"),
        "alias_config": Path(
            "/workspace/RoboTwin_geo/include/3d_semantic_train/"
            "semantic_field_release/configs/hammer.json"
        ),
        "utonia_checkpoint": Path("/workspace/checkpoints/utonia/utonia.pth"),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("local", "remote"), required=True)
    parser.add_argument(
        "--variant",
        action="append",
        choices=VARIANTS,
        required=True,
        help="Repeat for each already-complete variant to evaluate.",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--evaluation-seed", type=int, default=20260805)
    parser.add_argument("--kind", action="append", choices=("r1", "r2"))
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args()
    if len(set(args.variant)) != len(args.variant):
        parser.error("duplicate --variant values are not allowed")
    if args.kind and len(set(args.kind)) != len(args.kind):
        parser.error("duplicate --kind values are not allowed")
    return args


def build_verified_commands(args: argparse.Namespace) -> list[dict]:
    runtime = ROLE_RUNTIME[args.role]
    for label, path in runtime.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    tasks = build_tasks(args.role, variants=args.variant, num_workers=0 if args.role == "local" else 8)
    entries = []
    audits = []
    for task in tasks:
        report = audit_completed_run(task)
        if report.get("status") != "verified_complete":
            raise RuntimeError(f"unexpected audit result for {task.key}: {report}")
        checkpoint = task.run_dir / "best_sem.pt"
        entries.append(
            {
                "seed": int(task.seed),
                "variant": task.variant,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
            }
        )
        audits.append({"task": task.key, "status": report["status"]})

    manifest = {
        "evaluation_split": "test",
        "expected_evaluation_status": "held_out_test",
        "split_seed": SPLIT_SEED,
        "non_loss_protocol": {
            "val_ratio": VAL_RATIO,
            "test_ratio": TEST_RATIO,
        },
        "entries": entries,
    }
    output_root = (
        DEFAULT_OUTPUT_ROOTS[args.role]
        if args.output_root is None
        else args.output_root.expanduser().resolve()
    )
    commands = build_evaluation_commands(
        manifest,
        python_bin=runtime["python_bin"],
        dataset_root=runtime["dataset_root"],
        alias_config=runtime["alias_config"],
        utonia_checkpoint=runtime["utonia_checkpoint"],
        output_root=output_root,
        device=args.device,
        evaluation_seed=args.evaluation_seed,
    )
    selected_kinds = {"r1", "r2"} if not args.kind else set(args.kind)
    commands = [command for command in commands if command["kind"] in selected_kinds]
    print(
        json.dumps(
            {
                "role": args.role,
                "audits": audits,
                "output_root": str(output_root),
                "commands": len(commands),
                "mode": "run" if args.confirm_run else "audit",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return commands


def main() -> None:
    args = parse_args()
    commands = build_verified_commands(args)
    if not args.confirm_run:
        for command in commands:
            print(json.dumps({"label": command["label"], "kind": command["kind"], "argv": command["argv"]}))
        return
    run_commands(commands)


if __name__ == "__main__":
    main()
