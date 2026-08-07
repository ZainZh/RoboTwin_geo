#!/usr/bin/env python3
"""Read-only validation of the four persistent local Hammer resume targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from semantic_field_training_state import load_training_state, validate_training_state


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "outputs/paper_revision/hammer_semantic_loss_ablation_local4090_"
    "seed20260807_e4000_fastw4_resumable_restart20260806"
)
EXPECTED_WEIGHTS = {
    "full_fixed": (1.0, 0.2, 0.1),
    "no_ce": (0.0, 0.2, 0.1),
    "no_supcon": (1.0, 0.0, 0.1),
    "no_consistency": (1.0, 0.2, 0.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--epochs", type=int, default=1750)
    parser.add_argument("--run-name-epoch-tag", type=int, default=4000)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    output_root = cli.output_root.expanduser().resolve()
    reports = []
    for variant, expected_weights in EXPECTED_WEIGHTS.items():
        run_name = f"hammer_{variant}_e{cli.run_name_epoch_tag}_split42_seed{cli.seed}"
        run_dir = output_root / run_name
        resume_path = run_dir / "resume.pt"
        payload = load_training_state(resume_path)
        saved_args = dict(payload["args"])

        if Path(saved_args["output_dir"]).expanduser().resolve() != output_root:
            raise RuntimeError(f"stored output root mismatch: {resume_path}")
        if saved_args.get("run_name") != run_name:
            raise RuntimeError(f"stored run name mismatch: {resume_path}")
        if int(saved_args.get("seed", -1)) != cli.seed:
            raise RuntimeError(f"stored seed mismatch: {resume_path}")
        stored_target = int(saved_args.get("epochs", -1))
        if stored_target < cli.epochs:
            raise RuntimeError(f"stored target is below requested target: {resume_path}")
        actual_weights = (
            float(saved_args["sem_ce_weight"]),
            float(saved_args["sem_contrastive_weight"]),
            float(saved_args["sem_consistency_weight"]),
        )
        if actual_weights != expected_weights:
            raise RuntimeError(
                f"loss identity mismatch for {variant}: {actual_weights}"
            )

        expected_args = argparse.Namespace(**saved_args)
        expected_args.num_workers = cli.num_workers
        expected_args.epochs = cli.epochs
        expected_args.resume = None
        expected_args.resume_from = None
        expected_args.auto_resume = True
        epoch = int(payload["epoch"])
        if epoch < cli.epochs:
            validate_training_state(resume_path, expected_args)
            status = "verified_resumable_with_shortened_target"
        else:
            status = "verified_at_or_above_shortened_target"
        reports.append(
            {
                "variant": variant,
                "run_name": run_name,
                "resume_path": str(resume_path),
                "epoch": int(payload["epoch"]),
                "stored_target_epoch": stored_target,
                "requested_target_epoch": cli.epochs,
                "global_step": int(payload["global_step"]),
                "stored_num_workers": int(saved_args["num_workers"]),
                "requested_num_workers": cli.num_workers,
                "stored_config_sha256": payload["run_identity"]["config_sha256"],
                "loss_weights": actual_weights,
                "status": status,
            }
        )

    print(json.dumps({"output_root": str(output_root), "runs": reports}, indent=2))


if __name__ == "__main__":
    main()
