#!/usr/bin/env python3
"""Aggregate matched DP3 offline evaluations across independent training seeds.

The unit of replication in this report is a *training seed*. Diffusion-noise
repeats inside an individual evaluation JSON are first averaged and are never
treated as independent policy-training runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable


DEFAULT_ROUTES = ("field", "xyz", "partprob")
DEFAULT_STATES = ("raw", "ema")


def summarize(values: Iterable[float]) -> dict[str, float | int]:
    samples = [float(value) for value in values]
    if not samples:
        raise ValueError("Cannot summarize an empty sequence")
    mean = sum(samples) / len(samples)
    if len(samples) > 1:
        variance = sum((value - mean) ** 2 for value in samples) / (len(samples) - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0
    return {
        "n": len(samples),
        "mean": mean,
        "std": std,
        "sem": std / math.sqrt(len(samples)),
        "min": min(samples),
        "max": max(samples),
    }


def aggregate_reports(
    reports: dict[int, dict[str, dict[str, Any]]],
    routes: Iterable[str],
    states: Iterable[str] = DEFAULT_STATES,
    epochs: int = 300,
) -> dict[str, Any]:
    route_names = list(routes)
    state_names = list(states)
    seeds = sorted(reports)
    if len(seeds) < 2:
        raise ValueError("At least two independent training seeds are required")

    reference_protocol: dict[str, Any] | None = None
    reference_dataset: dict[str, Any] | None = None
    source_files: list[str] = []
    for seed in seeds:
        if set(reports[seed]) != set(route_names):
            raise ValueError(f"Seed {seed} does not contain exactly routes {route_names}")
        for route in route_names:
            report = reports[seed][route]
            if report.get("route") != route:
                raise ValueError(f"Route mismatch for seed {seed}: expected {route}")
            metadata_seed = int(report["checkpoint_metadata"]["seed"])
            if metadata_seed != seed:
                raise ValueError(f"Checkpoint seed mismatch: expected {seed}, got {metadata_seed}")
            metadata_epoch = report["checkpoint_metadata"].get("epoch")
            if metadata_epoch is not None and int(metadata_epoch) != epochs:
                raise ValueError(f"Checkpoint epoch mismatch: expected {epochs}, got {metadata_epoch}")
            dataset_signature = {
                "validation_episode_ids": report["dataset"]["validation_episode_ids"],
                "validation_samples": report["dataset"]["validation_samples"],
                "n_episodes": report["dataset"]["n_episodes"],
            }
            protocol_signature = {
                "repeat_seeds": report["protocol"]["repeat_seeds"],
                "repeats": report["protocol"]["repeats"],
                "paired_noise_across_raw_ema": report["protocol"][
                    "paired_noise_across_raw_ema"
                ],
            }
            if reference_dataset is None:
                reference_dataset = dataset_signature
                reference_protocol = protocol_signature
            if dataset_signature != reference_dataset:
                raise ValueError(f"Validation split mismatch at seed {seed}, route {route}")
            if protocol_signature != reference_protocol:
                raise ValueError(f"Diffusion evaluation protocol mismatch at seed {seed}, route {route}")
            for state in state_names:
                if state not in report["results"]:
                    raise ValueError(f"Missing {state} result at seed {seed}, route {route}")
            if "_source_file" in report:
                source_files.append(report["_source_file"])

    route_summaries: dict[str, Any] = {}
    pairwise: dict[str, Any] = {}
    seed_rankings: dict[str, Any] = {}
    for state in state_names:
        route_summaries[state] = {}
        for route in route_names:
            per_seed = {
                str(seed): float(reports[seed][route]["results"][state]["summary"]["mean"])
                for seed in seeds
            }
            route_summaries[state][route] = {
                "per_training_seed_mean_loss": per_seed,
                "across_training_seeds": summarize(per_seed.values()),
            }

        pairwise[state] = {}
        for left_index, left_route in enumerate(route_names):
            for right_route in route_names[left_index + 1 :]:
                deltas = {
                    str(seed): (
                        route_summaries[state][left_route]["per_training_seed_mean_loss"][str(seed)]
                        - route_summaries[state][right_route]["per_training_seed_mean_loss"][str(seed)]
                    )
                    for seed in seeds
                }
                pairwise[state][f"{left_route}_minus_{right_route}"] = {
                    "per_training_seed_delta": deltas,
                    "across_training_seeds": summarize(deltas.values()),
                    "left_lower_count": sum(delta < 0 for delta in deltas.values()),
                    "right_lower_count": sum(delta > 0 for delta in deltas.values()),
                    "tie_count": sum(delta == 0 for delta in deltas.values()),
                }

        seed_rankings[state] = {
            str(seed): sorted(
                route_names,
                key=lambda route: route_summaries[state][route]["per_training_seed_mean_loss"][str(seed)],
            )
            for seed in seeds
        }

    return {
        "schema_version": 1,
        "evaluation_kind": "fixed_split_diffusion_imitation_loss_across_training_seeds",
        "paper_claim_eligible": False,
        "eligibility_note": (
            f"Three-seed e{epochs} offline diagnostic on two validation episodes; "
            "not a task success rate and not sufficient for a paper main claim."
        ),
        "statistical_unit": "independent_policy_training_seed",
        "training_seeds": seeds,
        "training_epochs": epochs,
        "routes": route_names,
        "states": state_names,
        "matched_dataset": reference_dataset,
        "matched_diffusion_protocol": reference_protocol,
        "route_summaries": route_summaries,
        "pairwise": pairwise,
        "seed_rankings_best_to_worst": seed_rankings,
        "source_files": sorted(source_files),
    }


def load_reports(
    input_dir: Path,
    seeds: Iterable[int],
    routes: Iterable[str],
    epochs: int = 300,
) -> dict[int, dict[str, Any]]:
    reports: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        reports[seed] = {}
        for route in routes:
            path = input_dir / f"beat_cube_{route}_e{epochs}_seed{seed}.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            report = json.loads(path.read_text(encoding="utf-8"))
            report["_source_file"] = str(path.resolve())
            reports[seed][route] = report
    return reports


def write_csv(summary: dict[str, Any], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for state, route_payloads in summary["route_summaries"].items():
        for route, payload in route_payloads.items():
            aggregate = payload["across_training_seeds"]
            for seed, loss in payload["per_training_seed_mean_loss"].items():
                rows.append(
                    {
                        "state": state,
                        "route": route,
                        "training_seed": seed,
                        "mean_loss_over_diffusion_repeats": loss,
                        "across_seed_mean": aggregate["mean"],
                        "across_seed_sample_std": aggregate["std"],
                        "across_seed_sem": aggregate["sem"],
                        "n_training_seeds": aggregate["n"],
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--routes", nargs="+", default=list(DEFAULT_ROUTES))
    parser.add_argument("--epochs", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = load_reports(args.input_dir.resolve(), args.seeds, args.routes, args.epochs)
    summary = aggregate_reports(reports, args.routes, epochs=args.epochs)
    output_json = args.output_json.resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(summary, args.output_csv.resolve())
    print(json.dumps({"output_json": str(output_json), "output_csv": str(args.output_csv.resolve())}))


if __name__ == "__main__":
    main()
