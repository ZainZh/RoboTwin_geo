#!/usr/bin/env python3
"""Strictly summarize three-seed Hammer loss-ablation R1/R2 diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable


VARIANTS = ("full_fixed", "no_ce", "no_supcon", "no_consistency", "ce_only")
EXPECTED_CONDITIONS = tuple(
    [f"count_{value}" for value in (128, 256, 512, 1024, 5000)]
    + [f"dropout_{value}" for value in ("0.25", "0.5", "0.75")]
    + [f"noise_{value}" for value in ("0.0025", "0.005", "0.01")]
    + [f"crop_keep_{value}" for value in ("0.75", "0.5", "0.25")]
    + [f"outlier_replace_{value}" for value in ("0.05", "0.1", "0.2")]
    + [f"normal_view_keep_{value}" for value in ("0.75", "0.5", "0.25")]
)
METRIC_KEYS = (
    "mean_iou",
    "accuracy",
    "delta_miou_vs_aggregate_reference",
    "label_agreement_with_reference",
    "cosine_mean",
    "cosine_p05_mean",
    "embedding_l2_mean",
    "query_coord_drift_mean",
)


def summarize(values: Iterable[float]) -> dict[str, float | int]:
    samples = [float(value) for value in values]
    if not samples:
        raise ValueError("cannot summarize an empty sequence")
    mean = sum(samples) / len(samples)
    std = (
        math.sqrt(sum((value - mean) ** 2 for value in samples) / (len(samples) - 1))
        if len(samples) > 1
        else 0.0
    )
    return {
        "n_training_seeds": len(samples),
        "mean": mean,
        "sample_std": std,
        "sem": std / math.sqrt(len(samples)),
        "min": min(samples),
        "max": max(samples),
    }


def _record_signature(records: list[dict[str, Any]]) -> list[tuple[str, str]]:
    return [(str(row["category"]), str(row["model_id"])) for row in records]


def load_reports(
    manifest: dict[str, Any], evaluation_root: Path
) -> dict[int, dict[str, dict[str, Any]]]:
    evaluation_split = str(manifest.get("evaluation_split", "val"))
    expected_status = str(
        manifest.get(
            "expected_evaluation_status",
            "diagnostic_validation_checkpoint_selected_on_same_split",
        )
    )
    split_seed = int(manifest.get("split_seed", 42))
    training_protocol = manifest.get("non_loss_protocol", {})
    expected_val_ratio = float(training_protocol.get("val_ratio", 0.1))
    expected_test_ratio = float(training_protocol.get("test_ratio", 0.0))
    reports: dict[int, dict[str, dict[str, Any]]] = {}
    for entry in manifest["entries"]:
        seed = int(entry["seed"])
        variant = str(entry["variant"])
        label = f"seed{seed}_{variant}"
        r1_path = (
            evaluation_root / label / f"r1_{evaluation_split}_r5_fp32" / "results.json"
        )
        r2_path = (
            evaluation_root / label / f"r2_{evaluation_split}_t5_fp32" / "results.json"
        )
        if not r1_path.is_file() or not r2_path.is_file():
            raise FileNotFoundError(f"missing evaluation for {label}: {r1_path} / {r2_path}")
        r1 = json.loads(r1_path.read_text(encoding="utf-8"))
        r2 = json.loads(r2_path.read_text(encoding="utf-8"))
        expected_sha = entry["checkpoint_sha256"]
        for kind, report, experiment in (
            ("r1", r1, "R1_part_semantic_metrics"),
            ("r2", r2, "R2_fixed_query_support_perturbation"),
        ):
            if report.get("experiment") != experiment:
                raise ValueError(f"{label} {kind} experiment mismatch")
            if report.get("evaluation_status") != expected_status:
                raise ValueError(f"{label} {kind} evaluation status mismatch")
            if report["checkpoint"]["sha256"] != expected_sha:
                raise ValueError(f"{label} {kind} checkpoint SHA mismatch")
            if report["labels"] != ["Handle", "Head"]:
                raise ValueError(f"{label} {kind} label mismatch")
            split = report["split"]
            if split["name"] != evaluation_split or int(split["seed"]) != split_seed:
                raise ValueError(f"{label} {kind} split mismatch")
            for ratio_key, expected_ratio in (
                ("val_ratio", expected_val_ratio),
                ("test_ratio", expected_test_ratio),
            ):
                if not math.isclose(
                    float(split.get(ratio_key, -1.0)),
                    expected_ratio,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(f"{label} {kind} {ratio_key} mismatch")
        reports.setdefault(seed, {})[variant] = {
            "r1": r1,
            "r2": r2,
            "r1_path": str(r1_path),
            "r2_path": str(r2_path),
        }
    return reports


def validate_matched_protocol(reports: dict[int, dict[str, dict[str, Any]]]) -> None:
    if len(reports) != 3:
        raise ValueError(f"exactly three training seeds are required, got {len(reports)}")
    reference = None
    for seed, variants in sorted(reports.items()):
        if set(variants) != set(VARIANTS):
            raise ValueError(f"seed {seed} does not contain exactly {VARIANTS}")
        for variant, payload in variants.items():
            r1 = payload["r1"]
            r2 = payload["r2"]
            r1_args = r1["evaluation_args"]
            r2_args = r2["evaluation_args"]
            r2_conditions = sorted(
                (
                    row["normalization_mode"],
                    row["condition"],
                    row["condition_kind"],
                    float(row["condition_value"]),
                )
                for row in r2["aggregate_conditions"]
            )
            expected_keys = {
                (mode, condition)
                for mode in ("support", "reference")
                for condition in EXPECTED_CONDITIONS
            }
            actual_keys = {(row[0], row[1]) for row in r2_conditions}
            if actual_keys != expected_keys or len(r2_conditions) != len(expected_keys):
                raise ValueError(
                    f"{seed}/{variant} R2 conditions incomplete or duplicated: "
                    f"expected {len(expected_keys)}, got {len(r2_conditions)}"
                )

            signature = {
                "r1_split_records": _record_signature(r1["split"]["records"]),
                "r2_split_records": _record_signature(r2["split"]["records"]),
                "r1_repeats": int(r1_args["repeats"]),
                "r1_seed": int(r1_args["seed"]),
                "r1_amp": bool(r1_args["amp"]),
                "r2_trials": int(r2_args["trials"]),
                "r2_seed": int(r2_args["seed"]),
                "r2_amp": bool(r2_args["amp"]),
                "r2_fixed_query_protocol": r2["fixed_query_protocol"],
                "r2_conditions": r2_conditions,
            }
            if signature["r1_repeats"] != 5 or signature["r2_trials"] != 5:
                raise ValueError(f"{seed}/{variant} does not use R1x5/R2x5")
            if signature["r1_amp"] or signature["r2_amp"]:
                raise ValueError(f"{seed}/{variant} is not FP32 evaluation")
            if signature["r1_split_records"] != signature["r2_split_records"]:
                raise ValueError(f"{seed}/{variant} R1/R2 object split mismatch")
            if reference is None:
                reference = signature
            elif signature != reference:
                raise ValueError(f"evaluation protocol mismatch at {seed}/{variant}")


def _sum_confusions(confusions: list[list[list[int]]]) -> list[list[int]]:
    if not confusions:
        raise ValueError("no confusion matrices")
    size = len(confusions[0])
    result = [[0 for _ in range(size)] for _ in range(size)]
    for confusion in confusions:
        if len(confusion) != size or any(len(row) != size for row in confusion):
            raise ValueError("confusion matrix shape mismatch")
        for row in range(size):
            for column in range(size):
                result[row][column] += int(confusion[row][column])
    return result


def aggregate_reports(
    manifest: dict[str, Any], reports: dict[int, dict[str, dict[str, Any]]]
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    validate_matched_protocol(reports)
    seeds = sorted(reports)
    r1_rows: list[dict[str, Any]] = []
    part_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    r2_rows: list[dict[str, Any]] = []
    r1_summary: dict[str, Any] = {}
    r2_summary: dict[str, Any] = {}

    for variant in VARIANTS:
        seed_r1 = {seed: reports[seed][variant]["r1"] for seed in seeds}
        r1_summary[variant] = {
            "mean_iou": summarize(report["aggregate"]["mean_iou"] for report in seed_r1.values()),
            "accuracy": summarize(report["aggregate"]["accuracy"] for report in seed_r1.values()),
            "per_training_seed": {},
            "per_part": {},
            "pooled_confusion_across_training_seeds": _sum_confusions(
                [report["aggregate"]["confusion_matrix"] for report in seed_r1.values()]
            ),
        }
        for seed, report in seed_r1.items():
            aggregate = report["aggregate"]
            r1_summary[variant]["per_training_seed"][str(seed)] = {
                "mean_iou": aggregate["mean_iou"],
                "accuracy": aggregate["accuracy"],
            }
            r1_rows.append(
                {"variant": variant, "training_seed": seed, "mean_iou": aggregate["mean_iou"], "accuracy": aggregate["accuracy"]}
            )
            for row_index, row in enumerate(aggregate["confusion_matrix"]):
                for column_index, count in enumerate(row):
                    confusion_rows.append(
                        {"variant": variant, "training_seed": seed, "ground_truth_class": row_index, "predicted_class": column_index, "count": count}
                    )
        for label in ("Handle", "Head"):
            per_seed = {
                seed: next(item for item in report["aggregate"]["per_class"] if item["label"] == label)["iou"]
                for seed, report in seed_r1.items()
            }
            r1_summary[variant]["per_part"][label] = {
                "per_training_seed_iou": {str(seed): value for seed, value in per_seed.items()},
                "across_training_seeds": summarize(per_seed.values()),
            }
            for seed, value in per_seed.items():
                part_rows.append({"variant": variant, "training_seed": seed, "label": label, "iou": value})

        seed_r2 = {seed: reports[seed][variant]["r2"] for seed in seeds}
        reference_miou = {seed: report["reference"]["mean_iou"] for seed, report in seed_r2.items()}
        condition_maps = {
            seed: {
                (row["normalization_mode"], row["condition"]): row
                for row in report["aggregate_conditions"]
            }
            for seed, report in seed_r2.items()
        }
        condition_keys = sorted(next(iter(condition_maps.values())))
        r2_summary[variant] = {
            "reference_mean_iou": {
                "per_training_seed": {str(seed): value for seed, value in reference_miou.items()},
                "across_training_seeds": summarize(reference_miou.values()),
            },
            "conditions": {},
        }
        for key in condition_keys:
            mode, condition = key
            rows = {seed: condition_maps[seed][key] for seed in seeds}
            payload = {
                "condition_kind": next(iter(rows.values()))["condition_kind"],
                "condition_value": next(iter(rows.values()))["condition_value"],
                "metrics": {},
                "pooled_confusion_across_training_seeds": _sum_confusions(
                    [row["confusion_matrix"] for row in rows.values()]
                ),
            }
            for metric in METRIC_KEYS:
                values = {seed: row.get(metric) for seed, row in rows.items()}
                if all(value is not None for value in values.values()):
                    payload["metrics"][metric] = {
                        "per_training_seed": {str(seed): value for seed, value in values.items()},
                        "across_training_seeds": summarize(values.values()),
                    }
                    for seed, value in values.items():
                        r2_rows.append(
                            {"variant": variant, "training_seed": seed, "normalization_mode": mode, "condition": condition, "metric": metric, "value": value}
                        )
            r2_summary[variant]["conditions"][f"{mode}::{condition}"] = payload

    paper_claim_eligible = bool(manifest.get("paper_claim_eligible", False))
    if paper_claim_eligible:
        protocol = manifest.get("non_loss_protocol", {})
        independent_gate = (
            manifest.get("evaluation_protocol") == "independent_test"
            and manifest.get("evaluation_split") == "test"
            and manifest.get("expected_evaluation_status") == "held_out_test"
            and float(protocol.get("val_ratio", 0.0)) > 0.0
            and float(protocol.get("test_ratio", 0.0)) > 0.0
        )
        if not independent_gate:
            raise ValueError(
                "paper-claim eligibility requires an audited independent test split "
                "with positive validation and test ratios"
            )
    summary = {
        "schema_version": 1,
        "experiment": (
            "hammer_three_loss_ablation_r1_r2_independent_test_summary"
            if paper_claim_eligible
            else "hammer_three_loss_ablation_r1_r2_old_val_summary"
        ),
        "evaluation_protocol": manifest.get("evaluation_protocol", "legacy_val"),
        "paper_claim_eligible": paper_claim_eligible,
        "eligibility_note": manifest.get(
            "eligibility_note",
            "All reports use the legacy validation split also used for checkpoint selection. "
            "This is a fixed-protocol diagnosis, not an independent held-out test; stage 42 remains pending.",
        ),
        "statistical_unit": "independent_semantic_field_training_seed",
        "training_seeds": seeds,
        "variants": list(VARIANTS),
        "training_manifest": manifest,
        "r1": r1_summary,
        "r2": r2_summary,
    }
    tables = {
        "r1_seed_metrics.csv": r1_rows,
        "r1_per_part.csv": part_rows,
        "r1_confusion.csv": confusion_rows,
        "r2_conditions_long.csv": r2_rows,
    }
    return summary, tables


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected_runs = len(VARIANTS) * 3
    if not manifest.get("matrix_complete") or int(manifest.get("expected_runs", 0)) != expected_runs:
        raise ValueError(f"training audit manifest is not a complete {expected_runs}-run matrix")
    reports = load_reports(manifest, args.evaluation_root.expanduser().resolve())
    summary, tables = aggregate_reports(manifest, reports)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty summary directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    for filename, rows in tables.items():
        write_csv(output_dir / filename, rows)
    print(json.dumps({"summary": str(output_dir / 'summary.json'), "runs": expected_runs}))


if __name__ == "__main__":
    main()
