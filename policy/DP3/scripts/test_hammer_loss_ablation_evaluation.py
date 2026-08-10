from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from prepare_hammer_loss_ablation_evaluation import (
    EXPECTED_WEIGHTS,
    VARIANTS,
    audit_matrix,
    build_evaluation_commands,
    expected_paths,
    validate_commands,
    write_manifest_fail_closed,
    write_plan,
)
from summarize_hammer_loss_ablation_evaluation import (
    EXPECTED_CONDITIONS,
    aggregate_reports,
    validate_matched_protocol,
)


SEEDS = (20260805, 20260806, 20260807)


def _config(
    seed: int,
    variant: str,
    epochs: int = 4000,
    val_ratio: float = 0.1,
    test_ratio: float = 0.0,
) -> dict:
    ce, supcon, consistency = EXPECTED_WEIGHTS[variant]
    return {
        "dataset_root": "/data",
        "categories": ["Hammer"],
        "canonical_label_names": ["Handle", "Head"],
        "train_mode": "semantic",
        "split_seed": 42,
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
        "sem_ce_weight": ce,
        "sem_contrastive_weight": supcon,
        "sem_consistency_weight": consistency,
        "geo_dense_correspondence_weight": 0.0,
        "geo_metric_weight": 0.0,
        "geo_consistency_weight": 0.0,
        "occ_weight": 0.0,
        "finetune_utonia": False,
    }


def _write_fake_matrix(
    root: Path,
    *,
    val_ratio: float = 0.1,
    test_ratio: float = 0.0,
    epochs: int = 4000,
    artifact_epoch_tag: int | None = None,
) -> dict[int, Path]:
    roots = {seed: root / f"seed{seed}" for seed in SEEDS}
    artifact_tag = epochs if artifact_epoch_tag is None else artifact_epoch_tag
    for (seed, variant), entry in expected_paths(roots, artifact_tag, 42).items():
        entry["run_dir"].mkdir(parents=True)
        config = _config(
            seed, variant, epochs=epochs, val_ratio=val_ratio, test_ratio=test_ratio
        )
        entry["config"].write_text(json.dumps(config), encoding="utf-8")
        checkpoint_args = {
            key: value for key, value in config.items() if key != "canonical_label_names"
        }
        base = {
            "format": "semantic_field_weights_only_v1",
            "canonical_label_names": ["Handle", "Head"],
            "training_resume_supported": False,
            "projector_state_dict": {"weight": torch.ones(1)},
            "args": checkpoint_args,
        }
        best_args = dict(checkpoint_args)
        best_args["epochs"] = artifact_tag
        best_args["save_every"] = artifact_tag
        torch.save(
            {**base, "args": best_args, "epoch": min(31, epochs)}, entry["best_sem"]
        )
        torch.save({**base, "epoch": epochs}, entry["last"])
    return roots


def _fake_reports() -> tuple[dict, dict]:
    entries = []
    reports = {}
    for seed_index, seed in enumerate(SEEDS):
        reports[seed] = {}
        for variant_index, variant in enumerate(VARIANTS):
            value = 0.7 + seed_index * 0.01 + variant_index * 0.02
            sha = f"sha-{seed}-{variant}"
            entries.append({"seed": seed, "variant": variant, "checkpoint_sha256": sha})
            records = [{"category": "Hammer", "model_id": "object-1"}]
            r1 = {
                "experiment": "R1_part_semantic_metrics",
                "evaluation_status": "diagnostic_validation_checkpoint_selected_on_same_split",
                "checkpoint": {"sha256": sha},
                "labels": ["Handle", "Head"],
                "split": {"name": "val", "seed": 42, "records": records},
                "evaluation_args": {"repeats": 5, "seed": 20260805, "amp": False},
                "aggregate": {
                    "mean_iou": value,
                    "accuracy": value + 0.05,
                    "per_class": [
                        {"label": "Handle", "iou": value - 0.01},
                        {"label": "Head", "iou": value + 0.01},
                    ],
                    "confusion_matrix": [[10, 1], [2, 9]],
                },
            }
            conditions = []
            for mode in ("support", "reference"):
                for condition in EXPECTED_CONDITIONS:
                    if condition.startswith("count"):
                        kind = "resample_count"
                    elif condition.startswith("dropout"):
                        kind = "dropout"
                    elif condition.startswith("noise"):
                        kind = "gaussian_noise"
                    elif condition.startswith("crop"):
                        kind = "view_crop"
                    elif condition.startswith("outlier"):
                        kind = "aabb_shell_outliers"
                    else:
                        kind = "normal_facing_view"
                    conditions.append(
                        {
                            "normalization_mode": mode,
                            "condition": condition,
                            "condition_kind": kind,
                            "condition_value": 0.5,
                            "mean_iou": value - 0.05,
                            "accuracy": value,
                            "delta_miou_vs_aggregate_reference": -0.05,
                            "label_agreement_with_reference": 0.9,
                            "cosine_mean": 0.95,
                            "cosine_p05_mean": 0.9,
                            "embedding_l2_mean": 0.1,
                            "query_coord_drift_mean": 0.0,
                            "confusion_matrix": [[8, 2], [3, 7]],
                        }
                    )
            r2 = {
                "experiment": "R2_fixed_query_support_perturbation",
                "evaluation_status": "diagnostic_validation_checkpoint_selected_on_same_split",
                "checkpoint": {"sha256": sha},
                "labels": ["Handle", "Head"],
                "split": {"name": "val", "seed": 42, "records": records},
                "evaluation_args": {"trials": 5, "seed": 20260805, "amp": False},
                "fixed_query_protocol": {
                    "query_count": 2048,
                    "normalization_modes": ["support", "reference"],
                },
                "reference": {
                    "mean_iou": value,
                    "confusion_matrix": [[10, 1], [2, 9]],
                },
                "aggregate_conditions": conditions,
            }
            reports[seed][variant] = {"r1": r1, "r2": r2}
    return {"entries": entries}, reports


class TestHammerLossAblationEvaluation(unittest.TestCase):
    def test_audit_accepts_complete_strict_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = audit_matrix(_write_fake_matrix(Path(directory)))
        self.assertTrue(manifest["matrix_complete"])
        self.assertEqual(15, manifest["expected_runs"])
        self.assertEqual(15, len(manifest["entries"]))

    def test_audit_rejects_missing_and_loss_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            roots = _write_fake_matrix(Path(directory))
            paths = expected_paths(roots, 4000, 42)
            paths[(SEEDS[0], "full_fixed")]["best_sem"].unlink()
            with self.assertRaises(FileNotFoundError):
                audit_matrix(roots)
        with tempfile.TemporaryDirectory() as directory:
            roots = _write_fake_matrix(Path(directory))
            path = expected_paths(roots, 4000, 42)[(SEEDS[0], "no_ce")]["config"]
            config = json.loads(path.read_text(encoding="utf-8"))
            config["sem_ce_weight"] = 1.0
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaises(ValueError):
                audit_matrix(roots)
    def test_historical_best_budget_is_allowed_but_not_arbitrary_or_last(self):
        with tempfile.TemporaryDirectory() as directory:
            roots = _write_fake_matrix(
                Path(directory),
                test_ratio=0.2,
                epochs=1750,
                artifact_epoch_tag=4000,
            )
            manifest = audit_matrix(
                roots,
                epochs=1750,
                artifact_epoch_tag=4000,
                evaluation_protocol="independent_test",
                expected_test_ratio=0.2,
            )
            self.assertEqual(15, len(manifest["entries"]))
            path = expected_paths(roots, 4000, 42)[(SEEDS[0], "full_fixed")]["best_sem"]
            payload = torch.load(path, map_location="cpu", weights_only=True)
            payload["args"]["epochs"] = 3999
            payload["args"]["save_every"] = 3999
            torch.save(payload, path)
            with self.assertRaises(ValueError):
                audit_matrix(roots, epochs=1750, artifact_epoch_tag=4000)

        with tempfile.TemporaryDirectory() as directory:
            roots = _write_fake_matrix(
                Path(directory),
                test_ratio=0.2,
                epochs=1750,
                artifact_epoch_tag=4000,
            )
            path = expected_paths(roots, 4000, 42)[(SEEDS[0], "full_fixed")]["last"]
            payload = torch.load(path, map_location="cpu", weights_only=True)
            payload["args"]["epochs"] = 4000
            payload["args"]["save_every"] = 4000
            torch.save(payload, path)
            with self.assertRaises(ValueError):
                audit_matrix(roots, epochs=1750, artifact_epoch_tag=4000)


    def test_plan_has_exactly_30_locked_commands(self):
        manifest = {
            "entries": [
                {
                    "seed": seed,
                    "variant": variant,
                    "checkpoint": f"/{seed}/{variant}/best_sem.pt",
                    "checkpoint_sha256": f"sha-{seed}-{variant}",
                }
                for seed in SEEDS
                for variant in VARIANTS
            ]
        }
        commands = build_evaluation_commands(
            manifest,
            python_bin=Path("/env/python"),
            dataset_root=Path("/data"),
            alias_config=Path("/hammer.json"),
            utonia_checkpoint=Path("/utonia.pth"),
            output_root=Path("/eval"),
            device="cuda:0",
            evaluation_seed=20260805,
        )
        self.assertEqual(30, len(commands))
        r2 = next(command["argv"] for command in commands if command["kind"] == "r2")
        self.assertIn("--outlier-ratios", r2)
        self.assertIn("--normal-view-keep-ratios", r2)
        self.assertNotIn("--amp", r2)

    def test_independent_test_rejects_current_matrix_and_accepts_retrained_split(self):
        with tempfile.TemporaryDirectory() as directory:
            roots = _write_fake_matrix(Path(directory))
            with self.assertRaises(ValueError):
                audit_matrix(roots, evaluation_protocol="independent_test")
        with tempfile.TemporaryDirectory() as directory:
            roots = _write_fake_matrix(
                Path(directory),
                test_ratio=0.2,
                epochs=1750,
                artifact_epoch_tag=4000,
            )
            manifest = audit_matrix(
                roots,
                evaluation_protocol="independent_test",
                epochs=1750,
                artifact_epoch_tag=4000,
                expected_test_ratio=0.2,
            )
        self.assertTrue(manifest["paper_claim_eligible"])
        self.assertEqual("test", manifest["evaluation_split"])
        self.assertEqual(1750, manifest["training_epochs"])
        self.assertEqual(4000, manifest["artifact_epoch_tag"])
        commands = build_evaluation_commands(
            manifest,
            python_bin=Path("/env/python"),
            dataset_root=Path("/data"),
            alias_config=Path("/hammer.json"),
            utonia_checkpoint=Path("/utonia.pth"),
            output_root=Path("/eval"),
            device="cuda:0",
            evaluation_seed=20260805,
        )
        self.assertTrue(all("test" in command["output_dir"] for command in commands))
        self.assertTrue(
            all(
                command["argv"][command["argv"].index("--split") + 1] == "test"
                for command in commands
            )
        )

    def test_validate_is_cpu_only_and_rejects_stale_protocol(self):
        manifest = {
            "split_seed": 42,
            "evaluation_split": "test",
            "expected_evaluation_status": "held_out_test",
            "non_loss_protocol": {"val_ratio": 0.1, "test_ratio": 0.2},
            "entries": [
                {
                    "seed": SEEDS[0],
                    "variant": "full_fixed",
                    "checkpoint": "/checkpoint.pt",
                    "checkpoint_sha256": "checkpoint-sha",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            commands = build_evaluation_commands(
                manifest,
                python_bin=Path("/env/python"),
                dataset_root=Path("/data"),
                alias_config=Path("/hammer.json"),
                utonia_checkpoint=Path("/utonia.pth"),
                output_root=Path(directory),
                device="cuda:0",
                evaluation_seed=20260805,
            )
            for command in commands:
                output_dir = Path(command["output_dir"])
                output_dir.mkdir(parents=True)
                count_key = "repeats" if command["kind"] == "r1" else "trials"
                result = {
                    "experiment": command["experiment"],
                    "evaluation_status": "held_out_test",
                    "checkpoint": {"sha256": "checkpoint-sha"},
                    "split": {
                        "name": "test",
                        "seed": 42,
                        "val_ratio": 0.1,
                        "test_ratio": 0.2,
                    },
                    "evaluation_args": {
                        "seed": 20260805,
                        count_key: 5,
                        "amp": False,
                    },
                }
                (output_dir / "results.json").write_text(json.dumps(result))
            self.assertEqual(2, validate_commands(commands)["validated"])
            stale_path = Path(commands[0]["output_dir"]) / "results.json"
            stale = json.loads(stale_path.read_text())
            stale["split"]["name"] = "val"
            stale_path.write_text(json.dumps(stale))
            with self.assertRaises(ValueError):
                validate_commands(commands)

    def test_plan_and_manifest_writes_refuse_mismatch(self):
        command = {
            "argv": ["python", "evaluate.py"],
            "label": "one",
            "kind": "r1",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_plan([command], root)
            write_plan([command], root)
            with self.assertRaises(FileExistsError):
                write_plan([{**command, "label": "changed"}], root)
            manifest_path = root / "manifest.json"
            first = {"audit_created_at_utc": "first", "matrix_complete": True}
            second = {"audit_created_at_utc": "second", "matrix_complete": True}
            self.assertEqual(first, write_manifest_fail_closed(manifest_path, first))
            self.assertEqual(first, write_manifest_fail_closed(manifest_path, second))
            with self.assertRaises(FileExistsError):
                write_manifest_fail_closed(
                    manifest_path,
                    {"audit_created_at_utc": "third", "matrix_complete": False},
                )

    def test_summary_uses_training_seed_and_rejects_incomplete_r2(self):
        manifest, reports = _fake_reports()
        summary, tables = aggregate_reports(manifest, reports)
        self.assertFalse(summary["paper_claim_eligible"])
        self.assertEqual(
            3, summary["r1"]["full_fixed"]["mean_iou"]["n_training_seeds"]
        )
        self.assertEqual(15, len(tables["r1_seed_metrics.csv"]))
        self.assertEqual(30, len(tables["r1_per_part.csv"]))
        reports[SEEDS[0]]["full_fixed"]["r2"]["aggregate_conditions"].pop()
        with self.assertRaises(ValueError):
            validate_matched_protocol(reports)

    def test_summary_independent_claim_gate_cannot_be_enabled_by_flag_only(self):
        manifest, reports = _fake_reports()
        manifest.update(
            {
                "evaluation_protocol": "independent_test",
                "evaluation_split": "test",
                "expected_evaluation_status": "held_out_test",
                "paper_claim_eligible": True,
            }
        )
        for variants in reports.values():
            for payload in variants.values():
                for kind in ("r1", "r2"):
                    payload[kind]["evaluation_status"] = "held_out_test"
                    payload[kind]["split"]["name"] = "test"
        with self.assertRaises(ValueError):
            aggregate_reports(manifest, reports)
        manifest["non_loss_protocol"] = {"val_ratio": 0.1, "test_ratio": 0.2}
        summary, _ = aggregate_reports(manifest, reports)
        self.assertTrue(summary["paper_claim_eligible"])
        self.assertEqual("independent_test", summary["evaluation_protocol"])


if __name__ == "__main__":
    unittest.main()
