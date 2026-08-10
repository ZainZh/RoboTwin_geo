"""Pure helpers for auditable RoboTwin policy-evaluation results.

This module intentionally has no simulator or torch imports so the fixed-seed
protocol and its on-disk schema can be tested on CPU-only hosts.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any
import numpy as np



EPISODE_SCHEMA = "robotwin_policy_eval_episode_v1"
SUMMARY_SCHEMA = "robotwin_policy_eval_summary_v1"


def evaluation_start_seed(seed_group: int) -> int:
    """Return RoboTwin's first candidate seed for a CLI ``--seed`` group."""
    return 100000 * (1 + int(seed_group))


def fixed_candidate_seeds(seed_group: int, test_num: int) -> list[int]:
    """Return the exact candidate list used by a fixed-seed evaluation."""
    count = int(test_num)
    if count < 0:
        raise ValueError("test_num must be non-negative")
    start_seed = evaluation_start_seed(seed_group)
    return list(range(start_seed, start_seed + count))


def fixed_seed_protocol_enabled(args: Mapping[str, Any]) -> bool:
    config = args.get("fixed_seed_protocol", False)
    if isinstance(config, Mapping):
        return bool(config.get("enabled", False))
    return bool(config)


def should_run_expert_check(args: Mapping[str, Any]) -> bool:
    """Match RoboTwin's custom-asset rule without importing the simulator."""
    custom_hammer_eval = args.get("custom_hammer_eval") or {}
    custom_mug_eval = args.get("custom_mug_eval") or {}
    return not (
        bool(custom_hammer_eval.get("enabled"))
        or bool(custom_mug_eval.get("enabled"))
    )


def fast_eval_option_enabled(args: Mapping[str, Any], option: str) -> bool:
    """Return one explicit fast-eval option without changing legacy defaults."""
    config = args.get("fast_eval") or {}
    return bool(config.get(option, False)) if isinstance(config, Mapping) else False

def policy_observation_digest(observation: Mapping[str, Any]) -> str:
    """Hash exactly the raw fields consumed by the current DP3 routes."""
    digest = hashlib.sha256()

    def add_array(name: str, value: Any) -> None:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes())

    add_array("pointcloud", observation["pointcloud"])
    add_array("joint_action/vector", observation["joint_action"]["vector"])
    for placeholder, cloud in sorted((observation.get("object_pointcloud") or {}).items()):
        add_array(f"object_pointcloud/{placeholder}", cloud)
    return digest.hexdigest()



def evaluation_should_continue(
    *,
    candidate_count: int,
    accepted_count: int,
    test_num: int,
    fixed_seed_protocol: bool,
) -> bool:
    """Choose a fixed candidate budget or the legacy accepted-seed budget."""
    budget = int(test_num)
    if budget < 0:
        raise ValueError("test_num must be non-negative")
    progress = int(candidate_count) if fixed_seed_protocol else int(accepted_count)
    return progress < budget


class EvaluationResultRecorder:
    """Append per-candidate JSONL records and atomically write one summary."""

    def __init__(self, jsonl_path: str | Path):
        self.jsonl_path = Path(jsonl_path)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl_path.write_text("", encoding="utf-8")
        self.records: list[dict[str, Any]] = []

    def record_candidate(
        self,
        *,
        candidate_index: int,
        candidate_seed: int,
        accepted: bool,
        evaluated: bool,
        success: bool | None,
        status: str,
        reason: str | None = None,
        evaluation_index: int | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
        policy_steps: int | None = None,
        elapsed_seconds: float | None = None,
        initial_observation_digest: str | None = None,
    ) -> dict[str, Any]:
        if evaluated and success is None:
            raise ValueError("evaluated candidates require a boolean success value")
        if not evaluated and success is not None:
            raise ValueError("non-evaluated candidates require success=None")
        record = {
            "schema": EPISODE_SCHEMA,
            "candidate_index": int(candidate_index),
            "candidate_seed": int(candidate_seed),
            "accepted": bool(accepted),
            "evaluated": bool(evaluated),
            "success": None if success is None else bool(success),
            "status": str(status),
            "reason": None if reason is None else str(reason),
            "evaluation_index": None if evaluation_index is None else int(evaluation_index),
            "error_type": None if error_type is None else str(error_type),
            "error_message": None if error_message is None else str(error_message),
            "policy_steps": None if policy_steps is None else int(policy_steps),
            "elapsed_seconds": None if elapsed_seconds is None else float(elapsed_seconds),
            "initial_observation_digest": initial_observation_digest,
        }
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.records.append(record)
        return record

    def build_summary(
        self,
        *,
        start_seed: int,
        next_seed: int,
        requested_test_num: int,
        fixed_seed_protocol: bool,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        evaluated = [item for item in self.records if item["evaluated"]]
        accepted = [item for item in self.records if item["accepted"]]
        successful = [item for item in evaluated if item["success"]]
        skipped = [item for item in self.records if not item["evaluated"]]
        denominator = int(requested_test_num)
        summary = {
            "schema": SUMMARY_SCHEMA,
            "start_seed": int(start_seed),
            "next_seed": int(next_seed),
            "requested_test_num": denominator,
            "fixed_seed_protocol": bool(fixed_seed_protocol),
            "candidate_count": len(self.records),
            "accepted_count": len(accepted),
            "evaluated_count": len(evaluated),
            "success_count": len(successful),
            "success_rate": (len(successful) / denominator) if denominator else None,
            "evaluated_success_rate": (len(successful) / len(evaluated)) if evaluated else None,
            "actual_candidate_seeds": [item["candidate_seed"] for item in self.records],
            "actual_accepted_seeds": [item["candidate_seed"] for item in accepted],
            "actual_evaluated_seeds": [item["candidate_seed"] for item in evaluated],
            "successful_seeds": [item["candidate_seed"] for item in successful],
            "skipped_candidates": [
                {
                    "candidate_seed": item["candidate_seed"],
                    "status": item["status"],
                    "reason": item["reason"],
                    "error_type": item["error_type"],
                    "error_message": item["error_message"],
                }
                for item in skipped
            ],
        }
        if metadata:
            summary.update(dict(metadata))
        return summary

    def write_summary(
        self,
        summary_path: str | Path,
        *,
        start_seed: int,
        next_seed: int,
        requested_test_num: int,
        fixed_seed_protocol: bool,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        summary = self.build_summary(
            start_seed=start_seed,
            next_seed=next_seed,
            requested_test_num=requested_test_num,
            fixed_seed_protocol=fixed_seed_protocol,
            metadata=metadata,
        )
        destination = Path(summary_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        return summary
