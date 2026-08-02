#!/usr/bin/env python3
"""Merge nominal task-flow data with on-policy expert recovery episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


EPISODE_KEYS = ("episode_flow", "episode_pose", "episode_shoe_id")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def reference_flows(raw_dir: Path, episode_count: int) -> np.ndarray:
    values = []
    for episode in range(int(episode_count)):
        path = raw_dir / "reference_flow" / f"episode{episode}.npz"
        with np.load(path, allow_pickle=False) as archive:
            values.append(np.asarray(archive["predicted_flow"], dtype=np.float32))
    return np.stack(values)


def merge_payloads(
    nominal: dict[str, np.ndarray],
    recoveries: list[tuple[dict[str, np.ndarray], np.ndarray, dict]],
) -> dict[str, np.ndarray]:
    result = {key: value.copy() for key, value in nominal.items()}
    nominal_samples = int(len(nominal["state"]))
    sample_keys = [
        key
        for key, value in nominal.items()
        if key not in EPISODE_KEYS and value.ndim > 0 and len(value) == nominal_samples
    ]
    sample_parts = {key: [nominal[key]] for key in sample_keys}
    episode_parts = {key: [nominal[key]] for key in EPISODE_KEYS}
    recovery_flags = [np.zeros(nominal_samples, dtype=np.float32)]
    recovery_policy_actions = [np.zeros(nominal_samples, dtype=np.int64)]
    recovery_source_seeds = [np.full(nominal_samples, -1, dtype=np.int64)]
    episode_offset = int(len(nominal["episode_flow"]))
    episode_id_offset = int(np.max(nominal["episode_id"])) + 1

    for recovery, predicted_flow, manifest in recoveries:
        recovery_samples = int(len(recovery["state"]))
        recovery_episodes = int(len(recovery["episode_flow"]))
        if len(predicted_flow) != recovery_episodes:
            raise ValueError("reference-flow count does not match recovery episodes")
        for key in sample_keys:
            value = recovery[key].copy()
            if key == "episode_index":
                value = value + episode_offset
            elif key == "episode_id":
                value = value + episode_id_offset
            sample_parts[key].append(value.astype(nominal[key].dtype, copy=False))
        episode_parts["episode_flow"].append(predicted_flow.astype(np.float32))
        episode_parts["episode_pose"].append(recovery["episode_pose"])
        episode_parts["episode_shoe_id"].append(recovery["episode_shoe_id"])
        recovery_flags.append(np.ones(recovery_samples, dtype=np.float32))
        records = manifest.get("records", [])
        if len(records) != recovery_episodes:
            raise ValueError("recovery manifest record count does not match episodes")
        record_by_episode = {int(record["episode"]): record for record in records}
        if set(record_by_episode) != set(range(recovery_episodes)):
            raise ValueError("recovery manifest must contain every episode exactly once")
        local_episode = np.asarray(recovery["episode_index"], dtype=np.int64)
        depth_by_episode = np.asarray(
            [record_by_episode[index]["policy_actions"] for index in range(recovery_episodes)],
            dtype=np.int64,
        )
        seed_by_episode = np.asarray(
            [record_by_episode[index]["seed"] for index in range(recovery_episodes)],
            dtype=np.int64,
        )
        recovery_policy_actions.append(depth_by_episode[local_episode])
        recovery_source_seeds.append(seed_by_episode[local_episode])
        episode_offset += recovery_episodes
        episode_id_offset += recovery_episodes

    for key, parts in sample_parts.items():
        result[key] = np.concatenate(parts, axis=0)
    for key, parts in episode_parts.items():
        result[key] = np.concatenate(parts, axis=0)
    result["is_recovery_sample"] = np.concatenate(recovery_flags)
    result["recovery_policy_actions"] = np.concatenate(recovery_policy_actions)
    result["recovery_source_seed"] = np.concatenate(recovery_source_seeds)
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--nominal", type=Path, required=True)
    result.add_argument("--nominal-prediction", type=Path, required=True)
    result.add_argument("--recovery", action="append", type=Path, required=True)
    result.add_argument("--recovery-raw", action="append", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--prediction-output", type=Path, required=True)
    result.add_argument("--overwrite", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    if len(args.recovery) != len(args.recovery_raw):
        raise ValueError("--recovery and --recovery-raw counts must match")
    for path in (args.output, args.prediction_output):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"output already exists: {path}")
    nominal = load_npz(args.nominal)
    recovery_pairs = []
    recovery_metadata = []
    for dataset_path, raw_path in zip(args.recovery, args.recovery_raw):
        payload = load_npz(dataset_path)
        predicted = reference_flows(raw_path, len(payload["episode_flow"]))
        manifest_path = raw_path / "recovery_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        recovery_pairs.append((payload, predicted, manifest))
        policy_actions = [int(record["policy_actions"]) for record in manifest["records"]]
        recovery_metadata.append(
            {
                "dataset": str(dataset_path.resolve()),
                "raw": str(raw_path.resolve()),
                "episodes": int(len(payload["episode_flow"])),
                "samples": int(len(payload["state"])),
                "policy_actions": policy_actions,
            }
        )
    merged = merge_payloads(nominal, recovery_pairs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **merged)

    with np.load(args.nominal_prediction, allow_pickle=False) as archive:
        indices = np.asarray(archive["episode_indices"], dtype=np.int64)
        nominal_prediction = np.asarray(archive["predicted_flow"], dtype=np.float32)
    order = np.argsort(indices)
    if not np.array_equal(indices[order], np.arange(len(nominal["episode_flow"]))):
        raise ValueError("nominal prediction does not cover every nominal episode")
    all_prediction = [nominal_prediction[order]]
    all_prediction.extend(predicted for _, predicted, _ in recovery_pairs)
    all_prediction = np.concatenate(all_prediction, axis=0)
    args.prediction_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.prediction_output,
        episode_indices=np.arange(len(all_prediction), dtype=np.int64),
        predicted_flow=all_prediction,
    )
    metadata = {
        "schema_version": 1,
        "nominal": str(args.nominal.resolve()),
        "nominal_samples": int(len(nominal["state"])),
        "recovery": recovery_metadata,
        "total_samples": int(len(merged["state"])),
        "total_episodes": int(len(merged["episode_flow"])),
        "recovery_samples": int(np.sum(merged["is_recovery_sample"])),
        "output": str(args.output.resolve()),
        "prediction_output": str(args.prediction_output.resolve()),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
