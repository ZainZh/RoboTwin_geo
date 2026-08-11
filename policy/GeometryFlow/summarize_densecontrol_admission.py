#!/usr/bin/env python3
"""Join structural and online-replay checks into a dense-control admission set."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--structural-manifest", type=Path, required=True)
    result.add_argument("--evaluation-root", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--require-complete", action="store_true")
    return result


def latest_episode_log(evaluation_root: Path) -> Path:
    if evaluation_root.is_file():
        return evaluation_root
    candidates = list(evaluation_root.rglob("episodes.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"no episodes.jsonl below {evaluation_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON at {path}:{line_number}") from error
    return rows


def rejection_reason(row: dict) -> str:
    metrics = row.get("metrics", {})
    if not bool(metrics.get("gripper_open", False)):
        return "gripper_not_open"
    if not bool(metrics.get("shoe_ramp_contact", False)):
        return "no_ramp_contact"
    required = int(metrics.get("success_required_hold_steps", 0) or 0)
    maximum = int(metrics.get("max_success_hold_count", 0) or 0)
    if required > 0 and maximum < required:
        return "insufficient_stable_hold"
    if metrics.get("first_raw_success_step") is None:
        return "never_reached_pose_tolerance"
    return "task_failure"


def summarize(structural: dict, replay_rows: list[dict], replay_path: Path) -> dict:
    structural_rows = {
        int(row["episode_index"]): row
        for row in structural.get("episodes", [])
        if bool(row.get("structurally_valid", False))
    }
    replay_by_episode = {}
    for row in replay_rows:
        episode = int(row["episode_index"])
        if episode in replay_by_episode:
            raise ValueError(f"duplicate replay result for episode {episode}")
        replay_by_episode[episode] = row

    admissions = []
    for episode, candidate in sorted(structural_rows.items()):
        replay = replay_by_episode.get(episode)
        if replay is None:
            admissions.append(
                {
                    **candidate,
                    "replay_success": False,
                    "admitted": False,
                    "rejection_reason": "missing_replay_result",
                }
            )
            continue
        if int(replay["seed"]) != int(candidate["seed"]):
            raise ValueError(
                f"episode {episode} seed mismatch: replay={replay['seed']} "
                f"candidate={candidate['seed']}"
            )
        success = bool(replay.get("success", False))
        admissions.append(
            {
                **candidate,
                "replay_success": success,
                "admitted": success,
                "rejection_reason": None if success else rejection_reason(replay),
                "replay_metrics": replay.get("metrics", {}),
            }
        )

    expected = int(structural.get("structurally_valid_episodes", len(structural_rows)))
    expected_ids = set(structural_rows)
    replay_ids = set(replay_by_episode)
    reasons = Counter(
        row["rejection_reason"] for row in admissions if not row["admitted"]
    )
    admitted = [row for row in admissions if row["admitted"]]
    distribution = Counter(
        f"shoe{row['shoe_id']}_{row['active_arm']}" for row in admitted
    )
    return {
        "schema_version": 1,
        "structural_manifest": str(Path(structural["data_root"]) / "data"),
        "replay_episode_log": str(replay_path.resolve()),
        "expected_replays": expected,
        "found_replays": len(replay_by_episode),
        "admitted_episodes": len(admitted),
        "rejected_episodes": len(admissions) - len(admitted),
        "complete": replay_ids == expected_ids and len(expected_ids) == expected,
        "missing_replay_episode_indices": sorted(expected_ids - replay_ids),
        "unexpected_replay_episode_indices": sorted(replay_ids - expected_ids),
        "admitted_distribution": dict(sorted(distribution.items())),
        "rejection_reasons": dict(sorted(reasons.items())),
        "episodes": admissions,
    }


def main() -> None:
    args = parser().parse_args()
    structural_path = args.structural_manifest.expanduser().resolve()
    structural = json.loads(structural_path.read_text(encoding="utf-8"))
    replay_path = latest_episode_log(args.evaluation_root.expanduser().resolve())
    manifest = summarize(structural, load_jsonl(replay_path), replay_path)
    admitted = [row for row in manifest["episodes"] if row["admitted"]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "replay_admission_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "admitted_episodes.json").write_text(
        json.dumps(
            {
                "accepted_episode_indices": [row["episode_index"] for row in admitted],
                "accepted_seeds": [row["seed"] for row in admitted],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in manifest.items() if key != "episodes"}))
    if args.require_complete and not manifest["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
