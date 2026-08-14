#!/usr/bin/env python3
"""Select failure-matched windows from mixed expert continuations.

An already-held handoff contributes its placement prefix through release;
a not-yet-held handoff contributes its grasp prefix through closure.  The
selection is training-only and does not add a phase variable or rule to the
deployed policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def cutoff_for_gripper(
    gripper: np.ndarray,
    continuation: str,
    post_transition_frames: int,
) -> tuple[int, str]:
    value = np.asarray(gripper, dtype=np.float64).reshape(-1)
    if int(post_transition_frames) < 0:
        raise ValueError("post_transition_frames must be non-negative")
    if continuation in {"placement_only", "terminal_release_only"}:
        closed = np.flatnonzero(value < 0.5)
        if not len(closed):
            raise RuntimeError("placement-only continuation never holds the gripper closed")
        first_closed = int(closed[0])
        opened = np.flatnonzero(
            (np.arange(len(value)) > first_closed) & (value >= 0.5)
        )
        if not len(opened):
            raise RuntimeError("placement-only continuation never releases")
        transition = int(opened[0])
        kind = "first_release"
    elif continuation == "full_task":
        closed = np.flatnonzero(value < 0.5)
        if not len(closed):
            raise RuntimeError("full-task continuation never closes the gripper")
        transition = int(closed[0])
        kind = "first_close"
    else:
        raise ValueError(f"unknown expert continuation {continuation!r}")
    return transition + int(post_transition_frames), kind


def terminal_release_command_labels(
    payload: dict[str, np.ndarray],
    records: dict[int, dict],
    selected_episodes: np.ndarray,
    keep: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Return actions with expert terminal open commands causally aligned."""
    action = np.asarray(payload["action"], dtype=np.float32).copy()
    if action.ndim != 3 or action.shape[-1] != 14:
        raise ValueError("terminal release command labeling requires [N,H,14] actions")
    relabeled = 0
    for episode_value in selected_episodes:
        episode = int(episode_value)
        record = records[episode]
        if str(record["expert_continuation"]) != "terminal_release_only":
            continue
        handoff_frame = int(record["policy_chunks"])
        release_rows = (
            (payload["episode_id"] == episode)
            & (payload["frame_index"] >= handoff_frame)
            & keep
        )
        action[release_rows, :, 6] = 1.0
        relabeled += int(release_rows.sum())
    return action, relabeled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--episodes",
        nargs="+",
        type=int,
        help="Optional admitted episode ids to retain; defaults to all episodes.",
    )
    parser.add_argument("--post-transition-frames", type=int, default=15)
    parser.add_argument(
        "--terminal-release-command-label",
        action="store_true",
        help=(
            "For terminal-release continuations, supervise the expert's "
            "high-level open command from the admitted handoff onward instead "
            "of the actuator's delayed observed ramp."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {args.output}")
    with np.load(args.input, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = {int(record["episode"]): record for record in manifest["records"]}
    sample_count = len(payload["episode_id"])
    keep = np.zeros(sample_count, dtype=bool)
    summary = {}
    available = np.unique(np.asarray(payload["episode_id"], dtype=np.int64))
    selected_episodes = (
        available
        if args.episodes is None
        else np.asarray(sorted(set(args.episodes)), dtype=np.int64)
    )
    missing = sorted(set(selected_episodes.tolist()) - set(available.tolist()))
    if missing:
        raise KeyError(f"archive lacks selected episodes {missing}")
    for episode in selected_episodes:
        episode = int(episode)
        if episode not in records:
            raise KeyError(f"manifest lacks episode {episode}")
        with h5py.File(args.data_dir / f"episode{episode}.hdf5", "r") as root:
            gripper = np.asarray(root["endpose/left_gripper"], dtype=np.float64)
        continuation = str(records[episode]["expert_continuation"])
        cutoff, transition_kind = cutoff_for_gripper(
            gripper, continuation, args.post_transition_frames
        )
        selected = (payload["episode_id"] == episode) & (
            payload["frame_index"] <= cutoff
        )
        keep |= selected
        summary[str(episode)] = {
            "expert_continuation": continuation,
            "transition_kind": transition_kind,
            "last_kept_frame": int(cutoff),
            "kept_samples": int(selected.sum()),
        }
    relabeled_samples = 0
    if args.terminal_release_command_label:
        action, relabeled_samples = terminal_release_command_labels(
            payload, records, selected_episodes, keep
        )
        payload["action"] = action
    output = {}
    for key, value in payload.items():
        output[key] = (
            value[keep]
            if value.ndim > 0 and len(value) == sample_count
            else value
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)
    metadata = {
        "schema_version": 1,
        "input": str(args.input.resolve()),
        "data_dir": str(args.data_dir.resolve()),
        "manifest": str(args.manifest.resolve()),
        "output": str(args.output.resolve()),
        "input_samples": int(sample_count),
        "kept_samples": int(keep.sum()),
        "post_transition_frames": int(args.post_transition_frames),
        "terminal_release_command_label": bool(
            args.terminal_release_command_label
        ),
        "terminal_release_relabeled_samples": int(relabeled_samples),
        "selected_episodes": selected_episodes.astype(int).tolist(),
        "episodes": summary,
        "deployment_change": False,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
