#!/usr/bin/env python3
"""Replace observed gripper ramps with causal binary command targets.

Dense EEF demonstrations store the physical gripper position at the next
observation.  That value is feedback, not the high-level open/close command
that a policy must emit.  This utility detects each command onset from the
sign of the observed gripper motion and rebuilds every action horizon from the
resulting held binary command.  Cartesian action channels and all observation
fields are copied exactly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


GRIPPER_SPECS = ((9, 6, "left"), (19, 13, "right"))


def causal_command_sequence(
    current: np.ndarray,
    observed_next: np.ndarray,
    *,
    motion_threshold: float,
) -> tuple[np.ndarray, list[dict[str, int | str]]]:
    """Infer the held command from physical feedback motion onsets."""
    current = np.asarray(current, dtype=np.float64).reshape(-1)
    observed_next = np.asarray(observed_next, dtype=np.float64).reshape(-1)
    if current.shape != observed_next.shape or not len(current):
        raise ValueError("current and observed-next grippers must be equal non-empty vectors")
    if float(motion_threshold) <= 0.0:
        raise ValueError("motion threshold must be positive")
    command_open = bool(current[0] >= 0.5)
    command = np.empty(len(current), dtype=np.float32)
    transitions: list[dict[str, int | str]] = []
    for row, delta in enumerate(observed_next - current):
        next_open = command_open
        if float(delta) < -float(motion_threshold):
            next_open = False
        elif float(delta) > float(motion_threshold):
            next_open = True
        if next_open != command_open:
            transitions.append(
                {"row": int(row), "command": "open" if next_open else "close"}
            )
        command_open = next_open
        command[row] = float(command_open)
    return command, transitions


def relabel_payload(
    payload: dict[str, np.ndarray], *, motion_threshold: float
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    action = np.asarray(payload["action"], dtype=np.float32)
    state = np.asarray(payload["state"], dtype=np.float32)
    episode_id = np.asarray(payload["episode_id"], dtype=np.int64).reshape(-1)
    if action.ndim != 3 or action.shape[-1] != 14:
        raise ValueError(f"action must be [N,H,14], got {action.shape}")
    if state.ndim != 3 or state.shape[-1] != 20 or len(state) != len(action):
        raise ValueError(f"state must be sample-aligned [N,T,20], got {state.shape}")
    if episode_id.shape != (len(action),):
        raise ValueError("episode_id must have one value per action row")

    relabeled = action.copy()
    diagnostics: dict[str, dict] = {}
    for episode in np.unique(episode_id):
        rows = np.flatnonzero(episode_id == int(episode))
        if "frame_index" in payload:
            frame = np.asarray(payload["frame_index"])[rows]
            if np.any(np.diff(frame) < 0):
                raise ValueError(f"episode {int(episode)} rows are not time ordered")
        episode_summary: dict[str, object] = {"samples": int(len(rows))}
        for state_index, action_index, arm in GRIPPER_SPECS:
            command, transitions = causal_command_sequence(
                state[rows, -1, state_index],
                action[rows, 0, action_index],
                motion_threshold=motion_threshold,
            )
            for position, row in enumerate(rows):
                future = np.minimum(
                    np.arange(position, position + action.shape[1]), len(rows) - 1
                )
                relabeled[row, :, action_index] = command[future]
            episode_summary[arm] = {
                "initial_command": "open" if command[0] >= 0.5 else "close",
                "transitions": transitions,
                "open_targets": int((command >= 0.5).sum()),
                "closed_targets": int((command < 0.5).sum()),
            }
        diagnostics[str(int(episode))] = episode_summary

    result = {key: np.asarray(value).copy() for key, value in payload.items()}
    result["action"] = relabeled
    return result, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--motion-threshold", type=float, default=1e-3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    with np.load(args.input, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    output, diagnostics = relabel_payload(
        payload, motion_threshold=float(args.motion_threshold)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)
    transition_counts = {
        arm: int(
            sum(len(item[arm]["transitions"]) for item in diagnostics.values())
        )
        for _, _, arm in GRIPPER_SPECS
    }
    metadata = {
        "schema_version": 1,
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "samples": int(len(output["episode_id"])),
        "episodes": int(len(diagnostics)),
        "motion_threshold": float(args.motion_threshold),
        "transition_counts": transition_counts,
        "diagnostics": diagnostics,
        "deployment_change": False,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: metadata[key] for key in (
        "output", "samples", "episodes", "motion_threshold", "transition_counts"
    )}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
