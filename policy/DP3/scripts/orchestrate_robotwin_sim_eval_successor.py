#!/usr/bin/env python3
"""Fail-closed successor for the fixed-100 RoboTwin paper evaluation matrix."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


TASK_CONFIG = "demo_clean_3d_object_pc_hammer_formal_fixed_fast"
EXPECTED_SEEDS = list(range(100000, 100100))


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def result_root(root: Path) -> Path:
    return (
        root
        / "eval_result"
        / "beat_block_hammer"
        / "DP3"
        / TASK_CONFIG
    )


def complete_summary(root: Path, label: str, min_timestamp: str) -> Path | None:
    label_root = result_root(root) / label
    candidates = sorted(label_root.glob("*/summary.json"), reverse=True)
    for path in candidates:
        if path.parent.name < min_timestamp:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("schema") != "robotwin_policy_eval_summary_v1":
            continue
        if payload.get("task_config") != TASK_CONFIG:
            continue
        if payload.get("result_label") != label:
            continue
        if payload.get("fixed_seed_protocol") is not True:
            continue
        if int(payload.get("requested_test_num", -1)) != 100:
            continue
        if int(payload.get("evaluated_count", -1)) != 100:
            continue
        if payload.get("actual_evaluated_seeds") != EXPECTED_SEEDS:
            continue
        return path
    return None


def run_checked(command: list[str], *, env: dict[str, str] | None = None) -> str:
    print(f"[eval-successor] run: {' '.join(command)}", flush=True)
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        env=env,
    )
    return ""


def local_service_state(name: str) -> str:
    completed = subprocess.run(
        ["systemctl", "--user", "is-active", name],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return completed.stdout.strip() or "missing"


def supervisor_state(name: str) -> str:
    completed = subprocess.run(
        ["supervisorctl", "status", name],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    fields = completed.stdout.split()
    if len(fields) < 2:
        raise RuntimeError(f"Cannot parse supervisor state for {name}: {completed.stdout!r}")
    return fields[1]


def wait_for_stage(
    *,
    root: Path,
    expected: dict[str, str],
    min_timestamp: str,
    interval: int,
    controller: str,
) -> dict[str, Path]:
    while True:
        completed = {
            service: complete_summary(root, label, min_timestamp)
            for service, label in expected.items()
        }
        if all(completed.values()):
            for service, path in completed.items():
                print(f"[eval-successor] complete service={service} summary={path}", flush=True)
            return {key: value for key, value in completed.items() if value is not None}

        states = {}
        for service, path in completed.items():
            if path is not None:
                states[service] = "COMPLETE"
            elif controller == "local":
                states[service] = local_service_state(service)
            else:
                states[service] = supervisor_state(service)
        print(f"[eval-successor] waiting states={states}", flush=True)

        terminal = {"failed", "inactive", "missing", "FAILED", "FATAL", "BACKOFF", "EXITED", "STOPPED"}
        bad = {name: state for name, state in states.items() if state in terminal}
        if bad:
            raise RuntimeError(
                "Evaluation service stopped without a valid fixed-100 summary: "
                f"{bad}"
            )
        time.sleep(interval)


def start_supervisor_pair(names: list[str]) -> None:
    pending = []
    for name in names:
        state = supervisor_state(name)
        if state == "RUNNING":
            continue
        if state not in {"STOPPED", "EXITED"}:
            raise RuntimeError(f"Refusing to start {name} from state {state}")
        pending.append(name)
    if pending:
        run_checked(["supervisorctl", "start", *pending])


def run_local(args: argparse.Namespace, root: Path) -> None:
    service = "robotwin-sim-fast-planner-fixed100-seed05.service"
    wait_for_stage(
        root=root,
        expected={service: "paper-sim-field-joint14-e300"},
        min_timestamp=args.min_timestamp,
        interval=args.interval,
        controller="local",
    )
    env = os.environ.copy()
    env.update(
        {
            "TEST_NUM": "100",
            "ROUTES": "xyz partprob uniformprob shuffledprob utonia",
            "TASK_CONFIG_NAME": TASK_CONFIG,
        }
    )
    runner = root / "policy" / "DP3" / "scripts" / "run_robotwin_sim_beat_fixed100_seed05.sh"
    run_checked(["bash", str(runner)], env=env)


def remote_label(route: str, seed: str) -> str:
    return f"paper-sim-{route}-joint14-e300-trainseed{seed}"


def run_remote(args: argparse.Namespace, root: Path) -> None:
    base = [
        "robotwin_sim_beat_remote_fixed100_seed06",
        "robotwin_sim_beat_remote_fixed100_seed07",
    ]
    wait_for_stage(
        root=root,
        expected={
            base[0]: remote_label("utonia", "20260806"),
            base[1]: remote_label("utonia", "20260807"),
        },
        min_timestamp=args.min_timestamp,
        interval=args.interval,
        controller="remote",
    )

    xp = [name + "_xp" for name in base]
    start_supervisor_pair(xp)
    wait_for_stage(
        root=root,
        expected={
            xp[0]: remote_label("partprob", "20260806"),
            xp[1]: remote_label("partprob", "20260807"),
        },
        min_timestamp=args.min_timestamp,
        interval=args.interval,
        controller="remote",
    )

    usu = [name.replace("_xp", "_usu") for name in xp]
    start_supervisor_pair(usu)
    wait_for_stage(
        root=root,
        expected={
            usu[0]: remote_label("shuffledprob", "20260806"),
            usu[1]: remote_label("shuffledprob", "20260807"),
        },
        min_timestamp=args.min_timestamp,
        interval=args.interval,
        controller="remote",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("local", "remote"))
    parser.add_argument("--min-timestamp", required=True)
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    if args.interval < 30:
        parser.error("--interval must be >= 30 seconds")
    return args


def main() -> int:
    args = parse_args()
    root = repo_root()
    print(
        f"[eval-successor] role={args.role} root={root} "
        f"min_timestamp={args.min_timestamp}",
        flush=True,
    )
    if args.role == "local":
        run_local(args, root)
    else:
        run_remote(args, root)
    print(f"[eval-successor] role={args.role} all stages complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
