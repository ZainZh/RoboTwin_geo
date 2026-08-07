#!/usr/bin/env python3
"""Dry-run-by-default max-4 local queue for stage42 independent training.

The queue treats the four currently running legacy services as guarded slots.
It fills a slot only after that service either remains active/reserved or has a
completion marker.  An inactive legacy service without completion blocks all
new starts, preventing a legacy failure from being mistaken for free capacity.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Callable, Iterable

import hammer_remote_slot_filler as process_utils
from hammer_stage42_protocol import Stage42Task, VARIANTS, build_tasks


REPO_ROOT = Path(__file__).resolve().parents[3]
CURRENT_OUTPUT_ROOT = (
    REPO_ROOT
    / "outputs/paper_revision/"
    "hammer_semantic_loss_ablation_local4090_seed20260807_"
    "e4000_fastw4_resumable_restart20260806"
)
CURRENT_UNITS = {
    "full_fixed": "hammer-local-loss@full_fixed.service",
    "no_ce": "hammer-local-loss@no_ce.service",
    "no_supcon": "hammer-local-loss@no_supcon.service",
    "no_consistency": "hammer-local-loss@no_consistency.service",
}
GRAPHICS_CUDA_EXECUTABLES = {
    "/usr/libexec/gnome-remote-desktop-daemon",
}
_EVENT_LOG: Path | None = None


@dataclasses.dataclass(frozen=True)
class UnitState:
    load: str = "not-found"
    active: str = "inactive"
    sub: str = "dead"
    pid: int | None = None

    @property
    def reserves_slot(self) -> bool:
        return self.active in {"active", "activating", "reloading"}


@dataclasses.dataclass(frozen=True)
class LegacyGuard:
    variant: str
    unit: str
    completion: Path


def build_legacy_guards() -> list[LegacyGuard]:
    return [
        LegacyGuard(
            variant=variant,
            unit=unit,
            completion=(
                CURRENT_OUTPUT_ROOT
                / f"hammer_{variant}_e4000_split42_seed20260807"
                / "completion.json"
            ),
        )
        for variant, unit in CURRENT_UNITS.items()
    ]


def parse_systemd_show(text: str) -> dict[str, UnitState]:
    result: dict[str, UnitState] = {}
    for block in text.strip().split("\n\n"):
        values = {}
        for line in block.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                values[key] = value
        unit = values.get("Id")
        if not unit:
            continue
        pid_value = values.get("MainPID", "0")
        result[unit] = UnitState(
            load=values.get("LoadState", "not-found"),
            active=values.get("ActiveState", "inactive"),
            sub=values.get("SubState", "dead"),
            pid=int(pid_value) if pid_value.isdigit() and int(pid_value) > 0 else None,
        )
    return result


def query_units(units: Iterable[str]) -> dict[str, UnitState]:
    names = tuple(dict.fromkeys(units))
    completed = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            "--property=Id",
            "--property=LoadState",
            "--property=ActiveState",
            "--property=SubState",
            "--property=MainPID",
            *names,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode not in {0, 1}:
        raise subprocess.CalledProcessError(
            completed.returncode,
            completed.args,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    parsed = parse_systemd_show(completed.stdout)
    return {name: parsed.get(name, UnitState()) for name in names}


def ignored_graphics_cuda_pids(
    cuda_pids: set[int], commands: dict[int, tuple[str, ...]]
) -> set[int]:
    ignored = set()
    for pid in cuda_pids:
        command = commands.get(pid)
        if command and command[0] in GRAPHICS_CUDA_EXECUTABLES:
            ignored.add(pid)
    return ignored


def calculate_occupancy(
    *,
    cuda_pids: set[int],
    units: dict[str, UnitState],
    parents: dict[int, int],
    commands: dict[int, tuple[str, ...]],
    managed_units: Iterable[str],
) -> tuple[int, list[str], list[int]]:
    ignored = ignored_graphics_cuda_pids(cuda_pids, commands)
    training_or_unknown_cuda = cuda_pids.difference(ignored)
    occupancy = len(training_or_unknown_cuda)
    reservations = []
    for name in managed_units:
        state = units.get(name, UnitState())
        if not state.reserves_slot:
            continue
        has_cuda_descendant = state.pid is not None and any(
            process_utils.is_descendant(pid, state.pid, parents)
            for pid in training_or_unknown_cuda
        )
        if not has_cuda_descendant:
            occupancy += 1
            reservations.append(name)
    return occupancy, reservations, sorted(ignored)


def legacy_blockers(
    guards: list[LegacyGuard], units: dict[str, UnitState]
) -> list[str]:
    blocked = []
    for guard in guards:
        state = units.get(guard.unit, UnitState())
        if state.reserves_slot or guard.completion.is_file():
            continue
        blocked.append(
            f"{guard.variant}:unit={state.active}/{state.sub},"
            f"completion_missing={guard.completion}"
        )
    return blocked


def attempt_path(task: Stage42Task) -> Path:
    return (
        REPO_ROOT
        / "outputs/paper_revision/hammer_stage42_local_queue/attempts"
        / f"{task.run_name}.json"
    )


def classify_task(
    task: Stage42Task,
    *,
    active_run_names: set[str],
    units: dict[str, UnitState],
) -> str:
    if task.completion.is_file():
        return "complete_pending_final_artifact_audit"
    if task.run_name in active_run_names:
        return "active"
    state = units.get(task.program, UnitState())
    if state.reserves_slot:
        return "reserved"
    if task.run_dir.exists():
        return "blocked_incomplete_identity_audit_required"
    if attempt_path(task).exists():
        return "blocked_prior_attempt_without_artifact"
    if state.active == "failed" or state.sub == "failed":
        return "blocked_failed_unit"
    return "pending"


def systemd_run_command(task: Stage42Task) -> list[str]:
    wrapper = REPO_ROOT / "policy/DP3/scripts/run_hammer_stage42_independent_matrix.sh"
    return [
        "systemd-run",
        "--user",
        "--collect",
        f"--unit={task.program.removesuffix('.service')}",
        "--property=MemoryHigh=9G",
        "--property=MemoryMax=12G",
        "--property=MemorySwapMax=1G",
        "--property=Restart=on-failure",
        "--property=RestartSec=20",
        "--property=TimeoutStopSec=90",
        "/usr/bin/env",
        "PYTHONUNBUFFERED=1",
        "OMP_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
        "MKL_NUM_THREADS=1",
        "NUMEXPR_NUM_THREADS=1",
        "RUN_STAGE42=1",
        "STAGE42_ROLE=local",
        f"STAGE42_SEEDS={task.seed}",
        f"STAGE42_VARIANTS={task.variant}",
        f"STAGE42_NUM_WORKERS={task.num_workers}",
        "/bin/bash",
        str(wrapper),
    ]


def write_attempt(task: Stage42Task) -> Path:
    path = attempt_path(task)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"stage42 attempt already exists: {path}")
    payload = {
        "format": "hammer_stage42_local_queue_attempt_v1",
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "task": task.key,
        "run_name": task.run_name,
        "run_dir": str(task.run_dir),
        "command": systemd_run_command(task),
    }
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return path


def start_task(task: Stage42Task) -> None:
    write_attempt(task)
    subprocess.run(systemd_run_command(task), check=True)


def emit(event: str, **payload: object) -> None:
    line = json.dumps(
        {"event": event, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **payload},
        sort_keys=True,
    )
    print(line, flush=True)
    if _EVENT_LOG is not None:
        _EVENT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _EVENT_LOG.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def fill_once(
    *,
    tasks: list[Stage42Task],
    guards: list[LegacyGuard],
    max_active: int,
    allow_start: bool,
    starter: Callable[[Stage42Task], None] | None = None,
    cuda_pids: set[int] | None = None,
    units: dict[str, UnitState] | None = None,
    parents: dict[int, int] | None = None,
    commands: dict[int, tuple[str, ...]] | None = None,
) -> dict[str, object]:
    if cuda_pids is None:
        cuda_pids = process_utils.query_cuda_pids()
    if parents is None or commands is None:
        parents, commands = process_utils.read_process_snapshot()
    managed_units = [guard.unit for guard in guards] + [task.program for task in tasks]
    if units is None:
        units = query_units(managed_units)
    occupancy, reservations, ignored = calculate_occupancy(
        cuda_pids=cuda_pids,
        units=units,
        parents=parents,
        commands=commands,
        managed_units=managed_units,
    )
    blockers = legacy_blockers(guards, units)
    active_names = process_utils.active_formal_run_names(commands)
    states = {
        task.key: classify_task(
            task, active_run_names=active_names, units=units
        )
        for task in tasks
    }
    available = max(0, max_active - occupancy)
    selected: list[Stage42Task] = []
    if not blockers:
        for task in tasks:
            if len(selected) >= available:
                break
            if classify_task(task, active_run_names=active_names, units=units) == "pending":
                selected.append(task)

    starts: list[str] = []
    would_starts = [task.program for task in selected]
    if allow_start:
        launch = start_task if starter is None else starter
        for task in selected:
            # Final filesystem/unit recheck closes the controller-side TOCTOU
            # window; the Python per-run flock closes the remaining race.
            if classify_task(task, active_run_names=active_names, units=units) != "pending":
                continue
            launch(task)
            starts.append(task.program)
            occupancy += 1

    snapshot = {
        "mode": "run" if allow_start else "dry_run",
        "cuda_pids": sorted(cuda_pids),
        "ignored_graphics_cuda_pids": ignored,
        "occupancy": occupancy,
        "max_active": max_active,
        "reservations": reservations,
        "legacy_blockers": blockers,
        "states": states,
        "would_starts": would_starts,
        "starts": starts,
    }
    emit("stage42_local_queue_snapshot", **snapshot)
    return snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-active", type=int, default=4)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--confirm-start", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--event-log", type=Path)
    args = parser.parse_args()
    if args.max_active < 1 or args.max_active > 4:
        parser.error("--max-active must be in [1, 4]")
    if args.poll_seconds < 1:
        parser.error("--poll-seconds must be >= 1")
    if args.confirm_start and os.environ.get("RUN_STAGE42") != "1":
        parser.error("--confirm-start additionally requires RUN_STAGE42=1")
    return args


def main() -> None:
    global _EVENT_LOG
    args = parse_args()
    _EVENT_LOG = args.event_log
    tasks = build_tasks("local", num_workers=0)
    guards = build_legacy_guards()
    while True:
        try:
            fill_once(
                tasks=tasks,
                guards=guards,
                max_active=args.max_active,
                allow_start=args.confirm_start,
            )
        except Exception as error:
            emit("stage42_local_queue_error", error=repr(error))
            if args.once:
                raise
        if args.once:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
