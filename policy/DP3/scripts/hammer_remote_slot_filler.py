#!/usr/bin/env python3
"""Fail-closed supervisor slot filler for the remote Hammer formal matrix."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Callable, Iterable


FORMAL_SEEDS = ("20260805", "20260806")
PRIORITY_VARIANTS = ("no_ce", "no_supcon")
ALL_VARIANTS = ("full_fixed", "no_ce", "no_supcon", "no_consistency")
ORIGINAL_PROGRAMS = {
    "20260805": "hammer_seed20260805",
    "20260806": "hammer_seed20260806",
}
TASK_PROGRAMS = {
    ("20260805", "no_ce"): "hammer_slot_nce_seed20260805",
    ("20260806", "no_ce"): "hammer_slot_nce_seed20260806",
    ("20260805", "no_supcon"): "hammer_slot_nsup_seed20260805",
    ("20260806", "no_supcon"): "hammer_slot_nsup_seed20260806",
}
DEFAULT_OUTPUT_TEMPLATE = (
    "/workspace/RoboTwin_geo/outputs/paper_revision/"
    "hammer_semantic_loss_ablation_remote_seed{seed}_e4000_20260806"
)

_EVENT_LOG: Path | None = None

@dataclasses.dataclass(frozen=True)
class SupervisorProcess:
    state: str
    pid: int | None


@dataclasses.dataclass(frozen=True)
class FormalTask:
    seed: str
    variant: str
    output_root: Path
    program: str

    @property
    def run_name(self) -> str:
        return f"hammer_{self.variant}_e4000_split42_seed{self.seed}"

    @property
    def run_dir(self) -> Path:
        return self.output_root / self.run_name

    @property
    def inner_completion(self) -> Path:
        return self.run_dir / "completion.json"

    @property
    def outer_completion(self) -> Path:
        return self.output_root / "completed" / f"{self.run_name}.complete"

    @property
    def outer_manifest(self) -> Path:
        return self.output_root / "manifests" / f"{self.run_name}.sha256"


def build_tasks(output_template: str = DEFAULT_OUTPUT_TEMPLATE) -> list[FormalTask]:
    # Variant-major ordering starts both No-CE seeds before either faster
    # No-SupCon task, minimizing the matrix's long critical path.
    return [
        FormalTask(
            seed=seed,
            variant=variant,
            output_root=Path(output_template.format(seed=seed)),
            program=TASK_PROGRAMS[(seed, variant)],
        )
        for variant in PRIORITY_VARIANTS
        for seed in FORMAL_SEEDS
    ]


def parse_supervisor_status(text: str) -> dict[str, SupervisorProcess]:
    result: dict[str, SupervisorProcess] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        name, state = fields[0], fields[1].upper()
        match = re.search(r"\bpid\s+(\d+)\b", line)
        result[name] = SupervisorProcess(
            state=state,
            pid=int(match.group(1)) if match else None,
        )
    return result


def read_process_snapshot(
    proc_root: Path = Path("/proc"),
) -> tuple[dict[int, int], dict[int, tuple[str, ...]]]:
    parents: dict[int, int] = {}
    commands: dict[int, tuple[str, ...]] = {}
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            stat_fields = (entry / "stat").read_text(encoding="utf-8").split()
            parents[pid] = int(stat_fields[3])
            raw = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
        commands[pid] = tuple(
            item.decode("utf-8", errors="replace")
            for item in raw.split(b"\0")
            if item
        )
    return parents, commands


def run_name_from_command(command: Iterable[str]) -> str | None:
    tokens = tuple(command)
    for index, token in enumerate(tokens[:-1]):
        if token == "--run-name":
            return tokens[index + 1]
    return None


def active_formal_run_names(
    commands: dict[int, tuple[str, ...]],
) -> set[str]:
    result = set()
    for command in commands.values():
        joined = " ".join(command)
        if "train_semantic_field_loss_ablation_" not in joined:
            continue
        run_name = run_name_from_command(command)
        if run_name:
            result.add(run_name)
    return result


def is_descendant(pid: int, ancestor: int, parents: dict[int, int]) -> bool:
    seen: set[int] = set()
    current = pid
    while current > 1 and current not in seen:
        if current == ancestor:
            return True
        seen.add(current)
        current = parents.get(current, 0)
    return current == ancestor


def calculate_occupancy(
    cuda_pids: set[int],
    supervisor: dict[str, SupervisorProcess],
    parents: dict[int, int],
    managed_programs: Iterable[str],
) -> tuple[int, list[str]]:
    # Every CUDA compute process consumes a slot, even if it is an unrelated
    # workload. This makes the cap conservative on a shared or changed instance.
    occupancy = len(cuda_pids)
    reservations: list[str] = []
    for program in managed_programs:
        process = supervisor.get(program)
        if process is None:
            continue
        if process.state == "STARTING":
            occupancy += 1
            reservations.append(program)
            continue
        if process.state != "RUNNING":
            continue
        has_cuda_descendant = (
            process.pid is not None
            and any(
                is_descendant(cuda_pid, process.pid, parents)
                for cuda_pid in cuda_pids
            )
        )
        if not has_cuda_descendant:
            # Covers model startup, original supervisor variant transitions,
            # and a lock waiter. It prevents a transient fifth formal process.
            occupancy += 1
            reservations.append(program)
    return occupancy, reservations


def classify_task(
    task: FormalTask,
    active_run_names: set[str],
    supervisor: dict[str, SupervisorProcess],
) -> str:
    if task.outer_completion.is_file():
        return "complete"
    if task.inner_completion.is_file():
        return "finalizing"
    if task.run_name in active_run_names:
        return "active"
    if task.run_dir.exists():
        return "blocked_incomplete"
    program = supervisor.get(task.program)
    if program and program.state in {"STARTING", "RUNNING"}:
        return "reserved"
    if program is None:
        return "missing_program"
    if program.state == "STOPPED":
        return "pending"
    # EXITED/FATAL/BACKOFF are fail-closed. A one-shot task is never retried
    # automatically after an attempt; a human must inspect its log/status.
    return f"blocked_supervisor_{program.state.lower()}"


def query_cuda_pids() -> set[int]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = set()
    for line in completed.stdout.splitlines():
        token = line.strip()
        if token:
            result.add(int(token))
    return result


def query_supervisor(supervisorctl: str) -> dict[str, SupervisorProcess]:
    completed = subprocess.run(
        [supervisorctl, "status"],
        check=False,
        capture_output=True,
        text=True,
    )
    # supervisorctl status returns 3 whenever any listed program is not
    # RUNNING (normal for our STOPPED one-shot tasks). Other codes still make
    # scheduling fail closed.
    if completed.returncode not in {0, 3}:
        raise subprocess.CalledProcessError(
            completed.returncode,
            [supervisorctl, "status"],
            output=completed.stdout,
            stderr=completed.stderr,
        )
    return parse_supervisor_status(completed.stdout)


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
    tasks: list[FormalTask],
    max_active: int,
    supervisorctl: str,
    starter: Callable[[str], None] | None = None,
    cuda_pids: set[int] | None = None,
    supervisor: dict[str, SupervisorProcess] | None = None,
    parents: dict[int, int] | None = None,
    commands: dict[int, tuple[str, ...]] | None = None,
) -> dict[str, object]:
    if cuda_pids is None:
        cuda_pids = query_cuda_pids()
    if supervisor is None:
        supervisor = query_supervisor(supervisorctl)
    if parents is None or commands is None:
        parents, commands = read_process_snapshot()

    managed = list(ORIGINAL_PROGRAMS.values()) + [
        task.program for task in tasks
    ]
    occupancy, reservations = calculate_occupancy(
        cuda_pids, supervisor, parents, managed
    )
    active_names = active_formal_run_names(commands)
    states = {
        f"{task.seed}/{task.variant}": classify_task(
            task, active_names, supervisor
        )
        for task in tasks
    }
    starts: list[str] = []
    available = max(0, max_active - occupancy)
    start_program = starter
    if start_program is None:
        def start_program(program: str) -> None:
            subprocess.run([supervisorctl, "start", program], check=True)

    for task in tasks:
        if available <= 0:
            break
        state = classify_task(task, active_names, supervisor)
        if state != "pending":
            continue
        start_program(task.program)
        starts.append(task.program)
        available -= 1
        occupancy += 1
        supervisor[task.program] = SupervisorProcess("STARTING", None)

    snapshot = {
        "cuda_pids": sorted(cuda_pids),
        "occupancy": occupancy,
        "max_active": max_active,
        "reservations": reservations,
        "states": states,
        "starts": starts,
    }
    emit("slot_snapshot", **snapshot)
    return snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-active", type=int, default=4)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--supervisorctl", default="/usr/local/bin/supervisorctl")
    parser.add_argument("--output-template", default=DEFAULT_OUTPUT_TEMPLATE)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--event-log", type=Path)
    args = parser.parse_args()
    if args.max_active < 1 or args.max_active > 4:
        parser.error("--max-active must be in [1, 4]")
    if args.poll_seconds < 1:
        parser.error("--poll-seconds must be >= 1")
    return args


def main() -> None:
    global _EVENT_LOG
    args = parse_args()
    _EVENT_LOG = args.event_log
    tasks = build_tasks(args.output_template)
    while True:
        try:
            fill_once(
                tasks=tasks,
                max_active=args.max_active,
                supervisorctl=args.supervisorctl,
            )
        except Exception as error:
            # Monitoring continues, but any observation failure is fail-closed:
            # no starts occur from an uncertain snapshot.
            emit("slot_error", error=repr(error))
            if args.once:
                raise
        if args.once:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
