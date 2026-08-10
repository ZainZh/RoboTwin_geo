#!/usr/bin/env python3
"""Successor queue: legacy-priority remote work with stage42 backfill.

The currently deployed ``hammer_remote_slot_filler.py`` is intentionally left
unchanged. This successor is opt-in and requires ``--include-stage42``.
Legacy tasks stay first in priority order, while otherwise idle slots may run
stage42 before the legacy completion barrier. It reuses the tested scheduler.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
import time

import hammer_remote_slot_filler as legacy


DEFAULT_STAGE42_OUTPUT_TEMPLATE = (
    "/workspace/RoboTwin_geo/outputs/paper_revision/"
    "hammer_stage42_independent_remote6000ada_"
    "seed{seed}_e4000_split424242_20260806"
)
STAGE42_PROGRAMS = {
    "full_fixed": "hammer_stage42_remote_full_fixed_seed20260806",
    "no_ce": "hammer_stage42_remote_no_ce_seed20260806",
    "no_supcon": "hammer_stage42_remote_no_supcon_seed20260806",
    "no_consistency": "hammer_stage42_remote_no_consistency_seed20260806",
    "ce_only": "hammer_stage42_remote_ce_only_seed20260806",
}


@dataclasses.dataclass(frozen=True)
class Stage42RemoteTask(legacy.FormalTask):
    @property
    def run_name(self) -> str:
        return f"hammer_{self.variant}_e4000_split424242_seed{self.seed}"


def build_stage42_tasks(
    output_template: str = DEFAULT_STAGE42_OUTPUT_TEMPLATE,
) -> list[Stage42RemoteTask]:
    return [
        Stage42RemoteTask(
            seed="20260806",
            variant=variant,
            output_root=Path(output_template.format(seed="20260806")),
            program=program,
        )
        for variant, program in STAGE42_PROGRAMS.items()
    ]


def legacy_barrier_state(tasks: list[legacy.FormalTask]) -> dict[str, str]:
    return {
        f"{task.seed}/{task.variant}": (
            "complete" if task.outer_completion.is_file() else "incomplete"
        )
        for task in tasks
    }


def fill_once(
    *,
    legacy_tasks: list[legacy.FormalTask],
    stage42_tasks: list[Stage42RemoteTask],
    include_stage42: bool,
    max_active: int,
    supervisorctl: str,
    **injected,
) -> dict[str, object]:
    barrier = legacy_barrier_state(legacy_tasks)
    barrier_open = all(state == "complete" for state in barrier.values())
    active_phase = "stage42" if include_stage42 and barrier_open else "legacy"
    if include_stage42:
        # Preserve legacy priority, but do not leave an expensive GPU slot idle
        # when all incomplete legacy tasks are already active/reserved.
        selected = [*legacy_tasks, *stage42_tasks]
    else:
        selected = legacy_tasks
    snapshot = legacy.fill_once(
        tasks=selected,
        max_active=max_active,
        supervisorctl=supervisorctl,
        **injected,
    )
    snapshot["active_phase"] = active_phase
    snapshot["include_stage42"] = include_stage42
    snapshot["legacy_completion_barrier"] = barrier
    snapshot["legacy_barrier_open"] = barrier_open
    return snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-active", type=int, default=4)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--supervisorctl", default="/usr/local/bin/supervisorctl")
    parser.add_argument("--output-template", default=legacy.DEFAULT_OUTPUT_TEMPLATE)
    parser.add_argument("--stage42-output-template", default=DEFAULT_STAGE42_OUTPUT_TEMPLATE)
    parser.add_argument("--include-stage42", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--event-log", type=Path)
    args = parser.parse_args()
    if args.max_active < 1 or args.max_active > 4:
        parser.error("--max-active must be in [1, 4]")
    if args.poll_seconds < 1:
        parser.error("--poll-seconds must be >= 1")
    return args


def main() -> None:
    args = parse_args()
    legacy._EVENT_LOG = args.event_log
    legacy_tasks = legacy.build_tasks(args.output_template)
    stage42_tasks = build_stage42_tasks(args.stage42_output_template)
    while True:
        try:
            fill_once(
                legacy_tasks=legacy_tasks,
                stage42_tasks=stage42_tasks,
                include_stage42=args.include_stage42,
                max_active=args.max_active,
                supervisorctl=args.supervisorctl,
            )
        except Exception as error:
            legacy.emit("slot_error", error=repr(error), scheduler="stage42_successor")
            if args.once:
                raise
        if args.once:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
