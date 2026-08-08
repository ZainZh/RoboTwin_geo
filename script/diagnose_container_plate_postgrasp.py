#!/usr/bin/env python3
"""Run the rigid post-grasp gate for container-to-plate placement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from script.diagnose_hanging_mug_postgrasp import (
    build_environment_args,
    run_motion_sequence,
    summarize,
)
from script.smoke_test_expert_task import task_instance


AVAILABLE = {
    "002_bowl": list(range(6)),
    "021_cup": list(range(13)),
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--category", choices=sorted(AVAILABLE), required=True)
    result.add_argument("--container-ids", required=True)
    result.add_argument("--seed-start", type=int, default=0)
    result.add_argument("--trials-per-id", type=int, default=1)
    result.add_argument("--translation-cm", type=float, default=1.0)
    result.add_argument("--rotation-deg", type=float, default=10.0)
    result.add_argument("--hold-steps", type=int, default=250)
    result.add_argument("--output", type=Path, required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    container_ids = [
        int(value) for value in args.container_ids.split(",") if value.strip()
    ]
    if not container_ids or not set(container_ids).issubset(AVAILABLE[args.category]):
        raise ValueError("container IDs are unavailable for the selected category")
    records = []
    trial_index = 0
    for container_id in container_ids:
        for _ in range(int(args.trials_per_id)):
            scene_seed = int(args.seed_start) + trial_index
            trial_index += 1
            environment = build_environment_args(
                "demo_clean_3d_object_pc",
                args.output.parent,
                task_name="place_container_plate",
            )
            environment["container_geometry"] = {
                "allowed_categories": [args.category],
                "allowed_container_ids": [container_id],
            }
            task = task_instance("place_container_plate")
            try:
                task.setup_demo(now_ep_num=0, seed=scene_seed, **environment)
                arm = task.prepare_policy_placement_phase()
                if not task.plan_success:
                    raise RuntimeError("scripted container grasp/lift failed")
                sequence = run_motion_sequence(task, arm, args, scene_seed)
                stages = sequence["stages"]
                record = {
                    "status": "complete",
                    "category": str(task.actor_name),
                    "container_id": int(task.container_id),
                    "scene_seed": scene_seed,
                    "command": sequence["command"],
                    "stages": stages,
                    "maximum_drift": {
                        "translation_cm": max(
                            value["drift"]["translation_cm"]
                            for value in stages.values()
                        ),
                        "rotation_deg": max(
                            value["drift"]["rotation_deg"]
                            for value in stages.values()
                        ),
                    },
                }
            except Exception as error:
                record = {
                    "status": "error",
                    "category": args.category,
                    "container_id": container_id,
                    "scene_seed": scene_seed,
                    "error": f"{type(error).__name__}: {error}",
                }
            finally:
                try:
                    task.close_env(clear_cache=False)
                except Exception:
                    pass
            records.append(record)
            print(json.dumps(record), flush=True)
    complete = [record for record in records if record["status"] == "complete"]
    task_aligned_stable = sum(
        max(
            stage["drift"]["translation_cm"]
            for stage in record["stages"].values()
        )
        <= 0.5
        and max(
            stage["drift"]["symmetry_axis_deg"]
            for stage in record["stages"].values()
        )
        <= 3.0
        for record in complete
    )
    payload = {
        "schema_version": 1,
        "protocol": "container_plate_postgrasp_rigidity_v1",
        "settings": {
            "category": args.category,
            "container_ids": container_ids,
            "trials_per_id": int(args.trials_per_id),
            "translation_cm": float(args.translation_cm),
            "rotation_deg": float(args.rotation_deg),
            "hold_steps": int(args.hold_steps),
        },
        "summary": {
            **summarize(records),
            "task_aligned_stable_5mm_3deg_axis": int(task_aligned_stable),
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **payload["summary"]}))


if __name__ == "__main__":
    main()
