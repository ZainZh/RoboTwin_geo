from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np


def _build_route(trials: list[dict], weight: float) -> dict:
    selected = {}
    matching = [row for row in trials if np.isclose(float(row["direction_weight"]), float(weight))]
    for row in matching:
        transform = row.get("predicted_transform_a_from_b")
        if transform is None:
            raise ValueError(
                "validation trial is missing predicted_transform_a_from_b; rerun "
                "validate_ndf_shoe_ramp_se3.py with the updated exporter"
            )
        shoe_id = str(int(row["query_shoe_id"]))
        energy = float(row["total_energy"])
        current = selected.get(shoe_id)
        if current is not None and float(current["solver_energy"]) <= energy:
            continue
        selected[shoe_id] = {
            "goal_T_A_from_B": np.asarray(transform, dtype=np.float64).reshape(4, 4).tolist(),
            "solver_energy": energy,
            "confidence": 1.0,
            "trial": int(row.get("trial", 0)),
        }
    return selected


def build_goal_table(
    validation: Mapping,
    *,
    no_direction_weight: float,
    direction_weight: float,
) -> dict:
    trials = list(validation.get("trials", []))
    if not trials:
        raise ValueError("validation JSON contains no trials")
    return {
        "schema_version": 1,
        "transform_convention": "T_A_from_B",
        "routes": {
            "ndf_no_direction": {
                "direction_weight": float(no_direction_weight),
                "goals": _build_route(trials, no_direction_weight),
            },
            "ndf_direction": {
                "direction_weight": float(direction_weight),
                "goals": _build_route(trials, direction_weight),
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert NDF shoe-ramp validation trials into DP3 goal tables.")
    parser.add_argument("validation_json")
    parser.add_argument("output_json")
    parser.add_argument("--no_direction_weight", type=float, default=0.0)
    parser.add_argument("--direction_weight", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validation = json.loads(Path(args.validation_json).read_text(encoding="utf-8"))
    table = build_goal_table(
        validation,
        no_direction_weight=args.no_direction_weight,
        direction_weight=args.direction_weight,
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(table, indent=2), encoding="utf-8")
    counts = {
        route: len(route_data["goals"])
        for route, route_data in table["routes"].items()
    }
    print(f"wrote {output}: {counts}")


if __name__ == "__main__":
    main()
