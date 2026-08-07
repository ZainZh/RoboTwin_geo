#!/usr/bin/env python3
"""Create an auditable effective manifest after objective asset-gate failures."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_replacement(value: str) -> tuple[int, int, str]:
    fields = value.split(":", 2)
    if len(fields) != 3 or not fields[2].strip():
        raise argparse.ArgumentTypeError("replacement must be REJECTED:PROMOTED:REASON")
    try:
        rejected, promoted = int(fields[0]), int(fields[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("replacement ranks must be integers") from exc
    return rejected, promoted, fields[2].strip()


def build_effective_manifest(parent: dict, replacements: list[tuple[int, int, str]]) -> dict:
    assets_by_rank = {int(row["selection_rank"]): row for row in parent["assets"]}
    if len(assets_by_rank) != len(parent["assets"]):
        raise ValueError("parent manifest contains duplicate selection ranks")
    rejected_ranks = {row[0] for row in replacements}
    promoted_ranks = {row[1] for row in replacements}
    if len(rejected_ranks) != len(replacements) or len(promoted_ranks) != len(replacements):
        raise ValueError("replacement ranks must be one-to-one")

    effective_assets = [
        deepcopy(row)
        for row in parent["assets"]
        if int(row["selection_rank"]) not in rejected_ranks
    ]
    exclusions = []
    for rejected_rank, promoted_rank, reason in replacements:
        if rejected_rank not in assets_by_rank or promoted_rank not in assets_by_rank:
            raise ValueError("replacement references a missing selection rank")
        rejected = assets_by_rank[rejected_rank]
        promoted_source = assets_by_rank[promoted_rank]
        if promoted_source["partition"] != "reserve":
            raise ValueError(f"promoted rank {promoted_rank} is not frozen reserve")
        target_partition = rejected["partition"]
        for row in effective_assets:
            if int(row["selection_rank"]) == promoted_rank:
                row["original_partition"] = "reserve"
                row["partition"] = target_partition
                row["replacement_for_rank"] = rejected_rank
                row["asset_gate_status"] = "pending"
                break
        exclusions.append(
            {
                "selection_rank": rejected_rank,
                "name": rejected["name"],
                "partition": target_partition,
                "reason": reason,
                "replacement_rank": promoted_rank,
                "policy_outcomes_used": False,
            }
        )

    result = deepcopy(parent)
    result["schema_version"] = max(2, int(parent.get("schema_version", 1)))
    result["created_utc"] = datetime.now(timezone.utc).isoformat()
    result["assets"] = effective_assets
    result["effective_manifest"] = {
        "objective_asset_gate_only": True,
        "policy_outcomes_used": False,
        "exclusions": exclusions,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replacement", action="append", type=parse_replacement, required=True)
    args = parser.parse_args()
    parent = json.loads(args.parent.read_text(encoding="utf-8"))
    result = build_effective_manifest(parent, args.replacement)
    result["effective_manifest"]["parent_manifest"] = str(args.parent)
    result["effective_manifest"]["parent_manifest_sha256"] = sha256_file(args.parent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["effective_manifest"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
