#!/usr/bin/env python3
"""Create the minimal grouped-GLB index consumed by category NDF training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--asset-category-dir", type=Path, required=True)
    result.add_argument("--object-name", required=True)
    result.add_argument("--model-ids", nargs="+", type=int, required=True)
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--mesh-kind", choices=("visual", "collision"), default="visual")
    result.add_argument("--overwrite", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    category = args.asset_category_dir.expanduser().resolve()
    group = args.output_root.expanduser().resolve() / args.object_name
    annotation = group / "annotation.jsonl"
    if annotation.exists() and not args.overwrite:
        raise FileExistsError(annotation)
    rows = []
    for model_id in args.model_ids:
        mesh = category / args.mesh_kind / f"base{int(model_id)}.glb"
        if not mesh.is_file():
            raise FileNotFoundError(mesh)
        rows.append(
            {
                "object_name": args.object_name,
                "model_id": str(int(model_id)),
                "type_id": category.name,
                "glb_src": str(mesh),
                "glb_dst": str(mesh),
                "glb_exists": True,
                "masks": {},
                "hierarchyList": [],
            }
        )
    group.mkdir(parents=True, exist_ok=True)
    annotation.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    metadata = {
        "schema_version": 1,
        "object_name": args.object_name,
        "asset_category_dir": str(category),
        "mesh_kind": args.mesh_kind,
        "model_ids": [int(value) for value in args.model_ids],
        "count": len(rows),
        "held_out_from_policy_task": True,
        "annotation": str(annotation),
    }
    (group / "group_meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
