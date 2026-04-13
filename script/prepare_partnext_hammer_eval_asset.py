from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from partnext_hammer_eval_utils import (
    build_partnext_hammer_asset,
    render_preview_png,
    write_asset_package,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a RobotWin hammer asset from PartNext data.")
    parser.add_argument("--partnext_dir", type=Path, required=True)
    parser.add_argument("--annotation_path", type=Path, required=True)
    parser.add_argument("--output_modelname", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "assets" / "objects",
    )
    parser.add_argument("--reference_model_data", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prepared_asset = build_partnext_hammer_asset(
        partnext_dir=args.partnext_dir,
        annotation_path=args.annotation_path,
        output_modelname=args.output_modelname,
        reference_model_data_path=args.reference_model_data,
    )

    mesh = trimesh.load(prepared_asset.visual_glb_path, force="mesh")
    contact_pose = np.asarray(prepared_asset.model_data["contact_points_pose"][0], dtype=np.float64)
    functional_pose = np.asarray(prepared_asset.model_data["functional_matrix"][0], dtype=np.float64)
    preview_png = render_preview_png(
        mesh=mesh,
        contact_point=contact_pose[:3, 3],
        functional_point=functional_pose[:3, 3],
        handle_axis=contact_pose[:3, 0],
    )
    asset_dir = write_asset_package(
        output_root=args.output_root,
        prepared_asset=prepared_asset,
        preview_png=preview_png,
    )

    summary = {
        "asset_dir": str(asset_dir),
        "selected_glb": prepared_asset.source_meta["glb_dst"],
        "model_id": prepared_asset.source_meta["model_id"],
        "scale": prepared_asset.model_data["scale"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
