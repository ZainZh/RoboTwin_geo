from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import trimesh

from partnext_hammer_eval_utils import (
    build_partnext_hammer_asset,
    render_preview_ply,
    write_asset_package,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a RobotWin hammer asset from PartNext data.")
    parser.add_argument("--partnext_dir", type=Path, required=True)
    parser.add_argument("--annotation_path", type=Path, required=True)
    parser.add_argument("--output_modelname", type=str, default="partnext_hammer_eval")
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "assets" / "objects",
    )
    parser.add_argument("--reference_model_data", type=Path, default=None)
    parser.add_argument("--glb_name", type=str, default=None)
    parser.add_argument("--all", action="store_true", default=True,dest="prepare_all")
    return parser.parse_args()


def make_batch_output_modelname(output_modelname: str, glb_name: str) -> str:
    stem = Path(glb_name).stem.lower()
    stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    suffix = stem or "hammer"
    return f"{output_modelname}_{suffix}"


def prepare_asset_package(
    *,
    partnext_dir: Path,
    annotation_path: Path,
    output_modelname: str,
    output_root: Path,
    reference_model_data: Path | None = None,
    glb_name: str | None = None,
) -> dict:
    prepared_asset = build_partnext_hammer_asset(
        partnext_dir=partnext_dir,
        annotation_path=annotation_path,
        output_modelname=output_modelname,
        reference_model_data_path=reference_model_data,
        requested_glb_name=glb_name,
    )

    mesh = trimesh.load(prepared_asset.visual_glb_path, force="mesh")
    contact_pose = np.asarray(prepared_asset.model_data["contact_points_pose"][0], dtype=np.float64)
    functional_pose = np.asarray(prepared_asset.model_data["functional_matrix"][0], dtype=np.float64)
    target_pose = np.asarray(prepared_asset.model_data["target_pose"][0], dtype=np.float64)
    preview_ply = render_preview_ply(
        mesh=mesh,
        contact_point=contact_pose[:3, 3],
        target_point=target_pose[:3, 3],
        functional_point=functional_pose[:3, 3],
    )
    asset_dir = write_asset_package(
        output_root=output_root,
        prepared_asset=prepared_asset,
        preview_ply=preview_ply,
    )

    return {
        "asset_dir": str(asset_dir),
        "selected_glb": prepared_asset.source_meta["glb_dst"],
        "model_id": prepared_asset.source_meta["model_id"],
        "scale": prepared_asset.model_data["scale"],
    }


def prepare_asset_packages(
    *,
    partnext_dir: Path,
    annotation_path: Path,
    output_modelname: str,
    output_root: Path,
    reference_model_data: Path | None = None,
    glb_name: str | None = None,
    prepare_all: bool = False,
    single_prepare_fn=prepare_asset_package,
) -> list[dict]:
    if prepare_all:
        glb_names = sorted(path.name for path in partnext_dir.glob("*.glb"))
        if not glb_names:
            raise FileNotFoundError(f"no .glb files found under {partnext_dir}")
        return [
            single_prepare_fn(
                partnext_dir=partnext_dir,
                annotation_path=annotation_path,
                output_modelname=make_batch_output_modelname(output_modelname, current_glb_name),
                output_root=output_root,
                reference_model_data=reference_model_data,
                glb_name=current_glb_name,
            )
            for current_glb_name in glb_names
        ]

    return [
        single_prepare_fn(
            partnext_dir=partnext_dir,
            annotation_path=annotation_path,
            output_modelname=output_modelname,
            output_root=output_root,
            reference_model_data=reference_model_data,
            glb_name=glb_name,
        )
    ]


def main() -> int:
    args = parse_args()
    summaries = prepare_asset_packages(
        partnext_dir=args.partnext_dir,
        annotation_path=args.annotation_path,
        output_modelname=args.output_modelname,
        output_root=args.output_root,
        reference_model_data=args.reference_model_data,
        glb_name=args.glb_name,
        prepare_all=args.prepare_all,
    )
    if args.prepare_all:
        print(json.dumps({"count": len(summaries), "items": summaries}, indent=2))
    else:
        print(json.dumps(summaries[0], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
