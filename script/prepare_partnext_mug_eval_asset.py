from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import trimesh

from partnext_hammer_eval_utils import render_preview_ply
from partnext_mug_eval_utils import (
    DEFAULT_MUG_DESCRIPTION,
    build_partnext_mug_asset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare RobotWin mug assets from PartNext Mug_new data.")
    parser.add_argument("--partnext_dir", type=Path, required=True)
    parser.add_argument("--annotation_path", type=Path, required=True)
    parser.add_argument("--output_modelname", type=str, default="partnext_mug_eval_v9")
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "assets" / "objects",
    )
    parser.add_argument("--reference_model_data", type=Path, default=None)
    parser.add_argument("--glb_name", type=str, default=None)
    parser.add_argument("--all", action="store_true", dest="prepare_all")
    parser.add_argument(
        "--description_root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "description" / "objects_description",
    )
    return parser.parse_args()


def build_preview_ply(prepared_asset) -> bytes:
    mesh = trimesh.load(prepared_asset.visual_glb_path, force="mesh")
    contact_pose = np.asarray(prepared_asset.model_data["contact_points_pose"][0], dtype=np.float64)
    target_pose = np.asarray(prepared_asset.model_data["target_pose"][0], dtype=np.float64)
    functional_pose = np.asarray(prepared_asset.model_data["functional_matrix"][0], dtype=np.float64)
    return render_preview_ply(
        mesh=mesh,
        contact_point=contact_pose[:3, 3],
        target_point=target_pose[:3, 3],
        functional_point=functional_pose[:3, 3],
    )


def _write_object_description(
    *,
    description_root: Path,
    output_modelname: str,
    model_id: int,
) -> None:
    model_desc_dir = description_root / output_modelname
    model_desc_dir.mkdir(parents=True, exist_ok=True)
    (model_desc_dir / f"base{model_id}.json").write_text(
        json.dumps(DEFAULT_MUG_DESCRIPTION, indent=2),
        encoding="utf-8",
    )


def _write_asset_variant(
    *,
    asset_dir: Path,
    prepared_asset,
    model_id: int,
    preview_ply: bytes,
    description_root: Path,
) -> None:
    (asset_dir / "visual").mkdir(parents=True, exist_ok=True)
    (asset_dir / "collision").mkdir(parents=True, exist_ok=True)
    (asset_dir / "preview").mkdir(parents=True, exist_ok=True)

    shutil.copy2(prepared_asset.visual_glb_path, asset_dir / "visual" / f"base{model_id}.glb")
    shutil.copy2(prepared_asset.collision_glb_path, asset_dir / "collision" / f"base{model_id}.glb")
    (asset_dir / f"model_data{model_id}.json").write_text(
        json.dumps(prepared_asset.model_data, indent=2),
        encoding="utf-8",
    )
    (asset_dir / "preview" / f"overview{model_id}.ply").write_bytes(preview_ply)
    _write_object_description(
        description_root=description_root,
        output_modelname=prepared_asset.modelname,
        model_id=model_id,
    )
    for cleanup_path in getattr(prepared_asset, "cleanup_paths", ()):
        try:
            cleanup_path.unlink()
        except FileNotFoundError:
            pass


def prepare_asset_packages(
    *,
    partnext_dir: Path,
    annotation_path: Path,
    output_modelname: str,
    output_root: Path,
    reference_model_data: Path | None = None,
    glb_name: str | None = None,
    prepare_all: bool = False,
    description_root: Path | None = None,
) -> list[dict]:
    description_root = description_root or (Path(__file__).resolve().parents[1] / "description" / "objects_description")
    asset_dir = output_root / output_modelname
    if asset_dir.exists():
        raise FileExistsError(f"output asset directory already exists: {asset_dir}")

    output_root.mkdir(parents=True, exist_ok=True)
    description_root.mkdir(parents=True, exist_ok=True)

    if prepare_all:
        glb_names = [path.name for path in sorted(partnext_dir.glob("*.glb"))]
        if not glb_names:
            raise FileNotFoundError(f"no .glb files found under {partnext_dir}")
    else:
        glb_names = [glb_name]

    summaries: list[dict] = []
    first_points_info = None
    source_meta_summary: list[dict] = []

    for model_id, selected_glb_name in enumerate(glb_names):
        prepared_asset = build_partnext_mug_asset(
            partnext_dir=partnext_dir,
            annotation_path=annotation_path,
            output_modelname=output_modelname,
            reference_model_data_path=reference_model_data,
            requested_glb_name=selected_glb_name,
        )
        preview_ply = build_preview_ply(prepared_asset)
        _write_asset_variant(
            asset_dir=asset_dir,
            prepared_asset=prepared_asset,
            model_id=model_id,
            preview_ply=preview_ply,
            description_root=description_root,
        )
        if first_points_info is None:
            first_points_info = prepared_asset.points_info
        source_meta_summary.append(
            {
                "model_id": model_id,
                "selected_glb": prepared_asset.source_meta["glb_dst"],
                "source_meta": prepared_asset.source_meta,
            }
        )
        summaries.append(
            {
                "asset_dir": str(asset_dir),
                "selected_glb": prepared_asset.source_meta["glb_dst"],
                "model_id": model_id,
                "scale": prepared_asset.model_data["scale"],
            }
        )

    if first_points_info is not None:
        (asset_dir / "points_info.json").write_text(
            json.dumps(first_points_info, indent=2),
            encoding="utf-8",
        )
    (asset_dir / "source_meta.json").write_text(
        json.dumps(source_meta_summary, indent=2),
        encoding="utf-8",
    )
    return summaries


def main() -> None:
    args = parse_args()
    summaries = prepare_asset_packages(
        partnext_dir=args.partnext_dir,
        annotation_path=args.annotation_path,
        output_modelname=args.output_modelname,
        output_root=args.output_root,
        reference_model_data=args.reference_model_data,
        glb_name=args.glb_name,
        prepare_all=args.prepare_all,
        description_root=args.description_root,
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
