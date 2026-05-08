from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import trimesh

from partnext_hammer_eval_utils import (
    build_partnext_hammer_asset,
    render_preview_ply,
    write_asset_package,
)


DEFAULT_HAMMER_DESCRIPTION = {
    "raw_description": "hammer",
    "seen": [
        "silver hammer",
        "hammer for nails",
        "nail-driving hammer",
        "grippy handle hammer",
        "medium-sized metal hammer",
        "silver curved hammer head",
        "hammer with claw-shaped end",
        "hammer with two-tone handle",
        "plastic handle metal hammer",
        "handheld medium claw hammer",
        "yellow and black hammer grip",
        "black and yellow hammer grip",
    ],
    "unseen": [
        "hammer with black handle",
        "silver hammer head and claw",
        "hammer with claw and smooth head",
    ],
}


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
    parser.add_argument("--all", action="store_true", dest="prepare_all")
    parser.add_argument("--no_screen", action="store_true",default=True)
    parser.add_argument("--screen_task_name", type=str, default="beat_block_hammer")
    parser.add_argument("--screen_task_config", type=str, default="demo_clean_3d_partnext_objpc_hammer_eval")
    parser.add_argument("--screen_policy_name", type=str, default="DP3")
    parser.add_argument("--screen_seed_start", type=int, default=100000)
    parser.add_argument("--screen_num_seeds", type=int, default=10)
    return parser.parse_args()


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
        "model_id": 0,
        "scale": prepared_asset.model_data["scale"],
    }


def build_preview_ply(prepared_asset) -> bytes:
    mesh = trimesh.load(prepared_asset.visual_glb_path, force="mesh")
    contact_pose = np.asarray(prepared_asset.model_data["contact_points_pose"][0], dtype=np.float64)
    functional_pose = np.asarray(prepared_asset.model_data["functional_matrix"][0], dtype=np.float64)
    target_pose = np.asarray(prepared_asset.model_data["target_pose"][0], dtype=np.float64)
    return render_preview_ply(
        mesh=mesh,
        contact_point=contact_pose[:3, 3],
        target_point=target_pose[:3, 3],
        functional_point=functional_pose[:3, 3],
    )


def _default_description_root() -> Path:
    return Path(__file__).resolve().parents[1] / "description" / "objects_description"


def _write_object_description(
    *,
    description_root: Path,
    output_modelname: str,
    model_id: int,
) -> None:
    model_desc_dir = description_root / output_modelname
    model_desc_dir.mkdir(parents=True, exist_ok=True)
    (model_desc_dir / f"base{model_id}.json").write_text(
        json.dumps(DEFAULT_HAMMER_DESCRIPTION, indent=2),
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


def _build_screen_candidate_fn(
    *,
    output_root: Path,
    screen_settings: dict,
):
    import os
    import yaml

    from envs import CONFIGS_PATH
    from eval_policy import class_decorator, get_embodiment_config

    repo_root = Path(__file__).resolve().parents[1]
    task_name = screen_settings["task_name"]
    task_config = screen_settings["task_config"]
    policy_name = screen_settings["policy_name"]
    seed_start = int(screen_settings["seed_start"])
    num_seeds = int(screen_settings["num_seeds"])

    with open(repo_root / "task_config" / f"{task_config}.yml", "r", encoding="utf-8") as f:
        base_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    base_args["render_freq"] = 0
    base_args["eval_video_log"] = False
    base_args["task_name"] = task_name
    base_args["task_config"] = task_config
    base_args["ckpt_setting"] = task_config
    base_args["policy_name"] = policy_name

    embodiment_type = base_args.get("embodiment")
    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
        embodiment_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(current_embodiment_type):
        return embodiment_config[current_embodiment_type]["file_path"]

    base_args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
    base_args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
    base_args["dual_arm_embodied"] = True
    base_args["left_embodiment_config"] = get_embodiment_config(base_args["left_robot_file"])
    base_args["right_embodiment_config"] = get_embodiment_config(base_args["right_robot_file"])

    task_env = class_decorator(task_name)

    def screen_candidate(prepared_asset) -> dict:
        preview_ply = build_preview_ply(prepared_asset)
        result = {
            "accepted": False,
            "ok": 0,
            "unstable": 0,
            "target_pose_none": 0,
            "other_error": 0,
            "seed_errors": {},
        }

        with tempfile.TemporaryDirectory(prefix=".partnext_hammer_screen_", dir=output_root) as tmpdir:
            temp_modelname = Path(tmpdir).name
            temp_asset = replace(prepared_asset, modelname=temp_modelname)
            write_asset_package(
                output_root=output_root,
                prepared_asset=temp_asset,
                preview_ply=preview_ply,
            )

            current_args = dict(base_args)
            current_args["custom_hammer_eval"] = {
                "enabled": True,
                "modelname": temp_modelname,
                "model_id": 0,
            }
            for seed in range(seed_start, seed_start + num_seeds):
                try:
                    task_env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **current_args)
                    task_env.play_once()
                    result["ok"] += 1
                except Exception as exc:
                    error_text = str(exc)
                    if "unstable" in error_text.lower():
                        result["unstable"] += 1
                        error_name = "UnStableError"
                    elif "target_pose cannot be None" in error_text:
                        result["target_pose_none"] += 1
                        error_name = "target_pose_none"
                    else:
                        result["other_error"] += 1
                        error_name = type(exc).__name__
                    result["seed_errors"][str(seed)] = f"{error_name}: {error_text}"
                finally:
                    try:
                        task_env.close_env()
                    except Exception:
                        pass
        result["accepted"] = result["ok"] == num_seeds
        return result

    return screen_candidate


def prepare_asset_package_all(
    *,
    partnext_dir: Path,
    annotation_path: Path,
    output_modelname: str,
    output_root: Path,
    reference_model_data: Path | None = None,
    screen_candidates: bool = False,
    screen_candidate_fn=None,
    screen_settings: dict | None = None,
    description_root: Path | None = None,
) -> list[dict]:
    glb_names = sorted(path.name for path in partnext_dir.glob("*.glb"))
    if not glb_names:
        raise FileNotFoundError(f"no .glb files found under {partnext_dir}")

    output_root.mkdir(parents=True, exist_ok=True)
    asset_dir = output_root / output_modelname
    description_root = description_root or _default_description_root()
    if asset_dir.exists():
        raise FileExistsError(f"target asset directory already exists: {asset_dir}")

    if screen_candidates and screen_candidate_fn is None:
        if screen_settings is None:
            raise ValueError("screen_settings are required when screen_candidates=True")
        screen_candidate_fn = _build_screen_candidate_fn(
            output_root=output_root,
            screen_settings=screen_settings,
        )

    summaries: list[dict] = []
    screen_reports: list[dict] = []
    points_info = None
    for current_glb_name in glb_names:
        prepared_asset = build_partnext_hammer_asset(
            partnext_dir=partnext_dir,
            annotation_path=annotation_path,
            output_modelname=output_modelname,
            reference_model_data_path=reference_model_data,
            requested_glb_name=current_glb_name,
        )
        screen_report = None
        if screen_candidates:
            screen_report = screen_candidate_fn(prepared_asset)
            screen_reports.append(
                {
                    "selected_glb": prepared_asset.source_meta["glb_dst"],
                    **screen_report,
                }
            )
            if not screen_report.get("accepted", False):
                continue

        preview_ply = build_preview_ply(prepared_asset)
        model_id = len(summaries)
        _write_asset_variant(
            asset_dir=asset_dir,
            prepared_asset=prepared_asset,
            model_id=model_id,
            preview_ply=preview_ply,
            description_root=description_root,
        )
        if points_info is None:
            points_info = prepared_asset.points_info
        summaries.append(
            {
                "asset_dir": str(asset_dir),
                "selected_glb": prepared_asset.source_meta["glb_dst"],
                "model_id": model_id,
                "scale": prepared_asset.model_data["scale"],
                "screening": screen_report,
            }
        )

    if points_info is not None:
        (asset_dir / "points_info.json").write_text(
            json.dumps(points_info, indent=2),
            encoding="utf-8",
        )
    if screen_candidates:
        summary_path = output_root / f"{output_modelname}_screen_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "kept_model_ids": [item["model_id"] for item in summaries],
                    "kept_glbs": [item["selected_glb"] for item in summaries],
                    "screen_reports": screen_reports,
                    "screen_settings": screen_settings,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return summaries


def prepare_asset_packages(
    *,
    partnext_dir: Path,
    annotation_path: Path,
    output_modelname: str,
    output_root: Path,
    reference_model_data: Path | None = None,
    glb_name: str | None = None,
    prepare_all: bool = False,
    screen_candidates: bool = False,
    screen_candidate_fn=None,
    screen_settings: dict | None = None,
    description_root: Path | None = None,
) -> list[dict]:
    if prepare_all:
        return prepare_asset_package_all(
            partnext_dir=partnext_dir,
            annotation_path=annotation_path,
            output_modelname=output_modelname,
            output_root=output_root,
            reference_model_data=reference_model_data,
            screen_candidates=screen_candidates,
            screen_candidate_fn=screen_candidate_fn,
            screen_settings=screen_settings,
            description_root=description_root,
        )
    description_root = description_root or _default_description_root()
    summaries = [
        prepare_asset_package(
            partnext_dir=partnext_dir,
            annotation_path=annotation_path,
            output_modelname=output_modelname,
            output_root=output_root,
            reference_model_data=reference_model_data,
            glb_name=glb_name,
        )
    ]
    _write_object_description(
        description_root=description_root,
        output_modelname=output_modelname,
        model_id=0,
    )
    return summaries


def main() -> int:
    args = parse_args()
    screen_settings = {
        "task_name": args.screen_task_name,
        "task_config": args.screen_task_config,
        "policy_name": args.screen_policy_name,
        "seed_start": args.screen_seed_start,
        "num_seeds": args.screen_num_seeds,
    }
    summaries = prepare_asset_packages(
        partnext_dir=args.partnext_dir,
        annotation_path=args.annotation_path,
        output_modelname=args.output_modelname,
        output_root=args.output_root,
        reference_model_data=args.reference_model_data,
        glb_name=args.glb_name,
        prepare_all=args.prepare_all,
        screen_candidates=(args.prepare_all and not args.no_screen),
        screen_settings=screen_settings,
    )
    if args.prepare_all:
        print(json.dumps({"count": len(summaries), "items": summaries}, indent=2))
    else:
        print(json.dumps(summaries[0], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
