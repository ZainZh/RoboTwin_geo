"""Convert frozen Google Scanned Objects shoes into RoboTwin assets."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh
import vhacdx


GSO_TO_ROBOTWIN = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
DEFAULT_TARGET_LENGTH_M = 0.216
MAX_COLLISION_PART_VERTICES = 64
DEFAULT_SHOE_DESCRIPTION = {
    "raw_description": "shoe",
    "seen": [
        "low-top shoe",
        "single footwear item",
        "shoe with a sole and toe",
        "wearable shoe viewed from outside",
    ],
    "unseen": [
        "footwear with a distinct front and heel",
        "low shoe with an elongated sole",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-modelname", default="gso_shoe_blind_v1")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "assets" / "objects",
    )
    parser.add_argument(
        "--description-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "description" / "objects_description",
    )
    parser.add_argument(
        "--reference-model-data",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "assets"
        / "objects"
        / "041_shoe"
        / "model_data0.json",
    )
    parser.add_argument(
        "--reference-description",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "description"
        / "objects_description"
        / "041_shoe"
        / "base0.json",
    )
    parser.add_argument("--partitions", nargs="+", default=["development"])
    parser.add_argument("--target-length-m", type=float, default=DEFAULT_TARGET_LENGTH_M)
    parser.add_argument(
        "--max-loaded-height-m",
        type=float,
        default=0.160,
        help="Objective category-scope gate after uniform scaling.",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Replace only the exact generated model IDs selected by this invocation.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merged_scene_mesh(scene: trimesh.Scene) -> trimesh.Trimesh:
    # RoboTwin's pinned trimesh predates Scene.to_geometry().  Scene.dump with
    # concatenate=True applies node transforms and provides the same merged
    # geometry on both the old and current APIs.
    merged = scene.dump(concatenate=True)
    if isinstance(merged, trimesh.Trimesh):
        return merged
    geometries = list(merged) if isinstance(merged, (list, tuple, np.ndarray)) else []
    if not geometries:
        raise ValueError("scene contains no mergeable triangle geometry")
    return trimesh.util.concatenate(geometries)


def load_and_align_scene(obj_path: Path) -> tuple[trimesh.Scene, trimesh.Trimesh, np.ndarray]:
    scene = trimesh.load(obj_path, force="scene", process=False)
    if not isinstance(scene, trimesh.Scene) or not scene.geometry:
        raise ValueError(f"no visual geometry found in {obj_path}")
    scene.apply_transform(GSO_TO_ROBOTWIN)
    merged = merged_scene_mesh(scene)
    if not isinstance(merged, trimesh.Trimesh) or merged.is_empty:
        raise ValueError(f"could not merge geometry in {obj_path}")
    bounds = np.asarray(merged.bounds, dtype=np.float64)
    translation = np.eye(4, dtype=np.float64)
    translation[:3, 3] = [
        -0.5 * (bounds[0, 0] + bounds[1, 0]),
        -bounds[0, 1],
        -0.5 * (bounds[0, 2] + bounds[1, 2]),
    ]
    scene.apply_transform(translation)
    merged = merged_scene_mesh(scene)
    applied_transform = translation @ GSO_TO_ROBOTWIN
    return scene, merged, applied_transform


def build_model_data(
    mesh: trimesh.Trimesh,
    reference: dict,
    *,
    target_length_m: float,
) -> dict:
    extents = np.asarray(mesh.extents, dtype=np.float64)
    center = np.asarray(mesh.bounding_box.centroid, dtype=np.float64)
    if extents[2] <= 1e-8:
        raise ValueError("aligned shoe has zero longitudinal extent")
    scale_value = float(target_length_m / extents[2])
    scale = np.full((3,), scale_value, dtype=np.float64)

    height = float(extents[1])
    length = float(extents[2])
    functional_y = 0.16 * height
    contact_y = 0.52 * height

    result = copy.deepcopy(reference)
    result["center"] = center.tolist()
    result["extents"] = extents.tolist()
    result["scale"] = scale.tolist()
    result["transform_matrix"] = np.eye(4, dtype=np.float64).tolist()

    target_pose = np.asarray(result["target_pose"][0], dtype=np.float64)
    target_pose[:3, 3] = [0.0, functional_y, 0.0]
    result["target_pose"] = [target_pose.tolist()]

    functional_pose = np.asarray(result["functional_matrix"][0], dtype=np.float64)
    functional_pose[:3, 3] = [0.0, functional_y, 0.0]
    result["functional_matrix"] = [functional_pose.tolist()]

    contacts = np.asarray(result["contact_points_pose"], dtype=np.float64)
    contacts[:, :3, 3] = np.asarray([0.0, contact_y, 0.0], dtype=np.float64)
    result["contact_points_pose"] = contacts.tolist()

    orientation = np.asarray(result["orientation_point"], dtype=np.float64)
    orientation[:3, 3] = [0.0, contact_y, 0.49 * length]
    result["orientation_point"] = orientation.tolist()
    result["stable"] = True
    return result


def build_collision_scene(mesh: trimesh.Trimesh) -> tuple[trimesh.Scene, list[trimesh.Trimesh]]:
    vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
    triangles = np.ascontiguousarray(mesh.faces, dtype=np.uint32)
    packed_faces = np.concatenate(
        [np.full((len(triangles), 1), 3, dtype=np.uint32), triangles], axis=1
    ).reshape(-1)
    decomposition = vhacdx.compute_vhacd(
        vertices,
        packed_faces,
        maxConvexHulls=32,
        resolution=100000,
        minimumVolumePercentErrorAllowed=1.0,
        maxRecursionDepth=8,
        shrinkWrap=True,
        fillMode="flood",
        maxNumVerticesPerCH=MAX_COLLISION_PART_VERTICES,
        asyncACD=False,
        minEdgeLength=2,
        findBestPlane=False,
    )
    collision_parts = []
    scene = trimesh.Scene()
    for index, (part_vertices, part_faces) in enumerate(decomposition):
        part = trimesh.Trimesh(
            vertices=np.asarray(part_vertices, dtype=np.float64),
            faces=np.asarray(part_faces, dtype=np.int64),
            process=True,
        )
        if part.is_empty or not part.is_watertight or abs(float(part.volume)) <= 1e-10:
            continue
        name = f"vhacd_part_{index:02d}"
        scene.add_geometry(part, geom_name=name, node_name=name)
        collision_parts.append(part)
    if not collision_parts:
        raise ValueError("VHACD produced no valid collision parts")

    # VHACD's voxel surface can sit slightly above the scan's minimum. Add a
    # thin, conservative sole so resting contact is broad and deterministic.
    extents = np.asarray(mesh.extents, dtype=np.float64)
    sole_thickness = min(0.008, max(0.004, 0.06 * float(extents[1])))
    sole = trimesh.creation.box(
        extents=[0.80 * float(extents[0]), sole_thickness, 0.80 * float(extents[2])]
    )
    sole.apply_translation([0.0, 0.5 * sole_thickness, 0.0])
    scene.add_geometry(sole, geom_name="sole_support", node_name="sole_support")
    collision_parts.append(sole)
    return scene, collision_parts


def asset_gate(
    mesh: trimesh.Trimesh,
    collision_parts: list[trimesh.Trimesh],
    model_data: dict,
    *,
    max_loaded_height_m: float = 0.160,
) -> dict:
    extents = np.asarray(mesh.extents, dtype=np.float64)
    scale = np.asarray(model_data["scale"], dtype=np.float64)
    loaded = extents * scale
    reasons = []
    if len(mesh.vertices) < 500 or len(mesh.faces) < 500:
        reasons.append("visual_mesh_too_small")
    if not collision_parts or not all(part.is_watertight for part in collision_parts):
        reasons.append("collision_not_watertight")
    collision_face_count = sum(len(part.faces) for part in collision_parts)
    if collision_face_count > 4096:
        reasons.append("collision_too_complex")
    collision_vertex_count = sum(len(part.vertices) for part in collision_parts)
    collision_max_part_vertices = max((len(part.vertices) for part in collision_parts), default=0)
    if collision_max_part_vertices > MAX_COLLISION_PART_VERTICES:
        reasons.append("collision_part_too_many_vertices_for_physx")
    width, height, length = loaded.tolist()
    collision_vertices = np.concatenate(
        [np.asarray(part.vertices, dtype=np.float64) for part in collision_parts], axis=0
    )
    collision_min_y = float(collision_vertices[:, 1].min())
    bottom = collision_vertices[
        collision_vertices[:, 1] <= collision_min_y + max(1e-5, 0.002 / scale[1])
    ]
    bottom_width_coverage = float(np.ptp(bottom[:, 0]) / extents[0]) if len(bottom) else 0.0
    bottom_length_coverage = float(np.ptp(bottom[:, 2]) / extents[2]) if len(bottom) else 0.0
    if bottom_width_coverage < 0.50:
        reasons.append("collision_bottom_too_narrow")
    if bottom_length_coverage < 0.60:
        reasons.append("collision_bottom_too_short")
    if not 0.045 <= width <= 0.130:
        reasons.append("loaded_width_out_of_range")
    if not 0.025 <= height <= float(max_loaded_height_m):
        reasons.append("loaded_height_out_of_range")
    if not 0.214 <= length <= 0.218:
        reasons.append("loaded_length_out_of_range")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "visual_vertices": int(len(mesh.vertices)),
        "visual_faces": int(len(mesh.faces)),
        "visual_components": int(len(mesh.split(only_watertight=False))),
        "visual_watertight": bool(mesh.is_watertight),
        "collision_parts": int(len(collision_parts)),
        "collision_vertices": int(collision_vertex_count),
        "collision_max_part_vertices": int(collision_max_part_vertices),
        "collision_faces": int(collision_face_count),
        "collision_watertight": bool(
            collision_parts and all(part.is_watertight for part in collision_parts)
        ),
        "flat_sole_support": True,
        "bottom_width_coverage": bottom_width_coverage,
        "bottom_length_coverage": bottom_length_coverage,
        "loaded_extents_m": loaded.tolist(),
    }


def marker_scene(mesh: trimesh.Trimesh, model_data: dict) -> trimesh.Scene:
    preview_mesh = mesh.copy()
    preview_mesh.visual = trimesh.visual.ColorVisuals(
        preview_mesh, vertex_colors=[160, 160, 160, 255]
    )
    scene = trimesh.Scene(preview_mesh)
    colors_and_points = (
        ([214, 39, 40, 255], np.asarray(model_data["functional_matrix"])[0, :3, 3]),
        ([44, 160, 44, 255], np.asarray(model_data["contact_points_pose"])[0, :3, 3]),
        ([31, 119, 180, 255], np.asarray(model_data["orientation_point"])[:3, 3]),
    )
    radius = max(0.004, 0.015 * float(mesh.extents[2]))
    for color, point in colors_and_points:
        marker = trimesh.creation.icosphere(subdivisions=2, radius=radius)
        marker.apply_translation(point)
        marker.visual = trimesh.visual.ColorVisuals(marker, vertex_colors=color)
        scene.add_geometry(marker)
    return scene


def _safe_write(path: Path, payload: bytes | str, *, replace_existing: bool = False) -> None:
    if path.exists() and not replace_existing:
        raise FileExistsError(f"refusing to overwrite existing asset file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, bytes):
        path.write_bytes(payload)
    else:
        path.write_text(payload, encoding="utf-8")


def prepare_assets(
    *,
    manifest_path: Path,
    source_root: Path,
    output_root: Path,
    output_modelname: str,
    description_root: Path,
    reference_model_data_path: Path,
    reference_description_path: Path,
    partitions: set[str],
    target_length_m: float,
    max_loaded_height_m: float,
    replace_existing: bool = False,
) -> list[dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reference = json.loads(reference_model_data_path.read_text(encoding="utf-8"))
    if reference_description_path.is_file():
        reference_description = reference_description_path.read_text(encoding="utf-8")
    else:
        reference_description = json.dumps(DEFAULT_SHOE_DESCRIPTION, indent=2) + "\n"
    model_root = output_root / output_modelname
    summaries = []

    for asset in manifest["assets"]:
        if asset["partition"] not in partitions:
            continue
        model_id = int(asset["selection_rank"])
        # Objective-gate replacements keep their archive in the frozen reserve
        # partition while taking the rejected asset's effective partition.  Do
        # not duplicate source archives just to convert the promoted asset.
        source_partition = asset.get("original_partition", asset["partition"])
        source_dir = source_root / source_partition / asset["name"]
        obj_path = source_dir / "meshes" / "model.obj"
        if not obj_path.is_file():
            raise FileNotFoundError(f"missing unpacked source mesh: {obj_path}")

        scene, mesh, applied_transform = load_and_align_scene(obj_path)
        collision_scene, collision_parts = build_collision_scene(mesh)
        model_data = build_model_data(mesh, reference, target_length_m=target_length_m)
        gate = asset_gate(
            mesh,
            collision_parts,
            model_data,
            max_loaded_height_m=max_loaded_height_m,
        )

        visual_path = model_root / "visual" / f"base{model_id}.glb"
        collision_path = model_root / "collision" / f"base{model_id}.glb"
        model_data_path = model_root / f"model_data{model_id}.json"
        preview_path = model_root / "preview" / f"overview{model_id}.ply"
        description_path = description_root / output_modelname / f"base{model_id}.json"
        _safe_write(
            visual_path,
            scene.export(file_type="glb"),
            replace_existing=replace_existing,
        )
        _safe_write(
            collision_path,
            collision_scene.export(file_type="glb"),
            replace_existing=replace_existing,
        )
        _safe_write(
            model_data_path,
            json.dumps(model_data, indent=2) + "\n",
            replace_existing=replace_existing,
        )
        _safe_write(
            preview_path,
            marker_scene(mesh, model_data).export(file_type="ply"),
            replace_existing=replace_existing,
        )
        _safe_write(
            description_path,
            reference_description,
            replace_existing=replace_existing,
        )

        summaries.append(
            {
                "model_id": model_id,
                "name": asset["name"],
                "partition": asset["partition"],
                "source_partition": source_partition,
                "source_obj": str(obj_path),
                "source_obj_sha256": sha256_file(obj_path),
                "applied_transform": applied_transform.tolist(),
                "scale": model_data["scale"],
                "asset_gate": gate,
                "visual_path": str(visual_path),
                "collision_path": str(collision_path),
            }
        )

    if not summaries:
        raise ValueError(f"no manifest assets matched partitions {sorted(partitions)}")

    points_source = reference_model_data_path.parent / "points_info.json"
    points_destination = model_root / "points_info.json"
    if not points_destination.exists():
        _safe_write(points_destination, points_source.read_bytes())
    attribution_path = model_root / "ATTRIBUTION.md"
    if not attribution_path.exists():
        _safe_write(
            attribution_path,
            "# Asset attribution\n\n"
            "Source meshes are selected from Google Scanned Objects by Google Research "
            "and licensed under CC BY 4.0. Mesh coordinates were rotated, centered, "
            "uniformly scaled, and convex collision hulls were generated for RoboTwin.\n",
        )

    report_path = model_root / "conversion_report.json"
    existing = []
    if report_path.exists():
        existing = json.loads(report_path.read_text(encoding="utf-8"))["assets"]
    combined = {row["model_id"]: row for row in existing}
    combined.update({row["model_id"]: row for row in summaries})
    report_path.write_text(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "target_length_m": target_length_m,
                "max_loaded_height_m": max_loaded_height_m,
                "assets": [combined[key] for key in sorted(combined)],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return summaries


def main() -> None:
    args = parse_args()
    summaries = prepare_assets(
        manifest_path=args.manifest,
        source_root=args.source_root,
        output_root=args.output_root,
        output_modelname=args.output_modelname,
        description_root=args.description_root,
        reference_model_data_path=args.reference_model_data,
        reference_description_path=args.reference_description,
        partitions=set(args.partitions),
        target_length_m=float(args.target_length_m),
        max_loaded_height_m=float(args.max_loaded_height_m),
        replace_existing=bool(args.replace_existing),
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
