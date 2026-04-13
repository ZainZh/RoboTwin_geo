from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import trimesh

from partnext_hammer_eval_utils import (
    estimate_uniform_scale,
    find_annotation_row,
    load_annotation_rows,
    load_scene_geometries,
    render_preview_ply,
)


BODY_LABELS = ("Main Body", "Body")
HANDLE_LABELS = ("Handle",)
DEFAULT_MUG_DESCRIPTION = {
    "raw_description": "mug",
    "seen": [
        "black mug",
        "black ceramic mug",
        "ceramic mug smooth outside",
        "medium black mug with handle",
        "mug with curved black handle",
        "palm-sized mug glossy black body",
        "medium mug smooth rounded handle",
        "cylindrical mug with black finish",
        "black mug glossy ceramic material",
        "drinking mug shiny black exterior",
        "black mug with rounded sturdy handle",
        "glossy black mug medium-sized for drinking",
    ],
    "unseen": [
        "rounded mug with glossy shine",
        "mug for liquids black ceramic body",
        "cylindrical drinking mug black color",
    ],
}


@dataclass(frozen=True)
class PreparedMugAsset:
    modelname: str
    visual_glb_path: Path
    collision_glb_path: Path
    model_data: dict
    points_info: dict
    source_meta: dict
    cleanup_paths: tuple[Path, ...] = ()


def _parse_json_field(value):
    if isinstance(value, str):
        return json.loads(value)
    return value


def _walk_hierarchy(nodes):
    for node in nodes or []:
        yield node
        yield from _walk_hierarchy(node.get("children") or [])


def _merge_face_specs(into: dict[int, set[int]], other: dict[int, set[int]]) -> dict[int, set[int]]:
    for submesh_id, face_ids in other.items():
        into.setdefault(int(submesh_id), set()).update(int(face_id) for face_id in face_ids)
    return into


def _coerce_face_specs(mask_entry, default_submesh_id: int = 0) -> dict[int, set[int]]:
    face_specs: dict[int, set[int]] = {}
    if isinstance(mask_entry, list):
        face_specs.setdefault(int(default_submesh_id), set()).update(int(index) for index in mask_entry)
        return face_specs
    if isinstance(mask_entry, dict):
        for key, nested in mask_entry.items():
            next_submesh_id = int(key) if str(key).isdigit() else int(default_submesh_id)
            _merge_face_specs(face_specs, _coerce_face_specs(nested, default_submesh_id=next_submesh_id))
    return face_specs


def _has_any_face_specs(face_specs: dict[int, set[int]]) -> bool:
    return any(face_ids for face_ids in face_specs.values())


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        raise ValueError("cannot normalize near-zero vector")
    return vector / norm


def _orthonormal_frame(primary_axis: np.ndarray, secondary_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z_axis = _normalize(primary_axis)
    y_axis = np.asarray(secondary_axis, dtype=np.float64)
    y_axis = y_axis - z_axis * float(np.dot(y_axis, z_axis))
    if float(np.linalg.norm(y_axis)) <= 1e-8:
        fallbacks = (
            np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
            np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
            np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
        )
        for fallback in fallbacks:
            y_axis = fallback - z_axis * float(np.dot(fallback, z_axis))
            if float(np.linalg.norm(y_axis)) > 1e-8:
                break
    y_axis = _normalize(y_axis)
    x_axis = _normalize(np.cross(y_axis, z_axis))
    y_axis = _normalize(np.cross(z_axis, x_axis))
    return x_axis, y_axis, z_axis


def _build_pose(origin: np.ndarray, x_axis: np.ndarray, y_axis: np.ndarray, z_axis: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = _normalize(x_axis)
    pose[:3, 1] = _normalize(y_axis)
    pose[:3, 2] = _normalize(z_axis)
    pose[:3, 3] = np.asarray(origin, dtype=np.float64)
    return pose


def _collect_label_face_specs(row: dict, labels: tuple[str, ...]) -> dict[str, dict[int, set[int]]]:
    hierarchy = _parse_json_field(row["hierarchyList"])
    masks = _parse_json_field(row["masks"])

    face_specs_by_label: dict[str, dict[int, set[int]]] = {label: {} for label in labels}
    for node in _walk_hierarchy(hierarchy):
        node_name = str(node.get("name", ""))
        if node_name not in face_specs_by_label:
            continue
        mask_id = node.get("maskId")
        if mask_id is None:
            continue
        _merge_face_specs(face_specs_by_label[node_name], _coerce_face_specs(masks.get(str(mask_id), {})))
    return face_specs_by_label


def collect_body_and_handle_face_specs(row: dict) -> tuple[dict[int, set[int]], dict[int, set[int]], list[str]]:
    body_by_label = _collect_label_face_specs(row, BODY_LABELS)
    handle_by_label = _collect_label_face_specs(row, HANDLE_LABELS)
    body_specs: dict[int, set[int]] = {}
    handle_specs: dict[int, set[int]] = {}
    handle_labels_used: list[str] = []

    for _, specs in body_by_label.items():
        if _has_any_face_specs(specs):
            _merge_face_specs(body_specs, specs)
    for label, specs in handle_by_label.items():
        if _has_any_face_specs(specs):
            _merge_face_specs(handle_specs, specs)
            handle_labels_used.append(label)
    if not _has_any_face_specs(body_specs):
        raise ValueError("mug candidate is missing body face annotations")
    if not _has_any_face_specs(handle_specs):
        raise ValueError("mug candidate is missing handle face annotations")
    return body_specs, handle_specs, handle_labels_used


def _collect_region_geometry(
    geometry_meshes: list[trimesh.Trimesh],
    face_specs: dict[int, set[int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centroid_groups: list[np.ndarray] = []
    normal_groups: list[np.ndarray] = []
    vertex_groups: list[np.ndarray] = []
    for submesh_id, face_ids in sorted(face_specs.items()):
        if not face_ids:
            continue
        geometry = geometry_meshes[int(submesh_id)]
        face_indices = np.asarray(sorted(face_ids), dtype=np.int64)
        centroid_groups.append(np.asarray(geometry.triangles_center[face_indices], dtype=np.float64))
        normal_groups.append(np.asarray(geometry.face_normals[face_indices], dtype=np.float64))
        vertex_indices = np.unique(geometry.faces[face_indices].reshape(-1))
        vertex_groups.append(np.asarray(geometry.vertices[vertex_indices], dtype=np.float64))
    if not centroid_groups:
        raise ValueError("region face specs produced no geometry")
    return np.vstack(centroid_groups), np.vstack(normal_groups), np.vstack(vertex_groups)


def compute_mug_body_frame(
    geometry_meshes: list[trimesh.Trimesh],
    body_face_specs: dict[int, set[int]],
) -> dict:
    body_centroids, body_normals, body_vertices = _collect_region_geometry(geometry_meshes, body_face_specs)
    body_center = body_centroids.mean(axis=0)
    centered_vertices = body_vertices - body_center
    covariance = centered_vertices.T @ centered_vertices
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    axes = eigenvectors[:, order]

    up_axis = axes[:, 0]
    if float(np.dot(up_axis, np.asarray([0.0, 0.0, 1.0], dtype=np.float64))) < 0.0:
        up_axis = -up_axis

    radial_axis = axes[:, 1]
    radial_axis = radial_axis - up_axis * float(np.dot(radial_axis, up_axis))
    radial_axis = _normalize(radial_axis)
    secondary_axis = _normalize(np.cross(up_axis, radial_axis))

    projections = (body_vertices - body_center) @ up_axis
    bottom_threshold = float(np.quantile(projections, 0.05))
    top_threshold = float(np.quantile(projections, 0.95))
    bottom_vertices = body_vertices[projections <= bottom_threshold]
    top_vertices = body_vertices[projections >= top_threshold]
    bottom_center = bottom_vertices.mean(axis=0) if len(bottom_vertices) > 0 else body_center - 0.5 * np.ptp(projections) * up_axis
    top_center = top_vertices.mean(axis=0) if len(top_vertices) > 0 else body_center + 0.5 * np.ptp(projections) * up_axis

    side_mask = np.abs(body_normals @ up_axis) < 0.45
    side_centroids = body_centroids[side_mask] if np.any(side_mask) else body_centroids
    side_normals = body_normals[side_mask] if np.any(side_mask) else body_normals

    return {
        "body_center": body_center,
        "bottom_center": bottom_center,
        "top_center": top_center,
        "up_axis": up_axis,
        "radial_axis": radial_axis,
        "secondary_axis": secondary_axis,
        "side_centroids": side_centroids,
        "side_normals": side_normals,
    }


def compute_handle_frame(
    geometry_meshes: list[trimesh.Trimesh],
    handle_face_specs: dict[int, set[int]],
    body_center: np.ndarray,
    up_axis: np.ndarray,
) -> dict:
    handle_centroids, handle_normals, handle_vertices = _collect_region_geometry(geometry_meshes, handle_face_specs)
    handle_center = handle_centroids.mean(axis=0)

    planar_offset = handle_center - body_center
    planar_offset = planar_offset - up_axis * float(np.dot(planar_offset, up_axis))
    if float(np.linalg.norm(planar_offset)) > 1e-8:
        outward_axis = _normalize(planar_offset)
    else:
        centered = handle_vertices - handle_center
        covariance = centered.T @ centered
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        outward_axis = None
        for axis in eigenvectors[:, order].T:
            candidate_axis = axis - up_axis * float(np.dot(axis, up_axis))
            if float(np.linalg.norm(candidate_axis)) > 1e-8:
                outward_axis = _normalize(candidate_axis)
                break
        if outward_axis is None:
            fallback_axes = (
                np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
                np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
                np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
            )
            for fallback_axis in fallback_axes:
                candidate_axis = fallback_axis - up_axis * float(np.dot(fallback_axis, up_axis))
                if float(np.linalg.norm(candidate_axis)) > 1e-8:
                    outward_axis = _normalize(candidate_axis)
                    break
        if outward_axis is None:
            raise ValueError("failed to estimate mug handle outward axis")

    projections = (handle_centroids - body_center) @ outward_axis
    inner_threshold = float(np.quantile(projections, 0.15))
    inner_centroids = handle_centroids[projections <= inner_threshold]
    hanging_point = inner_centroids.mean(axis=0) if len(inner_centroids) > 0 else handle_center

    handle_span = float(np.ptp(projections))
    return {
        "handle_center": handle_center,
        "outward_axis": outward_axis,
        "hanging_point": hanging_point,
        "handle_span": handle_span,
        "handle_normals": handle_normals,
    }


def build_mug_contact_poses(
    body_frame: dict,
    handle_frame: dict,
    contact_count: int = 4,
) -> list[np.ndarray]:
    side_centroids = body_frame["side_centroids"]
    side_normals = body_frame["side_normals"]
    up_axis = body_frame["up_axis"]
    handle_dir = handle_frame["outward_axis"]
    cross_dir = _normalize(np.cross(up_axis, handle_dir))
    directions = [handle_dir, -handle_dir, cross_dir, -cross_dir]
    poses: list[np.ndarray] = []

    for direction in directions[: max(int(contact_count), 4)]:
        projections = (side_centroids - body_frame["body_center"]) @ direction
        alignments = side_normals @ direction
        threshold = float(np.quantile(projections, 0.85))
        mask = (projections >= threshold) & (alignments >= 0.35)
        if not np.any(mask):
            mask = projections >= threshold
        contact_point = side_centroids[mask].mean(axis=0)
        x_axis = _normalize(np.cross(direction, up_axis))
        y_axis = _normalize(direction)
        z_axis = _normalize(up_axis)
        poses.append(_build_pose(contact_point, x_axis, y_axis, z_axis))
        poses.append(_build_pose(contact_point, -x_axis, y_axis, -z_axis))
    return poses


def build_mug_functional_poses(body_frame: dict, handle_frame: dict) -> tuple[np.ndarray, np.ndarray]:
    up_axis = body_frame["up_axis"]
    outward_axis = handle_frame["outward_axis"]

    x_axis = _normalize(np.cross(up_axis, outward_axis))
    y_axis = _normalize(up_axis)
    z_axis = _normalize(outward_axis)
    hanging_pose = _build_pose(handle_frame["hanging_point"], x_axis, y_axis, z_axis)

    bottom_x, bottom_y, bottom_z = _orthonormal_frame(up_axis, outward_axis)
    bottom_pose = _build_pose(body_frame["bottom_center"], bottom_x, bottom_y, bottom_z)
    return hanging_pose, bottom_pose


def build_mug_collision_glb(
    geometry_meshes: list[trimesh.Trimesh],
    body_face_specs: dict[int, set[int]],
    handle_face_specs: dict[int, set[int]],
) -> Path:
    collision_scene = trimesh.Scene()

    def add_region(region_name: str, face_specs: dict[int, set[int]]):
        for submesh_id, face_ids in sorted(face_specs.items()):
            if not face_ids:
                continue
            geometry = geometry_meshes[int(submesh_id)]
            face_indices = np.asarray(sorted(face_ids), dtype=np.int64)
            submesh = geometry.submesh([face_indices], append=True, repair=False)
            if isinstance(submesh, list):
                if len(submesh) == 0:
                    continue
                submesh = trimesh.util.concatenate(submesh)
            if submesh is None or len(submesh.faces) == 0:
                continue
            collision_scene.add_geometry(submesh, geom_name=f"{region_name}_{submesh_id}")

    add_region("body", body_face_specs)
    add_region("handle", handle_face_specs)
    if len(collision_scene.geometry) == 0:
        raise ValueError("collision export produced zero components")

    with tempfile.NamedTemporaryFile(prefix="partnext_mug_collision_", suffix=".glb", delete=False) as tmp:
        collision_path = Path(tmp.name)
    collision_scene.export(collision_path)
    return collision_path


def build_model_data(
    mesh: trimesh.Trimesh,
    scale: float,
    contact_poses: list[np.ndarray],
    hanging_pose: np.ndarray,
    bottom_pose: np.ndarray,
    stable: bool = False,
) -> dict:
    center = np.asarray(mesh.bounding_box.centroid, dtype=np.float64)
    extents = np.asarray(mesh.extents, dtype=np.float64)
    target_pose = bottom_pose.copy()

    contact_desc = "Grasping the side of the cup."
    hanging_desc = (
        "Point0: The function point is inside the handle of the mug, and the function axis is perpendicular "
        "to the plane of the handle."
    )
    bottom_desc = (
        "Point1: The function point is at the bottom of the mug, and the function axis is perpendicular "
        "to the bottom of the cup, Used to put down the cup."
    )

    return {
        "center": center.tolist(),
        "extents": extents.tolist(),
        "scale": [float(scale), float(scale), float(scale)],
        "transform_matrix": np.eye(4, dtype=np.float64).tolist(),
        "target_pose": [target_pose.tolist()],
        "contact_points_pose": [pose.tolist() for pose in contact_poses],
        "functional_matrix": [hanging_pose.tolist(), bottom_pose.tolist()],
        "orientation_point": [hanging_pose.tolist()],
        "contact_points_group": [[i] for i in range(len(contact_poses))],
        "contact_points_mask": [True for _ in contact_poses],
        "contact_points_description": [contact_desc for _ in contact_poses],
        "functional_point_description": [hanging_desc, bottom_desc],
        "target_point_description": [""],
        "orientation_point_description": [hanging_desc],
        "contact_points_discription": [contact_desc for _ in contact_poses],
        "functional_point_discription": [hanging_desc, bottom_desc],
        "target_point_discription": [""],
        "stable": bool(stable),
    }


def build_partnext_mug_asset(
    partnext_dir: Path,
    annotation_path: Path,
    output_modelname: str,
    reference_model_data_path: Path | None = None,
    requested_glb_name: str | None = None,
) -> PreparedMugAsset:
    annotation_rows = load_annotation_rows(annotation_path)
    if reference_model_data_path is None:
        reference_model_data_path = Path(__file__).resolve().parents[1] / "assets" / "objects" / "039_mug" / "model_data0.json"
    reference_model_data = json.loads(reference_model_data_path.read_text(encoding="utf-8"))
    reference_loaded_extents = np.asarray(reference_model_data["extents"], dtype=np.float64) * np.asarray(
        reference_model_data["scale"], dtype=np.float64
    )

    if requested_glb_name is None:
        glb_paths = sorted(partnext_dir.glob("*.glb"))
        if not glb_paths:
            raise FileNotFoundError(f"no .glb files found under {partnext_dir}")
        candidate_glb = glb_paths[0]
    else:
        candidate_glb = partnext_dir / Path(requested_glb_name).name
        if not candidate_glb.is_file():
            raise FileNotFoundError(f"requested mug glb was not found: {candidate_glb}")

    annotation_row = find_annotation_row(annotation_rows, candidate_glb.name, fallback_model_id=candidate_glb.stem)
    mesh = trimesh.load(candidate_glb, force="mesh")
    geometry_meshes = load_scene_geometries(candidate_glb)
    body_face_specs, handle_face_specs, handle_labels_used = collect_body_and_handle_face_specs(annotation_row)

    body_frame = compute_mug_body_frame(geometry_meshes, body_face_specs)
    handle_frame = compute_handle_frame(
        geometry_meshes,
        handle_face_specs,
        body_center=body_frame["body_center"],
        up_axis=body_frame["up_axis"],
    )
    contact_poses = build_mug_contact_poses(body_frame, handle_frame)
    hanging_pose, bottom_pose = build_mug_functional_poses(body_frame, handle_frame)
    scale = estimate_uniform_scale(reference_loaded_extents, np.asarray(mesh.extents, dtype=np.float64))
    collision_glb_path = build_mug_collision_glb(geometry_meshes, body_face_specs, handle_face_specs)
    model_data = build_model_data(
        mesh=mesh,
        scale=scale,
        contact_poses=contact_poses,
        hanging_pose=hanging_pose,
        bottom_pose=bottom_pose,
        stable=bool(reference_model_data.get("stable", False)),
    )

    points_info = {
        "contact_points": [{"id": idx, "description": "mug side grasp point"} for idx in range(len(contact_poses))],
        "functional_points": [
            {"id": 0, "description": "mug handle hanging point"},
            {"id": 1, "description": "mug bottom placement point"},
        ],
    }
    source_meta = {
        "model_id": str(annotation_row.get("model_id", candidate_glb.stem)),
        "glb_dst": str(annotation_row.get("glb_dst", candidate_glb.name)),
        "source_mesh_path": str(candidate_glb),
        "annotation_object_name": str(annotation_row.get("object_name", "Mug")),
        "matched_labels": {
            "body": ["Main Body"],
            "handle": handle_labels_used,
        },
        "scale_estimate": {
            "reference_loaded_extents": reference_loaded_extents.tolist(),
            "candidate_extents": np.asarray(mesh.extents, dtype=np.float64).tolist(),
            "uniform_scale": float(scale),
        },
        "generated_points": {
            "contact_point": np.asarray(contact_poses[0], dtype=np.float64)[:3, 3].tolist(),
            "target_point": np.asarray(bottom_pose, dtype=np.float64)[:3, 3].tolist(),
            "functional_point": np.asarray(hanging_pose, dtype=np.float64)[:3, 3].tolist(),
            "up_axis": body_frame["up_axis"].tolist(),
            "handle_axis": handle_frame["outward_axis"].tolist(),
        },
    }
    return PreparedMugAsset(
        modelname=output_modelname,
        visual_glb_path=candidate_glb,
        collision_glb_path=collision_glb_path,
        model_data=model_data,
        points_info=points_info,
        source_meta=source_meta,
        cleanup_paths=(collision_glb_path,),
    )


__all__ = [
    "PreparedMugAsset",
    "build_model_data",
    "build_mug_collision_glb",
    "build_partnext_mug_asset",
    "collect_body_and_handle_face_specs",
    "compute_handle_frame",
    "compute_mug_body_frame",
    "build_mug_contact_poses",
    "build_mug_functional_poses",
]
