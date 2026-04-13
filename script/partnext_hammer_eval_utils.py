from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import trimesh

HANDLE_LABELS = ("Handle", "Shaft", "Grip", "Handle End")
STRIKING_LABELS = ("Hammer Head", "Head", "Nail Puller")


@dataclass(frozen=True)
class PreparedHammerAsset:
    modelname: str
    visual_glb_path: Path
    collision_glb_path: Path
    model_data: dict
    points_info: dict
    source_meta: dict


def _parse_json_field(value):
    if isinstance(value, str):
        return json.loads(value)
    return value


def _flatten_mask_indices(mask_entry) -> set[int]:
    face_ids: set[int] = set()
    if isinstance(mask_entry, dict):
        for nested in mask_entry.values():
            face_ids.update(_flatten_mask_indices(nested))
    elif isinstance(mask_entry, list):
        face_ids.update(int(index) for index in mask_entry)
    return face_ids


def _walk_hierarchy(nodes):
    for node in nodes or []:
        yield node
        children = node.get("children") or []
        yield from _walk_hierarchy(children)


def _load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(mesh_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"expected Trimesh for {mesh_path}, got {type(mesh).__name__}")
    if mesh.faces.size == 0 or mesh.vertices.size == 0:
        raise ValueError(f"mesh is empty: {mesh_path}")
    return mesh


def _collect_label_face_ids(row: dict, labels: tuple[str, ...]) -> dict[str, set[int]]:
    hierarchy = _parse_json_field(row["hierarchyList"])
    masks = _parse_json_field(row["masks"])

    face_ids_by_label: dict[str, set[int]] = {label: set() for label in labels}
    for node in _walk_hierarchy(hierarchy):
        node_name = str(node.get("name", ""))
        if node_name not in face_ids_by_label:
            continue
        mask_id = node.get("maskId")
        if mask_id is None:
            continue
        face_ids_by_label[node_name].update(_flatten_mask_indices(masks.get(str(mask_id), {})))
    return face_ids_by_label


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        raise ValueError("cannot normalize near-zero vector")
    return vector / norm


def _build_pose_matrix(origin: np.ndarray, primary_axis: np.ndarray, secondary_hint: np.ndarray) -> np.ndarray:
    x_axis = _normalize(primary_axis)
    secondary = np.asarray(secondary_hint, dtype=np.float64)
    if np.linalg.norm(np.cross(x_axis, secondary)) <= 1e-8:
        fallback_axes = (
            np.asarray([0.0, 0.0, 1.0]),
            np.asarray([0.0, 1.0, 0.0]),
            np.asarray([1.0, 0.0, 0.0]),
        )
        for fallback in fallback_axes:
            if np.linalg.norm(np.cross(x_axis, fallback)) > 1e-8:
                secondary = fallback
                break
    z_axis = _normalize(np.cross(x_axis, secondary))
    y_axis = _normalize(np.cross(z_axis, x_axis))

    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = x_axis
    pose[:3, 1] = y_axis
    pose[:3, 2] = z_axis
    pose[:3, 3] = np.asarray(origin, dtype=np.float64)
    return pose


def load_annotation_rows(annotation_path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in annotation_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def find_annotation_row(rows: list[dict], glb_name: str, fallback_model_id: str | None = None) -> dict:
    for row in rows:
        if str(row.get("glb_dst", "")) == glb_name:
            return row
    if fallback_model_id is not None:
        for row in rows:
            if str(row.get("model_id", "")) == fallback_model_id:
                return row
    raise KeyError(f"annotation row not found for {glb_name}")


def _region_labels(region: str) -> tuple[str, ...]:
    if region == "handle":
        return HANDLE_LABELS
    if region == "head":
        return STRIKING_LABELS
    raise ValueError(f"unsupported region: {region}")


def collect_region_face_ids(row: dict, region: str) -> set[int]:
    region_labels = _region_labels(region)
    hierarchy = _parse_json_field(row["hierarchyList"])
    masks = _parse_json_field(row["masks"])

    face_ids: set[int] = set()
    for node in _walk_hierarchy(hierarchy):
        node_name = str(node.get("name", ""))
        if node_name not in region_labels:
            continue
        mask_id = node.get("maskId")
        if mask_id is None:
            continue
        face_ids.update(_flatten_mask_indices(masks.get(str(mask_id), {})))
    return face_ids


def pick_striking_label(label_to_faces: dict[str, set[int]]) -> str:
    for label in STRIKING_LABELS:
        if label in label_to_faces and label_to_faces[label]:
            return label
    raise ValueError("no striking label candidates available")


def estimate_uniform_scale(
    reference_loaded_extents: np.ndarray, candidate_extents: np.ndarray
) -> float:
    reference_dominant_extent = float(np.max(reference_loaded_extents))
    candidate_dominant_extent = float(np.max(candidate_extents))
    if candidate_dominant_extent <= 0.0:
        raise ValueError("candidate extents must be positive")
    return float(np.clip(reference_dominant_extent / candidate_dominant_extent, 1e-3, 1e3))


def select_candidate_glb(
    partnext_dir: Path,
    annotation_rows: list[dict],
    reference_loaded_extents: np.ndarray,
) -> Path:
    glb_paths = sorted(partnext_dir.glob("*.glb"))
    if not glb_paths:
        raise FileNotFoundError(f"no .glb files found under {partnext_dir}")

    reference_loaded_extents = np.asarray(reference_loaded_extents, dtype=np.float64)
    reference_shape = reference_loaded_extents / float(np.max(reference_loaded_extents))

    best_path: Path | None = None
    best_score: float | None = None
    rejection_reasons: list[str] = []
    for glb_path in glb_paths:
        try:
            row = find_annotation_row(annotation_rows, glb_path.name, fallback_model_id=glb_path.stem)
        except KeyError:
            continue

        handle_faces = collect_region_face_ids(row, region="handle")
        head_faces = collect_region_face_ids(row, region="head")
        if not handle_faces or not head_faces:
            rejection_reasons.append(f"{glb_path.name}: missing annotated handle or head faces")
            continue

        try:
            mesh = _load_mesh(glb_path)
        except Exception as exc:
            rejection_reasons.append(f"{glb_path.name}: failed to load mesh ({exc})")
            continue

        extents = np.asarray(mesh.extents, dtype=np.float64)
        if np.any(extents <= 0.0):
            rejection_reasons.append(f"{glb_path.name}: non-positive mesh extents")
            continue
        dominant_extent = float(np.max(extents))
        if dominant_extent < 1e-4 or dominant_extent > 1e5:
            rejection_reasons.append(f"{glb_path.name}: dominant extent out of range")
            continue

        shape = extents / dominant_extent
        shape_score = float(np.linalg.norm(shape - reference_shape))
        size_score = abs(np.log(dominant_extent) - np.log(float(np.max(reference_loaded_extents))))
        score = shape_score + (0.2 * size_score)
        if best_score is None or score < best_score:
            best_path = glb_path
            best_score = score

    if best_path is None:
        if rejection_reasons:
            raise ValueError("; ".join(rejection_reasons))
        raise ValueError(f"no annotated hammer candidate found in {partnext_dir}")
    return best_path


def compute_handle_contact_point(mesh: trimesh.Trimesh, handle_face_ids: set[int]) -> tuple[np.ndarray, np.ndarray]:
    if not handle_face_ids:
        raise ValueError("handle face ids are required")

    handle_indices = np.asarray(sorted(handle_face_ids), dtype=np.int64)
    handle_centroids = np.asarray(mesh.triangles_center[handle_indices], dtype=np.float64)
    handle_center = handle_centroids.mean(axis=0)
    centered = handle_centroids - handle_center
    covariance = centered.T @ centered
    _, eigenvectors = np.linalg.eigh(covariance)
    handle_axis = _normalize(eigenvectors[:, -1])
    return handle_center, handle_axis


def compute_head_functional_point(
    mesh: trimesh.Trimesh,
    head_face_ids: set[int],
    handle_axis: np.ndarray,
    handle_contact_point: np.ndarray,
) -> np.ndarray:
    if not head_face_ids:
        raise ValueError("head face ids are required")

    head_indices = np.asarray(sorted(head_face_ids), dtype=np.int64)
    head_centroids = np.asarray(mesh.triangles_center[head_indices], dtype=np.float64)
    return head_centroids.mean(axis=0)


def build_model_data(
    mesh: trimesh.Trimesh,
    scale: float,
    contact_pose: np.ndarray,
    functional_pose: np.ndarray,
    target_point: np.ndarray,
    stable: bool = False,
) -> dict:
    center = np.asarray(mesh.bounding_box.centroid, dtype=np.float64)
    extents = np.asarray(mesh.extents, dtype=np.float64)
    target_pose = np.eye(4, dtype=np.float64)
    target_pose[:3, 3] = np.asarray(target_point, dtype=np.float64)

    return {
        "center": center.tolist(),
        "extents": extents.tolist(),
        "scale": [float(scale), float(scale), float(scale)],
        "transform_matrix": np.eye(4, dtype=np.float64).tolist(),
        "target_pose": [target_pose.tolist()],
        "contact_points_pose": [np.asarray(contact_pose, dtype=np.float64).tolist()],
        "functional_matrix": [np.asarray(functional_pose, dtype=np.float64).tolist()],
        "orientation_point": [np.asarray(functional_pose, dtype=np.float64).tolist()],
        "contact_points_group": [[0]],
        "contact_points_mask": [True],
        "contact_points_description": ["Grab the hammer's handle with the head facing outward."],
        "functional_point_description": ["Point 0: The head of the hammer is facing outward."],
        "target_point_description": ["The center of the handle part."],
        "orientation_point_description": ["Point 0: The head of the hammer is facing outward."],
        "contact_points_discription": ["Grab the hammer's handle with the head facing outward."],
        "functional_point_discription": ["Point 0: The head of the hammer is facing outward."],
        "target_point_discription": ["The center of the handle part."],
        "stable": bool(stable),
    }


def render_preview_ply(
    mesh: trimesh.Trimesh,
    contact_point: np.ndarray,
    target_point: np.ndarray,
    functional_point: np.ndarray,
) -> bytes:
    base_mesh = mesh.copy()
    if base_mesh.visual is None or len(getattr(base_mesh.visual, "vertex_colors", [])) != len(base_mesh.vertices):
        base_mesh.visual.vertex_colors = np.tile(np.asarray([180, 180, 180, 255], dtype=np.uint8), (len(base_mesh.vertices), 1))
    else:
        base_mesh.visual.vertex_colors = np.tile(np.asarray([180, 180, 180, 255], dtype=np.uint8), (len(base_mesh.vertices), 1))

    marker_radius = max(float(np.max(mesh.extents)) * 0.035, 1e-4)

    def make_marker(point, color):
        marker = trimesh.creation.icosphere(subdivisions=2, radius=marker_radius)
        marker.apply_translation(np.asarray(point, dtype=np.float64))
        marker.visual.vertex_colors = np.tile(np.asarray(color, dtype=np.uint8), (len(marker.vertices), 1))
        return marker

    preview_mesh = trimesh.util.concatenate([
        base_mesh,
        make_marker(contact_point, [214, 39, 40, 255]),
        make_marker(target_point, [44, 160, 44, 255]),
        make_marker(functional_point, [31, 119, 180, 255]),
    ])
    return preview_mesh.export(file_type="ply")


def write_asset_package(output_root: Path, prepared_asset: PreparedHammerAsset, preview_ply: bytes) -> Path:
    asset_dir = output_root / prepared_asset.modelname
    (asset_dir / "visual").mkdir(parents=True, exist_ok=True)
    (asset_dir / "collision").mkdir(parents=True, exist_ok=True)
    (asset_dir / "preview").mkdir(parents=True, exist_ok=True)

    shutil.copy2(prepared_asset.visual_glb_path, asset_dir / "visual" / "base0.glb")
    shutil.copy2(prepared_asset.collision_glb_path, asset_dir / "collision" / "base0.glb")
    (asset_dir / "model_data0.json").write_text(
        json.dumps(prepared_asset.model_data, indent=2),
        encoding="utf-8",
    )
    (asset_dir / "points_info.json").write_text(
        json.dumps(prepared_asset.points_info, indent=2),
        encoding="utf-8",
    )
    (asset_dir / "source_meta.json").write_text(
        json.dumps(prepared_asset.source_meta, indent=2),
        encoding="utf-8",
    )
    legacy_preview_path = asset_dir / "preview" / "overview.png"
    if legacy_preview_path.exists():
        legacy_preview_path.unlink()
    (asset_dir / "preview" / "overview.ply").write_bytes(preview_ply)
    return asset_dir


def _resolve_reference_model_data_path(reference_model_data_path: Path | None = None) -> Path:
    if reference_model_data_path is not None:
        return reference_model_data_path

    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        repo_root / "assets" / "objects" / "020_hammer" / "model_data0.json",
        Path("/home/zheng/github/RoboTwin_geo/assets/objects/020_hammer/model_data0.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("reference model_data0.json for 020_hammer was not found")


def build_partnext_hammer_asset(
    partnext_dir: Path,
    annotation_path: Path,
    output_modelname: str,
    reference_model_data_path: Path | None = None,
) -> PreparedHammerAsset:
    annotation_rows = load_annotation_rows(annotation_path)
    reference_model_data = json.loads(
        _resolve_reference_model_data_path(reference_model_data_path).read_text(encoding="utf-8")
    )
    reference_loaded_extents = np.asarray(reference_model_data["extents"], dtype=np.float64) * np.asarray(
        reference_model_data["scale"], dtype=np.float64
    )

    candidate_glb = select_candidate_glb(partnext_dir, annotation_rows, reference_loaded_extents)
    annotation_row = find_annotation_row(annotation_rows, candidate_glb.name, fallback_model_id=candidate_glb.stem)
    mesh = _load_mesh(candidate_glb)

    handle_face_ids = collect_region_face_ids(annotation_row, region="handle")
    striking_faces_by_label = _collect_label_face_ids(annotation_row, STRIKING_LABELS)
    striking_label = pick_striking_label(striking_faces_by_label)
    head_face_ids = striking_faces_by_label[striking_label]
    if not handle_face_ids:
        raise ValueError(f"selected candidate {candidate_glb.name} is missing handle face ids")
    if not head_face_ids:
        raise ValueError(f"selected candidate {candidate_glb.name} is missing head face ids")

    contact_point, handle_axis = compute_handle_contact_point(mesh, handle_face_ids)
    functional_point = compute_head_functional_point(mesh, head_face_ids, handle_axis, contact_point)
    if float(np.dot(functional_point - contact_point, handle_axis)) < 0.0:
        handle_axis = -handle_axis

    contact_pose = _build_pose_matrix(contact_point, handle_axis, functional_point - contact_point)
    functional_pose = _build_pose_matrix(functional_point, functional_point - contact_point, handle_axis)
    scale = estimate_uniform_scale(reference_loaded_extents, np.asarray(mesh.extents, dtype=np.float64))
    model_data = build_model_data(
        mesh=mesh,
        scale=scale,
        contact_pose=contact_pose,
        functional_pose=functional_pose,
        target_point=contact_point,
        stable=bool(reference_model_data.get("stable", False)),
    )

    points_info = {
        "contact_points": [{"id": 0, "description": "hammer handle grasp point"}],
        "functional_points": [{"id": 0, "description": "hammer striking head point"}],
    }
    source_meta = {
        "model_id": str(annotation_row.get("model_id", candidate_glb.stem)),
        "glb_dst": str(annotation_row.get("glb_dst", candidate_glb.name)),
        "source_mesh_path": str(candidate_glb),
        "annotation_object_name": str(annotation_row.get("object_name", "Hammer")),
        "matched_labels": {
            "handle": [label for label in HANDLE_LABELS if label in str(annotation_row.get("hierarchyList", ""))],
            "head": [striking_label],
        },
        "scale_estimate": {
            "reference_loaded_extents": reference_loaded_extents.tolist(),
            "candidate_extents": np.asarray(mesh.extents, dtype=np.float64).tolist(),
            "uniform_scale": float(scale),
        },
        "generated_points": {
            "contact_point": contact_point.tolist(),
            "target_point": contact_point.tolist(),
            "functional_point": functional_point.tolist(),
            "handle_axis": handle_axis.tolist(),
        },
    }
    return PreparedHammerAsset(
        modelname=output_modelname,
        visual_glb_path=candidate_glb,
        collision_glb_path=candidate_glb,
        model_data=model_data,
        points_info=points_info,
        source_meta=source_meta,
    )


__all__ = [
    "PreparedHammerAsset",
    "build_model_data",
    "build_partnext_hammer_asset",
    "collect_region_face_ids",
    "compute_handle_contact_point",
    "compute_head_functional_point",
    "estimate_uniform_scale",
    "find_annotation_row",
    "load_annotation_rows",
    "pick_striking_label",
    "render_preview_ply",
    "select_candidate_glb",
    "write_asset_package",
]
