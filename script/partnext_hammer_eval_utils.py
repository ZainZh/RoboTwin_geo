from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import trimesh

HANDLE_CORE_LABELS = (
    "Shaft",
    "Grip",
    "Anti-Slip Material",
    "Handle End",
    "Decoration",
)
HANDLE_LABELS = HANDLE_CORE_LABELS + ("Connector",)
STRIKING_LABELS = ("Hammer Head", "Nail Puller", "Connector")


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


def _flatten_mask_indices(mask_entry) -> set[int]:
    face_ids: set[int] = set()
    for nested_face_ids in _coerce_face_specs(mask_entry).values():
        face_ids.update(nested_face_ids)
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


def _geometry_index_from_name(name: str) -> int:
    if ".glb_" in name:
        suffix = name.rsplit(".glb_", 1)[1]
        if suffix.isdigit():
            return int(suffix)
    return 0


def load_scene_geometries(mesh_path: Path) -> list[trimesh.Trimesh]:
    scene = trimesh.load(mesh_path, force="scene")
    if isinstance(scene, trimesh.Trimesh):
        return [scene]
    if not isinstance(scene, trimesh.Scene):
        raise TypeError(f"expected Scene or Trimesh for {mesh_path}, got {type(scene).__name__}")
    geometry_items = sorted(scene.geometry.items(), key=lambda item: _geometry_index_from_name(item[0]))
    return [geometry.copy() for _, geometry in geometry_items]


def _collect_label_face_ids(row: dict, labels: tuple[str, ...]) -> dict[str, set[int]]:
    face_specs_by_label = _collect_label_face_specs(row, labels)
    face_ids_by_label: dict[str, set[int]] = {}
    for label, face_specs in face_specs_by_label.items():
        face_ids: set[int] = set()
        for nested_face_ids in face_specs.values():
            face_ids.update(nested_face_ids)
        face_ids_by_label[label] = face_ids
    return face_ids_by_label


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
    if region == "handle_core":
        return HANDLE_CORE_LABELS
    if region == "head":
        return STRIKING_LABELS
    raise ValueError(f"unsupported region: {region}")


def collect_region_face_specs(row: dict, region: str) -> dict[int, set[int]]:
    region_labels = _region_labels(region)
    hierarchy = _parse_json_field(row["hierarchyList"])
    masks = _parse_json_field(row["masks"])

    face_specs: dict[int, set[int]] = {}
    for node in _walk_hierarchy(hierarchy):
        node_name = str(node.get("name", ""))
        if node_name not in region_labels:
            continue
        mask_id = node.get("maskId")
        if mask_id is None:
            continue
        _merge_face_specs(face_specs, _coerce_face_specs(masks.get(str(mask_id), {})))
    return face_specs


def collect_region_face_ids(row: dict, region: str) -> set[int]:
    face_ids: set[int] = set()
    for nested_face_ids in collect_region_face_specs(row, region).values():
        face_ids.update(nested_face_ids)
    return face_ids


def _has_any_face_specs(face_specs) -> bool:
    if isinstance(face_specs, set):
        return bool(face_specs)
    return any(face_ids for face_ids in face_specs.values())


def pick_striking_label(label_to_faces: dict) -> str:
    for label in STRIKING_LABELS:
        if label not in label_to_faces:
            continue
        face_spec = label_to_faces[label]
        if isinstance(face_spec, dict):
            if _has_any_face_specs(face_spec):
                return label
        elif face_spec:
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
    requested_glb_name: str | None = None,
) -> Path:
    if requested_glb_name is not None:
        requested_glb_path = partnext_dir / Path(requested_glb_name).name
        if not requested_glb_path.is_file():
            raise FileNotFoundError(f"requested hammer glb was not found: {requested_glb_path}")
        row = find_annotation_row(
            annotation_rows,
            requested_glb_path.name,
            fallback_model_id=requested_glb_path.stem,
        )
        handle_faces = collect_region_face_specs(row, region="handle")
        head_faces = collect_region_face_specs(row, region="head")
        if not _has_any_face_specs(handle_faces) or not _has_any_face_specs(head_faces):
            raise ValueError(f"{requested_glb_path.name}: missing annotated handle or head faces")
        return requested_glb_path

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

        handle_faces = collect_region_face_specs(row, region="handle")
        head_faces = collect_region_face_specs(row, region="head")
        if not _has_any_face_specs(handle_faces) or not _has_any_face_specs(head_faces):
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


def _collect_region_face_geometry(
    geometry_meshes: list[trimesh.Trimesh],
    face_specs: dict[int, set[int]],
) -> tuple[np.ndarray, np.ndarray]:
    centroid_groups: list[np.ndarray] = []
    normal_groups: list[np.ndarray] = []
    for submesh_id, face_ids in sorted(face_specs.items()):
        if not face_ids:
            continue
        if submesh_id < 0 or submesh_id >= len(geometry_meshes):
            raise ValueError(f"submesh id {submesh_id} is out of range for {len(geometry_meshes)} geometries")
        geometry = geometry_meshes[submesh_id]
        face_indices = np.asarray(sorted(face_ids), dtype=np.int64)
        if np.any(face_indices < 0) or np.any(face_indices >= len(geometry.faces)):
            raise ValueError(f"submesh {submesh_id} has invalid face indices")
        centroid_groups.append(np.asarray(geometry.triangles_center[face_indices], dtype=np.float64))
        normal_groups.append(np.asarray(geometry.face_normals[face_indices], dtype=np.float64))
    if not centroid_groups:
        raise ValueError("face specs produced no triangle centroids")
    return np.vstack(centroid_groups), np.vstack(normal_groups)


def _collect_region_centroids(geometry_meshes: list[trimesh.Trimesh], face_specs: dict[int, set[int]]) -> np.ndarray:
    centroids, _ = _collect_region_face_geometry(geometry_meshes, face_specs)
    return centroids


def compute_handle_contact_point(
    geometry_meshes: list[trimesh.Trimesh] | trimesh.Trimesh,
    handle_face_specs: dict[int, set[int]] | set[int],
) -> tuple[np.ndarray, np.ndarray]:
    if not _has_any_face_specs(handle_face_specs):
        raise ValueError("handle face specs are required")

    if isinstance(geometry_meshes, trimesh.Trimesh):
        handle_indices = np.asarray(sorted(handle_face_specs), dtype=np.int64)
        handle_centroids = np.asarray(geometry_meshes.triangles_center[handle_indices], dtype=np.float64)
    else:
        handle_centroids = _collect_region_centroids(geometry_meshes, handle_face_specs)
    handle_center = handle_centroids.mean(axis=0)
    centered = handle_centroids - handle_center
    covariance = centered.T @ centered
    _, eigenvectors = np.linalg.eigh(covariance)
    handle_axis = _normalize(eigenvectors[:, -1])
    return handle_center, handle_axis


def compute_head_functional_point(
    geometry_meshes: list[trimesh.Trimesh] | trimesh.Trimesh,
    head_face_specs: dict[int, set[int]] | set[int],
    handle_axis: np.ndarray,
    handle_contact_point: np.ndarray,
) -> np.ndarray:
    if not _has_any_face_specs(head_face_specs):
        raise ValueError("head face specs are required")

    if isinstance(geometry_meshes, trimesh.Trimesh):
        head_indices = np.asarray(sorted(head_face_specs), dtype=np.int64)
        head_centroids = np.asarray(geometry_meshes.triangles_center[head_indices], dtype=np.float64)
        head_normals = np.asarray(geometry_meshes.face_normals[head_indices], dtype=np.float64)
    else:
        head_centroids, head_normals = _collect_region_face_geometry(geometry_meshes, head_face_specs)

    axis = _normalize(handle_axis)
    head_center = head_centroids.mean(axis=0)
    plane_offsets = head_centroids - head_center
    plane_offsets = plane_offsets - np.outer(plane_offsets @ axis, axis)

    covariance = plane_offsets.T @ plane_offsets
    _, eigenvectors = np.linalg.eigh(covariance)
    work_axis = eigenvectors[:, -1]
    work_axis = work_axis - axis * float(np.dot(work_axis, axis))
    if float(np.linalg.norm(work_axis)) <= 1e-8:
        offsets = head_centroids - np.asarray(handle_contact_point, dtype=np.float64)
        projections = offsets @ axis
        threshold = float(np.quantile(projections, 0.85))
        striking_centroids = head_centroids[projections >= threshold]
        if striking_centroids.size == 0:
            striking_centroids = head_centroids[[int(np.argmax(projections))]]
        return striking_centroids.mean(axis=0)
    work_axis = _normalize(work_axis)

    planar_normals = head_normals - np.outer(head_normals @ axis, axis)
    planar_normal_magnitudes = np.linalg.norm(planar_normals, axis=1)
    valid_planar = planar_normal_magnitudes > 1e-6
    planar_normal_dirs = np.zeros_like(planar_normals)
    planar_normal_dirs[valid_planar] = planar_normals[valid_planar] / planar_normal_magnitudes[valid_planar, None]

    best_mask = None
    best_score = None
    centered_projections = None
    centered_alignments = None
    for sign in (1.0, -1.0):
        direction = work_axis * sign
        projections = plane_offsets @ direction
        alignments = planar_normal_dirs @ direction
        valid_mask = valid_planar.copy()
        if not np.any(valid_mask):
            valid_mask = np.ones(len(head_centroids), dtype=bool)
        projection_threshold = float(np.quantile(projections[valid_mask], 0.85))
        candidate_mask = valid_mask & (projections >= projection_threshold) & (alignments >= 0.85)
        if not np.any(candidate_mask):
            alignment_threshold = float(np.quantile(alignments[valid_mask], 0.9))
            candidate_mask = valid_mask & (projections >= projection_threshold) & (alignments >= alignment_threshold)
        if not np.any(candidate_mask):
            top_index = int(np.argmax(projections + (0.25 * alignments)))
            candidate_mask = np.zeros(len(head_centroids), dtype=bool)
            candidate_mask[top_index] = True

        normalized_projection = (projections - float(np.min(projections))) / (float(np.ptp(projections)) + 1e-9)
        score = float(np.mean(alignments[candidate_mask]) + (0.2 * np.mean(normalized_projection[candidate_mask])))
        if best_score is None or score > best_score:
            best_score = score
            best_mask = candidate_mask
            centered_projections = projections
            centered_alignments = alignments

    if best_mask is None or not np.any(best_mask):
        combined = centered_projections + (0.25 * centered_alignments)
        return head_centroids[[int(np.argmax(combined))]].mean(axis=0)
    return head_centroids[best_mask].mean(axis=0)


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
    base_mesh.visual.vertex_colors = np.tile(
        np.asarray([180, 180, 180, 255], dtype=np.uint8),
        (len(base_mesh.vertices), 1),
    )

    marker_radius = max(float(np.max(mesh.extents)) * 0.1, 1e-4)

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
    requested_glb_name: str | None = None,
) -> PreparedHammerAsset:
    annotation_rows = load_annotation_rows(annotation_path)
    reference_model_data = json.loads(
        _resolve_reference_model_data_path(reference_model_data_path).read_text(encoding="utf-8")
    )
    reference_loaded_extents = np.asarray(reference_model_data["extents"], dtype=np.float64) * np.asarray(
        reference_model_data["scale"], dtype=np.float64
    )

    candidate_glb = select_candidate_glb(
        partnext_dir,
        annotation_rows,
        reference_loaded_extents,
        requested_glb_name=requested_glb_name,
    )
    annotation_row = find_annotation_row(annotation_rows, candidate_glb.name, fallback_model_id=candidate_glb.stem)
    mesh = _load_mesh(candidate_glb)
    geometry_meshes = load_scene_geometries(candidate_glb)

    handle_faces_by_label = _collect_label_face_specs(annotation_row, HANDLE_CORE_LABELS)
    handle_face_specs: dict[int, set[int]] = {}
    handle_labels_used: list[str] = []
    for label, face_specs in handle_faces_by_label.items():
        if _has_any_face_specs(face_specs):
            _merge_face_specs(handle_face_specs, face_specs)
            handle_labels_used.append(label)
    if not _has_any_face_specs(handle_face_specs):
        handle_faces_by_label = _collect_label_face_specs(annotation_row, HANDLE_LABELS)
        for label, face_specs in handle_faces_by_label.items():
            if _has_any_face_specs(face_specs):
                _merge_face_specs(handle_face_specs, face_specs)
                handle_labels_used.append(label)
    striking_faces_by_label = _collect_label_face_specs(annotation_row, STRIKING_LABELS)
    striking_label = pick_striking_label(striking_faces_by_label)
    head_face_specs = striking_faces_by_label[striking_label]
    if not _has_any_face_specs(handle_face_specs):
        raise ValueError(f"selected candidate {candidate_glb.name} is missing handle face ids")
    if not _has_any_face_specs(head_face_specs):
        raise ValueError(f"selected candidate {candidate_glb.name} is missing head face ids")

    contact_point, handle_axis = compute_handle_contact_point(geometry_meshes, handle_face_specs)
    functional_point = compute_head_functional_point(geometry_meshes, head_face_specs, handle_axis, contact_point)
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
            "handle": handle_labels_used,
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
    "collect_region_face_specs",
    "compute_handle_contact_point",
    "compute_head_functional_point",
    "estimate_uniform_scale",
    "find_annotation_row",
    "load_annotation_rows",
    "load_scene_geometries",
    "pick_striking_label",
    "render_preview_ply",
    "select_candidate_glb",
    "write_asset_package",
]
