"""
修复 092_teapot 的 contact_points_pose：
从把手外侧朝壶身方向接近，夹爪先接触把手。

contact_points_pose 格式与 bottle 一致（x_axis=[0,1,0]）。
方向以「从外侧朝壶身」为主（径向方向），加上小角度偏转。
"""

import json
import numpy as np
from pathlib import Path
import trimesh


def load_annotations(ann_path: Path) -> dict:
    anns = {}
    with open(ann_path) as f:
        for line in f:
            d = json.loads(line)
            anns[d['model_id']] = d
    return anns


def find_handle_mask_id(ann: dict):
    hierarchy = json.loads(ann['hierarchyList'])
    def search(nodes):
        for node in nodes:
            if node['name'] == 'Handle' and 'maskId' in node:
                return node['maskId']
            result = search(node.get('children', []))
            if result is not None:
                return result
        return None
    return search(hierarchy)


def get_handle_vertices(mesh, ann: dict):
    masks = json.loads(ann['masks'])
    mesh_face_num = json.loads(ann['mesh_face_num'])
    handle_mask_id = find_handle_mask_id(ann)
    if handle_mask_id is None:
        return None
    handle_mask = masks.get(str(handle_mask_id), {})
    if not handle_mask:
        return None
    mesh_indices = sorted(mesh_face_num.keys(), key=int)
    offsets = {}
    cumulative = 0
    for mi in mesh_indices:
        offsets[mi] = cumulative
        cumulative += mesh_face_num[mi]
    global_face_indices = []
    for mesh_idx, face_list in handle_mask.items():
        offset = offsets.get(mesh_idx, 0)
        global_face_indices.extend([f + offset for f in face_list])
    valid = [i for i in global_face_indices if i < len(mesh.faces)]
    if not valid:
        return None
    handle_faces = mesh.faces[valid]
    handle_verts = mesh.vertices[np.unique(handle_faces.flatten())]
    return handle_verts


def get_handle_grasp_point(handle_verts: np.ndarray, body_center: np.ndarray) -> np.ndarray:
    body_xz = np.array([body_center[0], body_center[2]])
    handle_xz = handle_verts[:, [0, 2]]
    radial_dist = np.linalg.norm(handle_xz - body_xz, axis=1)
    r_threshold = np.percentile(radial_dist, 70)
    outer_mask = radial_dist >= r_threshold
    outer_verts = handle_verts[outer_mask]
    y_median = np.median(outer_verts[:, 1])
    upper_mask = outer_verts[:, 1] >= y_median
    if upper_mask.sum() > 0:
        return outer_verts[upper_mask].mean(axis=0)
    return outer_verts.mean(axis=0)


def make_pose(position, z_axis):
    """x_axis=[0,1,0], z_axis=approach direction"""
    z = np.array(z_axis, dtype=float)
    z /= np.linalg.norm(z)
    x = np.array([0.0, 1.0, 0.0])
    x = x - np.dot(x, z) * z
    norm = np.linalg.norm(x)
    if norm < 1e-6:
        x = np.array([1.0, 0.0, 0.0])
        x = x - np.dot(x, z) * z
        norm = np.linalg.norm(x)
    x /= norm
    y = np.cross(z, x)
    def r(v): return round(float(v), 6)
    return [
        [r(x[0]), r(y[0]), r(z[0]), round(float(position[0]), 5)],
        [r(x[1]), r(y[1]), r(z[1]), round(float(position[1]), 5)],
        [r(x[2]), r(y[2]), r(z[2]), round(float(position[2]), 5)],
        [0.0, 0.0, 0.0, 1.0],
    ]


def generate_handle_poses(grasp_pt: np.ndarray, body_center: np.ndarray) -> list:
    """
    以「从外侧朝壶身」方向为主，生成 8 个接近方向。
    这样 pre-grasp 在把手外侧，夹爪向内移动时先碰到把手。
    """
    # 径向方向：从把手指向壶身（从外侧朝内）
    dx = body_center[0] - grasp_pt[0]
    dz = body_center[2] - grasp_pt[2]
    norm_xz = np.sqrt(dx * dx + dz * dz)
    if norm_xz < 1e-6:
        radial = np.array([1.0, 0.0])
    else:
        radial = np.array([dx / norm_xz, dz / norm_xz])

    poses = []
    # 8 个方向，以径向（0度=朝壶身）为中心，±90度范围
    # 0度=从外侧朝壶身（最好），90度=切线，180度=远离壶身
    for deg in [0, 15, -15, 30, -30, 50, -50, 70]:
        rad = np.radians(deg)
        c, s = np.cos(rad), np.sin(rad)
        # 绕Y轴旋转径向方向
        z_xz = np.array([
            radial[0] * c - radial[1] * s,
            radial[0] * s + radial[1] * c,
        ])
        z3d = np.array([z_xz[0], 0.0, z_xz[1]])
        poses.append(make_pose(grasp_pt, z3d))

    return poses


def main():
    src_dir = Path('/home/zhanglr/Datasets/Teapot')
    dst_dir = Path('assets/objects/092_teapot')

    annotations = load_annotations(src_dir / 'annotation.jsonl')
    glb_files = sorted(src_dir.glob('*.glb'))

    updated = 0
    no_handle = []

    for idx, glb_path in enumerate(glb_files):
        model_id = glb_path.stem
        if model_id not in annotations:
            continue
        md_path = dst_dir / f'model_data{idx}.json'
        if not md_path.exists():
            continue
        ann = annotations[model_id]
        with open(md_path) as f:
            md = json.load(f)
        body_center = np.array(md['center'])

        try:
            mesh = trimesh.load(str(glb_path), force='mesh')
        except Exception as e:
            continue

        handle_verts = get_handle_vertices(mesh, ann)
        if handle_verts is None or len(handle_verts) == 0:
            no_handle.append(idx)
            continue

        grasp_pt = get_handle_grasp_point(handle_verts, body_center)
        poses = generate_handle_poses(grasp_pt, body_center)

        md['contact_points_pose'] = poses
        md['contact_points_group'] = [list(range(len(poses)))]
        md['contact_points_mask'] = [True]
        md['contact_points_discription'] = ['Grasping the handle of the teapot'] * len(poses)

        with open(md_path, 'w') as f:
            json.dump(md, f, indent=4)

        scale = md['scale'][0]
        gp = grasp_pt * scale
        print(f"[{idx:2d}] grasp=[{gp[0]:.4f}, {gp[1]:.4f}, {gp[2]:.4f}]")
        updated += 1

    print(f"\nupdated {updated} models")


if __name__ == '__main__':
    main()
