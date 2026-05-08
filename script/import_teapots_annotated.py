"""
利用 PartNet 标注数据导入茶壶 GLB，自动提取 Handle 位置生成 contact_points_pose。

用法:
    cd ~/github/RoboTwin_geo
    conda activate RoboTwin
    python script/import_teapots_annotated.py --src ~/Datasets/Teapot --dst assets/objects/092_teapot --target-size 0.20

标注数据 (annotation.jsonl) 包含每个茶壶的部件分割 (Body/Spout/Handle/Lid 等)，
脚本用 Handle 面片的中心点作为抓取位置，生成 8 个方向的 contact_points_pose。
"""

import argparse
import json
import numpy as np
from pathlib import Path
import shutil

try:
    import trimesh
except ImportError:
    print("pip install trimesh")
    exit(1)

try:
    import coacd
except ImportError:
    print("pip install coacd")
    exit(1)


def load_annotations(ann_path: Path) -> dict:
    """读取 annotation.jsonl，返回 {model_id: annotation} 字典"""
    anns = {}
    with open(ann_path) as f:
        for line in f:
            d = json.loads(line)
            anns[d["model_id"]] = d
    return anns


def find_handle_mask_id(ann: dict):
    """从 hierarchyList 中找到 Handle 部件的 maskId"""
    hierarchy = json.loads(ann["hierarchyList"])

    def search(nodes):
        for node in nodes:
            if node["name"] == "Handle" and "maskId" in node:
                return node["maskId"]
            result = search(node.get("children", []))
            if result is not None:
                return result
        return None

    return search(hierarchy)


def get_handle_face_indices(ann: dict, handle_mask_id: int) -> list:
    """获取 Handle 的全局面索引（处理多 mesh 情况）"""
    masks = json.loads(ann["masks"])
    mesh_face_num = json.loads(ann["mesh_face_num"])

    handle_mask = masks.get(str(handle_mask_id), {})
    if not handle_mask:
        return []

    # 计算每个 mesh 的面索引偏移
    mesh_indices = sorted(mesh_face_num.keys(), key=int)
    offsets = {}
    cumulative = 0
    for mi in mesh_indices:
        offsets[mi] = cumulative
        cumulative += mesh_face_num[mi]

    # 合并所有 mesh 中的 Handle 面索引
    global_indices = []
    for mesh_idx, face_list in handle_mask.items():
        offset = offsets.get(mesh_idx, 0)
        global_indices.extend([f + offset for f in face_list])

    return global_indices


def get_handle_center(mesh, face_indices: list) -> np.ndarray:
    """从面索引提取 Handle 顶点的中心坐标"""
    valid = [i for i in face_indices if i < len(mesh.faces)]
    if not valid:
        return None
    handle_faces = mesh.faces[valid]
    handle_verts = mesh.vertices[np.unique(handle_faces.flatten())]
    return handle_verts.mean(axis=0)


def generate_contact_poses(handle_center: np.ndarray) -> list:
    """在 Handle 中心位置生成 8 个方向的 contact_points_pose (绕 Y 轴旋转)"""
    hx, hy, hz = handle_center.tolist()
    poses = []
    for i in range(8):
        angle = i * np.pi / 4  # 0, 45, 90, ..., 315 度
        c, s = np.cos(angle), np.sin(angle)
        # 绕 Y 轴旋转 (Y 是 GLB 坐标系的 up)
        pose = [
            [round(c, 6), 0.0, round(-s, 6), round(hx, 5)],
            [0.0, 1.0, 0.0, round(hy, 5)],
            [round(s, 6), 0.0, round(c, 6), round(hz, 5)],
            [0.0, 0.0, 0.0, 1.0],
        ]
        poses.append(pose)
    return poses


def coacd_decompose(mesh, threshold=0.05):
    """CoACD 凸分解"""
    mesh_coacd = coacd.Mesh(mesh.vertices, mesh.faces)
    parts = coacd.run_coacd(mesh_coacd, threshold=threshold)
    return [trimesh.Trimesh(vertices=v, faces=f) for v, f in parts]


def save_multi_part_glb(meshes, output_path: Path):
    """保存多部件碰撞体 GLB"""
    scene = trimesh.Scene()
    for i, m in enumerate(meshes):
        scene.add_geometry(m, node_name=f"part_{i}")
    scene.export(str(output_path))


def process_one(glb_path: Path, ann: dict, target_size: float):
    """处理单个 GLB: 加载网格、提取把手、生成碰撞体和 model_data"""
    mesh = trimesh.load(str(glb_path), force="mesh")

    # AABB
    bounds = mesh.bounds
    center = ((bounds[0] + bounds[1]) / 2).tolist()
    extents = (bounds[1] - bounds[0]).tolist()
    max_extent = max(extents)
    if max_extent < 1e-6:
        raise ValueError("模型尺寸接近零")
    scale_factor = target_size / max_extent

    # Handle 位置
    handle_mask_id = find_handle_mask_id(ann)
    handle_center = None
    if handle_mask_id is not None:
        face_indices = get_handle_face_indices(ann, handle_mask_id)
        if face_indices:
            handle_center = get_handle_center(mesh, face_indices)

    # contact_points_pose
    if handle_center is not None:
        contact_poses = generate_contact_poses(handle_center)
        contact_group = [list(range(8))]
        contact_mask = [True]
        contact_desc = ["Grasping the handle of the teapot"] * 8
    else:
        contact_poses = []
        contact_group = []
        contact_mask = []
        contact_desc = []

    # 清理碎片 + CoACD
    components = mesh.split(only_watertight=False)
    big_parts = [c for c in components if len(c.faces) > 50]
    cleaned = trimesh.util.concatenate(big_parts) if big_parts else mesh
    collision_parts = coacd_decompose(cleaned, threshold=0.05)

    center_y = center[1]
    model_data = {
        "center": center,
        "extents": extents,
        "scale": [scale_factor, scale_factor, scale_factor],
        "target_pose": [
            [[1.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, center_y],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]]
        ],
        "contact_points_pose": contact_poses,
        "transform_matrix": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ],
        "functional_matrix": [
            [[1.0, 0.0, 0.0, 0.0],
             [0.0, 6.123233995736766e-17, -1.0, center_y],
             [0.0, 1.0, 6.123233995736766e-17, 0.0],
             [0.0, 0.0, 0.0, 1.0]]
        ],
        "orientation_point": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, center_y * 2],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ],
        "contact_points_group": contact_group,
        "contact_points_mask": contact_mask,
        "contact_points_discription": contact_desc,
        "target_point_discription": ["The center of the teapot."],
        "functional_point_discription": [
            "Point 0: The center of the teapot and the functional axis is vertical and the bottom of the teapot is downward."
        ],
        "stable": True,
    }

    return mesh, collision_parts, model_data, handle_center


def main():
    parser = argparse.ArgumentParser(description="利用标注数据导入茶壶 (Handle 自动抓取点)")
    parser.add_argument("--src", type=str, required=True, help="GLB+annotation 源目录")
    parser.add_argument("--dst", type=str, required=True, help="输出资产目录")
    parser.add_argument("--target-size", type=float, default=0.20, help="目标最大尺寸(米)")
    parser.add_argument("--max-count", type=int, default=None, help="最多导入多少个")
    args = parser.parse_args()

    src_dir = Path(args.src).expanduser().resolve()
    dst_dir = Path(args.dst).expanduser().resolve()

    # 加载标注
    ann_path = src_dir / "annotation.jsonl"
    if not ann_path.exists():
        print(f"找不到标注文件: {ann_path}")
        return
    annotations = load_annotations(ann_path)
    print(f"加载了 {len(annotations)} 条标注")

    # 收集 GLB
    glb_files = sorted(src_dir.glob("*.glb"))
    if args.max_count:
        glb_files = glb_files[:args.max_count]
    print(f"找到 {len(glb_files)} 个 GLB 文件")
    print(f"目标尺寸: {args.target_size}m, 输出: {dst_dir}")
    print()

    # 清理旧数据
    collision_dir = dst_dir / "collision"
    visual_dir = dst_dir / "visual"
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    collision_dir.mkdir(parents=True)
    visual_dir.mkdir(parents=True)

    results = []
    for idx, glb_path in enumerate(glb_files):
        model_id = glb_path.stem
        has_ann = model_id in annotations
        print(f"[{idx:2d}] {model_id[:16]}... ", end="", flush=True)

        if not has_ann:
            print("跳过 (无标注)")
            continue

        try:
            ann = annotations[model_id]
            visual_mesh, collision_parts, model_data, handle_center = \
                process_one(glb_path, ann, args.target_size)

            # 保存
            visual_mesh.export(str(visual_dir / f"base{idx}.glb"))
            save_multi_part_glb(collision_parts, collision_dir / f"base{idx}.glb")
            with open(dst_dir / f"model_data{idx}.json", "w") as f:
                json.dump(model_data, f, indent=4)

            n_parts = len(collision_parts)
            handle_str = f"handle=[{handle_center[0]:.2f},{handle_center[1]:.2f},{handle_center[2]:.2f}]" if handle_center is not None else "NO HANDLE"
            scale = model_data["scale"][0]
            print(f"OK  scale={scale:.4f}  parts={n_parts}  {handle_str}")
            results.append(idx)

        except Exception as e:
            print(f"失败: {e}")

    # points_info.json
    with open(dst_dir / "points_info.json", "w") as f:
        json.dump({
            "contact_points": [{
                "id": 0,
                "description": "On the handle of the teapot",
                "usage": "Used for grasping the teapot by its handle."
            }],
            "functional_points": [{
                "id": 0,
                "description": "The center of the teapot",
                "usage": "Used to check teapot orientation and placement position."
            }]
        }, f, indent=4)

    print(f"\n完成! 成功导入 {len(results)}/{len(glb_files)} 个模型")
    print(f"成功的 model_id 列表: {results}")
    print(f"\n下一步: 运行稳定性测试脚本筛选可用模型")


if __name__ == "__main__":
    main()
