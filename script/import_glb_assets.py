"""
批量导入 GLB 模型到 RoboTwin 资产格式。

碰撞体使用 CoACD 凸分解, 生成多部件 GLB,
与项目现有资产 (053_teanet 32部件, 039_mug 27部件等) 一致。

用法:
    python script/import_glb_assets.py \
        --src ~/Datasets/Teapot \
        --dst assets/objects/092_teapot \
        --target-size 0.20 \
        --max-count 10
"""

import argparse
import json
import numpy as np
from pathlib import Path

try:
    import trimesh
except ImportError:
    print("需要安装 trimesh: pip install trimesh")
    exit(1)

try:
    import coacd
except ImportError:
    print("需要安装 coacd: pip install coacd")
    exit(1)


def coacd_decompose(mesh, threshold=0.05):
    """
    使用 CoACD 对网格做凸分解。
    返回 trimesh 网格列表。
    """
    mesh_coacd = coacd.Mesh(mesh.vertices, mesh.faces)
    parts = coacd.run_coacd(mesh_coacd, threshold=threshold)
    result = []
    for vertices, faces in parts:
        result.append(trimesh.Trimesh(vertices=vertices, faces=faces))
    return result


def process_glb(glb_path: Path, target_size: float):
    """
    加载 GLB, 清理碎片, 用 AABB 计算 center/extents, 用 CoACD 生成多部件碰撞体。
    """
    mesh = trimesh.load(str(glb_path), force="mesh")

    # AABB (在清理前计算, 因为 visual 用原始网格)
    bounds = mesh.bounds
    center = ((bounds[0] + bounds[1]) / 2).tolist()
    extents = (bounds[1] - bounds[0]).tolist()
    max_extent = max(extents)

    if max_extent < 1e-6:
        raise ValueError("模型尺寸接近零")

    scale_factor = target_size / max_extent

    # 清理碎片: 只保留面数 > 50 的连通分量
    components = mesh.split(only_watertight=False)
    big_parts = [c for c in components if len(c.faces) > 50]
    if not big_parts:
        # 如果没有大分量, 用整体
        cleaned = mesh
    else:
        cleaned = trimesh.util.concatenate(big_parts)

    # CoACD 凸分解
    collision_parts = coacd_decompose(cleaned, threshold=0.05)

    return scale_factor, center, extents, mesh, collision_parts


def save_multi_part_glb(meshes, output_path: Path):
    """
    多个 trimesh 网格保存为一个 GLB (多子网格),
    与项目现有碰撞体格式一致。
    """
    scene = trimesh.Scene()
    for i, m in enumerate(meshes):
        scene.add_geometry(m, node_name=f"part_{i}")
    scene.export(str(output_path))


def create_model_data(center, extents, scale_factor) -> dict:
    s = scale_factor
    center_y = center[1]

    return {
        "center": center,
        "extents": extents,
        "scale": [s, s, s],
        "target_pose": [
            [[1.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, center_y],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]]
        ],
        "contact_points_pose": [],
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
        "contact_points_group": [],
        "contact_points_mask": [],
        "contact_points_discription": [],
        "target_point_discription": [
            "The center of the teapot."
        ],
        "functional_point_discription": [
            "Point 0: The center of the teapot and the functional axis is vertical and the bottom of the teapot is downward."
        ],
        "stable": True
    }


def create_points_info() -> dict:
    return {
        "contact_points": [
            {
                "id": 0,
                "description": "On the handle of the teapot",
                "usage": "Used for grasping the teapot."
            }
        ],
        "functional_points": [
            {
                "id": 0,
                "description": "The center of the teapot",
                "usage": "Used to check teapot orientation and placement position."
            }
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="批量导入 GLB 到 RoboTwin 资产格式 (CoACD 凸分解)")
    parser.add_argument("--src", type=str, required=True, help="GLB 文件源目录")
    parser.add_argument("--dst", type=str, required=True, help="输出资产目录")
    parser.add_argument("--target-size", type=float, default=0.20,
                        help="目标最大尺寸 (米), 默认 0.20")
    parser.add_argument("--max-count", type=int, default=None,
                        help="最多导入多少个, 默认全部")
    args = parser.parse_args()

    src_dir = Path(args.src).expanduser().resolve()
    dst_dir = Path(args.dst).expanduser().resolve()

    glb_files = sorted(src_dir.glob("*.glb"))
    if args.max_count:
        glb_files = glb_files[:args.max_count]

    if not glb_files:
        print(f"在 {src_dir} 中没有找到 GLB 文件")
        return

    print(f"找到 {len(glb_files)} 个 GLB 文件")
    print(f"目标尺寸: {args.target_size}m ({args.target_size*100}cm)")
    print(f"输出目录: {dst_dir}")
    print(f"碰撞体: CoACD 多部件凸分解")
    print()

    collision_dir = dst_dir / "collision"
    visual_dir = dst_dir / "visual"
    collision_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    skipped = []

    for idx, glb_path in enumerate(glb_files):
        print(f"[{idx}] 处理 {glb_path.name} ... ", end="", flush=True)

        try:
            scale_factor, center, extents, visual_mesh, collision_parts = \
                process_glb(glb_path, args.target_size)

            max_ext = max(extents)
            sim_size = max_ext * scale_factor
            total_faces = sum(len(m.faces) for m in collision_parts)

            # 保存可视化网格
            dst_visual = visual_dir / f"base{idx}.glb"
            visual_mesh.export(str(dst_visual))

            # 保存碰撞体 (CoACD 多部件)
            dst_collision = collision_dir / f"base{idx}.glb"
            save_multi_part_glb(collision_parts, dst_collision)

            # 生成 model_data
            model_data = create_model_data(center, extents, scale_factor)
            model_data_path = dst_dir / f"model_data{idx}.json"
            with open(model_data_path, "w") as f:
                json.dump(model_data, f, indent=4)

            print(f"OK  scale={scale_factor:.4f}  sim={sim_size*100:.1f}cm  "
                  f"parts={len(collision_parts)}  faces={total_faces}")
            success_count += 1

        except Exception as e:
            print(f"跳过: {e}")
            skipped.append((glb_path.name, str(e)))

    # points_info.json
    points_info_path = dst_dir / "points_info.json"
    with open(points_info_path, "w") as f:
        json.dump(create_points_info(), f, indent=4)

    print()
    print(f"完成! 成功导入 {success_count}/{len(glb_files)} 个模型")
    if skipped:
        print(f"跳过 {len(skipped)} 个:")
        for name, reason in skipped:
            print(f"  {name}: {reason}")


if __name__ == "__main__":
    main()
