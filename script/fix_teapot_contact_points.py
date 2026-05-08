"""
修复 092_teapot 的 contact_points_pose：
将抓取点从把手中心改为壶身中心（与 bottle 相同的方式），
让夹爪能从侧面抓住壶身。

保持 x_axis=[0,1,0] (沿 Y/up 方向)，z_axis 在 XZ 平面旋转 8 个方向。
这和 001_bottle 的 contact_points_pose 格式完全一致。
"""

import json
import numpy as np
from pathlib import Path


def generate_body_center_poses(center_y: float) -> list:
    """在壶身中心生成 8 个方向的 contact_points_pose，格式与 bottle 一致"""
    poses = []
    # 8 个方向: 0, 45, 90, 135, 180, 225, 270, 315 度
    # z_axis 在 XZ 平面旋转 (approach direction)
    # x_axis 始终为 [0, 1, 0] (Y-up)
    angles = [0, 90, 180, 270, 45, 135, 225, 315]
    for deg in angles:
        rad = np.radians(deg)
        cz = round(np.cos(rad), 6)
        sz = round(np.sin(rad), 6)
        # 旋转矩阵: x_axis=[0,1,0], y_axis=[-sz,0,cz] (cross product), z_axis=[cz,0,sz] -> no
        # 参考 bottle: x_axis 指向 Y (up), z_axis 在 XZ 平面
        # R 的列向量: col0=x_axis, col1=y_axis, col2=z_axis
        # col0 = [0, 1, 0]  (gripper "up" = model Y)
        # col2 = [sz, 0, cz] (approach direction in XZ plane)
        # col1 = cross(col2, col0) = [cz*0-0, 0*sz-cz*0, 0-sz*1] ???
        # Actually: col1 = cross(col2, col0)
        # col2 x col0 = [0*0-cz*0, cz*0-sz*0, sz*1-0*0] = [0, 0, sz] ??? That's wrong
        # Let me just use the bottle pattern directly:
        # For bottle pose 0 (z=[0,0,1]): matrix rows are
        #   row0: [R00, R01, R02] = cols transposed
        # Actually the matrix is stored ROW-MAJOR. So matrix[row][col].
        # Column 0 (x_axis): [M[0][0], M[1][0], M[2][0]]
        # Column 2 (z_axis): [M[0][2], M[1][2], M[2][2]]
        #
        # For bottle with z_axis=[sz, 0, cz] and x_axis=[0, 1, 0]:
        # y_axis = z_axis x x_axis = [sz,0,cz] x [0,1,0] = [0*0-cz*1, cz*0-sz*0, sz*1-0*0] = [-cz, 0, sz]
        #
        # Matrix (row-major):
        # row0: [x0, y0, z0] = [0, -cz, sz]
        # row1: [x1, y1, z1] = [1,  0,  0 ]
        # row2: [x2, y2, z2] = [0,  sz, cz]

        pose = [
            [0.0, round(-cz, 6), round(sz, 6), 0.0],
            [1.0, 0.0, 0.0, round(center_y, 5)],
            [0.0, round(sz, 6), round(cz, 6), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        poses.append(pose)
    return poses


def main():
    asset_dir = Path("assets/objects/092_teapot")
    updated = 0

    for f in sorted(asset_dir.glob("model_data*.json")):
        mid = f.stem.replace("model_data", "")
        with open(f) as fp:
            data = json.load(fp)

        center_y = data["center"][1]
        data["contact_points_pose"] = generate_body_center_poses(center_y)
        data["contact_points_group"] = [list(range(8))]
        data["contact_points_mask"] = [True]
        data["contact_points_discription"] = ["Grasping the body of the teapot"] * 8

        with open(f, "w") as fp:
            json.dump(data, fp, indent=4)

        print(f"model_data{mid}: center_y={center_y:.4f} -> 8 body-center contact poses")
        updated += 1

    print(f"\n更新了 {updated} 个模型的 contact_points_pose")


if __name__ == "__main__":
    main()
