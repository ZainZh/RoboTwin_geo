"""
诊断物体稳定性: 在 SAPIEN 中加载物体, 运行物理, 打印位姿变化。
用法: python script/debug_stability.py 092_teapot 0
"""
import sys
sys.path.append(".")
import sapien
from sapien.render import set_global_config
import numpy as np
import json
from pathlib import Path
from envs.utils.transforms import cal_quat_dis

modelname = sys.argv[1] if len(sys.argv) > 1 else "092_teapot"
model_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0

# 加载 model_data
modeldir = Path("assets/objects") / modelname
with open(modeldir / f"model_data{model_id}.json") as f:
    model_data = json.load(f)
scale = model_data["scale"]
center = model_data["center"]
extents = model_data["extents"]

print(f"模型: {modelname}/base{model_id}")
print(f"scale: {scale}")
print(f"center: {center}")
print(f"extents: {extents}")
print(f"sim_size(cm): {[e*s*100 for e,s in zip(extents, scale)]}")
bottom_y = center[1] - extents[1] / 2
print(f"bottom_y (local): {bottom_y:.4f}")
print(f"bottom_offset (sim): {bottom_y * scale[1] * 100:.2f} cm")

# 创建场景
set_global_config(max_num_materials=50000, max_num_textures=50000)
scene = sapien.Scene()
scene.set_timestep(1 / 250)
scene.add_ground(0.74)  # 桌面高度

# 创建物体
collision_file = modeldir / "collision" / f"base{model_id}.glb"
visual_file = modeldir / "visual" / f"base{model_id}.glb"

builder = scene.create_actor_builder()
builder.set_physx_body_type("dynamic")
builder.add_multiple_convex_collisions_from_file(filename=str(collision_file), scale=scale)
builder.add_visual_from_file(filename=str(visual_file), scale=scale)
actor = builder.build(name=modelname)

# 放置: 正立, z=0.78
pose = sapien.Pose([0, 0, 0.78], [1, 0, 0, 0])
actor.set_pose(pose)

print(f"\n初始位姿: pos={pose.p.tolist()}, quat={pose.q.tolist()}")
print(f"\n--- 物理仿真 ---")

# 仿真 2500 步 (同 check_stable)
poses = []
for step in range(2500):
    scene.step()
    p = actor.get_pose()
    poses.append(p)
    if step % 250 == 0 or step == 2499:
        print(f"step {step:4d}: pos=[{p.p[0]:.4f}, {p.p[1]:.4f}, {p.p[2]:.4f}]  "
              f"quat=[{p.q[0]:.4f}, {p.q[1]:.4f}, {p.q[2]:.4f}, {p.q[3]:.4f}]")

# 检查稳定性 (同 _base_task.py 逻辑)
final_pose = poses[-1]
max_diff = 0
for p in poses[-200:]:
    diff = abs(cal_quat_dis(final_pose.q, p.q) * 180)
    max_diff = max(max_diff, diff)

print(f"\n最后200步最大角度变化: {max_diff:.4f}° (阈值: 3.0°)")
print(f"稳定: {'是' if max_diff < 3.0 else '否'}")

# 也打印位置变化
final_z = final_pose.p[2]
min_z = min(p.p[2] for p in poses[-200:])
max_z = max(p.p[2] for p in poses[-200:])
print(f"最后200步 Z 范围: [{min_z:.4f}, {max_z:.4f}], 变化: {(max_z-min_z)*100:.2f}cm")
