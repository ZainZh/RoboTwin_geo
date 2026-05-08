"""
交互式设置目标位姿：
1. 打开仿真窗口
2. 在窗口中拖动机械臂到目标位置
3. 按 'p' 打印当前左臂末端执行器的 pose
4. 按 'q' 退出

用法:
    cd ~/github/RoboTwin_geo
    conda activate RoboTwin
    python script/interactive_target.py
"""
import sys, os
sys.path.append(".")
import numpy as np
import yaml
import importlib
from envs._GLOBAL_CONFIGS import CONFIGS_PATH

task_name = "adjust_kettle"
with open("task_config/demo_clean.yml") as f:
    args = yaml.safe_load(f)
args['task_name'] = task_name
args['need_plan'] = True
args['render_freq'] = 20  # 开启渲染

with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml")) as f:
    _et = yaml.safe_load(f)
et = args.get("embodiment")
rf = _et[et[0]]["file_path"]
args["left_robot_file"] = rf
args["right_robot_file"] = rf
args["dual_arm_embodied"] = True
with open(os.path.join(rf, "config.yml")) as f:
    ec = yaml.safe_load(f)
args["left_embodiment_config"] = ec
args["right_embodiment_config"] = ec
args["embodiment_name"] = str(et[0])
args["save_path"] = f"./data/{task_name}/interactive"
args["task_config"] = "demo_clean"

envs_module = importlib.import_module(f"envs.{task_name}")
task_class = getattr(envs_module, task_name)
task = task_class()

print("正在初始化仿真环境...")
task.setup_demo(now_ep_num=0, seed=1, **args)

print("\n=== 仿真窗口已打开 ===")
print("在窗口中查看场景，然后在这里输入命令：")
print("  p  - 打印当前左臂/右臂末端执行器 pose")
print("  q  - 退出")
print()

while True:
    cmd = input("> ").strip().lower()
    if cmd == 'q':
        break
    elif cmd == 'p':
        left_ee = task.robot.get_left_ee_pose()
        right_ee = task.robot.get_right_ee_pose()
        print(f"\n左臂 EE pose: {[round(v, 4) for v in left_ee]}")
        print(f"右臂 EE pose: {[round(v, 4) for v in right_ee]}")

        # 也打印茶壶当前位姿
        kp = task.kettle.get_pose()
        print(f"茶壶 pose: pos=[{kp.p[0]:.4f}, {kp.p[1]:.4f}, {kp.p[2]:.4f}] q=[{kp.q[0]:.4f}, {kp.q[1]:.4f}, {kp.q[2]:.4f}, {kp.q[3]:.4f}]")
        print()
    else:
        print("未知命令，输入 p 或 q")
