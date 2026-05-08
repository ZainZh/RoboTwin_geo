"""
测试每个 model_id 的稳定性 (固定 seed, 变 model_id)。
"""
import sys, os
sys.path.append(".")
import numpy as np
import yaml
import importlib
from envs.utils.transforms import cal_quat_dis
from envs.utils.create_actor import UnStableError
from envs._GLOBAL_CONFIGS import CONFIGS_PATH
from envs._base_task import Base_Task

task_name = "adjust_kettle"

# 配置
with open("task_config/demo_clean.yml") as f:
    args = yaml.safe_load(f)
args['task_name'] = task_name
args['need_plan'] = True
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
args["save_path"] = f"./data/{task_name}/demo_clean"
args["task_config"] = "demo_clean"

# Patch load_actors 以固定 model_id
envs_module = importlib.import_module(f"envs.{task_name}")
task_class = getattr(envs_module, task_name)

original_load_actors = task_class.load_actors
forced_model_id = [None]

def patched_load_actors(self):
    original_load_actors(self)
    # 覆盖 model_id 后重新测试不现实, 直接记录选中了哪个
    print(f"    [model_id={self.model_id}, qpose_tag={self.qpose_tag}]", end=" ")

task_class.load_actors = patched_load_actors

task = task_class()

# 测试 50 个 seed
results = {}
for seed in range(50):
    try:
        task.setup_demo(now_ep_num=0, seed=seed, **args)
        mid = task.model_id
        qt = task.qpose_tag
        results.setdefault(mid, {"stable": 0, "unstable": 0})
        results[mid]["stable"] += 1
        print(f"seed {seed:3d}: STABLE")
    except UnStableError:
        mid = task.model_id if hasattr(task, 'model_id') else '?'
        results.setdefault(mid, {"stable": 0, "unstable": 0})
        results[mid]["unstable"] += 1
        print(f"seed {seed:3d}: UNSTABLE")
    except Exception as e:
        print(f"seed {seed:3d}: ERROR - {e}")

print(f"\n=== 按 model_id 统计 ===")
for mid in sorted(results.keys()):
    r = results[mid]
    total = r["stable"] + r["unstable"]
    pct = r["stable"] / total * 100 if total > 0 else 0
    print(f"model_id {mid}: {r['stable']}/{total} stable ({pct:.0f}%)")
