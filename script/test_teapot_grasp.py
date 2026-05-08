"""
快速测试每个 model_id 能否完成抓取+放置任务 (不仅是稳定性)。
每个模型跑 3 个 seed，打印 plan_success 和 check_success。
"""
import sys, os
sys.path.append(".")
import numpy as np
import yaml
import importlib
from envs.utils.create_actor import UnStableError
from envs._GLOBAL_CONFIGS import CONFIGS_PATH
from envs.utils import rand_create_actor
from pathlib import Path

task_name = "adjust_kettle"
with open("task_config/demo_clean.yml") as f:
    args = yaml.safe_load(f)
args['task_name'] = task_name
args['need_plan'] = True
args['render_freq'] = 0
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

envs_module = importlib.import_module(f"envs.{task_name}")
task_class = getattr(envs_module, task_name)

forced_id = [None]

def patched_load(self):
    self.qpose_tag = np.random.randint(0, 2)
    qposes = [[0.707, 0.707, 0, 0], [0.707, 0.707, 0, 0]]
    xlims = [[-0.12, -0.08], [0.08, 0.12]]
    self.model_id = forced_id[0]
    self.kettle = rand_create_actor(
        self, xlim=xlims[self.qpose_tag], ylim=[-0.13, -0.08], zlim=[0.78],
        rotate_rand=False, qpos=qposes[self.qpose_tag], modelname="092_teapot",
        convex=True, rotate_lim=(0, 0, 0.4), model_id=self.model_id,
    )
    self.delay(4)
    self.add_prohibit_area(self.kettle, padding=0.15)
    self.left_target_pose = [-0.25, -0.12, 0.95, 0, 1, 0, 0]
    self.right_target_pose = [0.25, -0.12, 0.95, 0, 1, 0, 0]

task_class.load_actors = patched_load
task = task_class()

# 测试一批 model_ids
test_ids = list(range(60))
seeds_per_model = 3

print(f"测试 {len(test_ids)} 个模型, 每个 {seeds_per_model} seed")
print(f"{'mid':>4} | {'stable':>6} | {'plan_ok':>7} | {'success':>7} | detail")
print("-" * 60)

good_models = []
for mid in test_ids:
    forced_id[0] = mid
    stable = 0
    plan_ok = 0
    success = 0
    for seed in range(seeds_per_model):
        try:
            task.setup_demo(now_ep_num=0, seed=seed, **args)
            stable += 1
            task.play_once()
            if task.plan_success:
                plan_ok += 1
                fp = task.kettle.get_functional_point(0)
                qt = task.qpose_tag
                ok = task.check_success()
                if not ok and seed == 0:
                    print(f"  DEBUG mid={mid} qt={qt} fp=[{fp[0]:.3f},{fp[1]:.3f},{fp[2]:.3f}] ok={ok}")
            if task.plan_success and task.check_success():
                success += 1
        except UnStableError:
            pass
        except Exception as e:
            pass
    detail = "GOOD" if success > 0 else ("plan_fail" if plan_ok == 0 and stable > 0 else "unstable" if stable == 0 else "task_fail")
    print(f"{mid:4d} | {stable:>6} | {plan_ok:>7} | {success:>7} | {detail}")
    if success > 0:
        good_models.append(mid)

print(f"\n可用模型 ({len(good_models)}): {good_models}")
