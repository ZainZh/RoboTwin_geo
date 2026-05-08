"""
为 adjust_kettle 的每个选定模型各生成一个 episode 视频。
"""
import sys, os
sys.path.append(".")
import numpy as np
import yaml
import importlib
from envs.utils.create_actor import UnStableError
from envs._GLOBAL_CONFIGS import CONFIGS_PATH
from envs.utils import rand_create_actor

task_name = "adjust_kettle"
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
args["save_path"] = f"./data/{task_name}/all_models"
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

model_ids = [2, 12, 13, 21, 33, 49, 56, 8, 20]

for mid in model_ids:
    forced_id[0] = mid
    for seed in range(10):
        try:
            args["save_path"] = f"./data/{task_name}/all_models/model{mid}"
            task.setup_demo(now_ep_num=0, seed=seed, **args)
            task.play_once()
            if task.plan_success and task.check_success():
                print(f"model {mid}: seed={seed} SUCCESS", flush=True)
                break
            else:
                print(f"model {mid}: seed={seed} plan={task.plan_success} check={task.check_success()}", flush=True)
        except UnStableError:
            print(f"model {mid}: seed={seed} unstable", flush=True)
        except Exception as e:
            print(f"model {mid}: seed={seed} error: {e}", flush=True)
    else:
        print(f"model {mid}: all seeds failed", flush=True)

print("\nDone!")
