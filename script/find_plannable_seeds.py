"""
扫描 seed，找出 N 个能 plan_success 的 seed，保存到 seeds.txt
"""
import sys, os
sys.path.append(".")
import numpy as np
import yaml
import importlib
import argparse
from envs.utils.create_actor import UnStableError
from envs._GLOBAL_CONFIGS import CONFIGS_PATH

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=100, help="target plannable seeds")
    parser.add_argument("--max_seed", type=int, default=2000, help="max seed to try")
    parser.add_argument("--output", type=str, default="data/adjust_kettle/plannable_seeds.txt")
    args_in = parser.parse_args()

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
    args["save_path"] = f"./data/{task_name}/_seed_search"
    args["task_config"] = "demo_clean"

    envs_module = importlib.import_module(f"envs.{task_name}")
    task_class = getattr(envs_module, task_name)
    task = task_class()

    plannable = []
    os.makedirs(os.path.dirname(args_in.output), exist_ok=True)

    for seed in range(args_in.max_seed):
        try:
            task.setup_demo(now_ep_num=0, seed=seed, **args)
            task.play_once()
            if task.plan_success:
                plannable.append(seed)
                with open(args_in.output, "w") as f:
                    f.write("\n".join(str(s) for s in plannable))
                print(f"[{len(plannable)}/{args_in.target}] seed={seed} OK", flush=True)
                if len(plannable) >= args_in.target:
                    break
        except UnStableError:
            pass
        except Exception:
            pass

    print(f"\nFound {len(plannable)} plannable seeds out of {seed+1} tried")
    print(f"Saved to {args_in.output}")

if __name__ == "__main__":
    main()
