"""
找出可靠的 seed：连续 N 次都 plan 成功才保留
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
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--max_seed", type=int, default=3000)
    parser.add_argument("--n_trials", type=int, default=2, help="每个seed连续成功次数")
    parser.add_argument("--output", type=str, default="data/adjust_kettle/reliable_seeds.txt")
    args_in = parser.parse_args()

    task_name = "adjust_kettle"
    with open("task_config/demo_clean.yml") as f:
        args = yaml.safe_load(f)
    args['task_name'] = task_name
    args['need_plan'] = True
    args['render_freq'] = 0
    args['use_seed'] = False
    args['data_type']['third_view'] = False

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

    reliable = []
    os.makedirs(os.path.dirname(args_in.output), exist_ok=True)

    for seed in range(args_in.max_seed):
        all_ok = True
        for trial in range(args_in.n_trials):
            try:
                task.setup_demo(now_ep_num=0, seed=seed, **args)
                task.play_once()
                if not task.plan_success:
                    all_ok = False
                    break
            except UnStableError:
                all_ok = False
                break
            except Exception:
                all_ok = False
                break
        if all_ok:
            reliable.append(seed)
            with open(args_in.output, "w") as f:
                f.write("\n".join(str(s) for s in reliable))
            print(f"[{len(reliable)}/{args_in.target}] seed={seed} OK ({args_in.n_trials} trials)", flush=True)
            if len(reliable) >= args_in.target:
                break

    print(f"\nFound {len(reliable)} reliable seeds out of {seed+1} tried")

if __name__ == "__main__":
    main()
