"""
用指定的 seed 列表采集数据。
用法:
    python script/collect_data_with_seeds.py adjust_kettle demo_clean data/adjust_kettle/plannable_seeds.txt 50
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
    parser.add_argument("task_name")
    parser.add_argument("task_config")
    parser.add_argument("seeds_file")
    parser.add_argument("num", type=int, help="how many seeds from the file to use (first N)")
    args_in = parser.parse_args()

    with open(args_in.seeds_file) as f:
        seeds = [int(line.strip()) for line in f if line.strip()]
    seeds = seeds[:args_in.num]
    print(f"使用前 {len(seeds)} 个 seed: {seeds[:10]}...")

    with open(f"task_config/{args_in.task_config}.yml") as f:
        args = yaml.safe_load(f)
    args['task_name'] = args_in.task_name
    args['need_plan'] = True
    args['data_type']['third_view'] = True
    args['episode_num'] = len(seeds)

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
    args["save_path"] = f"./data/{args_in.task_name}/{args_in.task_config}"
    args["task_config"] = args_in.task_config

    envs_module = importlib.import_module(f"envs.{args_in.task_name}")
    task_class = getattr(envs_module, args_in.task_name)
    task = task_class()

    for ep_idx, seed in enumerate(seeds):
        try:
            task.setup_demo(now_ep_num=ep_idx, seed=seed, **args)
            task.play_once()
            if task.plan_success and task.check_success():
                # 保存数据
                if hasattr(task, 'merge_pkl_to_hdf5_video'):
                    task.merge_pkl_to_hdf5_video()
                    if hasattr(task, 'remove_data_cache'):
                        task.remove_data_cache()
                print(f"[{ep_idx+1}/{len(seeds)}] seed={seed} SAVED", flush=True)
            else:
                print(f"[{ep_idx+1}/{len(seeds)}] seed={seed} failed (plan or check failed!)", flush=True)
        except Exception as e:
            print(f"[{ep_idx+1}/{len(seeds)}] seed={seed} error: {e}", flush=True)

if __name__ == "__main__":
    main()
