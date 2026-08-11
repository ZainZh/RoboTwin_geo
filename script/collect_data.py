from data_path_utils import resolve_data_paths
import sys

sys.path.append("./")

import sapien.core as sapien
from sapien.render import clear_cache
from collections import OrderedDict
import pdb
from envs import *
import yaml
import importlib
import json
import traceback
import os
import subprocess
import time
import fcntl
from argparse import ArgumentParser

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No such task")
    return env_instance


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def main(
    task_name=None,
    task_config=None,
    episode_num_override=None,
    episode_start_override=None,
    episode_stop_override=None,
    generate_instructions_override=None,
):

    task = class_decorator(task_name)
    config_path = f"./task_config/{task_config}.yml"

    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    if episode_num_override is not None:
        if int(episode_num_override) <= 0:
            raise ValueError("episode_num_override must be positive")
        args["episode_num"] = int(episode_num_override)
    if int(args.get("initial_seed", 0)) < 0:
        raise ValueError("initial_seed must be non-negative")
    if episode_start_override is not None:
        args["episode_start"] = int(episode_start_override)
    if episode_stop_override is not None:
        args["episode_stop"] = int(episode_stop_override)
    start = int(args.get("episode_start", 0))
    stop = int(args.get("episode_stop", args["episode_num"]))
    if not 0 <= start < stop <= int(args["episode_num"]):
        raise ValueError(
            f"episode range must satisfy 0 <= start < stop <= {args['episode_num']}, "
            f"got [{start}, {stop})"
        )
    if generate_instructions_override is not None:
        args["generate_instructions"] = bool(generate_instructions_override)

    args['task_name'] = task_name

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "missing embodiment files"
        return robot_file

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "number of embodiment config parameters should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    # show config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    args["embodiment_name"] = embodiment_name
    args['task_config'] = task_config
    data_paths = resolve_data_paths()
    # Description generation runs as a child process and resolves storage from
    # the environment.  Keep it on the same raw-data root as collection.
    os.environ["ROBOTWIN_RAW_DATA_ROOT"] = str(data_paths.raw_data_root)
    args["save_path"] = os.path.join(
        str(data_paths.raw_data_root), str(args["task_name"]), args["task_config"]
    )
    print(f"Data storage mode: {data_paths.mode}, raw root: {data_paths.raw_data_root}")
    run(task, args)


def run(TASK_ENV, args):
    epid = int(args.get("initial_seed", 0))
    suc_num, fail_num, seed_list = 0, 0, []

    print(f"Task Name: \033[34m{args['task_name']}\033[0m")

    # =========== Collect Seed ===========
    os.makedirs(args["save_path"], exist_ok=True)

    if not args["use_seed"]:
        print("\033[93m" + "[Start Seed and Pre Motion Data Collection]" + "\033[0m")
        args["need_plan"] = True

        if os.path.exists(os.path.join(args["save_path"], "seed.txt")):
            with open(os.path.join(args["save_path"], "seed.txt"), "r") as file:
                seed_list = file.read().split()
                if len(seed_list) != 0:
                    seed_list = [int(i) for i in seed_list]
                    suc_num = len(seed_list)
                    epid = max(epid, max(seed_list) + 1)
            print(f"Exist seed file, Start from: {epid} / {suc_num}")

        while suc_num < args["episode_num"]:
            try:
                TASK_ENV.setup_demo(now_ep_num=suc_num, seed=epid, **args)
                TASK_ENV.play_once()

                if TASK_ENV.plan_success and TASK_ENV.check_success():
                    print(f"simulate data episode {suc_num} success! (seed = {epid})")
                    seed_list.append(epid)
                    TASK_ENV.save_traj_data(suc_num)
                    suc_num += 1
                else:
                    print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                    fail_num += 1

                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
            except UnStableError as e:
                print(" -------------")
                print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                print("Error: ", e)
                print(" -------------")
                fail_num += 1
                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
                time.sleep(0.3)
            except Exception as e:
                # stack_trace = traceback.format_exc()
                print(" -------------")
                print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                print("Error: ", e)
                print(" -------------")
                fail_num += 1
                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
                time.sleep(1)

            epid += 1

            with open(os.path.join(args["save_path"], "seed.txt"), "w") as file:
                for sed in seed_list:
                    file.write("%s " % sed)

        print(f"\nComplete simulation, failed \033[91m{fail_num}\033[0m times / {epid} tries \n")
    else:
        print("\033[93m" + "Use Saved Seeds List".center(30, "-") + "\033[0m")
        with open(os.path.join(args["save_path"], "seed.txt"), "r") as file:
            seed_list = file.read().split()
            seed_list = [int(i) for i in seed_list]

    # =========== Collect Data ===========

    if args["collect_data"]:
        print("\033[93m" + "[Start Data Collection]" + "\033[0m")

        args["need_plan"] = False
        args["render_freq"] = 0
        args["save_data"] = True

        clear_cache_freq = args["clear_cache_freq"]

        st_idx = int(args.get("episode_start", 0))
        stop_idx = int(args.get("episode_stop", args["episode_num"]))

        def exist_hdf5(idx):
            file_path = os.path.join(args["save_path"], 'data', f'episode{idx}.hdf5')
            return os.path.exists(file_path)

        while exist_hdf5(st_idx):
            st_idx += 1

        for episode_idx in range(st_idx, stop_idx):
            print(f"\033[34mTask name: {args['task_name']}\033[0m")

            TASK_ENV.setup_demo(now_ep_num=episode_idx, seed=seed_list[episode_idx], **args)

            traj_data = TASK_ENV.load_tran_data(episode_idx)
            args["left_joint_path"] = traj_data["left_joint_path"]
            args["right_joint_path"] = traj_data["right_joint_path"]
            TASK_ENV.set_path_lst(args)
            if bool(args.get("record_dense_control", False)):
                # Exclude initialization/homing commands.  The trace begins at
                # the first command after the initial demonstration frame.
                TASK_ENV.reset_dense_control_trace()

            info = TASK_ENV.play_once()
            if not TASK_ENV.check_success():
                TASK_ENV.close_env(clear_cache=False)
                TASK_ENV.remove_data_cache()
                raise RuntimeError(
                    f"replayed episode {episode_idx} did not satisfy task success; "
                    "refusing to commit an HDF5 file"
                )
            info_file_path = os.path.join(args["save_path"], "scene_info.json")
            # Episode-range workers share this manifest.  Hold an OS lock for
            # the short read/modify/write section so parallel collection
            # cannot silently drop another worker's entry.
            with open(info_file_path, "a+", encoding="utf-8") as file:
                fcntl.flock(file.fileno(), fcntl.LOCK_EX)
                try:
                    file.seek(0)
                    content = file.read().strip()
                    info_db = json.loads(content) if content else {}
                    info_db[f"episode_{episode_idx}"] = info
                    file.seek(0)
                    file.truncate()
                    json.dump(info_db, file, ensure_ascii=False, indent=4)
                    file.flush()
                    os.fsync(file.fileno())
                finally:
                    fcntl.flock(file.fileno(), fcntl.LOCK_UN)

            TASK_ENV.close_env(clear_cache=((episode_idx + 1) % clear_cache_freq == 0))
            TASK_ENV.merge_pkl_to_hdf5_video()
            TASK_ENV.remove_data_cache()

        if bool(args.get("generate_instructions", True)):
            subprocess.run(
                [
                    "bash",
                    "gen_episode_instructions.sh",
                    str(args["task_name"]),
                    str(args["task_config"]),
                    str(args["language_num"]),
                ],
                cwd="description",
                check=True,
            )


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    parser = ArgumentParser()
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser.add_argument(
        "--episode_num",
        type=int,
        default=None,
        help="Temporarily cap collection for smoke tests; later runs can resume to the config value.",
    )
    parser.add_argument("--episode_start", type=int, default=None)
    parser.add_argument("--episode_stop", type=int, default=None)
    parser.add_argument(
        "--skip_instructions",
        action="store_true",
        help="Skip description generation for parallel workers; run it once after joining.",
    )
    parser = parser.parse_args()
    task_name = parser.task_name
    task_config = parser.task_config

    main(
        task_name=task_name,
        task_config=task_config,
        episode_num_override=parser.episode_num,
        episode_start_override=parser.episode_start,
        episode_stop_override=parser.episode_stop,
        generate_instructions_override=(False if parser.skip_instructions else None),
    )
