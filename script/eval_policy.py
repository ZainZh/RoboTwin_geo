import sys
import os
import subprocess
import json

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
from collections import deque
import traceback

import yaml
from datetime import datetime
import importlib
import argparse
import pdb

from generate_episode_instructions import *

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def parse_bool(value, *, default=False):
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"expected a boolean value, got {value!r}")


def load_evaluation_seed_file(path, *, test_num):
    if path in {None, ""}:
        return None
    seed_path = Path(path).expanduser().resolve()
    if not seed_path.is_file():
        raise FileNotFoundError(f"evaluation seed file does not exist: {seed_path}")
    payload = json.loads(seed_path.read_text(encoding="utf-8"))
    values = payload.get("accepted_seeds") if isinstance(payload, dict) else payload
    if not isinstance(values, list):
        raise ValueError(
            "evaluation seed file must be a JSON list or contain an accepted_seeds list"
        )
    if len(values) < int(test_num):
        raise ValueError(
            f"evaluation seed file has {len(values)} seeds, fewer than requested {test_num}"
        )
    seeds = []
    for value in values[: int(test_num)]:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"evaluation seeds must be integers, got {value!r}")
        seeds.append(int(value))
    if len(set(seeds)) != len(seeds):
        raise ValueError("evaluation seed list contains duplicates")
    return seeds


def git_snapshot(repo_root: Path) -> dict:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        return {"head": head, "dirty": bool(status), "status_short": status}
    except (OSError, subprocess.CalledProcessError) as error:
        return {"error": str(error)}


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e


def should_run_expert_check(args):
    custom_hammer_eval = args.get("custom_hammer_eval") or {}
    custom_mug_eval = args.get("custom_mug_eval") or {}
    return not (bool(custom_hammer_eval.get("enabled")) or bool(custom_mug_eval.get("enabled")))


def build_instruction_episode_info(task_name, task_env, episode_info=None):
    if episode_info is not None:
        return episode_info
    existing_info = getattr(task_env, "info", None)
    if (
        isinstance(existing_info, dict)
        and isinstance(existing_info.get("info"), dict)
        and len(existing_info["info"]) > 0
    ):
        return existing_info
    if task_name == "beat_block_hammer":
        hammer_asset_config = getattr(task_env, "hammer_asset_config", None)
        block = getattr(task_env, "block", None)
        if hammer_asset_config is None or block is None:
            raise ValueError("beat_block_hammer requires hammer_asset_config and block to build instruction info")
        block_pose = block.get_functional_point(0, "pose").p
        arm_tag = "left" if block_pose[0] < 0 else "right"
        return {"info": {"{A}": hammer_asset_config["info_asset_path"], "{a}": arm_tag}}
    if task_name == "hanging_mug":
        mug_asset_config = getattr(task_env, "mug_asset_config", None)
        if mug_asset_config is None:
            raise ValueError("hanging_mug requires mug_asset_config to build instruction info")
        return {"info": {"{A}": mug_asset_config["info_asset_path"], "{B}": "040_rack/base0"}}
    if task_name in {"place_shoe_rotating_block", "place_shoe_geometry_marker"}:
        arm_tag = getattr(task_env, "shoe_arm_name", None)
        if arm_tag not in {"left", "right"}:
            shoe = getattr(task_env, "shoe", None)
            if shoe is None:
                raise ValueError(f"{task_name} has no shoe actor after setup")
            arm_tag = "left" if float(shoe.get_pose().p[0]) < 0.0 else "right"
        return {
            "info": {
                "{A}": "shoe",
                "{B}": "target block",
                "{a}": str(arm_tag),
            }
        }
    raise ValueError(f"cannot build instruction info without expert_check for task {task_name}")


def resolve_episode_instruction(
        task_name,
        episode_info_list,
        instruction_type,
        test_num,
        policy_name,
        description_generator=None,
        random_choice_fn=None):
    description_generator = description_generator or generate_episode_descriptions

    try:
        results = description_generator(task_name, episode_info_list, test_num)
        if len(results) == 0:
            raise IndexError(f"No valid instructions generated for task {task_name}")
        instructions = results[0][instruction_type]
        if len(instructions) == 0:
            raise IndexError(f"No {instruction_type} instructions generated for task {task_name}")
        random_choice_fn = random_choice_fn or np.random.choice
        return random_choice_fn(instructions)
    except (FileNotFoundError, IndexError, KeyError):
        if policy_name == "DP3":
            fallback = task_name.replace("_", " ")
            print(
                f"[eval_policy] DP3 instruction fallback: task={task_name}, "
                f"instruction_type={instruction_type}, instruction='{fallback}'"
            )
            return fallback
        raise


def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def main(usr_args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    # checkpoint_num = usr_args['checkpoint_num']
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    save_dir = None
    video_save_dir = None
    video_size = None

    get_model = eval_function_decorator(policy_name, "get_model")

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    if "eval_video_log" in usr_args:
        args["eval_video_log"] = parse_bool(usr_args["eval_video_log"])

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting
    args["policy_start_phase"] = str(usr_args.get("policy_start_phase", "full_task"))
    # Fixed replay seeds may come from an earlier expert-screening pass.  Carry
    # this flag into the task arguments so eval_policy() can avoid rerunning the
    # expert (which is both expensive and can be nondeterministic under parallel
    # simulator launches).
    args["skip_expert_check_on_replay"] = parse_bool(
        usr_args.get("skip_expert_check_on_replay", False)
    )
    args["evaluation_start_episode_index"] = int(
        usr_args.get("evaluation_start_episode_index", 0)
    )
    args["seed_discovery_only"] = parse_bool(
        usr_args.get("seed_discovery_only", False)
    )

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

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
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    token_ablation = str(usr_args.get("se3_relation_token_ablation", "none"))
    evaluation_variant = "standard" if token_ablation == "none" else f"ablation-{token_ablation}"
    save_dir = Path(
        f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/"
        f"{evaluation_variant}/{current_time}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    args["eval_result_dir"] = str(save_dir)
    args["evaluation_variant"] = evaluation_variant

    if args["eval_video_log"]:
        video_save_dir = save_dir
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    # output camera config
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

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args["seed"]

    st_seed = 100000 * (1 + seed)
    suc_nums = []
    test_num = int(usr_args.get("test_num", 100))
    evaluation_seed_file = str(usr_args.get("evaluation_seed_file", "") or "")
    evaluation_seeds = load_evaluation_seed_file(
        evaluation_seed_file,
        test_num=test_num,
    )
    args["evaluation_seeds"] = evaluation_seeds
    topk = 1

    evaluation_manifest = {
        "schema_version": 1,
        "task_name": str(task_name),
        "task_config": str(task_config),
        "policy_name": str(policy_name),
        "checkpoint_setting": str(ckpt_setting),
        "checkpoint_num": int(usr_args.get("checkpoint_num", 0)),
        "expert_data_num": int(usr_args["expert_data_num"]),
        "training_seed": int(seed),
        "evaluation_start_seed": int(evaluation_seeds[0] if evaluation_seeds else st_seed),
        "evaluation_start_episode_index": int(
            usr_args.get("evaluation_start_episode_index", 0)
        ),
        "evaluation_seed_file": (
            str(Path(evaluation_seed_file).expanduser().resolve())
            if evaluation_seed_file
            else None
        ),
        # A seed-discovery run performs an expert rollout before the policy
        # rollout.  Even with the same integer seed that is not the same
        # initialization protocol as a one-pass fixed replay, so paired
        # comparisons must record and match this distinction.
        "skip_expert_check_on_replay": parse_bool(
            usr_args.get("skip_expert_check_on_replay", False)
        ),
        "fixed_seed_replay": bool(
            evaluation_seeds is not None
            and parse_bool(usr_args.get("skip_expert_check_on_replay", False))
        ),
        "seed_discovery_only": bool(args["seed_discovery_only"]),
        "evaluation_seeds": evaluation_seeds,
        "requested_episodes": int(test_num),
        "instruction_type": str(instruction_type),
        "evaluation_variant": str(evaluation_variant),
        "policy_start_phase": str(args["policy_start_phase"]),
        "eval_video_log": bool(args["eval_video_log"]),
        "policy_architecture": {
            "down_dims_override": usr_args.get("policy_down_dims"),
        },
        # Keep the complete geometry-policy interface auditable.  Previously
        # the episode metrics exposed some thresholds after execution, but the
        # manifest did not record checkpoint identities or disabled gates.  A
        # paper-scale paired run must be reconstructable even when one branch
        # does not execute and therefore emits no branch-specific metric.
        "geometry_policy_config": {
            key: value
            for key, value in usr_args.items()
            if key.startswith("geometry_")
            or key
            in {
                "max_eef_translation_delta_m",
                "max_eef_rotation_delta_rad",
            }
        },
        "se3_relation": {
            "enabled": parse_bool(
                usr_args.get("use_se3_relation_token", False), default=False
            ),
            "route": str(usr_args.get("se3_relation_route", "baseline")),
            "goal_table": str(usr_args.get("se3_relation_goal_table", "") or "") or None,
            "geometry_estimator_spec": str(
                usr_args.get("se3_geometry_estimator_spec", "") or ""
            )
            or None,
            "token_ablation": str(
                usr_args.get("se3_relation_token_ablation", "none")
            ),
        },
        "geometry_marker": args.get("geometry_marker", {}),
        "placement_success": args.get("placement_success", {}),
        "camera": args.get("camera", {}),
        "domain_randomization": args.get("domain_randomization", {}),
        "git": git_snapshot(Path(__file__).resolve().parents[1]),
    }
    (save_dir / "evaluation_manifest.json").write_text(
        json.dumps(evaluation_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Seed discovery needs the task expert but not a policy network or a
    # second simulator rollout.  Keeping this as a first-class protocol saves
    # substantial GPU/simulator time while preserving the same expert-admitted
    # seed criterion used by the subsequent fixed replay.
    model = None if args["seed_discovery_only"] else get_model(usr_args)
    st_seed, suc_num = eval_policy(task_name,
                                   TASK_ENV,
                                   args,
                                   model,
                                   st_seed,
                                   test_num=test_num,
                                   video_size=video_size,
                                   instruction_type=instruction_type)
    suc_nums.append(suc_num)

    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]

    file_path = os.path.join(save_dir, f"_result.txt")
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        # file.write(str(task_reward) + '\n')
        file.write("\n".join(map(str, np.array(suc_nums) / test_num)))

    print(f"Data has been saved to {file_path}")
    # return task_reward


def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                test_num=100,
                video_size=None,
                instruction_type=None):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    evaluation_seeds = args.get("evaluation_seeds")
    skip_expert_check = parse_bool(
        args.get("skip_expert_check_on_replay", False)
    )
    if skip_expert_check and evaluation_seeds is None:
        raise ValueError(
            "skip_expert_check_on_replay requires a fixed evaluation seed file"
        )
    expert_check = should_run_expert_check(args) and not skip_expert_check
    seed_discovery_only = parse_bool(args.get("seed_discovery_only", False))
    if seed_discovery_only and (not expert_check or evaluation_seeds is not None):
        raise ValueError(
            "seed_discovery_only requires automatic seeds and expert screening"
        )
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = int(args.get("evaluation_start_episode_index", 0))
    succ_seed = 0
    suc_test_seed_list = []

    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval")
    reset_func = eval_function_decorator(policy_name, "reset_model")

    now_seed = st_seed
    task_total_reward = 0
    clear_cache_freq = args["clear_cache_freq"]
    consecutive_setup_failures = 0
    max_setup_failures = max(1, int(args.get("max_setup_failures", 25)))

    args["eval_mode"] = True
    eval_result_dir = Path(args["eval_result_dir"])
    episode_log_path = eval_result_dir / "episodes.jsonl"
    accepted_seed_path = eval_result_dir / "accepted_eval_seeds.json"

    def persist_accepted_seeds():
        accepted_seed_path.write_text(
            json.dumps(
                {
                    "start_seed": int(
                        evaluation_seeds[0]
                        if evaluation_seeds is not None
                        else st_seed
                    ),
                    "accepted_seeds": [int(value) for value in suc_test_seed_list],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    while succ_seed < test_num:
        if evaluation_seeds is not None:
            now_seed = int(evaluation_seeds[succ_seed])
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        try:
            TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
            if expert_check:
                episode_info = TASK_ENV.play_once()
            else:
                episode_info = build_instruction_episode_info(task_name, TASK_ENV, episode_info=None)
            TASK_ENV.close_env()
        except UnStableError as e:
            TASK_ENV.close_env()
            if evaluation_seeds is not None:
                raise RuntimeError(
                    f"replayed evaluation seed {now_seed} became unstable"
                ) from e
            consecutive_setup_failures += 1
            if consecutive_setup_failures >= max_setup_failures:
                raise RuntimeError(
                    f"aborting after {consecutive_setup_failures} consecutive "
                    f"environment setup failures; last seed={now_seed}"
                ) from e
            now_seed += 1
            args["render_freq"] = render_freq
            continue
        except Exception as e:
            print(" -------------")
            print("Error: ", e)
            print(" -------------")
            TASK_ENV.close_env()
            if evaluation_seeds is not None:
                raise RuntimeError(
                    f"replayed evaluation seed {now_seed} raised an exception"
                ) from e
            consecutive_setup_failures += 1
            if consecutive_setup_failures >= max_setup_failures:
                raise RuntimeError(
                    f"aborting after {consecutive_setup_failures} consecutive "
                    f"environment setup failures; last seed={now_seed}, "
                    f"last error={e}"
                ) from e
            now_seed += 1
            args["render_freq"] = render_freq
            print("error occurs !")
            continue

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            consecutive_setup_failures = 0
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        else:
            if evaluation_seeds is not None:
                raise RuntimeError(
                    f"replayed evaluation seed {now_seed} no longer passes the expert check"
                )
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        if seed_discovery_only:
            persist_accepted_seeds()
            now_id += 1
            TASK_ENV.test_num += 1
            print(
                f"accepted expert seed {succ_seed}/{test_num}: {now_seed}",
                flush=True,
            )
            now_seed += 1
            continue

        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        if args["policy_start_phase"] == "placement":
            prepare_placement = getattr(TASK_ENV, "prepare_policy_placement_phase", None)
            if prepare_placement is None:
                raise RuntimeError(
                    f"task {task_name} does not implement prepare_policy_placement_phase()"
                )
            prepare_placement()

        episode_info_list = [episode_info["info"]]
        instruction = resolve_episode_instruction(
            args["task_name"],
            episode_info_list,
            instruction_type,
            test_num,
            args["policy_name"],
        )
        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction

        if TASK_ENV.eval_video_path is not None:
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        succ = False
        reset_func(model)
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            observation = TASK_ENV.get_obs()
            eval_func(TASK_ENV, model, observation)
            if TASK_ENV.eval_success:
                succ = True
                break
        # task_total_reward += TASK_ENV.episode_score
        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()

        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        metrics_fn = getattr(TASK_ENV, "get_evaluation_metrics", None)
        metrics = metrics_fn() if callable(metrics_fn) else {}
        policy_metrics_fn = getattr(model, "get_evaluation_metrics", None)
        if callable(policy_metrics_fn):
            policy_metrics = policy_metrics_fn()
            if isinstance(policy_metrics, dict):
                overlap = set(metrics) & set(policy_metrics)
                if overlap:
                    raise RuntimeError(
                        f"policy evaluation metrics collide with task metrics: {sorted(overlap)}"
                    )
                metrics.update(policy_metrics)
        episode_record = {
            "episode_index": int(now_id),
            "seed": int(now_seed),
            "success": bool(succ),
            "task_name": str(task_name),
            "task_config": str(args["task_config"]),
            "policy_name": str(args["policy_name"]),
            "checkpoint_setting": str(args["ckpt_setting"]),
            "evaluation_variant": str(args["evaluation_variant"]),
            "metrics": metrics,
        }
        with episode_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(episode_record, sort_keys=True) + "\n")
        persist_accepted_seeds()

        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        )
        # TASK_ENV._take_picture()
        if evaluation_seeds is None:
            now_seed += 1

    return now_seed, TASK_ENV.suc


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    return config


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    usr_args = parse_args_and_config()

    main(usr_args)
