"""
测试所有 092_teapot 模型的稳定性，输出可用 model_id 列表。

用法:
    cd ~/github/RoboTwin_geo
    conda activate RoboTwin
    python script/test_teapot_stability.py --seeds 30

完成后会打印可用的 model_id 列表，可直接复制到 envs/adjust_kettle.py 中。
"""

import sys
import os
sys.path.append(".")
import argparse
import numpy as np
import yaml
import importlib
from envs.utils.create_actor import UnStableError
from envs._GLOBAL_CONFIGS import CONFIGS_PATH
from pathlib import Path


def get_available_model_ids():
    """扫描 092_teapot 目录，返回所有有效的 model_id"""
    asset_dir = Path("assets/objects/092_teapot")
    ids = []
    for f in sorted(asset_dir.glob("model_data*.json")):
        mid = int(f.stem.replace("model_data", ""))
        collision = asset_dir / "collision" / f"base{mid}.glb"
        visual = asset_dir / "visual" / f"base{mid}.glb"
        if collision.exists() and visual.exists():
            ids.append(mid)
    return ids


def setup_args():
    """构建任务运行参数"""
    task_name = "adjust_kettle"
    with open("task_config/demo_clean.yml") as f:
        args = yaml.safe_load(f)
    args["task_name"] = task_name
    args["need_plan"] = True
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
    args["render_freq"] = 0  # 禁用渲染，只做物理稳定性测试
    return args


def test_model_ids(model_ids, num_seeds=30):
    """测试每个 model_id 在多个 seed 下的稳定性"""
    args = setup_args()

    # 加载任务类并 patch: 必须在 rand_create_actor 之前设置 model_id
    envs_module = importlib.import_module("envs.adjust_kettle")
    task_class = getattr(envs_module, "adjust_kettle")
    from envs.utils import rand_create_actor

    forced_id = [None]

    def patched_load(self):
        self.qpose_tag = np.random.randint(0, 2)
        qposes = [[0.707, 0.707, 0, 0], [0.707, 0.707, 0, 0]]
        xlims = [[-0.12, -0.08], [0.08, 0.12]]
        self.model_id = forced_id[0]
        self.kettle = rand_create_actor(
            self,
            xlim=xlims[self.qpose_tag],
            ylim=[-0.13, -0.08],
            zlim=[0.78],
            rotate_rand=False,
            qpos=qposes[self.qpose_tag],
            modelname="092_teapot",
            convex=True,
            rotate_lim=(0, 0, 0.4),
            model_id=self.model_id,
        )
        self.delay(4)
        self.add_prohibit_area(self.kettle, padding=0.15)
        self.left_target_pose = [-0.25, -0.12, 0.95, 0, 1, 0, 0]
        self.right_target_pose = [0.25, -0.12, 0.95, 0, 1, 0, 0]

    task_class.load_actors = patched_load
    task = task_class()

    results = {}
    for mid in model_ids:
        forced_id[0] = mid
        stable = 0
        total = 0
        for seed in range(num_seeds):
            total += 1
            try:
                task.setup_demo(now_ep_num=0, seed=seed, **args)
                stable += 1
            except UnStableError:
                pass
            except Exception as e:
                print(f"  model {mid} seed {seed}: ERROR - {e}")
        pct = stable / total * 100 if total > 0 else 0
        results[mid] = {"stable": stable, "total": total, "pct": pct}
        status = "PASS" if pct >= 80 else "FAIL"
        print(f"  model_id {mid:2d}: {stable:2d}/{total} stable ({pct:5.1f}%)  [{status}]")

    return results


def main():
    parser = argparse.ArgumentParser(description="092_teapot 稳定性批量测试")
    parser.add_argument("--seeds", type=int, default=30, help="每个模型测试多少个 seed")
    parser.add_argument("--threshold", type=float, default=80.0, help="稳定率阈值 (%%)")
    args = parser.parse_args()

    model_ids = get_available_model_ids()
    print(f"找到 {len(model_ids)} 个模型: {model_ids}")
    print(f"每个模型测试 {args.seeds} 个 seed, 阈值 {args.threshold}%")
    print()

    results = test_model_ids(model_ids, args.seeds)

    # 筛选
    passed = [mid for mid, r in results.items() if r["pct"] >= args.threshold]
    failed = [mid for mid, r in results.items() if r["pct"] < args.threshold]

    print(f"\n{'='*50}")
    print(f"通过 (>={args.threshold}%): {len(passed)} 个  {passed}")
    print(f"未通过: {len(failed)} 个  {failed}")

    if passed:
        id_list = str(passed)
        print(f"\n可直接用于 adjust_kettle.py:")
        print(f"  self.model_id = np.random.choice({id_list})")


if __name__ == "__main__":
    main()
