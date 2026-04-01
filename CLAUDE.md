# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RoboTwin 2.0 is a bimanual robotic manipulation benchmark and data generation platform. It provides:
- A simulation environment (SAPIEN 3.0 + PhysX) with 50+ manipulation tasks
- Tools to collect expert demonstration trajectories
- Evaluation infrastructure for 14+ policy baselines (DP, DP3, ACT, RDT, PI0, TinyVLA, DexVLA, etc.)

## Common Commands

**Install dependencies:**
```bash
bash script/_install.sh
```

**Collect demonstration data:**
```bash
bash collect_data.sh {task_name} {task_config} {gpu_id}
# Example:
bash collect_data.sh beat_block_hammer demo_randomized 0
```

**Train a policy (e.g., DP3):**
```bash
cd policy/DP3
bash train.sh {task_name} {task_config} {episode_num} {num_epochs} {batch_size}
# With object point cloud:
bash train_objpc.sh {task_name} {task_config} {ep_num} {num_epochs} {bs} "{A},{B}"
```

**Evaluate a policy:**
```bash
cd policy/DP3
bash eval.sh {task_name} {train_config} {eval_config} {ep_num} {num_epochs} {bs}
```

Task configs are YAML files in `task_config/`. Key configs: `demo_clean.yml`, `demo_randomized.yml`, `demo_clean_3d.yml`.

## Architecture

### Core Task Engine

`envs/_base_task.py` — `Base_Task(gym.Env)` is the central class. All 50+ task implementations inherit from it and override three methods:
- `load_actors()` — place task-specific objects into the SAPIEN scene
- `play_once()` — execute the expert demonstration trajectory
- `check_success()` — return whether the task has been completed

Tasks are loaded dynamically by name:
```python
envs_module = importlib.import_module(f"envs.{task_name}")
env_class = getattr(envs_module, task_name)
```

### Robot Control

`envs/robot/robot.py` — manages dual-arm bimanual robots. Integrates two motion planners:
- **mplib** (RRT-based) — default planner
- **CuRobo** — NVIDIA GPU-accelerated planner (optional)

`envs/robot/ik.py` — inverse kinematics solver.

Arms are addressed with `ArmTag("left")` / `ArmTag("right")`. Embodiments (robot URDFs) are selected via config and mapped through `task_config/_embodiment_config.yml`.

### Data Collection Pipeline

`script/collect_data.py` drives the collection loop. For each episode:
1. Loads task class and YAML config
2. Initializes scene (robots, cameras, objects)
3. Calls `play_once()` on the task
4. Saves multi-modal data (RGB, depth, point clouds, joint states, end-effector poses) to HDF5/zarr under `data/{task_name}/{task_config}/`

### Configuration System

All simulation parameters live in `task_config/*.yml`:
- Domain randomization knobs (lighting, backgrounds, clutter, table height)
- Camera sensor type and which cameras to record
- Data modalities to save (rgb, depth, pointcloud, qpos, endpose, segmentation)
- Episode count, save path, embodiment selection

`envs/_GLOBAL_CONFIGS.py` defines global constants: asset paths, grasp direction quaternions, and `ROTATE_NUM`.

### Policy Interface

Each policy in `policy/{PolicyName}/` implements a `deploy_policy.py` with a standardized observation/action interface. `script/eval_policy.py` loads any policy and runs rollouts. A `policy/Your_Policy/` template exists for adding new policies.

### Asset Organization

- `assets/embodiments/` — robot URDF files (aloha-agilex, piper, franka-panda, ARX-X5, ur5-wsg)
- `assets/objects/` — 127+ 3D object models (YCB, Objaverse) with convex/non-convex collision meshes
- `assets/background_texture/` — scene textures for domain randomization

## Adding a New Task

1. Create `envs/{task_name}.py` with a class named `{task_name}` inheriting from `Base_Task`
2. Implement `load_actors()`, `play_once()`, and `check_success()`
3. Add a task config entry in `task_config/` or reuse an existing config
4. Add evaluation step limit in `task_config/_eval_step_limit.yml`