# Progress Log

## Session: 2026-04-09

### Phase 1: Requirements & Discovery
- **Status:** complete
- **Started:** 2026-04-09 17:36 HKT
- Actions taken:
  - Confirmed that `planning-with-files` was explicitly requested.
  - Checked the repository root for existing planning files.
  - Read the skill and template files to match the expected workflow.
  - Initialized `task_plan.md`, `findings.md`, and `progress.md` in the project root.
  - Captured the concrete repo task: validate `policy/DP3/train_objpc_sam3.sh` and estimate per-frame SAM3 object-PCD fusion cost.
  - Inspected `policy/DP3/train_objpc_sam3.sh` and mapped the first-level dependency chain to preprocessing, training config, and SAM3 utility files.
  - Compared the SAM3 and non-SAM3 DP3 configs and identified that the `sam3` training script still points at `robot_dp3_objpc.yaml`.
  - Confirmed the relevant raw dataset and SAM3 model file exist locally, enabling targeted runtime validation.
  - Verified that the DP3 dataset loader resolves `../../../data/...` relative to the dataset module, which correctly targets `policy/DP3/data`.
- Files created/modified:
  - `task_plan.md` (created)
  - `findings.md` (created)
  - `progress.md` (created)
  - `task_plan.md` (updated)
  - `findings.md` (updated)
  - `progress.md` (updated)

### Phase 2: Planning & Structure
- **Status:** complete
- Actions taken:
  - Chose `policy/DP3/scripts/benchmark_objpc_sam3.py` as the primary runtime probe for per-frame fusion cost.
  - Defined validation strategy as shell syntax check, reproduce cwd failure, then exercise the extraction path directly.
- Files created/modified:
  - `task_plan.md` (updated)
  - `findings.md` (updated)
  - `progress.md` (updated)

### Phase 3: Implementation
- **Status:** complete
- Actions taken:
  - Reproduced the repo-root failure of `policy/DP3/train_objpc_sam3.sh`.
  - Fixed the missing `Path` import in `policy/DP3/scripts/sam3_pointcloud_utils.py`.
  - Installed `ftfy` into the active `RoboTwin` environment to satisfy the local CLIP dependency required by SAM3.
  - Added a compatibility shim so the local CLIP `SimpleTokenizer` is callable by Ultralytics SAM3.
  - Normalized cached SAM3 boxes to XYXY before reusing them as bbox prompts.
  - Made `policy/DP3/train_objpc_sam3.sh` and `policy/DP3/process_data_objpc_sam3.sh` self-locating and fail-fast with `set -euo pipefail`.
  - Updated the SAM3 training script to use `robot_dp3_objpc_sam3.yaml`.
- Files created/modified:
  - `policy/DP3/scripts/sam3_pointcloud_utils.py` (modified)
  - `policy/DP3/train_objpc_sam3.sh` (modified)
  - `policy/DP3/process_data_objpc_sam3.sh` (modified)

### Phase 4: Testing & Verification
- **Status:** complete
- Actions taken:
  - Re-ran the SAM3 benchmark until it completed successfully on 5 frames.
  - Verified shell syntax with `bash -n`.
  - Verified Python files compile with `python -m py_compile`.
  - Checked CUDA availability and confirmed this environment is CPU-only.
  - Verified that `policy/DP3/scripts/visualize_objpc_zarr.py` runs in `--no_show` mode with the current environment.
  - Confirmed older `objpc` zarrs may only expose merged `point_cloud`, while the SAM3 preprocessing path is configured to save per-placeholder clouds too.
  - Traced the repeated `imgsz=[640]` warning to Ultralytics `check_imgsz(...)` and confirmed the local SAM3 predictor is constructed without an explicit `imgsz` override.
  - Confirmed that a mid-run SAM3 zarr only contained group metadata, which explains why interrupted preprocessing could later trigger `KeyError('state')`.
- Files created/modified:
  - `progress.md` (updated)

### Phase 5: Delivery
- **Status:** in_progress
- Actions taken:
  - Patched `train_objpc_sam3.sh` to rebuild incomplete zarr outputs instead of trusting directory existence alone.
  - Patched `process_data_objpc_sam3.py` so the old all-at-end write pattern no longer controls visibility of output data.
  - Added `_self_` to `robot_dp3_objpc_sam3.yaml` to remove the Hydra composition-order warning.
  - Rechecked the SAM3 output directory: final `.zarr` is still absent, and the `.zarr.tmp` contains only empty `data/` and `meta/` groups so far.
  - Added a tested incremental zarr append helper built on DP3's `ReplayBuffer.add_episode(...)`.
  - Reworked `process_data_objpc_sam3.py` to persist one episode at a time into the final `.zarr` and update metadata after each episode.
  - Changed `train_objpc_sam3.sh` so incomplete SAM3 zarrs are resumed instead of deleted.
  - Inspected the installed Ultralytics SAM3 video-tracker implementation to determine whether it supports stable per-sequence tracking and whether it can handle online updates.
  - Confirmed that `SAM3VideoSemanticPredictor` preserves tracking state only within a single loaded finite video sequence, exposes `add_prompt(...)` for frame-level prompt updates, and is not a clean drop-in API for endless live streams.
  - Traced the eval runtime path and confirmed that `policy/DP3/eval_objpc_sam3.sh` already enables online SAM3 segmentation in `deploy_policy.py`.
  - Confirmed that eval currently uses `SAM3SemanticPredictor` plus previous-bbox reuse through `SAM3ProjectiveTracker`, not the Ultralytics video-predictor class.
  - Verified from the actual `hanging_mug/demo_clean_3d_object_pc` episode files that `head_camera` and `front_camera` RGB frames decode to `240x320`.
  - Traced the low resolution back to `task_config/_camera_config.yml` (`D435 = 320x240`, `Large_D435 = 640x480`) and confirmed `demo_clean_3d_object_pc.yml` currently uses `D435`.
- Files created/modified:
  - `policy/DP3/train_objpc_sam3.sh` (modified)
  - `policy/DP3/scripts/process_data_objpc_sam3.py` (modified)
  - `policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/robot_dp3_objpc_sam3.yaml` (modified)
  - `findings.md` (updated)
  - `progress.md` (updated)
  - `policy/DP3/scripts/incremental_objpc_zarr.py` (created)
  - `policy/DP3/scripts/test_incremental_objpc_zarr.py` (created)

## Test Results
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| Planning file initialization | Create project-local planning scaffold | Three planning files exist and reflect current context | Initialized `task_plan.md`, `findings.md`, and `progress.md` | pass |
| Shell syntax | `bash -n policy/DP3/train_objpc_sam3.sh` | Valid shell syntax | Exit code 0 | pass |
| Shell syntax | `bash -n policy/DP3/process_data_objpc_sam3.sh` | Valid shell syntax | Exit code 0 | pass |
| Python compile | `python -m py_compile policy/DP3/scripts/sam3_pointcloud_utils.py policy/DP3/scripts/benchmark_objpc_sam3.py policy/DP3/scripts/process_data_objpc_sam3.py` | Files compile cleanly | Exit code 0 | pass |
| Reproduce original cwd bug | `bash policy/DP3/train_objpc_sam3.sh ...` from repo root before patch | Immediate failure proves cwd sensitivity | Failed to find `process_data_objpc_sam3.sh` and `3D-Diffusion-Policy` | pass |
| SAM3 fusion benchmark | `cd policy/DP3 && python scripts/benchmark_objpc_sam3.py hanging_mug demo_clean_3d_object_pc --episode_idx 0 --num_frames 5 ...` | Successful extraction timing report | Completed with CPU timings for 5 frames | pass |
| CUDA availability | `python -c "import torch; ..."` | Confirm whether benchmark is CPU or GPU | `cuda_available False`, `device_count 0` | pass |
| Zarr visualizer dry-run | `python policy/DP3/scripts/visualize_objpc_zarr.py policy/DP3/data/hanging_mug-demo_clean_3d_object_pc-50-objpc.zarr --frame_idx 0 --placeholders 'A,B' --show_merged --no_show` | Confirm the visualizer works and report dataset availability | Printed merged `point_cloud`; placeholder arrays missing in this older zarr | pass |
| Shell syntax | `bash -n policy/DP3/train_objpc_sam3.sh` after incomplete-zarr guard | Valid shell syntax | Exit code 0 | pass |
| Python compile | `python -m py_compile policy/DP3/scripts/process_data_objpc_sam3.py` after preprocessing visibility fix | File compiles cleanly | Exit code 0 | pass |
| Incremental zarr unit test | `python policy/DP3/scripts/test_incremental_objpc_zarr.py` | Replay buffer should persist episodes across reopen and reset invalid stores | 2 tests passed | pass |
| Python compile | `python -m py_compile policy/DP3/scripts/process_data_objpc_sam3.py policy/DP3/scripts/incremental_objpc_zarr.py policy/DP3/scripts/sam3_pointcloud_utils.py` | Files compile cleanly | Exit code 0 | pass |

## Error Log
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-04-09 17:36 HKT | None | 0 | N/A |
| 2026-04-09 17:59 HKT | `NameError: Path` in `sam3_pointcloud_utils.py` | 1 | Added missing `Path` import |
| 2026-04-09 18:07 HKT | Missing `ftfy` during SAM3 model load | 1 | Installed `ftfy` |
| 2026-04-09 18:16 HKT | `SimpleTokenizer` not callable | 1 | Added compatibility shim |
| 2026-04-09 18:19 HKT | Bbox prompt shape `[1, 6]` incompatible with Ultralytics | 1 | Trimmed boxes to the first 4 values |

## 5-Question Reboot Check
| Question | Answer |
|----------|--------|
| Where am I? | Phase 5 |
| Where am I going? | Deliver the investigation results and next steps for SAM3-backed training and evaluation |
| What's the goal? | Validate the SAM3 object-PCD training path and measure frame-fusion cost |
| What have I learned? | The SAM3 extraction path now benchmarks successfully on CPU after fixing import, tokenizer, bbox, and cwd issues |
| What have I done? | Investigated blockers, patched the code, installed the missing dependency, and recorded timing evidence |

---
*Update after completing each phase or encountering errors.*

## Session: 2026-04-10

### Phase 1: Requirements & Discovery
- **Status:** in_progress
- Actions taken:
  - Restored planning context from the previous SAM-focused workstream.
  - Captured the new task: replace SAM segmentation with simulator segmentation for fused object-PCD VLA training/eval and create matching bash scripts.
  - Replaced `task_plan.md` with a new task-specific plan for the simulator-segmentation workstream.
  - Confirmed the simulator already exposes `mesh_segmentation`, `actor_segmentation`, and direct `object_pointcloud` observations.
  - Confirmed the current non-SAM `objpc` path already exists, but it is not the same as the requested “sim segmentation replaces SAM” fused multi-camera path.
  - Verified that `mesh_segmentation` comes from `Segmentation[..., 0]` while `actor_segmentation` comes from `Segmentation[..., 1]`.
  - Verified that oracle `object_pointcloud` extraction filters by actor ids from `Segmentation[..., 1]`, which makes `actor_segmentation` the closer match for placeholder-level object masks.
  - Confirmed from `scene_info.json` that collected `*_object_pc` datasets already persist placeholder-to-actor-id mappings, which the new offline actor-segmentation preprocessing can reuse.
  - Confirmed there is no currently collected local dataset with saved `actor_segmentation`, so verification will need synthetic tests plus a recollected smoke dataset.
- Files created/modified:
  - `task_plan.md` (replaced)
  - `findings.md` (updated)
  - `progress.md` (updated)

### Phase 2: Design & Structure
- **Status:** complete
- Actions taken:
  - Chose `actor_segmentation` as the canonical simulator segmentation source for the new fused object-PCD pipeline.
  - Defined a separate `objpc_actorseg` train/eval path so it does not collide with the existing oracle `objpc` flow or the SAM3 flow.
  - Decided to keep the segment-project-fuse structure and not shortcut to oracle `object_pointcloud`.
  - Decided to reuse incremental replay-buffer writes for the new offline preprocessing path.
- Files created/modified:
  - `task_plan.md` (updated)
  - `findings.md` (updated)
  - `progress.md` (updated)

### Phase 3: Implementation
- **Status:** complete
- Actions taken:
  - Added `policy/DP3/scripts/actorseg_pointcloud_utils.py` for multi-camera actor-segmentation projection and fusion in both offline and online modes.
  - Added `policy/DP3/scripts/process_data_objpc_actorseg.py` with resumable incremental zarr writing.
  - Added `policy/DP3/process_data_objpc_actorseg.sh`, `policy/DP3/train_objpc_actorseg.sh`, and `policy/DP3/eval_objpc_actorseg.sh`.
  - Added `policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/robot_dp3_objpc_actorseg.yaml` and `policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/task/demo_task_objpc_actorseg.yaml`.
  - Added `task_config/demo_clean_3d_actorseg.yml` and `task_config/demo_randomized_3d_actorseg.yml` as actor-segmentation collection/eval templates.
  - Extended `policy/DP3/deploy_policy.py` with an online `objpc_actorseg` branch and per-episode actor-id resolution.
  - Updated `envs/_base_task.py` so actor-segmentation collection preserves placeholder target metadata in `scene_info.json` even without oracle object-pointcloud collection.
  - Added `policy/DP3/scripts/test_actorseg_pointcloud_utils.py` as a synthetic unit-style test for the new projection logic.
- Files created/modified:
  - `envs/_base_task.py` (modified)
  - `policy/DP3/deploy_policy.py` (modified)
  - `policy/DP3/scripts/actorseg_pointcloud_utils.py` (created)
  - `policy/DP3/scripts/process_data_objpc_actorseg.py` (created)
  - `policy/DP3/scripts/test_actorseg_pointcloud_utils.py` (created)
  - `policy/DP3/process_data_objpc_actorseg.sh` (created)
  - `policy/DP3/train_objpc_actorseg.sh` (created)
  - `policy/DP3/eval_objpc_actorseg.sh` (created)
  - `policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/robot_dp3_objpc_actorseg.yaml` (created)
  - `policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/task/demo_task_objpc_actorseg.yaml` (created)
  - `task_config/demo_clean_3d_actorseg.yml` (created)
  - `task_config/demo_randomized_3d_actorseg.yml` (created)

### Phase 4: Testing & Verification
- **Status:** in_progress
- Actions taken:
  - Verified shell syntax for all new actor-segmentation bash scripts.
  - Verified Python compilation for the new actor-segmentation modules and the modified deploy/env files.
  - Ran the synthetic `actorseg` projection unit test successfully.
  - Ran a negative smoke test against the existing non-actorseg dataset and confirmed the new preprocessing path fails fast with a clear `missing_actor_segmentation` error.
- Files created/modified:
  - `findings.md` (updated)
  - `progress.md` (updated)
