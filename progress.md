# Progress Log

## 2026-04-24: Real Three-ZED Collection Design
- Read `include/xtrainer_clover/experiments/run_control.py` and identified the robot-control loop, button semantics, DP frame save path, and current RGB-only RealSense camera handling.
- Read `include/xtrainer_clover/scripts/format_obs.py` and confirmed the original DP save path writes per-frame pkl files with `obs["control"]` and activation flags.
- Read `include/xtrainer_clover/scripts/4_collect2train_data.py` and confirmed the existing xtrainer train conversion targets ACT/DP image datasets, not RoboTwin point-cloud HDF5.
- Read DP3 preprocessing utilities and confirmed canonical compatibility requires `/joint_action/vector`, `/pointcloud`, and optional `/object_pointcloud/{placeholder}` HDF5 datasets.
- Replaced the stale local planning files with the current real-data collection design task.
- Read `calibrate_three_zed_charuco_extrinsics.py` and confirmed it provides relative three-camera extrinsics to a reference camera, not a complete robot-base/world calibration.
- Read the generated `three_camera_charuco_extrinsics.yaml` and confirmed it includes `relative_to_reference` transforms plus residuals for all camera labels.
- Read `detect_charuco_to_robot_point.py` and `world_frame.yaml` to identify existing robot/base and board/world calibration pieces that could be wrapped by the new real collection folder.
- Added regression test `script/test_real_zed_collection_pipeline.py`.
- Added `script/real_zed_collection/real_zed_utils.py`.
- Added `script/real_zed_collection/postprocess_raw_to_robotwin_hdf5.py`.
- Added `script/real_zed_collection/collect_zed_robotwin_raw.py`.
- Added `script/real_zed_collection/segment_objects_sam.py`.
- Added `script/real_zed_collection/calibrate_three_zed_extrinsics.py` wrapper, config templates, and README.
- Verified `python -m unittest script/test_real_zed_collection_pipeline.py`.
- Verified `python -m py_compile` on all new real-ZED scripts and the test.
- Replaced `script/real_zed_collection/calibrate_three_zed_extrinsics.py` wrapper with an inline implementation of three-ZED Charuco calibration.
- Verified `python -m py_compile script/real_zed_collection/calibrate_three_zed_extrinsics.py` and re-ran `python -m unittest script/test_real_zed_collection_pipeline.py`.

## 2026-05-16: Real DP3 EEF Absolute-6D Branch
- Added EEF pose/action utility coverage in `script/test_eef_action_utils.py`.
- Added `policy/DP3/scripts/eef_action_utils.py` for local Dobot FK, pose transforms, rot6d encoding/decoding, EEF14/action20 conversion, and calibration-based world/base transform loading.
- Extended `process_data.py`, `process_data_objpc.py`, and `process_data_semantic_pointwise.py` with default-off `--action_mode=eef_absolute6d` support.
- Added EEF-specific preprocess wrappers, train wrappers, and Hydra task/model configs for scene baseline, objpc baseline, and semantic-pointwise hybrid.
- Updated real-ZED HDF5 postprocessing to preserve `/joint_action/control` for future target-action conversion while retaining fallback support for older HDF5 files.
- Updated real DP3 inference with `--action_mode eef_absolute6d`, action20 decoding through Dobot IK, and EEF agent-pos updates during observation reuse.
- Updated `/home/zheng/github/xtrainer_clover` via the `include/xtrainer_clover` symlink with Dobot `get_eef_pose`, `get_fk`, `get_ik`, ZMQ forwarding, and RobotEnv wrappers.
- Verified:
  - `python -m unittest script.test_eef_action_utils script.test_real_zed_inference_actions`
  - `python policy/DP3/scripts/test_eef_pointwise_wrappers.py`
  - `bash -n` on the new EEF shell wrappers
  - `python -m py_compile` on the new EEF utilities, wrappers, real inference, and xtrainer_clover interface files

## 2026-05-17: DP3 Main Point-Cloud Count Training Interface
- Started a new training-wrapper task to expose `point_cloud_num` across DP3 train scripts.
- Decided that default 1024 behavior should keep old artifact names, while non-default counts append `-pcN` to zarr/checkpoint settings.
- Decided that main context point count must remain separate from semantic/NDF/Utonia auxiliary feature point counts.
- Added `point_cloud_num` forwarding across DP3 train/process wrappers, including baseline, objpc, actorseg, NDF, semantic, Utonia, and EEF variants.
- Verified with `test_semantic_hybrid_training_resource_overrides.py`, `test_eef_pointwise_wrappers.py`, targeted `py_compile`, `bash -n` over DP3 shell wrappers, and `git diff --check`.
- Extended the same main point-cloud count interface to all DP3 eval wrappers.
- Added `deploy_policy.py` support for `usr_args["point_cloud_num"]`, overriding `cfg.task.shape_meta.obs.point_cloud.shape` before checkpoint loading and using the same value for online object-cloud resampling.
- Verified eval coverage with `test_eval_pointwise_script_interfaces.sh`, `test_eval_utonia_pointwise_hybrid_interface.sh`, `test_semantic_hybrid_training_resource_overrides.py`, `python -m py_compile policy/DP3/deploy_policy.py`, `bash -n policy/DP3/eval*.sh`, and `git diff --check`.
- Fixed the `--output_suffix` argparse failure for suffixes starting with `-` by switching all DP3 preprocess shell wrappers to `--output_suffix="${output_suffix}"`.
- Added fail-fast behavior to `train_objpc.sh` so a failed preprocessing step does not continue into Hydra training.
- Verified the fix with the resource override unit test, shell syntax checks, a mocked `process_data_objpc.sh` argument capture showing `--output_suffix=-objpc-pc2048`, EEF wrapper tests, eval pointwise interface tests, and `git diff --check`.

## 2026-05-19: Real DP3 EEF Async Robot-I/O Serialization
- Traced the EEF async inference crash to xtrainer `ZMQClientRobot.get_ik(...)` raising `ZMQError: Operation cannot be accomplished in current state`.
- Identified that async control sends `env.step(...)` from a background thread while the policy thread converts EEF action chunks through Dobot IK, both using the same ZeroMQ REQ socket.
- Added `robot_io_guard(...)` and a per-run `threading.RLock` in `script/real_zed_inference/real_dp3_inference.py`.
- Wrapped EEF IK conversion and real robot command execution in the shared robot I/O guard.
- Added regression coverage requiring both `get_ik` and `step` to run under the same lock.
- Verified:
  - `python -m unittest script.test_real_zed_inference_actions`
  - `python -m py_compile script/real_zed_inference/real_dp3_inference.py script/test_real_zed_inference_actions.py`
  - `git diff --check`

## 2026-05-19: Real EEF Pose Saving For Training
- Added tests showing raw collection should preserve measured EEF pose, postprocess should write `/eef_action/base_pose6`, and DP3 EEF training arrays should prefer measured pose over offline FK.
- Updated `collect_zed_robotwin_raw.py` so `_build_robot_snapshot(...)` captures 12D measured EEF pose from `eef_pose_base` or xtrainer `ee_pos_quat`, and `_save_raw_frame(...)` writes it into `robot_*.npz`.
- Updated `postprocess_raw_to_robotwin_hdf5.py` to propagate raw `eef_pose_base` into `/eef_action/base_pose6` and mark `eef_pose_source` in scene info.
- Updated `eef_action_utils.py` so `episode_eef_state_action_arrays(...)` accepts measured base-frame EEF poses, transforms them into the configured training frame, and keeps old offline-FK behavior when measured EEF pose is absent.
- Updated DP3 loaders (`object_pointcloud_utils.py`, `process_data.py`) so scene, objpc, semantic, and other object-pointcloud based EEF preprocessors can receive `eef_pose_base`.
- Verified:
  - `python -m unittest script.test_eef_action_utils`
  - `python -m unittest script.test_real_zed_collection_pipeline.RealZedCollectionPipelineTest.test_collect_robot_snapshot_preserves_true_eef_pose script.test_real_zed_collection_pipeline.RealZedCollectionPipelineTest.test_postprocess_writes_robotwin_hdf5_from_raw_episode_and_masks script.test_real_zed_collection_pipeline.RealZedCollectionPipelineTest.test_postprocess_can_write_pointclouds_in_left_base_frame`
  - `python policy/DP3/scripts/test_eef_pointwise_wrappers.py`
  - `python -m py_compile script/real_zed_collection/collect_zed_robotwin_raw.py script/real_zed_collection/postprocess/postprocess_raw_to_robotwin_hdf5.py policy/DP3/scripts/eef_action_utils.py policy/DP3/scripts/object_pointcloud_utils.py policy/DP3/scripts/process_data.py policy/DP3/scripts/process_data_objpc.py policy/DP3/scripts/process_data_semantic_pointwise.py script/test_real_zed_collection_pipeline.py script/test_eef_action_utils.py`
  - `git diff --check`
- Full `python -m unittest script.test_real_zed_collection_pipeline` still fails in this shell because `zarr` is not installed; the failing tests are unrelated DP image preprocess imports.

## 2026-05-20: Semantic vs Baseline EEF Gripper Diagnostics
- Read `outputs/real_zed_inference/semantic_eef_gripper.csv` and `outputs/real_zed_inference/baseline_eef_gripper.csv`.
- Found both runs have `command_follow_error_gripper=0`, so the real gripper follows the commanded gripper targets in the diagnostic data.
- Semantic run: 24 high command-gripper-delta runs at threshold `0.0199`, total 113 steps, average run length 4.7, maximum run length 15.
- Baseline run: 8 high command-gripper-delta runs at threshold `0.0199`, total 154 steps, average run length 19.25, maximum run length 59.
- Semantic command-gripper-delta-change is higher (`mean=0.00463`, `p90=0.01981`) than baseline (`mean=0.00213`, `p90=0.00691`), indicating more fragmented/chattery gripper command phases.

## Session: 2026-04-16 (`pour_kettle_mug`)

### Phase 1: Planning & Test Scope
- **Status:** complete
- Actions taken:
  - Confirmed the `pour_kettle_mug` design with the user and wrote the approved spec to `docs/superpowers/specs/2026-04-16-pour-kettle-mug-design.md`.
  - Committed the design spec separately as `a05c88a`.
  - Switched `task_plan.md` from an unrelated DP3 task to a task-specific implementation plan for `pour_kettle_mug`.
  - Reviewed existing lightweight test patterns under `script/` to choose a realistic test-first path.
  - Confirmed from asset metadata that `009_kettle` exposes a handle grasp point and a spout proxy point, and that `039_mug` exposes a bottom functional point suitable for deriving a mug-opening approximation.
- Files created/modified:
  - `docs/superpowers/specs/2026-04-16-pour-kettle-mug-design.md` (created earlier this session)
  - `task_plan.md` (replaced)
  - `findings.md` (updated)
  - `progress.md` (updated)

### Phase 2: Red Test
- **Status:** complete
- Actions taken:
  - Added `script/test_pour_kettle_mug_task_integration.py` as a lightweight source/config integration test.
  - Ran `python script/test_pour_kettle_mug_task_integration.py`.
  - Confirmed the test fails for the expected reasons:
    - missing `envs/pour_kettle_mug.py`
    - missing `description/task_instruction/pour_kettle_mug.json`
    - missing `pour_kettle_mug` mapping in `envs/object_pointcloud_targets.py`
    - missing `pour_kettle_mug` entry in `task_config/_eval_step_limit.yml`
- Files created/modified:
  - `script/test_pour_kettle_mug_task_integration.py` (created)
  - `task_plan.md` (updated)
  - `progress.md` (updated)

### Phase 3: Implementation
- **Status:** complete
- Actions taken:
  - Added `envs/pour_kettle_mug.py`.
  - Implemented the task as a left-arm-only geometry pouring task.
  - Randomized `009_kettle` across the three available URDF instances and `039_mug` across the available mug instances.
  - Added helper logic to:
    - approximate the mug opening center from the mug bottom point and scaled mug height
    - compute the current spout-to-end-effector transform after grasping
    - build pre-pour and final pour end-effector poses that keep the spout proxy above the mug
    - rotate the end effector about a world-axis derived from the spout offset to create a fixed pouring tilt
  - Added the language instruction template `description/task_instruction/pour_kettle_mug.json`.
  - Registered placeholder pointcloud mapping for `pour_kettle_mug` in `envs/object_pointcloud_targets.py`.
  - Added an eval step-limit entry in `task_config/_eval_step_limit.yml`.
- Files created/modified:
  - `envs/pour_kettle_mug.py` (created)
  - `description/task_instruction/pour_kettle_mug.json` (created)
  - `envs/object_pointcloud_targets.py` (modified)
  - `task_config/_eval_step_limit.yml` (modified)

### Phase 4: Verification
- **Status:** complete
- Actions taken:
  - Re-ran `python script/test_pour_kettle_mug_task_integration.py` and confirmed it passes with 5 tests.
  - Re-ran `python -m py_compile envs/pour_kettle_mug.py script/test_pour_kettle_mug_task_integration.py envs/object_pointcloud_targets.py` and confirmed clean compilation.
  - Checked direct module import and confirmed that both the new task and an existing task fail at the same `ModuleNotFoundError: sapien`, which shows the current shell environment cannot perform runtime env imports.
- Files created/modified:
  - `envs/pour_kettle_mug.py` (modified)
  - `task_plan.md` (updated)
  - `findings.md` (updated)
  - `progress.md` (updated)

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

## Session: 2026-04-10 (NDF Investigation)

### Phase 1: Experiment Surface Audit
- **Status:** complete
- Actions taken:
  - Restored planning context and switched the task plan from actor-segmentation implementation to NDF underperformance diagnosis.
  - Inspected `train_objpc.sh`, `train_ndf_pointwise.sh`, `train_semantic_pointwise.sh` and the matching eval scripts.
  - Compared `robot_dp3_objpc.yaml`, `robot_dp3_ndf_pointwise.yaml`, `robot_dp3_semantic_pointwise.yaml` and the matching task configs.
  - Traced the runtime observation path in `policy/DP3/deploy_policy.py`.
  - Confirmed that `semantic_pointwise` uses a different default training recipe (`use_ema=false`, `batch_size=110`) from `objpc` and `ndf_pointwise`.
- Files created/modified:
  - `task_plan.md` (replaced)
  - `findings.md` (updated)
  - `progress.md` (updated)

### Phase 2: Representation Path Comparison
- **Status:** complete
- Actions taken:
  - Inspected `process_data_ndf_pointwise.py`, `process_data_semantic_pointwise.py`, `ndf_feature_utils.py`, `semantic_feature_utils.py`, `object_pointcloud_utils.py`, `robot_dataset.py`, and `pointnet_extractor.py`.
  - Confirmed that both pointwise variants remove feature placeholders from the primary `point_cloud` and instead inject a second point-cloud branch for the feature placeholder.
  - Confirmed that baseline `objpc` and pointwise variants therefore do not share the same observation factorization or encoder structure.
  - Confirmed that NDF / semantic extra point-cloud branches are both encoded by the generic `PointNetEncoderXYZRGB` branch inside `DP3Encoder`.
- Files created/modified:
  - `findings.md` (updated)
  - `progress.md` (updated)

## Session: 2026-04-14 (ActorSeg Raw Geometry Audit)

### Phase 1: Current Data Capability Check
- **Status:** complete
- Actions taken:
  - Re-read the current project planning files to recover active context and prior actorseg-hybrid work.
  - Inspected `task_config/demo_clean_3d_actorseg.yml` and `task_config/demo_randomized_3d_actorseg.yml`.
  - Confirmed both actorseg configs collect RGB, actor segmentation, and a whole-scene pointcloud, but not depth or raw per-camera position buffers.
  - Inspected an actual collected file `data/hanging_mug/demo_clean_3d_actorseg/data/episode0.hdf5`.
  - Confirmed the dataset stores `actor_segmentation`, camera intrinsics/extrinsics, and `pointcloud (T, 1024, 6)`, but does not store raw per-camera `Position` or depth.
  - Traced the current `objpc` extraction path in `envs/camera/camera.py` and the current actorseg extraction path in `policy/DP3/scripts/actorseg_pointcloud_utils.py`.
  - Confirmed the core structural gap: `objpc` filters raw per-camera position pixels before downsampling, while actorseg filters an already-downsampled scene cloud after projection.
  - Recorded the conclusion that online eval can be upgraded immediately, but offline training requires a new richer collection config and recollection.
- Files created/modified:
  - `findings.md` (updated)
  - `progress.md` (updated)

## Session: 2026-04-14 (ObjPC 5000-Point Path)

### Phase 1: Scope & Consistency Audit
- **Status:** complete
- Actions taken:
  - Traced the full `object_pc` chain from collection config to HDF5, zarr preprocessing, DP3 task shape, and train/eval scripts.
  - Confirmed that changing only task config point counts would not be sufficient because preprocessing still defaulted to `1024` and DP3 task shape still expected `[1024, 6]`.
  - Confirmed existing collected `demo_clean_3d_object_pc` data is already stored as `(T, 1024, 6)` per object, so real 5000-point data requires recollection.
  - Chose an isolated 5000-point path to avoid breaking existing 1024-point checkpoints and scripts.
- Files created/modified:
  - `findings.md` (updated)
  - `progress.md` (updated)

### Phase 2: TDD & Implementation
- **Status:** complete
- Actions taken:
  - Added a failing regression test `policy/DP3/scripts/test_objpc_5000_path.py` covering collection config, preprocess wiring, and DP3 config shape consistency.
  - Added `task_config/demo_clean_3d_object_pc_5000.yml`.
  - Added `task_config/demo_randomized_3d_object_pc_5000.yml`.
  - Added `policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/task/demo_task_objpc_5000.yaml`.
  - Added `policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/robot_dp3_objpc_5000.yaml`.
  - Added `policy/DP3/train_objpc_5000.sh` and `policy/DP3/eval_objpc_5000.sh`.
  - Extended `policy/DP3/process_data_objpc.sh` with an explicit `target_num_points` argument so the new 5000-point path preprocesses correctly.
- Files created/modified:
  - `policy/DP3/scripts/test_objpc_5000_path.py` (created)
  - `task_config/demo_clean_3d_object_pc_5000.yml` (created)
  - `task_config/demo_randomized_3d_object_pc_5000.yml` (created)
  - `policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/task/demo_task_objpc_5000.yaml` (created)
  - `policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/robot_dp3_objpc_5000.yaml` (created)
  - `policy/DP3/train_objpc_5000.sh` (created)
  - `policy/DP3/eval_objpc_5000.sh` (created)
  - `policy/DP3/process_data_objpc.sh` (modified)

### Phase 3: Verification
- **Status:** complete
- Actions taken:
  - Ran `python policy/DP3/scripts/test_objpc_5000_path.py`.
  - Ran `bash -n policy/DP3/process_data_objpc.sh policy/DP3/train_objpc_5000.sh policy/DP3/eval_objpc_5000.sh`.
  - Ran `python -m py_compile policy/DP3/scripts/process_data_objpc.py policy/DP3/scripts/test_objpc_5000_path.py`.
- Files created/modified:
  - `findings.md` (updated)
  - `progress.md` (updated)

## Session: 2026-04-15 (Pointwise Preprocess OOM Fix)

### Phase 1: Root-Cause Investigation
- **Status:** complete
- Actions taken:
  - Inspected the failing wrapper `policy/DP3/process_data_semantic_pointwise_hybrid_feat5000.sh` and traced the actual work into `policy/DP3/scripts/process_data_semantic_pointwise.py`.
  - Confirmed the shell-side `Killed` happened at Python process launch rather than as a Python exception.
  - Computed the approximate memory footprint of `5000 x (3+128)` semantic pointwise arrays across a 50-demo run and confirmed it is on the order of `40+ GB` before extra overhead.
  - Identified the root cause as the base pointwise preprocessors buffering all frames in memory before writing zarr.
- Files created/modified:
  - `findings.md` (updated)
  - `progress.md` (updated)

### Phase 2: TDD & Implementation
- **Status:** complete
- Actions taken:
  - Added a failing regression test `policy/DP3/scripts/test_pointwise_preprocess_meta.py`.
  - Added `policy/DP3/scripts/pointwise_preprocess_meta.py` for shared incremental metadata handling.
  - Refactored `policy/DP3/scripts/process_data_semantic_pointwise.py` to incremental episode-at-a-time zarr appends via `ReplayBuffer`.
  - Refactored `policy/DP3/scripts/process_data_ndf_pointwise.py` to the same incremental append pattern so the NDF feat5000 path does not hit the same OOM next.
  - Preserved per-episode metadata and resumability semantics in both scripts.
- Files created/modified:
  - `policy/DP3/scripts/test_pointwise_preprocess_meta.py` (created)
  - `policy/DP3/scripts/pointwise_preprocess_meta.py` (created)
  - `policy/DP3/scripts/process_data_semantic_pointwise.py` (modified)
  - `policy/DP3/scripts/process_data_ndf_pointwise.py` (modified)

### Phase 3: Verification
- **Status:** complete
- Actions taken:
  - Ran `python policy/DP3/scripts/test_pointwise_preprocess_meta.py`.
  - Re-ran `python policy/DP3/scripts/test_ndf_pointwise_hybrid.py`.
  - Re-ran `python policy/DP3/scripts/test_semantic_pointwise_hybrid.py`.
  - Ran `python -m py_compile policy/DP3/scripts/pointwise_preprocess_meta.py policy/DP3/scripts/process_data_ndf_pointwise.py policy/DP3/scripts/process_data_semantic_pointwise.py`.
  - Ran `bash -n` on the relevant feat5000 wrapper scripts to ensure shell entrypoints remain valid.
  - Hardened the `feat5000` train wrappers so they only skip preprocessing when the target zarr contains the required replay-buffer datasets.
  - Re-ran `bash -n policy/DP3/train_semantic_pointwise_hybrid_feat5000.sh policy/DP3/train_ndf_pointwise_hybrid_feat5000.sh`.
- Files created/modified:
  - `findings.md` (updated)
  - `progress.md` (updated)

## Session: 2026-04-14 (Hybrid Feature5000 Paths)

### Phase 1: Design Audit
- **Status:** complete
- Actions taken:
  - Confirmed from `deploy_policy.py` that hybrid raw `point_cloud` and NDF/semantic pointwise branches are encoded independently and concatenated only after branch encoding.
  - Confirmed there is no per-point alignment requirement between the 1024-point raw branch and a denser feature branch.
  - Confirmed that requesting 5000 feature points from an underlying 1024-point object cloud would only repeat / pad points, so the new path only makes sense together with recollected 5000-point object-PCD data.
- Files created/modified:
  - `findings.md` (updated)
  - `progress.md` (updated)

### Phase 2: TDD & Implementation
- **Status:** complete
- Actions taken:
  - Added failing regression test `policy/DP3/scripts/test_hybrid_feature5000_path.py`.
  - Added isolated NDF feature5000 wrapper/config/script path.
  - Added isolated semantic feature5000 wrapper/config/script path.
  - Kept the raw hybrid `point_cloud` shape at `[1024, 6]`.
  - Set feature branch defaults to `5000` points and used a distinct `-feat5000` suffix to isolate checkpoint naming.
- Files created/modified:
  - `policy/DP3/scripts/test_hybrid_feature5000_path.py` (created)
  - `policy/DP3/scripts/process_data_ndf_pointwise_hybrid_feat5000.py` (created)
  - `policy/DP3/scripts/process_data_semantic_pointwise_hybrid_feat5000.py` (created)
  - `policy/DP3/process_data_ndf_pointwise_hybrid_feat5000.sh` (created)
  - `policy/DP3/process_data_semantic_pointwise_hybrid_feat5000.sh` (created)
  - `policy/DP3/train_ndf_pointwise_hybrid_feat5000.sh` (created)
  - `policy/DP3/train_semantic_pointwise_hybrid_feat5000.sh` (created)
  - `policy/DP3/eval_ndf_pointwise_hybrid_feat5000.sh` (created)
  - `policy/DP3/eval_semantic_pointwise_hybrid_feat5000.sh` (created)
  - `policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/task/demo_task_ndf_pointwise_hybrid_feat5000.yaml` (created)
  - `policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/task/demo_task_semantic_pointwise_hybrid_feat5000.yaml` (created)
  - `policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/robot_dp3_ndf_pointwise_hybrid_feat5000.yaml` (created)
  - `policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/robot_dp3_semantic_pointwise_hybrid_feat5000.yaml` (created)

### Phase 3: Verification
- **Status:** complete
- Actions taken:
  - Ran `python policy/DP3/scripts/test_hybrid_feature5000_path.py`.
  - Ran `bash -n` on all new feature5000 shell scripts.
  - Ran `python -m py_compile` on the new Python wrappers and the new regression test.
- Files created/modified:
  - `findings.md` (updated)
  - `progress.md` (updated)

### Phase 3: Evidence Gathering

## Session: 2026-04-13 (Semantic Hybrid Follow-Up)

### Phase 7: Discovery & Design
- **Status:** in_progress
- Actions taken:
  - Restored planning context and added a new follow-up phase for the semantic hybrid request.
  - Read `train_ndf_pointwise_hybrid.sh`, `process_data_ndf_pointwise_hybrid.sh`, `process_data_ndf_pointwise.py`, `deploy_policy.py`, and the matching NDF hybrid Hydra configs to pin down the exact hybrid pattern.
  - Read the existing semantic pointwise preprocess/train/eval chain and confirmed it still removes feature placeholders from the main merged `point_cloud`.
  - Confirmed the NDF hybrid pattern is two-part: keep the raw merged ObjPC in `point_cloud`, then append `ndf_point_cloud_A/B` as extra observation keys with explicit shape overrides.
  - Confirmed the semantic path can follow the same structure with a small extension instead of a new architecture.
- Files created/modified:
  - `task_plan.md` (updated)
  - `findings.md` (updated)
  - `progress.md` (updated)

### Phase 7: Implementation & Verification
- **Status:** complete
- Actions taken:
  - Added a failing regression test `policy/DP3/scripts/test_semantic_pointwise_hybrid.py` covering both the old semantic pointwise behavior and the new hybrid behavior.
  - Refactored `process_data_semantic_pointwise.py` to expose `build_parser()`, accept `--keep_feature_placeholders_in_context`, and reuse `pointwise_context_utils.py`.
  - Added the thin wrapper `policy/DP3/scripts/process_data_semantic_pointwise_hybrid.py`.
  - Added new shell entrypoints:
    - `policy/DP3/process_data_semantic_pointwise_hybrid.sh`
    - `policy/DP3/train_semantic_pointwise_hybrid.sh`
    - `policy/DP3/eval_semantic_pointwise_hybrid.sh`
  - Added new Hydra configs:
    - `policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/robot_dp3_semantic_pointwise_hybrid.yaml`
    - `policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/task/demo_task_semantic_pointwise_hybrid.yaml`
  - Extended `policy/DP3/deploy_policy.py` with `use_semantic_pointwise_hybrid` and reused the same “keep feature placeholders in main context” path as the NDF hybrid.
  - Verified the new semantic hybrid tests pass and the existing NDF hybrid regression test still passes.
  - Verified Python compilation and shell syntax for all newly added semantic hybrid entrypoints.
- Files created/modified:
  - `policy/DP3/scripts/test_semantic_pointwise_hybrid.py` (created)
  - `policy/DP3/scripts/process_data_semantic_pointwise.py` (modified)
  - `policy/DP3/scripts/process_data_semantic_pointwise_hybrid.py` (created)
  - `policy/DP3/process_data_semantic_pointwise_hybrid.sh` (created)
  - `policy/DP3/train_semantic_pointwise_hybrid.sh` (created)
  - `policy/DP3/eval_semantic_pointwise_hybrid.sh` (created)
  - `policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/robot_dp3_semantic_pointwise_hybrid.yaml` (created)
  - `policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/task/demo_task_semantic_pointwise_hybrid.yaml` (created)
  - `policy/DP3/deploy_policy.py` (modified)
  - `task_plan.md` (updated)
  - `findings.md` (updated)
  - `progress.md` (updated)

## Session: 2026-04-14 (NDF Hybrid Eval CLI Debug)

### Phase 1: Root Cause Isolation
- **Status:** complete
- Actions taken:
  - Inspected the user's failing `eval_ndf_pointwise_hybrid.sh` invocation against the script's positional argument contract.
  - Confirmed the eval command omitted the explicit empty `ndf_dgcnn_placeholders` argument, shifting `object_placeholders` to `3000`.
  - Confirmed this explains the log line `point cloud keys: ['point_cloud']` and the downstream checkpoint mismatch (`ndf_point_cloud_A` missing at eval model construction time).
  - Confirmed the checkpoint itself is structurally healthy for hybrid training because its state dict contains `obs_encoder.extractors.ndf_point_cloud_A.*`.
- Files created/modified:
  - `findings.md` (updated)
  - `progress.md` (updated)

### Phase 2: Compatibility Fix
- **Status:** complete
- Actions taken:
  - Added `policy/DP3/ndf_pointwise_arg_utils.sh` to normalize both explicit and legacy NDF pointwise CLI forms.
  - Patched these scripts to use the shared parser and emit a concise compatibility warning when auto-shifting arguments:
    - `policy/DP3/train_ndf_pointwise.sh`
    - `policy/DP3/eval_ndf_pointwise.sh`
    - `policy/DP3/train_ndf_pointwise_hybrid.sh`
    - `policy/DP3/eval_ndf_pointwise_hybrid.sh`
  - Added regression coverage in `policy/DP3/scripts/test_ndf_pointwise_arg_utils.sh` for the exact omitted-empty-argument form that caused this bug.
- Files created/modified:
  - `policy/DP3/ndf_pointwise_arg_utils.sh` (created)
  - `policy/DP3/train_ndf_pointwise.sh` (modified)
  - `policy/DP3/eval_ndf_pointwise.sh` (modified)
  - `policy/DP3/train_ndf_pointwise_hybrid.sh` (modified)
  - `policy/DP3/eval_ndf_pointwise_hybrid.sh` (modified)
  - `policy/DP3/scripts/test_ndf_pointwise_arg_utils.sh` (created)
  - `findings.md` (updated)
  - `progress.md` (updated)

### Phase 3: Verification
- **Status:** complete
- Actions taken:
  - Ran `bash policy/DP3/scripts/test_ndf_pointwise_arg_utils.sh`
  - Ran `bash -n policy/DP3/eval_ndf_pointwise_hybrid.sh`
  - Ran `bash -n policy/DP3/train_ndf_pointwise_hybrid.sh`
  - Ran `bash -n policy/DP3/eval_ndf_pointwise.sh`
  - Ran `bash -n policy/DP3/train_ndf_pointwise.sh`
- Files created/modified:
  - `progress.md` (updated)

## Session: 2026-04-14 (ActorSeg Hybrid Design)

### Phase 1: Discovery & Design
- **Status:** in_progress
- Actions taken:
  - Captured the new task: use actorseg mask-projected object point clouds as the main raw input while adding NDF or semantic hybrid branches, matching the already-successful objpc hybrid structure.
  - Re-read the current `objpc_actorseg` training/eval/preprocess chain.
  - Re-read the existing `ndf_pointwise_hybrid` and `semantic_pointwise_hybrid` paths to identify the minimum extension pattern.
  - Chose a non-invasive design: add separate actorseg+hybrid paths instead of mutating existing working scripts and configs.
- Files created/modified:
  - `task_plan.md` (updated)
  - `findings.md` (updated)
  - `progress.md` (updated)

### Phase 2: Test Design
- **Status:** complete
- Actions taken:
  - Added `policy/DP3/scripts/test_actorseg_pointwise_hybrid.py`.
  - First wrote the runtime tests too weakly by checking only output shapes; they passed immediately and did not prove actorseg point clouds were feeding the feature branches.
  - Tightened the assertions to require that `ndf_point_cloud_A` / `semantic_point_cloud_A` equal the mocked feature outputs, which correctly produced failing tests against the old runtime logic.
- Files created/modified:
  - `policy/DP3/scripts/test_actorseg_pointwise_hybrid.py` (created)

### Phase 3: Implementation
- **Status:** complete
- Actions taken:
  - Updated `policy/DP3/deploy_policy.py` so actorseg-extracted per-placeholder point clouds are reused for hybrid feature branches instead of being discarded after building the merged raw cloud.
  - Added actorseg+NDF hybrid preprocess/train/eval/config entrypoints.
  - Added actorseg+semantic hybrid preprocess/train/eval/config entrypoints.
  - Reused existing actorseg metadata requirements and incremental zarr writing.
  - Extended the NDF bash-argument compatibility helper to remain valid when actorseg camera-name parameters are appended to the end of the command line.
- Files created/modified:
  - `policy/DP3/deploy_policy.py` (modified)
  - `policy/DP3/ndf_pointwise_arg_utils.sh` (modified)
  - `policy/DP3/scripts/process_data_ndf_pointwise_actorseg_hybrid.py` (created)
  - `policy/DP3/scripts/process_data_semantic_pointwise_actorseg_hybrid.py` (created)
  - `policy/DP3/process_data_ndf_pointwise_actorseg_hybrid.sh` (created)
  - `policy/DP3/process_data_semantic_pointwise_actorseg_hybrid.sh` (created)
  - `policy/DP3/train_ndf_pointwise_actorseg_hybrid.sh` (created)
  - `policy/DP3/train_semantic_pointwise_actorseg_hybrid.sh` (created)
  - `policy/DP3/eval_ndf_pointwise_actorseg_hybrid.sh` (created)
  - `policy/DP3/eval_semantic_pointwise_actorseg_hybrid.sh` (created)
  - `policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/robot_dp3_objpc_actorseg_ndf_pointwise_hybrid.yaml` (created)
  - `policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/robot_dp3_objpc_actorseg_semantic_pointwise_hybrid.yaml` (created)
  - `policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/task/demo_task_objpc_actorseg_ndf_pointwise_hybrid.yaml` (created)
  - `policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/task/demo_task_objpc_actorseg_semantic_pointwise_hybrid.yaml` (created)

### Phase 4: Verification
- **Status:** complete
- Actions taken:
  - Ran `python policy/DP3/scripts/test_actorseg_pointwise_hybrid.py`
  - Ran `python policy/DP3/scripts/test_ndf_pointwise_hybrid.py`
  - Ran `python policy/DP3/scripts/test_semantic_pointwise_hybrid.py`
  - Ran `bash policy/DP3/scripts/test_ndf_pointwise_arg_utils.sh`
  - Ran `python -m py_compile policy/DP3/deploy_policy.py policy/DP3/scripts/process_data_ndf_pointwise_actorseg_hybrid.py policy/DP3/scripts/process_data_semantic_pointwise_actorseg_hybrid.py policy/DP3/scripts/test_actorseg_pointwise_hybrid.py`
  - Ran `bash -n` on all six new actorseg hybrid shell entrypoints
- Files created/modified:
  - `task_plan.md` (updated)
  - `findings.md` (updated)
  - `progress.md` (updated)
- **Status:** complete
- Actions taken:
  - Read the saved Hydra overrides for the existing local `objpc`, `ndf_pointwise`, and `semantic_pointwise` training runs.
  - Verified from the saved pointwise metadata that only `{A}` was augmented in both NDF and semantic experiments.
  - Verified from the saved zarr stores that all three variants have 50 episodes and aligned episode boundaries.
  - Verified from the saved metadata that the training artifacts span 9 mug instances from the same `039_mug` family.
  - Confirmed from `envs/hanging_mug.py` that eval samples from the same 10-instance mug family, so the current setup is not a strong novel-object benchmark.
  - Confirmed from `deploy_policy.yml` and `eval_policy.py` that `instruction_type: unseen` refers to unseen language instructions, not unseen objects.
  - Found a forwarding bug in `train_ndf_pointwise.sh`: `${ndf_dgcnn_/placeholders}` instead of `${ndf_dgcnn_placeholders}`.
  - Checked NDF checkpoint compatibility and confirmed:
    - `model_current.pth` matches the pointcloud backbone (`dgcnn=False`)
    - `model_best.pth` matches the DGCNN backbone (`dgcnn=True`)
  - Confirmed the current command examples use `model_current.pth` and do not request DGCNN.
  - Computed rough on-disk feature probes showing semantic features are much more temporally stable than the current NDF features.
  - Confirmed the current raw `data/hanging_mug/demo_clean_3d_object_pc` folder no longer matches the full source dataset used to build the 50-episode DP3 artifacts.
- Files created/modified:
  - `findings.md` (updated)
  - `progress.md` (updated)

### Phase 4: Root-Cause Synthesis
- **Status:** in_progress
- Actions taken:
  - Explained that the `ndf_dgcnn_placeholders` forwarding bug is real but irrelevant to the user's current pointnet-backed NDF runs when the argument is intentionally empty.
  - Fixed `policy/DP3/train_ndf_pointwise.sh` so DGCNN placeholder forwarding now works for future runs.
  - Restored `policy/DP3/train_semantic_pointwise.sh` defaults to a fairer base-DP3 recipe: `batch_size=256`, `val_batch_size=batch_size`, `use_ema=true`.
  - Verified both modified scripts with `bash -n`.
  - Synthesized the ranked diagnosis: current NDF underperformance is most likely caused by the integration path and only secondarily by the benchmark design; there is not enough evidence to conclude the NDF representation itself is useless.
  - Incorporated the user's additional evidence that the already-tested global-NDF path (`train_ndf.sh`) severely hurt mug approach behavior, which strengthens the diagnosis that pose-invariant global NDF descriptors are a poor direct drop-in conditioning signal for this DP3 setup.
- Files created/modified:
  - `policy/DP3/train_ndf_pointwise.sh` (modified)
  - `policy/DP3/train_semantic_pointwise.sh` (modified)
  - `findings.md` (updated)
  - `progress.md` (updated)

## Session: 2026-04-11 (NDF Hybrid Implementation)

### Phase 1: Design Confirmation
- **Status:** complete
- Actions taken:
  - Confirmed with the user that the correct next experiment is a separate `ndf_pointwise_hybrid` path.
  - Wrote and committed the design spec `docs/superpowers/specs/2026-04-11-ndf-pointwise-hybrid-design.md`.
  - Waited for and received user approval before implementation.
- Files created/modified:
  - `docs/superpowers/specs/2026-04-11-ndf-pointwise-hybrid-design.md` (created and committed)

### Phase 2: TDD Red Step
- **Status:** complete
- Actions taken:
  - Added `policy/DP3/scripts/test_ndf_pointwise_hybrid.py`.
  - Verified the new test failed in the intended way: `ndf_pointwise_hybrid` still produced a main `point_cloud` containing only `{B}` instead of `{A}+{B}`.
- Files created/modified:
  - `policy/DP3/scripts/test_ndf_pointwise_hybrid.py` (created)

### Phase 3: Implementation
- **Status:** complete
- Actions taken:
  - Added `policy/DP3/scripts/pointwise_context_utils.py` to centralize context-cloud construction.
  - Extended `policy/DP3/scripts/process_data_ndf_pointwise.py` with `--keep_feature_placeholders_in_context`.
  - Added `policy/DP3/scripts/process_data_ndf_pointwise_hybrid.py` as a thin wrapper for the new suffix and context behavior.
  - Added `policy/DP3/process_data_ndf_pointwise_hybrid.sh`, `policy/DP3/train_ndf_pointwise_hybrid.sh`, and `policy/DP3/eval_ndf_pointwise_hybrid.sh`.
  - Added `policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/robot_dp3_ndf_pointwise_hybrid.yaml`.
  - Added `policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/task/demo_task_ndf_pointwise_hybrid.yaml`.
  - Updated `policy/DP3/deploy_policy.py` so runtime hybrid eval keeps raw `{A}` in the main `point_cloud` while still adding `ndf_point_cloud_A/B`.
  - Updated `policy/DP3/Command.md` with train/eval examples for the hybrid path.
- Files created/modified:
  - `policy/DP3/scripts/pointwise_context_utils.py` (created)
  - `policy/DP3/scripts/process_data_ndf_pointwise.py` (modified)
  - `policy/DP3/scripts/process_data_ndf_pointwise_hybrid.py` (created)
  - `policy/DP3/process_data_ndf_pointwise_hybrid.sh` (created)
  - `policy/DP3/train_ndf_pointwise_hybrid.sh` (created)
  - `policy/DP3/eval_ndf_pointwise_hybrid.sh` (created)
  - `policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/robot_dp3_ndf_pointwise_hybrid.yaml` (created)
  - `policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/task/demo_task_ndf_pointwise_hybrid.yaml` (created)
  - `policy/DP3/deploy_policy.py` (modified)
  - `policy/DP3/Command.md` (modified)

### Phase 4: Verification
- **Status:** complete
- Actions taken:
  - Re-ran `policy/DP3/scripts/test_ndf_pointwise_hybrid.py` and confirmed the red test turned green.
  - Verified Python syntax with `python -m py_compile` on the changed Python files.
  - Verified shell syntax with `bash -n` on all three new hybrid shell scripts.
  - Updated `task_plan.md`, `findings.md`, and `progress.md` to reflect the completed implementation.
- Files created/modified:
  - `task_plan.md` (updated)
  - `findings.md` (updated)
  - `progress.md` (updated)

### Phase 5: Post-Implementation Bugfix
- **Status:** complete
- Actions taken:
  - Reproduced the user's first real-run failure in `process_data_ndf_pointwise_hybrid.py`: `argparse` rejected `--output_suffix` because the forwarded suffix value started with `-`.
  - Added a failing regression test for wrapper argv construction in `policy/DP3/scripts/test_ndf_pointwise_hybrid.py`.
  - Fixed the wrapper by introducing `build_hybrid_argv(...)` and forwarding `--output_suffix=-objpc-ndf-pointwise-hybrid` as a single attached option.
  - Re-ran the regression test and `py_compile`, and checked `--help` on the wrapper entrypoint.
- Files created/modified:
  - `policy/DP3/scripts/process_data_ndf_pointwise_hybrid.py` (modified)
  - `policy/DP3/scripts/test_ndf_pointwise_hybrid.py` (modified)
  - `findings.md` (updated)
  - `progress.md` (updated)

### Phase 6: NDF Load Warning Triage
- **Status:** complete
- Actions taken:
  - Inspected `policy/DP3/scripts/ndf_feature_utils.py` to confirm the NDF loader uses `strict=False`.
  - Traced the reported `unexpected_keys` to the optional decoder vector-feature head in `vnn_occupancy_net_pointnet_dgcnn.py`.
  - Confirmed the current DP3 NDF path only consumes regular local `features` through `forward_latent(...)`, not `vector_features`.
  - Concluded that this exact warning shape is benign when `missing_keys=[]`, but would become suspicious if missing keys or core encoder/decoder mismatches appeared.
- Files created/modified:
  - `findings.md` (updated)
  - `progress.md` (updated)

### Phase 7: ActorSeg Hybrid Placeholder Default Fix
- **Status:** complete
- Actions taken:
  - Reproduced the user's actorseg hybrid preprocessing failure and confirmed that `policy/DP3/ndf_pointwise_arg_utils.sh` emitted the default placeholder string as `\{A\},\{B\}` when `object_placeholders` was omitted.
  - Added regression coverage to `policy/DP3/scripts/test_ndf_pointwise_arg_utils.sh` for omitted-placeholder defaults across train, eval, eval-hybrid, and actorseg-process parsing.
  - Fixed the shared NDF arg helper by replacing the escaped inline default with a plain `DEFAULT_OBJECT_PLACEHOLDERS="{A},{B}"` constant.
  - Aligned `semantic_pointwise_actorseg_hybrid` train/process/eval scripts to the same plain default placeholder constant style.
  - Re-ran the shell regression test and `bash -n` on all touched actorseg hybrid scripts.
- Files created/modified:
  - `policy/DP3/ndf_pointwise_arg_utils.sh` (modified)
  - `policy/DP3/scripts/test_ndf_pointwise_arg_utils.sh` (modified)
  - `policy/DP3/train_semantic_pointwise_actorseg_hybrid.sh` (modified)
  - `policy/DP3/process_data_semantic_pointwise_actorseg_hybrid.sh` (modified)
  - `policy/DP3/eval_semantic_pointwise_actorseg_hybrid.sh` (modified)
  - `findings.md` (updated)
  - `progress.md` (updated)

### Phase 8: Pointwise Eval Interface Unification
- **Status:** complete
- Actions taken:
  - Added a dedicated regression test `policy/DP3/scripts/test_eval_pointwise_script_interfaces.sh` to capture the final `python script/eval_policy.py` argv emitted by the pointwise eval scripts.
  - Verified the new test failed against the old mixed interface, specifically because some scripts still auto-derived `ckpt_setting` instead of accepting it as the third positional argument.
  - Rewrote the pointwise eval scripts to use the fixed positional layout requested by the user: `task_name task_config ckpt_setting expert_data_num seed gpu_id ...`.
  - Removed old dual-signature/legacy compatibility branches from the touched eval scripts.
  - Updated the regression test to assert that the third argument is forwarded unchanged as `--ckpt_setting`.
  - Re-ran the regression test and shell syntax checks.
- Files created/modified:
  - `policy/DP3/eval_ndf_pointwise.sh` (modified)
  - `policy/DP3/eval_ndf_pointwise_hybrid.sh` (modified)
  - `policy/DP3/eval_ndf_pointwise_actorseg_hybrid.sh` (modified)
  - `policy/DP3/eval_semantic_pointwise.sh` (modified)
  - `policy/DP3/eval_semantic_pointwise_actorseg_hybrid.sh` (modified)
  - `policy/DP3/scripts/test_eval_pointwise_script_interfaces.sh` (new)
  - `findings.md` (updated)
  - `progress.md` (updated)

### Phase 9: ObjPC vs ActorSeg Comparison
- **Status:** complete
- Actions taken:
  - Compared `demo_clean_3d_object_pc.yml` and `demo_randomized_3d_object_pc.yml` to isolate which randomized factors change beyond textures and lighting.
  - Traced DP3 observation assembly in `policy/DP3/deploy_policy.py` to confirm that `agent_pos` is always present and `point_cloud` comes from either `object_pointcloud` or actorseg extraction depending on the path.
  - Verified in `envs/camera/camera.py` that `object_pointcloud` is built from raw per-camera `Position` buffers filtered by actor-id segmentation before per-object FPS/downsampling.
  - Verified in `policy/DP3/scripts/actorseg_pointcloud_utils.py` that actorseg extraction instead starts from the already-downsampled combined scene `episode["pointcloud"]` / `observation["pointcloud"]`, then projects that sparse cloud into actor-segmentation masks.
  - Concluded that the current actorseg path is structurally weaker than objpc, so they are not expected to match even with simulator truth masks.
- Files created/modified:
  - `findings.md` (updated)
  - `progress.md` (updated)

### Phase 10: Workspace Anchor And Cropped Real-ZED Capture
- **Status:** complete with hardware run pending
- Actions taken:
  - Re-read the existing real-ZED calibration, collection, export, and postprocess scripts.
  - Confirmed current collection saves full-resolution depth and downsampled RGB for all cameras, with no live crop path.
  - Added a new plan phase for explicit workspace-anchor calibration and cropped capture metadata.
  - Added workspace crop utility tests first and verified they failed before implementation.
  - Implemented workspace-frame loading, 3D bbox projection to camera ROI, crop-adjusted intrinsics, and cropped frame metadata.
  - Added `calibrate_workspace_frame.py` to define workspace/world from a placed Charuco board and report calibration residuals.
  - Updated `collect_zed_robotwin_raw.py` to save cropped RGB-D plus workspace metadata when `workspace_crop_enabled=true`.
  - Updated postprocess/export tools to prefer workspace calibration paths when `frame_mode=workspace`.
  - Added workspace crop arguments to HDF5 postprocess so already-collected full raw data can be re-exported in the new workspace frame without being discarded.
  - Updated workspace frame convention so tabletop workspace +Z is Charuco board -Z, keeping bbox z positive above the table.
  - Added serial-based camera calibration remapping for old and future data where raw labels and calibration labels differ.
  - Corrected the real-ZED collection config serial order to match the calibration labels.
  - Re-exported a fixed workspace preview under `outputs/real_zed_collection/previews/episode_20260425164903_fixed_workspace`.
  - Verified with `python -m unittest script/test_real_zed_collection_pipeline.py`, `python -m py_compile ...`, and script `--help` checks.
- Files created/modified:
  - `task_plan.md` (updated)
  - `findings.md` (updated)
  - `progress.md` (updated)
  - `script/real_zed_collection/workspace_crop_utils.py` (new)
  - `script/real_zed_collection/calibrate_workspace_frame.py` (new)
  - `script/real_zed_collection/collect_zed_robotwin_raw.py` (modified)
  - `script/real_zed_collection/real_zed_utils.py` (modified)
  - `script/real_zed_collection/postprocess_raw_to_robotwin_hdf5.py` (modified)
  - `script/real_zed_collection/export_raw_episode_pointcloud.py` (modified)
  - `script/real_zed_collection/configs/real_zed_collection.yaml` (modified)
  - `script/real_zed_collection/README.md` (modified)
  - `script/test_real_zed_collection_pipeline.py` (modified)

### Phase 11: Real SAM3 ObjPC Postprocessing
- **Status:** complete with full dataset run pending
- Actions taken:
  - Confirmed the collected raw dataset is at `/media/zheng/Extreme SSD/geo_mani_data/grasp_mug/real_zed_raw` with 61 episodes and three raw camera labels: `global,left,right`.
  - Confirmed the local SAM3 checkpoint exists at `/home/zheng/Datasets/sam3/sam3.pt`; the previously configured SAM2 checkpoint path does not exist on this machine.
  - Chose SAM3 for the first real-data pipeline so offline postprocessing and online inference can share the same prompt+bbox tracking behavior.
  - Added the first offline SAM3 episode mask generator scaffold at `script/real_zed_collection/segment_objects_sam3.py`.
  - Added `script/real_zed_collection/postprocess_real_zed_sam3_objpc_dataset.py` to generate SAM3 masks and write compact `train_objpc.sh`-compatible HDF5 episodes.
  - Added optional compact HDF5 output in `postprocess_raw_to_robotwin_hdf5.py` so real objpc training data can omit duplicated RGB-D observations by default.
  - Added an explicit SAM3 device override path; `--sam3_device auto` resolves to CUDA when available, otherwise CPU.
  - Verified unit coverage with `python -m unittest script.test_real_zed_collection_pipeline`.
  - Verified syntax with `python -m py_compile policy/DP3/scripts/sam3_pointcloud_utils.py script/real_zed_collection/segment_objects_sam3.py script/real_zed_collection/postprocess_real_zed_sam3_objpc_dataset.py script/real_zed_collection/postprocess_raw_to_robotwin_hdf5.py`.
  - Ran a compact 1-frame postprocess smoke and confirmed the HDF5 only contains `joint_action`, `pointcloud`, and `object_pointcloud`.
  - Ran a 1-frame SAM3 smoke; masks and object point clouds were written under `data/grasp_mug/demo_real_zed_sam3_objpc_sam3_smoke`.
  - Ran a 2-frame pipeline smoke through `policy/DP3/process_data_objpc.sh`; zarr output shape was `(1, 1024, 6)` for `data/point_cloud`.

### Phase 12: SSD Output And Debug Preview Mode
- **Status:** complete
- Actions taken:
  - Moved the real-SAM3 objpc batch script under `script/real_zed_collection/postprocess/`.
  - Changed its default output path to `/media/${USER}/Extreme SSD/geo_mani_data/<task>/robotwin_objpc/<task_config>`.
  - Added automatic repo data symlink creation at `data/<task>/<task_config>` for compatibility with `policy/DP3/train_objpc.sh`.
  - Added `--debug`, `--debug_stride`, and `--debug_max_frames`.
  - Added debug mask overlays under `debug/episode*/mask_overlays/<placeholder>/<camera>/overlay_*.png`.
  - Added merged colored `{A}/{B}` point cloud previews under `debug/episode*/pointclouds/frame_*_objects_ab.ply`.
  - Verified with a 1-frame SAM3 debug smoke: 6 overlay PNGs were written and the merged PLY contained 2048 vertices.

### Phase 13: Workspace-Constrained SAM Mask Generation
- **Status:** complete
- Actions taken:
  - Added `apply_mask_roi(...)` to zero SAM mask pixels outside a per-camera RGB ROI before saving.
  - Added automatic workspace-bbox projection to RGB ROI, with camera-matrix scaling for old recordings where RGB is 640x360 and depth is 1920x1080.
  - Added `--mask_roi_xyxy` for manual per-camera RGB ROI override.
  - Added per-frame depth-based workspace mask filtering in `segment_objects_sam3.py`; saved SAM masks now only keep pixels whose depth point transforms inside the workspace bbox.
  - Verified on a 1-frame SAM3 smoke that `workspace_filter=True` is written for all camera/object masks and the debug PLY still contains 2048 vertices.

### Phase 14: Interactive 2D Workspace Mask Mode
- **Status:** complete
- Actions taken:
  - Added `script/real_zed_collection/select_camera_workspace_masks.py` to display each camera's first raw RGB frame and let the user click a polygon.
  - The selector saves `workspace_mask.png`, `workspace_overlay.png`, and `workspace_masks_meta.json`.
  - Added `apply_mask_domain(...)` to restrict SAM masks to the clicked per-camera polygon mask.
  - Changed SAM3 inference in polygon mode so it crops to the polygon bbox and zeros pixels outside the polygon before calling the tracker, rather than only intersecting the output mask afterward.
  - Added cache invalidation for old non-polygon SAM masks so polygon mode regenerates them instead of reusing stale post-intersection masks.
  - Added `--camera_workspace_mask_root` to the SAM3 objpc batch postprocess path.
  - Added point-cloud reconstruction support for the same per-camera 2D masks in `postprocess_raw_to_robotwin_hdf5.py`.
  - When `--camera_workspace_mask_root` is set, depth workspace filtering is disabled by default; `--also_depth_workspace_filter` opts it back in.
  - Verified full tests with `python -m unittest script.test_real_zed_collection_pipeline` (`19` tests).
  - Verified syntax with `python -m py_compile script/real_zed_collection/select_camera_workspace_masks.py script/real_zed_collection/segment_objects_sam3.py script/real_zed_collection/postprocess/postprocess_real_zed_sam3_objpc_dataset.py script/real_zed_collection/postprocess/postprocess_raw_to_robotwin_hdf5.py script/test_real_zed_collection_pipeline.py`.
  - Verified a no-SAM smoke with synthetic per-camera masks: meta recorded all three mask labels and `workspace_mask_filter_enabled=False`.

### Phase 15: SAM3 Mask Quality Check
- **Status:** complete
- Actions taken:
  - Inspected existing smoke outputs and confirmed older previews had `sam_input_domain=0`, so they were not true polygon-input SAM results.
  - Ran a fresh 1-frame SAM3 check using `/media/zheng/Extreme SSD/geo_mani_data/grasp_mug/camera_workspace_masks` and confirmed all six camera/object entries have `sam_input_domain=True`.
  - Generated debug contact sheets:
    - `outputs/real_zed_collection/mask_debug/polygon_input_sam3_check/polygon_sam_debug_contact_sheet.png`
    - `outputs/real_zed_collection/mask_debug/polygon_input_sam3_check/prompt_compare_contact_sheet.png`
  - Compared `{A}:mug,{B}:box` against `{A}:cup,{B}:box`; `cup` recovered the left-camera `{A}` mask that `mug` missed.
  - Updated the real SAM3 batch default prompt to `{A}:cup,{B}:box` and aligned the standalone SAM3 script default confidence to `0.2`.
  - Added prompt-aware mask-cache invalidation so rerunning with a changed prompt regenerates masks even when existing masks were already generated with polygon input-domain mode.
  - Verified with `python -m py_compile script/real_zed_collection/segment_objects_sam3.py script/real_zed_collection/postprocess/postprocess_real_zed_sam3_objpc_dataset.py script/test_real_zed_collection_pipeline.py`.
  - Verified with `python -m unittest script.test_real_zed_collection_pipeline` (`20` tests).

### Phase 16: SAM2 Streaming Migration Discovery
- **Status:** in_progress
- Actions taken:
  - Read `include/SAM2_streaming/demo_webcam_box.py` and confirmed the manual bbox flow: draw a box, `load_first_frame`, `add_new_prompt`, then `track` each frame.
  - Read `include/SAM2_streaming/sam2/build_sam.py` and `sam2_camera_predictor.py` to identify the stable callable API for a wrapper.
  - Confirmed local SAM2 checkpoint exists at `/home/zheng/Datasets/sam2/sam2.1_hiera_large.pt`.
  - Confirmed matching config exists under `include/SAM2_streaming/configs/sam2.1/sam2.1_hiera_l.yaml`.
  - Confirmed current Python environment can import `sam2.build_sam.build_sam2_camera_predictor` and has Hydra/OmegaConf.
  - Mapped current SAM3 dependency surfaces: real postprocess scripts, DP3 `objpc_sam3` preprocessing/eval scripts, `sam3_pointcloud_utils.py`, and `deploy_policy.py` online eval branch.

### Phase 17: SAM2 Streaming Real ObjPC Path
- **Status:** complete with GPU runtime run pending
- Actions taken:
  - Added `script/real_zed_collection/sam2_tracking_utils.py`, a small wrapper around `build_sam2_camera_predictor`, `load_first_frame`, `add_new_prompt`, and `track`.
  - Added `script/real_zed_collection/select_sam2_bboxes.py` for manually selecting first-frame `{A}/{B}` bboxes in each camera.
  - Added `script/real_zed_collection/segment_objects_sam2.py` to run one SAM2 streaming tracker per camera and write masks in the same per-placeholder/per-camera layout used by the real objpc postprocess.
  - Added `script/real_zed_collection/postprocess/postprocess_real_zed_sam2_objpc_dataset.py` as the active real-data batch path; it writes compact train_objpc-compatible HDF5 and creates the repo-side `data/<task>/<task_config>` symlink.
  - Updated `script/real_zed_collection/README.md` so the active real-data flow is workspace polygon mask selection, SAM2 bbox selection, SAM2 tracking masks, and `demo_real_zed_sam2_objpc` conversion/training.
  - Added SAM2 unit coverage to `script/test_real_zed_collection_pipeline.py` for bbox prompt IO, SAM2 logit-to-placeholder mapping, fake-tracker mask writing, and SAM2 default output paths.
  - Added `policy/DP3/scripts/sam2_pointcloud_utils.py` for online SAM2 mask-to-scene-pointcloud projection during eval.
  - Added `policy/DP3/scripts/test_sam2_pointcloud_utils.py` to verify one-pass A/B tracking initialization, subsequent tracking without reinitialization, and bbox prompt filtering.
  - Updated `policy/DP3/deploy_policy.py` with a separate `objpc_sam2` branch, so online SAM2 eval can build the main DP3 `point_cloud` from SAM2-tracked masks without changing old objpc/SAM3 paths.
  - Added `policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/robot_dp3_objpc_sam2.yaml` and `policy/DP3/eval_objpc_sam2.sh`.
  - Verified SAM2 checkpoint/config discovery with shell checks:
    - `/home/zheng/Datasets/sam2/sam2.1_hiera_large.pt` exists.
    - `include/SAM2_streaming/configs/sam2.1/sam2.1_hiera_l.yaml` exists.
  - Verified with `python -m unittest script.test_real_zed_collection_pipeline`.
  - Verified with `python policy/DP3/scripts/test_sam2_pointcloud_utils.py`.
  - Verified syntax with `python -m py_compile script/real_zed_collection/sam2_tracking_utils.py script/real_zed_collection/select_sam2_bboxes.py script/real_zed_collection/segment_objects_sam2.py script/real_zed_collection/postprocess/postprocess_real_zed_sam2_objpc_dataset.py script/test_real_zed_collection_pipeline.py`.
  - Verified syntax with `python -m py_compile policy/DP3/deploy_policy.py policy/DP3/scripts/sam2_pointcloud_utils.py policy/DP3/scripts/test_sam2_pointcloud_utils.py`.
  - Verified `robot_dp3_objpc_sam2.yaml` composes with Hydra and keeps `point_cloud` shape `[1024, 6]`.
  - Verified `bash -n policy/DP3/eval_objpc_sam2.sh`.
  - Verified CLI argument surfaces with `--help` for `select_sam2_bboxes.py`, `segment_objects_sam2.py`, and `postprocess_real_zed_sam2_objpc_dataset.py`.
  - Verified active SAM2 files contain no direct `sam3` / `SAM3` / `ultralytics` references.
  - Did not run the actual SAM2 model in this shell because `torch.cuda.is_available()` is false and the upstream streaming predictor stores state on CUDA.

### Phase 18: SAM2 CUDA SDPA Autocast Fix
- **Status:** complete with user-side GPU rerun pending
- Actions taken:
  - Diagnosed the user traceback from `SAM2_streaming/sam2/modeling/sam/transformer.py` as a CUDA SDPA kernel-selection failure, not a data-format or bbox-prompt failure.
  - Confirmed the warnings showed the root cause: Q/K/V were float32, while available fast attention kernels require `Half` or `BFloat16`; the upstream SAM2 webcam demo uses bf16 autocast.
  - Added a failing regression test proving `SAM2StreamingObjectTracker` should run predictor calls inside autocast.
  - Updated `script/real_zed_collection/sam2_tracking_utils.py` so `load_first_frame`, `add_new_prompt`, and `track` run under `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` by default.
  - Enabled CUDA Flash/Memory Efficient/Math SDP kernels in the SAM2 checkpoint loader where the installed PyTorch exposes those switches.
  - Added `--sam2_autocast_dtype` to `segment_objects_sam2.py`, `postprocess_real_zed_sam2_objpc_dataset.py`, `eval_objpc_sam2.sh`, and the DP3 online SAM2 tracker factory path.
  - Documented the bf16 default and `float16` fallback in `script/real_zed_collection/README.md`.
  - Verified with `python -m unittest script.test_real_zed_collection_pipeline` (`25` tests).
  - Verified with `python policy/DP3/scripts/test_sam2_pointcloud_utils.py` (`3` tests).
  - Verified syntax with `python -m py_compile script/real_zed_collection/sam2_tracking_utils.py script/real_zed_collection/segment_objects_sam2.py script/real_zed_collection/postprocess/postprocess_real_zed_sam2_objpc_dataset.py policy/DP3/scripts/sam2_pointcloud_utils.py policy/DP3/deploy_policy.py script/test_real_zed_collection_pipeline.py`.
  - Verified `bash -n policy/DP3/eval_objpc_sam2.sh`.

### Phase 19: SAM2 Per-Episode Bbox Prompts
- **Status:** complete
- Actions taken:
  - Confirmed the user's concern: in data processing, SAM2 tracking should be reinitialized per recorded episode because object poses can differ across episodes.
  - Confirmed the tracker itself was already recreated inside `segment_episode_sam2(...)`, but the batch driver reused one global bbox prompt file for all raw episodes.
  - Added per-episode bbox prompt resolution to `postprocess_real_zed_sam2_objpc_dataset.py`.
  - The lookup priority is now:
    - `sam2_bbox_prompts/<raw_episode_name>/sam2_bbox_prompts.json`
    - `sam2_bbox_prompts/episode<processed_index>/sam2_bbox_prompts.json`
    - `sam2_bbox_prompts/episode_<processed_index:06d>/sam2_bbox_prompts.json`
    - fallback `sam2_bbox_prompts/sam2_bbox_prompts.json`
  - Added `--require_per_episode_bboxes` so production processing can fail fast instead of silently using a global prompt.
  - Added `--all_episodes`, `--per_episode_subdir`, and `--skip_existing` to `select_sam2_bboxes.py`.
  - Updated README instructions to use `--all_episodes --skip_existing` and `--require_per_episode_bboxes`.
  - Added regression coverage for per-episode prompt priority and required-per-episode behavior.
  - Verified targeted tests with `python -m unittest script.test_real_zed_collection_pipeline.RealZedCollectionPipelineTest.test_sam2_objpc_batch_prefers_per_episode_bbox_prompts script.test_real_zed_collection_pipeline.RealZedCollectionPipelineTest.test_sam2_objpc_batch_can_require_per_episode_bbox_prompts`.
  - Verified syntax with `python -m py_compile script/real_zed_collection/select_sam2_bboxes.py script/real_zed_collection/postprocess/postprocess_real_zed_sam2_objpc_dataset.py`.

### Phase 20: Legacy SAM/SAM3 Cleanup
- **Status:** complete
- Actions taken:
  - Removed old real-data SAM/SAM3 scripts and the DP3 `objpc_sam3` preprocessing/training/eval/config files.
  - Removed the SAM3 online eval branch from `policy/DP3/deploy_policy.py`.
  - Removed SAM3-specific unit tests and stale generic SAM documentation.
  - Verified active source search under `policy`, `script`, and `envs` has no `sam3`, `ultralytics`, `objpc_sam3`, or `segment_objects_sam.py` references.
  - Verified with `python -m unittest script.test_real_zed_collection_pipeline` (`14` tests).
  - Verified with `python policy/DP3/scripts/test_sam2_pointcloud_utils.py` (`3` tests).
  - Verified with `python policy/DP3/scripts/test_ndf_pointwise_hybrid.py`, `python policy/DP3/scripts/test_semantic_pointwise_hybrid.py`, `python policy/DP3/scripts/test_utonia_pointwise_hybrid.py`, and `python policy/DP3/scripts/test_actorseg_pointwise_hybrid.py`.
  - Verified syntax with `python -m py_compile` for the active SAM2/deploy/test files and `bash -n policy/DP3/eval_objpc_sam2.sh`.
  - Verified Hydra composition for `robot_dp3_objpc_sam2` and confirmed `point_cloud` shape remains `[1024, 6]`.

### Phase 21: SAM2 Prompt Selection UX
- **Status:** complete
- Actions taken:
  - Fixed accidental single-click/zero-area bbox selection so it no longer raises during rendering.
  - Extended SAM2 prompt files to store either bbox prompts or point prompts while keeping old bbox-only JSON readable.
  - Added selector key controls: `m` toggles bbox/point mode, left click adds a foreground point in point mode, and right click adds a background point.
  - Added first-frame SAM2 preview overlay in the selector, with `--disable_sam2_preview` for prompt-only operation.
  - Updated segment/postprocess/eval prompt loading so point prompt records can initialize SAM2 tracking.
  - Verified with `python -m unittest script.test_real_zed_collection_pipeline` and `python policy/DP3/scripts/test_sam2_pointcloud_utils.py`.

### Phase 22: Real-ZED DP Image Baseline
- **Status:** complete
- Actions taken:
  - Added `policy/DP/process_data_real_zed.py`, `policy/DP/process_data_real_zed.sh`, and `policy/DP/train_real_zed.sh` for a single-camera real-ZED DP image baseline.
  - Extended `process_data_real_zed.py` with `--camera_labels` and explicit raw-to-zarr mapping for `global,left,right`.
  - Added `policy/DP/diffusion_policy/dataset/robot_multi_image_dataset.py` for three-camera zarr loading and `head_cam/left_cam/right_cam` postprocess output.
  - Added `policy/DP/diffusion_policy/config/task/default_task_14_multicam.yaml` and `default_task_16_multicam.yaml`.
  - Updated `policy/DP/train.py` so every RGB obs shape is set from `head_camera_type`, not only `head_cam`.
  - Added `policy/DP/process_data_real_zed_multicam.sh` and `policy/DP/train_real_zed_multicam.sh`; the multicam train wrapper exposes `batch_size`, `val_batch_size`, and `gradient_accumulate_every` for GPU memory control.
  - Added a unit test that builds a synthetic three-camera raw episode and verifies zarr keys/shapes.
  - Verified with `python -m py_compile policy/DP/process_data_real_zed.py policy/DP/train.py policy/DP/diffusion_policy/dataset/robot_multi_image_dataset.py script/test_real_zed_collection_pipeline.py`.
  - Verified with `bash -n policy/DP/process_data_real_zed.sh policy/DP/train_real_zed.sh policy/DP/process_data_real_zed_multicam.sh policy/DP/train_real_zed_multicam.sh`.
  - Verified with `python -m unittest script.test_real_zed_collection_pipeline`.
  - Verified Hydra composition for both 14D and 16D multicam configs; both expose `head_cam`, `left_cam`, `right_cam`, and `agent_pos` through `RobotMultiImageDataset`.
  - Verified real-data smoke conversion for one episode:
    `python process_data_real_zed.py grasp_mug demo_real_zed_sam2_objpc 1 --camera_labels global,left,right --output_zarr /tmp/dp_real_zed_multicam_smoke.zarr`.

### Phase 23: DP Training Comparison With xtrainer
- **Status:** complete
- Actions taken:
  - Read RoboTwin `policy/DP` training entrypoints, preprocessors, datasets, config, image encoder, policy, and workspace loop.
  - Read xtrainer `ModelTrain/dp` pipeline, dataset, learner, data processing, and model code.
  - Identified `ModelTrain/model_train.py` as a separate ACT/DETR-style entrypoint, not the `dp/` diffusion-policy training path.
  - Summarized differences in data format, camera handling, image preprocessing, model feature dimensions, horizons, optimization loop, and checkpointing.

### Phase 24: Real DP3 Inference Scripts
- **Status:** complete with hardware run pending
- Actions taken:
  - Started implementation for real-robot inference scripts covering the user's trained DP3 baseline and semantic-pointwise-hybrid models.
  - Restored planning context and added Phase 14 to `task_plan.md`.
  - Read xtrainer real inference control flow and confirmed the 14D joint-action execution semantics.
  - Read DP3 deploy/model runner interfaces and confirmed the real script can call `get_model`, `encode_obs`, and `model.get_action(encoded_obs)`.
  - Read existing real-ZED capture/calibration utilities and SAM2 online point-cloud projection helpers for reuse.
  - Added `script/real_zed_inference/real_dp3_inference.py`.
  - Added `policy/DP3/real_infer_baseline.sh` for the user's `train.sh` DP3 baseline checkpoint.
  - Added `policy/DP3/real_infer_semantic_pointwise_hybrid.sh` for the user's semantic-pointwise-hybrid checkpoint and semantic model path.
  - Fixed wrapper root-path resolution after `--help` exposed that repo-root invocation jumped to `/home/zheng`.
  - Verified with `python -m py_compile script/real_zed_inference/real_dp3_inference.py`.
  - Verified with `bash -n policy/DP3/real_infer_baseline.sh policy/DP3/real_infer_semantic_pointwise_hybrid.sh`.
  - Verified CLI wiring with both shell wrappers using `--help`.
  - Did not run hardware execution in this session.
  - Diagnosed control-machine dependency failures as a `deploy_policy.py` top-level simulator import problem.
  - Changed `sapien.core` and `envs` imports in `policy/DP3/deploy_policy.py` to optional imports so real inference can load DP3 without full simulator dependencies.
  - Verified `deploy_policy` imports when `sapien` and `envs` are deliberately blocked by a temporary import hook.
  - Confirmed the user's follow-up: `ndf_feature_utils` was also being imported at real inference import time.
  - Moved direct NDF imports in `policy/DP3/deploy_policy.py` behind `get_ndf_utils()`.
  - Moved indirect NDF imports in `policy/DP3/scripts/object_pointcloud_utils.py` into the specific fallback functions that need them.
  - Verified baseline-style `encode_obs` still works while `sapien`, `envs`, and `ndf_feature_utils` are all deliberately blocked.

### Phase 25: Robot-Camera Calibration Design
- **Status:** design proposed
- Actions taken:
  - Read xtrainer `run_control.py` and current `collect_zed_robotwin_raw.py` button/servo semantics.
  - Confirmed Button B rising edge is the correct interaction to repurpose as "capture one calibration sample".
  - Confirmed xtrainer Dobot `ee_pos_quat` is zero-filled, so calibration must use `get_XYZrxryrz_state()`.
  - Verified OpenCV has `aruco` AprilTag dictionary support and `calibrateRobotWorldHandEye`.
  - Derived the correct target-on-gripper mapping for solving `T_base_from_camera` from held-AprilTag samples.

### Phase 26: Robot-Camera AprilTag Calibration Script
- **Status:** complete with hardware run pending
- Actions taken:
  - Added `script/test_robot_camera_apriltag_calibration.py` with a synthetic target-on-gripper calibration problem.
  - Verified the new test failed before the calibration module existed.
  - Added `script/real_zed_collection/calibrate_robot_camera_apriltag.py`.
  - Implemented reusable math helpers, Dobot pose conversion, AprilTag detection, OpenCV robot-world-hand-eye solving, residual reporting, YAML output, and an interactive ZED + xtrainer robot collection loop.
  - Kept Button A lock/servo semantics and Button B sample capture, with the existing collection-script servo safety checks.
  - Verified with `python -m unittest script.test_robot_camera_apriltag_calibration`.
  - Verified with `python -m py_compile script/real_zed_collection/calibrate_robot_camera_apriltag.py script/test_robot_camera_apriltag_calibration.py`.
  - Verified CLI wiring with `python script/real_zed_collection/calibrate_robot_camera_apriltag.py --help`.
  - Updated the marker detector to support generic OpenCV ArUco dictionaries through `--marker_dictionary`.
  - Added `--marker_dictionary auto` so a single-face ArUco marker with only id 4 can be tried without knowing the exact 4x4/5x5/6x6 family upfront.
  - Added unit coverage for dictionary alias normalization and auto dictionary expansion.
  - Re-verified with `python -m unittest script.test_robot_camera_apriltag_calibration`, `python -m py_compile script/real_zed_collection/calibrate_robot_camera_apriltag.py script/test_robot_camera_apriltag_calibration.py`, and `python script/real_zed_collection/calibrate_robot_camera_apriltag.py --help`.
  - Added config-driven camera label/serial mapping to `calibrate_three_zed_extrinsics.py` and `calibrate_robot_camera_apriltag.py`.
  - Added tests confirming `real_zed_collection.yaml` maps `global -> 38968158`, `left -> 31021548`, and `right -> 37856216`, and that robot-camera calibration prefers that mapping over stale calibration YAML serials.
  - Verified with `python -m unittest script.test_robot_camera_apriltag_calibration`, `python -m py_compile script/real_zed_collection/calibrate_three_zed_extrinsics.py script/real_zed_collection/calibrate_robot_camera_apriltag.py script/test_robot_camera_apriltag_calibration.py`, both calibration `--help` commands, and a direct serial-resolution smoke print.
  - Added preview-window bounds to `calibrate_robot_camera_apriltag.py` via `--window_width` and `--window_height`, defaulting to `1280x720`.
  - Added unit coverage that the display resize preserves aspect ratio and does not upscale smaller frames.
  - Re-verified with `python -m unittest script.test_robot_camera_apriltag_calibration`, `python -m py_compile script/real_zed_collection/calibrate_robot_camera_apriltag.py script/test_robot_camera_apriltag_calibration.py`, and `python script/real_zed_collection/calibrate_robot_camera_apriltag.py --help`.
  - Changed robot-camera calibration so ZED frame capture and marker detection run in a background worker thread, while the robot control loop reads the latest snapshot.
  - Added `LatestCameraDetection`, `CameraDetectionSnapshot`, and `start_camera_detection_worker` with a unit test using fake capture/detection functions.
  - Sample metadata now records `camera_timestamp_unix_sec` and `camera_frame_age_sec` so asynchronous capture latency can be inspected.
  - Re-verified with `python -m unittest script.test_robot_camera_apriltag_calibration`, `python -m py_compile script/real_zed_collection/calibrate_robot_camera_apriltag.py script/test_robot_camera_apriltag_calibration.py`, and `python script/real_zed_collection/calibrate_robot_camera_apriltag.py --help`.

### Phase 27: Robot-Base-Frame Real Point Clouds
- **Status:** complete with hardware/postprocess run pending
- Actions taken:
  - Added `--output_frame source|workspace|left_base|right_base` and `--robot_camera_calibration_path` to `postprocess_raw_to_robotwin_hdf5.py`.
  - Added the same `--output_frame` and robot-camera calibration path support to the SAM2 dataset postprocess driver.
  - Added matching `--output_frame` support to `script/real_zed_inference/real_dp3_inference.py`.
  - Implemented composition from multi-camera source/workspace frame into selected arm base frame using the robot-camera calibration YAML.
  - Updated HDF5 `cam2world_gl` and `extrinsic_cv` so projection metadata matches the transformed training point-cloud frame.
  - Added a synthetic test proving that `output_frame=left_base` shifts `/pointcloud` and camera metadata into the left-base transform.
  - Verified with targeted postprocess tests:
    `python -m unittest script.test_real_zed_collection_pipeline.RealZedCollectionPipelineTest.test_postprocess_can_write_pointclouds_in_left_base_frame script.test_real_zed_collection_pipeline.RealZedCollectionPipelineTest.test_postprocess_writes_robotwin_hdf5_from_raw_episode_and_masks`.
  - Verified syntax with `python -m py_compile script/real_zed_collection/postprocess/postprocess_raw_to_robotwin_hdf5.py script/real_zed_collection/postprocess/postprocess_real_zed_sam2_objpc_dataset.py script/real_zed_inference/real_dp3_inference.py script/test_real_zed_collection_pipeline.py`.
  - Verified CLI wiring with `python script/real_zed_collection/postprocess/postprocess_real_zed_sam2_objpc_dataset.py --help` and `python script/real_zed_inference/real_dp3_inference.py --help`.
  - Full `python -m unittest script.test_real_zed_collection_pipeline` still cannot complete in the current shell because `policy/DP/process_data_real_zed.py` imports missing dependency `zarr`.

### Phase 28: Real-ZED Direct Script Import Bootstrap
- **Status:** complete
- Actions taken:
  - Reproduced `ModuleNotFoundError: No module named 'script'` by running `select_sam2_bboxes.py --help` from a non-repository cwd in a regression test.
  - Added repo-root `sys.path` bootstrap to the direct-run real-ZED collection entry scripts before their `script.real_zed_collection.*` imports.
  - Removed the stale duplicate `REPO_ROOT = parents[3]` assignment in `segment_objects_sam2.py` so repo data links continue to resolve under this repository.
  - Verified with `python -m unittest script.test_real_zed_collection_pipeline.RealZedCollectionPipelineTest.test_sam2_bbox_selector_direct_script_help_from_non_repo_cwd`.
  - Verified syntax with `python -m py_compile` across the patched real-ZED entry scripts and test file.
  - Verified direct launch from `/tmp` with `python /home/zheng/github/RoboTwin_geo/script/real_zed_collection/select_sam2_bboxes.py --help`.

### Phase 29: Real-ZED Inference Coordinate-Frame Interface
- **Status:** complete
- Actions taken:
  - Added `policy/DP3/real_infer_arg_utils.sh` to resolve `output_frame` from explicit args, dataset meta, or task-config naming fallback.
  - Updated baseline and semantic real inference scripts so `output_frame` and robot-camera calibration are explicit positional interface fields before passthrough flags.
  - Kept backward-compatible passthrough behavior when the first optional argument starts with `--`.
  - Added `policy/DP3/scripts/test_real_infer_script_interfaces.sh` with fake-Python argument capture for baseline, semantic, explicit workspace override, and task-config-name fallback.
  - Verified with `bash policy/DP3/scripts/test_real_infer_script_interfaces.sh`.
  - Verified shell syntax with `bash -n policy/DP3/real_infer_arg_utils.sh policy/DP3/real_infer_baseline.sh policy/DP3/real_infer_semantic_pointwise_hybrid.sh policy/DP3/scripts/test_real_infer_script_interfaces.sh`.
  - Verified `demo_real_zed_sam2_objpc_rightbase` auto-resolves to `right_base` and the default right-arm robot-camera calibration path.

### Phase 30: Real-ZED Action Step Limiting
- **Status:** complete
- Actions taken:
  - Added `script/test_real_zed_inference_actions.py` covering per-joint and gripper action-delta clipping.
  - Verified the test failed before implementation because `limit_action_delta_for_execution` did not exist.
  - Added `limit_action_delta_for_execution` to real DP3 inference and applied it before action-delta safety checks and robot execution.
  - Added CLI controls `--max_executed_joint_delta` and `--max_executed_gripper_delta`, defaulting to `0.12` and `0.2`.
  - Verified with `python -m unittest script.test_real_zed_inference_actions`.
  - Verified syntax with `python -m py_compile script/real_zed_inference/real_dp3_inference.py script/test_real_zed_inference_actions.py`.
  - Verified CLI visibility with `python script/real_zed_inference/real_dp3_inference.py --help | rg "max_executed|max_action_delta|disable_action_delta"`.

### Phase 31: Portable SAM2 Paths
- **Status:** complete
- Actions taken:
  - Added a failing regression test for SAM2 root/checkpoint fallback when the requested path is machine-specific or missing.
  - Added SAM2 path resolvers that prefer `$SAM2_STREAMING_ROOT` and `$SAM2_CHECKPOINT`, then repository/current-user defaults.
  - Updated real-ZED SAM2 tracker loading to use the resolvers for both root and checkpoint paths.
  - Updated real DP3 inference defaults and the DP3 SAM2 deploy path to avoid hard-coded `/home/zheng` SAM2 checkpoint defaults.
  - Updated `eval_objpc_sam2.sh` to default through `$SAM2_CHECKPOINT` and `$SAM2_STREAMING_ROOT`.
  - Updated real semantic inference defaults to use `$SEMANTIC_CKPT_A` or the current user's `~/github/3d_semantic_train/...` path.
  - Verified with `python -m unittest script.test_real_zed_collection_pipeline.RealZedCollectionPipelineTest.test_sam2_paths_can_fallback_from_machine_specific_paths`.
  - Verified syntax with `python -m py_compile script/real_zed_collection/sam2_tracking_utils.py script/real_zed_inference/real_dp3_inference.py policy/DP3/deploy_policy.py script/test_real_zed_collection_pipeline.py`.
  - Verified shell syntax with `bash -n policy/DP3/eval_objpc_sam2.sh policy/DP3/real_infer_semantic_pointwise_hybrid.sh policy/DP3/real_infer_baseline.sh`.
  - Verified no remaining hard-coded `/home/zheng` SAM2 or real semantic checkpoint paths with `rg -n "/home/zheng/.+sam2|/home/zheng/github/SAM2_streaming|/home/zheng/github/3d_semantic_train" script policy -g '*.py' -g '*.sh'`.

### Phase 32: SAM2 Online BBox UI Robustness
- **Status:** complete
- Actions taken:
  - Added a regression test for online SAM2 bbox helper behavior on single-click zero-area boxes and HD image display scaling.
  - Verified the test failed before implementation because the helper functions did not exist.
  - Added `_try_normalize_bbox_xyxy`, `_display_scale_for_image`, and `_display_to_image_point` to `sam2_pointcloud_utils.py`.
  - Updated `select_bbox_for_image` to scale HD frames to a 1280x720 display canvas, map mouse coordinates back to original image coordinates, and show invalid-box status instead of raising on single clicks.
  - Verified with `python -m unittest test_sam2_pointcloud_utils` from `policy/DP3/scripts`.
  - Verified syntax with `python -m py_compile policy/DP3/scripts/sam2_pointcloud_utils.py policy/DP3/scripts/test_sam2_pointcloud_utils.py script/real_zed_inference/real_dp3_inference.py`.

### Phase 33: Real Semantic Inference SAM2 Latency Diagnosis
- **Status:** complete
- Actions taken:
  - Compared `real_dp3_inference.py`, `zed_sam2_realtime_bbox.py`, and `include/xtrainer_clover/experiments/run_inference.py`.
  - Confirmed real DP3 inference already uses camera capture threads, so the 15s stall is not caused by missing camera acquisition threading alone.
  - Identified major path differences from the fast SAM2 demo: three cameras instead of one, `HD1080` + `NEURAL` depth instead of `HD720` RGB-only, no 640-wide SAM2 resize by default, and synchronous point-cloud/SAM2/DP3 work in the main loop.
  - Identified the likely dominant code issue: online SAM2 object point-cloud extraction projects the full fused dense scene into every camera mask for every placeholder, rather than directly lifting masked depth pixels per camera.

### Phase 34: Real SAM2 Online Mask-Depth Object Point Clouds
- **Status:** complete with hardware timing pending
- Actions taken:
  - Added failing tests for direct SAM2 `mask + depth` lifting with workspace filtering and for scene point-cloud workspace filtering before robot-base output transforms.
  - Added `camera_frame_to_output_pc` so real scene point clouds are lifted to workspace, cropped by the configured 3D workspace bounds, then transformed into `workspace`, `right_base`, or `left_base`.
  - Updated real observations to carry `t_workspace_from_cam` and `workspace_bounds_m` per camera for online SAM2 object point-cloud extraction.
  - Updated SAM2 online extraction to use depth lifting when camera depth and `cam2world_gl` are available, falling back to the old scene-projection path only when depth data is absent.
  - Added `--sam2_image_width` with default `640` and changed real ZED inference default resolution to `HD720`.
  - Preserved `image_shape_hw` from saved bbox/point prompts so prompt coordinates can be scaled when SAM2 runs on resized images.
  - Updated live workspace ROI cropping to use the current ZED SDK intrinsics instead of calibration-file intrinsics, preventing stale-intrinsic crops when `--zed_resolution` changes.
  - Verified with `python -m unittest script.test_real_zed_inference_actions`, `python -m unittest test_sam2_pointcloud_utils` from `policy/DP3/scripts`, `python -m py_compile ...`, `bash -n ...`, and `python script/real_zed_inference/real_dp3_inference.py --help | rg "zed_resolution|sam2_image_width|output_frame|workspace_crop"`.

### Phase 35: Real SAM2 Object Point-Cloud Preview Script
- **Status:** complete with hardware run pending
- Actions taken:
  - Added `script/real_zed_inference/preview_sam2_object_pointcloud.py`, a standalone live preview that does not load DP3 or connect to the robot.
  - The script reuses live ZED startup, workspace filtering, SAM2 runtime loading, and online `mask + depth` object point-cloud reconstruction.
  - Added Open3D live visualization for `{A}/{B}` object clouds, optional RGB camera preview, optional gray scene cloud, placeholder/RGB color modes, frame timing prints, and robot-base output-frame support.
  - Added single-object CLI support through `--object_placeholders "{A}"`, headless FPS mode through `--no_open3d`, and larger/tunable Open3D display controls.
  - Added `script/test_real_sam2_object_pointcloud_preview.py` covering empty-row filtering and placeholder/RGB coloring without requiring Open3D or hardware.
  - Added a default workspace bounding-box LineSet and `--open3d_initial_zoom` to prevent Open3D from clamping zoom-out around a tiny single-object point cloud.
  - Added preview-specific fine-grained timing for frame snapshot, per-camera scene point-cloud construction, SAM2 tracking, mask-depth object lifting, merge/resample, Open3D rendering, image display, and loop total.
  - Added `fast_merge_object_point_clouds` / `fast_resample_point_cloud` for online SAM2 object clouds and exposed `--online_object_resample fast|fps`; preview and real inference default to `fast`.
  - Verified with `python -m unittest script.test_real_sam2_object_pointcloud_preview`, `python -m py_compile script/real_zed_inference/preview_sam2_object_pointcloud.py script/test_real_sam2_object_pointcloud_preview.py`, CLI help grep, and `python -c "import open3d as o3d; print(o3d.__version__)"`.

### Phase 36: Object-Only Preview And Per-Camera Parallelism
- **Status:** complete with hardware timing pending
- Actions taken:
  - Added RED tests for object-only preview skipping dense scene construction, shared per-camera worker parallelism, lazy scene pointcloud access in online SAM2 extraction, and real/preview CLI worker controls.
  - Added `run_camera_tasks_parallel(...)` to `policy/DP3/scripts/sam2_pointcloud_utils.py`; `max_workers <= 0` uses one worker per camera and `1` forces serial execution.
  - Updated online SAM2 object point-cloud extraction so per-camera tracker resize/init/track and mask-depth lifting run in parallel after any required interactive prompt selection.
  - Updated the standalone preview to default to `--object_only`, skipping `build_obs.<camera>.depth_to_pc`, `build_obs.merge_scene`, and `build_obs.resample_scene` unless `--show_scene` or `--no-object_only` is passed.
  - Updated real DP3 inference to parallelize per-camera scene construction and to pass `--parallel_camera_workers` into online SAM2 object extraction.
  - Verified with `python -m unittest test_sam2_pointcloud_utils` from `policy/DP3/scripts`.
  - Verified with `python -m unittest script.test_real_sam2_object_pointcloud_preview script.test_real_zed_inference_actions`.
  - Verified syntax with `python -m py_compile policy/DP3/scripts/sam2_pointcloud_utils.py script/real_zed_inference/preview_sam2_object_pointcloud.py script/real_zed_inference/real_dp3_inference.py script/test_real_sam2_object_pointcloud_preview.py script/test_real_zed_inference_actions.py`.

### Phase 37: Real Robot Execution Smoothing
- **Status:** complete with hardware tuning pending
- Actions taken:
  - Added RED tests for action substep interpolation, `execute_action(...)` sending multiple robot commands, and parser defaults.
  - Added `build_execution_substeps(...)` to split each clipped target action into smaller joint/gripper commands.
  - Updated `execute_action(...)` so non-initial executed policy actions are sent through substeps before returning the final commanded joints.
  - Added CLI controls `--execution_substeps` and `--execution_substep_sleep_sec`, defaulting to `3` and `0.02`.
  - Verified with `python -m unittest script.test_real_zed_inference_actions`.
  - Verified syntax with `python -m py_compile script/real_zed_inference/real_dp3_inference.py script/test_real_zed_inference_actions.py`.
  - Verified shell syntax with `bash -n policy/DP3/real_infer_baseline.sh policy/DP3/real_infer_semantic_pointwise_hybrid.sh`.
  - Verified CLI visibility with `python script/real_zed_inference/real_dp3_inference.py --help`.

### Phase 38: Dobot ServoJ Controller-Level Smoothing
- **Status:** complete with hardware restart/tuning pending
- Actions taken:
  - Added RED tests for `DobotApiMove.ServoJ(..., gain=...)`, `DobotRobot.command_joint_state(...)` using configured `servo_j_t/gain`, `DobotRobot.set_servo_params(...)` validation, and real inference startup parameter forwarding.
  - Updated `include/xtrainer_clover/dobot_control/robots/dobot_api.py` so `ServoJ` accepts a configurable `gain`.
  - Updated `include/xtrainer_clover/dobot_control/robots/dobot.py` so `DobotRobot` stores `servo_j_t/gain`, validates them, and passes them into `ServoJ`.
  - Added `set_servo_params` through `RobotEnv`, `BimanualRobot`, `ZMQServerRobot`, and `ZMQClientRobot`.
  - Updated real DP3 inference to call `configure_robot_servo_params(...)` after connecting to the robot server, with defaults `--servo_j_t 0.06 --servo_j_gain 300`.
  - Changed client-side execution substep defaults back to `--execution_substeps 1 --execution_substep_sleep_sec 0.0`, leaving substeps as an explicit fallback rather than the default smoothing mechanism.
  - Added a 3s timeout to the ZMQ `set_servo_params` request so an old, non-restarted robot server fails with a clear restart message instead of hanging indefinitely.

### Phase 39: Real Inference Command Acceleration Limiting
- **Status:** complete with hardware tuning pending
- Actions taken:
  - Interpreted the user's real hardware timing/action logs: raw policy deltas frequently reached `0.08-0.16 rad`, while observed execution deltas were usually smaller than commanded deltas, pointing to abrupt output/chunk targets rather than controller amplification.
  - Added a RED unit test for `prepare_action_for_execution(..., previous_command_delta=...)` so command delta ramps by `max_executed_*_delta_change` instead of jumping immediately to the velocity cap.
  - Added `limit_action_delta_change_for_execution(...)` after the existing per-step delta limiter.
  - Added CLI flags `--max_executed_joint_delta_change` and `--max_executed_gripper_delta_change`, defaulting to `0.0` for backward-compatible disabled behavior.
  - Threaded `previous_command_delta` through real inference execution and diagnostics so logs/CSV now include `cmd_arm_delta_change` and `cmd_gripper_delta_change`.
  - Verified with `PYTHONDONTWRITEBYTECODE=1 python -m unittest script.test_real_zed_inference_actions`.
  - Verified syntax with `PYTHONDONTWRITEBYTECODE=1 python -m py_compile script/real_zed_inference/real_dp3_inference.py script/test_real_zed_inference_actions.py`.
  - Verified shell wrappers with `bash -n policy/DP3/real_infer_baseline.sh policy/DP3/real_infer_semantic_pointwise_hybrid.sh`.
  - Verified CLI visibility with `python script/real_zed_inference/real_dp3_inference.py --help | rg -n "max_executed_joint_delta_change|max_executed_gripper_delta_change|reobserve_each_action|action_diagnostics_csv"`.
  - Verified whitespace with `git diff --check`.

### Phase 40: Async Real Robot Action Buffer Control
- **Status:** complete with hardware tuning pending
- Actions taken:
  - Compared the current RoboTwin real DP3 thread layout against xtrainer `run_inference.py`: both use camera capture threads and a main inference/control loop, but RoboTwin's semantic path blocks the main loop for long SAM2/point-cloud/model updates.
  - Added RED tests for an async controller that consumes submitted actions in order and keeps sending the last target when the action buffer is empty.
  - Added `AsyncActionController` to `real_dp3_inference.py`; it owns fixed-rate `env.step(...)`, keeps latest observed joints, applies existing velocity/acceleration limiters, writes diagnostics, and repeats the last target as keepalive during perception/model stalls.
  - Added CLI flags `--async_control`, `--async_control_hz`, `--async_control_max_idle_repeats`, and `--async_control_replace_buffer`.
  - Integrated `--async_control` into `run_real_inference`: the main thread now builds observations and submits DP3 action chunks, while the control thread executes continuously.
  - Kept old synchronous behavior as the default; async control is opt-in.
  - Verified with `PYTHONDONTWRITEBYTECODE=1 python -m unittest script.test_real_zed_inference_actions`.
  - Verified syntax with `PYTHONDONTWRITEBYTECODE=1 python -m py_compile script/real_zed_inference/real_dp3_inference.py script/test_real_zed_inference_actions.py`.
  - Verified shell wrappers with `bash -n policy/DP3/real_infer_baseline.sh policy/DP3/real_infer_semantic_pointwise_hybrid.sh`.
  - Verified CLI visibility with `python script/real_zed_inference/real_dp3_inference.py --help | rg -n "async_control|async_control_hz|async_control_max_idle_repeats|async_control_replace_buffer|max_executed_joint_delta_change"`.
  - Verified whitespace with `git diff --check`.

### Phase 41: Real Inference XYZ Safety Bounds Sync
- **Status:** complete
- Actions taken:
  - Compared real inference XYZ safety against the AprilTag calibration safety path.
  - Confirmed real inference still used the narrower xtrainer `run_inference.py` hard-coded workspace, while AprilTag calibration uses `run_control.py::check_pose_protection`.
  - Added RED tests covering the user's stopped pose and an out-of-range pose.
  - Updated `maybe_check_xyz_safety(...)` to use run-control-style bounds and to include the active limits in the RuntimeError message.
  - Added CLI flags `--xyz_left_x_min`, `--xyz_left_x_max`, `--xyz_right_x_min`, `--xyz_right_x_max`, `--xyz_y_min`, `--xyz_y_max`, `--xyz_left_z_min`, and `--xyz_right_z_min`.
  - Set defaults to left x `[-450,290]`, right x `[-290,450]`, y `[-750,-160]`, left z `>44`, right z `>40` for the current rig/test setting.
  - Verified with `PYTHONDONTWRITEBYTECODE=1 python -m unittest script.test_real_zed_inference_actions`.
  - Verified syntax with `PYTHONDONTWRITEBYTECODE=1 python -m py_compile script/real_zed_inference/real_dp3_inference.py script/test_real_zed_inference_actions.py`.
  - Verified shell wrappers with `bash -n policy/DP3/real_infer_baseline.sh policy/DP3/real_infer_semantic_pointwise_hybrid.sh`.
  - Verified CLI visibility with `python script/real_zed_inference/real_dp3_inference.py --help | rg -n "xyz_left|xyz_right|xyz_y|disable_xyz"`.
  - Verified whitespace with `git diff --check`.

### Phase 42: Real-ZED Offline Postprocess Calibration Auto-Selection
- **Status:** complete
- Actions taken:
  - Inspected `postprocess_real_zed_sam2_objpc_dataset.py`, `postprocess_raw_to_robotwin_hdf5.py`, real raw episode manifests, and the previously generated `scene_info.json`.
  - Confirmed the root cause: the SAM2 batch driver defaulted to repo `three_camera_workspace_extrinsics.yaml`, causing `postprocess_episode(...)` to bypass per-episode `calibration_snapshot.yaml`.
  - Added tests for `resolve_episode_postprocess_settings(...)` so `auto` uses the episode snapshot and only chooses workspace mode when a workspace calibration snapshot exists.
  - Updated the batch driver defaults to `--calibration_path auto --frame_mode auto`, resolves calibration/frame mode per episode, uses resolved settings for postprocess, and writes resolved settings into output metadata.
  - Updated the Real-ZED README command to use `auto` calibration/frame mode.
  - Verified with `python -m unittest script.test_real_zed_collection_pipeline`.
  - Verified syntax with `python -m py_compile script/real_zed_collection/postprocess/postprocess_real_zed_sam2_objpc_dataset.py script/test_real_zed_collection_pipeline.py`.

### Phase 43: Real-ZED Collection Workspace Crop Default
- **Status:** complete
- Actions taken:
  - Updated `script/real_zed_collection/configs/real_zed_collection.yaml` so future raw ZED collection enables workspace crop by default.
  - Set the default crop bounds to x `[-0.35, 0.35]`, y `[-0.25, 0.45]`, z `[0.0, 0.45]` while keeping `workspace_crop_resize_rgb=false`.
  - Verified config loading with a small YAML parse command.
  - Verified syntax with `python -m py_compile script/real_zed_collection/collect_zed_robotwin_raw.py`.

### Phase 44: DP Real-ZED Single-Camera Key Compatibility
- **Status:** complete
- Actions taken:
  - Added a regression test proving single-camera DP preprocessing maps a selected real-ZED camera such as `left` into zarr key `head_camera`, matching `RobotImageDataset`.
  - Updated `policy/DP/process_data_real_zed.py` so any single-camera run writes the selected camera stream as `head_camera`; multi-camera runs still write `head_camera`, `left_camera`, and `right_camera` separately.
  - Updated `policy/DP/train_real_zed.sh` to detect existing incompatible single-camera zarrs that lack `data/head_camera` and rebuild them automatically.
  - Verified with `python -m unittest script.test_real_zed_collection_pipeline`.
  - Verified syntax with `python -m py_compile policy/DP/process_data_real_zed.py script/test_real_zed_collection_pipeline.py` and `bash -n policy/DP/train_real_zed.sh policy/DP/process_data_real_zed.sh policy/DP/train_real_zed_multicam.sh`.

### Phase 45: Portable DP Real-ZED Dataset Packaging
- **Status:** complete
- Actions taken:
  - Updated `policy/DP/process_data_real_zed.py` so zarr metadata records only the source meta filename instead of the local absolute `/media/...` path, and marks the zarr as portable.
  - Added `policy/DP/package_real_zed_data.sh` to package an existing DP real-ZED zarr as a `.tar.gz` plus README for server extraction/training.
  - The packaging script sanitizes existing zarr attrs before compression, so older locally generated zarrs do not carry host-specific meta paths into the server package.
  - Verified targeted DP real-ZED preprocessing tests.
  - Verified script syntax with `bash -n policy/DP/package_real_zed_data.sh` and whitespace with `git diff --check`.

### Phase 46: Real Semantic-Pointwise Hybrid Regression Diagnosis
- **Status:** investigation complete, fix pending user decision
- Actions taken:
  - Inspected the latest `grasp_mug/demo_real_zed_sam2_objpc` HDF5, semantic hybrid zarr, preprocessing scripts, and real inference wrapper.
  - Confirmed the latest processed dataset is already in `right_base` output frame, so the primary issue is not missing robot-frame conversion.
  - Found severe `{A}` object-pointcloud outliers in the processed HDF5: 222 affected frames out of 1975 across 32 episodes, while `{B}` has no outlier under the same criterion.
  - Confirmed the zarr used by `train_semantic_pointwise_hybrid.sh` contains the same outliers in both `point_cloud` and `semantic_point_cloud_A`.
  - Found the processed dataset was generated with `workspace_crop_bounds_m=None`; online inference now does workspace filtering before transforming to `right_base`, so train/inference perception distributions can differ.
  - Found `real_infer_semantic_pointwise_hybrid.sh` still has duplicated default task assignments and a wrapper-level semantic checkpoint default that can differ from the checkpoint recorded in the training zarr meta.

### Phase 47: Real Semantic Inference Interface Cleanup
- **Status:** complete
- Actions taken:
  - Restored `policy/DP3/real_infer_semantic_pointwise_hybrid.sh` to a single `task_config` interface after user feedback; `ckpt_setting` is again derived as `${task_config}-objpc-semantic-pointwise-hybrid`.
  - Removed duplicated default `task_name/task_config` assignments from the semantic wrapper.
  - Changed the default semantic checkpoint path to the current-user training repo path, while still allowing `SEMANTIC_CKPT_A` to override it.
  - Updated the shell interface regression test to cover the single-config semantic call and the derived checkpoint setting.
  - Verified with `bash policy/DP3/scripts/test_real_infer_script_interfaces.sh` and `bash -n policy/DP3/real_infer_semantic_pointwise_hybrid.sh policy/DP3/real_infer_baseline.sh policy/DP3/scripts/test_real_infer_script_interfaces.sh`.

### Phase 48: Real Inference Frame-Argument Shorthand
- **Status:** complete
- Actions taken:
  - Reproduced the user-facing failure mode where `--right_base --auto` after semantic wrapper positional args was forwarded to Python as unknown argparse flags.
  - Added a regression test ensuring dashed frame shorthands are normalized before invoking `real_dp3_inference.py`.
  - Added `is_real_zed_output_frame_shorthand`, `normalize_real_zed_output_frame_token`, and `normalize_real_zed_calibration_token` to `real_infer_arg_utils.sh`.
  - Updated both `real_infer_semantic_pointwise_hybrid.sh` and `real_infer_baseline.sh` to accept `--right_base/--workspace/--source/--left_base` shorthand while preserving normal Python flags such as `--dry_run`.
  - Verified with `bash policy/DP3/scripts/test_real_infer_script_interfaces.sh` and `bash -n policy/DP3/real_infer_arg_utils.sh policy/DP3/real_infer_semantic_pointwise_hybrid.sh policy/DP3/real_infer_baseline.sh policy/DP3/scripts/test_real_infer_script_interfaces.sh`.

### Phase 49: Real-ZED DP Image Inference
- **Status:** complete with hardware run pending
- Actions taken:
  - User approved an independent DP real inference script with camera capture separated from async robot control.
  - Re-read DP real-ZED train scripts, DP dataset/config expectations, DP runner limitations, and the existing DP3 real inference control/safety implementation.
  - Added RED tests for dynamic DP RGB obs-key detection, single-camera `head_cam:<camera>` mapping, RGB resize/normalization, a DP runner that only forwards checkpoint-required camera keys, and safe async/dry-run parser defaults.
  - Added `script/real_zed_inference/real_dp_inference.py`, which loads DP checkpoints directly, starts lightweight RGB ZED camera threads, builds image observations, and submits DP action chunks to the existing async action controller.
  - Added `policy/DP/real_infer.sh`, deriving `ckpt_setting` from `task_config` plus camera labels and explicitly mapping single-camera checkpoints to the selected live camera.
  - Added `policy/DP/scripts/test_real_infer_script_interfaces.sh` for wrapper argument wiring.
  - Verified with `PYTHONDONTWRITEBYTECODE=1 python -m unittest script.test_real_dp_inference script.test_real_zed_inference_actions`.
  - Verified syntax with `PYTHONDONTWRITEBYTECODE=1 python -m py_compile script/real_zed_inference/real_dp_inference.py script/test_real_dp_inference.py` and `bash -n policy/DP/real_infer.sh policy/DP/scripts/test_real_infer_script_interfaces.sh`.
  - Verified wrapper wiring with `bash policy/DP/scripts/test_real_infer_script_interfaces.sh`, CLI visibility with `python script/real_zed_inference/real_dp_inference.py --help | rg -n "dp_camera_map|async_control|max_executed_joint_delta_change|ckpt_path|camera_labels"`, and whitespace with `git diff --check`.

### Phase 50: Optional-Camera Real-ZED Collection
- **Status:** complete with hardware run pending
- Actions taken:
  - Added RED tests proving `_resolve_cameras(...)` should accept a single camera label from calibration, accept two explicit camera labels/serials without calibration, and reject serial-count mismatch without calibration.
  - Updated `script/real_zed_collection/collect_zed_robotwin_raw.py` so `_resolve_cameras(...)` accepts any non-empty camera-label list instead of exactly three labels.
  - If a calibration path is available and the configured serial list length does not match the requested label subset, the script now resolves serials from calibration by label. This covers the common case where the config still has three serials but the CLI overrides `--camera_labels left`.
  - Updated the argparse description from three-ZED-specific wording to generic ZED collection wording.
  - Verified with `PYTHONDONTWRITEBYTECODE=1 python -m unittest script.test_real_zed_collection_pipeline`.
  - Verified syntax with `PYTHONDONTWRITEBYTECODE=1 python -m py_compile script/real_zed_collection/collect_zed_robotwin_raw.py script/test_real_zed_collection_pipeline.py`.

### Phase 51: Real-ZED Episode Global Camera Quality Check
- **Status:** investigation complete
- Actions taken:
  - Inspected `/media/zheng/Extreme SSD/geo_mani_data/grasp_mug/real_zed_raw/episode_20260502154520`.
  - Confirmed the manifest has 58 frames and lists all three cameras: `global`, `left`, and `right`.
  - Computed RGB/depth statistics for sampled and full frames.
  - Found `global` RGB is constant `[0,134,0]` for all 58 frames, RGB std is exactly zero, frame-to-frame RGB delta is zero, and valid depth rate is only `0.000448`.
  - Found `left` and `right` are normal: nonzero RGB variance and valid depth means around `0.941` and `0.882`.
  - Saved a contact-sheet preview at `outputs/real_zed_collection/previews/episode_20260502154520_quality_check/rgb_contact_sheet_frames_000_020_057.png`.

### Phase 52: DP Real-ZED Mixed-Resolution Training Fix
- **Status:** complete
- Actions taken:
  - Diagnosed the user traceback from `bash train_real_zed.sh grasp_mug demo_real_zed_sam2_objpc 49 0 14 0`.
  - Confirmed the first 49 processed episodes mix RGB shapes: 32 old `360x640` episodes and 17 new `1080x1920` episodes.
  - Confirmed `episode_20260502154520` is not in the current processed metadata, so the reported failure is a mixed-resolution zarr issue rather than the previously inspected black-`global` episode.
  - Added shell regression coverage that `train_real_zed.sh` and `train_real_zed_multicam.sh` pass `360,640` for `Large_L515` into their real-ZED preprocess scripts.
  - Updated `train_real_zed.sh` and `train_real_zed_multicam.sh` to derive `resize_hw` from `head_camera_type` and forward it to preprocessing.
  - Added `set -euo pipefail` to the DP real-ZED train/preprocess shell scripts so a preprocessing failure stops before Hydra training starts with a missing zarr.
  - Verified with `bash policy/DP/scripts/test_real_zed_train_script_interfaces.sh`.
  - Verified syntax with `bash -n policy/DP/train_real_zed.sh policy/DP/train_real_zed_multicam.sh policy/DP/process_data_real_zed.sh policy/DP/process_data_real_zed_multicam.sh policy/DP/scripts/test_real_zed_train_script_interfaces.sh`.
  - Verified collection pipeline tests in the RoboTwin environment with `/home/zheng/miniforge3/envs/RoboTwin/bin/python -m unittest script.test_real_zed_collection_pipeline`.

### Phase 53: Real DP3 In-Process Episode Reset
- **Status:** complete with hardware run pending
- Actions taken:
  - User approved adding an in-process reset key for real semantic DP3 inference so repeated 2000-step trials do not require restarting the full robot/ZED/SAM2/DP3 stack.
  - Planned the safe reset path around stopping the async action controller before moving the robot back to the initial photo pose.
  - Added a non-blocking terminal keyboard listener for `r` reset and `q` quit.
  - Connected the listener to `AsyncActionController` through an external stop event so reset requests stop stale action playback before robot reset.
  - Refactored real DP3 inference episode initialization so startup and keyboard reset both clear DP3 observation history and apply the configured initial gripper state.
  - Added `--reset_sam2_on_keyboard_reset`; default behavior keeps active SAM2 prompts/tracking across resets.
  - Added unit coverage for keyboard command handling, async external-stop behavior, and parser defaults.
  - Verified real-ZED inference action tests, Python syntax, shell syntax, wrapper interface tests, CLI visibility, and whitespace checks.

### Phase 54: Robot-Camera AprilTag Calibration First-Sample Fix
- **Status:** complete with hardware run pending
- Actions taken:
  - Diagnosed the first unusable calibration sample as a false Button-B release event at startup.
  - Added a test proving Button B capture now requires a press edge before the release edge.
  - Refactored button-state transition handling into a testable helper.
  - Initialized the button monitor from the current hardware key state instead of assuming all-zero state.
  - Cleared the global command state before starting the calibration button thread.
  - Verified the targeted test, full robot-camera calibration tests, Python syntax, and whitespace checks.

### Phase 55: Optional-Camera Real-ZED Extrinsic Calibration
- **Status:** complete with hardware run pending
- Actions taken:
  - Added tests proving `calibrate_three_zed_extrinsics.py` accepts a two-camera `real_zed_collection.yaml` and rejects single-camera calibration configs.
  - Replaced the runtime `exactly 3` label/serial validation with `_resolve_requested_cameras(...)`, which accepts any 2+ unique camera labels.
  - Made `--labels` variable-length while preserving config-driven defaults from `real_zed_collection.yaml`.
  - Updated runtime warnings and saved YAML metadata to report the actual configured camera count.
  - Verified targeted tests, full robot-camera calibration tests, Python syntax, and whitespace checks.

### Phase 56: Real-ZED Exposure Lock And Quality Watchdog
- **Status:** complete with hardware threshold tuning pending
- Actions taken:
  - Added fixed ZED image-control fields to `Args` and `real_zed_collection.yaml`: auto exposure disabled, exposure `22`, gain `12`, and white-balance temperature `4500` by default.
  - Applied those image controls inside `zed_capture_loop(...)` immediately after opening each ZED.
  - Added image-quality metrics for RGB mean/std, overexposed ratio, underexposed ratio, and valid-depth ratio.
  - Added repeated-warning logic for bad quality streaks, covering overexposure, underexposure, low RGB variance/black-screen-like frames, and low valid-depth ratio.
  - Saved `image_quality_values` and `image_quality_bad` into each camera `.npz`, plus JSON-friendly `camera_quality` metadata in each manifest frame.
  - Added tests for fixed default image controls and overexposure detection.
  - Verified targeted tests, full real-ZED collection pipeline tests, real-ZED inference action tests, Python syntax, and whitespace checks.

### Phase 57: Real-ZED Initial Frame Preview
- **Status:** complete with hardware UI check pending
- Actions taken:
  - Added `preview_before_collection`, `preview_before_collection_timeout_sec`, and `preview_window_name` to the collection args/config.
  - Added initial camera-frame waiting and side-by-side preview canvas rendering for all active labels.
  - Inserted the preview gate immediately after ZED thread initialization and before xtrainer robot initialization.
  - Made Enter, `c`, or Space continue; `q` or Esc aborts and cleans up camera/writer threads before robot initialization.
  - Made the preview auto-skip with a warning when no GUI display is available.
  - Added unit coverage for preview defaults and multi-camera canvas rendering.
  - Verified full real-ZED collection pipeline tests, real-ZED inference action tests, Python syntax, and whitespace checks.

### Phase 58: Real-ZED SAM2 Postprocess Debug Video And Keyframe PLY
- **Status:** complete with visual inspection pending
- Actions taken:
  - Added default debug keyframe selection at `0%, 20%, 40%, 60%, 80%, 100%` of each processed episode.
  - Added SAM2 tracking debug video generation in `postprocess_real_zed_sam2_objpc_dataset.py`, writing `debug/episode*/tracking_videos/sam2_tracking.mp4`.
  - The MP4 overlays per-placeholder SAM2 masks on the selected camera views for every processed frame in the demo.
  - Extended debug point-cloud export so each keyframe writes the existing merged `objects_ab.ply` plus separate per-object fused PLY files such as `frame_000000_A.ply` and `frame_000000_B.ply`.
  - Kept stride-based debug frame selection available with `--debug_frame_mode stride`.
  - Added unit coverage for keyframe selection, per-object PLY output, and two-view debug canvas rendering.
  - Verified RED first with the three new targeted tests failing for missing functions/interfaces.
  - Verified GREEN with the targeted tests passing.
  - Verified full collection pipeline tests with `/home/zheng/miniforge3/envs/RoboTwin/bin/python -m unittest script.test_real_zed_collection_pipeline`.
  - Verified syntax with `PYTHONDONTWRITEBYTECODE=1 python -m py_compile script/real_zed_collection/postprocess/postprocess_real_zed_sam2_objpc_dataset.py script/test_real_zed_collection_pipeline.py`.
  - Verified whitespace with `git diff --check -- script/real_zed_collection/postprocess/postprocess_real_zed_sam2_objpc_dataset.py script/test_real_zed_collection_pipeline.py`.

### Phase 59: Real-ZED Dense Per-Object Point Cloud Default
- **Status:** complete
- Actions taken:
  - Added a failing regression test proving the SAM2 objpc batch postprocess default should be 5000 object points per placeholder.
  - Updated `postprocess_real_zed_sam2_objpc_dataset.py` so `--object_point_num` defaults to `5000`.
  - Updated `postprocess_raw_to_robotwin_hdf5.py` function and CLI defaults so direct raw-to-HDF5 conversion also saves 5000 points per object by default.
  - Preserved explicit `--object_point_num` override for smaller debug runs.
  - Verified the new targeted test passes.
  - Verified full collection pipeline tests with `/home/zheng/miniforge3/envs/RoboTwin/bin/python -m unittest script.test_real_zed_collection_pipeline`.
  - Verified syntax with `PYTHONDONTWRITEBYTECODE=1 python -m py_compile script/real_zed_collection/postprocess/postprocess_real_zed_sam2_objpc_dataset.py script/real_zed_collection/postprocess/postprocess_raw_to_robotwin_hdf5.py script/test_real_zed_collection_pipeline.py`.
  - Verified whitespace with `git diff --check -- script/real_zed_collection/postprocess/postprocess_real_zed_sam2_objpc_dataset.py script/real_zed_collection/postprocess/postprocess_raw_to_robotwin_hdf5.py script/test_real_zed_collection_pipeline.py`.

### Phase 60: Semantic Hybrid Training Resource Controls
- **Status:** complete
- Actions taken:
  - Added a text regression test for `train_semantic_pointwise_hybrid.sh` resource-control overrides.
  - Confirmed RED by running the test against the old script and observing missing `dataloader_num_workers` and related parameters.
  - Added script parameters:
    - `dataloader_num_workers=${17:-4}`
    - `val_dataloader_num_workers=${18:-2}`
    - `pin_memory=${19:-true}`
    - `val_pin_memory=${20:-false}`
    - `max_val_steps=${21:-2}`
  - Forwarded those parameters to Hydra as `dataloader.num_workers`, `val_dataloader.num_workers`, `dataloader.pin_memory`, `val_dataloader.pin_memory`, and `training.max_val_steps`.
  - Chose defaults that keep train-side throughput reasonable while reducing validation/DataLoader memory spikes at epoch boundaries.
  - Verified the new targeted test with `python policy/DP3/scripts/test_semantic_hybrid_training_resource_overrides.py`.
  - Verified shell syntax with `bash -n policy/DP3/train_semantic_pointwise_hybrid.sh`.
  - Verified whitespace with `git diff --check -- policy/DP3/train_semantic_pointwise_hybrid.sh policy/DP3/scripts/test_semantic_hybrid_training_resource_overrides.py`.

### Phase 61: DP3 Training Resource Controls Across Scripts
- **Status:** complete
- Actions taken:
  - Expanded the resource-control regression test to cover direct DP3 train wrappers plus the base `train.sh`/`train_rgb.sh` helper path.
  - Confirmed RED across the unmodified wrappers before implementation.
  - Added resource parameters to `train.sh`, `train_rgb.sh`, `scripts/train_policy.sh`, and `scripts/train_policy_rgb.sh`.
  - Added the same controls to objpc, objpc_5000, objpc_actorseg, NDF, NDF pointwise, NDF hybrid, NDF feature-count, NDF interaction, NDF actorseg hybrid, semantic pointwise, semantic hybrid feature-count, semantic actorseg hybrid, and Utonia hybrid wrappers.
  - Extended `ndf_pointwise_arg_utils.sh` so the NDF pointwise family parses resource defaults from the common argument path.
  - Preserved the existing semantic hybrid default `semantic_point_num=1024` and avoided changing zarr/checkpoint naming.
  - Verified the targeted resource-interface test with `python policy/DP3/scripts/test_semantic_hybrid_training_resource_overrides.py`.
  - Verified shell syntax with `bash -n` over all touched train wrappers and helper scripts.
  - Verified every targeted wrapper contains `training.max_val_steps=${max_val_steps}` and `dataloader.num_workers=${dataloader_num_workers}` using `rg -L` checks.
  - Verified whitespace with `git diff --check` over all touched train/resource test files.
- Notes:
  - Running `python policy/DP3/scripts/test_hybrid_feature5000_path.py` still fails in this environment for pre-existing reasons: current `train_ndf_pointwise_hybrid_feat5000.sh` defaults to `ndf_point_num=512` while the test expects 5000, and the active Python environment lacks `zarr`. This is unrelated to the resource-control change.

### Phase 62: Real Semantic Checkpoint Branch-Mismatch Diagnosis
- **Status:** complete
- Actions taken:
  - Inspected the failing teapot checkpoint payload directly and confirmed it has no `semantic_point_cloud_A/B` extractor keys.
  - Confirmed the same checkpoint's saved Hydra config only includes `point_cloud` and `agent_pos`, while the semantic zarr and meta do contain `semantic_point_cloud_A`.
  - Identified the root cause as a training wrapper edge case: if the semantic zarr already exists but `semantic_ckpt_A/B` are not passed, the old script trains a non-semantic model under the semantic-hybrid suffix.
  - Added regression coverage for using meta `feature_placeholders` to enable semantic branches in `train_semantic_pointwise_hybrid.sh`.
  - Updated `train_semantic_pointwise_hybrid.sh` so existing semantic meta can enable `semantic_point_cloud_A/B` even when preprocessing is not rerun.
  - Added real-inference validation to reject semantic checkpoint arguments when the DP3 checkpoint was not trained with matching semantic point-cloud branches.
  - Updated `real_infer_semantic_pointwise_hybrid.sh` default semantic point count to `1024`, matching the current training default.
  - Verified `policy/DP3/scripts/test_real_infer_script_interfaces.sh`.
  - Verified `python policy/DP3/scripts/test_semantic_hybrid_training_resource_overrides.py`.
  - Verified `python -m unittest script.test_real_zed_inference_actions`.
  - Verified shell syntax and Python syntax checks for the touched files.

### Phase 63: Real Inference ZED Auto Image Controls
- **Status:** complete
- Actions taken:
  - Traced real inference ZED startup to `zed_capture_loop(...)` and confirmed it inherited fixed collection defaults because no image-control args were forwarded.
  - Added real inference CLI options for `--zed_auto_exposure`, `--zed_exposure`, `--zed_gain`, and `--zed_whitebalance_temp`.
  - Changed real inference defaults to auto exposure/gain and auto white balance by using `--zed_auto_exposure` default true and `--zed_whitebalance_temp 0`.
  - Forwarded those options into `CollectionArgs` and each live `zed_capture_loop(...)` thread.
  - Added parser regression tests for the new automatic defaults and fixed-control override path.
  - Verified targeted parser tests, full real inference action tests, real inference wrapper interface tests, Python syntax, shell syntax, and whitespace checks.

### Phase 64: LZ xtrainer EEF Route Read
- **Status:** read-only analysis complete
- Actions taken:
  - Read `include/lz_xtrainer/experiments/run_control.py`, `run_inference.py`, `dobot_control/agents/dobot_agent.py`, `dobot_control/robots/dobot.py`, `dobot_control/env.py`, `ModelTrain/dp/train_dp.py`, and related dataset utilities.
  - Confirmed LZ collects EEF actions by converting leader joints to Dobot FK and writing absolute EEF pose into `control`, while still executing teleoperation through joint ServoJ.
  - Confirmed LZ trains default absolute EEF 6D actions (`predict_eef_6d=True`, `predict_eef_6d_delta=False`), yielding 20D bimanual action vectors.
  - Confirmed LZ real inference converts predicted absolute EEF 6D actions back to rotvec, solves IK, and sends joint commands through ServoJ rather than directly using ServoP.
  - Compared with current `include/xtrainer_clover` and real-ZED collection: current code records joint actions only and has zero `ee_pos_quat`, so EEF training needs explicit collection/postprocess work.

### Phase 65: Dobot Offline FK Consistency Check Tool
- **Status:** complete
- Actions taken:
  - Added a RED unit test for FK helper behavior before implementing the diagnostic tool.
  - Added `script/real_zed_collection/check_dobot_fk_consistency.py`, a read-only real-hardware tool that compares local DH FK, Dobot `PositiveSolution`, and Dobot `GetPose` for current robot joints.
  - The tool defaults to the xtrainer controller TCP assumption `SetTool(1, 0, 0, 197, 0, 0, 0)` through `--tool_z_m 0.197 --tool 1`.
  - It supports left/right/both arms, repeated samples, JSON output, and configurable pass/fail thresholds.
  - Verified unit tests, Python syntax, CLI help, and whitespace checks.

### Phase 66: Right-Base EEF Reference Frame
- **Status:** complete
- Actions taken:
  - Added RED coverage for `right_base` EEF transform composition, real inference parser support, and wrapper defaults.
  - Implemented `left_base/right_base` support in `load_world_from_base_transforms`.
  - Changed EEF preprocessing and real inference defaults from `workspace` to `right_base`.
  - Renamed EEF zarr/checkpoint suffixes to include `rightbase`, preventing accidental reuse of old workspace-frame EEF artifacts.
  - Added EEF dataset-frame validation against `real_zed_sam2_objpc_meta.json`.
  - Verified targeted unit tests, wrapper tests, shell syntax, Python syntax, and whitespace checks.

### Phase 67: Global-Camera EEF Wrapper Set
- **Status:** complete
- Actions taken:
  - Added RED coverage requiring `global` EEF wrappers to use `source/reference_camera` and `*-global` suffixes.
  - Added Python preprocess wrappers for scene, objpc, semantic pointwise, and semantic hybrid global EEF zarr generation.
  - Added shell preprocess wrappers with default `eef_frame_mode=reference_camera`.
  - Added train wrappers for scene, objpc, and semantic hybrid global EEF checkpoints.
  - Added real inference wrappers for baseline and semantic hybrid global EEF checkpoints.
  - Verified wrapper tests, shell syntax, Python syntax, and whitespace checks.

### Phase 68: EEF Training Empty Zarr Path Fix
- **Status:** complete
- Actions taken:
  - Read the failing `train_objpc_eef_absolute6d_global.sh` path and confirmed the logged Hydra overrides lacked `task.dataset.zarr_path`.
  - Added RED regression tests requiring shared train helpers to accept an explicit zarr path and objpc EEF wrappers to pass it.
  - Updated `scripts/train_policy.sh` and `scripts/train_policy_rgb.sh` to accept optional argument 14 and forward `task.dataset.zarr_path`.
  - Updated EEF baseline/objpc rightbase/global train wrappers to compute and pass their generated zarr path explicitly.
  - Added `set -euo pipefail` to those EEF wrappers so preprocessing failures do not fall through into training.
  - Verified `python policy/DP3/scripts/test_eef_pointwise_wrappers.py`.
  - Verified `python policy/DP3/scripts/test_semantic_hybrid_training_resource_overrides.py`.
  - Verified shell syntax for the touched train helper and EEF wrapper scripts.
  - Verified with a fake `python` capture that `train_policy.sh` now forwards `task.dataset.zarr_path=../../../data/grasp_mug-demo_real_zed_sam2_objpc_global-50-objpc-eef-absolute6d-global.zarr`.
  - Verified `git diff --check`.

### Phase 69: Raw RGB Episode Export Utility
- **Status:** complete
- Actions taken:
  - Added RED coverage for exporting selected-camera raw RGB images from a synthetic real-ZED raw episode.
  - Added `script/real_zed_collection/export_raw_rgb_images.py`.
  - The script resolves `task_name + episode` to a raw episode directory, where `episode` can be a zero-based sorted index, an `episode_*` directory name, a bare timestamp, or a full path.
  - It exports PNG files under per-camera subdirectories and writes `export_summary.json`.
  - It supports `--camera_labels`, `--frame_stride`, `--max_frames`, and `--rgb_key rgb|full_rgb_debug`.
  - Verified the targeted export test.
  - Verified `python -m py_compile script/real_zed_collection/export_raw_rgb_images.py`.
  - Verified absolute-path `--help` from `/tmp`.
  - Full `script.test_real_zed_collection_pipeline` currently fails two unrelated DP zarr tests because this environment lacks the `zarr` package.
  - Verified `git diff --check`.

### Phase 70: Raw Colored Point Cloud Episode Export Utility
- **Status:** complete
- Actions taken:
  - Added RED coverage for exporting a fused colored PLY from a synthetic real-ZED raw episode and calibration snapshot.
  - Added `script/real_zed_collection/export_raw_colored_pointclouds.py`.
  - The script shares the RGB exporter's task/episode resolution behavior and supports sorted episode index, episode name/timestamp, or full path.
  - It converts raw `rgb + depth_m + camera_matrix` to colored point clouds, transforms each camera into the calibration frame, and exports fused/per-camera/both PLY outputs.
  - It supports `--camera_labels`, `--frame_stride`, `--max_frames`, `--rgb_key`, `--depth_key`, `--point_num`, and xyz crop bounds.
  - Verified the new colored pointcloud export test.
  - Verified both RGB and colored pointcloud export tests together.
  - Verified `python -m py_compile script/real_zed_collection/export_raw_colored_pointclouds.py`.
  - Verified absolute-path `--help` from `/tmp`.
  - Verified `git diff --check`.

### Phase 71: Real DP3 EEF IK Failure Diagnostics
- **Status:** complete
- Actions taken:
  - Traced the user's EEF inference stack to `eef_policy_action_to_joint_action(...)` and `env.get_ik(eef_base)`.
  - Confirmed xtrainer's ZMQ robot server did not return exceptions for failed robot methods, which can make the client appear stuck at `env.get_ik`.
  - Added RED tests for remote robot error payload handling and for DP3 EEF IK failures reporting the target pose.
  - Updated `include/xtrainer_clover/dobot_control/robots/robot_node.py` so server-side robot method exceptions are returned as structured errors, and `get_ik` raises them client-side with a 5s receive timeout.
  - Updated `script/real_zed_inference/real_dp3_inference.py` so EEF IK failures include rounded `eef_world`, `eef_base`, and the underlying exception.
  - Verified targeted xtrainer and real-inference tests.
  - Verified full `script.test_xtrainer_servo_params`.
  - Verified full `script.test_real_zed_inference_actions`.
  - Verified Python syntax checks for touched files.
  - Verified `git diff --check`.

### Phase 72: Real-ZED DP EEF Image Policy
- **Status:** complete with hardware run pending
- Actions taken:
  - Read `policy/DP/process_data_real_zed.py`, DP train/infer wrappers, DP image datasets/configs, and DP3 EEF utilities.
  - Confirmed DP image baseline zarr currently stores joint `state`/`action`, while DP3 EEF utilities already provide EEF14 state and 20D absolute-6D action conversion from `joint_action` plus optional measured `/eef_action/base_pose6`.
  - Added DP real-ZED preprocessing support for `--action_mode eef_absolute6d`, producing EEF14 `state` and 20D absolute-6D EEF `action`.
  - Added `robot_dp_20.yaml`, `default_task_20.yaml`, and `default_task_20_multicam.yaml`.
  - Added single-camera and multicam DP EEF training wrappers with `*-eef-absolute6d-global` zarr/checkpoint naming.
  - Added DP EEF real inference support in `real_dp_inference.py`, converting policy EEF actions through Dobot IK before existing ServoJ smoothing/async control.
  - Added wrapper interface coverage for DP EEF training and inference.
  - Verified shell syntax, wrapper interface tests, Python compile, DP20 YAML shape checks, real inference help exposure, a RoboTwin-conda synthetic EEF zarr smoke, and `git diff --check`.

### Phase 73: Real DP3 Snapshot And SAM2 Reselect Hotkeys
- **Status:** complete with hardware run pending
- Actions taken:
  - Read the current `KeyboardCommandListener`, async/sync real DP3 inference loops, live ZED observation construction, and EEF real-infer wrappers.
  - Design choice: use `s` for low-cost snapshot saving of current RGB/depth/calibration plus an already-built fused scene point cloud, and use `b` for robot reset plus clearing SAM2 tracking/prompts so the next semantic update reopens bbox selection.
  - Added keyboard listener support for `s` save snapshot and `b` reset/reselect commands.
  - Added `save_live_inference_snapshot(...)`, which writes per-camera `rgb.png`, `camera_data.npz`, `manifest.json`, `scene_pointcloud.npy`, and `scene_pointcloud.ply` under `outputs/real_zed_inference/snapshots`.
  - Wired snapshot saving into both async and sync DP3 real inference loops immediately after live observation construction.
  - Updated reset handling so `b` clears SAM2 tracking state and prompt cache without restoring loaded prompts.
  - Added parser flags: `--keyboard_save_snapshot_key`, `--keyboard_reselect_key`, `--snapshot_output_dir`, and `--snapshot_scene_point_num`.
  - Verified targeted RED/GREEN tests, full `script.test_real_zed_inference_actions`, Python syntax, EEF wrapper shell syntax, help output for new flags, and whitespace.
  - `policy/DP3/scripts/test_real_infer_script_interfaces.sh` still fails on an existing semantic wrapper `ckpt_setting` mismatch (`unit_real_zed_cfg_global-...` vs the test's old `unit_real_zed_cfg-...` expectation), unrelated to the new hotkeys.
