# Findings & Decisions

## Real Three-ZED Collection Pipeline (2026-04-24)

### Existing xtrainer control path
- `include/xtrainer_clover/experiments/run_control.py` uses three RealSense RGB camera threads, `BimanualAgent`, two `DobotAgent`s, `RobotEnv`, and `ZMQClientRobot`.
- The main loop preserves button-driven lock/servo/recording semantics through the shared `what_to_do` state.
- During DP recording, the script saves one pkl per frame through `scripts.format_obs.save_dp_frame(...)`.
- The saved DP frame currently contains robot `obs`, `control`, activation flags, and RGB images named `base_rgb`, `left_wrist_rgb`, and `right_wrist_rgb`.
- The current xtrainer collection path does not record depth, camera intrinsics/extrinsics, per-camera point clouds, fused scene point clouds, or object point clouds.

### Real-ZED DP image baseline notes (2026-04-27)
- `policy/DP` is an image-based DP baseline. Its original preprocess/dataset path only writes and loads `head_camera`, while `deploy_policy.py` already constructs `head_cam`, `left_cam`, and `right_cam` observations during inference.
- `MultiImageObsEncoder` already supports multiple RGB keys through `shape_meta.obs`; the missing pieces were a multi-camera zarr layout and a dataset/config that exposes all three image observations.
- The real-ZED raw RGB frames are `360x640`, so `Large_L515` is the correct camera config for this real image baseline unless the frames are explicitly resized during preprocessing.
- The compact SAM2 objpc HDF5 files do not store `/observation/*/rgb`, but `real_zed_sam2_objpc_meta.json` links each processed episode back to the raw episode directory. DP image zarr generation should therefore read RGB from raw ZED frame NPZs and joint vectors from the compact HDF5.
- The three-camera branch maps raw camera labels as `global -> head_camera`, `left -> left_camera`, and `right -> right_camera`; DP training sees those as `head_cam`, `left_cam`, and `right_cam`.

### RoboTwin `policy/DP` vs xtrainer `ModelTrain/dp` (2026-04-28)
- RoboTwin `policy/DP` is a Hydra/zarr-based fork of the diffusion_policy image workspace. It preprocesses RoboTwin HDF5 or real-ZED raw/meta into zarr with `head_camera`, optional `left_camera/right_camera`, `state`, `action`, and `meta/episode_ends`.
- xtrainer `ModelTrain/dp` is a standalone training pipeline around `Agent`, `Dataset`, and `DiffusionPolicy`. It reads xtrainer trajectory folders of per-frame `.pkl` files, optionally loads images lazily or through memmap, and builds train/eval split from directory names.
- RoboTwin DP default is `horizon=8`, `n_obs_steps=3`, `n_action_steps=6`, `num_epochs=600`, `batch_size=128`, action dim 14/16 by config, image ResNet18 feature dim 512 per camera, and ImageNet normalization.
- xtrainer DP default is `pred_horizon=16`, `obs_horizon=1`, `action_horizon=8`, `epochs=300`, `batch_size=32`, action dim fixed at 14 in `Agent`, ResNet18 projected to `image_output_size=32` per camera, image resize to `240x320`, random crop to `216x288`, and normalization with mean/std 128 for RGB.
- RoboTwin DP conditions diffusion through `MultiImageObsEncoder` global condition and uses `DiffusionUnetImagePolicy`; xtrainer DP manually concatenates encoded modalities in order `eef, hand_pos, img, pos, touch` and feeds a custom `ConditionalUnet1D`.
- RoboTwin DP currently uses only RGB + joint vector for real-ZED DP baseline; xtrainer DP can select `representation_type` such as `img-pos`, `eef`, `hand_pos`, `touch`, and `depth`, with optional delta action modes.

### Real DP3 inference requirements (2026-04-28)
- xtrainer `experiments/run_inference.py` commands the real robot with 14D joint actions through `RobotEnv.step(action, np.array([1,1]))`, clamps gripper dimensions `6` and `13` into `[0,1]`, and initializes observations from `env.get_obs()["joint_positions"]`.
- RoboTwin DP3 deployment should reuse `policy/DP3/deploy_policy.py:get_model()` plus `encode_obs(...)`, because that path already loads the correct checkpoint, EMA setting, NDF/semantic models, and DP3 `RobotRunner` action queue.
- For the user's baseline `train.sh` model, the real observation only needs `joint_action/vector` and a fused `/pointcloud` with shape `[1024,6]`.
- For the user's `train_semantic_pointwise_hybrid.sh` model, the real observation needs both the hybrid main context point cloud and `object_pointcloud/{A,B}` so `encode_obs(...)` can compute `semantic_point_cloud_A` from the semantic checkpoint.
- Existing `policy/DP3/scripts/sam2_pointcloud_utils.py` can produce `{A}/{B}` object point clouds online if the observation contains per-camera `rgb`, `intrinsic_cv`, and `extrinsic_cv`; this matches the real-ZED live camera frames once they are transformed into the workspace/world frame.
- `RobotRunner.get_action(policy, observation)` accepts one encoded observation at a time and returns an action chunk. A real execution script should update the DP3 observation cache, iterate the returned actions, send them to `RobotEnv.step`, and rebuild live observations between actions.
- For real-ZED online SAM2 projection, `extrinsic_cv` must be world-to-camera. The live inference script therefore stores `invert_transform(T_world_from_camera)` in `observation/<camera>/extrinsic_cv` while keeping `cam2world_gl` as `T_world_from_camera`.
- The real inference wrappers default to dry-run mode and require `--execute` to command the robot. This prevents accidentally moving hardware during model/camera/prompt checks.
- Real control machines should not need full RoboTwin simulator dependencies for DP3 real inference. The blocking import chain was `policy/DP3/deploy_policy.py` importing `sapien.core` and `envs` at module import time, even though real inference only needs `get_model` and `encode_obs`.
- A second unnecessary real-inference dependency chain was `deploy_policy.py -> object_pointcloud_utils.py -> ndf_feature_utils.py`. Baseline and semantic-pointwise-hybrid real inference do not need NDF, so both direct and indirect NDF imports should be lazy.

### Robot-camera calibration design notes (2026-04-28)
- xtrainer/RoboTwin teleoperation uses Button A short press for lock/unlock, Button A long press for servo, and Button B rising edge for recording/green-light events.
- `RobotEnv.get_obs()["ee_pos_quat"]` is currently zero-filled in xtrainer's Dobot driver, so robot-camera calibration cannot rely on that field.
- The usable robot pose source is `RobotEnv.get_XYZrxryrz_state()`, which returns concatenated left/right Cartesian poses `[x,y,z,rx,ry,rz]` from Dobot `GetPose()`.
- For an AprilTag cube held by the gripper and observed by a fixed external camera, each sample gives `T_base_from_gripper_i` from the robot and `T_camera_from_tag_i` from tag PnP.
- This is a target-on-gripper / fixed-camera calibration. With OpenCV `calibrateRobotWorldHandEye`, map algorithm world to the fixed camera frame and algorithm camera to the moving tag frame:
  - input `world2cam` as `T_tag_from_camera_i = inverse(T_camera_from_tag_i)`
  - input `base2gripper` as `T_gripper_from_base_i = inverse(T_base_from_gripper_i)`
  - output `base2world` as `T_camera_from_base`, whose inverse is the needed `T_base_from_camera`
  - output `gripper2cam` as `T_tag_from_gripper`, which is the fixed tag/cube mounting transform
- Left and right arms should be calibrated separately because xtrainer exposes separate left/right Dobot base frames.
- Implemented `script/real_zed_collection/calibrate_robot_camera_apriltag.py` as a standalone interactive collector/solver. It opens one ZED by calibration label or serial number, detects ArUco/AprilTag markers with OpenCV ArUco, captures samples with Button B, and writes `t_camera_from_base`, `t_base_from_camera`, `t_tag_from_gripper`, per-sample residuals, debug images, and raw sample JSON/YAML.
- The calibration script keeps xtrainer-style Button A lock/servo behavior and reuses the same servo safety checks from `collect_zed_robotwin_raw.py`.
- The default pose conversion assumes Dobot Cartesian pose units are millimeters and Euler angles are degrees (`--pose_xyz_unit mm --pose_rotation_mode euler_deg --pose_euler_order xyz`); these are explicit CLI parameters so hardware validation can adjust convention if needed.
- The marker detector now accepts `--marker_dictionary`, including explicit dictionaries such as `DICT_4X4_50` and an `auto` mode that tries common ArUco and AprilTag dictionaries. For the user's single-face ArUco marker with id 4, run with `--tag_id 4` and either the known dictionary or `--marker_dictionary auto`.
- Three-ZED extrinsic calibration and robot-camera calibration now default to `script/real_zed_collection/configs/real_zed_collection.yaml` for `camera_labels`/`zed_serials`. Explicit `--serials` in three-ZED calibration and explicit `--zed_serial` in robot-camera calibration still override this config.
- Robot-camera calibration now runs ZED frame capture and marker detection in a background worker thread. The robot control loop reads the latest camera snapshot and records `camera_frame_age_sec` for every captured sample, so image processing latency is visible without blocking servo control.
- Real-ZED HDF5 postprocessing and real DP3 inference now support `output_frame` values `source`, `workspace`, `left_base`, and `right_base`. For `left_base/right_base`, the code composes each camera transform as `T_base_from_robot_camera @ inverse(T_source_from_robot_camera) @ T_source_from_camera_i`, so scene and object point clouds plus `cam2world_gl/extrinsic_cv` are expressed in the selected arm base frame.
- When postprocessing outputs to a robot base frame, workspace 3D crop bounds are not applied after the frame transform, because those bounds live in workspace/source coordinates. Use collection-time workspace crop or 2D workspace masks when training in base frame.

### Existing DP3 training data expectations
- `policy/DP3/scripts/process_data.py` expects RoboTwin-style HDF5 episodes under `data/<task>/<task_config>/data/episode{i}.hdf5`.
- The minimum baseline DP3 fields are `/joint_action/vector` and `/pointcloud`.
- Object-pointcloud and pointwise feature branches read optional `/object_pointcloud/{placeholder}` datasets.
- `policy/DP3/scripts/object_pointcloud_utils.py` loads `/observation/head_camera/intrinsic_cv`, `extrinsic_cv`, `cam2world_gl`, and `mesh_segmentation` as simulator fallback metadata, but if `/object_pointcloud/{A}` exists the preprocessors use it directly.
- Existing NDF, semantic, Utonia, and interaction branches can all reuse canonical object point clouds and generate their own feature zarr outputs later.

### Design implication
- Real-world collection should not try to mimic simulator actor segmentation online. It should record calibrated RGB-D/pointcloud data once, then derive `/pointcloud` and `/object_pointcloud/{A,B,...}` in a postprocess step using real segmentation masks.
- A model-agnostic canonical HDF5 episode is the correct compatibility boundary. Model-specific zarr preprocessing should remain separate and reuse the existing DP3 scripts.

### Calibration findings
- `/home/zheng/github/geometry_awareness_manipulation/scripts/tools/calibrate_three_zed_charuco_extrinsics.py` opens three ZED cameras, detects one shared Charuco board across multiple poses, and writes `three_camera_charuco_extrinsics.yaml`.
- Its output contains per-camera serial numbers, camera intrinsics, residual statistics, and `relative_to_reference.<label>.t_ref_from_cam`.
- This is sufficient for three-ZED relative fusion in the reference camera frame.
- It is not sufficient by itself to define the canonical robot training frame. For DP3-style real data, we still need either `T_robot_base_from_ref_camera` or an explicit table/world frame transform that is fixed across collection and inference.
- Existing scripts in `geometry_awareness_manipulation` include robot/base-related utilities such as `detect_charuco_to_robot_point.py`, but the new RoboTwin real collection path should treat reference-camera-to-robot-base calibration as a first-class config input.

### First implementation
- Added `script/real_zed_collection/` as a separate execution folder so real collection code does not mix with `policy/DP3`.
- `collect_zed_robotwin_raw.py` keeps the original button semantics and robot joint-space recording path while replacing RealSense RGB capture with three ZED RGB-D capture threads.
- Raw data is saved as one manifest plus per-frame `robot_*.npz` and `<camera>_*.npz` files.
- `segment_objects_sam.py` is separate from collection and writes masks under `<mask_root>/<placeholder>/<camera>/mask_XXXXXX.png`.
- `postprocess_raw_to_robotwin_hdf5.py` converts one raw episode plus masks into RoboTwin-compatible HDF5 with `/joint_action/vector`, `/pointcloud`, `/object_pointcloud/{placeholder}`, and per-camera observation groups.
- First version uses `frame_mode=reference_camera`; robot-base/table-frame support remains an explicit extension.

### Workspace crop extension requirements (2026-04-27)
- The collected raw data currently uses the `global` ZED camera as the reference/world frame. This is useful for fusion but awkward for stable physical workspace crop ranges.
- The desired next version should define an explicit `workspace` frame from a deliberately placed Charuco board after camera extrinsic calibration.
- Calibration health can be checked without robot hand-eye calibration by detecting the same workspace board from multiple ZEDs, transforming each camera's `T_cam_from_board` into the current reference frame, and measuring pairwise residuals. Large translation/rotation residuals indicate the three-camera extrinsics or physical camera positions have drifted.
- Future collection should be allowed to save cropped RGB/depth to reduce disk use, but each cropped frame must preserve enough metadata for later point-cloud reconstruction: crop pixel box, adjusted crop intrinsics, original image/depth shape, `T_workspace_from_camera`, and workspace bbox parameters.
- It is safer to optionally save occasional full-frame debug snapshots even when normal frames are cropped.

### Workspace crop implementation notes (2026-04-27)
- `script/real_zed_collection/calibrate_workspace_frame.py` defines the workspace frame from a single deliberately placed Charuco board and writes a calibration YAML that still contains the original three-camera calibration plus `workspace` and `relative_to_workspace` sections.
- Calibration health is measured by transforming each camera's observed board pose into the reference-camera frame and comparing all samples against the averaged anchor pose.
- `load_three_zed_calibration(..., frame_mode="workspace")` now returns `t_world_from_cam` as `T_workspace_from_camera`.
- Live cropped collection stores cropped `rgb`, cropped `depth_m`, crop-adjusted `camera_matrix`, original shapes, ROI boxes, workspace bounds, and `t_workspace_from_camera` per camera NPZ.
- When `frame_mode=workspace`, postprocess and point-cloud export prefer `workspace_calibration_snapshot_path` / `workspace_calibration_path` from the manifest.
- Existing full raw episodes can be reprocessed with `postprocess_raw_to_robotwin_hdf5.py --frame_mode workspace --workspace_crop_* ...`; this crops fused scene/object point clouds in workspace coordinates and stores cropped per-camera RGB/depth observations.
- The first workspace anchor file treated Charuco board +Z as workspace +Z. For the user's tabletop setup, board +Z points into the table, so the correct long-term convention is `workspace +Z = board -Z` with a right-handed rotation `diag(1,-1,-1)`.
- Old raw episodes can have camera label and serial order mismatches. The observed example mapped `raw global -> calibration right`, `raw left -> calibration global`, and `raw right -> calibration left`. Serial-based remapping is now the correct default for preview/postprocess/cropped collection.
- With cropped collection and `workspace_crop_resize_rgb=false`, RGB is saved at the crop ROI's native camera resolution. Full-frame HD1080 RGB for all three cameras would add about 18 MB/frame uncompressed before any depth data, so full-frame RGB should be kept as debug snapshots unless compressed image storage is added.

## New Task: `pour_kettle_mug` (2026-04-16)

### Requirements
- Implement a new environment task named `pour_kettle_mug`.
- Use only the left arm.
- Grasp `009_kettle`, keep `039_mug` untouched on the table, and move the spout above the mug before tilting into a pouring pose.
- Use geometry-only success criteria; no liquid simulation is required.
- Randomize the mug position within a small region.
- Integrate the task with:
  - dynamic task import from `envs.{task_name}`
  - language instruction generation via `description/task_instruction/{task}.json`
  - object-pointcloud placeholder mapping
  - eval step limit configuration

### Research Findings
- `script/collect_data.py` and related entrypoints dynamically import tasks by `envs.{task_name}` and instantiate a same-named class.
- `Base_Task._init_task_env_()` initializes the scene, robot, camera, and calls `load_actors()` before running `play_once()`.
- `009_kettle` is a URDF articulation asset with:
  - contact point `0` on `link_1` usable as a handle grasp point
  - functional point `0` on `link_2` usable as a spout proxy
- `039_mug` is a standard actor asset with:
  - multiple contact points for generic grasping
  - functional point `1` at the mug bottom, suitable as a base reference for approximating the mug opening center
- Existing object descriptions already exist for both `009_kettle` and `039_mug`, so no new object-description JSON files are needed.
- Existing lightweight tests in `script/` prefer parsing source files or checking config/text integration rather than requiring a full simulator rollout.

### Implementation Direction
- Favor a source-level integration test first, not a heavy simulator test.
- Use a minimal new test to verify:
  - the task source exists
  - the class name matches the module name
  - the source includes the expected info placeholders and pointcloud-target wiring assumptions where possible
- Then implement the environment file and task metadata changes.

### Verification Findings
- The new source-level integration test `python script/test_pour_kettle_mug_task_integration.py` now passes with `5` tests.
- `python -m py_compile envs/pour_kettle_mug.py script/test_pour_kettle_mug_task_integration.py envs/object_pointcloud_targets.py` passes.
- Direct runtime module import for any `envs.*` task currently fails in this shell environment because `sapien` is not installed here; the same failure occurs for an existing task (`envs.place_object_basket`), so this is an environment limitation rather than a `pour_kettle_mug`-specific regression.
- Remaining risk: the final pouring quaternion and spawn ranges have not been validated in a live simulator rollout inside this session.

## Requirements
- User explicitly invoked `planning-with-files`.
- Initialize persistent planning files in the project root for this repository.
- Keep future discovery, especially external or untrusted content, in this file rather than `task_plan.md`.
- Current repository goal: evaluate whether the 3D representation method improves VLA task success rate, with edits mainly under `policy/DP3`.
- Current near-term task: check whether `policy/DP3/train_objpc_sam3.sh` can work and estimate how much time is needed to fuse object PCDs for one frame using the segment-and-project path.
- Longer-term direction: improve NDF and semantic training separately, then integrate them together and compare against baseline.
- New task on 2026-04-10: replace SAM-based segmentation with simulator-provided segmentation to build fused object point clouds for DP3 VLA training and evaluation, and add matching train/eval bash scripts.
- New task on 2026-04-10 (later): diagnose why `train_ndf_pointwise` is not improving over `train_objpc`, while `train_semantic_pointwise` improves substantially.

## Research Findings
- No existing `task_plan.md`, `findings.md`, or `progress.md` were present in the repository root at initialization.
- The `planning-with-files` templates are available under `/home/zheng/.agents/skills/planning-with-files/templates/`.
- The workflow expects project-local planning files and ongoing updates after each phase or meaningful discovery.
- `policy/DP3/train_objpc_sam3.sh` first preprocesses data into `./data/${task_name}-${task_config}-${expert_data_num}-objpc-sam3.zarr` by calling `process_data_objpc_sam3.sh` if the zarr directory does not already exist.
- The same script then trains inside `policy/DP3/3D-Diffusion-Policy` with `python train_dp3.py --config-name=robot_dp3_objpc.yaml`, overriding `task.dataset.zarr_path` to the generated `-objpc-sam3.zarr` dataset.
- Relevant SAM3 object-PCD code exists under `policy/DP3/scripts/sam3_pointcloud_utils.py` and a dedicated benchmark entrypoint exists at `policy/DP3/scripts/benchmark_objpc_sam3.py`.
- `policy/DP3/process_data_objpc_sam3.sh` runs `python scripts/process_data_objpc_sam3.py` and therefore assumes the caller's working directory is `policy/DP3`.
- `policy/DP3/train_objpc_sam3.sh` also assumes the caller's working directory is `policy/DP3`: it invokes `bash process_data_objpc_sam3.sh`, checks `./data/...`, and runs `cd 3D-Diffusion-Policy`.
- The raw dataset directory `data/hanging_mug/demo_clean_3d_object_pc` exists in the repo root, and prior DP3 zarr outputs already exist under `policy/DP3/data/`.
- The SAM3 model file `/home/zheng/Datasets/sam3/sam3.pt` exists.
- `robot_dp3_objpc_sam3.yaml` differs from `robot_dp3_objpc.yaml` mainly by using `demo_task_objpc_sam3` and adding the `SAM3` tag; the task configs have the same shape metadata and differ mainly in the default dataset suffix (`-objpc.zarr` vs `-objpc-sam3.zarr`).
- The DP3 dataset loader resolves `zarr_path` relative to `diffusion_policy_3d/dataset/robot_dataset.py`, so `../../../data/...` correctly points to `policy/DP3/data`.
- Before patching, running `bash policy/DP3/train_objpc_sam3.sh ...` from the repo root failed immediately because the script could not locate `process_data_objpc_sam3.sh`, could not `cd` into `3D-Diffusion-Policy`, and then tried to open `train_dp3.py` from the wrong directory.
- Runtime blockers found and addressed in the SAM3 extraction path:
- Missing `Path` import in `sam3_pointcloud_utils.py`
- Missing `ftfy` package in the active `RoboTwin` environment
- Local CLIP `SimpleTokenizer` was not callable in the way Ultralytics SAM3 expects
- SAM3 returned 6-value boxes while the tracker's bbox reuse path required 4-value XYXY boxes
- After patching, `policy/DP3/scripts/benchmark_objpc_sam3.py` completed successfully for `hanging_mug / demo_clean_3d_object_pc / episode 0` over 5 frames using prompts `{A}: mug` and `{B}: rack`.
- Measured benchmark timing on this machine:
- `frame_total_ms`: first `28719.51`, mean `22544.42`, p95 `27344.03`
- `frame_total_ms_steady`: mean `21000.64`, p95 `21711.73`
- `{A}_ms`: first `17554.46`, mean `11924.64`, p95 `16277.47`
- `{B}_ms`: first `11165.04`, mean `10619.78`, p95 `11122.44`
- The benchmark ran with `torch.cuda.is_available() == False` and `torch.cuda.device_count() == 0`, so the measured fusion cost is CPU-only.
- There is already a visualization utility at `policy/DP3/scripts/visualize_objpc_zarr.py` that reads a DP3 zarr, optionally overlays per-placeholder point clouds, and can export `.ply` files.
- `open3d` is installed in the active `RoboTwin` environment, so the visualizer can run here.
- Dry-running the visualizer on `policy/DP3/data/hanging_mug-demo_clean_3d_object_pc-50-objpc.zarr` showed that older `objpc` zarrs may only contain the merged `data/point_cloud` dataset and may not include `object_point_cloud_A/B`.
- `process_data_objpc_sam3.py` writes per-placeholder datasets like `object_point_cloud_A` and `object_point_cloud_B` when `--save_placeholder_point_clouds` is enabled, and `process_data_objpc_sam3.sh` already passes that flag.
- The repeated warning `imgsz=[640] must be multiple of max stride 14, updating to [644]` comes from Ultralytics `check_imgsz(...)`.
- Current `SAM3ProjectiveTracker` creates `SAM3SemanticPredictor` with overrides that do not set `imgsz`, so the predictor falls back to `640`.
- For this SAM3 path, stride is `14`, so `640` is invalid and Ultralytics promotes it to `644` while printing a warning on each invocation.
- The correct fix is to set an explicit valid `imgsz` in the predictor overrides, rather than suppressing logs globally.
- Mid-preprocessing, `hanging_mug-demo_clean_3d_object_pc-50-objpc-sam3.zarr` contained only `.zgroup`, `data/.zgroup`, and `meta/.zgroup`. That is expected with the old preprocessing implementation because it buffered arrays in memory and only created `state`, `action`, `point_cloud`, and `episode_ends` after all episodes finished.
- The earlier `KeyError('state')` is consistent with an interrupted preprocessing run: the final zarr directory existed, but required datasets had never been written.
- `train_objpc_sam3.sh` now validates that `data/point_cloud`, `data/state`, `data/action`, and `meta/episode_ends` exist before skipping preprocessing. If not, it invokes preprocessing again and lets the preprocessing script resume or reset as needed.
- A stale `...objpc-sam3.zarr.tmp` from the older temp-publish approach is now treated as legacy state; the current implementation writes directly to the final `.zarr` per episode.
- `robot_dp3_objpc_sam3.yaml` now includes `_self_` in the Hydra defaults list, which removes the missing-`_self_` warning.
- Current recheck: `policy/DP3/data/hanging_mug-demo_clean_3d_object_pc-50-objpc-sam3.zarr` does not exist.
- Current recheck: `policy/DP3/data/hanging_mug-demo_clean_3d_object_pc-50-objpc-sam3.zarr.tmp` exists, but it only contains `.zgroup`, `data/.zgroup`, and `meta/.zgroup`; `data/` and `meta/` still have no arrays.
- This means the preprocess job has not yet reached the final write stage. If the user-facing job is still printing `processing episode: ...`, this state is consistent with the current implementation. If the user-facing job has already exited, then it stopped before publishing any datasets.
- The preprocessing path has now been changed to write incrementally per episode through DP3's `ReplayBuffer.add_episode(...)` instead of buffering all episodes in Python lists.
- New helper module: `policy/DP3/scripts/incremental_objpc_zarr.py`.
- The incremental-write test passes: `policy/DP3/scripts/test_incremental_objpc_zarr.py`.
- `process_data_objpc_sam3.py` now opens or resets a resumable replay buffer at the final `.zarr` path, resumes from `buffer.n_episodes`, and writes metadata JSON after each completed episode.
- `train_objpc_sam3.sh` no longer deletes incomplete SAM3 zarr outputs before preprocessing; it now reports that preprocessing will resume.
- This change reduces risk substantially: if preprocessing stops after episode `k`, episodes `0..k-1` remain persisted in the final `.zarr` and are visible immediately on disk.
- Ultralytics `SAM3VideoSemanticPredictor` resets its source on every `predictor(...)` call because it goes through `BasePredictor.stream_inference(...) -> setup_source(...)`, so tracker state is not preserved automatically across separate calls.
- `SAM3VideoSemanticPredictor.init_state()` asserts `dataset.mode == "video"` and allocates prompt state as `[None] * num_frames`, which means the stock video-tracker path is designed for a finite indexed sequence, not an open-ended live stream.
- The same class does support prompt updates within one loaded sequence through `add_prompt(frame_idx=..., text=..., bboxes=..., labels=...)`; text prompts are global to the sequence, while geometric prompts are frame-indexed.
- `LoadStreams` in Ultralytics uses `mode = "stream"` and can represent effectively infinite sources, but that does not match the stock SAM3 video predictor's `dataset.mode == "video"` requirement.
- Practical implication for this repo: using the SAM3 video tracker per episode or per camera sequence is structurally compatible, but true endless online tracking would require a wrapper, chunked/sliding-window processing, or lower-level state management instead of repeated top-level `predictor(source=..., stream=True)` calls.
- The current eval path already has an online SAM3 hook: `policy/DP3/eval_objpc_sam3.sh` launches `script/eval_policy.py` with `--config_name robot_dp3_objpc_sam3`.
- In eval, `policy/DP3/deploy_policy.py` sets `use_sam3_objpc = "objpc_sam3" in usr_args["config_name"]`, builds one `SAM3ProjectiveTracker`, and stores it on the loaded policy model.
- During each eval step, `encode_obs(...)` calls `extract_placeholder_point_cloud_sam3_online(...)` for every placeholder, merges the per-placeholder clouds, and writes the merged result into `obs["point_cloud"]`.
- The current online SAM3 eval path does not use `SAM3VideoSemanticPredictor`; it uses `SAM3SemanticPredictor` through `SAM3ProjectiveTracker`, with a simple tracking heuristic: reuse the previous frame's bbox when available, otherwise refresh with text, and force text refresh every `sam3_text_refresh_every` frames.
- Eval resets SAM3 state at episode boundaries in `reset_model(...)` by clearing `sam3_tracking_state` and setting `sam3_frame_idx = 0`.
- The eval shell entrypoint exposes the main SAM3 knobs already: `sam3_prompt_map`, `sam3_model`, `sam3_camera_names`, and `sam3_text_refresh_every`.
- The current `hanging_mug/demo_clean_3d_object_pc` dataset stores `head_camera` and `front_camera` RGB frames as JPEG-encoded `240x320` images; direct decode of episode 0 frame 0 confirmed shape `(240, 320, 3)` for both cameras.
- The default camera config in `task_config/_camera_config.yml` uses `D435: w=320, h=240` and also provides `Large_D435: w=640, h=480`.
- `task_config/demo_clean_3d_object_pc.yml` currently sets `head_camera_type: D435` and `wrist_camera_type: D435`, so the low resolution is coming from task config rather than from the SAM3 code path.
- `sam3_pointcloud_utils.py` projects the full 3D scene point cloud into mask pixels using the stored `intrinsic_cv`/`extrinsic_cv` matrices and the mask image size; lower RGB resolution therefore directly reduces mask detail and the number/stability of points selected for the object cloud, especially for small or thin structures.
- Raising RGB resolution should help SAM-based segmentation and 3D selection quality, but the benefit is on the segment-and-project stage, not on the DP3 policy directly. If eval also uses online SAM, the eval environment resolution needs to be raised too; improving only the offline training demos will not fix low-resolution online eval masks.
- Because `Large_D435` is `640x480`, moving from `320x240` to `640x480` increases input pixels by 4x, so SAM runtime and storage/render cost will rise substantially. A small A/B pilot before recollecting the full dataset is the pragmatic next step.
- The simulator already exposes two segmentation sources through `envs/_base_task.py`: `mesh_segmentation` and `actor_segmentation`.
- The simulator also already exposes direct per-placeholder oracle object point clouds through `observation["object_pointcloud"]` when `data_type.object_pointcloud=true`.
- Existing non-SAM training and eval entrypoints already exist:
- `policy/DP3/train_objpc.sh`
- `policy/DP3/eval_objpc.sh`
- Existing online eval logic in `policy/DP3/deploy_policy.py` already prefers `observation["object_pointcloud"]` when `use_object_pointcloud` is enabled and `use_sam3_objpc` is false.
- Existing offline fallback logic in `policy/DP3/scripts/object_pointcloud_utils.py` can derive object point clouds from simulator segmentation, but it currently only loads `head_camera` segmentation from the dataset and does not implement the newer SAM-style multi-camera fused path.
- `envs/camera/camera.py` returns segmentation as a colorized image derived from the simulator's `Segmentation` buffer:
- channel 0 for mesh-level labels
- channel 1 for actor-level labels
- `mesh_segmentation` and `actor_segmentation` are not equivalent in this repo:
- `mesh_segmentation` uses `Segmentation[..., 0]` and corresponds to visual-shape / mesh-level labels.
- `actor_segmentation` uses `Segmentation[..., 1]` and corresponds to actor/entity-level labels, which is the closer match to per-object instance masks.
- Current simulator segmentation export colorizes the label map for storage/visualization instead of preserving raw integer ids; for a robust placeholder-specific projection path, using raw actor ids internally would be preferable to reusing the colorized output as the canonical representation.
- The oracle `object_pointcloud` path filters camera point clouds directly by actor ids using `Segmentation[..., 1]`, so it is naturally aligned with `actor_segmentation`, not `mesh_segmentation`.
- Collected `scene_info.json` files for `*_object_pc` datasets already contain placeholder-to-actor-id mappings under `episode_k.object_pointcloud.targets.{placeholder}.actor_ids`; this is sufficient to drive offline actor-segmentation projection without using stored oracle point clouds.
- There is currently no local dataset under `data/` that already contains saved `actor_segmentation` or `mesh_segmentation` frames, so end-to-end validation of the new pipeline will require recollecting a small dataset or enabling segmentation in an eval task config.
- Implemented a new `objpc_actorseg` pipeline:
- offline utility: `policy/DP3/scripts/actorseg_pointcloud_utils.py`
- offline preprocessing: `policy/DP3/scripts/process_data_objpc_actorseg.py`
- shell entrypoints: `policy/DP3/process_data_objpc_actorseg.sh`, `policy/DP3/train_objpc_actorseg.sh`, `policy/DP3/eval_objpc_actorseg.sh`
- Hydra configs: `robot_dp3_objpc_actorseg.yaml`, `demo_task_objpc_actorseg.yaml`
- collection/eval task config templates: `task_config/demo_clean_3d_actorseg.yml`, `task_config/demo_randomized_3d_actorseg.yml`
- Online eval support for the new pipeline is now wired into `policy/DP3/deploy_policy.py` via a dedicated `use_actorseg_objpc` branch.
- `envs/_base_task.py` now preserves object-target metadata when `actor_segmentation` or `mesh_segmentation` is enabled, even if `data_type.object_pointcloud` is false; this is intended so actor-segmentation datasets can still save placeholder->actor_ids in `scene_info.json` without storing per-frame oracle object point clouds.
- The new offline actor-segmentation preprocessing path is fail-fast: if the requested cameras do not contain saved `actor_segmentation`, it raises an explicit error instead of silently generating zero point clouds.
- Negative verification against the existing `demo_clean_3d_object_pc` dataset confirmed this guard works as intended: preprocessing aborts with `missing_actor_segmentation` for `head_camera` and `front_camera`.
- The new actor-segmentation extraction path uses the stored colorized segmentation images and placeholder actor ids to select projected scene points. This mirrors the SAM segment-project-fuse structure, but depends on the current color-palette encoding of simulator labels.
- For the new requested work, the likely cleanest replacement for SAM is not direct `object_pointcloud`, but a new fused multi-camera simulator-segmentation projection path that mirrors the structure of the SAM3 pipeline while removing the learned 2D segmenter.
- `train_objpc.sh`, `train_ndf_pointwise.sh`, and `train_semantic_pointwise.sh` are not a controlled apples-to-apples comparison:
- `train_semantic_pointwise.sh` changes the training recipe by default (`training.use_ema=false`, `dataloader.batch_size=110`, `val_dataloader.batch_size=110`), while `objpc` and `ndf_pointwise` keep the base DP3 defaults (`use_ema=True`, `batch_size=256`).
- The saved Hydra overrides confirm the existing local runs used those differing defaults:
- `objpc`: only dataset path / setting overrides
- `ndf_pointwise`: only dataset path / extra obs key overrides
- `semantic_pointwise`: dataset path / extra obs key overrides plus `training.use_ema=false` and `batch_size=110`
- For the local `hanging_mug` experiments currently on disk, both `ndf_pointwise` and `semantic_pointwise` only augment placeholder `{A}` (the mug), not `{B}` (the rack). `feature_placeholders` in both metadata files are `['{A}']`.
- In `deploy_policy.py`, pointwise NDF / semantic runs do not keep the raw `{A}` object cloud inside the primary `point_cloud`. Instead:
- primary `point_cloud` becomes only the context placeholders (here `{B}` rack)
- the mug is provided separately via `ndf_point_cloud_A` or `semantic_point_cloud_A`
- Therefore this is not “baseline objpc plus extra feature”; it is a different observation factorization.
- `DP3Encoder` creates one PointNet branch per point-cloud observation key. So baseline `objpc` has one point-cloud branch, while `ndf_pointwise` / `semantic_pointwise` have two point-cloud branches (`point_cloud` + feature cloud), meaning the encoder architecture also changes.
- The pointwise feature branches are both consumed by `PointNetEncoderXYZRGB`, which is a generic per-point MLP + max-pool encoder. There is no NDF-specific alignment or correspondence module on top of the NDF field inside DP3.
- The saved NDF zarr and semantic zarr both contain 50 episodes (`meta/episode_ends` length 50) and the same step counts as the baseline objpc zarr, so the current comparison is using aligned DP3 training data volumes.
- The saved metadata for both pointwise variants shows the training set covered mug assets:
- `039_mug/base0`
- `039_mug/base1`
- `039_mug/base2`
- `039_mug/base3`
- `039_mug/base4`
- `039_mug/base6`
- `039_mug/base7`
- `039_mug/base8`
- `039_mug/base9`
- This means the training artifacts already span 9 mug instances from the same asset family, not a single mug instance.
- `envs/hanging_mug.py` samples `self.mug_id = np.random.choice([i for i in range(10)])`, so eval also draws from the same 10-instance `039_mug` family. This is not a strong novel-object benchmark; at most one mug instance (`base5`) is absent from the saved training metadata.
- `policy/DP3/deploy_policy.yml` sets `instruction_type: unseen`, and `script/eval_policy.py` writes this value into `_result.txt`. In this repo, “unseen” refers to unseen language instructions, not unseen object instances.
- The saved semantic eval result file reports `Instruction Type: unseen` and success `0.36`; this should not be interpreted as a novel-object result.
- `train_ndf_pointwise.sh` contains a bug in the preprocessing call: it passes `"${ndf_dgcnn_/placeholders}"` instead of `"${ndf_dgcnn_placeholders}"`. This prevents the user from correctly forwarding `ndf_dgcnn_placeholders` through the train script.
- That forwarding bug does not affect the user's current pointnet-backed NDF setup when `ndf_dgcnn_placeholders` is intentionally empty, but it would break any future DGCNN-backed NDF run.
- The currently used NDF checkpoint in `policy/DP3/Command.md` is `/home/zheng/train_results/Mug/checkpoints/model_current.pth`, not `model_best.pth`.
- Checkpoint-architecture compatibility check:
- `model_current.pth` cleanly matches `dgcnn=False` and mismatches under `dgcnn=True`
- `model_best.pth` cleanly matches `dgcnn=True` and mismatches under `dgcnn=False`
- This means the local NDF experiment is currently using a pointcloud-backbone “current” checkpoint rather than the DGCNN-backed “best” checkpoint.
- Because of the `train_ndf_pointwise.sh` forwarding bug, switching to the DGCNN-backed best checkpoint through the train script is currently blocked.
- The saved pointwise feature statistics suggest semantic features are much more temporally stable than NDF features in the current pipeline:
- over the first 200 frames, NDF frame-mean feature std is about `0.0108`
- over the first 200 frames, semantic frame-mean feature std is about `0.0012`
- flattened adjacent-frame cosine similarity is about `0.764` for NDF vs `0.999` for semantic
- These are rough probes, not full invariance metrics, but they are consistent with semantic features behaving like a much more stable object/part code and NDF behaving like a more frame-sensitive local geometry field.
- Both NDF and semantic pointwise features are unit-norm on disk (mean feature norm about `1.0`), so the underperformance signal is not explained by a simple feature-scale explosion.
- The current raw dataset folder under `data/hanging_mug/demo_clean_3d_object_pc` now contains only `episode0.hdf5` and `scene_info.json` with only `episode_0`, while the saved DP3 zarrs clearly contain 50 episodes. Therefore the current raw demo directory is no longer the exact source snapshot used to build the existing 50-episode artifacts.
- The requested fairness cleanup has now been applied:
- `train_ndf_pointwise.sh` correctly forwards `ndf_dgcnn_placeholders`
- `train_semantic_pointwise.sh` now defaults back to `batch_size=256`, `val_batch_size=batch_size`, and `use_ema=true`
- `use_ema` in this codebase means training and saving an exponential moving average copy of the policy weights. In practice it usually stabilizes eval and often gives a modest improvement, but the impact is task-dependent rather than guaranteed to be large.
- New task on 2026-04-13: explain how `train_ndf_pointwise_hybrid.sh` combines baseline object-PCD observations with NDF pointwise features, then build the analogous `semantic_pointwise_hybrid` path.
- New task on 2026-04-14: add actorseg-based hybrid training/eval for both NDF and semantic, mirroring the successful `objpc + feature-branch` hybrid structure while using mask-project-fuse object point clouds like `train_objpc_actorseg`.
- User-provided experiment evidence: the existing global-NDF path (`train_ndf.sh` / `robot_dp3_ndf`) was already tested using `policy/DP3/checkpoints/hanging_mug-demo_clean_3d_object_pc-objpc-ndf-50_0`, and it caused a severe behavioral regression: the robot often failed to approach the mug, consistent with losing or underusing object location information.
- This is consistent with the current global-NDF integration design:
- `compute_ndf_feature(...)` explicitly normalizes the object point cloud by centering and scaling it before extracting the descriptor, making the NDF feature pose-invariant by construction, see `normalize_object_point_cloud(...)` in [ndf_feature_utils.py](/home/zheng/github/RoboTwin_geo/policy/DP3/scripts/ndf_feature_utils.py#L431).
- `process_data_ndf.py` stores that descriptor as a low-dimensional extra observation `ndf_feat_A`, not as a localized point set, while the scene point cloud remains a merged `{A}+{B}` cloud, see [process_data_ndf.py](/home/zheng/github/RoboTwin_geo/policy/DP3/scripts/process_data_ndf.py#L153).
- Therefore the policy receives an object-identity / geometry descriptor with no explicit correspondence back to the mug points inside the merged scene cloud. This makes “global NDF causes localization regression” a credible integration failure mode rather than evidence that NDF is fundamentally useless.
- Current best diagnosis for the NDF underperformance is:
- Most likely: the current NDF integration path is suboptimal for DP3.
- Evidence:
- NDF pointwise observations are fed as a separate high-dimensional point-cloud branch into the generic `PointNetEncoderXYZRGB` path, not a correspondence-aware or geometry-aware downstream head.
- The primary `point_cloud` no longer contains the `{A}` object when NDF pointwise is enabled; `{A}` is isolated into its own branch, so the policy sees a different factorization than baseline `objpc`.
- NDF pointwise features are materially less temporally stable than semantic pointwise features in the current saved dataset probes.
- Secondary likely cause: the current benchmark underexposes the kind of generalization where NDF would be expected to help most.
- Evidence:
- current eval is “unseen instruction”, not unseen object
- train/eval both operate on the same `039_mug` family with broad overlap of mug ids
- Less likely but still plausible: the current pointnet-backed NDF checkpoint is not the strongest available NDF model for this use case.
- Evidence:
- command examples use `model_current.pth`
- `model_best.pth` corresponds to a different backbone family (`dgcnn=True`)
- There is not enough evidence to conclude “NDF cannot help this task.” The current evidence only supports “this particular NDF integration + benchmark setup is not showing an advantage.”
- The approved mitigation experiment has now been implemented as a separate `ndf_pointwise_hybrid` path rather than changing the existing `ndf_pointwise` behavior in place.
- `policy/DP3/scripts/pointwise_context_utils.py` now owns the shared logic for deciding whether feature placeholders stay in the main merged `point_cloud`.
- `policy/DP3/deploy_policy.py` now treats config names containing `ndf_pointwise_hybrid` as a distinct runtime mode:
- `ndf_point_cloud_A/B` generation is unchanged
- the main `point_cloud` keeps all placeholders instead of dropping feature placeholders
- `policy/DP3/scripts/process_data_ndf_pointwise.py` now supports `--keep_feature_placeholders_in_context`, and the new wrapper `policy/DP3/scripts/process_data_ndf_pointwise_hybrid.py` uses that flag together with the `-objpc-ndf-pointwise-hybrid` dataset suffix.
- New hybrid entrypoints now exist:
- `policy/DP3/process_data_ndf_pointwise_hybrid.sh`
- `policy/DP3/train_ndf_pointwise_hybrid.sh`
- `policy/DP3/eval_ndf_pointwise_hybrid.sh`
- `policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/robot_dp3_ndf_pointwise_hybrid.yaml`
- `policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/task/demo_task_ndf_pointwise_hybrid.yaml`
- The focused regression test `policy/DP3/scripts/test_ndf_pointwise_hybrid.py` now proves:
- old `ndf_pointwise` excludes `{A}` from the main `point_cloud`
- new `ndf_pointwise_hybrid` keeps `{A}` and `{B}` in the main `point_cloud`
- both still emit `ndf_point_cloud_A`
- Fresh verification evidence for the hybrid work:
- `python policy/DP3/scripts/test_ndf_pointwise_hybrid.py` passed with `2` tests
- `python -m py_compile policy/DP3/deploy_policy.py policy/DP3/scripts/process_data_ndf_pointwise.py policy/DP3/scripts/process_data_ndf_pointwise_hybrid.py policy/DP3/scripts/pointwise_context_utils.py policy/DP3/scripts/test_ndf_pointwise_hybrid.py` passed
- `bash -n policy/DP3/process_data_ndf_pointwise_hybrid.sh` passed
- `bash -n policy/DP3/train_ndf_pointwise_hybrid.sh` passed
- `bash -n policy/DP3/eval_ndf_pointwise_hybrid.sh` passed
- `train_ndf_pointwise_hybrid.sh` keeps the baseline merged raw `point_cloud` because its preprocess wrapper adds `--keep_feature_placeholders_in_context`, while still injecting `ndf_point_cloud_A/B` through `+task.dataset.extra_obs_keys=[...]` and dynamic shape overrides. This is the exact pattern to mirror for a semantic hybrid.
- The current semantic pointwise path still removes feature placeholders from the main `point_cloud`:
  - offline preprocessing in `process_data_semantic_pointwise.py` sets `context_placeholders = placeholders - feature_placeholders`
  - runtime `deploy_policy.py` only keeps feature placeholders in the main `point_cloud` when `use_ndf_pointwise_hybrid=True`
  - there is no `use_semantic_pointwise_hybrid` branch yet
- The minimum semantic-hybrid delta is therefore:
  - allow semantic preprocessing to accept `--keep_feature_placeholders_in_context`
  - add a thin wrapper `process_data_semantic_pointwise_hybrid.py`
  - add matching `train/eval/process` shell entrypoints and Hydra config/task-config names
  - extend runtime `deploy_policy.py` with a `use_semantic_pointwise_hybrid` mode that reuses the same context-building helper as NDF hybrid
- The requested `semantic_pointwise_hybrid` path has now been implemented with the same observation structure as the NDF hybrid:
  - main `point_cloud` keeps the baseline merged raw ObjPC
  - `semantic_point_cloud_A/B` remain extra point-cloud branches
  - low-dim `agent_pos` is unchanged
- Important encoder clarification:
  - DP3 does not concatenate raw point cloud, semantic pointwise cloud, and agent state at the point level
  - instead, [DP3Encoder](/home/zheng/github/RoboTwin_geo/policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/model/vision/pointnet_extractor.py#L194) encodes each point-cloud key with its own PointNet extractor, encodes low-dim state with a separate MLP, then concatenates the branch-level features
  - therefore `semantic_pointwise_hybrid` means `f_raw_objpc || f_semantic_pointwise || f_agent_pos`, not `[xyz|rgb|semantic|state]` inside one shared PointNet
- New semantic hybrid entrypoints now exist:
  - `policy/DP3/scripts/process_data_semantic_pointwise_hybrid.py`
  - `policy/DP3/process_data_semantic_pointwise_hybrid.sh`
  - `policy/DP3/train_semantic_pointwise_hybrid.sh`
  - `policy/DP3/eval_semantic_pointwise_hybrid.sh`
- New task on 2026-04-14 (later): add a richer actor-segmentation data collection config so actorseg extraction can use per-camera raw geometry instead of the already-downsampled scene pointcloud, and verify how close the upgraded actorseg clouds are to `object_pointcloud`.
- Current actorseg offline training data in `data/hanging_mug/demo_clean_3d_actorseg/data/episode0.hdf5` contains:
  - `observation/*/actor_segmentation`
  - `observation/*/intrinsic_cv`
  - `observation/*/extrinsic_cv`
  - `observation/*/cam2world_gl`
  - RGB
  - a precomputed whole-scene `pointcloud (T, 1024, 6)`
- The same file does **not** contain per-camera raw `Position` buffers or depth maps, so current actorseg training cannot be losslessly upgraded to `objpc`-quality extraction without recollecting data.
- The current actorseg collection configs also confirm this limitation:
  - `task_config/demo_clean_3d_actorseg.yml` has `depth: false`
  - `task_config/demo_randomized_3d_actorseg.yml` has `depth: false`
  - both keep `actor_segmentation: true` and `pointcloud: true`
- The current `objpc` path and current actorseg path are structurally different:
  - `objpc` in `envs/camera/camera.py` directly filters raw per-camera `Position` pixels by actor ids from `Segmentation[..., 1]`, merges per-camera object points, then downsamples per object
  - actorseg in `policy/DP3/scripts/actorseg_pointcloud_utils.py` starts from an already-downsampled whole-scene `pointcloud` and projects those sparse points back into actor-segmentation images to recover A/B
- Because the whole-scene `pointcloud` is already downsampled to `1024` points in `envs/camera/camera.py` before actorseg extraction sees it, current actorseg can differ substantially from `objpc`, especially for fine structures like mug handles and rack hooks.
- If actorseg extraction is changed to use the same per-camera raw `Position` buffer as `objpc`, together with the same actor-segmentation truth mask and the same cameras, then actorseg and `objpc` should become very close in practice. The major current gap is implementation order, not label quality.
- New task on 2026-04-14 (later): raise the `object_pointcloud` path to 5000 points without breaking existing 1024-point checkpoints.
- Current `object_pc` training and evaluation cannot be switched to 5000 points by changing only the collection config:
  - collection-side `object_pointcloud.point_num` controls how many real object points are saved into HDF5
  - preprocessing `process_data_objpc.py` still defaults to `--target_num_points 1024`
  - DP3 task config `demo_task_objpc.yaml` still declares `point_cloud.shape: [1024, 6]`
- Existing collected `demo_clean_3d_object_pc` data is already stored as `object_pointcloud/{A,B}: (T, 1024, 6)`, so it cannot be losslessly upgraded to real 5000-point object clouds without recollection.
- To avoid breaking existing 1024-point checkpoints, a separate 5000-point path is safer than mutating the current default path in place.
- The new isolated 5000-point ObjPC path now exists:
  - collection configs:
    - `task_config/demo_clean_3d_object_pc_5000.yml`
    - `task_config/demo_randomized_3d_object_pc_5000.yml`
  - DP3 configs:
    - `policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/task/demo_task_objpc_5000.yaml`
    - `policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/robot_dp3_objpc_5000.yaml`
  - scripts:
    - `policy/DP3/train_objpc_5000.sh`
    - `policy/DP3/eval_objpc_5000.sh`
- `policy/DP3/process_data_objpc.sh` now accepts an explicit fifth positional argument `target_num_points`, so the 5000-point training path can preprocess to 5000 without changing the old 1024-point path.
- Training impact expectation for `1024 -> 5000`:
  - point count grows by about `4.88x`
  - PointNet/point-cloud encoding cost and memory usually increase substantially, often near-linearly in practice
  - training speed should be expected to drop noticeably and GPU memory usage to rise materially
- New task on 2026-04-14 (later): keep the hybrid main raw `point_cloud` branch at `1024`, but raise only the NDF / semantic feature point-cloud branches to `5000`.
- Current DP3 hybrid structure makes this feasible:
  - `point_cloud` and `ndf_point_cloud_*` / `semantic_point_cloud_*` are encoded as separate branches
  - they are concatenated after branch-level encoding
  - there is no requirement that the raw branch and feature branch use the same point count or per-point alignment
- Important limitation:
  - if the underlying object point cloud source is still only `1024` real points, requesting `5000` feature points only causes repeated resampling / padding, not new geometry information
  - therefore the new `feature5000` path only makes sense together with recollected `5000`-point `object_pc` data
- A new isolated `feature5000` hybrid path now exists for both NDF and semantic:
  - NDF:
    - `policy/DP3/process_data_ndf_pointwise_hybrid_feat5000.sh`
    - `policy/DP3/train_ndf_pointwise_hybrid_feat5000.sh`
    - `policy/DP3/eval_ndf_pointwise_hybrid_feat5000.sh`
    - `policy/DP3/scripts/process_data_ndf_pointwise_hybrid_feat5000.py`
    - `policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/task/demo_task_ndf_pointwise_hybrid_feat5000.yaml`
    - `policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/robot_dp3_ndf_pointwise_hybrid_feat5000.yaml`
  - Semantic:
    - `policy/DP3/process_data_semantic_pointwise_hybrid_feat5000.sh`
    - `policy/DP3/train_semantic_pointwise_hybrid_feat5000.sh`
    - `policy/DP3/eval_semantic_pointwise_hybrid_feat5000.sh`
    - `policy/DP3/scripts/process_data_semantic_pointwise_hybrid_feat5000.py`
    - `policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/task/demo_task_semantic_pointwise_hybrid_feat5000.yaml`
    - `policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/robot_dp3_semantic_pointwise_hybrid_feat5000.yaml`
- The new `feature5000` paths keep:
  - main raw `point_cloud.shape = [1024, 6]`
  - `ndf_point_cloud_*` / `semantic_point_cloud_*` default point count = `5000`
  - distinct train/eval/checkpoint naming via the suffix `-feat5000`
- New task on 2026-04-15: fix `process_data_semantic_pointwise_hybrid_feat5000.sh` being killed during preprocessing.
- Root cause of the `Killed` failure:
  - `process_data_semantic_pointwise.py` and `process_data_ndf_pointwise.py` originally accumulated every frame of `point_cloud`, feature point clouds, `state`, and `action` in Python lists and only wrote zarr at the very end.
  - With `semantic_num_points=5000`, a single semantic frame is about `5000 x (3+128) x 4 bytes ~= 2.5 MB`.
  - For roughly `16750` training frames in a 50-demo `hanging_mug` run, semantic pointwise arrays alone are about `40.9 GB` before accounting for raw point clouds, actions/states, and Python container overhead.
  - Therefore the shell-side `Killed` is consistent with the Linux OOM killer, not with a Python exception or checkpoint incompatibility.
- The fix was applied at the shared base preprocess layer, not just the feat5000 wrappers:
  - `policy/DP3/scripts/process_data_semantic_pointwise.py`
  - `policy/DP3/scripts/process_data_ndf_pointwise.py`
- Both preprocessors now:
  - open or resume a replay buffer with `open_or_reset_replay_buffer(...)`
  - build arrays only for one episode at a time
  - append that episode immediately with `append_episode_to_buffer(...)`
  - update metadata JSON after each episode
- New shared metadata helper:
  - `policy/DP3/scripts/pointwise_preprocess_meta.py`
- Because the fix is in the base preprocess scripts, all wrapper paths that call them inherit the improvement automatically:
  - regular pointwise
  - hybrid
  - feat5000 wrappers
- The new `feat5000` training wrappers were also hardened against stale partial preprocess directories:
  - `policy/DP3/train_semantic_pointwise_hybrid_feat5000.sh`
  - `policy/DP3/train_ndf_pointwise_hybrid_feat5000.sh`
- They now validate that the target zarr contains at least `data/point_cloud`, `data/state`, `data/action`, and `meta/episode_ends` before skipping preprocessing. This avoids the old “directory exists but preprocess died halfway” failure mode after an OOM kill.
  - `policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/robot_dp3_semantic_pointwise_hybrid.yaml`
  - `policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/task/demo_task_semantic_pointwise_hybrid.yaml`
- Fresh verification evidence for the semantic hybrid work:
  - `python policy/DP3/scripts/test_semantic_pointwise_hybrid.py` passed with `3` tests
  - `python policy/DP3/scripts/test_ndf_pointwise_hybrid.py` still passed with `3` tests
  - `python -m py_compile policy/DP3/deploy_policy.py policy/DP3/scripts/process_data_semantic_pointwise.py policy/DP3/scripts/process_data_semantic_pointwise_hybrid.py policy/DP3/scripts/test_semantic_pointwise_hybrid.py` passed
  - `bash -n policy/DP3/process_data_semantic_pointwise_hybrid.sh` passed
  - `bash -n policy/DP3/train_semantic_pointwise_hybrid.sh` passed
  - `bash -n policy/DP3/eval_semantic_pointwise_hybrid.sh` passed
- New debugging result on 2026-04-14 for `eval_ndf_pointwise_hybrid.sh`:
  - The checkpoint mismatch was not caused by a changed model definition in training.
  - The user invocation omitted the explicit empty `ndf_dgcnn_placeholders` argument:
    - intended order: `... ndf_device ndf_dgcnn_placeholders object_placeholders checkpoint_num`
    - actual call placed `"{A},{B}"` into `ndf_dgcnn_placeholders` and `3000` into `object_placeholders`
  - Evidence:
    - eval log showed `[DP3Encoder] point cloud keys: ['point_cloud']`, which means `ndf_point_cloud_A` was never injected into the eval model shape meta
    - the checkpoint state dict *does* contain `obs_encoder.extractors.ndf_point_cloud_A.*`, which proves the training run that produced it had the NDF hybrid branch enabled
  - Therefore the checkpoint is structurally consistent with hybrid training; the eval-time CLI parsing was wrong.
  - Fix implemented: `policy/DP3/ndf_pointwise_arg_utils.sh` now normalizes both the explicit and legacy call forms, and the following scripts now auto-correct the omitted-empty-argument form:
    - `policy/DP3/eval_ndf_pointwise_hybrid.sh`
    - `policy/DP3/train_ndf_pointwise_hybrid.sh`
    - `policy/DP3/eval_ndf_pointwise.sh`
    - `policy/DP3/train_ndf_pointwise.sh`
- Current actorseg+hybrid design constraint:
  - training must use actorseg-projected fused object clouds as the main `point_cloud`
  - hybrid branches must still add `ndf_point_cloud_A/B` or `semantic_point_cloud_A/B` on top of that raw actorseg point cloud
  - eval must perform the same online actorseg extraction once per placeholder and reuse that same per-placeholder cloud for both the main merged `point_cloud` and the feature branches
- The cleanest implementation path is to add separate pipelines rather than mutating existing working ones:
  - `objpc_actorseg_ndf_pointwise_hybrid`
  - `objpc_actorseg_semantic_pointwise_hybrid`
- Actorseg hybrid runtime support is now implemented in `deploy_policy.py` by reusing the per-placeholder actorseg-extracted point clouds for both:
  - the merged raw `point_cloud`
  - the `ndf_point_cloud_A/B` or `semantic_point_cloud_A/B` feature branches
  This keeps online eval semantics aligned with offline preprocessing.
- New actorseg hybrid preprocessing entrypoints now exist:
  - `policy/DP3/scripts/process_data_ndf_pointwise_actorseg_hybrid.py`
  - `policy/DP3/scripts/process_data_semantic_pointwise_actorseg_hybrid.py`
  - `policy/DP3/process_data_ndf_pointwise_actorseg_hybrid.sh`
  - `policy/DP3/process_data_semantic_pointwise_actorseg_hybrid.sh`
- New actorseg hybrid train/eval entrypoints now exist:
  - `policy/DP3/train_ndf_pointwise_actorseg_hybrid.sh`
  - `policy/DP3/train_semantic_pointwise_actorseg_hybrid.sh`
  - `policy/DP3/eval_ndf_pointwise_actorseg_hybrid.sh`
  - `policy/DP3/eval_semantic_pointwise_actorseg_hybrid.sh`
- New Hydra configs now exist:
  - `policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/robot_dp3_objpc_actorseg_ndf_pointwise_hybrid.yaml`
  - `policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/robot_dp3_objpc_actorseg_semantic_pointwise_hybrid.yaml`
  - `policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/task/demo_task_objpc_actorseg_ndf_pointwise_hybrid.yaml`
  - `policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/task/demo_task_objpc_actorseg_semantic_pointwise_hybrid.yaml`
- Targeted regression coverage now exists in `policy/DP3/scripts/test_actorseg_pointwise_hybrid.py`, proving:
  - actorseg+NDF hybrid keeps merged raw actorseg `point_cloud` and emits `ndf_point_cloud_A`
  - actorseg+semantic hybrid keeps merged raw actorseg `point_cloud` and emits `semantic_point_cloud_A`
- Fresh verification evidence for the actorseg hybrid work:
  - `python policy/DP3/scripts/test_actorseg_pointwise_hybrid.py` passed with `2` tests
  - `python policy/DP3/scripts/test_ndf_pointwise_hybrid.py` still passed with `3` tests
  - `python policy/DP3/scripts/test_semantic_pointwise_hybrid.py` still passed with `3` tests
  - `bash policy/DP3/scripts/test_ndf_pointwise_arg_utils.sh` passed
  - `python -m py_compile policy/DP3/deploy_policy.py policy/DP3/scripts/process_data_ndf_pointwise_actorseg_hybrid.py policy/DP3/scripts/process_data_semantic_pointwise_actorseg_hybrid.py policy/DP3/scripts/test_actorseg_pointwise_hybrid.py` passed
  - `bash -n` passed for:
    - `policy/DP3/process_data_ndf_pointwise_actorseg_hybrid.sh`
    - `policy/DP3/process_data_semantic_pointwise_actorseg_hybrid.sh`
    - `policy/DP3/train_ndf_pointwise_actorseg_hybrid.sh`
    - `policy/DP3/train_semantic_pointwise_actorseg_hybrid.sh`
    - `policy/DP3/eval_ndf_pointwise_actorseg_hybrid.sh`
    - `policy/DP3/eval_semantic_pointwise_actorseg_hybrid.sh`
- Follow-up bug found during first real user run: `process_data_ndf_pointwise_hybrid.py` originally forwarded `--output_suffix` as two argv items, and the value `-objpc-ndf-pointwise-hybrid` was parsed as a new option because it starts with `-`.
- Root-cause fix: `policy/DP3/scripts/process_data_ndf_pointwise_hybrid.py` now builds argv through `build_hybrid_argv(...)` and passes `--output_suffix=-objpc-ndf-pointwise-hybrid` as an attached option.
- The regression test `policy/DP3/scripts/test_ndf_pointwise_hybrid.py` now includes a dedicated wrapper-argv check for this case and passes with `3` tests.
- Additional NDF-loading finding: the warning
  `unexpected_keys=['decoder.vector_basis.map_to_feat.weight', 'decoder.fc_vec_alpha.weight', 'decoder.fc_vec_alpha.bias']`
  is generally benign for the current DP3 pointwise NDF path when `missing_keys=[]`.
- Reason:
- `policy/DP3/scripts/ndf_feature_utils.py` builds `VNNOccNet(..., return_features=True, return_vector_features=False)` and loads checkpoints with `strict=False`.
- The reported unexpected keys belong to the optional decoder vector-feature head, which is only instantiated when `return_vector_features=True`, see `vnn_occupancy_net_pointnet_dgcnn.py`.
- Current DP3 feature extraction only calls `model.forward_latent(z, pts)` / `model.forward_latent(z, query)` and uses the regular `features` output, not `vector_features`.
- Therefore this exact warning means “the checkpoint contains extra decoder weights that the current loader ignores,” not “the main encoder/decoder weights are missing.”
- Risk rule:
- acceptable: `missing_keys=[]` and only optional vector-feature decoder keys are unexpected
- suspicious: any `missing_keys`, or unexpected keys under encoder / core decoder blocks / backbone family mismatch

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| Start with a generic scaffold instead of guessing a task-specific plan | The user has requested the planning workflow but has not yet provided the concrete implementation task |
| Track initialization state immediately | This creates a clean handoff point for the next task in the session |
| Follow the `train_objpc_sam3.sh -> process_data_objpc_sam3.sh -> sam3_pointcloud_utils.py` chain first | This is the shortest path to determine if the script can run and where fusion timing should be measured |
| Prefer a targeted benchmark before full training | The user asked for per-frame fusion time, and the repo already includes a dedicated benchmark path that isolates this cost |
| Patch repo-side SAM3 compatibility issues before judging the pipeline | The initial failures were caused by fixable code and environment mismatches rather than the core segmentation-project idea itself |
| Prefer fixing the warning at the predictor configuration layer | This removes the spam without hiding potentially useful warnings from other parts of the pipeline |
| Replace all-at-end writes with per-episode replay-buffer appends | The user explicitly wants already-computed episodes persisted and visible during long SAM3 preprocessing runs |
| Treat SAM3 video tracking as a per-sequence tool, not a drop-in endless-stream API | The installed Ultralytics implementation requires `dataset.mode == "video"` and initializes state against a fixed `num_frames` |
| Treat the new request as “replace the 2D segmenter, keep the segment-project-fuse idea” unless the user explicitly asks to switch to oracle `object_pointcloud` | The user explicitly asked to use simulator segmentation instead of SAM, not to bypass segmentation entirely |
| Treat the NDF question as a diagnosis problem, not a one-step benchmark-design explanation | The current code and saved runs already expose multiple confounders beyond “NDF only helps on novel objects” |
| Add `ndf_pointwise_hybrid` as a new named path instead of redefining `ndf_pointwise` | This preserves the meaning of existing checkpoints while isolating the “keep raw `{A}` in the main cloud” hypothesis |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| No prior planning context existed in the repo | Created a fresh planning scaffold |
| `train_objpc_sam3.sh` is cwd-sensitive | Treat this as a likely usability bug and verify it through execution checks |
| `sam3_pointcloud_utils.py` referenced `Path` without importing it | Fixed in repo |
| Local CLIP tokenizer did not match Ultralytics SAM3 expectations | Added an in-repo compatibility shim and installed `ftfy` |
| SAM3 bbox reuse path cached 6-value boxes | Trimmed boxes to 4-value XYXY before reuse |
| Incomplete final `.zarr` directories were reused as if preprocessing had succeeded | Added completeness checks in `train_objpc_sam3.sh` and moved SAM3 preprocessing to resumable per-episode persistence |
| All-at-end writes still left the user exposed to total progress loss before the final publish stage | Reworked SAM3 preprocessing to persist each episode incrementally and resume from partial replay-buffer state |

## Resources
- Project root: `/home/zheng/github/RoboTwin_geo`
- Skill file: `/home/zheng/.agents/skills/planning-with-files/SKILL.md`
- Template file: `/home/zheng/.agents/skills/planning-with-files/templates/task_plan.md`
- Template file: `/home/zheng/.agents/skills/planning-with-files/templates/findings.md`
- Template file: `/home/zheng/.agents/skills/planning-with-files/templates/progress.md`
- Training script: `/home/zheng/github/RoboTwin_geo/policy/DP3/train_objpc_sam3.sh`
- Benchmark script: `/home/zheng/github/RoboTwin_geo/policy/DP3/scripts/benchmark_objpc_sam3.py`
- Fusion utilities: `/home/zheng/github/RoboTwin_geo/policy/DP3/scripts/sam3_pointcloud_utils.py`
- Raw dataset example: `/home/zheng/github/RoboTwin_geo/data/hanging_mug/demo_clean_3d_object_pc`
- Existing DP3 outputs: `/home/zheng/github/RoboTwin_geo/policy/DP3/data`
- SAM3 weights: `/home/zheng/Datasets/sam3/sam3.pt`
- Dataset loader: `/home/zheng/github/RoboTwin_geo/policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/dataset/robot_dataset.py`
- Visualizer: `/home/zheng/github/RoboTwin_geo/policy/DP3/scripts/visualize_objpc_zarr.py`
- Incremental zarr helper: `/home/zheng/github/RoboTwin_geo/policy/DP3/scripts/incremental_objpc_zarr.py`
- Incremental zarr test: `/home/zheng/github/RoboTwin_geo/policy/DP3/scripts/test_incremental_objpc_zarr.py`

## Visual/Browser Findings
- None yet.

## New Findings
- `policy/DP3/ndf_pointwise_arg_utils.sh` had a bash-specific default-value bug: inside quoted parameter expansion, the escaped placeholder default `\{A\},\{B\}` was preserved literally and reached Python as `["\\{A}", "\\{B}"]`.
- That bug only surfaced when users omitted `object_placeholders`, which is why earlier explicit-placeholder invocations worked while the actorseg hybrid default path failed at `require_actor_id_targets(...)`.
- The direct semantic actorseg hybrid shell scripts were not failing for the same reason, but aligning them to the same plain `DEFAULT_OBJECT_PLACEHOLDERS="{A},{B}"` convention removes ambiguity and future-proofs the interface.
- The user explicitly does not want compatibility branches for eval invocation. For the pointwise eval family, the desired contract is the same as `eval_semantic_pointwise_hybrid.sh`: the third positional argument is already the full `ckpt_setting`, and scripts should forward it unchanged instead of deriving suffixes from `task_config`.
- The new shell regression test captures the final argv sent to `script/eval_policy.py`, which is the right verification layer for these wrappers because it checks the observable interface rather than shell internals.
- `objpc` and `actorseg` are not equivalent in the current implementation. `objpc` filters raw per-camera `Position` buffers by actor id first and only then downsamples per object, while `actorseg` starts from the already-downsampled combined scene `observation["pointcloud"]` and then projects that sparse cloud back into segmentation masks.
- Because of that ordering difference, `actorseg` currently loses object detail much earlier than `objpc`, especially for small structures like mug handles and rack hooks. This explains why `objpc` can outperform `actorseg` even when both use simulator truth labels.
- Real ZED raw dataset discovery for `grasp_mug`:
  - raw root: `/media/zheng/Extreme SSD/geo_mani_data/grasp_mug/real_zed_raw`
  - episodes with manifests: 61
  - raw camera labels: `global,left,right`
  - old raw manifest serial mapping differs from calibration labels, so serial-based remapping must stay enabled during postprocess.
- For real-data object masks, SAM3 is the better first implementation target on this machine:
  - SAM3 checkpoint exists: `/home/zheng/Datasets/sam3/sam3.pt`
  - configured SAM2 checkpoint is absent: `/home/zheng/Datasets/sam_gdino/sam_checkpoints/sam2.1_hiera_large.pt`
  - DP3 online eval already has `SAM3ProjectiveTracker`, allowing the same text-refresh plus bbox-reuse behavior to be reused for inference.
- Real objpc HDF5 should be compact by default:
  - `policy/DP3/scripts/object_pointcloud_utils.py::load_hdf5` only reads `/joint_action/vector`, `/pointcloud`, and `/object_pointcloud/*` for `process_data_objpc.py`.
  - Storing full RGB-D observations in every real training HDF5 unnecessarily duplicates raw data and can dominate disk usage, especially for old recordings where RGB is resized to match 1080p depth during postprocess.
  - The new SAM3 objpc batch driver therefore defaults to no observation storage and exposes `--store_observations` for debugging/downstream use.
- Local SAM3 smoke result:
  - On this sandbox, `torch.cuda.is_available()` is false, so SAM3 falls back to CPU.
  - A 1-frame/3-camera/2-object smoke completed and wrote non-empty masks for most camera/object pairs, but full dataset processing should be run in the normal GPU-visible environment with `--sam3_device cuda:0` or `--sam3_device auto`.
- For training, saving processed real data directly on the SSD is feasible as long as DP3 sees a repo-side path:
  - `train_objpc.sh` still expects `../../data/<task>/<task_config>` from `policy/DP3`.
  - The batch postprocess script now writes the actual data to `/media/${USER}/Extreme SSD/geo_mani_data/<task>/robotwin_objpc/<task_config>` by default.
  - It creates `data/<task>/<task_config>` as a symlink to the SSD output unless `--no_link_repo_data` is passed.
- Debug preview mode should be sampled, not full-frame/full-episode by default:
  - mask overlays are PNGs and can be many files across 3 cameras x 2 objects x N frames.
  - merged `{A}/{B}` PLY previews are useful for sanity checking segmentation/projection alignment, and `--debug_stride` plus `--debug_max_frames` keeps them bounded.
- Workspace-projected 2D ROI alone is not sufficient for the current real ZED view:
  - With the saved workspace bbox, the projected RGB ROI spans the full 640x360 frame for all three cameras.
  - Therefore saved SAM masks now also use per-frame depth-based gating: each mask pixel is kept only if its depth point transforms into the workspace bbox.
  - Manual `--mask_roi_xyxy` remains useful as an extra coarse image-space restriction, but the 3D guarantee comes from depth-based workspace filtering.
- The user prefers explicit 2D image-domain workspace masks over depth/3D workspace gating for SAM:
  - `select_camera_workspace_masks.py` now lets the user click a polygon on each camera's first RGB frame.
  - Passing `--camera_workspace_mask_root` makes SAM3 see only the clicked polygon image domain: the image is cropped to the polygon bbox and pixels outside the polygon are zeroed before tracker inference.
  - Saved SAM masks and HDF5 point clouds also use only pixels inside the clicked polygon.
  - Old cached masks from a non-polygon run are treated as incompatible and regenerated in polygon mode.
  - In this mode, depth workspace filtering is disabled by default to avoid reintroducing the previous behavior.
- Real SAM3 mask quality diagnosis:
  - Existing `*_workspace_mask_sam3_smoke` outputs were not generated with polygon input-domain SAM; their metadata has `sam_input_domain=0`, so those previews can look worse than the intended polygon-input pipeline.
  - A fresh 1-frame run with the real per-camera polygon masks confirmed `sam_input_domain=True` for all six camera/object masks.
  - With prompt `{A}:mug`, the left-camera `{A}` mask was empty, while the same frame and polygon domain with `{A}:cup` produced a non-empty left-camera mask. This points to prompt wording / low-resolution text detection as the main failure mode, not SAM's segmentation capacity.
  - The raw RGB frames used by the saved data are `640x360`, so SAM3 is operating on low-resolution images. This can make small/partially occluded objects much more prompt-sensitive than normal high-resolution SAM demos.
  - SAM3 mask cache validity now includes the object prompt text, so changing `{A}:mug` to `{A}:cup` regenerates affected masks instead of silently reusing stale results.
- SAM2 streaming replacement direction:
  - `include/SAM2_streaming` is a symlink to `/home/zheng/github/SAM2_streaming` and provides a real-time `SAM2CameraPredictor`.
  - The core API is `build_sam2_camera_predictor(config, checkpoint)`, then `predictor.load_first_frame(frame)`, `predictor.add_new_prompt(frame_idx=0, obj_id=id, bbox=[[x0,y0],[x1,y1]])`, and later `predictor.track(frame)`.
  - `demo_webcam_box.py` confirms the intended manual initialization flow: user draws a bbox once, then subsequent frames use tracker memory instead of text prompts.
  - Local SAM2 checkpoint discovery found `/home/zheng/Datasets/sam2/sam2.1_hiera_large.pt`; matching config exists at `include/SAM2_streaming/configs/sam2.1/sam2.1_hiera_l.yaml`.
  - The active Python environment can import `sam2.build_sam.build_sam2_camera_predictor` and has Hydra/OmegaConf installed.
  - The streaming predictor code hard-codes CUDA devices in its state initialization, so practical use should be on the GPU runtime. A wrapper should own device setup and fail clearly if CUDA/SAM2 deps are unavailable.
  - New real-data segmentation should be bbox-initialized SAM2 tracking, not SAM3 text detection. This removes the main observed failure mode: prompt sensitivity under low-resolution RGB.
- SAM2 streaming implementation findings:
  - The new active real-data path uses `sam2_tracking_utils.py`, `select_sam2_bboxes.py`, `segment_objects_sam2.py`, and `postprocess/postprocess_real_zed_sam2_objpc_dataset.py`.
  - `postprocess_real_zed_sam2_objpc_dataset.py` imports no SAM3/Ultralytics code; legacy SAM/SAM3 active scripts and DP3 entrypoints have been removed so the maintained real-data segmentation path is SAM2-only.
  - The bbox prompt file is per camera and per placeholder, so `{A}` and `{B}` can be manually initialized independently in `global,left,right`.
  - SAM2 receives the polygon-limited RGB input domain by default when `--camera_workspace_mask_root` is passed, matching the user's requested “only segment inside clicked workspace” behavior.
  - The generated HDF5 boundary remains the same compact objpc format: `/joint_action/vector`, `/pointcloud`, and `/object_pointcloud/{A,B}`, so `policy/DP3/process_data_objpc.sh` and `train_objpc.sh` stay compatible.
- SAM2 online eval implementation findings:
  - `robot_dp3_objpc_sam2` is architecture-compatible with normal `objpc` checkpoints; the difference is only how `point_cloud` is reconstructed online in `deploy_policy.py`.
  - `eval_objpc_sam2.sh` accepts an explicit `ckpt_setting`, so it can evaluate checkpoints trained with `train_objpc.sh` on `demo_real_zed_sam2_objpc` without requiring a separate SAM2 training suffix.
  - The online path tracks all placeholders together per camera, then projects each SAM2 mask back into the current scene point cloud. This avoids advancing the SAM2 streaming state once per placeholder inside a single policy observation.
  - If `--sam2_bbox_prompt_path` is provided, bbox prompts are reused; otherwise the first frame opens interactive OpenCV bbox selection windows. Interactive prompts are cleared on episode reset so stale boxes are not silently reused across episodes.
- SAM2 SDPA runtime error finding:
  - The traceback ending in `RuntimeError: No available kernel` came from `torch.nn.functional.scaled_dot_product_attention` inside the SAM2 mask decoder.
  - The preceding warnings are the diagnostic signal: Q/K/V were `float`, while the available Flash/Memory Efficient attention kernels require `Half` or `BFloat16`; CUDNN SDPA was also disabled.
  - `include/SAM2_streaming/demo_webcam_box.py` opens a global `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` context before using the camera predictor, but the first wrapper implementation did not.
  - The fix is to run `load_first_frame`, `add_new_prompt`, and `track` under CUDA autocast. The wrapper now defaults to `--sam2_autocast_dtype bfloat16`, with `float16` and `none` exposed for hardware-specific fallback.
- SAM2 per-episode bbox finding:
  - SAM2 tracker state must be reset and initialized per recorded episode, not per training epoch.
  - The tracker was already recreated inside `segment_episode_sam2(...)`, but the batch driver initially loaded one global `sam2_bbox_prompts.json` and reused it for every raw episode.
  - That global prompt reuse is only safe if every episode starts with the same image-space object pose. For randomized real data, each raw episode needs its own first-frame `{A}/{B}` bboxes.
  - The batch driver now prefers `sam2_bbox_prompts/<raw_episode_name>/sam2_bbox_prompts.json`, then `episode<index>/sam2_bbox_prompts.json`, then `episode_<index:06d>/sam2_bbox_prompts.json`, with global `sam2_bbox_prompts.json` only as fallback unless `--require_per_episode_bboxes` is set.
- Real-ZED direct-run import finding:
  - Running `python script/real_zed_collection/select_sam2_bboxes.py` can omit the repository root from `sys.path`, so absolute imports like `from script.real_zed_collection...` fail with `ModuleNotFoundError: No module named 'script'`.
  - Direct-run real-ZED entry scripts should insert `Path(__file__).resolve().parents[2]` or `parents[3]` for postprocess scripts before importing `script.*`.
- Real-ZED right-base task-config finding:
  - The current `grasp_mug_new` processed HDF5 data is linked at `data/grasp_mug_new/demo_real_zed_sam2_objpc`, and its meta records `output_frame=right_base`.
  - Training with `task_config=demo_real_zed_sam2_objectpc_rightbase` fails because no repo data link exists at `data/grasp_mug_new/demo_real_zed_sam2_objectpc_rightbase`.
  - A failed baseline preprocessing run can leave an empty zarr directory under `policy/DP3/data`, so reusing that exact wrong config can skip preprocessing unless the zarr is rebuilt.
- Real-ZED inference frame-interface finding:
  - Real inference scripts now expose `output_frame` and robot-camera calibration as positional optional args rather than hiding them in passthrough flags.
  - `auto` resolution first reads `data/<task>/<task_config>/real_zed_sam2_objpc_meta.json`; if that file is unavailable, it infers `right_base`, `left_base`, or `workspace` from the `task_config` name.
  - For `right_base` or `left_base`, `auto` calibration resolves to the default `robot_camera_apriltag_<arm>_global.yaml` under `script/real_zed_collection/calibration`.
- Real-ZED action step finding:
  - The `Action delta safety stop` is triggered by a large joint-space action jump between consecutive commands, not by the camera pipeline.
  - Before the fix, only the first action was interpolated; later DP3 actions were sent directly via `env.step(action)`.
  - Real inference now clips the actually executed arm-joint delta and gripper delta before safety checking, keeping `--max_action_delta` as a hard stop rather than the mechanism for smoothing.

---
*Update this file after every 2 view/browser/search operations.*
