# Task Plan: Real Three-ZED Data Collection For DP3

## Goal

Design a real-robot data collection pipeline that keeps the robot-control behavior consistent with `include/xtrainer_clover/experiments/run_control.py`, records synchronized data from three ZED cameras, and postprocesses each demonstration once into simulator-compatible RoboTwin/DP3 training data that can feed objpc, NDF, semantic, Utonia, and future feature branches.

## Current Phase

Processed-dataset semantic field visualization

## Phases

### Phase 1: Context Discovery
- [x] Read the existing real-robot control and save loop.
- [x] Read existing xtrainer frame serialization.
- [x] Read DP3 HDF5/zarr preprocessing expectations.
- **Status:** complete

### Phase 2: Pipeline Design
- [x] Separate raw recording from canonical postprocessing.
- [x] Define the canonical HDF5 fields needed by current DP3 preprocessors.
- [x] Identify where real data must differ from simulator actor/object point clouds.
- **Status:** in_progress

### Phase 3: User Review
- [x] Present architecture and trade-offs.
- [x] Get approval before writing collection/postprocess scripts.
- **Status:** complete

### Phase 4: Implementation
- [x] Add the real ZED collection script.
- [x] Add the raw-to-RoboTwin-HDF5 postprocess script.
- [x] Add SAM-mask generation script.
- [x] Add validation utilities for point cloud alignment and field compatibility.
- **Status:** complete

### Phase 5: Verification
- [x] Add and run unit coverage for raw-to-HDF5 conversion.
- [x] Run Python syntax checks on new scripts.
- [ ] Run on real ZED/robot hardware.
- **Status:** complete with hardware run pending

### Phase 6: Workspace Anchor And Cropped Capture
- [x] Add a standalone workspace-anchor script that uses a placed Charuco board to define a stable workspace/world frame.
- [x] Add calibration-health checks comparing per-camera board poses after transforming into the current reference frame.
- [x] Add reusable workspace crop utilities for 3D bbox projection, 2D ROI clipping, image/depth crop, and crop-adjusted intrinsics.
- [x] Update real ZED raw collection so future recordings can save cropped RGB/depth with workspace metadata while retaining optional full-frame debug snapshots.
- [x] Verify geometry utilities and collection script syntax without requiring real hardware.
- **Status:** complete with hardware run pending

### Phase 7: Real SAM3 ObjPC Dataset Postprocessing
- [x] Add an offline SAM3 mask generator for `{A}=mug` and `{B}=box` that can reuse bbox prompts between frames.
- [x] Add a batch driver that converts raw real-ZED episodes into `data/<task>/<task_config>/data/episode*.hdf5`.
- [x] Keep the output compatible with `policy/DP3/train_objpc.sh` without changing existing DP3 training scripts.
- [x] Verify helper behavior, syntax, and a small pilot conversion path.
- [ ] Run full SAM3 postprocessing over all real episodes on a GPU-visible runtime.
- **Status:** complete with full dataset run pending

### Phase 8: SSD Output And Debug Previews
- [x] Default real SAM3 objpc postprocess output to the external SSD path.
- [x] Add repo-side symlink creation so `policy/DP3/train_objpc.sh` can still load data from `data/<task>/<config>`.
- [x] Add debug mask overlays on original RGB frames.
- [x] Add merged `{A}/{B}` colored `.ply` object point cloud previews.
- [x] Verify debug output on a 1-frame SAM3 smoke.
- **Status:** complete

### Phase 9: Workspace-Constrained SAM Masks
- [x] Add static RGB ROI masking for SAM masks, using workspace bbox projection by default.
- [x] Add manual per-camera RGB ROI override through `--mask_roi_xyxy`.
- [x] Add per-frame depth-based workspace gating so saved masks only keep pixels whose depth point lies inside the workspace bbox.
- [x] Verify mask metadata and debug point cloud output on a 1-frame SAM3 smoke.
- **Status:** complete

### Phase 10: Interactive 2D Camera Workspace Masks
- [x] Add an interactive first-frame polygon selector for each camera.
- [x] Save per-camera 2D workspace masks and overlays.
- [x] Restrict SAM input images to the clicked per-camera polygon masks before inference.
- [x] Keep saved SAM masks restricted to the clicked per-camera polygon masks.
- [x] Restrict HDF5 point-cloud reconstruction to the clicked per-camera polygon masks.
- [x] Invalidate old non-polygon SAM mask caches when polygon mode is enabled.
- [x] Make polygon mask mode disable depth workspace gating by default, with an opt-in `--also_depth_workspace_filter`.
- **Status:** complete

### Phase 11: SAM2 Streaming Tracking Migration
- [x] Read `include/SAM2_streaming` and identify the bbox-initialized real-time tracking API.
- [x] Locate local SAM2 checkpoint/config availability.
- [x] Add a repo-local SAM2 tracking adapter that does not depend on SAM3/Ultralytics.
- [x] Add an interactive per-camera/per-placeholder bbox initializer for real recordings.
- [x] Add a SAM2-tracking real postprocess path that writes the same RoboTwin HDF5 fields.
- [x] Add real eval integration so the first frame initializes A/B boxes on each camera and later frames call `track`.
- [x] Move active real-data docs/scripts from SAM3 naming to SAM2 naming.
- **Status:** complete with GPU/hardware run pending

### Phase 12: Legacy SAM/SAM3 Cleanup
- [x] Remove old SAM/SAM3 real-data segmentation scripts and postprocess entrypoints.
- [x] Remove DP3 `objpc_sam3` preprocessing, training, eval, and config files.
- [x] Remove SAM3 branches from DP3 deployment so only `objpc_sam2` remains for online mask tracking.
- [x] Update tests/docs so active code no longer imports SAM3/Ultralytics or the old generic SAM path.
- [x] Verify active source search, Python syntax, SAM2 tests, DP3 hybrid tests, shell syntax, and Hydra SAM2 config composition.
- **Status:** complete

### Phase 13: Real-ZED DP Image Baseline
- [x] Add a single-camera DP zarr conversion path from real-ZED raw RGB plus compact SAM2 objpc HDF5 joint vectors.
- [x] Add an independent three-camera DP zarr conversion path using `global,left,right -> head_camera,left_camera,right_camera`.
- [x] Add a multi-camera DP dataset/config branch so `head_cam/left_cam/right_cam` all enter `MultiImageObsEncoder`.
- [x] Keep the existing single-camera DP scripts/configs unchanged.
- [x] Verify single-camera conversion, three-camera conversion, shell syntax, Python syntax, and unit coverage.
- **Status:** complete

### Phase 14: Real DP3 Inference Scripts
- [x] Read xtrainer real inference/control path and DP3 deploy interfaces.
- [x] Add a real-ZED DP3 inference driver that fuses three ZED point clouds and commands the xtrainer robot env.
- [x] Add baseline and semantic-pointwise-hybrid shell wrappers for the user's trained checkpoints.
- [x] Keep SAM2 object tracking only where semantic hybrid needs `{A}/{B}` object point clouds.
- [x] Verify syntax and wrapper argument wiring without requiring hardware.
- [ ] Run on real ZED/robot hardware.
- **Status:** complete with hardware run pending

### Phase 15: Robot-Base To Camera AprilTag Calibration
- [x] Read xtrainer/RoboTwin teleoperation semantics and identify the usable robot Cartesian pose source.
- [x] Design target-on-gripper AprilTag calibration for separate left/right Dobot bases.
- [x] Add a standalone interactive ZED + robot calibration script under `script/real_zed_collection`.
- [x] Add synthetic unit coverage for the OpenCV robot-world-hand-eye transform convention.
- [x] Verify unit test, syntax, and CLI wiring without requiring hardware.
- [ ] Run the calibration on real robot/ZED hardware.
- **Status:** complete with hardware run pending

### Phase 16: Real SAM2 Online Latency Reduction
- [x] Diagnose why semantic real inference is much slower than the standalone ZED SAM2 tracker.
- [x] Replace fused-scene back-projection with direct `mask + depth` object point-cloud lifting.
- [x] Enforce workspace 3D filtering before points enter scene/object observations for all output frames.
- [x] Add SAM2 input downscale control so tracking does not require 1080p frames.
- [x] Add a standalone real-time SAM2 object point-cloud preview script for hardware-side unit testing.
- [x] Add single-object, headless FPS, and larger Open3D preview controls.
- [x] Add object-only preview mode that skips dense full-scene point-cloud construction by default.
- [x] Add shared per-camera worker parallelism for preview object reconstruction, real scene construction, and online SAM2 object extraction.
- [x] Verify with unit tests, syntax checks, and CLI visibility.
- [ ] Run on real ZED/robot hardware and inspect `--profile_timing`.
- **Status:** complete with hardware timing pending

### Phase 17: Real Robot Execution Smoothing
- [x] Diagnose why real inference can feel abrupt even after action delta limiting.
- [x] Add execution-side action substeps so each policy action is sent as several smaller robot commands.
- [x] Expose CLI controls for substep count and inter-substep sleep.
- [x] Move the default smoothing path down to Dobot `ServoJ(t, gain)` through the xtrainer ZMQ robot server.
- [x] Add execution-side delta-change limiting to reduce acceleration-like jumps between consecutive commands.
- [x] Add action diagnostics for policy delta, commanded delta, commanded delta change, and observed execution delta.
- [x] Add optional async action-buffer control thread so robot ServoJ commands continue while SAM2/DP3 inference is running.
- [x] Verify with unit tests, syntax checks, shell checks, and CLI help visibility.
- [ ] Tune the smoothing parameters on real hardware.
- **Status:** complete with hardware tuning pending

### Phase 18: Real-ZED Postprocess Calibration Selection
- [x] Diagnose why offline SAM2 objpc postprocess can produce visibly misaligned fused point clouds.
- [x] Compare the batch driver calibration defaults against per-episode raw-data manifests.
- [x] Update the batch driver to prefer collection-time calibration snapshots by default.
- [x] Preserve explicit calibration override behavior for intentional reprocessing.
- [x] Verify with targeted unit tests, full real-ZED collection pipeline tests, and syntax checks.
- **Status:** complete

### Phase 19: Real-ZED DP Image Inference
- [x] Add a real-ZED DP image-policy inference driver that avoids point-cloud/SAM work.
- [x] Detect required RGB streams from the loaded checkpoint config instead of assuming three cameras.
- [x] Reuse the proven real robot safety, ServoJ smoothing, and async action-buffer control pattern.
- [x] Add shell wrapper(s) for single-camera and multi-camera DP checkpoints.
- [x] Verify unit behavior, syntax, wrapper wiring, and CLI visibility without requiring hardware.
- [ ] Run on real ZED/robot hardware.
- **Status:** complete with hardware run pending

### Phase 20: Optional-Camera Real-ZED Collection
- [x] Remove the raw collection script's hard requirement that exactly three ZED cameras are active.
- [x] Keep the default three-camera config unchanged.
- [x] Resolve camera serials from calibration when the requested camera label subset differs from config `zed_serials`.
- [x] Add unit coverage for one-camera, two-camera, and invalid serial-count cases.
- [x] Verify collection pipeline tests and syntax without requiring hardware.
- [ ] Run on real ZED/robot hardware.
- **Status:** complete with hardware run pending

### Phase 21: DP Real-ZED Mixed-Resolution Training Fix
- [x] Diagnose DP zarr preprocessing failure on mixed old/new real-ZED episodes.
- [x] Make DP real-ZED training wrappers pass a fixed resize target from `head_camera_type`.
- [x] Stop training immediately if real-ZED preprocessing fails.
- [x] Add shell coverage for single-camera and multicam wrapper preprocessing args.
- [x] Verify shell syntax and RoboTwin-environment unit tests.
- **Status:** complete

### Phase 22: Real DP3 In-Process Episode Reset
- [x] Add a non-blocking keyboard command path for real DP3 inference.
- [x] Make `r` stop any async controller, return the robot to the initial photo pose, reset policy/controller state, and restart the episode budget.
- [x] Make `q` stop inference without restarting the Python process.
- [x] Keep SAM2 prompts/tracking by default, with an opt-in flag to clear SAM2 state on reset.
- [x] Verify behavior with hardware-free unit tests, syntax checks, and wrapper checks.
- **Status:** complete with hardware run pending

### Phase 23: Optional-Camera Real-ZED Extrinsic Calibration
- [x] Remove the Charuco extrinsic calibration script's hard requirement that exactly three ZED cameras are active.
- [x] Resolve camera labels/serials from `real_zed_collection.yaml`, `--labels/--serials`, or connected serial auto-discovery for any 2+ unique cameras.
- [x] Keep the default three-camera behavior unchanged when the config still lists `global,left,right`.
- [x] Record `camera_count` in the saved calibration YAML.
- [x] Add unit coverage for two-camera config acceptance and one-camera rejection.
- **Status:** complete with hardware run pending

### Phase 24: Real-ZED Exposure Lock And Quality Watchdog
- [x] Add fixed ZED exposure, gain, and white-balance controls to the raw collection config and capture loop.
- [x] Record the requested image-control settings in each raw episode manifest.
- [x] Compute per-camera image quality metrics for saved RGB/depth frames.
- [x] Warn on repeated overexposure, underexposure, low RGB variance, or low valid-depth ratio.
- [x] Save compact image-quality values in camera `.npz` files and per-frame camera-quality metadata in `manifest.json`.
- [x] Verify the new behavior with hardware-free unit tests and syntax checks.
- **Status:** complete with hardware threshold tuning pending

### Phase 25: Real-ZED Initial Frame Preview
- [x] Add a first-frame preview gate after ZED camera threads initialize and before robot initialization.
- [x] Render all active camera labels side-by-side in one OpenCV window.
- [x] Continue only after Enter, `c`, or Space; abort before robot initialization on `q` or Esc.
- [x] Skip the GUI preview automatically when no display server is available.
- [x] Add config flags to enable/disable preview and tune the wait timeout/window name.
- [x] Verify preview rendering and collection pipeline tests without hardware.
- **Status:** complete with hardware UI check pending

### Phase 26: Real-ZED DP EEF Image Policy
- [x] Add DP zarr preprocessing support for EEF14 state and 20D absolute-6D EEF action.
- [x] Add DP Hydra config/task variants for 20D actions.
- [x] Add single-camera and multicam EEF training wrappers with `*-eef-absolute6d-global` naming.
- [x] Add real DP EEF inference support that converts policy EEF actions through Dobot IK before ServoJ execution.
- [x] Add shell/interface tests and syntax checks without requiring hardware.
- **Status:** complete with hardware run pending

### Phase 38: Real DP3 Snapshot And SAM2 Reselect Hotkeys
- [x] Add a keyboard `s` command that saves current live camera frames and fused colored point cloud snapshot.
- [x] Add a keyboard command that resets the robot and forces SAM2 bbox re-selection for swapped objects.
- [x] Wire the new commands into both async and sync DP3 real inference loops.
- [x] Keep baseline/semantic EEF wrappers compatible without changing their positional arguments.
- [x] Verify behavior with unit tests, shell syntax, Python syntax, and wrapper checks.
- **Status:** complete with hardware run pending

### Phase 26: DP3 Main Point-Cloud Count Training Interface
- [x] Add failing tests that lock the desired shell-wrapper behavior.
- [x] Add `point_cloud_num` to baseline, objpc, semantic, NDF, Utonia, actorseg, and EEF training wrappers.
- [x] Forward the count into preprocessing `--target_num_points` and Hydra `point_cloud` shape overrides.
- [x] Use `-pcN` suffixes for non-default main point-cloud counts to avoid zarr/checkpoint collisions.
- [x] Verify tests plus shell syntax for touched wrappers.
- **Status:** complete

### Phase 27: DP3 Eval Main Point-Cloud Count Interface
- [x] Add failing tests for eval wrappers and deploy-time shape override.
- [x] Add `point_cloud_num` to all DP3 eval shell wrappers.
- [x] Append `-pcN` to checkpoint settings for non-default main point-cloud counts.
- [x] Pass `--point_cloud_num` through `eval_policy.py` overrides into `deploy_policy.py`.
- [x] Override `cfg.task.shape_meta.obs.point_cloud.shape` before checkpoint loading.
- [x] Verify eval wrapper tests, deploy compile, shell syntax, and diff whitespace.
- **Status:** complete

### Phase 26: Right-Base EEF Reference Frame
- [x] Extend EEF frame handling from `workspace/reference_camera` to `left_base/right_base`.
- [x] Make EEF preprocessing and real inference default to `right_base`.
- [x] Add `rightbase` to EEF zarr/checkpoint suffixes to avoid mixing workspace-frame artifacts.
- [x] Validate real-ZED dataset `output_frame` against `--eef_frame_mode` during EEF preprocessing.
- [x] Verify EEF transform tests, real inference parser/action tests, wrapper tests, shell syntax, Python syntax, and whitespace checks.
- **Status:** complete

### Phase 27: Global-Camera EEF Wrapper Set
- [x] Add independent global/reference-camera EEF preprocess wrappers.
- [x] Add independent global/reference-camera EEF train wrappers for scene, objpc, and semantic hybrid.
- [x] Add independent global/reference-camera real inference wrappers for baseline and semantic hybrid.
- [x] Use `*-eef-absolute6d-global` suffixes to avoid mixing global and right-base EEF artifacts.
- [x] Verify wrapper tests, shell syntax, Python syntax, and whitespace checks.
- **Status:** complete

### Phase 26: Real DP3 EEF Absolute-6D Training And Inference
- [x] Validate local Dobot FK against controller `PositiveSolution`/`GetPose` on hardware.
- [x] Add shared EEF pose/action utilities for joint14 -> EEF14 and EEF14 -> action20.
- [x] Add optional EEF absolute-6D action conversion to scene, objpc, and semantic-pointwise DP3 preprocessing.
- [x] Add independent EEF zarr suffixes, task configs, train scripts, and real inference wrappers.
- [x] Add xtrainer_clover FK/IK/ZMQ/RobotEnv interfaces so real inference can decode EEF actions through Dobot IK and still execute with ServoJ smoothing.
- [x] Verify EEF utility tests, real inference action tests, shell syntax, and Python syntax checks without hardware.
- [ ] Run EEF inference on real robot hardware.
- **Status:** complete with hardware run pending

### Phase 26: Real-ZED SAM2 Postprocess Debug Video And Keyframes
- [x] Generate a per-demo MP4 that overlays SAM2 tracked masks on the active camera views for every processed frame.
- [x] Switch default debug point-cloud frame selection to demo start, 20%, 40%, 60%, 80%, and final frame.
- [x] Keep the existing merged `{A}/{B}` debug PLY and add per-object fused PLY files for each keyframe.
- [x] Preserve the old stride/max debug-frame mode behind an explicit `--debug_frame_mode stride`.
- [x] Verify with targeted RED/GREEN tests, full collection pipeline tests, syntax checks, and whitespace checks.
- **Status:** complete with hardware/postprocess visual inspection pending

### Phase 27: Real-ZED Dense Per-Object Point Clouds
- [x] Change real-ZED SAM2 objpc batch postprocess default `object_point_num` from 1024 to 5000.
- [x] Change the lower-level raw-to-RoboTwin-HDF5 postprocess default to the same 5000 points per object.
- [x] Keep explicit `--object_point_num` override available for smaller/faster debug runs.
- [x] Add regression coverage for the new dense default.
- [x] Verify collection pipeline tests, syntax checks, and whitespace checks.
- **Status:** complete

### Phase 28: Semantic Hybrid Training Resource Controls
- [x] Add explicit DataLoader worker and pin-memory parameters to `train_semantic_pointwise_hybrid.sh`.
- [x] Add a `training.max_val_steps` script parameter to cap epoch-boundary validation memory/time.
- [x] Keep training throughput-oriented defaults while making validation more memory-safe.
- [x] Add regression coverage for the new script interface.
- [x] Verify script syntax and whitespace.
- **Status:** complete

### Phase 29: DP3 Training Resource Controls Across Scripts
- [x] Propagate DataLoader worker, pin-memory, and validation-step controls to baseline, RGB, objpc, NDF, semantic, Utonia, actorseg, feat, and interaction train wrappers.
- [x] Extend the shared NDF pointwise argument parser so NDF pointwise variants inherit the same resource defaults.
- [x] Keep defaults consistent across scripts: train workers 4, val workers 2, train pin-memory true, val pin-memory false, max val steps 2.
- [x] Add regression coverage that all target train wrappers forward the resource overrides.
- [x] Verify shell syntax, targeted tests, missing-override grep checks, and whitespace.
- **Status:** complete

### Phase 30: Semantic Checkpoint Branch-Mismatch Guardrails
- [x] Diagnose why a semantic-named teapot checkpoint failed against a semantic-hybrid inference graph.
- [x] Confirm the checkpoint was trained without `semantic_point_cloud_A` even though the zarr/meta contains that key.
- [x] Update the semantic hybrid train wrapper to use meta `feature_placeholders` when reusing an existing semantic zarr.
- [x] Add real-inference validation for extra semantic ckpts not present in the checkpoint state_dict.
- [x] Align the real semantic wrapper default point count with the current 1024-point training default.
- [x] Verify targeted tests, real inference action tests, shell syntax, Python syntax, and whitespace.
- **Status:** complete

### Phase 31: Real Inference ZED Image Controls
- [x] Locate where real DP3 inference configures ZED exposure and white balance.
- [x] Add real inference CLI controls for auto/fixed exposure and white balance.
- [x] Default real inference to automatic exposure/gain and automatic white balance.
- [x] Keep fixed exposure/gain/white-balance override flags available.
- [x] Verify parser tests, real inference tests, wrapper interface tests, syntax, and whitespace.
- **Status:** complete

### Phase 32: EEF Training Zarr Path Forwarding
- [x] Diagnose `PathNotFoundError("nothing found at path ''")` in `train_objpc_eef_absolute6d_global.sh`.
- [x] Confirm the generic `scripts/train_policy.sh` did not forward `task.dataset.zarr_path` explicitly.
- [x] Add an optional zarr-path parameter to shared DP3 train helpers.
- [x] Update EEF baseline/objpc train wrappers to pass their generated zarr path explicitly.
- [x] Add regression coverage for the path forwarding contract.
- [x] Verify targeted tests, shell syntax, mock Hydra arguments, and whitespace.
- **Status:** complete

### Phase 33: Raw RGB Episode Export Utility
- [x] Add a standalone script to export saved raw real-ZED RGB images for one task/episode.
- [x] Support episode index, timestamp/name, or full episode directory path.
- [x] Support camera filtering, frame stride, max frame limit, and `rgb`/`full_rgb_debug` key selection.
- [x] Save per-camera PNG files plus an export summary JSON.
- [x] Add regression coverage using a synthetic raw episode.
- [x] Verify targeted test, Python compilation, absolute-path help invocation, and whitespace.
- **Status:** complete

### Phase 34: Raw Colored Point Cloud Episode Export Utility
- [x] Add a standalone script matching the raw RGB export task/episode interface.
- [x] Convert raw `rgb + depth_m + camera_matrix` into colored point clouds.
- [x] Use per-episode calibration snapshots for fused reference/workspace-frame output.
- [x] Support fused, per-camera, or both export modes.
- [x] Support camera filtering, frame stride, max frame limit, depth/RGB key selection, point cap, and xyz crop bounds.
- [x] Add regression coverage using a synthetic raw episode and calibration snapshot.
- [x] Verify targeted tests, Python compilation, absolute-path help invocation, and whitespace.
- **Status:** complete

### Phase 35: Real DP3 EEF IK Failure Diagnostics
- [x] Trace the EEF inference failure to `policy_actions_to_joint_actions -> env.get_ik`.
- [x] Confirm xtrainer ZMQ server previously did not return exceptions to the client.
- [x] Add client-side surfacing for remote robot errors from `get_ik`.
- [x] Add DP3-side EEF IK failure context including `eef_world` and `eef_base`.
- [x] Verify xtrainer robot-node tests, real inference action tests, syntax checks, and whitespace.
- **Status:** complete

### Phase 36: Real DP3 EEF Async Robot-I/O Serialization
- [x] Diagnose `ZMQError: Operation cannot be accomplished in current state` during EEF async inference.
- [x] Identify the root cause as concurrent use of one xtrainer ZMQ REQ socket by async `step/get_obs` and main-thread EEF IK conversion.
- [x] Add a shared robot I/O lock to serialize IK and `env.step` calls without changing the policy/action representation.
- [x] Add regression coverage requiring EEF IK and command execution to use the shared lock.
- [x] Verify the real inference action tests, Python compilation, and whitespace check.
- **Status:** complete

### Phase 37: Real EEF Pose Collection For EEF Training
- [x] Trace the current collection/postprocess/EEF preprocessing path for robot pose data.
- [x] Add raw collection support for saving measured Dobot TCP pose from `env.get_obs()["ee_pos_quat"]` as `eef_pose_base`.
- [x] Persist measured EEF poses into processed HDF5 under `/eef_action/base_pose6`.
- [x] Make DP3 EEF preprocessing prefer `/eef_action/base_pose6` over offline FK when present, while preserving FK fallback for old data.
- [x] Add regression tests for raw snapshot saving, HDF5 persistence, and real-pose EEF state/action conversion.
- [x] Verify targeted tests, EEF wrapper tests, Python compilation, and whitespace.
- **Status:** complete

### Phase 38: Processed-Dataset Semantic Field Visualization
- [x] Use processed HDF5 `/pointcloud` as the scene cloud source.
- [x] Use processed HDF5 `/object_pointcloud/{placeholder}` as the segmented object point-cloud source.
- [x] Compute semantic pointwise embeddings through the same DP3 semantic feature utility used by training.
- [x] Fit shared PCA by semantic checkpoint by default so same-class objects share colors.
- [x] Prefer raw full-scene RGB/depth reconstruction for scene overlay when real-ZED metadata is available, without workspace cropping.
- [x] Default to removing matching full-scene object points and inserting semantic-colored object points, avoiding nearest-neighbor recoloring of nearby table/background points.
- [x] Add a z-min deletion gate for `cut_replace` so nearby tabletop points can be preserved while semantic-colored object points are inserted.
- [x] Add predicted-label coloring through semantic logits, with PCA retained as an optional continuous-embedding visualization mode.
- [x] Align PCA coloring with `visualize_utonia_universal_field.py` `surface_sem_embedding_joint_pca.ply` by using the same `torch.pca_lowrank` projection.
- [x] Align visualization semantic forward mode with `visualize_utonia_universal_field.py` direct point-cloud mode by using fallback normals and random query sampling by default.
- [x] Align visualization semantic input colors with the debug PLY placeholder colors used by `visualize_utonia_universal_field.py`.
- [x] Export per-object semantic PCA PLY files, scene overlay PLY files, and a JSON summary.
- [x] Add unit and synthetic-HDF5 regression coverage.
- **Status:** complete

### Phase 39: Real Semantic Feature-Input Consistency
- [x] Audit offline semantic preprocessing feature inputs.
- [x] Audit sim/real semantic deployment feature inputs.
- [x] Compare current training/inference behavior against the validated debug-placeholder visualization path.
- [x] Add an explicit semantic feature-input mode to preprocessing, eval, and real inference wrappers.
- [x] Put corrected semantic features under a non-conflicting `-semdebugref` suffix.
- [ ] Regenerate semantic zarrs and retrain checkpoints with the corrected distribution.
- **Status:** implementation complete; retraining pending

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| Keep robot teleoperation/control logic aligned with `run_control.py` | Avoid changing safety, button, servo, and action semantics while adding perception recording |
| Do not compute model features during collection | NDF, semantic, and Utonia variants should all reuse the same raw/canonical data |
| Use canonical HDF5 as the compatibility boundary | Existing DP3 preprocess scripts already consume HDF5 episode files and then emit zarr |
| Use reference-camera frame in v1 | Current policy is joint-space; robot-base hand-eye calibration is useful but not required for a fixed camera setting |
| Add explicit workspace anchor instead of implicit last calibration pose | Keeps crop coordinates physically meaningful and repeatable across sessions |
| Save cropped RGB/depth plus crop metadata rather than only fused point clouds | Reduces storage while preserving postprocess flexibility for masks and feature extraction |
| Use SAM3 for the first real-data postprocess path | The local SAM3 checkpoint exists and the DP3 online eval path already uses the same SAM3 tracker; the configured SAM2 checkpoint is absent |
| Migrate active real tracking to SAM2 streaming bbox initialization | SAM2 streaming removes the observed SAM3 text-prompt instability and matches the planned real eval workflow |
| Remove legacy SAM/SAM3 code paths after SAM2 migration | Avoid dependency/import pollution and keep the real-data segmentation stack centered on one maintained tracker |
| Use a separate DP multicam branch instead of modifying the old DP dataset | Keeps the single-camera baseline reproducible while allowing three RGB streams to train through the existing `MultiImageObsEncoder` |
| Calibrate robot base to camera with a held AprilTag cube | Makes the point-cloud frame recoverable after robot-base movement and supports future robot-base-frame training/inference |

## Key Questions

1. What ZED SDK wrapper is available in the target robot environment?
2. How should object masks be supplied in real data: manual annotation, live SAM/grounding, saved masks, or a first-frame interactive workflow?
3. Which world frame should be canonical: robot base frame or a calibrated table/world frame?

## Errors Encountered

| Error | Attempt | Resolution |
|-------|---------|------------|
| SAM2 postprocess failed in `scaled_dot_product_attention` with `RuntimeError: No available kernel` | User ran `postprocess_real_zed_sam2_objpc_dataset.py` on GPU; warnings showed Q/K/V were float32 and Flash/Memory Efficient attention kernels were unavailable for that dtype | Wrapped SAM2 predictor calls in CUDA autocast, defaulting to `bfloat16`, and enabled CUDA SDP kernels in the SAM2 loader |
| Real DP3 wrapper opened `/home/zheng/script/...` from repo-root invocation | Initial wrappers used `cd ../..`, which only works when launched from `policy/DP3` | Switched both wrappers to resolve the repository root from `${BASH_SOURCE[0]}` |
| EEF objpc training failed with `PathNotFoundError("nothing found at path ''")` | `train_objpc_eef_absolute6d_global.sh` used the shared train helper without an explicit `task.dataset.zarr_path`, and Hydra resolved the dataset path to an empty string | Added optional zarr-path forwarding in shared train helpers and passed `../../../data/<task>-<config>-<num><suffix>.zarr` from EEF train wrappers |
| EEF async inference failed with `ZMQError: Operation cannot be accomplished in current state` | Main thread called Dobot IK while the async control thread was sending `step/get_obs` through the same xtrainer ZMQ REQ socket | Added a shared robot I/O lock around EEF IK and robot command execution |
| Full `script.test_real_zed_collection_pipeline` still fails in the current local env | Two DP image preprocess tests import `zarr`, which is not installed in this shell environment | Ran the targeted real-EFF collection/postprocess tests plus EEF utility/wrapper tests instead |

## Notes

- Re-read this file before implementation.
- Log discoveries in `findings.md`.
- Log commands and verification in `progress.md`.
