# Task Plan: Real Three-ZED Data Collection For DP3

## Goal

Design a real-robot data collection pipeline that keeps the robot-control behavior consistent with `include/xtrainer_clover/experiments/run_control.py`, records synchronized data from three ZED cameras, and postprocesses each demonstration once into simulator-compatible RoboTwin/DP3 training data that can feed objpc, NDF, semantic, Utonia, and future feature branches.

## Current Phase

SAM2 streaming tracking migration for real objpc training/eval data

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

## Notes

- Re-read this file before implementation.
- Log discoveries in `findings.md`.
- Log commands and verification in `progress.md`.
