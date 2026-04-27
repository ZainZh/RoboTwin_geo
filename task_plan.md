# Task Plan: Real Three-ZED Data Collection For DP3

## Goal

Design a real-robot data collection pipeline that keeps the robot-control behavior consistent with `include/xtrainer_clover/experiments/run_control.py`, records synchronized data from three ZED cameras, and postprocesses each demonstration once into simulator-compatible RoboTwin/DP3 training data that can feed objpc, NDF, semantic, Utonia, and future feature branches.

## Current Phase

Workspace anchor and live crop support

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

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| Keep robot teleoperation/control logic aligned with `run_control.py` | Avoid changing safety, button, servo, and action semantics while adding perception recording |
| Do not compute model features during collection | NDF, semantic, and Utonia variants should all reuse the same raw/canonical data |
| Use canonical HDF5 as the compatibility boundary | Existing DP3 preprocess scripts already consume HDF5 episode files and then emit zarr |
| Use reference-camera frame in v1 | Current policy is joint-space; robot-base hand-eye calibration is useful but not required for a fixed camera setting |
| Add explicit workspace anchor instead of implicit last calibration pose | Keeps crop coordinates physically meaningful and repeatable across sessions |
| Save cropped RGB/depth plus crop metadata rather than only fused point clouds | Reduces storage while preserving postprocess flexibility for masks and feature extraction |

## Key Questions

1. What ZED SDK wrapper is available in the target robot environment?
2. How should object masks be supplied in real data: manual annotation, live SAM/grounding, saved masks, or a first-frame interactive workflow?
3. Which world frame should be canonical: robot base frame or a calibrated table/world frame?

## Errors Encountered

| Error | Attempt | Resolution |
|-------|---------|------------|

## Notes

- Re-read this file before implementation.
- Log discoveries in `findings.md`.
- Log commands and verification in `progress.md`.
