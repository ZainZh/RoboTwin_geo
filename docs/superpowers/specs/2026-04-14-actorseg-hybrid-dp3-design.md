# ActorSeg Hybrid DP3 Design

## Summary

Add two new DP3 pipelines that use simulator actor-segmentation mask projection as the raw object-pointcloud source and
still keep the successful hybrid observation structure:

- `objpc_actorseg_ndf_pointwise_hybrid`
- `objpc_actorseg_semantic_pointwise_hybrid`

Each pipeline keeps the main merged actorseg object point cloud in `obs.point_cloud` and adds a second feature branch
for `{A}` / `{B}` as needed.

## Problem

The current repo already has:

- `objpc_actorseg`: raw mask-project-fuse object point clouds from simulator actor segmentation
- `ndf_pointwise_hybrid`: oracle object-pointcloud baseline plus `ndf_point_cloud_{placeholder}`
- `semantic_pointwise_hybrid`: oracle object-pointcloud baseline plus `semantic_point_cloud_{placeholder}`

What is still missing is the combination of these ideas:

- raw point clouds should come from actorseg mask projection, not direct oracle object point clouds
- NDF or semantic pointwise features should still be added as extra branches on top of that raw actorseg point cloud

This is the closest simulator analogue to a future real-world segmentation-plus-projection pipeline.

## Goal

Create separate train/eval paths for:

1. actorseg raw point cloud + NDF pointwise hybrid
2. actorseg raw point cloud + semantic pointwise hybrid

The design must preserve the current successful hybrid observation structure and must not change any existing working
pipelines.

## Non-Goals

- Do not modify the semantics of existing `objpc_actorseg`, `ndf_pointwise_hybrid`, or `semantic_pointwise_hybrid`.
- Do not redesign the DP3 encoder.
- Do not change how actor segmentation is collected from the simulator.
- Do not add SAM-based logic here.

## Approaches Considered

### A. New dedicated actorseg+hybrid pipelines

Add separate preprocess/train/eval/config paths that directly generate actorseg-based hybrid observations.

Pros:

- Keeps old experiment definitions intact
- Easiest to reason about
- Matches how the current repo names and isolates experiment families

Cons:

- Adds several new scripts/configs

### B. Reuse `objpc_actorseg.zarr` and post-process into hybrid zarr

Generate actorseg raw zarr first, then derive NDF / semantic hybrid zarr in a second step.

Pros:

- Some preprocessing reuse

Cons:

- More indirection
- Eval still needs a separate online hybrid path
- Harder to keep train/eval symmetry obvious

### C. Mutate current hybrid paths to optionally use actorseg

Add flags to existing NDF / semantic hybrid scripts and runtime code.

Pros:

- Fewer top-level files

Cons:

- Pollutes already-stable paths
- Makes experiment names ambiguous
- Higher regression risk

## Decision

Use Approach A.

Create new dedicated paths:

- `objpc_actorseg_ndf_pointwise_hybrid`
- `objpc_actorseg_semantic_pointwise_hybrid`

## Data Flow

### Offline Preprocessing

For each episode and frame:

1. Use actor segmentation plus camera intrinsics/extrinsics to project scene point clouds into per-placeholder object
   point clouds, exactly like `process_data_objpc_actorseg.py`.
2. Merge all placeholders into the main raw `point_cloud`.
3. For placeholders that have feature models:
   - compute `ndf_point_cloud_{placeholder}` or
   - compute `semantic_point_cloud_{placeholder}`
4. Write one zarr containing:
   - `point_cloud`
   - `state`
   - `action`
   - optional `object_point_cloud_A/B` for debugging
   - optional `ndf_point_cloud_A/B` or `semantic_point_cloud_A/B`

This preserves the successful hybrid structure while changing only the raw point-cloud source.

### Online Eval

At each eval step:

1. Extract per-placeholder actorseg object point clouds once.
2. Merge them into the main raw `point_cloud`.
3. Reuse the same extracted per-placeholder point clouds to compute:
   - `ndf_point_cloud_A/B` or
   - `semantic_point_cloud_A/B`

The same placeholder clouds must feed both the raw branch and the feature branch, so train/eval semantics stay aligned.

## Observation Structure

### NDF ActorSeg Hybrid

- `point_cloud`: merged raw actorseg object point cloud for `{A}+{B}`
- `ndf_point_cloud_A/B`: `[xyz | ndf_feature]`
- `agent_pos`

### Semantic ActorSeg Hybrid

- `point_cloud`: merged raw actorseg object point cloud for `{A}+{B}`
- `semantic_point_cloud_A/B`: `[xyz | semantic_feature]`
- `agent_pos`

DP3 will still encode each point-cloud key with its own PointNet branch and then concatenate branch-level features.
This design does not change encoder semantics.

## File Plan

### New Files

- `policy/DP3/scripts/process_data_ndf_pointwise_actorseg_hybrid.py`
- `policy/DP3/scripts/process_data_semantic_pointwise_actorseg_hybrid.py`
- `policy/DP3/process_data_ndf_pointwise_actorseg_hybrid.sh`
- `policy/DP3/process_data_semantic_pointwise_actorseg_hybrid.sh`
- `policy/DP3/train_ndf_pointwise_actorseg_hybrid.sh`
- `policy/DP3/train_semantic_pointwise_actorseg_hybrid.sh`
- `policy/DP3/eval_ndf_pointwise_actorseg_hybrid.sh`
- `policy/DP3/eval_semantic_pointwise_actorseg_hybrid.sh`
- `policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/robot_dp3_objpc_actorseg_ndf_pointwise_hybrid.yaml`
- `policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/robot_dp3_objpc_actorseg_semantic_pointwise_hybrid.yaml`
- `policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/task/demo_task_objpc_actorseg_ndf_pointwise_hybrid.yaml`
- `policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/task/demo_task_objpc_actorseg_semantic_pointwise_hybrid.yaml`
- focused tests for runtime observation building and wrapper arg plumbing

### Updated Files

- `policy/DP3/deploy_policy.py`
- `policy/DP3/Command.md`

## Naming

Dataset suffixes:

- `-objpc-actorseg-ndf-pointwise-hybrid.zarr`
- `-objpc-actorseg-semantic-pointwise-hybrid.zarr`

Train settings:

- `${task_config}-objpc-actorseg-ndf-pointwise-hybrid`
- `${task_config}-objpc-actorseg-semantic-pointwise-hybrid`

Config names:

- `robot_dp3_objpc_actorseg_ndf_pointwise_hybrid`
- `robot_dp3_objpc_actorseg_semantic_pointwise_hybrid`

## Compatibility

- Existing `objpc_actorseg` remains unchanged.
- Existing `ndf_pointwise_hybrid` remains unchanged.
- Existing `semantic_pointwise_hybrid` remains unchanged.
- New checkpoints and zarrs get separate names and directories.

## Test Plan

### Required Failing-Then-Passing Tests

Add focused tests proving:

- actorseg+NDF hybrid runtime keeps raw merged `point_cloud` and still emits `ndf_point_cloud_A`
- actorseg+semantic hybrid runtime keeps raw merged `point_cloud` and still emits `semantic_point_cloud_A`
- wrapper scripts / wrapper Python entrypoints attach the correct output suffix and mode

### Verification

- `bash -n` for new shell scripts
- `python -m py_compile` for new/updated Python files
- run the focused tests

## Risks

- actorseg extraction is noisier than oracle object point clouds, so the hybrid gain may be smaller than in the oracle
  pipeline
- online eval cost increases because actorseg extraction and feature extraction happen in the same loop
- if code reuse is done carelessly, existing actorseg or hybrid paths could regress

## Success Criteria

- The repo has separate actorseg+NDF hybrid and actorseg+semantic hybrid train/eval paths
- Both paths preserve the raw merged actorseg object point cloud while adding their feature branches
- Existing working pipelines remain behaviorally unchanged
