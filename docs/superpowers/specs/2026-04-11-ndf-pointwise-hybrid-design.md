# NDF Pointwise Hybrid Design

## Summary
Add a new DP3 path that keeps the baseline merged object point cloud in `obs.point_cloud` and also injects `ndf_point_cloud_{placeholder}` as an extra observation. This is intended to test whether NDF pointwise features help when they are added on top of the baseline object-PCD signal rather than replacing the feature placeholder inside the main context cloud.

## Problem
The current `ndf_pointwise` path removes NDF-feature placeholders from the main `point_cloud` context and places them only in `ndf_point_cloud_{placeholder}`. For `hanging_mug`, that means `{A}` mug is removed from the main cloud and only `{B}` rack remains in `point_cloud`. This changes the observation factorization relative to `objpc` and may discard localization cues that the baseline policy uses successfully.

The existing global-NDF path is not an adequate substitute because it produces pose-invariant low-dimensional descriptors and has already shown severe localization regressions during mug approach.

## Goal
Create a separate `ndf_pointwise_hybrid` pipeline that answers one specific question:

Does NDF pointwise help when the policy still receives the full baseline merged object point cloud?

## Non-Goals
- Do not change the behavior of existing `objpc`, `ndf`, `ndf_pointwise`, or `semantic_pointwise` pipelines.
- Do not redesign DP3 encoders or add correspondence-specific NDF heads in this change.
- Do not change raw data collection formats.

## Approaches Considered

### A. New `ndf_pointwise_hybrid` pipeline
Add a distinct preprocess/train/eval/config path whose only semantic difference from `ndf_pointwise` is that `point_cloud` keeps all placeholders.

Pros:
- Clean experiment isolation
- No ambiguity when comparing old and new results
- Low risk to existing scripts

Cons:
- Adds one more named pipeline

### B. Modify existing `ndf_pointwise`
Change the current behavior so feature placeholders remain in `point_cloud`.

Pros:
- Smaller code diff

Cons:
- Invalidates prior meaning of `ndf_pointwise`
- Makes existing checkpoints and notes harder to interpret

### C. Add a runtime/config flag to `ndf_pointwise`
Support both behaviors behind a flag such as `keep_feature_placeholders_in_context`.

Pros:
- Flexible

Cons:
- More branching in scripts and deploy path
- Easier to run experiments with inconsistent settings

## Decision
Use Approach A. Create a new `ndf_pointwise_hybrid` path.

## Data Flow

### Offline Preprocessing
Reuse the current `process_data_ndf_pointwise.py` logic with one behavioral change:

- Current `ndf_pointwise`:
  - `point_cloud` merges only context placeholders
  - `ndf_point_cloud_{placeholder}` stores feature point clouds for modeled placeholders

- New `ndf_pointwise_hybrid`:
  - `point_cloud` merges all placeholders, matching baseline `objpc`
  - `ndf_point_cloud_{placeholder}` is still produced for modeled placeholders

This preserves the baseline scene/object localization signal while adding NDF pointwise features as an auxiliary branch.

### Online Eval
In `deploy_policy.py`, the new config name should:
- build `obs.point_cloud` from all available object placeholders
- also compute `obs.ndf_point_cloud_{placeholder}` for placeholders with NDF models

The existing `ndf_pointwise` logic should remain unchanged.

## Files To Add Or Update

### New Files
- `policy/DP3/scripts/process_data_ndf_pointwise_hybrid.py`
- `policy/DP3/process_data_ndf_pointwise_hybrid.sh`
- `policy/DP3/train_ndf_pointwise_hybrid.sh`
- `policy/DP3/eval_ndf_pointwise_hybrid.sh`
- `policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/robot_dp3_ndf_pointwise_hybrid.yaml`
- `policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/task/demo_task_ndf_pointwise_hybrid.yaml`
- one focused test covering the changed observation semantics

### Updated Files
- `policy/DP3/deploy_policy.py`
- `policy/DP3/Command.md` with new example commands

## Naming
- Dataset suffix: `-objpc-ndf-pointwise-hybrid.zarr`
- Train setting: `${task_config}-objpc-ndf-pointwise-hybrid`
- Config name: `robot_dp3_ndf_pointwise_hybrid`

This keeps the meaning explicit: baseline object-PCD plus NDF pointwise branch.

## Compatibility
- Old checkpoints remain valid and unchanged.
- Old scripts keep their current behavior.
- New experiments get a separate checkpoint directory and eval output path.

## Test Plan

### Required Failing-Then-Passing Test
Add a small targeted test that proves:
- old `ndf_pointwise` excludes feature placeholders from `point_cloud`
- new `ndf_pointwise_hybrid` includes feature placeholders in `point_cloud`
- both paths still produce `ndf_point_cloud_A` when `{A}` has an NDF model

### Verification
- `bash -n` for new shell scripts
- `python -m py_compile` for new/updated Python files
- run the focused test

## Risks
- The hybrid path may still not improve performance if the generic PointNet branch cannot exploit NDF features well.
- The larger effective observation signal could increase overfitting or training instability.
- If NDF feature extraction itself is too noisy, retaining raw `point_cloud` may only mitigate harm rather than create a gain.

## Success Criteria
- The repo has a distinct `ndf_pointwise_hybrid` train/eval path.
- The new path preserves baseline merged object-PCD inputs while adding NDF pointwise features.
- Existing paths remain behaviorally unchanged.
- The repo can now test whether the NDF issue is primarily caused by replacing raw `{A}` geometry inside the main point-cloud branch.
