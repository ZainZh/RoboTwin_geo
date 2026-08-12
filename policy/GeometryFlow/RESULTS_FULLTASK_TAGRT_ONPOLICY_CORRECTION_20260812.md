# Full-task TAGRT on-policy correction gate — 2026-08-12

## Decision

Do **not** scale the current deterministic `dense_joint26` regression policy or
collect more corrections with the same protocol.  Camera NDF/TAGRT geometry is
accurate enough to be consumed by the policy, and expert corrections are learned
offline, but neither a one-state nor a four-state correction batch closes the
complete online manipulation task.

The next minimum decision experiment keeps the camera NDF/TAGRT tokens, temporal
Transformer, dense action interface, demonstrations, split, and Raw comparison
fixed, and changes only the deterministic regression action decoder to the
already implemented flow-matching decoder.

## Fixed learned-policy contract

Deployment remains:

```text
camera A/B point clouds
  -> frozen NDF functional-frame ensemble
  -> local anchor-flow + global remaining-SE(3) TAGRT tokens
  -> relation/temporal Transformer
  -> 15-step dense_joint26 action block
```

Simulator state and the expert are used only to collect/admit correction labels.
There is no online expert, planner, residual controller, stage gate, or simulator
pose input.

## Baseline gate before correction

The exact train-scene initializations are seeds `173,175,176,184,185,186`, source
episodes `49..54`, and shoes `8,9,2,3,4,7`.

| Policy | Complete success | Left gripper closes |
|---|---:|---:|
| Raw regression | 0/6 | 4/6 |
| TAGRT regression | 0/6 | 6/6 |
| TAGRT with zero geometry | 0/6 | 0/6 |

Correct TAGRT beat zero geometry on the previously recorded combined action
metric in 5/6 scenes.  This supports geometry consumption, but not complete-task
success.

## One-state correction gate

The original TAGRT policy was executed on seed 173 until immediately before its
first predicted left-gripper close.  The simulator expert then executed a
successful full continuation.  This yielded 146 observation frames, 1,924 dense
controls, and 127 policy training samples.

The correction was train-only; the original validation and test episodes were
unchanged.  Continuing from the calibrated checkpoints also reused the original
normalization statistics to avoid a silent coordinate-system change.

| Training | Offline correction normalized MSE | Online seed-173 success | First close control |
|---|---:|---:|---:|
| TAGRT before | 0.08521 | 0/1 | 750 |
| TAGRT, weight 4, LR 1e-4 | 0.00307 | 0/1 | never |
| TAGRT, weight 1, LR 3e-5 | 0.02481 | 0/1 | 1,908 |
| Raw, weight 4, LR 1e-4 | 0.00374 | 0/1 | 900 |

The correction is therefore learned, but fitting one continuation changes the
earlier rollout distribution.  The high-weight model never closes; the
conservative model closes too late.  Expert handoff from the updated TAGRT at
policy chunk 50 and chunk 25 was no longer recoverable, so those failed
continuations were rejected.

## Four-state correction gate

The original TAGRT policy was run on all six paired scenes.  A continuation was
admitted only when direct expert execution from the policy-induced pre-close
state passed the complete task success check.

| Seed | Shoe | Admission |
|---:|---:|---|
| 173 | 8 | success |
| 175 | 9 | rejected: final task check failed |
| 176 | 2 | rejected: expert plan failed |
| 184 | 3 | success |
| 185 | 4 | success |
| 186 | 7 | success |

The four admitted episodes produced 559 `15 x 26` dense-action samples.  They
were reindexed as episodes `60..63` and added only to training:

- train: 6,241 samples, including 559 correction samples;
- validation: original 1,904 samples, no correction samples;
- test: original 2,029 samples, no correction samples.

The frozen perception stack remained accurate on the admitted corrections:

- NDF functional-frame median error: 1.98 degrees;
- NDF p90 error: 6.49 degrees;
- camera-relative median rotation error: 2.62 degrees;
- camera-relative median translation error: 2.95 cm;
- camera goal-frame median error: 1.18 degrees and 0.20 cm.

Both Raw and TAGRT were continued from their calibrated regression checkpoints
with identical data, LR `3e-5`, correction weight `1`, and frozen original
statistics.

| Policy | Best validation normalized motion MSE | Correction normalized MSE before -> after | Complete success |
|---|---:|---:|---:|
| Raw regression | 0.06503 | 0.10110 -> 0.03888 | 0/6 |
| TAGRT regression | 0.02903 | 0.07050 -> 0.00869 | 0/6 |

TAGRT still has roughly half the nominal validation error and learns the
correction states substantially better than Raw, but neither improvement
translates into online task completion.  After correction TAGRT closes in four
of six scenes; the first close moves to controls `770..1155` when it occurs.
No scene enters and holds the complete pose-success tolerance.

## Online NDF degeneracy found and fixed

The first TAGRT seed-186 run terminated because all three NDF heads produced a
degenerate zero rotation after severe long-rollout object motion/occlusion.
Online inference now:

1. validates every ensemble rotation as finite, orthonormal, and right-handed;
2. aggregates only valid members;
3. assigns zero source confidence whenever any member is invalid;
4. reuses the last valid source rotation, or identity before the first valid
   estimate, to keep the policy input finite;
5. reports the invalid member count.

Nineteen online-frame/deployment tests pass.  The corrected seed-186 run no
longer crashes, reports three invalid members and zero confidence at the end,
and still fails the task.  The strict paired outcome is therefore Raw `0/6`,
TAGRT `0/6`, not an evaluation exception.

## Interpretation

This gate does not falsify task-aligned geometry as a policy condition:

- geometry changes the learned action and gripper decisions;
- wrong/zero geometry is worse offline and at the grasp-decision gate;
- frozen NDF/camera geometry is accurate on admitted correction states;
- TAGRT has a large paired offline advantage over Raw.

It does falsify the current claim that sparse nominal behavior cloning plus a
few expert continuations is sufficient for a deterministic long-horizon dense
joint regression policy.  The remaining bottleneck is the action-learning and
online rollout interface: the regressor averages/moves between sequential
action modes, and small changes alter the state distribution before the
corrected state is reached.

## Next decision experiment

Run the capacity-matched conditions already supported by the same policy code:

1. `raw_flow`;
2. `tagrt_flow`;

on the exact four-correction archive and episode split above.  Evaluate the same
six seeds.  Scale only if `tagrt_flow` obtains non-zero complete success and
beats `raw_flow`; if both remain zero, stop investing in this small full-task
backbone and port the unchanged TAGRT memory interface to a proven full-task
policy backbone.

## Canonical artifacts

- Raw data: `/shared2/sz/TAGRT-v1/simulator_data/place_shoe_geometry_marker/fulltask_tagrt_correction_preclose_train6_v1`
- Processed corrections: `/shared2/sz/TAGRT-v1/processed_datasets/fulltask_tagrt_v2/correction_preclose_train4_v1`
- Merged paired dataset: `/shared2/sz/TAGRT-v1/processed_datasets/fulltask_tagrt_v2/gate2_left60_correction_train4_v1/fulltask_history3_dense15.npz`
- Raw model: `/shared2/sz/TAGRT-v1/models/fulltask_tagrt_v2/gate2_left60_correction_train4_v1_raw_seed0/raw_reg_seed0.pt`
- TAGRT model: `/shared2/sz/TAGRT-v1/models/fulltask_tagrt_v2/gate2_left60_correction_train4_v1_tagrt_seed0/tagrt_reg_seed0.pt`
- Evaluation roots: `/shared2/sz/TAGRT-v1/experiment_outputs/robotwin_eval_result/place_shoe_geometry_marker/GeometryFlow.deploy_fulltask_tagrt_v2/demo_clean_3d_object_pc_geometry_marker_densecontrol_gate2_train30/`
