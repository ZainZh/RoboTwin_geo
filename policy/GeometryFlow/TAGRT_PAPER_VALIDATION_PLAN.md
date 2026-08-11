# TAGRT Paper Validation Plan

## Fixed method and claim

The method remains a learned visuomotor policy, not a residual planner:

camera point clouds -> NDF functional correspondence -> local anchor-relation
tokens + global remaining-SE(3) token -> relation/temporal Transformer ->
continuous robot action chunk + independent continuous gripper head.

The target claim is that task-aligned, relation-preserving geometry improves
the accuracy, closed-loop robustness, and object generalization of a learned
manipulation policy over capacity-matched raw point-cloud conditioning.

## Gate 1: single-scene interface admission (passed)

- Exact dense-control replay admitted 9/10 demonstrations.
- TAGRT reduced the nine-trajectory fitting MSE by 13.68% relative to Raw.
- Under identical seed and matched gripper-head calibration:
  correct TAGRT succeeded 1/1, Raw failed 0/1, and zero-geometry TAGRT failed
  0/1.

This admits the complete pipeline but is not a rate or generalization result.

## Gate 2: replication batch (next)

Collect 30 replay-admitted training trajectories, balanced across active arm
and the training shoe identities 2/3/4/7/8/9.  Preserve every rejected replay
and its rejection reason in a manifest.

Train capacity-matched Raw and TAGRT policies with policy seeds 0/1/2.  Apply
the same two-stage optimization to both: full motion-policy training followed
by frozen-motion gripper-head calibration.  Do not tune on online test seeds.

Evaluate every checkpoint on the same 10 feasible, non-training scene seeds.
Required conditions are correct TAGRT, Raw, and same-checkpoint zero geometry.
Record task success, grasp success, stable placement, EEF-to-object progress,
first grasp time, final SE(3) error, NDF disagreement, and failure category.

Advance only if the direction is consistent across all three policy seeds and
TAGRT improves paired online success/progress without a new hand-written gate.

## Gate 3: strict object generalization

Build the paper split without trajectory leakage:

- train shoes: 2/3/4/7/8/9;
- validation shoes: 1/6;
- held-out test shoes: 0/5.

Target at least 120 replay-admitted training demonstrations, balanced by shoe
and active arm.  Validation-shoe demonstrations may select checkpoints and
hyperparameters but are never optimized as training samples.  Test-shoe
demonstrations are used only to identify feasible evaluation seeds and measure
expert/replay ceilings, never for training, selection, or tuning.

Run paired Raw/TAGRT evaluation with confidence intervals.  The primary metric
is full-task success; grasp, placement, stable hold, and pose error are
pre-declared secondary metrics.

## Gate 4: paper ablations and robustness

After Gate 3 is positive, run:

1. local-only, global-only, zero, and wrong-goal/shuffled relation tokens;
2. camera NDF versus oracle geometry as a perception upper bound;
3. NDF versus PCA/raw geometry encoder controls;
4. camera occlusion, point dropout, and calibration-noise stress tests;
5. regression versus flow-matching action decoder as a decoder ablation;
6. one interface-matched mainstream backbone experiment for portability.

The flow decoder and second backbone are supporting ablations, not changes to
the central task-aligned geometry claim.

## Reporting discipline

Use paired evaluation seeds and report Wilson or bootstrap 95% confidence
intervals.  Keep training-internal pilots separate from held-out results.
Simulator object pose may generate supervision and evaluation metrics, but the
deployed policy input must remain camera point clouds plus robot state.
