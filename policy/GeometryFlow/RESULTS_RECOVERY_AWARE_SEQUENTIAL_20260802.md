# Recovery-aware sequential learning: 2026-08-02 decision record

## Scope

This record covers the fold-0 shoe-placement development experiments run on
physical GPU 2.  The remote 4090 was not used and no dataset was copied across
sites.  All closed-loop numbers below use the same replay seed `100002`, so they
are diagnostics rather than paper-scale success estimates.

## What is established

1. A camera-selected object-in-gripper relation is a useful geometric
   representation.  On the fixed scene its fit cost is `1.8095e-4 m^2`, below
   the locked `2.5e-4 m^2` acceptance threshold.  The corresponding geometric
   relation planner previously reached `2.08 cm / 1.27 deg` and succeeded on
   this seed.  Geometry itself is therefore not the observed bottleneck.
2. PCA correspondence is an important rotation bottleneck.  With the original
   coupled decoder, replacing only PCA endpoint rotation by the privileged
   rotation improved the minimum error from `2.65 cm / 135.04 deg` to
   `4.93 cm / 73.54 deg`.
3. Causal factorization helps when the supplied rotation is correct.  The
   factorized decoder reduced the privileged-rotation result further to
   `3.93 cm / 56.07 deg`; its unit test also verifies that changing rotational
   flow cannot change predicted translation.
4. Direct deterministic action regression is the present failure point.  It
   turns a multi-stage recovery plan into a small averaged delta and then never
   reaches the geometric goal.  Adding more recovery trajectories, matching
   the camera-flow distribution, distillation, and a positive goal-aligned
   rotation parameterization did not repair this closed-loop failure.

## Critical data-interface bug found and fixed

The first cross-fitted camera-flow builder sampled a fresh anchor set from the
relation-phase query cloud, but training paired that flow with the task
dataset's fixed episode anchor identities.  Equal tensor shapes hid the point
identity mismatch and corrupted displacement and SE(3) tokens.

`build_grasp_relation_flow_prediction.py` now recovers each rigid transform
from the online flow and applies the transform to the dataset anchor set.  The
new unit test checks exact transfer of a translation-plus-rotation schedule.
The corrected 74-episode artifact is:

`outputs/geometry_flow/task_flow_grasp_relation_crossfit_anchor_consistent_d1248_r2d4_fold0.npz`

An offline action-prior consistency probe changed as follows:

- broken camera flow: mean geometry-candidate action error `0.113`;
- anchor-consistent camera flow: `0.072`;
- oracle flow: `0.074`.

Therefore, all earlier camera-matched-policy failures built with the broken
artifact are invalid evidence against NDF, UTONIA, DINO, or camera geometry.
The corrected experiments below remain valid and show that the data fix alone
is not sufficient.

## Controlled results

| Training / decoder | Offline endpoint (cm / deg) | Fixed-seed closed loop: min (cm / deg) | Decision |
|---|---:|---:|---|
| Original factorized, oracle flow, 20 recoveries | `1.288 / 5.076` | camera relation `8.02 / 26.06` | Best learned camera diagnostic, still 0/1 |
| Factorized, oracle flow, 24 recoveries | `1.258 / 5.744` | camera relation `15.78 / 74.15` | More recovery BC made action averaging worse |
| Goal-aligned, oracle flow, 24 recoveries | `1.365 / 4.076` | camera relation `10.25 / 94.69` | Offline gain did not transfer |
| Goal-aligned, broken camera-matched flow | `1.789 / 5.826` | `13.15 / 126.89` | Invalid training artifact; retained only as diagnosis |
| Direct factorized, anchor-consistent camera flow | `2.085 / 7.689` | `19.16 / 135.08` | Reject direct BC |
| Goal-aligned, anchor-consistent camera flow | `2.255 / 5.038` | `17.52 / 104.56` | Reject constrained direct BC |

The corrected direct model averaged only `1.52 mm` translation per step and
made zero geometric progress.  The corrected goal-aligned model averaged
`0.185 deg` rotation and also made zero progress.

## Why direct BC fails

Recovery is collected by invoking the scripted `place_actor` expert after an
on-policy perturbation.  That expert executes a sequence: pre-place waypoint,
final placement waypoint, release, and settle.  A single remaining object-flow
endpoint does not identify which of these subgoals generates the next delta.

The measured first-action statistics support this diagnosis:

- nominal recovery-direction cosine: `0.715`;
- old recovery cosine: `0.339`;
- new on-policy camera recovery cosine: `0.311`;
- only `39.0%` of new recovery first actions have cosine above `0.5`.

Forcing every action to point toward the final object rotation makes MSE prefer
nearly zero magnitude; unconstrained MSE averages the incompatible stages.
This is an action/sequence interface problem, not evidence that the geometric
encoder direction is wrong.

## Locked next method

The next model should be a learned policy with a structured geometric action
prior, not a pure analytic planner and not another token-concatenation variant:

1. Estimate sparse rigid object flow and confidence from camera A/B clouds.
2. Use the selected object-in-gripper relation to lift object flow into two EEF
   SE(3) waypoint tokens: pre-place and final-place.
3. Let a temporal policy predict the discrete stage and a bounded residual
   action around the selected geometric waypoint action.
4. Learn release/settle decisions and residuals from nominal plus on-policy
   recovery data.  The geometric prior remains observable; privileged object
   pose is used only for labels and evaluation.
5. Compare against the identical policy with zero, shuffled, and wrong-object
   action priors, plus the pure geometric-prior controller.  This separates
   representation value from planner value and from extra parameters.

This preserves a learning-centered T-ASE story: geometry does not merely add
tokens; it constrains the action hypothesis space and exposes task stage,
while learning handles contact, residual correction, and recovery.

## Scale-up gate

Do not start the cross-fold large experiment yet.  First require, on five
locked development seeds:

- nonzero strict success;
- a paired improvement over the pure geometric prior in either success or
  stable pose error;
- nonzero flow progress without action-magnitude collapse;
- degradation under shuffled/wrong-object action priors.

Only after this gate should GPU 2 run cross-fold seeds.  The remote 4090 is
reserved for independent unit/offline checks or locally recollected small data.
