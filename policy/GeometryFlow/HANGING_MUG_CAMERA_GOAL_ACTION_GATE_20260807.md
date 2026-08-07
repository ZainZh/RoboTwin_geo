# Camera-Only Goal-Frame Action Gate on Hanging Mug — 2026-08-07

## Question

Does a task-aligned geometric representation improve a learned short-horizon
manipulation policy on a new object/task when the representation is computed
only from live A/B camera point clouds?

This is a development gate, not a paper-scale benchmark.  The held-out set has
two mug identities and four paired simulator snapshots.  Three independently
trained policy seeds reuse those four snapshots, so the 12 rollouts per method
are not 12 independent scenes.

## Method under test

The experiment keeps the action generator learned.  There is no geometric
planner, inverse-kinematics action prior, simulator pose input, or learned
residual controller.

1. A frozen equivariant frame ensemble estimates the current mug rotation from
   the segmented A point cloud.
2. Trimmed ICP registers the segmented rack B point cloud to one frozen
   training reference and estimates the desired mug rotation.
3. The current/goal relation is encoded as a zero-translation relative rotation
   token.  Translation is exactly zero by construction.
4. The camera-derived goal rotation also defines the policy's `goal_action`
   coordinate frame.  Demonstrated rotation actions are learned in this frame
   and predictions are transformed back to the world frame at runtime.
5. A Transformer action-chunk policy consumes point clouds, proprioception, and
   the relation token.  Closed loop executes one learned action and replans.

The online provider reproduces the offline camera token on all 124 held-out
frames with mean rotation discrepancy below `1e-5` degrees.  Simulator object
poses are used only for evaluation metrics.

The frame models in this gate use the NDF-style equivariant architecture but
are task-trained frame estimators.  They do **not** use the pretrained
`shoe.pth`; earlier experiments found that shoe checkpoint transfers negatively
to mugs.

## Three-seed offline held-out action result

All values are mean ± sample standard deviation over policy seeds 0/1/2 on 124
held-out frames.  Lower is better.

| Condition | Delta translation (mm) | Delta rotation (deg) | Endpoint translation (cm) | Endpoint rotation (deg) |
|---|---:|---:|---:|---:|
| Exact-zero token | 2.638 ± 0.095 | 2.068 ± 0.133 | 1.464 ± 0.064 | 12.055 ± 0.831 |
| Camera geometry + goal-action frame | **2.575 ± 0.019** | **1.895 ± 0.200** | **1.431 ± 0.010** | **10.936 ± 1.361** |

The camera geometry condition wins endpoint rotation in all three policy seeds.
Its per-seed endpoint-rotation gains are 1.472, 0.281, and 1.604 degrees, for a
mean gain of 1.119 degrees.  This is about twice the mean gain of the earlier
world-action additive-token experiment (0.490 degrees), supporting geometric
canonicalization rather than unconstrained feature concatenation.

## Strict paired closed-loop result

Success requires final translation below 2 cm and rotation below 10 degrees.
Each policy receives the same in-process SAPIEN snapshot, keeps the gripper
closed, executes at most ten one-step receding-horizon calls, and uses the
camera-only token described above.

| Policy seed | Exact-zero successes | Camera geometry successes |
|---|---:|---:|
| 0 | 1/4 | 1/4 |
| 1 | 1/4 | 0/4 |
| 2 | 0/4 | 2/4 |
| Pooled descriptive count | 2/12 | **3/12** |

Across the 12 paired rollouts, mean final metrics are:

| Condition | Translation (cm) | Rotation (deg) |
|---|---:|---:|
| Exact-zero token | **1.836** | **20.514** |
| Camera geometry + goal-action frame | 2.064 | 23.349 |

The success count is weakly positive, but the mean endpoint metrics are not.
One seed-2 geometry rollout suffered approximately 49 degrees of hand-object
rotation drift and ended at 54.9 degrees error.  This single catastrophic
contact trajectory heavily affects the four-scene mean.

The seed-1 episode-2 geometry rollout is an important boundary case: it reaches
7.66 degrees rotation error but ends at 2.129 cm translation, missing the strict
success threshold by 1.29 mm.  The current token contains no translation, so
this is not evidence that its rotation estimate is wrong.

## Grasp-slip audit

Hand-object drift strongly correlates with final rotation error:

| Condition | Pearson correlation: rotation slip vs final rotation error |
|---|---:|
| Exact-zero token | 0.744 |
| Camera geometry + goal-action frame | 0.830 |

Restricting descriptively to rollouts whose final hand-object rotation drift is
at most 20 degrees gives:

| Condition | Rollouts | Successes | Final translation (cm) | Final rotation (deg) |
|---|---:|---:|---:|---:|
| Exact-zero token | 8 | 2 | **1.542** | 14.161 |
| Camera geometry + goal-action frame | 8 | **3** | 1.675 | **14.104** |

This post-hoc subset does not prove causality, because slip can itself depend on
the policy action.  It does show that gross grasp drift is a major error
amplifier.  Stable grasp is the mechanical interface that makes an EEF action
correspond to the intended object motion.  It is not the only missing factor:
even low-slip trials do not show a large mean geometry advantage.

## Current conclusion

The result is promising but **not closed** as a performance claim.

- Camera-only geometry is deployable and causally useful: it improves held-out
  action metrics in all three seeds and raises strict paired successes from
  2/12 to 3/12.
- Expressing actions in the estimated goal frame is more effective and more
  interpretable than simply adding a global geometry embedding.
- Rotation-only geometry cannot correct remaining translation or a changing
  hand-object relation.
- Grasp slip explains much of the closed-loop variance, but not all of it.
- Therefore the next experiment should not scale this exact model yet.

## Locked next experiment

Stay within geometry-enhanced policy learning:

1. Add a camera/proprioception-derived hand-object relation token alongside the
   existing object-target relation token.  Do not use it to analytically plan.
2. Predict both rotation and translation in the task-aligned goal frame; retain
   exact-zero/clean/shuffled causal controls for each relation.
3. Train with short post-grasp correction demonstrations that include moderate
   contact/slip perturbations, while separately reporting stable-grasp and
   slip-stress strata.
4. Repeat the frozen four-snapshot gate first.  Promote to new mug identities
   and a larger benchmark only if the geometry condition wins all policy seeds
   and improves both strict success and endpoint means.

## Machine-readable artifacts

- `outputs/geometry_flow/hanging_mug_cross_task_v1/dev24_task_flow_camera_rotation_temporal.npz`
- `outputs/geometry_flow/hanging_mug_cross_task_v1/hanging_mug_camera_rotation_calibration.npz`
- `outputs/geometry_flow/hanging_mug_cross_task_v1/closedloop_id05_pilot4_snapshots.json`
- `outputs/geometry_flow/hanging_mug_cross_task_v1/closedloop_id05_pilot4_goal_action_seed{0,1,2}_10calls.json`
- `outputs/geometry_flow/hanging_mug_cross_task_v1/camera_rotation_temporal_goal_action_{zero,predicted}_seed{0,1,2}/`
