# Strict Cross-Fit Short-Horizon Geometry Tokens — 2026-08-09

## Frozen claim and scope

This experiment tests one narrow claim: a learned short-horizon manipulation
policy benefits from task-aligned object geometry when the geometry is
estimated from camera point clouds and supplied as structured local and global
tokens. It does not use an analytic controller, inverse kinematics, a residual
planner, stage prediction, or a grasp token at policy inference.

The evaluated pipeline is:

1. current and target segmented camera point clouds;
2. a category NDF ensemble estimates the current functional frame, while the
   target functional frame is fitted from the target camera observation;
3. local per-anchor functional coordinates/flow and one global remaining
   relative SE(3) token condition a Transformer;
4. the Transformer predicts a six-step, 14-D bimanual action chunk.

Simulator state is used only to construct training labels and diagnostics. The
policy inputs are camera point clouds, predicted geometry tokens, and robot
proprioception.

## Leakage controls and data

The object split is frozen as train shoes `2,3,4,7,8,9`, validation shoes
`1,6`, and blind test shoes `0,5`.

Training-object NDF features are strictly leave-one-object-out cross-fitted:
the NDF ensemble used for a training row never saw that row's shoe object.
Validation and test use the final NDF ensemble trained only on the six training
objects. Confidence calibration uses cross-fitted training-object disagreement
only.

The policy dataset contains 3,200 short-horizon samples constructed from 1,600
source states and exactly two real-camera target relations per source. It is
balanced to 32 source states per demonstration and restricted to approximately
20 cm / 90 degrees. The rigid-grasp expert is label-only and is unavailable at
inference.

Primary artifacts:

- `outputs/geometry_flow/shoe_multigoal_relation_v1/fold0_strict_crossfit_medoid_p95_temporal_frames.json`
- `outputs/geometry_flow/shoe_multigoal_relation_v1/task_flow_fold0_strict_crossfit_ndf_camera.json`
- `outputs/geometry_flow/shoe_multigoal_relation_v1/task_flow_fold0_strict_crossfit_multigoal2_rel_bal32_short20cm90deg.json`

## NDF functional-frame reliability

The estimator uses three independently trained NDF frame heads, medoid
aggregation, a 95th-percentile disagreement gate calibrated on cross-fitted
training objects, and causal temporal branch stabilization. The temporal rule
does not predict task stage or actions. It only rejects/repairs discontinuous
equivalent frame branches before token construction.

On blind test frames, the confidence gate accepts 95.49% of observations. On
the 1,377 accepted observations, functional-frame rotation error is 2.15
degrees median and 4.29 degrees at p90, with zero greater-than-90-degree flips.
The temporal check identified 80 ambiguous frames over all splits and repaired
69. Rejected frames include the rare large branch flips.

The final camera-derived relative task frame on all blind test rows has 2.50 cm
/ 4.14 degrees mean error, 2.16 cm / 2.98 degrees median error, and 4.72 cm /
5.31 degrees p90 error. The camera target-frame fit alone has 0.21 cm / 1.40
degrees mean blind error.

These values support using NDF as a category-level functional-frame estimator.
They do not support treating raw NDF descriptors as a complete policy input.

## Three-seed offline policy results

All conditions use 1,219,779 parameters, identical splits, optimization, and
action targets. `Normal` receives the correct two-level geometry tokens;
`zero` removes them; `shuffled` assigns geometry from another row. Values are
mean +/- sample standard deviation over policy seeds 0, 1, and 2.

| Trained condition | Endpoint translation (cm) | Endpoint rotation (deg) | Per-step translation (mm) | Per-step rotation (deg) |
|---|---:|---:|---:|---:|
| Correct geometry | **5.023 +/- 0.081** | **18.660 +/- 1.732** | **8.485 +/- 0.154** | **3.131 +/- 0.287** |
| Zero geometry | 6.903 +/- 0.223 | 52.734 +/- 1.202 | 11.570 +/- 0.361 | 8.807 +/- 0.218 |
| Shuffled geometry | 5.817 +/- 0.860 | 49.872 +/- 3.607 | 9.795 +/- 1.410 | 8.318 +/- 0.600 |

The dominant stable gain is rotation/direction prediction. Correct geometry
reduces endpoint rotation by 64.6% relative to zero geometry and by 62.6%
relative to shuffled geometry.

## Same-checkpoint causal token intervention

For each normal checkpoint, weights, camera clouds, proprioception, and test
rows are held fixed. Only the geometry tokens are replaced by zeros or shuffled
tokens at evaluation.

| Input to the same learned checkpoint | Endpoint translation (cm) | Endpoint rotation (deg) |
|---|---:|---:|
| Correct tokens | **5.023 +/- 0.081** | **18.660 +/- 1.732** |
| Tokens zeroed | 5.956 +/- 0.098 | 51.259 +/- 1.853 |
| Tokens shuffled | 5.415 +/- 0.146 | 54.732 +/- 1.120 |

Because no model weight or other observation changes, this is direct evidence
that the learned policy uses the task-aligned relation rather than merely
benefiting from extra parameters.

Intervention artifacts are stored as
`policy_local_global_rotaux01_seed{0,1,2}_normal/interventions_test05.json`
under `outputs/geometry_flow/shoe_multigoal_relation_v1/`.

## Exact paired closed-loop result

The online evaluator restores the same in-process SAPIEN snapshot for every
variant, executes six receding-horizon policy calls, and keeps executing after
the first threshold crossing so all variants receive the same action budget.
Final success requires translation at most 4 cm and rotation at most 15
degrees.

Ten manifest states were reproducible in all three policy-seed evaluations.
The table uses this common set.

| Policy seed | Correct geometry | Same weights, geometry zeroed | Gain |
|---|---:|---:|---:|
| 0 | 8/10 (80%) | 5/10 (50%) | +30 pp |
| 1 | 6/10 (60%) | 3/10 (30%) | +30 pp |
| 2 | 6/10 (60%) | 3/10 (30%) | +30 pp |
| Mean +/- SD | **66.7 +/- 11.5%** | **36.7 +/- 11.5%** | **+30.0 +/- 0.0 pp** |

Final-pose means over the common states are:

| Condition | Translation (cm) | Rotation (deg) |
|---|---:|---:|
| Correct geometry | **3.293 +/- 0.069** | **9.932 +/- 0.822** |
| Same weights, geometry zeroed | 3.880 +/- 0.395 | 14.119 +/- 0.838 |

Correct geometry improves the paired final pose by 0.587 cm and 4.188 degrees
on average. After averaging each scene over the three policy seeds, six scenes
favor correct geometry, zero favor zero geometry, and four tie. The two-sided
exact sign-test value for the six non-tied scenes is 0.03125. The pooled 20/30
versus 11/30 count is descriptive because three policies share each scene.

Closed-loop artifacts:

- `outputs/geometry_flow/shoe_multigoal_relation_v1/closedloop_seed0_clean_zero_pilot_ep02345.json`
- `outputs/geometry_flow/shoe_multigoal_relation_v1/closedloop_seed0_clean_zero_ep06_19.json`
- `outputs/geometry_flow/shoe_multigoal_relation_v1/closedloop_seed12_clean_zero_valid11.json`

## Supported conclusion

The current evidence supports the mechanism and the route:

1. cross-fitted NDF can recover a stable functional frame from camera point
   clouds on blind shoes when ambiguity is explicitly gated;
2. two-level task-aligned relation tokens produce large, seed-stable offline
   action gains, especially for rotation;
3. same-checkpoint interventions causally attribute the gain to correct token
   content;
4. the effect transfers to exact paired closed-loop execution and is positive
   in every policy seed.

This freezes the short-horizon two-level geometry-token architecture for the
next evidence gate. NDF is one estimator inside the method, not the paper's
only contribution; it can later be replaced by another category representation
without changing the policy interface.

## Remaining limitations and next gate

This is not yet a complete T-ASE claim.

- Closed-loop evidence is one task and one object category, with only ten
  common reproducible scenes. A second category/task with an independently
  trained category estimator is required.
- Several manifest states fail before policy execution because motion-planner
  replay cannot reproduce the nominal endpoint within tolerance. They are
  paired protocol errors, not policy failures, but the next benchmark should
  use stored simulator snapshots that reproduce all states without planning.
- Some trajectories cross the success threshold and then leave it under the
  forced six-call protocol. Both first-threshold and fixed-budget-final success
  should be reported. Policy-side terminal/hold behavior may be trained, but no
  task-stage predictor is needed.
- Translation is now the main execution bottleneck. The next model change
  should target action decoding/endpoint consistency while holding the geometry
  representation fixed.

The next confirmatory experiment should therefore preserve this architecture,
collect a fully replayable fixed-state benchmark, and repeat the predeclared
correct/zero/shuffled comparisons on one additional short-horizon category.
