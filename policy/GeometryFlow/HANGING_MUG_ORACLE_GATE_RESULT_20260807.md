# Hanging-Mug Cross-Task Oracle Gate — 2026-08-07

## Question

Does the task-aligned global relative SE(3) token provide action-learning
value beyond local point/flow tokens when the architecture is transferred
from shoe placement to a short-horizon mug-hanging phase?

This is an oracle representation ceiling, not a deployable camera-only result.
The global functional frame is derived from simulator task state. Its purpose
is to decide whether estimating that relation from NDF/UTONIA features is worth
the next experiment.

## Frozen data contract

- Task: `hanging_mug`, right gripper already holding the mug.
- Development identities: mug IDs 0--7; three admitted episodes per ID.
- Difficulty: eight episodes at each of 2--4 cm / 10--25 degrees,
  3--5 cm / 20--35 degrees, and 4--6 cm / 30--45 degrees.
- Admission: initial state fails 2 cm / 10 degrees and expert endpoint passes
  2 cm / 10 degrees.
- Dataset: 24 HDF5 episodes, 501 action samples after static-frame collapse.
- Fold 0 effective train IDs: 2, 3, 4, 7; validation IDs: 1, 6; held-out test
  IDs: 0, 5. Generic fold entries 8 and 9 are absent from this development set
  and therefore contribute no samples.
- Test samples per run: 124.

The collection audit passed exactly:

| audit | value |
|---|---:|
| episodes / videos | 24 / 24 |
| episodes per mug ID | 3 |
| episodes per difficulty | 8 |
| initial failures / admitted endpoints | 24 / 24 |
| initial error, mean | 2.85 cm / 23.46 deg |
| endpoint error, mean | 0.54 cm / 4.07 deg |
| worst admitted endpoint | 1.58 cm / 8.49 deg |
| demonstrations using two re-alignments | 12 |
| first-pass failures rescued by second re-alignment | 12 |

The original one-pass partial set (16 admitted episodes) is retained at
`dev24_onepass_partial_16` as diagnostics and is excluded from training.

## Conditions

All conditions use the same action horizon, optimizer, split, action targets,
and early-stopping rule.

1. `raw`: point tokens only (0.834 M parameters).
2. `zero_global`: local interaction/flow-token architecture with the explicit
   global relative-frame token zeroed (1.123 M parameters).
3. `full_oracle`: identical to `zero_global`, with the oracle relative
   functional SE(3) token enabled (1.123 M parameters).

The primary causal comparison is `full_oracle` versus `zero_global`, because
these models have identical capacity and differ only in the global token.

## Held-out offline results

Endpoint errors, lower is better:

| seed | raw trans (cm) | zero trans (cm) | full trans (cm) | raw rot (deg) | zero rot (deg) | full rot (deg) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.457 | 1.436 | 1.444 | 12.333 | 12.102 | 11.516 |
| 1 | 1.429 | 1.494 | 1.438 | 11.707 | 11.580 | 10.913 |
| 2 | 1.593 | 1.440 | 1.369 | 12.519 | 10.668 | 9.970 |
| mean +/- sample SD | 1.493 +/- 0.088 | 1.457 +/- 0.033 | **1.417 +/- 0.042** | 12.186 +/- 0.426 | 11.450 +/- 0.726 | **10.800 +/- 0.779** |

Per-step active-arm errors:

| condition | translation (mm) | rotation (deg) |
|---|---:|---:|
| raw | 2.692 +/- 0.124 | 2.088 +/- 0.073 |
| zero_global | 2.633 +/- 0.042 | 1.971 +/- 0.119 |
| full_oracle | **2.573 +/- 0.063** | **1.867 +/- 0.130** |

Effect sizes across seeds:

- Full versus raw: endpoint rotation improves by 1.386 degrees (11.4%) and
  endpoint translation by 0.076 cm (5.1%) on the mean.
- Full versus the capacity-matched zero-global control: endpoint rotation
  improves in all three seeds by 0.650 +/- 0.057 degrees (5.7%).
- Full versus zero-global endpoint translation improves by 0.040 cm (2.7%) on
  the mean, but it is not seed-consistent (full loses seed 0 by 0.008 cm).

## Interpretation

The cross-task oracle gate is positive for the intended claim: an explicit
task-aligned relative SE(3) token improves learned functional orientation, and
the benefit cannot be explained only by the larger interaction-token model.
This repeats the shoe-task mechanism on a semantically different object-object
relation (mug handle to rack) and makes a camera-only estimator experiment
worthwhile.

The evidence is not yet a paper-level manipulation result:

- it is offline action prediction, not paired closed-loop task success;
- the global relation is privileged simulator state;
- the held-out set contains two mug identities and 124 temporally correlated
  samples;
- three optimization seeds do not replace new scenes or new object identities.

Consequently the defensible claim is **oracle task-aligned geometry transfers
and consistently reduces orientation error**. It is not yet valid to claim
camera-only generalization or improved hanging success.

## Current-mug camera-frame gate

The next experiment replaced only the **current mug functional frame** with a
frame estimated from the mug camera point cloud.  The rack/goal frame remains
the task-state label in this experiment.  It is therefore a partial-perception
causal gate, not a fully camera-only policy.

Two frame estimators used the same vector-neuron NDF architecture, the same
task-aligned functional-frame supervision, the same object split, and three
training seeds:

1. `shoe-pretrained`: initialized from `/shared2/sz/model/ndf/shoe.pth` and
   fully fine-tuned on the development mugs;
2. `random-equivariant`: identical architecture trained from random
   initialization on the development mugs.

Held-out single-model frame errors were:

| initialization | mean SO(3) error (deg) | median (deg) | p90 (deg) | flip rate |
|---|---:|---:|---:|---:|
| shoe-pretrained | 11.179 +/- 0.545 | 5.795 +/- 1.065 | 14.058 +/- 2.547 | 2.69% |
| random-equivariant | **8.745 +/- 0.712** | **4.231 +/- 0.393** | **9.216 +/- 2.070** | 2.42% |

The validation-selected two-model random ensemble reached 8.036-degree mean,
3.638-degree median, and 7.407-degree p90 test error.  Its disagreement gate
accepted 85.5% of test samples; accepted samples had 5.254-degree mean error,
but one 180-degree-like flip was still accepted.  Confidence is therefore
useful for triage but not yet safety-calibrated.

The frame-confidence interface was corrected before the policy runs:
confidence now gates both the source-frame consumers and the global relative
frame token.  Previously only the source field was populated, which could let
a low-confidence relative token enter the policy unmasked.

### Held-out policy results

The primary controls are capacity-matched:

- `zero_global`: global relation token exactly zeroed;
- `shuffled`: predicted global tokens shuffled across episodes;
- `predicted`: correctly aligned predicted global token.

Endpoint rotation errors for the random-equivariant estimator were:

| seed | raw (deg) | zero (deg) | shuffled (deg) | predicted (deg) |
|---:|---:|---:|---:|---:|
| 0 | 12.333 | 12.102 | 12.203 | **11.762** |
| 1 | 11.707 | 11.580 | 12.607 | **11.181** |
| 2 | 12.519 | 10.668 | 10.883 | **10.326** |
| mean +/- sample SD | 12.186 +/- 0.426 | 11.450 +/- 0.726 | 11.898 +/- 0.902 | **11.090 +/- 0.722** |

The predicted token beats the exact-zero control in all three seeds by
0.361 +/- 0.034 degrees, and beats shuffled tokens in all three seeds by
0.808 +/- 0.538 degrees.  It retains 55.4% of the oracle-versus-zero endpoint
rotation gain.  Endpoint translation is 1.447 +/- 0.042 cm versus
1.457 +/- 0.033 cm for zero; this small translation difference is not
seed-consistent and is not claimed as a positive result.

The shoe-pretrained predicted token also beat zero and shuffled in all three
seeds, reaching 11.157 +/- 0.656 degrees and retaining 45.0% of the oracle
rotation gain.  Despite its substantially worse standalone frame error, its
downstream mean is only 0.068 degrees worse than random initialization and the
per-seed ranking is not uniform.  Standalone frame error is therefore useful
but is not a sufficient policy-selection metric.

### Updated interpretation

This gate supports two distinct conclusions:

1. **The task-aligned relation token is causally useful.**  Correct predicted
   tokens beat both exact-zero and shuffled-token controls in every seed.
2. **The shoe NDF checkpoint is not the source of the mug gain.**  It is a
   negative-transfer initialization for mug frame estimation: random
   initialization improves mean frame error by 2.435 degrees under the same
   architecture and supervision.

Accordingly, the paper claim should be encoder-agnostic: the contribution is a
task-aligned geometric relation interface, with an equivariant encoder adapted
to the task.  We must not claim that category-mismatched NDF pretraining itself
improves mug manipulation.  UTONIA remains a representation comparison, not a
required component of the method.

## Fully camera-derived rotation-token gate

The rack/goal orientation was then removed from task-state inputs.  The
deployable rotation-token path is:

1. camera-A mug cloud -> task-trained vector-neuron frame ensemble;
2. camera-B rack cloud -> trimmed ICP against one training reference rack;
3. one goal-orientation label from reference episode 15 calibrates the rack
   geometry to the task frame;
4. predicted goal rotation times predicted current rotation inverse -> global
   relative rotation token.

Reference candidates came only from train IDs 2, 3, 4, and 7; the reference
was selected using validation IDs 1 and 6.  Test IDs 0 and 5 were not used to
select the reference, ICP settings, frame models, or confidence thresholds.
The token translation is **exactly zero** in every sample, so no task-state
translation can enter this gate.

### Necessary failure ablation

The first camera-only version used independent per-frame estimates.  Its test
relation error was 8.587-degree mean, 4.294-degree median, and 8.191-degree
p90, but it retained three near-180-degree frame flips (2.42%).  The resulting
policy did not pass:

| condition | endpoint rotation, mean +/- sample SD (deg) |
|---|---:|
| zero | 11.450 +/- 0.726 |
| shuffled camera token | 11.454 +/- 0.688 |
| camera token without symmetry stabilization | 11.469 +/- 0.508 |

Predicted minus zero was slightly worse by 0.019 degrees on the mean and lost
in seeds 1 and 2.  This failed result is retained; it shows that good median
frame error is insufficient when the representation contains rare discrete
axis flips.

### Label-free symmetry stabilization

Every vector-neuron frame has four determinant-positive axis-sign variants.
Within each episode, the stabilized estimator keeps the first camera estimate
and, at each later observation, selects the proper variant closest on SO(3) to
the preceding stabilized frame.  Temporal confidence requires an observable
step below 30 degrees and a best-versus-runner-up margin above 30 degrees.  No
simulator pose, action target, or success label is used by this operation.

On held-out test frames this changed exactly the three flipped rows and gave:

| metric | independent frames | stabilized frames |
|---|---:|---:|
| current-frame mean error | 8.036 deg | **4.116 deg** |
| current-frame p90 error | 7.407 deg | **7.198 deg** |
| current-frame flips >=90 deg | 3 / 124 | **0 / 124** |
| relative-token mean error | 8.587 deg | **4.666 deg** |
| relative-token p90 error | 8.191 deg | **7.987 deg** |
| relative-token maximum error | 175.300 deg | **14.796 deg** |
| confidence acceptance | 85.5% | **98.4%** |

### Three-seed policy result

Endpoint rotation errors, lower is better:

| seed | raw (deg) | zero (deg) | shuffled (deg) | camera predicted (deg) |
|---:|---:|---:|---:|---:|
| 0 | 12.333 | 12.102 | 11.774 | **11.643** |
| 1 | 11.707 | 11.580 | 10.916 | **10.862** |
| 2 | 12.519 | 10.668 | 11.221 | **10.375** |
| mean +/- sample SD | 12.186 +/- 0.426 | 11.450 +/- 0.726 | 11.304 +/- 0.435 | **10.960 +/- 0.640** |

The camera-predicted token beats exact zero in every seed by
0.490 +/- 0.214 degrees and beats shuffled camera tokens in every seed by
0.344 +/- 0.437 degrees.  It retains 75.3% of the oracle-versus-zero endpoint
rotation gain even though its translation field is zero.  It also improves on
raw points by 1.226 +/- 0.799 degrees.  Endpoint translation is
1.445 +/- 0.075 cm versus 1.457 +/- 0.033 cm for zero, but loses seed 1 and is
therefore not claimed as a consistent translation gain.

This is the first offline gate in this task where the complete rotation token
is camera-derived and both causal controls pass in all three seeds.  It does
not yet establish closed-loop success, full camera SE(3), or new-rack
generalization.  The held-out evidence is still only two mug identities, six
episodes, and 124 temporally correlated action samples; the rack geometry is
the same asset in every split.

## Next locked gate

1. Freeze the camera-frame ensemble, reference episode, ICP parameters,
   temporal thresholds, and the three selected policy checkpoints.
2. Implement the identical stateful symmetry stabilization in runtime and run
   paired closed-loop evaluation on fixed initial states for held-out mug IDs
   0 and 5: raw, zero, shuffled, and camera-predicted.
3. Report success, final SO(3), final translation, monotonic error reduction,
   every estimator rejection, and every runtime failure.  Offline action error
   alone cannot pass this gate.
4. If held-out closed-loop success passes, evaluate blind mug IDs 8 and 9 and
   then replicate the frozen token interface in ACT/DP3.  A learned camera
   translation token is a later ablation; it is not required for the current
   orientation claim and must not delay the paired success test.
