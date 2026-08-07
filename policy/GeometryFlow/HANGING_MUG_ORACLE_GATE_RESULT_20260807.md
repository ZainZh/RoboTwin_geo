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

## Next locked gate

1. Train a camera-only relative functional-frame estimator on development mug
   IDs only, first with NDF and then with UTONIA if the feature cache is
   available.
2. Replace the oracle global token with the predicted token without changing
   the action-policy architecture or action targets.
3. Compare `raw`, `zero_global`, `predicted`, `predicted-shuffled`, and oracle
   on fixed held-out identities and paired initial states.
4. Only if predicted geometry retains a useful fraction of the oracle gain,
   run paired closed-loop success evaluation and then replicate in ACT/DP3.

