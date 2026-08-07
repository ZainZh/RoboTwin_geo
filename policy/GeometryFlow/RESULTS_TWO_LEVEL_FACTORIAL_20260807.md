# Two-Level Task-Aligned Geometry Token Factorial — 2026-08-07

## Scope and protocol status

This note records the frozen fold-1 mechanistic factorial defined in
`TWO_LEVEL_FACTORIAL_PROTOCOL_20260807.md`.  All variants use the same policy
phase, action chunk, nominal/recovery data split, early stopping, and three
policy seeds.  The external evaluation replays the same 32 stored SAPIEN
states (16 held-out GSO shoes, two snapshots per shoe) for every policy.

The 32 states are a post-hoc mechanistic set for the newly trained point/local
ablations, not a new blind benchmark.  The previously frozen clean-versus-zero
functional-frame intervention remains the confirmatory causal comparison.

## Three-seed offline held-out action results

Values are mean ± sample standard deviation over policy seeds 0, 1, and 2.
Lower is better.

| Condition | Endpoint translation (cm) | Endpoint rotation (deg) | Flow trajectory Chamfer (cm) |
|---|---:|---:|---:|
| Raw point tokens, 0.834 M params | 1.878 ± 0.221 | 8.514 ± 0.683 | — |
| Capacity-matched point tokens, 1.112 M params | 2.061 ± 0.215 | 8.821 ± 0.210 | — |
| Local relation tokens, no flow loss | 1.830 ± 0.120 | 9.016 ± 0.268 | 5.816 ± 0.124 |
| Local relation tokens + flow loss | 1.758 ± 0.102 | 8.702 ± 0.241 | 3.016 ± 0.047 |
| Local flow tokens + zero global token | 1.721 ± 0.072 | 8.733 ± 0.165 | 2.993 ± 0.023 |
| Full two-level tokens | **1.334 ± 0.014** | **3.789 ± 0.530** | **2.685 ± 0.070** |

Offline interpretation:

- Increasing point-policy capacity does not explain the full model's gain.
- Flow supervision reliably improves the learned geometry diagnostic
  (trajectory Chamfer 5.816 to 3.016 cm), but its action improvement is small.
- The large and seed-stable gain, especially in rotation, appears only when the
  policy receives the task-aligned global relative SE(3) token.

## External paired closed loop: policy seed 0

Success requires translation below 4 cm and rotation below 15 degrees within
six receding-horizon policy calls.  There were 32/32 complete states and zero
runtime errors.

| Condition | Successes | Success rate | Final translation mean (cm) | Final rotation mean (deg) |
|---|---:|---:|---:|---:|
| Raw point tokens | 14/32 | 43.8% | 4.385 | 14.659 |
| Capacity-matched point tokens | 15/32 | 46.9% | 4.427 | 14.636 |
| Local relation tokens, no flow loss | 14/32 | 43.8% | 4.246 | 14.818 |
| Local relation tokens + flow loss | 13/32 | 40.6% | 4.253 | 13.407 |
| Local flow tokens + zero global token | 13/32 | 40.6% | 4.446 | 15.134 |
| Full two-level tokens | **27/32** | **84.4%** | **3.715** | **9.539** |

All methods start from the same mean error, 5.194 cm / 12.956 degrees.  The
full model reduces both components by 1.479 cm / 3.417 degrees.  Every other
condition increases mean rotation error, except local flow, which still
increases it by 0.451 degrees.

### Exact paired success comparisons

The counts below use exactly matched states.  `Full only` means the full model
succeeded while the comparison failed; `comparison only` is the reverse.
Two-sided exact McNemar p-values are reported without multiple-comparison
correction and should remain secondary to the pre-frozen clean/zero contrast.

| Comparison | Full only | Comparison only | Exact p |
|---|---:|---:|---:|
| Raw point tokens | 15 | 2 | 0.00235 |
| Capacity-matched point tokens | 14 | 2 | 0.00418 |
| Local relation tokens, no flow loss | 15 | 2 | 0.00235 |
| Local relation tokens + flow loss | 17 | 3 | 0.00258 |
| Local flow tokens + zero global token | 16 | 2 | 0.00131 |

The full model has lower final translation and rotation than the raw point
baseline on 78.1% of the paired states.  Mean paired improvements are 0.670 cm
and 5.120 degrees.

## Current conclusion

The evidence supports the architecture direction, but not every originally
proposed component equally:

1. The main supported mechanism is the explicit, task-aligned relative SE(3)
   token.  It resolves rotation/direction ambiguity that raw points and local
   correspondence tokens often leave unresolved.
2. Local point-relation tokens are useful for structured conditioning, but do
   not independently outperform a capacity-matched point policy in closed loop.
3. Future-flow supervision is a geometrically meaningful auxiliary loss, not a
   demonstrated primary performance source.  It should be presented as a
   regularizer unless later task/seed evidence shows a stable recovery gain.
4. The method still has a recoverability boundary.  Some failures occur under
   larger perturbations or grasp slip, and a few states are baseline-only wins.

This is strong enough to justify completing policy seeds 1 and 2 and then
moving to a new task/category gate.  It is not yet sufficient for a T-ASE claim:
cross-seed closed-loop replication and cross-task evidence remain mandatory.

## Interrupted-evaluation robustness

`evaluate_pose_correction_closedloop.py` now supports `--resume-existing`.
It validates the frozen protocol and manifest identity, skips completed
`(variant, episode)` pairs, and atomically continues an interrupted result.
Checkpoint and estimator provenance are recorded for new/resumed outputs.

At the time of this note, external seed 1 has 1/32 states saved and seed 2 has
not started.  No partial result is used as cross-seed evidence.
