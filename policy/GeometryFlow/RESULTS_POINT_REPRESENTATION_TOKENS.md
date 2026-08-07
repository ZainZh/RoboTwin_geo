# Point-representation token benchmark

Date: 2026-07-31

## Question

The experiment tests whether the earlier negative NDF/UTONIA results were
caused by global pooling. It compares raw XYZ, RGB, PCA-local coordinates,
NDF, the NDF equivariant vector, and UTONIA under one parameter-matched grasp
architecture.

This is a grasp-relation prediction screen. It does not claim closed-loop
manipulation success.

## Controlled protocol

- 50 successful shoe demonstrations and 10 shoe identities.
- Five object-disjoint folds. Every shoe is held out exactly once.
- Two policy seeds for the primary comparison, giving 100 held-out
  episode-seed predictions per condition.
- The same 256 camera points, grasp labels, active-arm input, 64-D descriptor
  interface, optimizer, early stopping rule, and 657,284 trainable parameters
  are used by every primary condition.
- NDF uses its 256 invariant scalar channels. Its final equivariant triplet is
  tested separately.
- UTONIA uses the official 1,386-D five-stage descriptor. High-dimensional
  descriptors are standardized and PCA-whitened to 64 dimensions using
  training shoes only.
- Seven labeled virtual-gripper keypoints query either one global mean/max
  object token or the global token plus all 256 aligned point tokens. A
  differentiable Kabsch fit produces a proper SE(3) grasp.
- The `point_shuffle` controls permute frozen descriptors inside each object
  while keeping XYZ and the descriptor multiset unchanged. They test whether
  local point/feature correspondence is causally useful.

The grasp cache and representation cache align exactly in episode ID, shoe ID,
arm, point order, XYZ, and NDF values; maximum XYZ/NDF discrepancy is zero.

## Primary two-seed, five-fold result

| Condition | Translation (cm) | Rotation (deg) | Keypoint RMSE (cm) | 3 cm / 30 deg |
| --- | ---: | ---: | ---: | ---: |
| global raw XYZ | **1.737** | 42.43 | 1.732 | 48% |
| token raw XYZ | 1.877 | **32.82** | **1.570** | **63%** |
| global PCA local | 2.041 | 48.47 | 1.872 | 30% |
| token PCA local | 1.900 | 44.88 | 1.794 | 44% |
| global NDF scalar | 2.308 | 55.23 | 2.113 | 25% |
| token NDF scalar | 1.930 | 53.85 | 2.238 | 27% |
| global NDF vector | 1.888 | 41.39 | 1.707 | 44% |
| token NDF vector | 2.246 | 42.52 | 1.981 | 46% |
| global UTONIA all stages | 2.166 | 57.24 | 2.176 | 23% |
| token UTONIA all stages | 2.153 | 54.80 | 2.207 | 31% |

Raw point tokens change rotation by `-9.60 deg`, keypoint RMSE by `-0.162 cm`,
and 3 cm/30 deg accuracy by `+15` points relative to global raw pooling.
Both seeds agree on the rotation sign (`-9.14` and `-10.07 deg`). The
shoe-cluster bootstrap intervals still cross zero: rotation
`[-21.89,+3.84] deg`, keypoint `[-0.349,+0.061] cm`, and success
`[-4.0,+31.6]` points. The effect is therefore promising but heterogeneous
across shoe identities, not yet a stable paper claim.

Frozen descriptors do not improve the token baseline:

| Token representation minus raw token | Translation (cm) | Rotation (deg) | Keypoint (cm) | Success points |
| --- | ---: | ---: | ---: | ---: |
| PCA local | +0.023 | +12.05 | +0.224 | -19 |
| NDF scalar | +0.054 | +21.03 | +0.668 | -36 |
| NDF vector | +0.370 | +9.70 | +0.411 | -17 |
| UTONIA all stages | +0.277 | +21.97 | +0.637 | -32 |

The rotation penalty relative to raw token has a shoe-cluster 95% interval of
`[+8.46,+30.05] deg` for NDF and `[+4.80,+35.05] deg` for UTONIA. NDF also
reduces success by 36 points (`[-51.3,-16.2]`), while UTONIA changes success
by -32 points (`[-55.6,+1.0]`).

Local descriptor alignment is not established. Clean NDF versus point-shuffled
NDF changes translation/rotation by `-0.031 cm/-2.58 deg`, with intervals
`[-0.224,+0.088] cm` and `[-6.86,+0.42] deg`. Clean UTONIA changes them by
`+0.132 cm/+2.75 deg`, with intervals `[-0.121,+0.395] cm` and
`[-2.74,+7.28] deg`. Seed signs are not consistently favorable.

## UTONIA layer screen

One additional five-fold seed separates the official UTONIA stages. This was
an exploratory screen over five choices and is not an independent
confirmation.

| Stage | Global T/R | Token T/R | Token - global rotation | Success global -> token |
| --- | ---: | ---: | ---: | ---: |
| s0, 54-D | 1.853 cm / 54.59 deg | 2.056 cm / 39.92 deg | **-14.67 deg** | 24% -> 48% |
| s1, 108-D | 1.672 / 43.34 | 2.890 / 53.65 | +10.32 deg | 50% -> 22% |
| s2, 216-D | 1.931 / 47.17 | 2.475 / 46.75 | -0.42 deg | 34% -> 42% |
| s3, 432-D | 1.578 / 60.52 | 2.452 / 54.28 | -6.24 deg | 22% -> 26% |
| s4, 576-D | 2.042 / 58.59 | 2.227 / 55.34 | -3.25 deg | 30% -> 22% |

For s0, the shoe-cluster interval for the token/global rotation change is
`[-21.35,-6.99] deg` and success changes by +24 points
(`[+5.3,+41.5]`). Translation changes by +0.203 cm and its interval crosses
zero. The absolute s0 token result is still worse than the two-seed raw-token
result in rotation (39.92 versus 32.82 deg) and success (48% versus 63%).
Because s0 was selected after testing five layers, it is a fine-tuning
candidate rather than evidence that frozen UTONIA beats raw geometry.

## Zero-gated residual diagnostic

Early concatenation may let 64 descriptor channels swamp three exact XYZ
channels. A second architecture encodes XYZ separately and adds descriptors
through a zero-initialized learned gate. Both seeds completed fold 0 before
the remaining GPU expansion was stopped by the server execution limit.

On this diagnostic fold, the gate largely recovers the raw-token operating
point, but no descriptor condition clearly beats raw token. Clean-versus-
shuffled effects are mixed across translation, rotation, and keypoint error.
These 16 held-out episode-seed predictions are useful architecture diagnostics
but are intentionally excluded from the five-fold conclusion.

## Decision

The experiment rejects two simple explanations:

1. **Global pooling was the only problem.** Raw XYZ benefits from local
   queries, but frozen NDF/UTONIA point tokens still do not beat raw tokens.
2. **Any large pretrained per-point encoder will help once tokenized.** Full
   UTONIA and NDF are substantially worse than raw token on orientation and
   grasp success, and their clean/shuffled controls do not establish useful
   pointwise correspondence.

The useful narrowed route is:

```text
camera XYZ point tokens
  -> task/gripper queries over local points
  -> explicit contact or gripper-object relation
  -> guaranteed-rigid SE(3) task flow
  -> flow-conditioned action decoder
```

Pretrained features should enter only through a conservative residual, and
only after task-specific part refinement. The best candidate is a shallow
UTONIA feature initialized from s0, not the concatenated five-stage field.
Refinement supervision can be obtained from demonstrations: contact points,
gripper swept volume, successful grasp-frame anchors, and cross-shoe
contrastive positives. A point-shuffled control must remain in every run.

The next gate should require a task-refined feature model to beat raw XYZ token
by at least 5 degrees rotation or 10 success points on both seeds, with a
shoe-cluster interval excluding zero. Only after that gate should the selected
token model be integrated into DP3 and evaluated in paired closed-loop
rollouts. If it fails, the paper contribution should be the structured
gripper-object relation/task-flow interface rather than a general frozen
feature encoder.

## Artifacts

- Training: `train_representation_token_grasp.py`
- Aggregation: `summarize_representation_token_grasp.py`
- Unit tests: `test_representation_token_grasp.py` and
  `test_summarize_representation_token_grasp.py`
- Primary machine-readable result:
  `outputs/geometry_flow/representation_token_grasp_crossfold_summary.json`
- UTONIA stages:
  `outputs/geometry_flow/representation_token_grasp_utonia_stages_summary.json`
- Gated fold-0 diagnostic:
  `outputs/geometry_flow/representation_token_grasp_gated_fold0_summary.json`
