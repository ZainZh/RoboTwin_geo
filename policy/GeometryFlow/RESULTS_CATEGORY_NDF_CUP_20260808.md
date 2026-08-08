# Category-specific NDF cup result — 2026-08-08

## Locked research route

This experiment stays inside the agreed learned-policy route:

```text
segmented current camera cloud of operated object A
  -> category-specific NDF
  -> task-aligned signed functional axis
  -> per-anchor [axis, dot(axis, B-A), cross(axis, B-A)] relation token
  -> A/B point tokens + robot proprioception
  -> Transformer
  -> learned 6-step, 14D EEF action chunk
```

There is no analytic action prior, IK nominal trajectory, residual planner,
stage prediction, or UTONIA branch in this experiment. Simulator object axes
are used only as task-alignment supervision during training; NDF inference and
policy inference use the current camera point cloud.

## Geometry-disjoint NDF pretraining

- Category: `assets/objects/021_cup`
- NDF pretraining assets: IDs `4,6,7,8,9,10,11,12`
- Manipulation-task assets: IDs `0,1,2,3,5`
- Therefore no manipulation test geometry occurs in NDF occupancy pretraining.
- NDF output includes a 16-vector equivariant head compatible with the existing
  shoe checkpoint format.
- Occupancy training stopped at epoch 107; best validation occupancy loss was
  `0.170778` at epoch 27.
- Checkpoint (not committed):
  `outputs/geometry_flow/category_ndf/training/cup_021_geom_disjoint_vector16_v1/checkpoints/model_best.pth`

The cup functional direction encoded by the task assets is local `-Y`, not
local `Z`. This is read from each asset's functional transform and is the axis
used for task alignment.

## Camera-cloud functional-axis result

Fold-0 task-alignment test uses held-out manipulation instances `0` and `5`
(`450` relation-phase observations).

| Axis estimator | Mean error | Median error | Within 15 deg |
|---|---:|---:|---:|
| Raw category NDF vector | 116.205 deg | 114.815 deg | 0.0% |
| Constant train-set mean axis | 7.549 deg | 4.987 deg | 85.1% |
| Task-aligned category NDF | **3.345 deg** | **2.127 deg** | **99.1%** |

Interpretation: the raw vector head is not task aligned. Full task alignment is
necessary, after which the camera-only NDF axis beats the constant-axis prior
on unseen manipulation objects.

Result directory (not committed):
`outputs/geometry_flow/container_plate_geometry_v2/cup_ndf_functional_neg_y_full_fold0_seed0`

## Policy protocol

- Dataset: `recovery_100x_taskflow.npz`
- 100 balanced demonstrations, 1139 relation-phase samples.
- Fold 0: train object IDs `2,3`, validation ID `1`, test IDs `0,5`.
- Same Transformer, action loss, object split, point clouds, actions, and
  proprioception for clean and zero-axis conditions.
- Source-axis relation branch pre-tanh gate initialization is fixed to `0.3`.
- Zero control keeps the same branch and parameter count but supplies an exact
  zero axis.
- Early stopping uses validation normalized action MSE.

The original zero gate was falsified as a fusion design: it converged to an
effective gate of about `0.00028`, making clean, zero, and shuffled conditions
numerically identical. A small positive initialization prevents gradient
starvation while preserving an exact-zero capacity control.

## Independently trained clean versus zero control

Endpoint test errors on the two unseen fold-0 objects:

| Seed | Clean NDF trans. | Zero trans. | Clean NDF rot. | Zero rot. |
|---:|---:|---:|---:|---:|
| 0 | **0.4668 cm** | 0.4883 cm | **3.2513 deg** | 3.6214 deg |
| 1 | **0.5329 cm** | 0.5388 cm | **3.6331 deg** | 3.6537 deg |
| 2 | 0.5391 cm | **0.5269 cm** | 3.4871 deg | **3.4838 deg** |
| Mean | **0.5129 cm** | 0.5180 cm | **3.4572 deg** | 3.5863 deg |

Mean improvement is `0.97%` in endpoint translation and `3.60%` in endpoint
rotation. Clean wins 2/3 seeds on each endpoint metric. This is positive but
not yet a strong paper-level multi-task result.

## Frozen-checkpoint causal intervention

Each clean-trained checkpoint is frozen. Test point clouds, actions,
proprioception, normalization, rows, and model weights remain identical. Only
`source_axis3` is replaced at inference.

Three-seed means:

| Frozen input | Endpoint trans. | Endpoint rot. | Clean improvement |
|---|---:|---:|---:|
| Correct task-aligned NDF axis | **0.5129 cm** | **3.4572 deg** | — |
| Zero axis | 0.5255 cm | 3.6177 deg | 2.39% / 4.44% |
| Sign-flipped NDF axis | 0.5331 cm | 3.7577 deg | 3.78% / 8.00% |
| Isotropic random episode axis | 0.5212 cm | 3.5482 deg | 1.59% / 2.57% |
| Different-episode axis | 0.5078 cm | 3.4776 deg | -1.01% / 0.59% |

Correct NDF beats zero on both metrics in all three frozen seeds (seed 1
translation differs by only `0.00002 cm`). Correct NDF beats sign-flipped and
isotropic axes in 3/3 seeds for rotation and 2/3 seeds for translation.

The different-episode corruption is a weak control here: correct and shuffled
axes differ by only 10.8 degrees on average because the task's cup axes are
concentrated around one world direction. It is retained for completeness but
must not be used as the primary causal claim. Zero, sign-flip, and isotropic
interventions are the informative controls.

Counterfactual result files (not committed):

- `cup_ndf_axis_gate0p3_fold0_seed0_counterfactual.json`
- `cup_ndf_axis_gate0p3_fold0_seed1_counterfactual.json`
- `cup_ndf_axis_gate0p3_fold0_seed2_counterfactual.json`

## Current conclusion

The cup result answers the immediate doubt:

1. A category-specific cup NDF can recover a correct task functional axis from
   held-out camera point clouds after task alignment (`3.345 deg`).
2. A task-aligned relation-token policy can use that axis causally across three
   seeds: zero, random, and sign-flipped interventions degrade frozen-policy
   orientation, and usually translation.
3. The independently trained policy currently yields modest mean gains over a
   capacity-matched zero-axis policy, not yet a publication-complete gain.

Therefore the NDF route is not falsified. The remaining bottleneck is evidence
scale and task identifiability, not inability of cup NDF to produce the axis.

## Required next evidence

1. Repeat the fixed `0.3` setting on more object-disjoint cup splits; do not tune
   the gate again on those splits.
2. Run closed-loop paired clean/zero/random-axis evaluation on identical scene
   seeds, reporting task success and final pose error.
3. Add one second short-horizon category with a functionally important axis and
   its own category-specific NDF.
4. Add a from-scratch equivariant encoder control to separate NDF pretraining
   value from the value of task-aligned axis supervision itself.

Until these are complete, the justified claim is a validated mechanism and a
promising learned-policy route, not a finished T-ASE result.
