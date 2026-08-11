# Continuous TAGRT-DP3 main-line protocol (2026-08-11)

## Frozen method

```text
current/goal segmented camera point clouds
  -> frozen NDF functional frames and correspondence confidence
  -> local 128 x 9 anchor-flow tokens + global 9D remaining SE(3)
  -> confidence-weighted zero-start local/global DP3 adapters
  -> one DP3 diffusion decoder
  -> six 14D bimanual EEF action deltas
```

`tagrt_anchor_flow` contains, for every current-object anchor, its offset to
the left EEF, its offset to the right EEF, and its remaining rigid task flow.
`tagrt_global` is derived from the same current/goal frame pair. Both are
present at every observation. Geometry confidence multiplies their adapter
features continuously in `[0, 1]`.

This configuration contains no high/low-error recovery head, hard SE(3)
threshold, task-stage predictor, analytic controller, simulator-pose policy
input, or nominal-plus-residual action replacement. The learned binary gripper
decoder uses the same continuously conditioned observation context.

## Pilot comparison

Train one seed on shoe IDs 2/3/4/7/8/9, select only on IDs 1/6, and evaluate
once on held-out IDs 0/5. The first causal check uses one checkpoint with
identical diffusion noise under three inference interventions:

1. correct anchor-flow and global SE(3);
2. both geometry levels zeroed;
3. episode-level shuffled geometry from another task goal.

The immediate promotion requirement is that correct geometry improves both
active-arm endpoint translation and rotation over zero and shuffled geometry
without reducing active-gripper accuracy. Only a promoted checkpoint proceeds
to matched closed-loop simulation. Diagnostic recovery datasets and 18-degree
gates are excluded from this experiment.

## Seed-0 pilot result

The compact 6.74M-parameter DP3 was trained from scratch for 120 epochs. Its
base encoder consumed only current/goal XYZ clouds and 20D proprioception; the
two geometry adapters consumed anchor flow and global SE(3). Checkpoints
60/80/100/120 were compared only on validation shoes 1/6. Epoch 120 was selected
because it had the best active-arm MAE, endpoint rotation, and gripper accuracy;
its translation was within 0.007 cm of the best checkpoint.

Validation shoes 1/6, three matched diffusion samples per row:

| condition | active-arm MAE | endpoint translation | endpoint rotation | gripper accuracy |
|---|---:|---:|---:|---:|
| correct TAGRT | **0.01110** | **1.641 cm** | **4.680 deg** | **96.26%** |
| zero geometry | 0.01190 | 1.832 cm | 5.983 deg | 96.21% |
| shuffled geometry | 0.01200 | 1.780 cm | 6.575 deg | 96.20% |

After selection, the frozen epoch-120 checkpoint was evaluated once on
held-out shoes 0/5:

| condition | active-arm MAE | endpoint translation | endpoint rotation | gripper accuracy |
|---|---:|---:|---:|---:|
| correct TAGRT | **0.00968** | **1.518 cm** | **4.979 deg** | **97.98%** |
| zero geometry | 0.01168 | 1.716 cm | 5.376 deg | 96.14% |
| shuffled geometry | 0.01102 | 1.689 cm | 5.189 deg | 96.75% |

Correct geometry is better in every aggregate metric under both interventions
and on both object-disjoint splits. The result passes the offline causal gate.
It supports continuous DP3 use of the representation, not yet closed-loop task
success. The next and only promoted experiment is matched post-grasp placement
on the existing development seeds with this checkpoint frozen.

Artifacts:

- checkpoint: `/shared2/sz/TAGRT-v1/models/dp3/place_shoe_geometry_marker-tagrt_continuous_anchorflow_seed0-8880_0/120.ckpt`;
- validation records: `/shared2/sz/TAGRT-v1/results/continuous_tagrt_seed0/val_epoch120.json`;
- held-out records: `/shared2/sz/TAGRT-v1/results/continuous_tagrt_seed0/test_epoch120.json`.
