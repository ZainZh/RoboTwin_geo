# DP3 backbone integration: TAGRT-v1 pilot (2026-08-09)

## Question

Can the frozen two-level Task-Aligned Geometric Relation Token (TAGRT-v1)
improve a mainstream diffusion-policy backbone without changing the paper's
geometry-conditioned learning claim?

## Interface

The DP3 policy consumes two role-factored camera point clouds, 20-D bimanual
end-effector proprioception, and six-step 14-D Cartesian action chunks. TAGRT
adds two observation-derived conditions:

1. a 256-by-4 local stream containing task-aligned functional coordinates for
   A and B plus an A/B role bit;
2. a 9-D global remaining-SE(3) stream containing translation and two rotation
   columns.

The local and global streams use separate zero-initialized residual adapters.
Both are multiplied by the frozen frame estimator's confidence. There is no
planner, inverse kinematics, simulator pose input, residual controller, stage
head, or grasp token in the policy.

The training split contains shoe IDs 2/3/4/7/8/9, validation contains 1/6, and
blind evaluation contains 0/5. The model has 6.66 M parameters. For each seed,
one correct-geometry checkpoint is evaluated with identical diffusion noise
under correct, zero, and within-split shuffled geometry.

## Three-seed blind action prediction

Values are mean plus/minus sample standard deviation over seeds 0/1/2.

| condition | raw action MAE | active-arm MAE | endpoint translation | endpoint rotation |
|---|---:|---:|---:|---:|
| correct TAGRT | 0.01418 +/- 0.00173 | 0.02271 +/- 0.00318 | 4.051 +/- 0.547 cm | 25.191 +/- 1.435 deg |
| zero geometry | 0.02049 +/- 0.00147 | 0.03469 +/- 0.00345 | 5.223 +/- 0.338 cm | 48.822 +/- 0.274 deg |
| shuffled geometry | 0.02341 +/- 0.00149 | 0.04048 +/- 0.00257 | 5.670 +/- 0.562 cm | 64.542 +/- 2.316 deg |

Correct TAGRT improves over the same checkpoint with zero geometry by 30.9%
in raw action MAE, 34.6% in active-arm MAE, 22.7% in endpoint translation, and
48.4% in endpoint rotation. Correct geometry beats both zero and shuffled
geometry on every metric for all three training seeds.

## What this establishes

This passes the offline causal mechanism gate for DP3. The earlier pooled-flow
DP3 adapter was weak and seed-unstable even with oracle features; the new result
shows that DP3 is not intrinsically incompatible with geometry. The important
changes are role-factored A/B inputs, per-point functional coordinates, an
explicit global remaining transform, and dedicated safe fusion.

This result does not yet establish closed-loop success or end-to-end full-task
success. The next evidence ladder is:

1. deploy this checkpoint in the controlled post-grasp closed-loop benchmark;
2. train one DP3 policy over the complete observe-grasp-transport-place-release
   trajectory without an explicit stage label;
3. report grasp success, conditional placement success, stable-release success,
   and full-task success under matched correct/zero/shuffled interventions;
4. reproduce the effect on a natural functional-geometry task.

The post-grasp benchmark remains necessary as the causal mechanism experiment;
the full-task benchmark is necessary as the external-validity experiment.
