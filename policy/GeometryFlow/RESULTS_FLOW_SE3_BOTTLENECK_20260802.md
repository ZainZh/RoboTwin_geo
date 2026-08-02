# Flow-SE(3) bottleneck learned-policy gate

Date: 2026-08-02

## Question

Can object geometry improve a learned manipulation policy when geometry is a
structured policy input, rather than an analytic controller that overrides the
policy action?

The deployment rule in this gate is strict: every Cartesian arm delta is
predicted by the learned policy.  A uniquely closed gripper selects the active
arm channel, and camera-derived flow progress gates release.  No analytic
SE(3) controller supplies an arm action.  The `oracle_endpoint` run below is a
diagnostic geometry input only; it still uses the same learned action decoder.

## Architecture tested

```text
A/B camera point clouds
  -> replaceable geometry backend (PCA now; NDF/UTONIA compatible)
  -> sparse rigid object task flow
       -> per-anchor point-flow tokens
       -> per-step Kabsch SE(3) tokens (centroid delta + rotation 6D)
  -> proprioception token
  -> Transformer action-query decoder
  -> six-step Cartesian action chunk
```

The action decoder has no raw-point-cloud bypass.  The `zero_flow` control has
the identical parameterization but receives zero point-flow, zero SE(3), and
zero progress inputs.  The Kabsch tokens are computed from the supplied flow
and current camera anchors; they do not use simulator object pose.

Training uses fold-0 object-disjoint splits and only relation/placement rows:
shoes `2,3,4,7,8,9` train, `1,6` validate, and `0,5` test.  There are 2,796
nominal and 1,095 on-policy recovery rows in the train split, so recovery is
already 28.1% of the placement data.  The configured 70:30 loss mix changes
its weight only slightly.

## Offline gate: held-out shoes

One policy seed, 796 held-out nominal placement samples:

| Model/input | Endpoint translation | Endpoint rotation | Active-arm selection |
|---|---:|---:|---:|
| Point-flow bottleneck, zero flow | 2.251 cm | 6.457 deg | 95.6% |
| Point-flow bottleneck, PCA flow | 1.826 cm | 7.851 deg | 88.7% |
| Point + SE(3) bottleneck, zero flow | 2.255 cm | 6.826 deg | 90.8% |
| Point + SE(3) bottleneck, PCA flow | **1.473 cm** | **3.758 deg** | 74.9% |
| Frozen SE(3) policy, oracle flow | **1.194 cm** | **3.267 deg** | 72.2% |
| Frozen SE(3) policy, shuffled PCA flow | 2.553 cm | 7.993 deg | 47.9% |

Relative to the matched zero-flow model, PCA improves endpoint translation by
34.7% and rotation by 44.9%.  A marginally valid but wrong rigid flow destroys
both gains and performs worse than zero flow.  Oracle flow further improves
both metrics.  This is a positive causal representation result, not merely an
extra-capacity effect.

The explicit per-step SE(3) tokens are necessary in this pilot: point-flow
tokens alone recover translation but make rotation worse.  The frozen
oracle/PCA comparison also localizes most of the remaining rotation error to
the learned action decoder/data, while leaving a smaller perception gap.

An auxiliary arm-routing loss with weight 0.1 was rejected.  It changed
active-arm selection only from 74.9% to 75.5% while degrading PCA endpoint
errors to 1.774 cm / 4.916 deg.  The normalized-action magnitude used by this
auxiliary is not a suitable physical routing target.

## Closed-loop survival gate

All runs use the identical held-out seed `100002`, shoe 5, 120 learned arm
actions, six actions per predicted chunk, receding-horizon camera refresh, and
the same closed-gripper arm routing/release gate.

| Learned policy input | Success | Min translation | Min rotation | Final/max flow progress |
|---|---:|---:|---:|---:|
| Zero flow | 0/1 | **7.11 cm** | 81.87 deg | 0.733 |
| PCA camera flow | 0/1 | 9.25 cm | 66.54 deg | 1.000 |
| Oracle endpoint flow diagnostic | 0/1 | 8.53 cm | **58.15 deg** | 1.000 |

PCA and oracle geometry improve rotation relative to zero flow, but neither
approaches the success region and translation remains poor.  Both geometry
routes report progress 1.0 despite large true pose error, after which the
release gate opens.  Most importantly, oracle endpoint geometry still fails;
the current failure therefore cannot be assigned primarily to the PCA/NDF/
UTONIA backend.

## Decision

The representation route passes its offline causal gate but the current
behavior-cloning sequential policy fails its closed-loop gate.  It is valid to
claim that a rigid point-flow + explicit SE(3) bottleneck improves held-out
object action prediction.  It is not valid to claim learned task success or to
start paper-scale task evaluation from this checkpoint.

The next experiment should keep this exact representation fixed and replace
the data/policy loop:

1. collect iterative DAgger states from the new SE(3)-bottleneck policy, not
   the older flow model;
2. query expert recovery at several depths (roughly 1, 2, 4, 8, and 12 chunks)
   and include overshoot/post-goal states;
3. decode a single observable active arm in a 7D action space rather than
   learning two redundant arm trajectories and masking one afterward;
4. train a progress/stop/release head on real rollout states, because nearest
   flow-step progress is overconfident under pose error;
5. require an oracle-flow learned-policy closed-loop gain before comparing PCA,
   UTONIA, NDF, or collecting more nominal demonstrations.

The paper candidate remains an **SE(3)-consistent multi-resolution geometric
flow bottleneck for recovery-aware learned manipulation**, with zero-flow,
shuffled-flow, and oracle-flow causal controls.  Closed-loop recovery learning,
not another encoder sweep, is the next decisive gate.

## Artifacts

- `flow_se3_bottleneck_recovery30_fold0_seed0/summary_seed0.json`
- `flow_se3_bottleneck_recovery30_fold0_seed0/causal_flow_eval.json`
- learned checkpoint: `flow_se3_bottleneck_recovery30_fold0_seed0/fold0_flow_seed0.pt`
- rejected routing-loss run: `flow_se3_bottleneck_route010_recovery30_fold0_seed0/`
- closed-loop evaluation settings:
  - `flow-se3-bottleneck-chunk6-fold0-seed100002`
  - `zero-se3-bottleneck-chunk6-fold0-seed100002`
  - `oracle-se3-bottleneck-chunk6-fold0-seed100002`

