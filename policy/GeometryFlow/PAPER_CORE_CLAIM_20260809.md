# TAGRT-v1: paper core claim and backbone boundary

## One-sentence claim

Category-level geometric representations improve learned short-horizon
manipulation when they are converted into mutually consistent, task-aligned
local and global relation tokens and fused through reliability-preserving
adapters; raw descriptors or direct token concatenation are not reliably
useful.

## Problem

Modern manipulation policies can observe object point clouds, yet must still
infer which object parts define the task, the remaining current-to-goal
relation, and how that relation maps to actions.  A pooled NDF/UTONIA feature or
an unconstrained appended token is easy for a policy to ignore, may be
redundant with raw observations, and may introduce coordinate ambiguity or
train/test estimator shift.  Consequently, extra geometry sometimes helps and
sometimes hurts even when the underlying geometric encoder is accurate.

The paper studies a deliberately narrow question: how should a learned policy
use an object representation, rather than whether a geometric controller can
solve the task?

## Method

TAGRT converts segmented current and target camera point clouds into a frozen
representation contract:

1. A category-specific equivariant estimator (currently NDF) recovers a
   functional frame and confidence from the operated object.  The target frame
   is recovered from its camera cloud.
2. Confidence gating, SO(3) medoid aggregation, and causal temporal branch
   stabilization suppress equivalent-axis flips.  This module stabilizes the
   representation; it does not predict task stage or actions.
3. Each point receives local functional coordinates/task-flow features, while
   the same pair of frames produces an explicit global remaining SE(3) token.
   The two levels are mutually consistent.
4. A zero-initialized local adapter and zero-start global gate condition the
   learned policy without corrupting its original point representation when
   geometry is absent or initially untrusted.
5. Robot proprioception and conditioned point tokens enter a learned
   action-chunk policy, which predicts six 14-D bimanual Cartesian action
   deltas.  There is no planner, IK action prior, residual controller, stage
   predictor, grasp token, or simulator pose at inference.

Training uses object-disjoint splits, leave-one-object-out cross-fitted geometry
for policy-training rows, and relation-identifying multi-goal examples.  For a
fixed current observation, different camera goals change the relation tokens
and action labels together, forcing the policy to identify and use the desired
relation.

## Main methodological contributions

1. **Task-aligned two-level geometry tokens:** per-point functional relation
   features and an explicit remaining SE(3) token derived from one consistent
   frame pair.
2. **Reliability-preserving policy fusion:** no-op-at-initialization local and
   global geometry pathways that retain a strong baseline instead of directly
   corrupting point tokens.
3. **Identifiable, leakage-controlled learning:** multi-goal conditioning and
   cross-fitted category-estimator outputs remove goal redundancy and
   in-sample representation leakage.
4. **Causal evaluation of geometry use:** correct/zero/shuffled tokens and
   same-checkpoint interventions under identical simulator snapshots separate
   genuine geometry use from capacity, regularization, and scene effects.

## Evidence boundary

The frozen controlled Transformer establishes the mechanism on blind shoe and
handled-mug identities.  Across three policy seeds, correct geometry improves
closed-loop success over the same checkpoint with geometry zeroed by 30
percentage points on the common shoe scenes.  On handled mugs the descriptive
pooled result is 19/36 versus 8/36, with 7 scene-level wins, 0 losses, and 5
ties.  These results support the representation-and-fusion claim, but larger
blind scene sets and a mainstream-backbone replication remain necessary for a
strong T-ASE submission.

## Mainstream-backbone interface

Replacing the small controlled Transformer with DP3 or another mainstream
learned policy does not change the method if the following interface remains
fixed:

```text
camera A/B point clouds
  -> category representation and confidence
  -> local functional relation tokens + global remaining SE(3)
  -> zero-start geometry adapters on the chosen policy backbone
  -> learned action distribution/chunk
```

The geometry estimator, two-level token definition, multi-goal training data,
object splits, causal controls, and camera-only inference stay unchanged.  Only
the consumer of the tokens changes.  For DP3, the local adapter augments its
point features and the global token conditions the diffusion denoiser through
a zero-start projection/gate.  For a VLA, the same tokens enter spatial
cross-attention or gated conditioning while language remains an orthogonal
input.

Backbone replication therefore strengthens rather than redirects the story:
the current Transformer is the controlled mechanism study, and DP3 is the
first external-validity test.  A positive DP3 result supports a
backbone-general geometry-conditioning framework.  A negative result would
limit the claim to the present action-chunk architecture and must be reported;
it would not justify changing the frozen geometry representation after seeing
test outcomes.

DP3 should be the next backbone because it is already integrated in this
repository, natively consumes 3D observations, and permits an interface-matched
geometry-off baseline.  Large VLA backbones should follow only after the DP3
test and after collecting task/language diversity sufficient for a meaningful
VLA claim.
