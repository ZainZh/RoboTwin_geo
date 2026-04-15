# NDF Interact Hybrid Design

## Goal

Add an `objpc`-only NDF interaction feature path for DP3 training and evaluation.

The new path extends the current `ndf_pointwise_hybrid` structure:

- keep raw merged object point cloud: `point_cloud`
- keep self NDF branches: `ndf_point_cloud_A`, `ndf_point_cloud_B`
- add cross-object NDF interaction branches:
  - `ndf_interact_point_cloud_A_from_B`
  - `ndf_interact_point_cloud_B_from_A`

This change must not alter existing `ndf_pointwise_hybrid`, `semantic_pointwise_hybrid`, or any actorseg path.

## Behavior

For each placeholder pair `A`, `B`:

- self branch:
  - unchanged
  - computed by querying object `A` points with model `A`, object `B` points with model `B`
- interact branch:
  - `A_from_B`: query sampled world points from object `A`, but compute local NDF features using object `B` as support with model `B`
  - `B_from_A`: query sampled world points from object `B`, but compute local NDF features using object `A` as support with model `A`

Output shape for each interact branch:

- `[N, 3 + ndf_feat_dim]`
- first 3 dims remain the query object's world xyz
- remaining dims are cross-object local NDF features from the support object's model

## Partial Availability

Interaction branches are created only when the support-side NDF model exists.

Examples:

- only `A` model exists:
  - self: `ndf_point_cloud_A`
  - interact: `ndf_interact_point_cloud_B_from_A`
  - do not create `ndf_point_cloud_B`
  - do not create `ndf_interact_point_cloud_A_from_B`

- both `A` and `B` models exist:
  - create all four NDF branches

## Integration Strategy

Create a new independent path rather than changing the existing hybrid path.

New path naming:

- preprocess/train/eval suffix: `-objpc-ndf-pointwise-hybrid-interact`
- Hydra config: `robot_dp3_ndf_pointwise_hybrid_interact.yaml`
- task config: `demo_task_ndf_pointwise_hybrid_interact.yaml`
- shell entrypoints:
  - `process_data_ndf_pointwise_hybrid_interact.sh`
  - `train_ndf_pointwise_hybrid_interact.sh`
  - `eval_ndf_pointwise_hybrid_interact.sh`

This keeps old checkpoints and experiment naming stable.

## Implementation

### Feature Computation

Add a new utility function in `policy/DP3/scripts/ndf_feature_utils.py`:

- `compute_ndf_interact_pointwise_cloud(model, support_object_point_cloud, query_object_point_cloud, device, target_num_points)`

Logic:

1. take support object xyz
2. take query object xyz
3. sample query world xyz to `target_num_points`
4. normalize support xyz to support-centered normalized frame
5. transform sampled query world xyz into that same support-normalized frame
6. run support model latent on support cloud
7. run local feature query on transformed query points
8. return `[query_world_xyz | queried_local_features]`

This mirrors the query/support logic used by the interaction demo path.

### Offline Preprocess

Based on current `process_data_ndf_pointwise.py`:

- keep self branches unchanged
- additionally build interact branches per frame when the corresponding support-side model exists
- save them as extra replay buffer arrays

### Runtime Eval

Based on current `deploy_policy.py` NDF hybrid path:

- keep self branches unchanged
- after placeholder point clouds are extracted, compute interact branches online using the same query/support rule

### Shape Meta

The new config adds optional point-cloud observations:

- `ndf_interact_point_cloud_A_from_B`
- `ndf_interact_point_cloud_B_from_A`

Each branch uses shape `[ndf_point_num, 3 + ndf_feat_dim]`.

## Testing

Add tests for:

1. feature utility:
   - interact function returns `[N, 3 + feat_dim]`
   - world xyz prefix matches query object sample coordinates

2. preprocess argv:
   - wrapper adds new output suffix

3. deploy policy:
   - when only `A` model exists, only `B_from_A` interact branch appears
   - when both models exist, both interact branches appear

4. shell interface:
   - train/eval scripts parse arguments and pass interact config names correctly

## Risks

- More point-cloud encoder branches increase memory and runtime
- Cross-object features may over-dominate grasp behavior if point counts are too large
- Because support normalization is object-centric, very large query distances may produce weak or unstable local features

## Recommendation

Keep this path separate and start with:

- `ndf_point_num = 128` or `256`

Do not start with large interact branch point counts. The goal is to test whether interaction information helps, not to maximize branch capacity immediately.
