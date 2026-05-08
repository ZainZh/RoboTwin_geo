# PartNext Hammer Eval Integration Design

## Summary

This design adds an eval-only path for replacing the default `020_hammer/base0` asset in `beat_block_hammer` with a PartNext hammer asset, while keeping the existing task success logic unchanged.

The integration is intentionally split into two stages:

1. An offline asset-preparation step converts one selected PartNext hammer into a minimal RobotWin-compatible asset package.
2. An eval-time task-config switch makes `beat_block_hammer` load that prepared asset instead of the default hammer.

The first version only targets simulation inference and evaluation. It does not change data collection, training, or baseline task definitions.

## Goals

- Allow `beat_block_hammer` to run eval with a custom PartNext hammer asset.
- Keep the existing `beat_block_hammer.check_success()` logic unchanged.
- Avoid modifying or overwriting the original `020_hammer` asset.
- Use `annotation.jsonl` as the primary source for identifying handle and head regions.
- Produce an explicit visualization of the selected contact point and functional point on the full hammer mesh before eval use.

## Non-Goals

- No training-data regeneration for the custom hammer in the first version.
- No data-collection or scripted-demo support as a release requirement, even though the generated metadata should remain compatible with future scripted use.
- No runtime parsing of the full PartNext annotation file inside the task loop.
- No modification of policy checkpoints or policy-side inference logic.

## Constraints

- The new hammer must not be confused with the original `020_hammer`.
- The `beat_block_hammer` task must keep its current success check:
  - the hammer functional point is compared against the target block point
  - actor contact is still required
- The hammer head functional point must correspond to the striking side, not the claw / nail-puller side.
- The first version should be controlled through task config, not by changing eval CLI shape.

## Current System

### Task

`envs/beat_block_hammer.py` currently hardcodes:

- `modelname="020_hammer"`
- `model_id=0`

The task uses:

- `self.hammer.get_functional_point(0, "pose")` during success checking
- `self.grasp_actor(self.hammer, ...)` and `self.place_actor(..., functional_point_id=0, ...)` during scripted expert rollout

### Asset Loading

`envs/utils/create_actor.py` loads object assets from:

- `assets/objects/<modelname>/visual/base<id>.glb` or fallback mesh
- `assets/objects/<modelname>/collision/base<id>.glb` or fallback mesh
- `assets/objects/<modelname>/model_data<id>.json`

If `model_data*.json` is absent, semantic points such as `contact_points_pose` and `functional_matrix` are unavailable.

### Eval Path

`script/eval_policy.py` loads the selected `task_config/*.yml`, passes all task-config fields into `TASK_ENV.setup_demo(..., **args)`, and then instantiates the task. This allows the new behavior to be activated entirely through task config.

## Proposed Architecture

### Stage 1: Offline Asset Preparation

Add a standalone preparation script that builds a RobotWin-compatible hammer asset package from one selected PartNext hammer.

Responsibilities:

- scan PartNext hammer candidates
- choose one candidate hammer
- parse `annotation.jsonl`
- identify handle and head regions from part annotations
- estimate a usable global scale
- derive:
  - `contact_points_pose[0]`
  - `functional_matrix[0]`
- generate a preview visualization of the full hammer mesh with marked points
- write a new isolated RobotWin asset directory

This preparation step runs once per chosen candidate and produces a reusable on-disk asset package for future eval runs.

### Stage 2: Eval-Time Asset Override

Add a small task-side override in `beat_block_hammer` that checks a new task-config field, for example:

```yaml
custom_hammer_eval:
  enabled: true
  modelname: partnext_hammer_eval
  model_id: 0
```

Behavior:

- if `enabled` is false or absent, keep loading `020_hammer/base0`
- if `enabled` is true, load the prepared custom hammer asset instead

This keeps the eval command shape stable and makes A/B comparison a matter of switching task configs.

## Asset Selection

The PartNext hammer source directory contains many candidate meshes with widely varying scales and proportions. The first version will automatically choose one candidate rather than attempting batch import of the entire category.

Selection policy:

1. Enumerate available hammer meshes from `/home/zheng/Datasets/PartNext_mesh/Hammer`.
2. Compute each mesh bounding-box extents.
3. Compare candidate aspect ratio and dominant length to the existing RobotWin `020_hammer/base0`.
4. Reject obviously degenerate candidates, including extreme aspect ratios or implausibly large or tiny raw bounds.
5. Choose the closest remaining candidate by normalized size-and-shape similarity.

The selected candidate id and source path are recorded into `source_meta.json`.

## Annotation-First Part Extraction

The primary source of part semantics is `annotation.jsonl`, not geometry heuristics.

For the selected candidate:

1. Find the matching annotation row by `glb_dst` or `model_id`.
2. Parse `hierarchyList`.
3. Parse `masks`.
4. Build semantic face groups from annotation labels.

Region definitions:

- head region:
  - preferred labels: `Hammer Head`, `Head`
  - optional additive labels: `Nail Puller`
- handle region:
  - preferred labels: `Handle`, `Shaft`, `Grip`, `Handle End`

If labels are nested, all matching descendants are merged into the corresponding region.

Fallback rule:

- only if annotation lookup fails or required masks are empty, use geometry-based heuristics as a fallback
- fallback is not the primary path

## Scale Estimation

The prepared asset must load at a physically plausible size in RobotWin.

Scale estimation policy:

1. Read the original RobotWin hammer metadata from `assets/objects/020_hammer/model_data0.json`.
2. Compute the reference loaded hammer size as `reference_extents = extents * scale`.
3. Compute the raw candidate extents from the selected PartNext mesh.
4. Choose a single isotropic scale factor that makes the candidate dominant length match the reference dominant length.
5. Clamp the final scale to a conservative range to avoid pathological values.

The first version uses isotropic scale only. Non-uniform scale is explicitly out of scope.

## Contact Point Generation

`contact_points_pose[0]` should lie on the handle and remain compatible with future scripted usage.

Generation policy:

1. Extract the handle-region submesh.
2. Fit a handle principal axis from the handle geometry.
3. Select a representative point near the longitudinal center of the handle region.
4. Build a local pose matrix whose translation is that handle point.
5. Orient the contact frame so the gripper closing direction is consistent with grasping the handle rather than the head.

The first version does not optimize grasp quality beyond this semantic-handle placement. The goal is semantic correctness and compatibility, not perfect antipodal grasp synthesis.

## Functional Point Generation

`functional_matrix[0]` must identify the striking side of the hammer head and must not land on the claw or nail-puller side.

Generation policy:

1. Extract head-related annotated subregions.
2. Prefer the `Hammer Head` region over `Nail Puller` whenever both are present.
3. Use the handle principal axis to define a head-facing frame.
4. Within the chosen head region, identify the broader striking side by comparing transverse spread perpendicular to the handle axis.
5. Place the functional point at a representative location on that broader striking side.
6. Build a local pose matrix for `functional_matrix[0]`.

This preserves the existing task meaning: the point used in `check_success()` is still the hammer head used to strike the block.

## Output Asset Format

The preparation script writes a new isolated asset package:

```text
assets/objects/partnext_hammer_eval/
  visual/base0.glb
  collision/base0.glb
  model_data0.json
  points_info.json
  source_meta.json
  preview/
    overview.png
```

### `model_data0.json`

The first version includes at least:

- `scale`
- `contact_points_pose`
- `functional_matrix`
- `orientation_point`
- `target_pose`
- `center`
- `extents`
- `stable`

Descriptions are included so the asset remains understandable in the same style as existing RobotWin assets.

### `points_info.json`

The first version records simple semantic descriptions such as:

- contact point 0: hammer handle grasp point
- functional point 0: hammer striking head point

### `source_meta.json`

This records:

- selected PartNext candidate id
- source mesh path
- annotation row identifiers
- scale estimation summary
- generated point summary

## Task-Config Integration

Add a new task config for eval, for example:

- `task_config/demo_clean_3d_partnext_hammer_eval.yml`

It will inherit the standard `beat_block_hammer` eval settings but add:

```yaml
custom_hammer_eval:
  enabled: true
  modelname: partnext_hammer_eval
  model_id: 0
```

No existing task config is modified by default. The original clean and randomized configs continue to load `020_hammer`.

## Runtime Behavior

At runtime:

1. `script/eval_policy.py` reads the chosen task config.
2. The task config is passed into `beat_block_hammer.setup_demo(..., **args)`.
3. `beat_block_hammer.load_actors()` reads `custom_hammer_eval`.
4. If enabled, it loads `partnext_hammer_eval/base0`; otherwise it loads `020_hammer/base0`.
5. Policy inference proceeds unchanged.
6. Success checking proceeds unchanged.

This design isolates the custom-object substitution to the environment asset layer only.

## Validation Plan

### Asset Validation

- verify the new asset directory exists and contains all required files
- verify `model_data0.json` is parseable and includes the required semantic fields
- verify preview output shows:
  - full hammer mesh
  - contact point
  - functional point
  - handle axis

### Environment Validation

- instantiate `beat_block_hammer` with the custom eval task config
- confirm the hammer actor loads successfully
- confirm `get_contact_point(0, "pose")` returns a pose
- confirm `get_functional_point(0, "pose")` returns a pose
- confirm `check_success()` can run without asset-metadata failure

### Eval Validation

- run a short smoke eval with the custom task config
- confirm the task completes end-to-end without environment-side asset errors
- compare with the original hammer config to ensure the override path is isolated

## Risks and Mitigations

### Risk: annotation labels vary across assets

Mitigation:

- use a label-priority table instead of a single exact name
- record the matched labels in `source_meta.json`
- fall back to geometry only when annotation is missing or empty

### Risk: scale estimate is plausible but still poor for physics

Mitigation:

- anchor scale to the original RobotWin hammer loaded dimensions
- keep isotropic scale conservative
- include preview and metadata summary for manual review

### Risk: chosen functional point lands on the wrong head side

Mitigation:

- prefer explicit `Hammer Head` annotation over `Nail Puller`
- use width-based side selection relative to the handle axis
- visually inspect the generated point before enabling eval

### Risk: future work needs scripted demos

Mitigation:

- generate `contact_points_pose[0]` now, even though eval-only support is the current requirement

## Out of Scope for This Spec

- batch import of multiple PartNext hammers
- training or collection with the new hammer
- policy-side adaptation for custom object embeddings
- changing the reward or success logic for `beat_block_hammer`

## Acceptance Criteria

This design is complete when all of the following are true:

- a new isolated RobotWin asset package exists for one selected PartNext hammer
- the package includes valid scale, contact point, and functional point metadata
- the generated functional point corresponds to the striking head side, not the claw side
- a dedicated eval task config can switch `beat_block_hammer` to the new asset
- `script/eval_policy.py` can run `beat_block_hammer` end-to-end with that task config
- the original `020_hammer` asset and existing task configs remain unchanged by default
