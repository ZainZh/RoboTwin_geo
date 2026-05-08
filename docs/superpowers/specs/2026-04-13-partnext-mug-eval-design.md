# PartNext Mug Eval Design

## Goal

Integrate `PartNext_mesh/Mug_new` into RobotWin as a `hanging_mug` evaluation-only asset family.

The first version must:

- support `sim eval/inference` only
- keep `hanging_mug` scripted logic and success logic unchanged
- generate RobotWin-compatible mug assets from `Mug_new`
- support `--all` conversion into a single RobotWin asset directory
- follow the same principles already used for the PartNext hammer integration

Out of scope for this version:

- demo collection
- training data generation
- DP3 preprocessing changes
- screening or filtering mugs during asset generation

## Task Constraints

`hanging_mug` currently assumes:

- the mug is a rigid object loaded through `create_actor(...)`
- the mug exposes `contact_points_pose`
- the mug exposes `functional_matrix[0]` for hanging
- the mug exposes a bottom-related pose for generic placement compatibility

The task logic must remain unchanged. The new assets must adapt to the task, not the other way around.

## Output Layout

`--all` conversion will generate one consolidated asset family:

- `assets/objects/<output_modelname>/visual/base0.glb ...`
- `assets/objects/<output_modelname>/collision/base0.glb ...`
- `assets/objects/<output_modelname>/model_data0.json ...`
- `assets/objects/<output_modelname>/points_info.json`
- `assets/objects/<output_modelname>/preview/overview0.ply ...`
- `description/objects_description/<output_modelname>/base0.json ...`

This matches the current multi-model RobotWin convention and the newer hammer pipeline.

## Asset Semantics

`Mug_new` annotation is expected to provide:

- `Body -> Main Body`
- `Handle`
- optionally multiple `Handle` nodes
- optionally `Contents`

The generator will treat:

- all `Handle` regions as handle geometry
- `Main Body` as body geometry
- `Contents` as ignored for contact/functional semantics and ignored for collision export unless required for mesh integrity

## Generated Points

### Contact Points

The mug must expose several grasp poses similar in spirit to official `039_mug`.

Generation rule:

- compute the body center from the `Main Body` region
- estimate the mug vertical axis from the full body geometry
- estimate the dominant horizontal axis of the body cross-section
- generate `4` canonical side grasp poses around the mug body
- optionally add `2` extra poses if the geometry supports stable diagonal side grasps

The contact points are intended for side grasping of the mug body, not for grabbing the handle.

### Functional Point 0

`functional_matrix[0]` must represent the hanging semantics:

- point location: handle inner-center or handle centerline center, biased toward the opening used for hanging
- axis convention: perpendicular to the handle plane
- purpose: this is the point used by `hanging_mug.place_actor(... functional_point_id=0 ...)`

If multiple handles exist:

- choose the largest valid handle loop by geometric extent and face coverage

### Functional Point 1

`functional_matrix[1]` must represent a bottom placement semantics:

- point location: mug bottom center
- axis convention: perpendicular to mug bottom

This is added for compatibility with existing RobotWin object conventions even if `hanging_mug` mainly uses point `0`.

### Target Pose

`target_pose[0]` will be the mug bottom center pose.

This follows the same convention used by existing mug assets.

## Collision Geometry

Collision geometry must not reuse the raw visual mesh directly.

Generation rule:

- export split collision geometry from annotated semantic regions
- create at least:
  - one body collision component
  - one handle collision component
- if multiple handles exist, export one collision component per valid handle

The goal is the same as the hammer fix:

- simpler convex decomposition input
- more stable rigid-body resting behavior
- better support for scripted grasp and hanging

## Scale

Scale is estimated by matching the dominant physical extent of official `039_mug` reference assets.

The generator will:

- read one official `039_mug/model_data*.json` reference
- compute loaded extents as `extents * scale`
- estimate a uniform scale for each PartNext mug

No anisotropic scaling is allowed in this version.

## Object Descriptions

Each generated mug must also create:

- `description/objects_description/<output_modelname>/baseN.json`

These description files are required by RobotWin instruction generation even though DP3 does not consume language directly.

The initial description template can be generic mug language and does not need part-specific wording.

## Task Override

`hanging_mug` will get an eval-only override similar to hammer:

- `custom_mug_eval.enabled`
- `custom_mug_eval.modelname`
- `custom_mug_eval.model_id`

`model_id` supports:

- a single integer
- a list of integers, round-robin by episode index

Default behavior stays unchanged:

- if `custom_mug_eval.enabled` is false or absent, keep using official `039_mug`

## Spawn Pose Alignment

For custom mug eval assets, the mug root pose should align semantically with the official mug family.

Generation and runtime should preserve:

- similar initial upright orientation
- similar side-grasp accessibility
- similar hanging-point orientation relative to the world frame

If necessary, runtime spawn pose alignment will follow the same principle used for custom hammer:

- align a reference semantic frame between official and custom assets

For mugs, the preferred reference frame is the primary side grasp contact frame.

## CLI Behavior

The new mug asset preparation script will mirror the hammer CLI style:

- `--partnext_dir`
- `--annotation_path`
- `--output_modelname`
- `--glb_name`
- `--all`

`--all` behavior:

- convert every eligible `Mug_new/*.glb`
- write a single RobotWin asset directory
- assign contiguous `model_id`
- do not screen out any mug in this version

## Validation

Validation must include:

- unit tests for mug annotation parsing and point generation
- unit tests for split collision export
- unit tests for `custom_mug_eval` config resolution
- unit tests for spawn-pose alignment behavior if runtime alignment is added
- `py_compile` on new and modified Python files

Manual smoke validation target:

- generate one `Mug_new` asset family with `--all`
- point `hanging_mug` eval config to the new family
- run short `sim eval/inference`
- confirm the mug can spawn, be grasped, and be used in hanging behavior without changing task logic

## Recommended Implementation Units

The implementation should be split into these units:

- `script/partnext_mug_eval_utils.py`
  - annotation parsing
  - scale estimation
  - contact point generation
  - hanging functional point generation
  - collision export
- `script/prepare_partnext_mug_eval_asset.py`
  - CLI and package writing
- `envs/hanging_mug.py`
  - eval-only mug override
- `task_config/...partnext_mug_eval...yml`
  - eval config
- `script/test_partnext_mug_eval_utils.py`
  - geometry and asset tests
- `script/test_hanging_mug_eval_override.py`
  - override and spawn behavior tests

## Recommendation

Use the hammer integration pattern directly:

- same eval-only philosophy
- same consolidated `--all` asset layout
- same description-file generation
- same runtime override design

The main object-specific differences are:

- mug needs multi-contact side grasps instead of a single handle grasp
- mug functional semantics are handle-inside for hanging, not striking-face for beating
- mug collision should separate body and handle rather than handle and head
