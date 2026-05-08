# Pour Kettle Mug Design

## Goal

Add a new task `pour_kettle_mug` for RoboTwin.

The task is a geometry-only pouring task:

- use only the left arm
- grasp `009_kettle`
- keep `039_mug` static on the table
- move the kettle spout above the mug opening
- tilt the kettle into a clear pouring pose

The task does not simulate liquid and does not require the mug to be touched.

## Behavior

The episode should look like this:

1. spawn a random `009_kettle` instance on the left side of the table
2. spawn a random `039_mug` instance in a small randomized area near the middle-right of the table
3. keep the mug upright on the table
4. use the left arm to grasp the kettle handle
5. lift the kettle clear of the table
6. move the kettle to a pre-pour pose above the mug
7. rotate and lower into a final pour pose with the spout proxy above the mug opening
8. stop in that pose and evaluate success

The right arm remains unused and should stay at its default home pose.

## Asset Assumptions

### Kettle

Use `rand_create_sapien_urdf_obj()` with `modelname="009_kettle"` and random `modelid` from the three currently available URDF submodels.

Use the existing asset metadata:

- contact point `0` on `link_1` as the kettle handle grasp point
- functional point `0` on `link_2` as the spout proxy point

The kettle should remain a movable articulation. Do not fix its root link.

### Mug

Use `rand_create_actor()` with `modelname="039_mug"` and a random mug `model_id` from the available set.

Use the existing mug metadata:

- functional point `1` as the mug bottom-center proxy

The mug stays upright on the table and is never grasped.

## Scene Layout

### Kettle Spawn

Spawn the kettle in a conservative region on the left half of the table so the left arm can reach it without sweeping across the mug.

Recommended initial region:

- `xlim`: `[-0.26, -0.16]`
- `ylim`: `[-0.10, 0.04]`
- yaw-only randomization with a small range

The kettle orientation should be randomized only enough to avoid overfitting to one pose. Do not allow extreme initial rotations that make the grasp unstable.

### Mug Spawn

Spawn the mug upright in a small region near the middle-right area of the table.

Recommended initial region:

- `xlim`: `[0.02, 0.10]`
- `ylim`: `[-0.02, 0.08]`
- keep the mug upright
- yaw can be fixed or slightly randomized; large rotation variation is unnecessary for the first version

This keeps the task focused on the pouring motion rather than large-scale search or collision-heavy planning.

## Action Sequence

Implement the demo with a simple deterministic sequence.

### Step 1: Grasp Kettle

Use the left arm to grasp the kettle at contact point `0`.

Recommended call shape:

- `arm_tag = ArmTag("left")`
- `self.grasp_actor(self.kettle, arm_tag=arm_tag, pre_grasp_dis=0.08, contact_point_id=0)`

### Step 2: Lift Clear

Lift the kettle vertically after grasping so the body clears the table before lateral motion.

Recommended displacement:

- `z += 0.10` to `0.14`

### Step 3: Move To Pre-Pour Pose

Move to a pose above the mug opening before applying the final tilt.

The pre-pour pose should:

- place the spout proxy above the mug opening center
- keep some safety height above the mug
- keep the kettle closer to upright than the final pose

### Step 4: Move To Final Pour Pose

Move into a final pour pose with:

- the kettle spout proxy horizontally aligned with the mug opening center within a small tolerance
- the spout proxy vertically above the mug opening by a small positive offset
- the kettle tilted enough to clearly represent pouring

The demo does not need to place the kettle back on the table.

## Pose Construction

The implementation should avoid trying to invent a liquid model. It only needs a stable geometric target.

### Mug Opening Center Approximation

Approximate the mug opening center from the mug metadata and dimensions:

1. take `mug_bottom = self.mug.get_functional_point(1)`
2. estimate mug height from `self.mug.config["extents"][2] * self.mug.config["scale"][2]`
3. define `mug_opening_center = mug_bottom[:3] + [0, 0, mug_height * 0.8]`

This is preferable to using the raw actor origin because the mug origin is not guaranteed to coincide with the opening center.

### Spout Proxy

Use `self.kettle.get_functional_point(0)` as the kettle spout proxy in both motion construction and success checking.

### Final Tilt

Use a fixed pour quaternion for the left arm in the first version.

The exact quaternion can be tuned during implementation, but the design goal is:

- the kettle must be visibly tilted
- the spout side must face downward toward the mug opening
- the pose must remain reachable by the left arm from the chosen spawn regions

The first implementation should prefer a stable fixed pour orientation over a more general but brittle orientation solver.

## Success Criteria

The task succeeds only if all conditions below hold.

### 1. Mug Remains Stable

The mug must remain upright on the table.

Recommended checks:

- mug tilt from world-up stays below a small threshold
- mug base height stays near its initial table height

This prevents collisions that knock or drag the mug.

### 2. Spout Proxy Is Above Mug Opening

Let:

- `spout = self.kettle.get_functional_point(0)[:3]`
- `opening = mug_opening_center`

Require:

- horizontal error `norm((spout - opening)[:2]) < 0.04`
- vertical offset `0.02 < spout[2] - opening[2] < 0.10`

This enforces a plausible pouring position rather than just a nearby pose.

### 3. Kettle Is Tilted

The kettle body must clearly deviate from upright.

Recommended check:

- compare the kettle local up axis against world up
- require tilt angle greater than roughly `45` degrees

This keeps the success condition tied to pouring rather than hovering.

### 4. Grasp Still Held

The left gripper should still be closed on the kettle at the final pose.

This avoids counting accidental ballistic motion or post-release drift as success.

## Information Returned For Language Generation

`play_once()` should populate:

- `"{A}" -> "009_kettle/base{selected_kettle_id}"`
- `"{B}" -> "039_mug/base{selected_mug_id}"`
- `"{a}" -> "left"`

This keeps the task compatible with the existing instruction generation pipeline.

## Object Point Cloud Mapping

Register default placeholder mappings for this task:

- `"{A}" -> "kettle"`
- `"{B}" -> "mug"`

This should be added in `envs/object_pointcloud_targets.py`.

## Files To Change

### New Files

- `envs/pour_kettle_mug.py`
- `description/task_instruction/pour_kettle_mug.json`

### Existing Files

- `envs/object_pointcloud_targets.py`
- `task_config/_eval_step_limit.yml`

No new object description files are needed because both `009_kettle` and `039_mug` already have description JSON files.

## Testing

### Smoke Test

Run one rendered episode with a debug config or an existing clean config with `episode_num=1` and `collect_data=false`.

The smoke test should confirm:

- kettle grasp succeeds
- kettle clears the table
- final pose lands above the mug
- mug is not knocked over

### Small Seed Sweep

Run `3` to `5` seeds without rendering and check:

- no frequent planner failure
- no frequent kettle-table or kettle-mug collision
- success condition matches the visible result

### Regression Checks

After the task is implemented:

- ensure `script/collect_data.py` can import `envs.pour_kettle_mug`
- ensure instruction generation works with the returned placeholders
- ensure object-pointcloud metadata resolves for placeholders `A` and `B`

## Risks

- the kettle functional point may not perfectly correspond to the physical spout tip, so the final thresholds may need minor tuning
- a too-large mug randomization region can turn the task into a planning benchmark instead of a pouring benchmark
- a too-aggressive tilt quaternion may cause the kettle body to collide with the mug

## Recommendation

Start with a conservative, stable implementation:

- small mug randomization
- small kettle yaw randomization
- fixed left-arm pour quaternion
- simple geometric success check

Do not optimize for realism in the first pass. Optimize for a robust and readable benchmark task that matches the existing RoboTwin task style.
