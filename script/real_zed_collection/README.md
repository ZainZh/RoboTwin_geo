# Real Three-ZED Collection

This folder keeps real-robot collection separate from DP3 training code.

## Pipeline

1. Calibrate relative extrinsics between the three fixed ZED cameras.

```bash
python script/real_zed_collection/calibrate_three_zed_extrinsics.py \
  --labels global ego side \
  --reference_label global \
  --output_config /home/zheng/github/geometry_awareness_manipulation/config/device/three_camera_charuco_extrinsics.yaml
```

2. Collect raw robot and ZED data.

```bash
python script/real_zed_collection/collect_zed_robotwin_raw.py \
  --save_data_path ./real_data \
  --project_name hanging_mug_real \
  --calibration_path /home/zheng/github/geometry_awareness_manipulation/config/device/three_camera_charuco_extrinsics.yaml \
  --camera_labels global,ego,side
```

The control loop follows the original `include/xtrainer_clover/experiments/run_control.py` button semantics:

- Button A short press toggles torque lock.
- Button A long press toggles servo mode.
- Button B toggles recording while servo is active.

3. Generate masks with a SAM-style model.

```bash
python script/real_zed_collection/segment_objects_sam.py \
  --raw_episode_dir ./real_data/hanging_mug_real/real_zed_raw/episode_YYYYMMDDHHMMSS \
  --output_mask_root ./real_data/hanging_mug_real/masks/episode_000000 \
  --object_prompts "{A}:mug,{B}:rack" \
  --camera_labels global,ego,side
```

4. Convert one raw episode to RoboTwin-compatible HDF5.

```bash
python script/real_zed_collection/postprocess_raw_to_robotwin_hdf5.py \
  --raw_episode_dir ./real_data/hanging_mug_real/real_zed_raw/episode_YYYYMMDDHHMMSS \
  --output_dir ./data/hanging_mug/demo_real_zed \
  --episode_index 0 \
  --calibration_path /home/zheng/github/geometry_awareness_manipulation/config/device/three_camera_charuco_extrinsics.yaml \
  --camera_labels global,ego,side \
  --object_prompts "{A}:mug,{B}:rack" \
  --mask_root ./real_data/hanging_mug_real/masks/episode_000000
```

The output HDF5 contains the fields consumed by current DP3 preprocessing:

- `/joint_action/vector`
- `/pointcloud`
- `/object_pointcloud/{A}`
- `/object_pointcloud/{B}`
- `/observation/<camera>/rgb`
- `/observation/<camera>/depth`
- `/observation/<camera>/intrinsic_cv`
- `/observation/<camera>/cam2world_gl`

After this step, reuse existing DP3 scripts such as `train_objpc.sh`, `train_ndf_pointwise_hybrid.sh`, or `train_semantic_pointwise_hybrid.sh` with the new task config name.
