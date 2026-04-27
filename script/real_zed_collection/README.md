# Real Three-ZED Collection

This folder keeps real-robot collection separate from DP3 training code.

## Pipeline

1. Calibrate relative extrinsics between the three fixed ZED cameras.

```bash
python script/real_zed_collection/calibrate_three_zed_extrinsics.py \
  --labels global left right \
  --serials 38968158 31021548 37856216 \
  --reference_label global \
  --charuco_config_name charuco_300_9x14_20mm_15mm \
  --output_config script/real_zed_collection/calibration/three_camera_charuco_extrinsics.yaml
```

Local Charuco YAML files can be stored under `script/real_zed_collection/charuco_config/`.

2. Optionally define a workspace/world frame and check whether the weekend camera calibration still agrees.

Place the Charuco board at the desired workspace origin, keep it visible to all three ZEDs, then run:

```bash
python script/real_zed_collection/calibrate_workspace_frame.py \
  --calibration_path script/real_zed_collection/calibration/three_camera_charuco_extrinsics.yaml \
  --output_config script/real_zed_collection/calibration/three_camera_workspace_extrinsics.yaml \
  --workspace_bbox -0.35 0.35 -0.25 0.45 0.0 0.45
```

The script writes `workspace.t_workspace_from_ref`, `relative_to_workspace`, the default workspace bbox, and a health check with per-camera residuals. Use `--fail_on_bad_check` if you want the command to exit non-zero when residuals exceed thresholds.

3. Collect raw robot and ZED data.

```bash
python script/real_zed_collection/collect_zed_robotwin_raw.py \
  --save_data_path ./real_data \
  --project_name hanging_mug_real \
  --calibration_path script/real_zed_collection/calibration/three_camera_charuco_extrinsics.yaml \
  --camera_labels global,left,right
```

By default, the collection script reads `script/real_zed_collection/configs/real_zed_collection.yaml`. Use `--config <path>` to point to another YAML, and any CLI flags you pass will override the YAML values.
The default collection config keeps depth at camera resolution, stores RGB at `640x360`, samples at `5 Hz`, and writes frames asynchronously to protect the robot control loop.

To save only workspace-cropped RGB-D on the next collection run:

```bash
python script/real_zed_collection/collect_zed_robotwin_raw.py \
  --workspace_crop_enabled \
  --workspace_calibration_path script/real_zed_collection/calibration/three_camera_workspace_extrinsics.yaml \
  --workspace_crop_x_min -0.35 --workspace_crop_x_max 0.35 \
  --workspace_crop_y_min -0.25 --workspace_crop_y_max 0.45 \
  --workspace_crop_z_min 0.0 --workspace_crop_z_max 0.45 \
  --workspace_crop_margin_px 32
```

Each cropped camera frame stores adjusted `camera_matrix`, `depth_crop_box_xyxy`, original image/depth shape, `t_workspace_from_camera`, and workspace bounds. Keep `workspace_crop_resize_rgb=false` unless you intentionally want resized/distorted crop images; depth remains the geometry reference.

The control loop follows the original `include/xtrainer_clover/experiments/run_control.py` button semantics:

- Button A short press toggles torque lock.
- Button A long press toggles servo mode.
- Button B toggles recording while servo is active.

4. Generate masks with a SAM-style model.

```bash
python script/real_zed_collection/segment_objects_sam.py \
  --raw_episode_dir ./real_data/hanging_mug_real/real_zed_raw/episode_YYYYMMDDHHMMSS \
  --output_mask_root ./real_data/hanging_mug_real/masks/episode_000000 \
  --object_prompts "{A}:mug,{B}:rack" \
  --camera_labels global,left,right
```

5. Convert one raw episode to RoboTwin-compatible HDF5.

```bash
python script/real_zed_collection/postprocess_raw_to_robotwin_hdf5.py \
  --raw_episode_dir ./real_data/hanging_mug_real/real_zed_raw/episode_YYYYMMDDHHMMSS \
  --output_dir ./data/hanging_mug/demo_real_zed \
  --episode_index 0 \
  --calibration_path script/real_zed_collection/calibration/three_camera_charuco_extrinsics.yaml \
  --camera_labels global,left,right \
  --object_prompts "{A}:mug,{B}:rack" \
  --mask_root ./real_data/hanging_mug_real/masks/episode_000000 \
  --frame_mode workspace \
  --workspace_crop_x_min -0.35 --workspace_crop_x_max 0.35 \
  --workspace_crop_y_min -0.25 --workspace_crop_y_max 0.45 \
  --workspace_crop_z_min 0.0 --workspace_crop_z_max 0.45
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
