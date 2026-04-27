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
By default, `--flip_workspace_z` is enabled because a Charuco board placed on the table usually has board `+Z` pointing into the table. The saved workspace frame therefore uses `workspace +Z = board -Z`, so a normal tabletop crop can use `z_min=0` and `z_max>0`.

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
When `workspace_crop_enabled=true` and `workspace_crop_resize_rgb=false`, cropped RGB is saved at the crop ROI's original camera resolution instead of being resized to `640x360`.

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

4. Select per-camera 2D workspace masks.

For real runs, first click the valid working area in each camera's first frame. The polygon is saved as a per-camera binary mask and will be used as the SAM input domain and the point-cloud reconstruction domain.

```bash
python script/real_zed_collection/select_camera_workspace_masks.py \
  --raw_root "/media/${USER}/Extreme SSD/geo_mani_data/grasp_mug/real_zed_raw" \
  --output_mask_root "/media/${USER}/Extreme SSD/geo_mani_data/grasp_mug/camera_workspace_masks" \
  --camera_labels global,left,right
```

Controls: left click adds a vertex, right click or `u` undoes the last point, `r` resets the current camera, and Enter/Space/`s` saves the current polygon and moves to the next camera.

5. Select first-frame SAM2 tracking boxes.

For the current real pipeline, initialize tracking manually. Draw one bbox for `{A}` and `{B}` in each camera's first frame:

```bash
python script/real_zed_collection/select_sam2_bboxes.py \
  --raw_root "/media/${USER}/Extreme SSD/geo_mani_data/grasp_mug/real_zed_raw" \
  --output_bbox_root "/media/${USER}/Extreme SSD/geo_mani_data/grasp_mug/sam2_bbox_prompts" \
  --camera_labels global,left,right \
  --object_placeholders "{A},{B}" \
  --all_episodes --skip_existing
```

Controls: default mode is bbox prompt; drag with the left mouse button to draw a box. Press `m` to switch between bbox and point prompt modes. In point mode, left click adds a foreground point and right click adds a background point, which is useful when the object is partially occluded. Press `p` to force-refresh the SAM2 preview for the current prompt. Enter/Space/`s` saves the current object, `r` resets the current object prompt, and `q`/Esc aborts. The window header shows the current episode index/name, camera, object placeholder, mode, and preview mask pixel count. By default the selector runs a SAM2 first-frame preview and overlays the current mask with a dimmed background and contour; pass `--disable_sam2_preview` if you only want to save prompts.
Because object poses can differ across recorded episodes, the recommended mode is `--all_episodes`, which saves prompts under `sam2_bbox_prompts/<raw_episode_name>/sam2_bbox_prompts.json`. The postprocess script will use those per-episode prompts first. If you only save one global `sam2_bbox_prompts.json`, it is treated as a fallback and is only safe when every episode starts from the same image-space object pose.

6. Generate SAM2 tracking masks and convert raw episodes.

For the current `grasp_mug` setting with `{A}=cup/mug` and `{B}=box`, use the SAM2 tracking batch driver. By default it writes the processed dataset to the external SSD:

`/media/${USER}/Extreme SSD/geo_mani_data/grasp_mug/robotwin_objpc/demo_real_zed_sam2_objpc`

It also creates a repo-side symlink at `data/grasp_mug/demo_real_zed_sam2_objpc`, so existing DP3 scripts can still find the data without learning a new path.

```bash
python script/real_zed_collection/postprocess/postprocess_real_zed_sam2_objpc_dataset.py \
  --raw_root "/media/${USER}/Extreme SSD/geo_mani_data/grasp_mug/real_zed_raw" \
  --task_name grasp_mug \
  --task_config demo_real_zed_sam2_objpc \
  --object_prompts "{A}:cup,{B}:box" \
  --camera_labels global,left,right \
  --bbox_prompt_root "/media/${USER}/Extreme SSD/geo_mani_data/grasp_mug/sam2_bbox_prompts" \
  --require_per_episode_bboxes \
  --calibration_path script/real_zed_collection/calibration/three_camera_workspace_extrinsics.yaml \
  --frame_mode workspace \
  --camera_workspace_mask_root "/media/${USER}/Extreme SSD/geo_mani_data/grasp_mug/camera_workspace_masks" \
  --sam2_checkpoint "/home/zheng/Datasets/sam2/sam2.1_hiera_large.pt" \
  --sam2_config "sam2.1/sam2.1_hiera_l.yaml" \
  --sam2_device cuda \
  --sam2_autocast_dtype bfloat16 \
  --debug --debug_stride 20 --debug_max_frames 5
```

By default this batch path stores compact training HDF5 files: `/joint_action/vector`, `/pointcloud`, and `/object_pointcloud/{A,B}`. It does not duplicate RGB-D observations into every training HDF5. Add `--store_observations` only when you explicitly need the images/depth for debugging or another downstream pipeline.
With `--camera_workspace_mask_root`, SAM2 sees the polygon-limited image domain during tracking and the saved masks/reconstructed point clouds are also restricted to the clicked 2D polygon for each camera. Pass `--disable_mask_input_domain` only if you want SAM2 to track on the full RGB image while still cropping the saved masks/point clouds afterward.
SAM2 streaming defaults to `--sam2_autocast_dtype bfloat16`, matching the upstream webcam demo and avoiding CUDA SDPA "No available kernel" failures on environments where float32 attention kernels are unavailable. Use `float16` if your GPU does not support bf16, and `none` only if you explicitly want full-float inference.
When `--debug` is enabled, the script saves SAM2 mask overlays under `debug/episode*/mask_overlays_sam2/` and merged `{A}/{B}` colored object clouds under `debug/episode*/pointclouds/frame_*_objects_ab.ply`. Use `--debug_stride` and `--debug_max_frames` to keep these previews small.
If an old repo-side data directory already exists, either move it away or pass `--overwrite_repo_link`. Pass `--no_link_repo_data` only when you do not want `train_objpc.sh` compatibility through a symlink.

Then build the DP3 objpc zarr from `policy/DP3`:

```bash
cd policy/DP3
bash process_data_objpc.sh grasp_mug demo_real_zed_sam2_objpc <episode_num> "{A},{B}"
```

or train directly:

```bash
cd policy/DP3
bash train_objpc.sh grasp_mug demo_real_zed_sam2_objpc <episode_num> <seed> <gpu_id> true "{A},{B}"
```

For online eval/inference with SAM2 tracking, use the SAM2 eval wrapper. If `sam2_bbox_prompt_path` is omitted, the first eval frame will open OpenCV windows and ask you to draw `{A}` and `{B}` bboxes for each selected camera.

```bash
cd policy/DP3
bash eval_objpc_sam2.sh \
  grasp_mug <eval_task_config> demo_real_zed_sam2_objpc-objpc \
  <episode_num> <seed> <gpu_id> "{A},{B}" 3000 \
  "/media/${USER}/Extreme SSD/geo_mani_data/grasp_mug/sam2_bbox_prompts" \
  global,left,right \
  "/home/zheng/Datasets/sam2/sam2.1_hiera_large.pt" cuda:0
```

For single-episode/manual debugging with precomputed masks, convert one raw episode to RoboTwin-compatible HDF5:

```bash
python script/real_zed_collection/postprocess/postprocess_raw_to_robotwin_hdf5.py \
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
