# import packages and module here
import sys

import torch
import sapien.core as sapien
import traceback
import os
import numpy as np
from envs import *
from hydra import initialize, compose
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig
from hydra import main as hydra_main
import pathlib
from omegaconf import OmegaConf

import yaml
from datetime import datetime
import importlib
import dill
import importlib.util

from hydra import initialize, compose
from omegaconf import OmegaConf
from datetime import datetime

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)

sys.path.append(os.path.join(parent_directory, '3D-Diffusion-Policy'))
sys.path.append(os.path.join(parent_directory, 'scripts'))

from dp3_policy import *
from ndf_feature_utils import (
    compute_ndf_feature,
    compute_ndf_interact_pointwise_cloud,
    compute_ndf_pointwise_cloud,
    load_ndf_model,
)
from object_pointcloud_utils import merge_object_point_clouds, parse_placeholder_list
from pointwise_context_utils import build_context_point_cloud


def placeholder_feature_key(placeholder: str) -> str:
    return f"ndf_feat_{placeholder.strip('{}')}"


def placeholder_pointcloud_key(placeholder: str) -> str:
    return f"ndf_point_cloud_{placeholder.strip('{}')}"


def placeholder_interact_pointcloud_key(query_placeholder: str, support_placeholder: str) -> str:
    return (
        f"ndf_interact_point_cloud_{query_placeholder.strip('{}')}_from_{support_placeholder.strip('{}')}"
    )


def placeholder_semantic_pointcloud_key(placeholder: str) -> str:
    return f"semantic_point_cloud_{placeholder.strip('{}')}"


def get_semantic_utils():
    # Import lazily so baseline / NDF eval does not depend on semantic-field runtime deps.
    from semantic_feature_utils import compute_semantic_pointwise_cloud, load_semantic_model

    return compute_semantic_pointwise_cloud, load_semantic_model


def get_sam3_utils():
    from sam3_pointcloud_utils import (
        SAM3ProjectiveTracker,
        build_placeholder_prompt_map_from_targets,
        extract_placeholder_point_cloud_sam3_online,
        parse_camera_list,
        parse_prompt_map,
    )

    return (
        SAM3ProjectiveTracker,
        build_placeholder_prompt_map_from_targets,
        extract_placeholder_point_cloud_sam3_online,
        parse_camera_list,
        parse_prompt_map,
    )


def get_actorseg_utils():
    from actorseg_pointcloud_utils import (
        extract_placeholder_point_cloud_actorseg_online,
        parse_camera_list,
    )

    return extract_placeholder_point_cloud_actorseg_online, parse_camera_list


def resolve_task_object_pointcloud_targets(task_name):
    repo_root = pathlib.Path(parent_directory).parents[1]
    registry_path = repo_root / "envs" / "object_pointcloud_targets.py"
    if not registry_path.is_file():
        return {}
    spec = importlib.util.spec_from_file_location("robotwin_object_pointcloud_targets", registry_path)
    if spec is None or spec.loader is None:
        return {}
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    getter = getattr(module, "get_task_object_pointcloud_targets", None)
    if getter is None or not callable(getter):
        return {}
    targets = getter(task_name)
    return targets if isinstance(targets, dict) else {}


def resolve_task_env_actor_ids(task_env, placeholders):
    result = {}

    info_targets = getattr(task_env, "info", {}).get("object_pointcloud", {}).get("targets", {})
    if isinstance(info_targets, dict):
        for placeholder in placeholders:
            actor_ids = info_targets.get(placeholder, {}).get("actor_ids", [])
            if len(actor_ids) > 0:
                result[str(placeholder)] = [int(actor_id) for actor_id in actor_ids]

    if len(result) == len(placeholders):
        return result

    object_pointcloud_targets = getattr(task_env, "object_pointcloud_targets", {})
    if isinstance(object_pointcloud_targets, dict):
        for placeholder in placeholders:
            actor_ids = object_pointcloud_targets.get(placeholder, {}).get("actor_ids", [])
            if len(actor_ids) > 0:
                result[str(placeholder)] = [int(actor_id) for actor_id in actor_ids]

    if len(result) == len(placeholders):
        return result

    cfg_targets = getattr(task_env, "object_pointcloud_config", {})
    if isinstance(cfg_targets, dict):
        cfg_targets = cfg_targets.get("targets", {})
    else:
        cfg_targets = {}
    task_method_targets = {}
    getter = getattr(task_env, "get_object_pointcloud_targets", None)
    if getter is not None and callable(getter):
        try:
            task_method_targets = getter() or {}
        except Exception:
            task_method_targets = {}
    registry_targets = resolve_task_object_pointcloud_targets(getattr(task_env, "task_name", None))
    target_specs = cfg_targets or task_method_targets or registry_targets

    resolve_attr = getattr(task_env, "_resolve_target_attr_path", None)
    flatten_refs = getattr(task_env, "_flatten_target_refs", None)
    resolve_actor_ids = getattr(task_env, "_resolve_target_actor_ids", None)
    if not callable(resolve_attr) or not callable(flatten_refs) or not callable(resolve_actor_ids):
        return result

    for placeholder in placeholders:
        if placeholder in result:
            continue
        attr_spec = target_specs.get(placeholder)
        if attr_spec is None:
            continue
        attr_paths = attr_spec if isinstance(attr_spec, (list, tuple)) else [attr_spec]
        resolved_targets = []
        try:
            for attr_path in attr_paths:
                resolved_targets.extend(flatten_refs(resolve_attr(attr_path)))
            actor_ids = resolve_actor_ids(resolved_targets)
        except Exception:
            continue
        if len(actor_ids) > 0:
            result[str(placeholder)] = [int(actor_id) for actor_id in actor_ids]

    return result


def encode_obs(observation, model):  # Post-Process Observation
    obs = dict()
    obs['agent_pos'] = observation['joint_action']['vector']
    point_cloud = observation['pointcloud']
    object_pointcloud = observation.get('object_pointcloud', {})
    use_actorseg_objpc = bool(getattr(model, "use_actorseg_objpc", False))
    use_sam3_objpc = bool(getattr(model, "use_sam3_objpc", False))
    use_ndf_pointwise = bool(getattr(model, "use_ndf_pointwise", False))
    use_ndf_pointwise_hybrid = bool(getattr(model, "use_ndf_pointwise_hybrid", False))
    use_ndf_pointwise_interact = bool(getattr(model, "use_ndf_pointwise_interact", False))
    use_semantic_pointwise = bool(getattr(model, "use_semantic_pointwise", False))
    use_semantic_pointwise_hybrid = bool(getattr(model, "use_semantic_pointwise_hybrid", False))

    if use_actorseg_objpc:
        per_placeholder_clouds = []
        actorseg_object_pointcloud = {}
        for placeholder in getattr(model, "object_placeholders", []):
            actor_ids = getattr(model, "actorseg_actor_ids_by_placeholder", {}).get(placeholder, [])
            object_pc, _ = model.actorseg_extract_fn(
                observation,
                placeholder=placeholder,
                actor_ids=actor_ids,
                camera_names=getattr(model, "actorseg_camera_names", ["head_camera", "front_camera"]),
                target_num_points=int(getattr(model, "target_num_points", 1024)),
                segmentation_key=str(getattr(model, "actorseg_segmentation_key", "actor_segmentation")),
            )
            actorseg_object_pointcloud[str(placeholder)] = object_pc
            per_placeholder_clouds.append(object_pc)
        point_cloud = merge_object_point_clouds(
            per_placeholder_clouds,
            target_num_points=int(getattr(model, "target_num_points", 1024)),
        )
        object_pointcloud = actorseg_object_pointcloud

    elif use_sam3_objpc:
        per_placeholder_clouds = []
        frame_idx = int(getattr(model, "sam3_frame_idx", 0))
        for placeholder in getattr(model, "object_placeholders", []):
            object_pc, _ = model.sam3_extract_fn(
                observation,
                placeholder=placeholder,
                prompt=getattr(model, "sam3_prompt_map", {}).get(placeholder, placeholder.strip("{}")),
                camera_names=getattr(model, "sam3_camera_names", ["head_camera", "front_camera"]),
                tracker=getattr(model, "sam3_tracker"),
                tracking_state_by_camera=getattr(model, "sam3_tracking_state", {}).setdefault(placeholder, {}),
                target_num_points=int(getattr(model, "target_num_points", 1024)),
                target_extents=None,
                min_mask_points=int(getattr(model, "sam3_min_mask_points", 16)),
                text_refresh_every=int(getattr(model, "sam3_text_refresh_every", 15)),
                frame_idx=frame_idx,
            )
            per_placeholder_clouds.append(object_pc)
        point_cloud = merge_object_point_clouds(
            per_placeholder_clouds,
            target_num_points=int(getattr(model, "target_num_points", 1024)),
        )
        model.sam3_frame_idx = frame_idx + 1

    elif getattr(model, "use_object_pointcloud", False) and len(object_pointcloud) > 0:
        placeholders = getattr(model, "object_placeholders", [])
        if use_ndf_pointwise or use_semantic_pointwise:
            feature_placeholders = set(getattr(model, "ndf_models", {}).keys()) | set(
                getattr(model, "semantic_models", {}).keys()
            )
            point_cloud, _ = build_context_point_cloud(
                object_pointcloud,
                placeholders=placeholders,
                feature_placeholders=sorted(feature_placeholders),
                target_num_points=int(getattr(model, "target_num_points", 1024)),
                keep_feature_placeholders_in_context=(
                    use_ndf_pointwise_hybrid or use_semantic_pointwise_hybrid
                ),
            )
        else:
            ordered_point_clouds = [object_pointcloud[key] for key in placeholders if key in object_pointcloud]
            if len(ordered_point_clouds) == 0:
                ordered_point_clouds = list(object_pointcloud.values())
            point_cloud = merge_object_point_clouds(
                ordered_point_clouds,
                target_num_points=int(getattr(model, "target_num_points", 1024)),
            )

    obs['point_cloud'] = point_cloud

    if use_ndf_pointwise:
        feat_dim = int(getattr(model, "ndf_feat_dim", 256))
        for placeholder, ndf_model in getattr(model, "ndf_models", {}).items():
            pointcloud_key = placeholder_pointcloud_key(placeholder)
            point_num = int(getattr(model, "ndf_point_num_by_placeholder", {}).get(placeholder, 128))
            object_pc = object_pointcloud.get(placeholder)
            if object_pc is None:
                obs[pointcloud_key] = np.zeros((point_num, 3 + feat_dim), dtype=np.float32)
                continue
            obs[pointcloud_key] = compute_ndf_pointwise_cloud(
                model=ndf_model,
                object_point_cloud=object_pc,
                device=getattr(model, "ndf_device", torch.device("cpu")),
                target_num_points=point_num,
            ).astype(np.float32)
        if use_ndf_pointwise_interact:
            for support_placeholder, ndf_model in getattr(model, "ndf_models", {}).items():
                support_object_pc = object_pointcloud.get(support_placeholder)
                for query_placeholder in getattr(model, "object_placeholders", []):
                    if query_placeholder == support_placeholder:
                        continue
                    pointcloud_key = placeholder_interact_pointcloud_key(query_placeholder, support_placeholder)
                    point_num = int(
                        getattr(model, "ndf_interact_point_num_by_pair", {}).get(
                            (query_placeholder, support_placeholder),
                            int(getattr(model, "ndf_point_num_by_placeholder", {}).get(support_placeholder, 128)),
                        )
                    )
                    query_object_pc = object_pointcloud.get(query_placeholder)
                    if support_object_pc is None or query_object_pc is None:
                        obs[pointcloud_key] = np.zeros((point_num, 3 + feat_dim), dtype=np.float32)
                        continue
                    obs[pointcloud_key] = compute_ndf_interact_pointwise_cloud(
                        model=ndf_model,
                        support_object_point_cloud=support_object_pc,
                        query_object_point_cloud=query_object_pc,
                        device=getattr(model, "ndf_device", torch.device("cpu")),
                        target_num_points=point_num,
                    ).astype(np.float32)

    if use_semantic_pointwise:
        compute_semantic_pointwise_cloud, _ = get_semantic_utils()
        default_sem_dim = int(getattr(model, "semantic_feat_dim", 128))
        for placeholder, semantic_artifacts in getattr(model, "semantic_models", {}).items():
            pointcloud_key = placeholder_semantic_pointcloud_key(placeholder)
            point_num = int(getattr(model, "semantic_point_num_by_placeholder", {}).get(placeholder, 128))
            feat_dim = int(getattr(model, "semantic_feat_dim_by_placeholder", {}).get(placeholder, default_sem_dim))
            object_pc = object_pointcloud.get(placeholder)
            if object_pc is None:
                obs[pointcloud_key] = np.zeros((point_num, 3 + feat_dim), dtype=np.float32)
                continue
            obs[pointcloud_key] = compute_semantic_pointwise_cloud(
                artifacts=semantic_artifacts,
                object_point_cloud=object_pc,
                target_num_points=point_num,
            ).astype(np.float32)

    if use_ndf_pointwise or use_semantic_pointwise:
        return obs

    for placeholder, ndf_model in getattr(model, "ndf_models", {}).items():
        feature_key = placeholder_feature_key(placeholder)
        feat_dim = int(getattr(model, "ndf_feat_dim", 256))
        object_pc = object_pointcloud.get(placeholder)
        if object_pc is None:
            obs[feature_key] = np.zeros((feat_dim,), dtype=np.float32)
            continue
        obs[feature_key] = compute_ndf_feature(
            model=ndf_model,
            object_point_cloud=object_pc,
            device=getattr(model, "ndf_device", torch.device("cpu")),
        ).astype(np.float32)
    return obs


def resolve_ndf_models(usr_args):
    model_specs = {}
    ckpt_a = usr_args.get("ndf_ckpt_A", "none")
    ckpt_b = usr_args.get("ndf_ckpt_B", "none")
    if ckpt_a not in {None, "", "none"}:
        model_specs["{A}"] = ckpt_a
    if ckpt_b not in {None, "", "none"}:
        model_specs["{B}"] = ckpt_b
    return model_specs


def resolve_semantic_models(usr_args):
    model_specs = {}
    ckpt_a = usr_args.get("semantic_ckpt_A", "none")
    ckpt_b = usr_args.get("semantic_ckpt_B", "none")
    if ckpt_a not in {None, "", "none"}:
        model_specs["{A}"] = ckpt_a
    if ckpt_b not in {None, "", "none"}:
        model_specs["{B}"] = ckpt_b
    return model_specs


def resolve_checkpoint_path(usr_args, use_rgb: bool) -> pathlib.Path:
    suffix = "_w_rgb" if use_rgb else ""
    return pathlib.Path(
        os.path.join(
            parent_directory,
            f"./checkpoints/{usr_args['task_name']}-{usr_args['ckpt_setting']}-{usr_args['expert_data_num']}{suffix}_{usr_args['seed']}/{usr_args['checkpoint_num']}.ckpt",
        )
    )


def infer_checkpoint_use_ema(ckpt_path: pathlib.Path):
    if not ckpt_path.is_file():
        return None
    try:
        payload = torch.load(ckpt_path.open("rb"), pickle_module=dill, map_location="cpu")
    except Exception:
        return None

    state_dicts = payload.get("state_dicts", {}) if isinstance(payload, dict) else {}
    if "ema_model" in state_dicts:
        return True

    payload_cfg = payload.get("cfg") if isinstance(payload, dict) else None
    try:
        if payload_cfg is not None:
            training_cfg = getattr(payload_cfg, "training", None)
            if training_cfg is None and hasattr(payload_cfg, "get"):
                training_cfg = payload_cfg.get("training", None)
            if training_cfg is not None:
                use_ema = getattr(training_cfg, "use_ema", None)
                if use_ema is None and hasattr(training_cfg, "get"):
                    use_ema = training_cfg.get("use_ema", None)
                if use_ema is not None:
                    return bool(use_ema)
    except Exception:
        pass

    # Checkpoint without ema_model should be treated as non-EMA to avoid
    # accidentally evaluating an uninitialized ema_model instance.
    return False


def get_model(usr_args):
    config_path = "./3D-Diffusion-Policy/diffusion_policy_3d/config"
    config_name = f"{usr_args['config_name']}.yaml"

    with initialize(config_path=config_path, version_base='1.2'):
        cfg = compose(config_name=config_name)

    now = datetime.now()
    run_dir = f"data/outputs/{now:%Y.%m.%d}/{now:%H.%M.%S}_{usr_args['config_name']}_{usr_args['task_name']}"

    hydra_runtime_cfg = {
        "job": {
            "override_dirname": usr_args['task_name']
        },
        "run": {
            "dir": run_dir
        },
        "sweep": {
            "dir": run_dir,
            "subdir": "0"
        }
    }

    OmegaConf.set_struct(cfg, False)
    cfg.hydra = hydra_runtime_cfg
    cfg.task_name = usr_args["task_name"]
    cfg.expert_data_num = usr_args["expert_data_num"]
    cfg.raw_task_name = usr_args["task_name"]
    cfg.policy.use_pc_color = usr_args['use_rgb']

    use_actorseg_objpc = "objpc_actorseg" in usr_args["config_name"]
    use_sam3_objpc = "objpc_sam3" in usr_args["config_name"]
    use_ndf_pointwise = "ndf_pointwise" in usr_args["config_name"]
    use_ndf_pointwise_hybrid = "ndf_pointwise_hybrid" in usr_args["config_name"]
    use_ndf_pointwise_interact = "ndf_pointwise_hybrid_interact" in usr_args["config_name"]
    use_semantic_pointwise = "semantic_pointwise" in usr_args["config_name"]
    use_semantic_pointwise_hybrid = "semantic_pointwise_hybrid" in usr_args["config_name"]
    use_object_pointcloud = (
        (("objpc" in usr_args["config_name"]) and not use_sam3_objpc and not use_actorseg_objpc)
        or use_ndf_pointwise
        or use_semantic_pointwise
    )
    object_placeholders = parse_placeholder_list(usr_args.get("object_placeholders", "{A},{B}"))
    target_num_points = int(cfg.task.shape_meta.obs.point_cloud.shape[0])
    ndf_feat_dim = 256
    ndf_point_num = int(usr_args.get("ndf_point_num", 128))
    ndf_device = torch.device(usr_args.get("ndf_device", "cuda:0") if torch.cuda.is_available() else "cpu")
    ndf_model_specs = resolve_ndf_models(usr_args)
    dgcnn_placeholders = set(parse_placeholder_list(usr_args.get("ndf_dgcnn_placeholders", "")))
    semantic_point_num = int(usr_args.get("semantic_point_num", 128))
    semantic_device = torch.device(usr_args.get("semantic_device", "cuda:0") if torch.cuda.is_available() else "cpu")
    semantic_model_specs = resolve_semantic_models(usr_args)
    semantic_feat_dim_by_placeholder = {}
    ckpt_file = resolve_checkpoint_path(usr_args, use_rgb=bool(usr_args.get("use_rgb", False)))
    checkpoint_use_ema = infer_checkpoint_use_ema(ckpt_file)
    if checkpoint_use_ema is not None:
        cfg.training.use_ema = bool(checkpoint_use_ema)
    actorseg_extract_fn = None
    actorseg_camera_names = []
    actorseg_segmentation_key = "actor_segmentation"
    sam3_tracker = None
    sam3_extract_fn = None
    sam3_camera_names = []
    sam3_prompt_map = {}
    sam3_text_refresh_every = int(usr_args.get("sam3_text_refresh_every", 15))
    sam3_min_mask_points = int(usr_args.get("sam3_min_mask_points", 16))

    if use_actorseg_objpc:
        extract_placeholder_point_cloud_actorseg_online, parse_actorseg_camera_list = get_actorseg_utils()
        actorseg_extract_fn = extract_placeholder_point_cloud_actorseg_online
        actorseg_camera_names = parse_actorseg_camera_list(
            usr_args.get("actorseg_camera_names", "head_camera,front_camera")
        )

    if use_sam3_objpc:
        (
            SAM3ProjectiveTracker,
            build_placeholder_prompt_map_from_targets,
            extract_placeholder_point_cloud_sam3_online,
            parse_sam3_camera_list,
            parse_sam3_prompt_map,
        ) = get_sam3_utils()
        sam3_tracker = SAM3ProjectiveTracker(
            model_path=str(usr_args.get("sam3_model", "/home/zheng/Datasets/sam3/sam3.pt")),
            conf=float(usr_args.get("sam3_conf", 0.50)),
            verbose=False,
        )
        sam3_extract_fn = extract_placeholder_point_cloud_sam3_online
        sam3_camera_names = parse_sam3_camera_list(usr_args.get("sam3_camera_names", "head_camera,front_camera"))
        prompt_overrides = parse_sam3_prompt_map(usr_args.get("sam3_prompt_map", ""))
        target_specs = resolve_task_object_pointcloud_targets(usr_args["task_name"])
        sam3_prompt_map = build_placeholder_prompt_map_from_targets(
            object_placeholders,
            target_specs,
            prompt_overrides=prompt_overrides,
        )

    if use_semantic_pointwise:
        _, load_semantic_model = get_semantic_utils()
        semantic_models = {}
        for placeholder, checkpoint in semantic_model_specs.items():
            semantic_models[placeholder] = load_semantic_model(
                checkpoint=checkpoint,
                device=semantic_device,
            )
            semantic_feat_dim_by_placeholder[placeholder] = int(semantic_models[placeholder]["sem_embedding_dim"])
    else:
        semantic_models = {}

    if use_ndf_pointwise:
        for placeholder in object_placeholders:
            checkpoint = ndf_model_specs.get(placeholder)
            if checkpoint in {None, "", "none"}:
                continue
            pointcloud_key = placeholder_pointcloud_key(placeholder)
            cfg.task.shape_meta.obs[pointcloud_key] = {
                "shape": [ndf_point_num, 3 + ndf_feat_dim],
                "type": "point_cloud",
            }
        if use_ndf_pointwise_interact:
            for support_placeholder in object_placeholders:
                checkpoint = ndf_model_specs.get(support_placeholder)
                if checkpoint in {None, "", "none"}:
                    continue
                for query_placeholder in object_placeholders:
                    if query_placeholder == support_placeholder:
                        continue
                    pointcloud_key = placeholder_interact_pointcloud_key(
                        query_placeholder,
                        support_placeholder,
                    )
                    cfg.task.shape_meta.obs[pointcloud_key] = {
                        "shape": [ndf_point_num, 3 + ndf_feat_dim],
                        "type": "point_cloud",
                    }
    elif "ndf" in usr_args["config_name"]:
        for placeholder in object_placeholders:
            checkpoint = ndf_model_specs.get(placeholder)
            if checkpoint in {None, "", "none"}:
                continue
            feature_key = placeholder_feature_key(placeholder)
            cfg.task.shape_meta.obs[feature_key] = {
                "shape": [ndf_feat_dim],
                "type": "low_dim",
            }
    if use_semantic_pointwise:
        for placeholder in object_placeholders:
            artifacts = semantic_models.get(placeholder)
            if artifacts is None:
                continue
            pointcloud_key = placeholder_semantic_pointcloud_key(placeholder)
            cfg.task.shape_meta.obs[pointcloud_key] = {
                "shape": [semantic_point_num, 3 + int(artifacts["sem_embedding_dim"])],
                "type": "point_cloud",
            }
    OmegaConf.set_struct(cfg, True)

    DP3_Model = DP3(cfg, usr_args)
    DP3_Model.use_actorseg_objpc = use_actorseg_objpc
    DP3_Model.use_sam3_objpc = use_sam3_objpc
    DP3_Model.use_object_pointcloud = use_object_pointcloud
    DP3_Model.use_ndf_pointwise = use_ndf_pointwise
    DP3_Model.use_ndf_pointwise_hybrid = use_ndf_pointwise_hybrid
    DP3_Model.use_ndf_pointwise_interact = use_ndf_pointwise_interact
    DP3_Model.use_semantic_pointwise = use_semantic_pointwise
    DP3_Model.use_semantic_pointwise_hybrid = use_semantic_pointwise_hybrid
    DP3_Model.object_placeholders = object_placeholders
    DP3_Model.target_num_points = target_num_points
    DP3_Model.actorseg_extract_fn = actorseg_extract_fn
    DP3_Model.actorseg_camera_names = actorseg_camera_names
    DP3_Model.actorseg_segmentation_key = actorseg_segmentation_key
    DP3_Model.actorseg_actor_ids_by_placeholder = {
        placeholder: []
        for placeholder in object_placeholders
    }
    DP3_Model.sam3_tracker = sam3_tracker
    DP3_Model.sam3_extract_fn = sam3_extract_fn
    DP3_Model.sam3_camera_names = sam3_camera_names
    DP3_Model.sam3_prompt_map = sam3_prompt_map
    DP3_Model.sam3_text_refresh_every = sam3_text_refresh_every
    DP3_Model.sam3_min_mask_points = sam3_min_mask_points
    DP3_Model.sam3_tracking_state = {
        placeholder: {}
        for placeholder in object_placeholders
    }
    DP3_Model.sam3_frame_idx = 0
    DP3_Model.ndf_feat_dim = ndf_feat_dim
    DP3_Model.ndf_point_num_by_placeholder = {
        placeholder: ndf_point_num
        for placeholder in object_placeholders
        if placeholder in ndf_model_specs
    }
    DP3_Model.ndf_interact_point_num_by_pair = {
        (query_placeholder, support_placeholder): ndf_point_num
        for support_placeholder in object_placeholders
        if support_placeholder in ndf_model_specs
        for query_placeholder in object_placeholders
        if query_placeholder != support_placeholder
    }
    DP3_Model.ndf_device = ndf_device
    DP3_Model.ndf_models = {}
    for placeholder, checkpoint in ndf_model_specs.items():
        DP3_Model.ndf_models[placeholder] = load_ndf_model(
            checkpoint=checkpoint,
            dgcnn=placeholder in dgcnn_placeholders,
            device=ndf_device,
            latent_dim=ndf_feat_dim,
        )
    DP3_Model.semantic_device = semantic_device
    DP3_Model.semantic_point_num_by_placeholder = {
        placeholder: semantic_point_num
        for placeholder in object_placeholders
        if placeholder in semantic_models
    }
    DP3_Model.semantic_feat_dim_by_placeholder = semantic_feat_dim_by_placeholder
    DP3_Model.semantic_feat_dim = max(semantic_feat_dim_by_placeholder.values(), default=128)
    DP3_Model.semantic_models = semantic_models
    return DP3_Model


def eval(TASK_ENV, model, observation):
    if getattr(model, "use_actorseg_objpc", False):
        camera_obs = observation.get("observation", {})
        missing = []
        for camera_name in getattr(model, "actorseg_camera_names", []):
            camera_info = camera_obs.get(camera_name)
            if camera_info is None:
                missing.append(f"{camera_name}:missing_camera")
                continue
            if getattr(model, "actorseg_segmentation_key", "actor_segmentation") not in camera_info:
                missing.append(f"{camera_name}:missing_{getattr(model, 'actorseg_segmentation_key', 'actor_segmentation')}")
        if missing:
            raise RuntimeError(
                "Actor-segmentation eval requires the selected cameras to expose simulator segmentation in the "
                f"observation, but missing entries were found: {', '.join(missing)}"
            )
        resolved_actor_ids = resolve_task_env_actor_ids(TASK_ENV, getattr(model, "object_placeholders", []))
        if len(resolved_actor_ids) > 0:
            for placeholder in getattr(model, "object_placeholders", []):
                model.actorseg_actor_ids_by_placeholder[placeholder] = list(resolved_actor_ids.get(placeholder, []))

    obs = encode_obs(observation, model)  # Post-Process Observation
    # instruction = TASK_ENV.get_instruction()

    if len(
            model.env_runner.obs
    ) == 0:  # Force an update of the observation at the first frame to avoid an empty observation window, `obs_cache` here can be modified
        model.update_obs(obs)

    actions = model.get_action()  # Get Action according to observation chunk

    for action in actions:  # Execute each step of the action
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation, model)
        model.update_obs(obs)  # Update Observation, `update_obs` here can be modified


def reset_model(
        model):  # Clean the model cache at the beginning of every evaluation episode, such as the observation window
    model.env_runner.reset_obs()
    if hasattr(model, "actorseg_actor_ids_by_placeholder"):
        model.actorseg_actor_ids_by_placeholder = {
            placeholder: []
            for placeholder in getattr(model, "object_placeholders", [])
        }
    if hasattr(model, "sam3_tracking_state"):
        model.sam3_tracking_state = {
            placeholder: {}
            for placeholder in getattr(model, "object_placeholders", [])
        }
    if hasattr(model, "sam3_frame_idx"):
        model.sam3_frame_idx = 0
