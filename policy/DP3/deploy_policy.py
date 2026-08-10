# import packages and module here
import sys
import math

import json
import torch
import traceback
import os
import numpy as np
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

try:
    import sapien.core as sapien
except Exception:
    sapien = None

try:
    from envs import *  # noqa: F401,F403
except Exception as exc:
    _ENVS_IMPORT_ERROR = exc
else:
    _ENVS_IMPORT_ERROR = None

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)

sys.path.append(os.path.join(parent_directory, '3D-Diffusion-Policy'))
sys.path.append(os.path.join(parent_directory, 'scripts'))
from safe_dp3_checkpoint import load_weights_only_checkpoint

from dp3_policy import *
from object_pointcloud_utils import merge_object_point_clouds, parse_placeholder_list
from pointwise_context_utils import build_context_point_cloud
from se3_relation_token_utils import (
    OBSERVATION_RELATION_ROUTES,
    RELATION_TOKEN_DIM,
    RELATION_TOKEN_KEY,
    SUPPORTED_RELATION_ROUTES,
)
from geometry_relation_estimator import create_estimator_from_spec
from se3_relation_deploy_utils import (
    SUPPORTED_TOKEN_ABLATIONS,
    build_deploy_relation_token,
)


def placeholder_feature_key(placeholder: str) -> str:
    return f"ndf_feat_{placeholder.strip('{}')}"


def placeholder_pointcloud_key(placeholder: str) -> str:
    return f"ndf_point_cloud_{placeholder.strip('{}')}"


def placeholder_interact_pointcloud_key(query_placeholder: str, support_placeholder: str) -> str:
    return (
        f"ndf_interact_point_cloud_{query_placeholder.strip('{}')}_from_{support_placeholder.strip('{}')}"
    )


def placeholder_relation_pointcloud_key(query_placeholder: str, support_placeholder: str) -> str:
    return (
        f"ndf_relation_point_cloud_{query_placeholder.strip('{}')}_in_{support_placeholder.strip('{}')}"
    )


def placeholder_relation_token_key(query_placeholder: str, support_placeholder: str) -> str:
    return f"ndf_relation_token_{query_placeholder.strip('{}')}_in_{support_placeholder.strip('{}')}"


def placeholder_semantic_pointcloud_key(placeholder: str) -> str:
    return f"semantic_point_cloud_{placeholder.strip('{}')}"


def placeholder_utonia_pointcloud_key(placeholder: str) -> str:
    return f"utonia_point_cloud_{placeholder.strip('{}')}"


def get_semantic_utils():
    # Import lazily so baseline / NDF eval does not depend on semantic-field runtime deps.
    from semantic_feature_utils import compute_semantic_pointwise_cloud, load_semantic_model

    return compute_semantic_pointwise_cloud, load_semantic_model


compute_ndf_feature = None
compute_ndf_interact_pointwise_cloud = None
compute_ndf_relation_pointwise_cloud = None
compute_ndf_relation_token = None
compute_ndf_pointwise_cloud = None
load_ndf_model = None


def get_ndf_utils():
    # Import lazily so baseline / semantic real inference does not require NDF runtime deps.
    global compute_ndf_feature
    global compute_ndf_interact_pointwise_cloud
    global compute_ndf_relation_pointwise_cloud
    global compute_ndf_relation_token
    global compute_ndf_pointwise_cloud
    global load_ndf_model
    from ndf_feature_utils import (
        compute_ndf_feature as imported_compute_ndf_feature,
        compute_ndf_interact_pointwise_cloud as imported_compute_ndf_interact_pointwise_cloud,
        compute_ndf_relation_pointwise_cloud as imported_compute_ndf_relation_pointwise_cloud,
        compute_ndf_relation_token as imported_compute_ndf_relation_token,
        compute_ndf_pointwise_cloud as imported_compute_ndf_pointwise_cloud,
        load_ndf_model as imported_load_ndf_model,
    )

    if compute_ndf_feature is None:
        compute_ndf_feature = imported_compute_ndf_feature
    if compute_ndf_interact_pointwise_cloud is None:
        compute_ndf_interact_pointwise_cloud = imported_compute_ndf_interact_pointwise_cloud
    if compute_ndf_relation_pointwise_cloud is None:
        compute_ndf_relation_pointwise_cloud = imported_compute_ndf_relation_pointwise_cloud
    if compute_ndf_relation_token is None:
        compute_ndf_relation_token = imported_compute_ndf_relation_token
    if compute_ndf_pointwise_cloud is None:
        compute_ndf_pointwise_cloud = imported_compute_ndf_pointwise_cloud
    if load_ndf_model is None:
        load_ndf_model = imported_load_ndf_model

    return (
        compute_ndf_feature,
        compute_ndf_interact_pointwise_cloud,
        compute_ndf_relation_pointwise_cloud,
        compute_ndf_pointwise_cloud,
        load_ndf_model,
    )


def get_utonia_utils():
    from utonia_feature_utils import compute_utonia_pointwise_cloud, load_utonia_model

    return compute_utonia_pointwise_cloud, load_utonia_model


def get_matched_utonia_utils():
    from matched_utonia_feature_utils import compute_utonia_features_at_queries

    return compute_utonia_features_at_queries


def get_utonia_projection_utils():
    from utonia_projection_utils import load_projection_artifact, project_utonia_cloud

    return load_projection_artifact, project_utonia_cloud


def get_sam2_utils():
    from sam2_pointcloud_utils import (
        build_sam2_tracker_factory,
        extract_placeholder_point_clouds_sam2_online,
        load_sam2_bbox_prompt_file,
        parse_camera_list,
    )

    return (
        build_sam2_tracker_factory,
        extract_placeholder_point_clouds_sam2_online,
        load_sam2_bbox_prompt_file,
        parse_camera_list,
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


def parse_bool_arg(value, *, default=False):
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


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
    use_sam2_objpc = bool(getattr(model, "use_sam2_objpc", False))
    use_ndf_pointwise = bool(getattr(model, "use_ndf_pointwise", False))
    use_ndf_pointwise_hybrid = bool(getattr(model, "use_ndf_pointwise_hybrid", False))
    use_ndf_pointwise_interact = bool(getattr(model, "use_ndf_pointwise_interact", False))
    use_ndf_relation_token = bool(getattr(model, "use_ndf_relation_token", False))
    use_semantic_pointwise = bool(getattr(model, "use_semantic_pointwise", False))
    use_semantic_pointwise_hybrid = bool(getattr(model, "use_semantic_pointwise_hybrid", False))
    use_se3_relation_token = bool(getattr(model, "use_se3_relation_token", False))
    use_utonia_pointwise = bool(getattr(model, "use_utonia_pointwise", False))
    use_utonia_pointwise_hybrid = bool(getattr(model, "use_utonia_pointwise_hybrid", False))

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

    elif use_sam2_objpc:
        sam2_object_pointcloud, _ = model.sam2_extract_all_fn(
            observation,
            placeholders=getattr(model, "object_placeholders", []),
            camera_names=getattr(model, "sam2_camera_names", ["head_camera", "front_camera"]),
            tracker_factory=getattr(model, "sam2_tracker_factory"),
            tracking_state_by_camera=getattr(model, "sam2_tracking_state_by_camera", {}),
            bbox_prompts_by_camera=getattr(model, "sam2_bbox_prompts_by_camera", {}),
            target_num_points=int(getattr(model, "target_num_points", 1024)),
            min_mask_points=int(getattr(model, "sam2_min_mask_points", 16)),
            interactive_init=bool(getattr(model, "sam2_interactive_init", True)),
        )
        per_placeholder_clouds = [
            sam2_object_pointcloud[placeholder]
            for placeholder in getattr(model, "object_placeholders", [])
            if placeholder in sam2_object_pointcloud
        ]
        point_cloud = merge_object_point_clouds(
            per_placeholder_clouds,
            target_num_points=int(getattr(model, "target_num_points", 1024)),
        )
        object_pointcloud = sam2_object_pointcloud

    elif getattr(model, "use_object_pointcloud", False) and len(object_pointcloud) > 0:
        placeholders = getattr(model, "object_placeholders", [])
        if use_ndf_pointwise or use_semantic_pointwise or use_utonia_pointwise:
            feature_placeholders = set(getattr(model, "ndf_models", {}).keys()) | set(
                getattr(model, "semantic_models", {}).keys()
            ) | set(getattr(model, "utonia_models", {}).keys())
            point_cloud, _ = build_context_point_cloud(
                object_pointcloud,
                placeholders=placeholders,
                feature_placeholders=sorted(feature_placeholders),
                target_num_points=int(getattr(model, "target_num_points", 1024)),
                keep_feature_placeholders_in_context=(
                    use_ndf_pointwise_hybrid or use_semantic_pointwise_hybrid or use_utonia_pointwise_hybrid
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

    if use_se3_relation_token:
        obs[RELATION_TOKEN_KEY] = build_deploy_relation_token(
            observation=observation, object_pointcloud=object_pointcloud, model=model
        )

    if use_ndf_pointwise:
        _, compute_ndf_interact_pointwise_cloud, compute_ndf_relation_pointwise_cloud, compute_ndf_pointwise_cloud, _ = get_ndf_utils()
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

        if bool(getattr(model, "use_ndf_pointwise_relation", False)):
            for support_placeholder, ndf_model in getattr(model, "ndf_models", {}).items():
                support_object_pc = object_pointcloud.get(support_placeholder)
                for query_placeholder in getattr(model, "object_placeholders", []):
                    if query_placeholder == support_placeholder:
                        continue
                    pointcloud_key = placeholder_relation_pointcloud_key(query_placeholder, support_placeholder)
                    point_num = int(
                        getattr(model, "ndf_relation_point_num_by_pair", {}).get(
                            (query_placeholder, support_placeholder),
                            int(getattr(model, "ndf_point_num_by_placeholder", {}).get(support_placeholder, 128)),
                        )
                    )
                    query_object_pc = object_pointcloud.get(query_placeholder)
                    if support_object_pc is None or query_object_pc is None:
                        obs[pointcloud_key] = np.zeros((point_num, 3 + feat_dim), dtype=np.float32)
                        continue
                    obs[pointcloud_key] = compute_ndf_relation_pointwise_cloud(
                        model=ndf_model,
                        support_object_point_cloud=support_object_pc,
                        query_object_point_cloud=query_object_pc,
                        device=getattr(model, "ndf_device", torch.device("cpu")),
                        target_num_points=point_num,
                    ).astype(np.float32)

        if use_ndf_relation_token:
            for support_placeholder, ndf_model in getattr(model, "ndf_models", {}).items():
                support_object_pc = object_pointcloud.get(support_placeholder)
                for query_placeholder in getattr(model, "object_placeholders", []):
                    if query_placeholder == support_placeholder:
                        continue
                    token_key = placeholder_relation_token_key(query_placeholder, support_placeholder)
                    token_dim = int(getattr(model, "ndf_relation_token_dim", 9 + feat_dim))
                    query_object_pc = object_pointcloud.get(query_placeholder)
                    if support_object_pc is None or query_object_pc is None:
                        obs[token_key] = np.zeros((token_dim,), dtype=np.float32)
                        continue
                    point_num = int(
                        getattr(model, "ndf_relation_token_point_num_by_pair", {}).get(
                            (query_placeholder, support_placeholder),
                            int(getattr(model, "ndf_point_num_by_placeholder", {}).get(support_placeholder, 128)),
                        )
                    )
                    obs[token_key] = compute_ndf_relation_token(
                        model=ndf_model,
                        support_object_point_cloud=support_object_pc,
                        query_object_point_cloud=query_object_pc,
                        device=getattr(model, "ndf_device", torch.device("cpu")),
                        target_num_points=point_num,
                        gate_near_ratio=float(getattr(model, "ndf_relation_token_gate_near_ratio", 0.35)),
                        gate_far_ratio=float(getattr(model, "ndf_relation_token_gate_far_ratio", 0.75)),
                        valid_query_radius=float(getattr(model, "ndf_relation_token_valid_query_radius", 0.75)),
                        valid_fraction_min=float(getattr(model, "ndf_relation_token_valid_fraction_min", 0.25)),
                        valid_fraction_max=float(getattr(model, "ndf_relation_token_valid_fraction_max", 0.65)),
                        projection_dim=int(getattr(model, "ndf_relation_token_projection_dim", 0)),
                        projection_seed=int(getattr(model, "ndf_relation_token_projection_seed", 0)),
                        gate_geometry=bool(getattr(model, "ndf_relation_token_gate_geometry", False)),
                    ).astype(np.float32)

    if use_semantic_pointwise:
        semantic_frame_index = int(getattr(model, "semantic_policy_frame_index", 0))
        compute_semantic_pointwise_cloud, _ = get_semantic_utils()
        semantic_output_mode = str(
            getattr(model, "semantic_policy_output_mode", "embedding")
        )
        use_matched_utonia = semantic_output_mode in {
            "matched_utonia",
            "matched_utonia128",
        }
        use_projected_utonia = semantic_output_mode == "matched_utonia128"
        compute_matched_utonia = (
            get_matched_utonia_utils() if use_matched_utonia else None
        )
        default_sem_dim = int(getattr(model, "semantic_feat_dim", 128))
        for placeholder, semantic_artifacts in getattr(model, "semantic_models", {}).items():
            pointcloud_key = (
                placeholder_utonia_pointcloud_key(placeholder)
                if use_matched_utonia
                else placeholder_semantic_pointcloud_key(placeholder)
            )
            point_num = int(getattr(model, "semantic_point_num_by_placeholder", {}).get(placeholder, 128))
            if use_matched_utonia:
                feat_dim = int(
                    getattr(model, "utonia_feat_dim_by_placeholder", {}).get(
                        placeholder, getattr(model, "utonia_feat_dim", 576)
                    )
                )
            else:
                feat_dim = int(getattr(model, "semantic_feat_dim_by_placeholder", {}).get(placeholder, default_sem_dim))
            object_pc = object_pointcloud.get(placeholder)
            if object_pc is None:
                if use_matched_utonia:
                    raise ValueError(
                        f"matched_utonia is missing object support for {placeholder}"
                    )
                obs[pointcloud_key] = np.zeros((point_num, 3 + feat_dim), dtype=np.float32)
                continue
            semantic_cloud = compute_semantic_pointwise_cloud(
                artifacts=semantic_artifacts,
                object_point_cloud=object_pc,
                target_num_points=point_num,
                placeholder=placeholder,
                semantic_input_color_mode=str(getattr(model, "semantic_input_color_mode", "debug_placeholder")),
                semantic_forward_mode=str(getattr(model, "semantic_forward_mode", "reference")),
                output_mode="xyz" if use_matched_utonia else semantic_output_mode,
                output_seed=int(getattr(model, "semantic_policy_output_seed", 20260807)),
                output_frame_index=semantic_frame_index,
            ).astype(np.float32)
            if use_matched_utonia:
                utonia_artifacts = getattr(model, "utonia_models", {}).get(placeholder)
                if utonia_artifacts is None:
                    raise ValueError(
                        f"matched_utonia is missing Utonia artifacts for {placeholder}"
                    )
                matched_cloud = compute_matched_utonia(
                    artifacts=utonia_artifacts,
                    object_point_cloud=object_pc,
                    query_world_xyz=semantic_cloud[:, :3],
                    placeholder=placeholder,
                    color_mode=str(
                        getattr(model, "matched_utonia_color_mode", "debug_placeholder")
                    ),
                    normal_mode=str(
                        getattr(model, "matched_utonia_normal_mode", "fallback")
                    ),
                ).astype(np.float32)
                if use_projected_utonia:
                    _, project_utonia_cloud = get_utonia_projection_utils()
                    matched_cloud = project_utonia_cloud(
                        matched_cloud,
                        getattr(model, "matched_utonia_projection")["matrix"],
                    )
                expected_shape = (point_num, 3 + feat_dim)
                if matched_cloud.shape != expected_shape:
                    raise ValueError(
                        f"matched_utonia produced shape {matched_cloud.shape} for "
                        f"{placeholder}; expected {expected_shape}"
                    )
                if not np.isfinite(matched_cloud).all():
                    raise ValueError(
                        f"matched_utonia produced NaN/Inf for {placeholder}"
                    )
                if not np.array_equal(matched_cloud[:, :3], semantic_cloud[:, :3]):
                    raise ValueError(
                        f"matched_utonia changed semantic query XYZ for {placeholder}"
                    )
                obs[pointcloud_key] = matched_cloud
            else:
                obs[pointcloud_key] = semantic_cloud

        model.semantic_policy_frame_index = semantic_frame_index + 1
    if use_utonia_pointwise:
        compute_utonia_pointwise_cloud, _ = get_utonia_utils()
        default_utonia_dim = int(getattr(model, "utonia_feat_dim", 96))
        for placeholder, utonia_artifacts in getattr(model, "utonia_models", {}).items():
            pointcloud_key = placeholder_utonia_pointcloud_key(placeholder)
            point_num = int(getattr(model, "utonia_point_num_by_placeholder", {}).get(placeholder, 128))
            feat_dim = int(getattr(model, "utonia_feat_dim_by_placeholder", {}).get(placeholder, default_utonia_dim))
            object_pc = object_pointcloud.get(placeholder)
            if object_pc is None:
                obs[pointcloud_key] = np.zeros((point_num, 3 + feat_dim), dtype=np.float32)
                continue
            obs[pointcloud_key] = compute_utonia_pointwise_cloud(
                artifacts=utonia_artifacts,
                object_point_cloud=object_pc,
                target_num_points=point_num,
            ).astype(np.float32)

    if use_ndf_pointwise or use_semantic_pointwise or use_utonia_pointwise:
        return obs

    ndf_models = getattr(model, "ndf_models", {})
    compute_ndf_feature = None
    if len(ndf_models) > 0:
        compute_ndf_feature, _, _, _, _ = get_ndf_utils()
    for placeholder, ndf_model in ndf_models.items():
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


def resolve_utonia_spec(usr_args):
    return {
        "checkpoint": usr_args.get("utonia_checkpoint", "auto"),
        "repo_id": usr_args.get("utonia_repo_id", "Pointcept/Utonia"),
        "upcast_levels": int(usr_args.get("utonia_upcast_levels", 0)),
    }


def resolve_utonia_feature_placeholders(usr_args, object_placeholders):
    selected = parse_placeholder_list(usr_args.get("utonia_feature_placeholders", ""))
    if len(selected) == 0:
        return list(object_placeholders)
    selected_set = set(selected)
    return [placeholder for placeholder in object_placeholders if placeholder in selected_set]


def resolve_checkpoint_path(usr_args, use_rgb: bool) -> pathlib.Path:
    suffix = "_w_rgb" if use_rgb else ""
    return pathlib.Path(
        os.path.join(
            parent_directory,
            f"./checkpoints/{usr_args['task_name']}-{usr_args['ckpt_setting']}-{usr_args['expert_data_num']}{suffix}_{usr_args['seed']}/{usr_args['checkpoint_num']}.ckpt",
        )
    )


def resolve_safe_checkpoint_path(usr_args) -> pathlib.Path | None:
    value = str(usr_args.get("safe_checkpoint_path", "") or "").strip()
    if not value:
        return None
    path = pathlib.Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"safe DP3 checkpoint does not exist: {path}")
    return path


def inspect_safe_policy_metadata(
    checkpoint_path: pathlib.Path,
    *,
    expected_task_name: str,
) -> dict:
    """Read deployment metadata exclusively through PyTorch's restricted loader."""
    payload = load_weights_only_checkpoint(checkpoint_path, mmap=True)
    metadata = payload["metadata"]
    required = {"task_name", "use_ema", "shape_meta"}
    missing = sorted(required.difference(metadata))
    if missing:
        raise ValueError(f"Safe DP3 checkpoint metadata is missing {missing}")
    if metadata["task_name"] != str(expected_task_name):
        raise ValueError(
            "Safe DP3 checkpoint task metadata mismatch: "
            f"expected {expected_task_name!r}, got {metadata['task_name']!r}"
        )
    if not isinstance(metadata["use_ema"], bool):
        raise ValueError("Safe DP3 checkpoint metadata 'use_ema' must be a bool")
    if metadata["use_ema"] and payload.get("ema_model") is None:
        raise ValueError("Safe DP3 checkpoint requests EMA but ema_model is absent")
    del payload
    return metadata


def validate_safe_policy_shape_meta(metadata: dict, cfg) -> None:
    expected = OmegaConf.to_container(cfg.task.shape_meta, resolve=True)
    if metadata["shape_meta"] != expected:
        raise ValueError(
            "Safe DP3 checkpoint shape_meta mismatch: "
            f"expected {expected!r}, got {metadata['shape_meta']!r}"
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
    safe_checkpoint_path = resolve_safe_checkpoint_path(usr_args)
    safe_checkpoint_metadata = None
    if safe_checkpoint_path is not None:
        cfg.setting = str(usr_args.get("ckpt_setting", cfg.setting))
        safe_checkpoint_metadata = inspect_safe_policy_metadata(
            safe_checkpoint_path,
            expected_task_name=str(cfg.task.name),
        )
        cfg.training.use_ema = bool(safe_checkpoint_metadata["use_ema"])

    checkpoint_encoder_metadata = safe_checkpoint_metadata or {}
    requested_fusion_mode = str(
        usr_args.get("pointcloud_fusion_mode", "") or ""
    ).strip()
    checkpoint_fusion_mode = str(
        checkpoint_encoder_metadata.get("pointcloud_fusion_mode", "") or ""
    ).strip()
    if (
        requested_fusion_mode
        and checkpoint_fusion_mode
        and requested_fusion_mode != checkpoint_fusion_mode
    ):
        raise ValueError(
            "Requested pointcloud_fusion_mode does not match safe checkpoint: "
            f"{requested_fusion_mode!r} != {checkpoint_fusion_mode!r}"
        )
    pointcloud_fusion_mode = (
        requested_fusion_mode
        or checkpoint_fusion_mode
        or str(cfg.policy.get("pointcloud_fusion_mode", "concat"))
    )
    attention_heads = int(
        usr_args.get(
            "semantic_attention_heads",
            checkpoint_encoder_metadata.get(
                "semantic_attention_heads",
                cfg.policy.get("semantic_attention_heads", 4),
            ),
        )
    )
    part_pool_temperature = float(
        usr_args.get(
            "part_pool_temperature",
            checkpoint_encoder_metadata.get(
                "part_pool_temperature",
                cfg.policy.get("part_pool_temperature", 1.0),
            ),
        )
    )
    gate_trainable_default = checkpoint_encoder_metadata.get(
        "semantic_gate_trainable",
        cfg.policy.get("semantic_gate_trainable", True),
    )
    semantic_gate_trainable = parse_bool_arg(
        usr_args.get("semantic_gate_trainable", gate_trainable_default),
        default=bool(gate_trainable_default),
    )
    semantic_gate_scale = float(usr_args.get("semantic_gate_scale", 1.0))
    if not math.isfinite(semantic_gate_scale):
        raise ValueError("semantic_gate_scale must be finite")
    OmegaConf.update(
        cfg, "policy.pointcloud_fusion_mode", pointcloud_fusion_mode, force_add=True
    )
    OmegaConf.update(
        cfg, "policy.semantic_attention_heads", attention_heads, force_add=True
    )
    OmegaConf.update(
        cfg, "policy.part_pool_temperature", part_pool_temperature, force_add=True
    )
    OmegaConf.update(
        cfg, "policy.semantic_gate_trainable", semantic_gate_trainable, force_add=True
    )
    OmegaConf.update(
        cfg, "policy.semantic_gate_scale", semantic_gate_scale, force_add=True
    )

    use_actorseg_objpc = "objpc_actorseg" in usr_args["config_name"]
    use_sam2_objpc = "objpc_sam2" in usr_args["config_name"]
    use_ndf_pointwise = "ndf_pointwise" in usr_args["config_name"] or "ndf_relation_hybrid" in usr_args["config_name"]
    use_ndf_pointwise_hybrid = "ndf_pointwise_hybrid" in usr_args["config_name"]
    use_ndf_pointwise_interact = "ndf_pointwise_hybrid_interact" in usr_args["config_name"]
    use_ndf_pointwise_relation = "ndf_relation_hybrid" in usr_args["config_name"]
    use_ndf_relation_token = parse_bool_arg(usr_args.get("use_ndf_relation_token", False), default=False)
    use_se3_relation_token = parse_bool_arg(usr_args.get("use_se3_relation_token", False), default=False)
    use_semantic_pointwise = "semantic_pointwise" in usr_args["config_name"]
    use_semantic_pointwise_hybrid = "semantic_pointwise_hybrid" in usr_args["config_name"]
    use_utonia_pointwise = "utonia_pointwise" in usr_args["config_name"]
    use_utonia_pointwise_hybrid = "utonia_pointwise_hybrid" in usr_args["config_name"]
    use_object_pointcloud = (
        (("objpc" in usr_args["config_name"]) and not use_sam2_objpc and not use_actorseg_objpc)
        or use_ndf_pointwise
        or use_semantic_pointwise
        or use_utonia_pointwise
    )
    object_placeholders = parse_placeholder_list(usr_args.get("object_placeholders", "{A},{B}"))
    point_cloud_shape = list(cfg.task.shape_meta.obs.point_cloud.shape)
    point_cloud_dim = int(point_cloud_shape[1]) if len(point_cloud_shape) > 1 else 6
    point_cloud_num = int(usr_args.get("point_cloud_num", point_cloud_shape[0]))
    cfg.task.shape_meta.obs.point_cloud.shape = [point_cloud_num, point_cloud_dim]
    target_num_points = point_cloud_num
    ndf_feat_dim = 256
    ndf_point_num = int(usr_args.get("ndf_point_num", 128))
    ndf_relation_token_projection_dim = int(usr_args.get("ndf_relation_token_projection_dim", 0))
    ndf_relation_token_dim = 9 + (
        ndf_relation_token_projection_dim if ndf_relation_token_projection_dim > 0 else ndf_feat_dim
    )
    ndf_device = torch.device(usr_args.get("ndf_device", "cuda:0") if torch.cuda.is_available() else "cpu")
    ndf_model_specs = resolve_ndf_models(usr_args)
    dgcnn_placeholders = set(parse_placeholder_list(usr_args.get("ndf_dgcnn_placeholders", "")))
    se3_relation_route = str(usr_args.get("se3_relation_route", "oracle"))
    if se3_relation_route not in SUPPORTED_RELATION_ROUTES:
        raise ValueError(
            f"Unsupported SE(3) relation route {se3_relation_route!r}; "
            f"expected one of {SUPPORTED_RELATION_ROUTES}"
        )
    se3_relation_goal_table_path = str(usr_args.get("se3_relation_goal_table", "") or "")
    se3_relation_goal_table = None
    if (
        use_se3_relation_token
        and se3_relation_route not in OBSERVATION_RELATION_ROUTES
        and se3_relation_route != "oracle"
    ):
        if not se3_relation_goal_table_path:
            raise ValueError(
                f"se3_relation_goal_table is required for route={se3_relation_route}"
            )
        with open(se3_relation_goal_table_path, "r", encoding="utf-8") as handle:
            se3_relation_goal_table = json.load(handle)
    se3_relation_token_ablation = str(
        usr_args.get("se3_relation_token_ablation", "none")
    ).lower()
    if se3_relation_token_ablation not in SUPPORTED_TOKEN_ABLATIONS:
        raise ValueError(
            f"Unsupported SE(3) token ablation {se3_relation_token_ablation!r}; "
            f"expected one of {SUPPORTED_TOKEN_ABLATIONS}"
        )
    se3_geometry_estimator = None
    if (
        use_se3_relation_token
        and se3_relation_route in OBSERVATION_RELATION_ROUTES
        and se3_relation_token_ablation != "zero"
    ):
        if se3_relation_token_ablation == "constant_goal":
            geometry_spec_path = str(
                usr_args.get("se3_ablation_estimator_spec", "") or ""
            )
            if not geometry_spec_path:
                raise ValueError(
                    "se3_ablation_estimator_spec is required for "
                    "se3_relation_token_ablation=constant_goal"
                )
        else:
            geometry_spec_path = str(
                usr_args.get("se3_geometry_estimator_spec", "") or ""
            )
            if not geometry_spec_path:
                raise ValueError(
                    f"se3_geometry_estimator_spec is required for "
                    f"route={se3_relation_route}"
                )
        se3_geometry_estimator = create_estimator_from_spec(
            geometry_spec_path,
            device_override=str(usr_args.get("se3_geometry_device", ndf_device)),
        )
    semantic_point_num = int(usr_args.get("semantic_point_num", 128))
    semantic_input_color_mode = str(usr_args.get("semantic_input_color_mode", "debug_placeholder"))
    semantic_forward_mode = str(usr_args.get("semantic_forward_mode", "reference"))
    semantic_policy_output_mode = str(usr_args.get("semantic_policy_output_mode", "embedding"))
    semantic_policy_output_seed = int(usr_args.get("semantic_policy_output_seed", 20260807))
    if semantic_policy_output_mode not in {
        "embedding",
        "xyz",
        "part_prob",
        "uniform_prob",
        "shuffled_prob",
        "matched_utonia",
        "matched_utonia128",
    }:
        raise ValueError(
            f"Unsupported semantic_policy_output_mode: {semantic_policy_output_mode}"
        )
    use_matched_utonia = semantic_policy_output_mode in {
        "matched_utonia",
        "matched_utonia128",
    }
    use_projected_utonia = semantic_policy_output_mode == "matched_utonia128"
    if use_matched_utonia and not use_semantic_pointwise:
        raise ValueError(
            f"semantic_policy_output_mode={semantic_policy_output_mode} requires a semantic_pointwise config"
        )
    if use_matched_utonia and use_utonia_pointwise:
        raise ValueError(
            "semantic_policy_output_mode=matched_utonia cannot be combined with the legacy "
            "utonia_pointwise config route"
        )
    if use_matched_utonia and safe_checkpoint_path is None:
        raise ValueError(
            "semantic_policy_output_mode=matched_utonia requires safe_checkpoint_path"
        )
    semantic_device = torch.device(usr_args.get("semantic_device", "cuda:0") if torch.cuda.is_available() else "cpu")
    semantic_model_specs = resolve_semantic_models(usr_args)
    semantic_feat_dim_by_placeholder = {}
    utonia_point_num = int(usr_args.get("utonia_point_num", 128))
    utonia_device = torch.device(usr_args.get("utonia_device", "cuda:0") if torch.cuda.is_available() else "cpu")
    utonia_spec = resolve_utonia_spec(usr_args)
    utonia_feature_placeholders = resolve_utonia_feature_placeholders(usr_args, object_placeholders)
    utonia_feat_dim_by_placeholder = {}
    if safe_checkpoint_path is None:
        ckpt_file = resolve_checkpoint_path(usr_args, use_rgb=bool(usr_args.get("use_rgb", False)))
        checkpoint_use_ema = infer_checkpoint_use_ema(ckpt_file)
        if checkpoint_use_ema is not None:
            cfg.training.use_ema = bool(checkpoint_use_ema)
    actorseg_extract_fn = None
    actorseg_camera_names = []
    actorseg_segmentation_key = "actor_segmentation"
    sam2_tracker_factory = None
    sam2_extract_all_fn = None
    sam2_camera_names = []
    sam2_bbox_prompts_by_camera = {}
    sam2_bbox_prompts_persistent = False
    sam2_interactive_init = parse_bool_arg(usr_args.get("sam2_interactive_init", True), default=True)
    sam2_min_mask_points = int(usr_args.get("sam2_min_mask_points", 16))

    if use_actorseg_objpc:
        extract_placeholder_point_cloud_actorseg_online, parse_actorseg_camera_list = get_actorseg_utils()
        actorseg_extract_fn = extract_placeholder_point_cloud_actorseg_online
        actorseg_camera_names = parse_actorseg_camera_list(
            usr_args.get("actorseg_camera_names", "head_camera,front_camera")
        )

    if use_sam2_objpc:
        (
            build_sam2_tracker_factory,
            extract_placeholder_point_clouds_sam2_online,
            load_sam2_bbox_prompt_file,
            parse_sam2_camera_list,
        ) = get_sam2_utils()
        sam2_extract_all_fn = extract_placeholder_point_clouds_sam2_online
        sam2_camera_names = parse_sam2_camera_list(usr_args.get("sam2_camera_names", "head_camera,front_camera"))
        repo_root = pathlib.Path(parent_directory).parents[1]
        sam2_tracker_factory = build_sam2_tracker_factory(
            placeholders=object_placeholders,
            sam2_root=str(usr_args.get("sam2_root", repo_root / "include" / "SAM2_streaming")),
            config=str(usr_args.get("sam2_config", "sam2.1/sam2.1_hiera_l.yaml")),
            checkpoint=str(
                usr_args.get(
                    "sam2_checkpoint",
                    os.environ.get(
                        "SAM2_CHECKPOINT",
                        str(pathlib.Path.home() / "Datasets" / "sam2" / "sam2.1_hiera_large.pt"),
                    ),
                )
            ),
            device=str(usr_args.get("sam2_device", "cuda:0")),
            autocast_dtype=str(usr_args.get("sam2_autocast_dtype", "bfloat16")),
        )
        sam2_bbox_prompt_path = str(usr_args.get("sam2_bbox_prompt_path", "") or "").strip()
        if sam2_bbox_prompt_path:
            sam2_bbox_prompts_by_camera = load_sam2_bbox_prompt_file(
                sam2_bbox_prompt_path,
                camera_names=sam2_camera_names,
                placeholders=object_placeholders,
            )
            sam2_bbox_prompts_persistent = True
            sam2_interactive_init = False if "sam2_interactive_init" not in usr_args else sam2_interactive_init

    if use_semantic_pointwise:
        _, load_semantic_model = get_semantic_utils()
        semantic_models = {}
        for placeholder, checkpoint in semantic_model_specs.items():
            semantic_models[placeholder] = load_semantic_model(
                checkpoint=checkpoint,
                device=semantic_device,
            )
            if semantic_policy_output_mode == "embedding":
                output_feature_dim = int(semantic_models[placeholder]["sem_embedding_dim"])
            elif semantic_policy_output_mode in {"part_prob", "uniform_prob", "shuffled_prob"}:
                output_feature_dim = len(semantic_models[placeholder].get("canonical_label_names", []))
                if output_feature_dim <= 0:
                    raise ValueError(f"No semantic labels available for {placeholder}")
            else:
                output_feature_dim = 0
            semantic_feat_dim_by_placeholder[placeholder] = output_feature_dim
    else:
        semantic_models = {}

    if use_utonia_pointwise or use_matched_utonia:
        _, load_utonia_model = get_utonia_utils()
        utonia_artifacts = load_utonia_model(
            device=utonia_device,
            checkpoint=str(utonia_spec["checkpoint"]),
            repo_id=str(utonia_spec["repo_id"]),
            upcast_levels=int(utonia_spec["upcast_levels"]),

        )
        selected_utonia_placeholders = (
            [
                placeholder
                for placeholder in semantic_models
                if placeholder in utonia_feature_placeholders
            ]
            if use_matched_utonia
            else utonia_feature_placeholders
        )
        if use_matched_utonia and len(selected_utonia_placeholders) == 0:
            raise ValueError(
                "matched_utonia requires at least one semantic checkpoint selected for "
                "Utonia features"
            )
        if use_matched_utonia and set(selected_utonia_placeholders) != set(semantic_models):
            missing = sorted(set(semantic_models).difference(selected_utonia_placeholders))
            raise ValueError(
                "matched_utonia requires Utonia artifacts for every semantic placeholder; "
                f"missing {missing}"
            )
        utonia_models = {
            placeholder: utonia_artifacts
            for placeholder in selected_utonia_placeholders
        }
        utonia_feat_dim_by_placeholder = {
            placeholder: int(utonia_artifacts["feature_dim"])
            for placeholder in selected_utonia_placeholders
        }
    else:
        utonia_models = {}
    matched_utonia_projection = None
    if use_projected_utonia:
        projection_path = str(
            usr_args.get("matched_utonia_projection_path", "") or ""
        ).strip()
        if not projection_path:
            raise ValueError(
                "matched_utonia128 requires matched_utonia_projection_path"
            )
        raw_dims = sorted(set(utonia_feat_dim_by_placeholder.values()))
        if len(raw_dims) != 1:
            raise ValueError(
                f"matched_utonia128 requires one shared Utonia feature dim, got {raw_dims}"
            )
        projection_output_dim = int(
            usr_args.get("matched_utonia_projection_dim", 128)
        )
        load_projection_artifact, _ = get_utonia_projection_utils()
        matched_utonia_projection = load_projection_artifact(
            projection_path,
            expected_input_dim=raw_dims[0],
            expected_output_dim=projection_output_dim,
        )
        utonia_feat_dim_by_placeholder = {
            placeholder: projection_output_dim
            for placeholder in utonia_feat_dim_by_placeholder
        }

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
        if use_ndf_pointwise_relation:
            for support_placeholder in object_placeholders:
                checkpoint = ndf_model_specs.get(support_placeholder)
                if checkpoint in {None, "", "none"}:
                    continue
                for query_placeholder in object_placeholders:
                    if query_placeholder == support_placeholder:
                        continue
                    pointcloud_key = placeholder_relation_pointcloud_key(
                        query_placeholder,
                        support_placeholder,
                    )
                    cfg.task.shape_meta.obs[pointcloud_key] = {
                        "shape": [ndf_point_num, 3 + ndf_feat_dim],
                        "type": "point_cloud",
                    }
        if use_ndf_relation_token:
            for support_placeholder in object_placeholders:
                checkpoint = ndf_model_specs.get(support_placeholder)
                if checkpoint in {None, "", "none"}:
                    continue
                for query_placeholder in object_placeholders:
                    if query_placeholder == support_placeholder:
                        continue
                    token_key = placeholder_relation_token_key(query_placeholder, support_placeholder)
                    cfg.task.shape_meta.obs[token_key] = {
                        "shape": [ndf_relation_token_dim],
                        "type": "low_dim",
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
    if use_se3_relation_token:
        cfg.task.shape_meta.obs[RELATION_TOKEN_KEY] = {
            "shape": [RELATION_TOKEN_DIM],
            "type": "low_dim",
        }
    if use_semantic_pointwise:
        for placeholder in object_placeholders:
            artifacts = semantic_models.get(placeholder)
            if artifacts is None:
                continue
            pointcloud_key = (
                placeholder_utonia_pointcloud_key(placeholder)
                if use_matched_utonia
                else placeholder_semantic_pointcloud_key(placeholder)
            )
            output_feat_dim = (
                utonia_feat_dim_by_placeholder[placeholder]
                if use_matched_utonia
                else semantic_feat_dim_by_placeholder[placeholder]
            )
            cfg.task.shape_meta.obs[pointcloud_key] = {
                "shape": [semantic_point_num, 3 + output_feat_dim],
                "type": "point_cloud",
            }
    if use_utonia_pointwise:
        for placeholder in object_placeholders:
            artifacts = utonia_models.get(placeholder)
            if artifacts is None:
                continue
            pointcloud_key = placeholder_utonia_pointcloud_key(placeholder)
            cfg.task.shape_meta.obs[pointcloud_key] = {
                "shape": [utonia_point_num, 3 + int(artifacts["feature_dim"])],
                "type": "point_cloud",
            }
    OmegaConf.set_struct(cfg, True)
    if safe_checkpoint_metadata is not None:
        validate_safe_policy_shape_meta(safe_checkpoint_metadata, cfg)


    DP3_Model = DP3(cfg, usr_args)
    DP3_Model.use_actorseg_objpc = use_actorseg_objpc
    DP3_Model.use_sam2_objpc = use_sam2_objpc
    DP3_Model.use_object_pointcloud = use_object_pointcloud
    DP3_Model.use_ndf_pointwise = use_ndf_pointwise
    DP3_Model.use_ndf_pointwise_hybrid = use_ndf_pointwise_hybrid
    DP3_Model.use_ndf_pointwise_interact = use_ndf_pointwise_interact
    DP3_Model.use_ndf_pointwise_relation = use_ndf_pointwise_relation
    DP3_Model.use_ndf_relation_token = use_ndf_relation_token
    DP3_Model.use_se3_relation_token = use_se3_relation_token
    DP3_Model.se3_relation_route = se3_relation_route
    DP3_Model.se3_relation_goal_table = se3_relation_goal_table
    DP3_Model.se3_geometry_estimator = se3_geometry_estimator
    DP3_Model.se3_relation_token_ablation = se3_relation_token_ablation
    DP3_Model.use_semantic_pointwise = use_semantic_pointwise
    DP3_Model.use_semantic_pointwise_hybrid = use_semantic_pointwise_hybrid
    DP3_Model.use_utonia_pointwise = use_utonia_pointwise
    DP3_Model.use_utonia_pointwise_hybrid = use_utonia_pointwise_hybrid
    DP3_Model.object_placeholders = object_placeholders
    DP3_Model.target_num_points = target_num_points
    DP3_Model.actorseg_extract_fn = actorseg_extract_fn
    DP3_Model.actorseg_camera_names = actorseg_camera_names
    DP3_Model.actorseg_segmentation_key = actorseg_segmentation_key
    DP3_Model.actorseg_actor_ids_by_placeholder = {
        placeholder: []
        for placeholder in object_placeholders
    }
    DP3_Model.sam2_tracker_factory = sam2_tracker_factory
    DP3_Model.sam2_extract_all_fn = sam2_extract_all_fn
    DP3_Model.sam2_camera_names = sam2_camera_names
    DP3_Model.sam2_bbox_prompts_by_camera = sam2_bbox_prompts_by_camera
    DP3_Model.sam2_bbox_prompts_persistent = sam2_bbox_prompts_persistent
    DP3_Model.sam2_interactive_init = sam2_interactive_init
    DP3_Model.sam2_min_mask_points = sam2_min_mask_points
    DP3_Model.sam2_tracking_state_by_camera = {}
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
    DP3_Model.ndf_relation_point_num_by_pair = {
        (query_placeholder, support_placeholder): ndf_point_num
        for support_placeholder in object_placeholders
        if support_placeholder in ndf_model_specs
        for query_placeholder in object_placeholders
        if query_placeholder != support_placeholder
    }
    DP3_Model.ndf_relation_token_dim = ndf_relation_token_dim
    DP3_Model.ndf_relation_token_point_num_by_pair = {
        (query_placeholder, support_placeholder): ndf_point_num
        for support_placeholder in object_placeholders
        if support_placeholder in ndf_model_specs
        for query_placeholder in object_placeholders
        if query_placeholder != support_placeholder
    }
    DP3_Model.ndf_relation_token_gate_near_ratio = float(
        usr_args.get("ndf_relation_token_gate_near_ratio", 0.35)
    )
    DP3_Model.ndf_relation_token_gate_far_ratio = float(
        usr_args.get("ndf_relation_token_gate_far_ratio", 0.75)
    )
    DP3_Model.ndf_relation_token_valid_query_radius = float(
        usr_args.get("ndf_relation_token_valid_query_radius", 0.75)
    )
    DP3_Model.ndf_relation_token_valid_fraction_min = float(
        usr_args.get("ndf_relation_token_valid_fraction_min", 0.25)
    )
    DP3_Model.ndf_relation_token_valid_fraction_max = float(
        usr_args.get("ndf_relation_token_valid_fraction_max", 0.65)
    )
    DP3_Model.ndf_relation_token_projection_dim = ndf_relation_token_projection_dim
    DP3_Model.ndf_relation_token_projection_seed = int(
        usr_args.get("ndf_relation_token_projection_seed", 0)
    )
    DP3_Model.ndf_relation_token_gate_geometry = parse_bool_arg(
        usr_args.get("ndf_relation_token_gate_geometry", False),
        default=False,
    )
    DP3_Model.ndf_device = ndf_device
    DP3_Model.ndf_models = {}
    load_ndf_model = None
    if len(ndf_model_specs) > 0:
        _, _, _, _, load_ndf_model = get_ndf_utils()
    for placeholder, checkpoint in ndf_model_specs.items():
        DP3_Model.ndf_models[placeholder] = load_ndf_model(
            checkpoint=checkpoint,
            dgcnn=placeholder in dgcnn_placeholders,
            device=ndf_device,
            latent_dim=ndf_feat_dim,
        )
    DP3_Model.semantic_device = semantic_device
    DP3_Model.semantic_input_color_mode = semantic_input_color_mode
    DP3_Model.semantic_forward_mode = semantic_forward_mode
    DP3_Model.semantic_policy_output_mode = semantic_policy_output_mode
    DP3_Model.semantic_policy_output_seed = semantic_policy_output_seed
    DP3_Model.semantic_policy_frame_index = 0
    DP3_Model.matched_utonia_color_mode = str(
        usr_args.get("matched_utonia_color_mode", "debug_placeholder")
    )
    DP3_Model.matched_utonia_normal_mode = str(
        usr_args.get("matched_utonia_normal_mode", "fallback")
    )
    DP3_Model.matched_utonia_projection = matched_utonia_projection
    DP3_Model.semantic_point_num_by_placeholder = {
        placeholder: semantic_point_num
        for placeholder in object_placeholders
        if placeholder in semantic_models
    }
    DP3_Model.semantic_feat_dim_by_placeholder = semantic_feat_dim_by_placeholder
    DP3_Model.semantic_feat_dim = max(semantic_feat_dim_by_placeholder.values(), default=128)
    DP3_Model.semantic_models = semantic_models
    DP3_Model.utonia_device = utonia_device
    DP3_Model.utonia_point_num_by_placeholder = {
        placeholder: utonia_point_num
        for placeholder in object_placeholders
        if placeholder in utonia_models
    }
    DP3_Model.utonia_feat_dim_by_placeholder = utonia_feat_dim_by_placeholder
    DP3_Model.utonia_feat_dim = max(utonia_feat_dim_by_placeholder.values(), default=96)
    DP3_Model.utonia_models = utonia_models
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

    retain_tail_observations = bool(
        getattr(TASK_ENV, "eval_mode", False)
        and getattr(TASK_ENV, "fast_eval", {}).get("retain_action_tail_observations", False)
    )
    retained_observation_count = max(
        1, int(getattr(model.env_runner, "n_obs_steps", len(actions)))
    )
    first_retained_action = max(0, len(actions) - retained_observation_count)

    for action_index, action in enumerate(actions):  # Execute each step of the action
        TASK_ENV.take_action(action)
        if TASK_ENV.eval_success or TASK_ENV.take_action_cnt >= TASK_ENV.step_lim:
            break
        if retain_tail_observations and action_index < first_retained_action:
            # These frames cannot enter the next n_obs_steps window. Keep the
            # explicit frame counter aligned for frame-dependent controls.
            if hasattr(model, "semantic_policy_frame_index"):
                model.semantic_policy_frame_index += 1
            continue
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
    if hasattr(model, "sam2_tracking_state_by_camera"):
        model.sam2_tracking_state_by_camera = {}
    if hasattr(model, "sam2_bbox_prompts_by_camera") and not bool(getattr(model, "sam2_bbox_prompts_persistent", False)):
        model.sam2_bbox_prompts_by_camera = {}
    if hasattr(model, "semantic_policy_frame_index"):
        model.semantic_policy_frame_index = 0
