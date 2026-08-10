from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


_ADAPTER_PREFIXES = (
    "obs_encoder.semantic_gates.",
    "obs_encoder.semantic_extractors.",
    "obs_encoder.semantic_input_projections.",
    "obs_encoder.semantic_attentions.",
    "obs_encoder.semantic_query_projections.",
    "obs_encoder.semantic_output_mlps.",
    "obs_encoder.semantic_task_queries.",
    "obs_encoder.part_pool_mlps.",
)


def vanilla_to_fusion_key(key: str) -> str:
    key = str(key)
    if key.startswith("obs_encoder."):
        return "obs_encoder.base_encoder." + key[len("obs_encoder.") :]
    return key


def validate_vanilla_warmstart_metadata(
    metadata: Mapping[str, Any],
    *,
    target_shape_meta: Mapping[str, Any],
    expected_raw_task_name: str,
    require_ema: bool,
) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise ValueError("Warm-start checkpoint metadata must be a mapping")
    task_name = str(metadata.get("task_name", ""))
    expected_prefix = f"{expected_raw_task_name}-"
    if not task_name.startswith(expected_prefix):
        raise ValueError(
            "Warm-start checkpoint task mismatch: "
            f"expected prefix {expected_prefix!r}, got {task_name!r}"
        )
    fusion_mode = str(metadata.get("pointcloud_fusion_mode", "concat"))
    if fusion_mode != "concat":
        raise ValueError(
            "Warm-start source must be Strict Vanilla/concat, got "
            f"pointcloud_fusion_mode={fusion_mode!r}"
        )
    if require_ema and metadata.get("use_ema") is not True:
        raise ValueError(
            "warmstart_use_ema_weights=true requires a checkpoint with use_ema=true"
        )

    source_shape_meta = metadata.get("shape_meta")
    if not isinstance(source_shape_meta, Mapping):
        raise ValueError("Warm-start checkpoint metadata is missing shape_meta")
    source_obs = source_shape_meta.get("obs")
    target_obs = target_shape_meta.get("obs")
    if not isinstance(source_obs, Mapping) or not isinstance(target_obs, Mapping):
        raise ValueError("Warm-start source and target must both define obs shape_meta")

    required_obs = ("point_cloud", "agent_pos")
    for key in required_obs:
        if key not in source_obs or key not in target_obs:
            raise ValueError(f"Warm-start source/target is missing required obs key {key!r}")
        if dict(source_obs[key]) != dict(target_obs[key]):
            raise ValueError(
                f"Warm-start observation shape mismatch for {key!r}: "
                f"{source_obs[key]!r} != {target_obs[key]!r}"
            )
    source_secondary = [
        key
        for key, value in source_obs.items()
        if key not in required_obs and value.get("type") == "point_cloud"
    ]
    if source_secondary:
        raise ValueError(
            "Warm-start source is not Strict Vanilla; unexpected secondary point "
            f"cloud keys: {sorted(source_secondary)}"
        )
    if dict(source_shape_meta.get("action", {})) != dict(
        target_shape_meta.get("action", {})
    ):
        raise ValueError(
            "Warm-start action shape mismatch: "
            f"{source_shape_meta.get('action')!r} != "
            f"{target_shape_meta.get('action')!r}"
        )
    return {
        "source_task_name": task_name,
        "source_fusion_mode": fusion_mode,
        "source_epoch": metadata.get("epoch"),
        "source_seed": metadata.get("seed"),
    }


def load_shared_vanilla_state(
    target_model: torch.nn.Module,
    source_state: Mapping[str, torch.Tensor],
) -> dict[str, int]:
    if not isinstance(source_state, Mapping):
        raise ValueError("Warm-start source state_dict must be a mapping")
    target_state = target_model.state_dict()
    mapped_state: dict[str, torch.Tensor] = {}
    missing: list[tuple[str, str]] = []
    skipped_normalizer: list[str] = []
    shape_mismatch: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []

    for source_key, source_value in source_state.items():
        target_key = vanilla_to_fusion_key(source_key)
        if target_key not in target_state:
            if str(source_key).startswith("normalizer."):
                skipped_normalizer.append(str(source_key))
                continue
            missing.append((str(source_key), target_key))
            continue
        if not isinstance(source_value, torch.Tensor):
            raise ValueError(f"Warm-start state entry {source_key!r} is not a tensor")
        expected_shape = tuple(target_state[target_key].shape)
        actual_shape = tuple(source_value.shape)
        if expected_shape != actual_shape:
            shape_mismatch.append((target_key, actual_shape, expected_shape))
            continue
        mapped_state[target_key] = source_value

    if missing:
        raise ValueError(
            "Warm-start source has keys absent from fusion target: "
            f"{missing[:5]}"
        )
    if shape_mismatch:
        raise ValueError(
            "Warm-start shared tensor shape mismatch: "
            f"{shape_mismatch[:5]}"
        )
    accounted = len(mapped_state) + len(skipped_normalizer)
    if accounted != len(source_state):
        raise RuntimeError(
            "Warm-start accounted for "
            f"{accounted} of {len(source_state)} tensors"
        )

    target_model.load_state_dict(mapped_state, strict=False)
    return {
        "source_tensor_count": len(source_state),
        "loaded_tensor_count": len(mapped_state),
        "skipped_normalizer_tensor_count": len(skipped_normalizer),
        "loaded_numel": int(sum(value.numel() for value in mapped_state.values())),
    }


def freeze_vanilla_backbone_for_adapter(
    model: torch.nn.Module,
) -> dict[str, int]:
    trainable_names: list[str] = []
    frozen_names: list[str] = []
    trainable_numel = 0
    frozen_numel = 0
    for name, parameter in model.named_parameters():
        trainable = name.startswith(_ADAPTER_PREFIXES)
        parameter.requires_grad_(trainable)
        if trainable:
            trainable_names.append(name)
            trainable_numel += parameter.numel()
        else:
            frozen_names.append(name)
            frozen_numel += parameter.numel()
    if not trainable_names:
        raise ValueError(
            "freeze_vanilla_backbone_for_adapter found no semantic adapter parameters"
        )
    if not any(name.startswith("obs_encoder.semantic_gates.") for name in trainable_names):
        raise ValueError("Semantic fusion model has no trainable semantic gate")
    return {
        "trainable_tensor_count": len(trainable_names),
        "frozen_tensor_count": len(frozen_names),
        "trainable_numel": int(trainable_numel),
        "frozen_numel": int(frozen_numel),
    }
