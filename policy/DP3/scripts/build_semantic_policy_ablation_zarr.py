#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from semantic_field_eval_utils import sha256_file


def stable_softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    shifted = array - np.max(array, axis=axis, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.maximum(exponent.sum(axis=axis, keepdims=True), 1e-12)


def transform_semantic_cloud(
    cloud: np.ndarray,
    *,
    mode: str,
    logit_weight: np.ndarray | None = None,
    logit_bias: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(cloud, dtype=np.float32)
    if values.ndim < 2 or values.shape[-1] < 3:
        raise ValueError(f"semantic cloud must end in at least 3 channels, got {values.shape}")
    xyz = values[..., :3]
    if mode == "xyz":
        return xyz.copy()
    if mode != "part_prob":
        raise ValueError(f"unsupported semantic ablation mode: {mode}")
    if logit_weight is None or logit_bias is None:
        raise ValueError("part_prob mode requires logit weight and bias")

    embeddings = values[..., 3:]
    weight = np.asarray(logit_weight, dtype=np.float32)
    bias = np.asarray(logit_bias, dtype=np.float32)
    if weight.ndim != 2 or bias.shape != (weight.shape[0],):
        raise ValueError(
            f"invalid logit head shapes: weight={weight.shape}, bias={bias.shape}"
        )
    if embeddings.shape[-1] != weight.shape[1]:
        raise ValueError(
            f"embedding dim {embeddings.shape[-1]} does not match logit head "
            f"input dim {weight.shape[1]}"
        )
    logits = embeddings @ weight.T + bias
    probabilities = stable_softmax(logits, axis=-1)
    return np.concatenate([xyz, probabilities], axis=-1).astype(np.float32)


def _safe_load_logit_head(checkpoint_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    import torch

    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError as error:
        raise RuntimeError(
            "PyTorch with weights_only checkpoint loading is required"
        ) from error
    projector = state["projector_state_dict"]
    decoder = projector["semantic_decoder"]
    weight_key = next(
        (
            key
            for key in (
                "logit_head.weight",
                "module.logit_head.weight",
            )
            if key in decoder
        ),
        None,
    )
    if weight_key is None:
        raise KeyError("semantic decoder checkpoint has no logit_head.weight")
    bias_key = weight_key.replace("weight", "bias")
    if bias_key not in decoder:
        raise KeyError("semantic decoder checkpoint has no matching logit_head.bias")
    weight = decoder[weight_key].detach().float().cpu().numpy()
    bias = decoder[bias_key].detach().float().cpu().numpy()
    labels = [str(value) for value in state["canonical_label_names"]]
    if len(labels) != weight.shape[0]:
        raise RuntimeError(
            f"label count {len(labels)} does not match logit rows {weight.shape[0]}"
        )
    return weight, bias, labels


def _infer_sidecar(path: Path) -> Path:
    return path.with_name(f"{path.stem}_meta.json")


def _copy_source_zarr(source: Path, destination: Path) -> None:
    try:
        shutil.copytree(source, destination, copy_function=_copy_reflink_or_regular)
    except Exception:
        if destination.exists():
            shutil.rmtree(destination)
        raise


def _copy_reflink_or_regular(source: str, destination: str) -> str:
    # copy2 is portable; filesystems with transparent CoW may deduplicate later.
    return shutil.copy2(source, destination)


def _replace_array(
    *,
    root,
    key: str,
    mode: str,
    weight: np.ndarray | None,
    bias: np.ndarray | None,
    batch_frames: int,
) -> dict:
    source_array = root[f"data/{key}"]
    source_shape = tuple(int(value) for value in source_array.shape)
    source_dtype = str(source_array.dtype)
    source_chunks = tuple(int(value) for value in source_array.chunks)
    sample = transform_semantic_cloud(
        np.asarray(source_array[:1], dtype=np.float32),
        mode=mode,
        logit_weight=weight,
        logit_bias=bias,
    )
    output_channels = int(sample.shape[-1])
    output_shape = (*source_shape[:-1], output_channels)
    output_chunks = (*source_chunks[:-1], output_channels)
    compressor = source_array.compressor

    temporary_key = f"{key}__ablation_tmp"
    if temporary_key in root["data"]:
        del root["data"][temporary_key]
    output = root["data"].create_dataset(
        temporary_key,
        shape=output_shape,
        chunks=output_chunks,
        dtype=np.float32,
        compressor=compressor,
        overwrite=False,
    )
    frame_count = source_shape[0]
    for start in range(0, frame_count, int(batch_frames)):
        end = min(frame_count, start + int(batch_frames))
        transformed = transform_semantic_cloud(
            np.asarray(source_array[start:end], dtype=np.float32),
            mode=mode,
            logit_weight=weight,
            logit_bias=bias,
        )
        output[start:end] = transformed
    source_xyz = np.asarray(source_array[..., :3])
    output_xyz = np.asarray(output[..., :3])
    if not np.array_equal(source_xyz, output_xyz):
        raise RuntimeError(f"query xyz changed while deriving {key}")
    if mode == "part_prob":
        probability_sum = np.asarray(output[..., 3:]).sum(axis=-1)
        if not np.allclose(probability_sum, 1.0, atol=1e-5):
            raise RuntimeError(f"part probabilities for {key} do not sum to one")

    del root["data"][key]
    root["data"].move(temporary_key, key)
    return {
        "key": key,
        "source_shape": list(source_shape),
        "output_shape": list(output_shape),
        "source_dtype": source_dtype,
        "output_dtype": "float32",
        "source_chunks": list(source_chunks),
        "output_chunks": list(output_chunks),
        "xyz_bitwise_identical": True,
    }


def build_ablation_zarr(args: argparse.Namespace) -> dict:
    import zarr

    source = args.source_zarr.expanduser().resolve()
    destination = args.output_zarr.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"source zarr does not exist: {source}")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite output path: {destination}")
    if args.batch_frames <= 0:
        raise ValueError("batch_frames must be positive")

    source_sidecar = args.source_meta or _infer_sidecar(source)
    source_meta = None
    if source_sidecar.is_file():
        source_meta = json.loads(source_sidecar.read_text(encoding="utf-8"))

    weight = None
    bias = None
    labels = None
    checkpoint_manifest = None
    if args.mode == "part_prob":
        if args.semantic_checkpoint is None:
            raise ValueError("part_prob mode requires --semantic-checkpoint")
        checkpoint = args.semantic_checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"semantic checkpoint does not exist: {checkpoint}")
        weight, bias, labels = _safe_load_logit_head(checkpoint)
        checkpoint_manifest = {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
            "canonical_label_names": labels,
            "logit_weight_shape": list(weight.shape),
        }

    _copy_source_zarr(source, destination)
    try:
        root = zarr.open(str(destination), mode="a")
        if "data" not in root:
            raise KeyError("source zarr has no data group")
        keys = args.semantic_keys or sorted(
            key for key in root["data"].array_keys() if key.startswith("semantic_point_cloud_")
        )
        if not keys:
            raise RuntimeError("no semantic point-cloud arrays were selected")
        missing = [key for key in keys if key not in root["data"]]
        if missing:
            raise KeyError(f"semantic arrays missing from source zarr: {missing}")
        arrays = [
            _replace_array(
                root=root,
                key=key,
                mode=args.mode,
                weight=weight,
                bias=bias,
                batch_frames=args.batch_frames,
            )
            for key in keys
        ]
    except Exception:
        shutil.rmtree(destination)
        raise

    output_meta = copy_meta = dict(source_meta or {})
    output_meta.update(
        {
            "output_zarr": str(destination),
            "derived_at_utc": datetime.now(timezone.utc).isoformat(),
            "representation_route": args.mode,
            "source_zarr": str(source),
            "source_meta": str(source_sidecar) if source_sidecar.is_file() else None,
            "source_meta_sha256": (
                sha256_file(source_sidecar) if source_sidecar.is_file() else None
            ),
            "semantic_checkpoint": checkpoint_manifest,
            "canonical_label_names": labels,
            "semantic_feat_dim": (
                0 if args.mode == "xyz" else int(len(labels or []))
            ),
            "derived_arrays": arrays,
            "matched_invariants": {
                "episode_ends_unchanged": True,
                "state_unchanged": True,
                "action_unchanged": True,
                "primary_point_cloud_unchanged": True,
                "semantic_query_xyz_bitwise_identical": True,
            },
        }
    )
    output_sidecar = args.output_meta or _infer_sidecar(destination)
    output_sidecar.write_text(
        json.dumps(output_meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    manifest = {
        "source_zarr": str(source),
        "output_zarr": str(destination),
        "output_meta": str(output_sidecar.resolve()),
        "mode": args.mode,
        "arrays": arrays,
        "semantic_checkpoint": checkpoint_manifest,
        "matched_invariants": output_meta["matched_invariants"],
    }
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Derive matched XYZ-only or xyz+part-probability DP3 zarr data "
            "from an existing semantic point-cloud zarr."
        )
    )
    parser.add_argument("--source-zarr", type=Path, required=True)
    parser.add_argument("--output-zarr", type=Path, required=True)
    parser.add_argument("--source-meta", type=Path, default=None)
    parser.add_argument("--output-meta", type=Path, default=None)
    parser.add_argument("--mode", choices=["xyz", "part_prob"], required=True)
    parser.add_argument("--semantic-checkpoint", type=Path, default=None)
    parser.add_argument("--semantic-keys", nargs="*", default=None)
    parser.add_argument("--batch-frames", type=int, default=100)
    return parser


def main() -> None:
    build_ablation_zarr(build_argparser().parse_args())


if __name__ == "__main__":
    main()
