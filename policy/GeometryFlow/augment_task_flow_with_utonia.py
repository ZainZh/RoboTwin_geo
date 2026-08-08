#!/usr/bin/env python3
"""Attach frozen UTONIA descriptors to both objects in a task-flow dataset.

The output point tensors contain ``[XYZ, descriptor]``.  RGB is used by UTONIA
during extraction but deliberately omitted from the policy input, so the
shared descriptor adapter has one unambiguous role and an all-zero gate is an
exact XYZ-only fallback.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from .extract_utonia_features import (
    build_transform,
    estimate_outward_normals,
    move_tensors,
    normalize_object_coordinates,
    upcast_to_input,
)


STAGE_SLICES = {
    "all": slice(0, 1386),
    "s0": slice(0, 54),
    "s1": slice(54, 162),
    "s2": slice(162, 378),
    "s3": slice(378, 810),
    "s4": slice(810, 1386),
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--utonia-root", type=Path, required=True)
    result.add_argument("--checkpoint", default="utonia")
    result.add_argument("--repo-id", default="Pointcept/Utonia")
    result.add_argument("--download-root", type=Path)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--stage", choices=tuple(STAGE_SLICES), default="s0")
    result.add_argument("--grid-size", type=float, default=0.01)
    result.add_argument("--normal-neighbors", type=int, default=16)
    result.add_argument("--color-mode", choices=("rgb", "zero"), default="rgb")
    result.add_argument("--no-l2-normalize", action="store_true")
    result.add_argument(
        "--shuffle-within-cloud",
        action="store_true",
        help="Negative control: randomly detach descriptors from their XYZ points.",
    )
    result.add_argument("--seed", type=int, default=73)
    result.add_argument("--overwrite", action="store_true")
    return result


def encode_cloud(
    cloud: np.ndarray,
    *,
    model,
    transform,
    utonia,
    device: torch.device,
    normal_neighbors: int,
    color_mode: str,
) -> np.ndarray:
    world_xyz = np.asarray(cloud[:, :3], dtype=np.float32)
    color = np.clip(np.asarray(cloud[:, 3:6], dtype=np.float32), 0.0, 1.0)
    if color_mode == "zero":
        color = np.zeros_like(color)
    point = transform(
        {
            "coord": normalize_object_coordinates(world_xyz),
            "color": color * 255.0,
            "normal": estimate_outward_normals(
                world_xyz, neighbors=int(normal_neighbors)
            ),
        }
    )
    point = move_tensors(utonia.data.collate_fn([point]), device)
    with torch.inference_mode():
        descriptor = upcast_to_input(model(point), levels=4)
    value = descriptor.detach().float().cpu().numpy().astype(np.float32)
    if value.shape[0] != world_xyz.shape[0]:
        raise RuntimeError(
            f"UTONIA returned {value.shape[0]} descriptors for {len(world_xyz)} points"
        )
    if not np.all(np.isfinite(value)):
        raise RuntimeError("UTONIA produced non-finite descriptors")
    return value


def main() -> None:
    args = parser().parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    sys.path.insert(0, str(args.utonia_root.resolve()))
    import utonia  # type: ignore

    utonia.utils.set_seed(int(args.seed))
    model = utonia.load(
        args.checkpoint,
        repo_id=args.repo_id,
        download_root=(
            str(args.download_root.resolve()) if args.download_root else None
        ),
        custom_config={
            "enable_flash": False,
            "enc_patch_size": [1024 for _ in range(5)],
        },
    )
    device = torch.device(args.device)
    model = model.to(device).eval()
    transform = build_transform(utonia, float(args.grid_size))

    with np.load(args.dataset, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    for key in ("points_a", "points_b"):
        if key not in payload:
            raise KeyError(f"task-flow dataset is missing {key}")
        if payload[key].ndim != 3 or payload[key].shape[-1] < 6:
            raise ValueError(f"{key} must be [samples, points, XYZRGB]")

    selected = STAGE_SLICES[args.stage]
    rng = np.random.default_rng(int(args.seed))
    diagnostics: dict[str, dict[str, float | int]] = {}
    for key in ("points_a", "points_b"):
        output = []
        for index, cloud in enumerate(np.asarray(payload[key], dtype=np.float32)):
            descriptor = encode_cloud(
                cloud,
                model=model,
                transform=transform,
                utonia=utonia,
                device=device,
                normal_neighbors=int(args.normal_neighbors),
                color_mode=args.color_mode,
            )[:, selected]
            if not args.no_l2_normalize:
                descriptor = descriptor / np.maximum(
                    np.linalg.norm(descriptor, axis=-1, keepdims=True), 1e-8
                )
            if args.shuffle_within_cloud:
                descriptor = descriptor[rng.permutation(len(descriptor))]
            output.append(np.concatenate((cloud[:, :3], descriptor), axis=-1))
            if (index + 1) % 50 == 0 or index + 1 == len(payload[key]):
                print(
                    json.dumps({"key": key, "completed": index + 1}),
                    flush=True,
                )
        value = np.asarray(output, dtype=np.float32)
        payload[key] = value
        diagnostics[key] = {
            "samples": int(value.shape[0]),
            "points": int(value.shape[1]),
            "point_channels": int(value.shape[2]),
            "descriptor_abs_mean": float(np.abs(value[..., 3:]).mean()),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    metadata = {
        "schema_version": 1,
        "dataset": str(args.dataset.resolve()),
        "output": str(args.output.resolve()),
        "encoder": "Pointcept/Utonia",
        "checkpoint": args.checkpoint,
        "stage": args.stage,
        "descriptor_dim": int(payload["points_a"].shape[-1] - 3),
        "l2_normalized": not args.no_l2_normalize,
        "color_mode": args.color_mode,
        "shuffle_within_cloud": bool(args.shuffle_within_cloud),
        "diagnostics": diagnostics,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
