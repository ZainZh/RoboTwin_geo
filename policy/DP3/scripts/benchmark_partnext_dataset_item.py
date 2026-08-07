#!/usr/bin/env python3
"""Measure matched PartNext dataset item latency for environment audits."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--utonia-root", type=Path, required=True)
    parser.add_argument("--alias-config", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260805)
    return parser.parse_args()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.utonia_root.resolve()))
    sys.path.insert(0, str(args.release_root.resolve()))

    from my_datasets.partnext_canonical_field import (
        _concatenate_meshes,
        _load_geometry_meshes,
        build_partnext_canonical_label_space,
    )
    from my_datasets.partnext_universal_field import PartNextUniversalFieldDataset

    labels = build_partnext_canonical_label_space(
        dataset_root=args.dataset_root,
        categories=["Hammer"],
        label_level="config",
        alias_config_path=args.alias_config,
        ignore_labels=("Other",),
    )
    dataset = PartNextUniversalFieldDataset(
        dataset_root=args.dataset_root,
        split="train",
        categories=["Hammer"],
        num_support_points=5000,
        num_query_surface_points=2048,
        num_query_occ_points=1536,
        num_query_probe_points=1024,
        balanced_sampling_ratio=0.0,
        query_sampling_ratio=0.75,
        val_ratio=0.1,
        test_ratio=0.0,
        split_seed=42,
        label_level="config",
        alias_config_path=args.alias_config,
        canonical_label_names=labels,
        rotation_mode="so3",
        jitter_std=0.005,
        second_view=True,
    )
    index = args.index % len(dataset)
    record = dataset.records[index]
    mesh = _concatenate_meshes(_load_geometry_meshes(record.mesh_path))

    durations: list[float] = []
    for repeat in range(args.repeats):
        seed = args.seed + repeat
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        started = time.perf_counter()
        sample = dataset[index]
        durations.append(time.perf_counter() - started)
        if int(sample["query_surface_points_view1"].shape[0]) != 2048:
            raise RuntimeError("unexpected query surface shape")

    cgroup_cpu_max = None
    cpu_max_path = Path("/sys/fs/cgroup/cpu.max")
    if cpu_max_path.is_file():
        cgroup_cpu_max = cpu_max_path.read_text(encoding="utf-8").strip()
    payload = {
        "model_id": record.model_id,
        "dataset_length": len(dataset),
        "index": index,
        "repeats": args.repeats,
        "durations_seconds": durations,
        "first_seconds": durations[0],
        "steady_mean_seconds": float(np.mean(durations[1:])),
        "steady_std_seconds": float(np.std(durations[1:], ddof=1)),
        "ray_backend": f"{type(mesh.ray).__module__}.{type(mesh.ray).__name__}",
        "cpu_affinity_count": len(os.sched_getaffinity(0)),
        "cgroup_cpu_max": cgroup_cpu_max,
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "trimesh": package_version("trimesh"),
            "rtree": package_version("rtree"),
            "embreex": package_version("embreex"),
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
