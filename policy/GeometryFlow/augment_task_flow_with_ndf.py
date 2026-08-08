#!/usr/bin/env python3
"""Attach camera-only pointwise NDF descriptors to task-flow observations.

NDF is evaluated directly on every current A point cloud.  Simulator object
poses are never used to construct policy inputs.  The generated clean, zero,
point-shuffled, and episode-shuffled datasets share exactly the same shapes so
they can be compared with a capacity-matched gated descriptor adapter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .ndf_adapter import NdfPointwiseAdapter


CORRUPTIONS = (
    "clean",
    "zero",
    "point_shuffled",
    "episode_shuffled",
    "sign_flipped",
    "isotropic_axis",
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", type=Path, required=True)
    result.add_argument("--ndf-checkpoint", type=Path, required=True)
    result.add_argument("--output-prefix", type=Path, required=True)
    result.add_argument(
        "--descriptor",
        choices=("vector", "projective_vector", "scalar", "all"),
        default="vector",
    )
    result.add_argument("--batch-size", type=int, default=32)
    result.add_argument("--device", default="cuda:0")
    result.add_argument(
        "--corruptions", nargs="+", choices=CORRUPTIONS, default=list(CORRUPTIONS)
    )
    result.add_argument("--overwrite", action="store_true")
    return result


def select_descriptor(features: np.ndarray, kind: str) -> np.ndarray:
    if features.shape[-1] < 259:
        raise ValueError(
            "NDF checkpoint does not expose the expected 256D scalar + 3D vector feature"
        )
    if kind == "vector":
        return features[..., -3:]
    if kind == "projective_vector":
        # The legacy NDF vector head learns an unoriented functional axis: v
        # and -v are equivalent across examples.  The six unique entries of
        # vv^T retain the axis and magnitude while removing that sign gauge.
        vector = features[..., -3:]
        x, y, z = np.moveaxis(vector, -1, 0)
        return np.stack((x * x, y * y, z * z, x * y, x * z, y * z), axis=-1)
    if kind == "scalar":
        return features[..., :256]
    if kind == "all":
        return features[..., :259]
    raise ValueError(f"unknown descriptor kind {kind!r}")


def mean_vector_axis(features: np.ndarray) -> np.ndarray:
    """Return the signed task-aligned NDF axis for each camera observation."""
    if features.shape[-1] < 259:
        raise ValueError("source-axis extraction requires the NDF vector head")
    axis = np.asarray(features[..., -3:], dtype=np.float32).mean(axis=1)
    norm = np.linalg.norm(axis, axis=-1, keepdims=True)
    return np.divide(
        axis,
        np.maximum(norm, 1e-6),
        out=np.zeros_like(axis),
        where=norm > 1e-6,
    ).astype(np.float32)


def point_shuffle(features: np.ndarray, episode: np.ndarray, frame: np.ndarray) -> np.ndarray:
    result = np.empty_like(features)
    for index in range(len(features)):
        generator = np.random.default_rng(
            1729 + int(episode[index]) * 100003 + int(frame[index]) * 17
        )
        result[index] = features[index, generator.permutation(features.shape[1])]
    return result


def wrong_episode_map(episode_shoe: np.ndarray) -> dict[int, int]:
    episodes = list(range(len(episode_shoe)))
    result = {}
    for position, episode in enumerate(episodes):
        for offset in range(1, len(episodes) + 1):
            candidate = episodes[(position + offset) % len(episodes)]
            if int(episode_shoe[candidate]) != int(episode_shoe[episode]):
                result[episode] = candidate
                break
        if episode not in result:
            raise ValueError("episode-shuffled control needs at least two shoe identities")
    return result


def episode_shuffle(
    features: np.ndarray,
    episode: np.ndarray,
    frame: np.ndarray,
    episode_shoe: np.ndarray,
) -> np.ndarray:
    mapping = wrong_episode_map(episode_shoe)
    lookup = {
        int(ep): np.flatnonzero(episode == ep)
        for ep in np.unique(episode).astype(np.int64)
    }
    result = np.empty_like(features)
    for index in range(len(features)):
        source_indices = lookup[mapping[int(episode[index])]]
        source_frames = frame[source_indices]
        source = source_indices[int(np.argmin(np.abs(source_frames - frame[index])))]
        result[index] = features[source]
    return result


def deterministic_isotropic_axis(episode: np.ndarray) -> np.ndarray:
    """Generate one deterministic random unit axis per demonstration episode."""
    episode = np.asarray(episode, dtype=np.int64)
    result = np.empty((len(episode), 3), dtype=np.float32)
    for value in np.unique(episode):
        generator = np.random.default_rng(9_170_113 + int(value) * 104_729)
        axis = generator.normal(size=3).astype(np.float32)
        axis /= max(float(np.linalg.norm(axis)), 1e-6)
        result[episode == value] = axis
    return result


def output_path(prefix: Path, corruption: str) -> Path:
    return prefix.parent / f"{prefix.name}_{corruption}.npz"


def main() -> None:
    args = parser().parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    prefix = args.output_prefix.expanduser().resolve()
    outputs = {name: output_path(prefix, name) for name in args.corruptions}
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"outputs already exist: {existing}")
    with np.load(args.dataset, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    points_a = np.asarray(payload["points_a"][..., :3], dtype=np.float32)
    if points_a.ndim != 3 or points_a.shape[-1] != 3:
        raise ValueError(f"points_a must be [samples,points,channels], got {points_a.shape}")
    adapter = NdfPointwiseAdapter(
        args.ndf_checkpoint,
        device=args.device,
        include_vector_direction=True,
    )
    batches = []
    for start in range(0, len(points_a), int(args.batch_size)):
        stop = min(start + int(args.batch_size), len(points_a))
        batches.append(adapter.encode_batch(points_a[start:stop]))
        print(json.dumps({"encoded": stop, "total": len(points_a)}), flush=True)
    ndf_features = np.concatenate(batches, axis=0)
    descriptor = select_descriptor(ndf_features, args.descriptor)
    source_axis = mean_vector_axis(ndf_features)
    shuffled_descriptor = episode_shuffle(
        descriptor,
        payload["episode_index"],
        payload["frame_index"],
        payload["episode_shoe_id"],
    )
    shuffled_axis = episode_shuffle(
        source_axis,
        payload["episode_index"],
        payload["frame_index"],
        payload["episode_shoe_id"],
    )
    controls = {
        "clean": descriptor,
        "zero": np.zeros_like(descriptor),
        "point_shuffled": point_shuffle(
            descriptor, payload["episode_index"], payload["frame_index"]
        ),
        "episode_shuffled": shuffled_descriptor,
        # These two controls change only the explicit functional axis.  Keeping
        # descriptors intact makes clean/sign-flipped an exact signed-semantic
        # test if pointwise descriptors are enabled in a future policy variant.
        "sign_flipped": descriptor,
        "isotropic_axis": descriptor,
    }
    axis_controls = {
        "clean": source_axis,
        "zero": np.zeros_like(source_axis),
        "point_shuffled": source_axis,
        "episode_shuffled": shuffled_axis,
        "sign_flipped": -source_axis,
        "isotropic_axis": deterministic_isotropic_axis(payload["episode_index"]),
    }
    prefix.parent.mkdir(parents=True, exist_ok=True)
    source_metadata_path = args.dataset.with_suffix(".json")
    source_metadata = (
        json.loads(source_metadata_path.read_text(encoding="utf-8"))
        if source_metadata_path.is_file()
        else {}
    )
    for corruption, path in outputs.items():
        augmented = dict(payload)
        augmented["source_axis3"] = axis_controls[corruption].astype(np.float32)
        augmented["points_a"] = np.concatenate(
            (points_a, controls[corruption]), axis=-1
        ).astype(np.float32)
        augmented["points_b"] = np.concatenate(
            (
                np.asarray(payload["points_b"][..., :3], dtype=np.float32),
                np.zeros(
                    (*payload["points_b"].shape[:2], descriptor.shape[-1]),
                    dtype=np.float32,
                ),
            ),
            axis=-1,
        )
        np.savez_compressed(path, **augmented)
        metadata = dict(source_metadata)
        metadata.update(
            {
                "schema_version": 2,
                "output": str(path),
                "source_dataset": str(args.dataset.expanduser().resolve()),
                "ndf_checkpoint": str(args.ndf_checkpoint.expanduser().resolve()),
                "ndf_evaluation": "directly on each current camera point cloud",
                "ndf_descriptor": str(args.descriptor),
                "ndf_descriptor_dim": int(descriptor.shape[-1]),
                "source_axis3": "normalized mean signed NDF vector direction",
                "ndf_corruption": corruption,
                "point_channels": int(augmented["points_a"].shape[-1]),
                "deployment_inputs_for_future_model": [
                    "current camera point clouds A and B",
                    "frozen pointwise NDF descriptors computed from A",
                    "robot proprioception",
                ],
            }
        )
        path.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"output": str(path), "corruption": corruption}), flush=True)


if __name__ == "__main__":
    main()
