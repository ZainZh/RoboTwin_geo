#!/usr/bin/env python3
"""Validate that matched semantic-policy zarr routes differ only as intended."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from build_semantic_policy_ablation_zarr import _derangement


ROUTE_NAMES = ("field", "xyz", "partprob", "uniform", "shuffled")
INVARIANT_ARRAYS = (
    "meta/episode_ends",
    "data/action",
    "data/state",
    "data/point_cloud",
)


def _array(root, path: str):
    try:
        return root[path]
    except KeyError as error:
        raise ValueError(f"zarr is missing required array {path!r}") from error


def _assert_same_array(
    reference,
    candidate,
    *,
    label: str,
    batch_frames: int,
) -> None:
    if reference.shape != candidate.shape:
        raise ValueError(
            f"{label} shape mismatch: {reference.shape} != {candidate.shape}"
        )
    if reference.dtype != candidate.dtype:
        raise ValueError(
            f"{label} dtype mismatch: {reference.dtype} != {candidate.dtype}"
        )
    if len(reference.shape) == 0:
        if not np.array_equal(np.asarray(reference[()]), np.asarray(candidate[()])):
            raise ValueError(f"{label} value mismatch")
        return
    for start in range(0, int(reference.shape[0]), batch_frames):
        end = min(int(reference.shape[0]), start + batch_frames)
        if not np.array_equal(np.asarray(reference[start:end]), np.asarray(candidate[start:end])):
            raise ValueError(f"{label} value mismatch in frames [{start}, {end})")


def _semantic_keys(field_root, requested: Sequence[str] | None) -> list[str]:
    if requested:
        keys = list(requested)
    else:
        keys = sorted(
            key
            for key in field_root["data"].array_keys()
            if key.startswith("semantic_point_cloud_")
        )
    if not keys:
        raise ValueError("field zarr contains no semantic_point_cloud_* arrays")
    if len(set(keys)) != len(keys):
        raise ValueError(f"semantic keys contain duplicates: {keys}")
    return keys


def _check_probabilities(
    values: np.ndarray,
    *,
    label: str,
    probability_atol: float,
) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} contains non-finite probabilities")
    if np.any(values < -probability_atol):
        raise ValueError(f"{label} contains negative probabilities")
    if not np.allclose(values.sum(axis=-1), 1.0, rtol=0.0, atol=probability_atol):
        raise ValueError(f"{label} probabilities do not sum to one")


def validate_matched_routes(
    route_paths: Mapping[str, Path],
    *,
    semantic_keys: Sequence[str] | None = None,
    batch_frames: int = 128,
    shuffle_seed: int = 20260807,
    probability_atol: float = 1e-5,
    expected_query_points: int | None = None,
    expected_field_feature_dim: int | None = None,
) -> dict:
    """Validate five matched routes and return a JSON-serializable report.

    The shuffle check reconstructs the builder's per-frame derangement. This is
    stronger than comparing sorted rows and remains well-defined when two points
    happen to have identical probability vectors.
    """
    import zarr

    if batch_frames <= 0:
        raise ValueError("batch_frames must be positive")
    if probability_atol < 0:
        raise ValueError("probability_atol must be non-negative")
    missing_routes = [name for name in ROUTE_NAMES if name not in route_paths]
    extra_routes = sorted(set(route_paths) - set(ROUTE_NAMES))
    if missing_routes or extra_routes:
        raise ValueError(
            f"route mapping mismatch: missing={missing_routes}, extra={extra_routes}"
        )

    resolved_paths = {name: Path(route_paths[name]).expanduser().resolve() for name in ROUTE_NAMES}
    missing_paths = [f"{name}={path}" for name, path in resolved_paths.items() if not path.is_dir()]
    if missing_paths:
        raise FileNotFoundError("zarr route directories do not exist: " + ", ".join(missing_paths))
    roots = {name: zarr.open(str(path), mode="r") for name, path in resolved_paths.items()}
    for name, root in roots.items():
        if "data" not in root or "meta" not in root:
            raise ValueError(f"{name} zarr must contain data and meta groups")

    keys = _semantic_keys(roots["field"], semantic_keys)
    invariant_report = {}
    for path in INVARIANT_ARRAYS:
        reference = _array(roots["field"], path)
        for route in ROUTE_NAMES[1:]:
            _assert_same_array(
                reference,
                _array(roots[route], path),
                label=f"{route}:{path}",
                batch_frames=batch_frames,
            )
        invariant_report[path] = {
            "shape": list(reference.shape),
            "dtype": str(reference.dtype),
            "bitwise_identical_across_routes": True,
        }

    semantic_report = {}
    for key in keys:
        arrays = {name: _array(root, f"data/{key}") for name, root in roots.items()}
        field = arrays["field"]
        if len(field.shape) != 3 or field.shape[-1] < 4:
            raise ValueError(f"field:{key} must have shape [frames, points, 3+F], got {field.shape}")
        frame_count, query_points = int(field.shape[0]), int(field.shape[1])
        if expected_query_points is not None and query_points != expected_query_points:
            raise ValueError(
                f"field:{key} query-point count {query_points} != expected {expected_query_points}"
            )
        field_feature_dim = int(field.shape[-1]) - 3
        if (
            expected_field_feature_dim is not None
            and field_feature_dim != expected_field_feature_dim
        ):
            raise ValueError(
                f"field:{key} feature dim {field_feature_dim} != expected "
                f"{expected_field_feature_dim}"
            )

        for route, array in arrays.items():
            if len(array.shape) != 3 or tuple(array.shape[:2]) != (frame_count, query_points):
                raise ValueError(
                    f"{route}:{key} frame/point shape {array.shape[:2]} != "
                    f"{(frame_count, query_points)}"
                )
        if int(arrays["xyz"].shape[-1]) != 3:
            raise ValueError(f"xyz:{key} must have exactly 3 channels, got {arrays['xyz'].shape}")
        probability_channels = int(arrays["partprob"].shape[-1]) - 3
        if probability_channels < 2:
            raise ValueError(
                f"partprob:{key} must have xyz plus at least two classes, "
                f"got {arrays['partprob'].shape}"
            )
        expected_probability_channels = probability_channels + 3
        for route in ("uniform", "shuffled"):
            if int(arrays[route].shape[-1]) != expected_probability_channels:
                raise ValueError(
                    f"{route}:{key} channels {arrays[route].shape[-1]} != "
                    f"partprob channels {expected_probability_channels}"
                )

        for route, array in arrays.items():
            for start in range(0, frame_count, batch_frames):
                end = min(frame_count, start + batch_frames)
                field_xyz = np.asarray(field[start:end, ..., :3])
                route_xyz = np.asarray(array[start:end, ..., :3])
                if not np.array_equal(field_xyz, route_xyz):
                    raise ValueError(
                        f"{route}:{key} query XYZ mismatch in frames [{start}, {end})"
                    )

        coincident_probability_rows = 0
        total_probability_rows = frame_count * query_points
        uniform_value = np.float32(1.0 / probability_channels)
        for start in range(0, frame_count, batch_frames):
            end = min(frame_count, start + batch_frames)
            part = np.asarray(arrays["partprob"][start:end, ..., 3:])
            uniform = np.asarray(arrays["uniform"][start:end, ..., 3:])
            shuffled = np.asarray(arrays["shuffled"][start:end, ..., 3:])
            _check_probabilities(part, label=f"partprob:{key}", probability_atol=probability_atol)
            _check_probabilities(uniform, label=f"uniform:{key}", probability_atol=probability_atol)
            _check_probabilities(shuffled, label=f"shuffled:{key}", probability_atol=probability_atol)
            if not np.array_equal(uniform, np.full_like(uniform, uniform_value)):
                raise ValueError(
                    f"uniform:{key} is not exactly 1/{probability_channels} in "
                    f"frames [{start}, {end})"
                )
            for local_frame in range(end - start):
                global_frame = start + local_frame
                order = _derangement(query_points, seed=shuffle_seed + global_frame)
                if np.any(order == np.arange(query_points)):
                    raise AssertionError("internal derangement unexpectedly has a fixed index")
                expected = part[local_frame, order]
                actual = shuffled[local_frame]
                if not np.array_equal(actual, expected):
                    raise ValueError(
                        f"shuffled:{key} does not match the seeded per-frame derangement "
                        f"at frame {global_frame}"
                    )
                coincident_probability_rows += int(
                    np.count_nonzero(np.all(actual == part[local_frame], axis=-1))
                )

        semantic_report[key] = {
            "frames": frame_count,
            "query_points": query_points,
            "field_feature_dim": field_feature_dim,
            "probability_classes": probability_channels,
            "channels": {name: int(array.shape[-1]) for name, array in arrays.items()},
            "query_xyz_bitwise_identical": True,
            "probabilities_normalized": True,
            "uniform_exact": True,
            "shuffle_multiset_preserved": True,
            "shuffle_seeded_derangement_exact": True,
            "shuffle_fixed_indices": 0,
            "shuffle_value_coincidences": coincident_probability_rows,
            "shuffle_total_rows": total_probability_rows,
        }

    return {
        "status": "pass",
        "routes": {name: str(path) for name, path in resolved_paths.items()},
        "parameters": {
            "batch_frames": batch_frames,
            "shuffle_seed": shuffle_seed,
            "probability_atol": probability_atol,
            "expected_query_points": expected_query_points,
            "expected_field_feature_dim": expected_field_feature_dim,
        },
        "invariants": invariant_report,
        "semantic_arrays": semantic_report,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate fair, matched field/XYZ/part-probability/control policy zarr routes."
    )
    for route in ROUTE_NAMES:
        parser.add_argument(f"--{route}-zarr", type=Path, required=True)
    parser.add_argument("--semantic-keys", nargs="*", default=None)
    parser.add_argument("--batch-frames", type=int, default=128)
    parser.add_argument("--shuffle-seed", type=int, default=20260807)
    parser.add_argument("--probability-atol", type=float, default=1e-5)
    parser.add_argument("--expected-query-points", type=int, default=None)
    parser.add_argument("--expected-field-feature-dim", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    paths = {name: getattr(args, f"{name}_zarr") for name in ROUTE_NAMES}
    report = validate_matched_routes(
        paths,
        semantic_keys=args.semantic_keys,
        batch_frames=args.batch_frames,
        shuffle_seed=args.shuffle_seed,
        probability_atol=args.probability_atol,
        expected_query_points=args.expected_query_points,
        expected_field_feature_dim=args.expected_field_feature_dim,
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output_json is not None:
        output = args.output_json.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        temporary.replace(output)
    print(rendered)


if __name__ == "__main__":
    main()
