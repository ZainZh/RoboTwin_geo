#!/usr/bin/env python3
"""Fail-closed input audit for matched RoboTwin DINOv2 RGB-D lifting.

This tool does not synthesize depth from a sparse point cloud.  A dataset is
eligible only when every requested camera contains synchronized RGB, dense
depth, and calibration, and its episode lengths match the policy field zarr.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np


DEPTH_KEYS = ("depth", "depth_m", "depth_mm")
SEGMENTATION_KEYS = (
    "actor_segmentation",
    "mesh_segmentation",
    "foreground_mask",
    "mask",
)


class IssueCollector:
    def __init__(self) -> None:
        self._issues: dict[str, list[str]] = defaultdict(list)

    def add(self, code: str, detail: str) -> None:
        self._issues[str(code)].append(str(detail))

    def report(self) -> list[dict]:
        return [
            {
                "code": code,
                "count": len(details),
                "examples": details[:5],
            }
            for code, details in sorted(self._issues.items())
        ]

    def __bool__(self) -> bool:
        return bool(self._issues)


def _episode_index(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("episode") or not stem[7:].isdigit():
        raise ValueError(f"invalid episode filename: {path.name}")
    return int(stem[7:])


def _discover_episode_paths(dataset_dir: Path, expected_episodes: int | None) -> list[Path]:
    data_dir = dataset_dir / "data" if (dataset_dir / "data").is_dir() else dataset_dir
    paths = sorted(data_dir.glob("episode*.hdf5"), key=_episode_index)
    if not paths:
        raise FileNotFoundError(f"no episode*.hdf5 files found under {data_dir}")
    indices = [_episode_index(path) for path in paths]
    expected_indices = list(range(len(paths)))
    if indices != expected_indices:
        raise ValueError(
            f"episode files must be contiguous from zero: found {indices}, expected {expected_indices}"
        )
    if expected_episodes is not None and len(paths) != int(expected_episodes):
        raise ValueError(
            f"episode count {len(paths)} does not match expected {expected_episodes}"
        )
    return paths


def _frame_indices(frame_count: int, max_frames_per_episode: int | None) -> list[int]:
    if frame_count <= 0:
        return []
    if max_frames_per_episode is not None:
        return list(range(min(frame_count, int(max_frames_per_episode))))
    return list(range(frame_count))


def _matrix_has_frame_axis(dataset: h5py.Dataset, frame_count: int) -> bool:
    return dataset.ndim >= 3 and int(dataset.shape[0]) == int(frame_count)


def _read_frame_or_static(dataset: h5py.Dataset, frame_index: int, frame_count: int) -> np.ndarray:
    if _matrix_has_frame_axis(dataset, frame_count):
        return np.asarray(dataset[frame_index])
    return np.asarray(dataset[()])


def _decode_rgb_frame(dataset: h5py.Dataset, frame_index: int) -> np.ndarray:
    value = dataset[frame_index]
    if isinstance(value, (bytes, np.bytes_)):
        import cv2

        encoded = np.frombuffer(bytes(value).rstrip(b"\0"), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("JPEG decode returned None")
        return image
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError(f"RGB frame must be HxWx3+, got {image.shape}")
    return image


def _dataset_frames_match(dataset: h5py.Dataset, frame_count: int) -> bool:
    return dataset.ndim >= 1 and int(dataset.shape[0]) == int(frame_count)


def _finite_dataset(dataset: h5py.Dataset, frame_indices: Iterable[int], frame_count: int) -> bool:
    if _matrix_has_frame_axis(dataset, frame_count):
        return all(np.isfinite(np.asarray(dataset[index])).all() for index in frame_indices)
    return bool(np.isfinite(np.asarray(dataset[()])).all())


def _find_first_key(group: h5py.Group, candidates: Sequence[str]) -> str | None:
    return next((key for key in candidates if key in group), None)


def _audit_policy_zarr(
    *,
    field_zarr: Path,
    raw_episode_lengths: Sequence[int],
    expected_query_points: int,
    semantic_keys: Sequence[str] | None,
    issues: IssueCollector,
) -> dict:
    import zarr

    if not field_zarr.is_dir():
        issues.add("missing_field_zarr", str(field_zarr))
        return {"path": str(field_zarr), "checked": False}
    root = zarr.open(str(field_zarr), mode="r")
    if "meta/episode_ends" not in root:
        issues.add("missing_policy_episode_ends", str(field_zarr))
        return {"path": str(field_zarr), "checked": False}
    episode_ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
    starts = np.concatenate([np.zeros(1, dtype=np.int64), episode_ends[:-1]])
    policy_lengths = (episode_ends - starts).tolist()
    expected_lengths = [int(length) - 1 for length in raw_episode_lengths]
    if policy_lengths != expected_lengths:
        issues.add(
            "policy_episode_length_mismatch",
            f"policy={policy_lengths}, expected_raw_T_minus_1={expected_lengths}",
        )

    if "data" not in root:
        issues.add("missing_policy_data_group", str(field_zarr))
        return {"path": str(field_zarr), "checked": False}
    keys = list(semantic_keys or sorted(
        key for key in root["data"].array_keys() if key.startswith("semantic_point_cloud_")
    ))
    if not keys:
        issues.add("missing_semantic_query_arrays", str(field_zarr))
    total_frames = int(episode_ends[-1]) if episode_ends.size else 0
    query_arrays = {}
    for key in keys:
        path = f"data/{key}"
        if path not in root:
            issues.add("missing_semantic_query_array", path)
            continue
        array = root[path]
        query_arrays[key] = {"shape": list(array.shape), "dtype": str(array.dtype)}
        if len(array.shape) != 3 or int(array.shape[0]) != total_frames:
            issues.add("semantic_query_shape_mismatch", f"{key}: {array.shape}")
            continue
        if int(array.shape[1]) != int(expected_query_points):
            issues.add(
                "semantic_query_point_count_mismatch",
                f"{key}: {array.shape[1]} != {expected_query_points}",
            )
        for start in range(0, total_frames, 512):
            xyz = np.asarray(array[start : min(total_frames, start + 512), ..., :3])
            if not np.isfinite(xyz).all():
                issues.add("nonfinite_semantic_query_xyz", f"{key}: frames {start}+")
                break

    for key in ("action", "state", "point_cloud"):
        path = f"data/{key}"
        if path not in root:
            issues.add("missing_policy_invariant_array", path)
        elif int(root[path].shape[0]) != total_frames:
            issues.add("policy_invariant_length_mismatch", f"{path}: {root[path].shape}")
    return {
        "path": str(field_zarr),
        "checked": True,
        "episode_count": int(len(episode_ends)),
        "episode_lengths": policy_lengths,
        "expected_raw_T_minus_1": expected_lengths,
        "total_frames": total_frames,
        "semantic_query_arrays": query_arrays,
        "frame_mapping_matches": policy_lengths == expected_lengths,
    }


def audit_robotwin_dinov2_inputs(
    *,
    dataset_dir: Path,
    field_zarr: Path,
    camera_names: Sequence[str] = ("head_camera", "front_camera"),
    expected_episodes: int | None = None,
    expected_query_points: int = 128,
    semantic_keys: Sequence[str] | None = None,
    max_episodes: int | None = None,
    max_frames_per_episode: int | None = None,
    require_foreground_mask: bool = False,
    min_depth_m: float = 1e-6,
    max_depth_m: float = 10.0,
) -> dict:
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    field_zarr = Path(field_zarr).expanduser().resolve()
    if len(camera_names) < 2:
        raise ValueError("visibility-aware multiview lifting requires at least two cameras")
    if max_episodes is not None and max_episodes <= 0:
        raise ValueError("max_episodes must be positive")
    if max_frames_per_episode is not None and max_frames_per_episode <= 0:
        raise ValueError("max_frames_per_episode must be positive")
    if not (0.0 <= min_depth_m < max_depth_m):
        raise ValueError("depth bounds must satisfy 0 <= min_depth_m < max_depth_m")

    all_paths = _discover_episode_paths(dataset_dir, expected_episodes)
    selected_paths = all_paths[:max_episodes] if max_episodes is not None else all_paths
    limited_by_episode = len(selected_paths) != len(all_paths)
    issues = IssueCollector()
    raw_episode_lengths: list[int] = []
    camera_summary = {
        str(name): {
            "episodes_present": 0,
            "frames": 0,
            "rgb_frames_checked": 0,
            "depth_frames_checked": 0,
            "depth_valid_pixels": 0,
            "depth_total_pixels": 0,
            "depth_keys": defaultdict(int),
            "segmentation_keys": defaultdict(int),
            "camera_keysets": defaultdict(int),
        }
        for name in camera_names
    }
    limited_by_frame = False

    for episode_path in selected_paths:
        episode_index = _episode_index(episode_path)
        with h5py.File(episode_path, "r") as root:
            if "joint_action/vector" not in root:
                issues.add("missing_joint_action_vector", episode_path.name)
                continue
            frame_count = int(root["joint_action/vector"].shape[0])
            raw_episode_lengths.append(frame_count)
            if frame_count < 2:
                issues.add("too_short_episode", f"{episode_path.name}: {frame_count}")
            checked_frames = _frame_indices(frame_count, max_frames_per_episode)
            if max_frames_per_episode is not None and len(checked_frames) < frame_count:
                limited_by_frame = True
            if "observation" not in root:
                issues.add("missing_observation_group", episode_path.name)
                continue
            observation = root["observation"]
            for camera_name in camera_names:
                summary = camera_summary[str(camera_name)]
                if camera_name not in observation:
                    issues.add("missing_camera", f"{episode_path.name}:{camera_name}")
                    continue
                camera = observation[camera_name]
                summary["episodes_present"] += 1
                summary["frames"] += frame_count
                summary["camera_keysets"][",".join(sorted(camera.keys()))] += 1
                decoded_rgb_shapes: dict[int, tuple[int, int]] = {}

                for required in ("rgb", "intrinsic_cv", "extrinsic_cv"):
                    if required not in camera:
                        issues.add(
                            f"missing_{required}", f"{episode_path.name}:{camera_name}"
                        )
                if "rgb" in camera:
                    rgb = camera["rgb"]
                    if not _dataset_frames_match(rgb, frame_count):
                        issues.add(
                            "rgb_frame_count_mismatch",
                            f"{episode_path.name}:{camera_name}:{rgb.shape} vs {frame_count}",
                        )
                    else:
                        for frame_index in checked_frames:
                            try:
                                image = _decode_rgb_frame(rgb, frame_index)
                            except Exception as error:
                                issues.add(
                                    "rgb_decode_failure",
                                    f"{episode_path.name}:{camera_name}:{frame_index}:{error}",
                                )
                                continue
                            summary["rgb_frames_checked"] += 1
                            decoded_rgb_shapes[frame_index] = (
                                int(image.shape[0]),
                                int(image.shape[1]),
                            )
                            if not np.isfinite(image).all():
                                issues.add(
                                    "nonfinite_rgb",
                                    f"{episode_path.name}:{camera_name}:{frame_index}",
                                )

                for calibration_key in ("intrinsic_cv", "extrinsic_cv", "cam2world_gl"):
                    if calibration_key not in camera:
                        continue
                    calibration = camera[calibration_key]
                    if calibration.ndim >= 3 and not _dataset_frames_match(
                        calibration, frame_count
                    ):
                        issues.add(
                            "calibration_frame_count_mismatch",
                            f"{episode_path.name}:{camera_name}:{calibration_key}:{calibration.shape}",
                        )
                    elif not _finite_dataset(
                        calibration, checked_frames, frame_count
                    ):
                        issues.add(
                            "nonfinite_calibration",
                            f"{episode_path.name}:{camera_name}:{calibration_key}",
                        )

                    expected_shapes = {
                        "intrinsic_cv": {(3, 3)},
                        "extrinsic_cv": {(3, 4), (4, 4)},
                        "cam2world_gl": {(4, 4)},
                    }[calibration_key]
                    sample_matrix = _read_frame_or_static(
                        calibration, checked_frames[0], frame_count
                    )
                    if tuple(sample_matrix.shape) not in expected_shapes:
                        issues.add(
                            "invalid_calibration_shape",
                            f"{episode_path.name}:{camera_name}:{calibration_key}:{sample_matrix.shape}",
                        )
                depth_key = _find_first_key(camera, DEPTH_KEYS)
                if depth_key is None:
                    issues.add("missing_dense_depth", f"{episode_path.name}:{camera_name}")
                else:
                    summary["depth_keys"][depth_key] += 1
                    depth = camera[depth_key]
                    if depth.ndim != 3 or not _dataset_frames_match(depth, frame_count):
                        issues.add(
                            "depth_shape_or_frame_mismatch",
                            f"{episode_path.name}:{camera_name}:{depth_key}:{depth.shape}",
                        )
                    else:
                        for frame_index in checked_frames:
                            depth_frame = np.asarray(depth[frame_index], dtype=np.float32)
                            if depth_key == "depth_mm":
                                depth_frame = depth_frame / 1000.0
                            if (
                                frame_index in decoded_rgb_shapes
                                and tuple(depth_frame.shape) != decoded_rgb_shapes[frame_index]
                            ):
                                issues.add(
                                    "rgb_depth_shape_mismatch",
                                    f"{episode_path.name}:{camera_name}:{frame_index}:"
                                    f"rgb={decoded_rgb_shapes[frame_index]},depth={depth_frame.shape}",
                                )
                            finite = np.isfinite(depth_frame)
                            valid = finite & (depth_frame > min_depth_m) & (
                                depth_frame < max_depth_m
                            )
                            summary["depth_frames_checked"] += 1
                            summary["depth_valid_pixels"] += int(valid.sum())
                            summary["depth_total_pixels"] += int(depth_frame.size)
                            if not np.any(valid):
                                issues.add(
                                    "no_valid_depth",
                                    f"{episode_path.name}:{camera_name}:{frame_index}",
                                )

                segmentation_key = _find_first_key(camera, SEGMENTATION_KEYS)
                if segmentation_key is not None:
                    summary["segmentation_keys"][segmentation_key] += 1
                elif require_foreground_mask:
                    issues.add(
                        "missing_foreground_mask", f"{episode_path.name}:{camera_name}"
                    )

    policy = _audit_policy_zarr(
        field_zarr=field_zarr,
        raw_episode_lengths=raw_episode_lengths,
        expected_query_points=expected_query_points,
        semantic_keys=semantic_keys,
        issues=issues,
    )
    scope_complete = not limited_by_episode and not limited_by_frame
    if not scope_complete:
        issues.add(
            "incomplete_smoke_scope",
            f"selected_episodes={len(selected_paths)}/{len(all_paths)}, "
            f"max_frames_per_episode={max_frames_per_episode}",
        )

    rendered_camera_summary = {}
    for name, summary in camera_summary.items():
        total_depth = int(summary["depth_total_pixels"])
        rendered_camera_summary[name] = {
            **{
                key: int(value)
                for key, value in summary.items()
                if key not in {"depth_keys", "segmentation_keys", "camera_keysets"}
            },
            "depth_valid_ratio": (
                float(summary["depth_valid_pixels"]) / total_depth if total_depth else None
            ),
            "depth_keys": dict(sorted(summary["depth_keys"].items())),
            "segmentation_keys": dict(sorted(summary["segmentation_keys"].items())),
            "camera_keysets": dict(sorted(summary["camera_keysets"].items())),
        }
    blocking_reasons = issues.report()
    eligible = not blocking_reasons and scope_complete
    status = "pass" if eligible else "blocked"
    return {
        "schema_version": "robotwin_dinov2_input_audit_v1",
        "status": status,
        "eligible_for_visibility_aware_dinov2_zarr": eligible,
        "dataset_dir": str(dataset_dir),
        "field_zarr": str(field_zarr),
        "requirements": {
            "camera_names": [str(name) for name in camera_names],
            "minimum_camera_count": 2,
            "dense_depth_required": True,
            "foreground_mask_required": bool(require_foreground_mask),
            "expected_episodes": expected_episodes,
            "expected_query_points": expected_query_points,
            "depth_bounds_m": [float(min_depth_m), float(max_depth_m)],
        },
        "scope": {
            "discovered_episodes": len(all_paths),
            "selected_episodes": len(selected_paths),
            "max_frames_per_episode": max_frames_per_episode,
            "complete_dataset_audit": scope_complete,
            "raw_episode_lengths": raw_episode_lengths,
            "raw_total_frames": int(sum(raw_episode_lengths)),
        },
        "cameras": rendered_camera_summary,
        "matched_policy": policy,
        "blocking_reasons": blocking_reasons,
        "no_depth_synthesis_or_sparse_pointcloud_substitution": True,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether a RoboTwin dataset can support calibrated, depth-aware, "
            "matched-query DINOv2 lifting. Exits 2 when blocked."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--field-zarr", type=Path, required=True)
    parser.add_argument("--camera-names", default="head_camera,front_camera")
    parser.add_argument("--expected-episodes", type=int, default=None)
    parser.add_argument("--expected-query-points", type=int, default=128)
    parser.add_argument("--semantic-keys", nargs="*", default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-frames-per-episode", type=int, default=None)
    parser.add_argument("--require-foreground-mask", action="store_true")
    parser.add_argument("--min-depth-m", type=float, default=1e-6)
    parser.add_argument("--max-depth-m", type=float, default=10.0)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    camera_names = [value.strip() for value in args.camera_names.split(",") if value.strip()]
    report = audit_robotwin_dinov2_inputs(
        dataset_dir=args.dataset_dir,
        field_zarr=args.field_zarr,
        camera_names=camera_names,
        expected_episodes=args.expected_episodes,
        expected_query_points=args.expected_query_points,
        semantic_keys=args.semantic_keys,
        max_episodes=args.max_episodes,
        max_frames_per_episode=args.max_frames_per_episode,
        require_foreground_mask=args.require_foreground_mask,
        min_depth_m=args.min_depth_m,
        max_depth_m=args.max_depth_m,
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output_json is not None:
        output = args.output_json.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        temporary.replace(output)
    print(rendered)
    if report["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
