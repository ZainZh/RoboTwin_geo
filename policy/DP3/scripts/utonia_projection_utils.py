#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


PROJECTION_FORMAT = "utonia_frozen_orthogonal_projection_v1"


def build_frozen_orthogonal_projection(
    input_dim: int,
    output_dim: int,
    *,
    seed: int,
) -> np.ndarray:
    input_dim = int(input_dim)
    output_dim = int(output_dim)
    if input_dim <= 0 or output_dim <= 0:
        raise ValueError("projection dimensions must be positive")
    if output_dim > input_dim:
        raise ValueError("output_dim cannot exceed input_dim")
    generator = np.random.default_rng(int(seed))
    gaussian = generator.standard_normal((input_dim, output_dim), dtype=np.float64)
    basis, triangular = np.linalg.qr(gaussian, mode="reduced")
    signs = np.sign(np.diag(triangular))
    signs[signs == 0] = 1.0
    basis *= signs[None, :]
    # Preserve the expected squared norm of an isotropic input after reducing
    # 576 channels to 128, while retaining mutually orthogonal columns.
    basis *= np.sqrt(float(input_dim) / float(output_dim))
    matrix = np.asarray(basis, dtype=np.float32)
    if matrix.shape != (input_dim, output_dim) or not np.isfinite(matrix).all():
        raise RuntimeError("failed to construct a finite projection matrix")
    return matrix


def matrix_sha256(matrix: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(matrix, dtype=np.float32))
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def save_projection_artifact(
    path: Path,
    matrix: np.ndarray,
    *,
    seed: int,
) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    values = np.ascontiguousarray(np.asarray(matrix, dtype=np.float32))
    temporary = destination.with_name(f".{destination.name}.tmp.npz")
    np.savez_compressed(
        temporary,
        format=np.asarray(PROJECTION_FORMAT),
        matrix=values,
        input_dim=np.asarray(values.shape[0], dtype=np.int64),
        output_dim=np.asarray(values.shape[1], dtype=np.int64),
        seed=np.asarray(int(seed), dtype=np.int64),
        matrix_sha256=np.asarray(matrix_sha256(values)),
    )
    temporary.replace(destination)


def load_projection_artifact(
    path: str | Path,
    *,
    expected_input_dim: int | None = None,
    expected_output_dim: int | None = None,
) -> dict:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Utonia projection artifact is unavailable: {source}")
    with np.load(source, allow_pickle=False) as payload:
        artifact_format = str(payload["format"].item())
        matrix = np.asarray(payload["matrix"], dtype=np.float32)
        seed = int(payload["seed"].item())
        recorded_sha = str(payload["matrix_sha256"].item())
    if artifact_format != PROJECTION_FORMAT:
        raise ValueError(f"unsupported projection artifact format: {artifact_format}")
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError(f"projection matrix must be finite and 2D, got {matrix.shape}")
    if expected_input_dim is not None and matrix.shape[0] != int(expected_input_dim):
        raise ValueError(
            f"projection input dim mismatch: expected {expected_input_dim}, got {matrix.shape[0]}"
        )
    if expected_output_dim is not None and matrix.shape[1] != int(expected_output_dim):
        raise ValueError(
            f"projection output dim mismatch: expected {expected_output_dim}, got {matrix.shape[1]}"
        )
    actual_sha = matrix_sha256(matrix)
    if actual_sha != recorded_sha:
        raise ValueError("projection matrix SHA-256 mismatch")
    return {"path": source, "matrix": matrix, "seed": seed, "matrix_sha256": actual_sha}


def project_utonia_cloud(cloud: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(cloud, dtype=np.float32)
    projection = np.asarray(matrix, dtype=np.float32)
    if values.ndim < 2 or values.shape[-1] != 3 + projection.shape[0]:
        raise ValueError(
            f"Utonia cloud must end in {3 + projection.shape[0]} channels, got {values.shape}"
        )
    projected = np.matmul(values[..., 3:], projection).astype(np.float32)
    result = np.concatenate([values[..., :3], projected], axis=-1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("projected Utonia cloud contains NaN/Inf")
    return result
