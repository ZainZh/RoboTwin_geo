"""Transform Cartesian delta actions through a camera-estimated task frame."""

from __future__ import annotations

import numpy as np


def frame9_rotation(frame9: np.ndarray) -> np.ndarray:
    """Recover a proper rotation matrix from translation + first two columns."""
    value = np.asarray(frame9, dtype=np.float64)
    if value.shape[-1] != 9:
        raise ValueError(f"frame9 must end in 9 values, got {value.shape}")
    first = value[..., 3:6]
    first = first / np.maximum(
        np.linalg.norm(first, axis=-1, keepdims=True), 1e-12
    )
    second = value[..., 6:9]
    second = second - first * np.sum(first * second, axis=-1, keepdims=True)
    second = second / np.maximum(
        np.linalg.norm(second, axis=-1, keepdims=True), 1e-12
    )
    third = np.cross(first, second)
    return np.stack((first, second, third), axis=-1).astype(np.float32)


def pose9_rotation(pose9: np.ndarray) -> np.ndarray:
    """Decode pose-6D stored as ``R[:, :2].reshape(6)``.

    ``target_frame9`` stores two complete columns consecutively, whereas the
    historical object ``pose9`` dataset uses NumPy's row-major flattening of a
    ``(3, 2)`` slice.  Keeping the two decoders explicit prevents silently
    interpreting one layout as the other.
    """
    value = np.asarray(pose9, dtype=np.float64)
    if value.shape[-1] != 9:
        raise ValueError(f"pose9 must end in 9 values, got {value.shape}")
    columns = value[..., 3:9].reshape(*value.shape[:-1], 3, 2)
    frame = np.concatenate(
        (value[..., :3], columns[..., :, 0], columns[..., :, 1]), axis=-1
    )
    return frame9_rotation(frame)


def transform_action14(
    action: np.ndarray,
    frame9: np.ndarray,
    *,
    to_local: bool,
) -> np.ndarray:
    """Rotate translation and rotation-vector deltas, preserving grippers."""
    value = np.asarray(action, dtype=np.float32)
    if value.shape[-1] != 14:
        raise ValueError(f"action must end in 14 values, got {value.shape}")
    rotations = frame9_rotation(frame9)
    while rotations.ndim < value.ndim + 1:
        rotations = np.expand_dims(rotations, axis=-3)
    transform = rotations if to_local else np.swapaxes(rotations, -1, -2)
    result = value.copy()
    for start in (0, 3, 7, 10):
        result[..., start : start + 3] = np.einsum(
            "...i,...ij->...j", value[..., start : start + 3], transform
        )
    return result


def action14_to_target_frame(
    action: np.ndarray, frame9: np.ndarray
) -> np.ndarray:
    return transform_action14(action, frame9, to_local=True)


def action14_from_target_frame(
    action: np.ndarray, frame9: np.ndarray
) -> np.ndarray:
    return transform_action14(action, frame9, to_local=False)
