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


def frame9_transform(frame9: np.ndarray) -> np.ndarray:
    """Decode metric ``[position, rotation-column-6D]`` into a transform."""
    value = np.asarray(frame9, dtype=np.float32)
    if value.shape[-1] != 9:
        raise ValueError(f"frame9 must end in 9 values, got {value.shape}")
    result = np.zeros((*value.shape[:-1], 4, 4), dtype=np.float32)
    result[..., :3, :3] = frame9_rotation(value)
    result[..., :3, 3] = value[..., :3]
    result[..., 3, 3] = 1.0
    return result


def transform_point_features_to_target_frame(
    points: np.ndarray, frame9: np.ndarray
) -> np.ndarray:
    """Express XYZ in a target frame while preserving aligned features."""
    value = np.asarray(points, dtype=np.float32)
    if value.ndim < 2 or value.shape[-1] < 3:
        raise ValueError(f"points must end in [N,C>=3], got {value.shape}")
    frame = np.asarray(frame9, dtype=np.float32)
    rotation = frame9_rotation(frame)
    translation = frame[..., :3]
    result = value.copy()
    result[..., :3] = np.einsum(
        "...ni,...ij->...nj",
        value[..., :3] - translation[..., None, :],
        rotation,
    )
    return result


def transform_eef_state20_to_target_frame(
    state: np.ndarray, frame9: np.ndarray
) -> np.ndarray:
    """Canonicalize two Cartesian EEF poses and preserve gripper values.

    The 20-D state stores ``[position3, R[:, :2].reshape(6), gripper]`` for
    each arm.  Unlike delta actions, positions require the frame origin as
    well as its rotation.
    """
    value = np.asarray(state, dtype=np.float32)
    if value.shape[-1] != 20:
        raise ValueError(f"state must end in 20 values, got {value.shape}")
    frame = np.asarray(frame9, dtype=np.float32)
    if frame.shape[-1] != 9:
        raise ValueError(f"frame9 must end in 9 values, got {frame.shape}")
    target_from_world = np.swapaxes(frame9_rotation(frame), -1, -2)
    world_from_target = np.swapaxes(target_from_world, -1, -2)
    translation = frame[..., :3]
    result = value.copy()
    for offset in (0, 10):
        result[..., offset : offset + 3] = np.einsum(
            "...i,...ij->...j",
            value[..., offset : offset + 3] - translation,
            world_from_target,
        )
        pose = np.concatenate(
            (
                value[..., offset : offset + 3],
                value[..., offset + 3 : offset + 9],
            ),
            axis=-1,
        )
        world_from_eef = pose9_rotation(pose)
        target_from_eef = target_from_world @ world_from_eef
        result[..., offset + 3 : offset + 9] = target_from_eef[
            ..., :, :2
        ].reshape(*value.shape[:-1], 6)
    return result


def relative_frame9_to_target_frame(
    relative_frame9: np.ndarray, target_frame9: np.ndarray
) -> np.ndarray:
    """Change a world-expressed relative SE(3) token into target coordinates."""
    relative = np.asarray(relative_frame9, dtype=np.float32)
    target = np.asarray(target_frame9, dtype=np.float32)
    if relative.shape[-1] != 9 or target.shape[-1] != 9:
        raise ValueError(
            "relative_frame9 and target_frame9 must both end in 9 values"
        )
    world_from_target = frame9_rotation(target)
    target_from_world = np.swapaxes(world_from_target, -1, -2)
    relative_rotation = frame9_rotation(relative)
    local_rotation = target_from_world @ relative_rotation @ world_from_target
    local_translation = np.einsum(
        "...i,...ij->...j", relative[..., :3], world_from_target
    )
    return np.concatenate(
        (
            local_translation,
            local_rotation[..., :, 0],
            local_rotation[..., :, 1],
        ),
        axis=-1,
    ).astype(np.float32)
