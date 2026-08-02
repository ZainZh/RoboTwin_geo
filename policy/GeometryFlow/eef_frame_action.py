"""Convert Cartesian delta actions between world and instantaneous EEF frames."""

from __future__ import annotations

import numpy as np

from .canonicalize_task_flow_dataset import pose7_to_matrix


def action_world_to_eef(action: np.ndarray, eef_pose7: np.ndarray) -> np.ndarray:
    value = np.asarray(action, dtype=np.float64).copy()
    poses = np.asarray(eef_pose7, dtype=np.float64).reshape(2, 7)
    for arm, offset in enumerate((0, 7)):
        world_from_eef = pose7_to_matrix(poses[arm])[:3, :3]
        value[offset : offset + 3] = value[offset : offset + 3] @ world_from_eef
        value[offset + 3 : offset + 6] = (
            value[offset + 3 : offset + 6] @ world_from_eef
        )
    return value.astype(np.float32)


def action_eef_to_world(action: np.ndarray, eef_pose7: np.ndarray) -> np.ndarray:
    value = np.asarray(action, dtype=np.float64).copy()
    poses = np.asarray(eef_pose7, dtype=np.float64).reshape(2, 7)
    for arm, offset in enumerate((0, 7)):
        world_from_eef = pose7_to_matrix(poses[arm])[:3, :3]
        value[offset : offset + 3] = value[offset : offset + 3] @ world_from_eef.T
        value[offset + 3 : offset + 6] = (
            value[offset + 3 : offset + 6] @ world_from_eef.T
        )
    return value.astype(np.float32)


def payload_with_eef_frame_actions(
    payload: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Return a shallow payload copy with every action expressed locally."""
    result = dict(payload)
    world = np.asarray(payload["action"], dtype=np.float32)
    local = np.empty_like(world)
    current = np.asarray(payload["current_eef_pose7"], dtype=np.float32)
    future = np.asarray(payload["future_eef_pose7"], dtype=np.float32)
    for sample in range(len(world)):
        for step in range(world.shape[1]):
            source_pose = current[sample] if step == 0 else future[sample, step - 1]
            local[sample, step] = action_world_to_eef(
                world[sample, step], source_pose
            )
    result["action"] = local
    result["action_frame_eef"] = np.asarray(1, dtype=np.int64)
    return result

