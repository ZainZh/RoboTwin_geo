"""Camera-deployable hand--object interaction-frame tokens.

The task frame says how the operated object should move toward the goal.  This
module supplies the complementary relation: how the currently closed gripper
is attached to the estimated functional frame of the operated object.  Both
training and deployment derive it from observable EEF proprioception and the
same camera/NDF frame used by the task token.
"""

from __future__ import annotations

import numpy as np

from .functional_action_frame import frame9_rotation, pose9_rotation


HAND_OBJECT_TRANSLATION_SCALE_M = 0.2


def current_frame_from_goal_relative(
    goal_pose9: np.ndarray, relative_frame9: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Recover ``world_from_current`` from goal pose and goal-from-current token."""

    goal = np.asarray(goal_pose9, dtype=np.float32).reshape(9)
    relative = np.asarray(relative_frame9, dtype=np.float32).reshape(9)
    goal_rotation = pose9_rotation(goal)
    relative_rotation = frame9_rotation(relative)
    current_position = goal[:3] - relative[:3]
    current_rotation = relative_rotation.T @ goal_rotation
    return current_position.astype(np.float32), current_rotation.astype(np.float32)


def active_hand_object_frame11(
    current_position: np.ndarray,
    current_rotation: np.ndarray,
    eef_state20: np.ndarray,
    *,
    translation_scale_m: float = HAND_OBJECT_TRANSLATION_SCALE_M,
) -> np.ndarray:
    """Encode the object functional frame in the observably closed EEF frame.

    The output is ``[p/scale, R[:,0], R[:,1], active_left, active_right]``.
    Gripper values in RoboTwin are zero when closed, so the active arm is the
    smaller of the two proprioceptive gripper scalars.  This is not a learned
    task-stage label.
    """

    if float(translation_scale_m) <= 0.0:
        raise ValueError("translation_scale_m must be positive")
    position = np.asarray(current_position, dtype=np.float32).reshape(3)
    rotation = np.asarray(current_rotation, dtype=np.float32).reshape(3, 3)
    state = np.asarray(eef_state20, dtype=np.float32).reshape(20)
    active_right = bool(float(state[19]) < float(state[9]))
    offset = 10 if active_right else 0
    eef_pose9 = state[offset : offset + 9]
    eef_rotation = pose9_rotation(eef_pose9)
    relative_position = eef_rotation.T @ (position - eef_pose9[:3])
    relative_rotation = eef_rotation.T @ rotation
    active = np.asarray(
        [0.0 if active_right else 1.0, 1.0 if active_right else 0.0],
        dtype=np.float32,
    )
    return np.concatenate(
        (
            relative_position / float(translation_scale_m),
            relative_rotation[:, 0],
            relative_rotation[:, 1],
            active,
        )
    ).astype(np.float32)


def active_hand_object_frame11_from_goal_relative(
    goal_pose9: np.ndarray,
    relative_frame9: np.ndarray,
    eef_state20: np.ndarray,
    *,
    translation_scale_m: float = HAND_OBJECT_TRANSLATION_SCALE_M,
) -> np.ndarray:
    current_position, current_rotation = current_frame_from_goal_relative(
        goal_pose9, relative_frame9
    )
    return active_hand_object_frame11(
        current_position,
        current_rotation,
        eef_state20,
        translation_scale_m=translation_scale_m,
    )
