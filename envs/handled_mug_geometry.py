"""Pure geometry definitions for the handle-aware mug task."""

from __future__ import annotations

import numpy as np
import transforms3d as t3d


HANDLED_MUG_IDS = tuple(range(10))

# Columns are [handle, side, downward].  Mesh inspection shows that every
# handled 039_mug asset has local +Y as its height axis and local +Z as its
# handle direction.  +Z x -X = -Y, so this is a proper right-handed frame.
CUP_FUNCTIONAL_ROTATION = np.array(
    [
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)

# A flat marker's local +X is its indicated handle direction.  Its second and
# third axes are negated so that +X x -Y = -Z (the downward placement axis).
MARKER_FUNCTIONAL_BASE_ROTATION = np.diag([1.0, -1.0, -1.0])


def mug_functional_matrix() -> np.ndarray:
    """Return the category-level handled-mug frame at the mug bottom center."""

    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = CUP_FUNCTIONAL_ROTATION
    return result


def handle_marker_functional_matrix(
    marker_x: float, marker_y: float, marker_yaw: float
) -> np.ndarray:
    """Return the desired handled-mug frame encoded by the visible marker."""

    marker = np.eye(4, dtype=np.float64)
    marker[:3, :3] = t3d.euler.euler2mat(0.0, 0.0, float(marker_yaw))
    marker[:3, 3] = [float(marker_x), float(marker_y), 0.0]
    marker[:3, :3] = marker[:3, :3] @ MARKER_FUNCTIONAL_BASE_ROTATION
    return marker


def signed_axis_error_deg(source: np.ndarray, target: np.ndarray) -> float:
    """Angular error for a signed task axis."""

    source = np.asarray(source, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    source /= max(float(np.linalg.norm(source)), 1e-12)
    target /= max(float(np.linalg.norm(target)), 1e-12)
    return float(np.degrees(np.arccos(np.clip(source @ target, -1.0, 1.0))))
