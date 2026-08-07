from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .augment_task_flow_with_oracle_frame import oracle_relative_frame


def test_oracle_relative_frame_encodes_translation_and_rotation_columns() -> None:
    payload = {
        "goal_translation_error_xyz_m": np.asarray(
            [[0.1, -0.2, 0.3]], dtype=np.float32
        ),
        "goal_rotation_error_rotvec": np.asarray(
            [[0.0, 0.0, np.pi / 2]], dtype=np.float32
        ),
    }
    result = oracle_relative_frame(payload)
    expected_rotation = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    assert result.shape == (1, 9)
    assert np.allclose(result[0, :3], payload["goal_translation_error_xyz_m"][0])
    assert np.allclose(result[0, 3:6], expected_rotation[:, 0], atol=1e-6)
    assert np.allclose(result[0, 6:9], expected_rotation[:, 1], atol=1e-6)
