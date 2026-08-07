from __future__ import annotations

import json
import unittest

import numpy as np

from script.evaluate_pose_correction_closedloop import _snapshot_to_jsonable


class SemanticSnapshotSerializationTest(unittest.TestCase):
    def test_numpy_snapshot_becomes_json_primitives(self):
        value = {
            "array": np.asarray([[1.0, 2.0]], dtype=np.float32),
            "scalar": np.int64(3),
            "nested": (np.float64(4.5), True, None),
        }
        converted = _snapshot_to_jsonable(value)
        self.assertEqual(
            converted,
            {
                "array": [[1.0, 2.0]],
                "scalar": 3,
                "nested": [4.5, True, None],
            },
        )
        json.dumps(converted, sort_keys=True)

    def test_unsupported_snapshot_value_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "unsupported value"):
            _snapshot_to_jsonable({"bad": object()})


if __name__ == "__main__":
    unittest.main()
