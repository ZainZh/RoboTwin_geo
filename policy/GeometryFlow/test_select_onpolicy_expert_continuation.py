import numpy as np

from policy.GeometryFlow.select_onpolicy_expert_continuation import select_rows


def test_selects_complete_expert_suffix_only():
    payload = {
        "episode_id": np.array([0, 0, 0, 0, 1, 1, 1]),
        "frame_index": np.array([0, 1, 2, 3, 0, 1, 2]),
        "action": np.arange(7 * 2).reshape(7, 2),
        "scalar": np.array(3),
    }
    records = [
        {
            "episode": 0,
            "policy_controls": 2,
            "recovered": True,
            "expert_continuation": "full_task",
        },
        {
            "episode": 1,
            "policy_controls": 1,
            "recovered": True,
            "expert_continuation": "full_task",
        },
    ]
    selected, summary = select_rows(payload, records)
    np.testing.assert_array_equal(selected["episode_id"], [0, 0, 1, 1])
    np.testing.assert_array_equal(selected["frame_index"], [2, 3, 1, 2])
    assert selected["scalar"] == 3
    assert summary["0"]["kept_samples"] == 2
    assert summary["1"]["first_kept_frame"] == 1
