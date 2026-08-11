from pathlib import Path

from .summarize_densecontrol_admission import rejection_reason, summarize


def test_rejection_reason_prefers_actionable_terminal_failure():
    assert rejection_reason({"metrics": {"gripper_open": False}}) == "gripper_not_open"
    assert (
        rejection_reason(
            {"metrics": {"gripper_open": True, "shoe_ramp_contact": False}}
        )
        == "no_ramp_contact"
    )
    assert (
        rejection_reason(
            {
                "metrics": {
                    "gripper_open": True,
                    "shoe_ramp_contact": True,
                    "success_required_hold_steps": 25,
                    "max_success_hold_count": 20,
                }
            }
        )
        == "insufficient_stable_hold"
    )


def test_summary_joins_episode_and_seed_and_preserves_rejection():
    structural = {
        "data_root": "/tmp/data",
        "structurally_valid_episodes": 2,
        "episodes": [
            {
                "episode_index": 0,
                "seed": 22,
                "shoe_id": 2,
                "active_arm": "left",
                "structurally_valid": True,
            },
            {
                "episode_index": 1,
                "seed": 23,
                "shoe_id": 3,
                "active_arm": "right",
                "structurally_valid": True,
            },
        ],
    }
    replay = [
        {"episode_index": 0, "seed": 22, "success": True, "metrics": {}},
        {
            "episode_index": 1,
            "seed": 23,
            "success": False,
            "metrics": {
                "gripper_open": True,
                "shoe_ramp_contact": True,
                "success_required_hold_steps": 25,
                "max_success_hold_count": 20,
            },
        },
    ]
    result = summarize(structural, replay, Path("episodes.jsonl"))
    assert result["complete"]
    assert result["admitted_episodes"] == 1
    assert result["rejection_reasons"] == {"insufficient_stable_hold": 1}
    assert result["admitted_distribution"] == {"shoe2_left": 1}
