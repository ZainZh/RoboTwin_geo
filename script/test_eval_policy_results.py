import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from eval_policy_results import (  # noqa: E402
    EPISODE_SCHEMA,
    SUMMARY_SCHEMA,
    EvaluationResultRecorder,
    evaluation_should_continue,
    fixed_candidate_seeds,
    fast_eval_option_enabled,
    fixed_seed_protocol_enabled,
    policy_observation_digest,
    should_run_expert_check,
)


class FixedSeedProtocolTest(unittest.TestCase):
    def test_seed_zero_maps_exactly_to_100000_through_100099(self):
        seeds = fixed_candidate_seeds(seed_group=0, test_num=100)
        self.assertEqual(len(seeds), 100)
        self.assertEqual(seeds[0], 100000)
        self.assertEqual(seeds[-1], 100099)
        self.assertEqual(seeds, list(range(100000, 100100)))

    def test_fixed_budget_stops_after_candidates_even_when_one_is_skipped(self):
        self.assertTrue(evaluation_should_continue(
            candidate_count=99,
            accepted_count=98,
            test_num=100,
            fixed_seed_protocol=True,
        ))
        self.assertFalse(evaluation_should_continue(
            candidate_count=100,
            accepted_count=99,
            test_num=100,
            fixed_seed_protocol=True,
        ))
        self.assertTrue(evaluation_should_continue(
            candidate_count=100,
            accepted_count=99,
            test_num=100,
            fixed_seed_protocol=False,
        ))

    def test_formal_task_config_disables_video_and_expert_screening(self):
        config_path = REPO_ROOT / "task_config" / "demo_clean_3d_object_pc_hammer_formal_fixed.yml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.assertFalse(config["eval_video_log"])
        self.assertTrue(fixed_seed_protocol_enabled(config))
        self.assertFalse(should_run_expert_check(config))
        self.assertEqual(config["custom_hammer_eval"], {
            "enabled": True,
            "modelname": "020_hammer",
            "model_id": [0],
            "spawn_pose_mode": "match_reference_contact",
        })

    def test_fast_formal_config_preserves_protocol_and_disables_unused_visuals(self):
        base_path = REPO_ROOT / "task_config" / "demo_clean_3d_object_pc_hammer_formal_fixed.yml"
        fast_path = REPO_ROOT / "task_config" / "demo_clean_3d_object_pc_hammer_formal_fixed_fast.yml"
        base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
        fast = yaml.safe_load(fast_path.read_text(encoding="utf-8"))
        self.assertTrue(fixed_seed_protocol_enabled(fast))
        self.assertFalse(should_run_expert_check(fast))
        self.assertEqual(fast["custom_hammer_eval"], base["custom_hammer_eval"])
        self.assertEqual(fast["episode_num"], base["episode_num"])
        self.assertFalse(fast["data_type"]["rgb"])
        self.assertFalse(fast["data_type"]["third_view"])
        self.assertFalse(fast["data_type"]["camera_config"])
        self.assertTrue(fast_eval_option_enabled(fast, "skip_redundant_preflight"))
        self.assertTrue(fast_eval_option_enabled(fast, "skip_eval_planner"))
        self.assertTrue(fast_eval_option_enabled(fast, "skip_post_action_render"))
        self.assertTrue(fast_eval_option_enabled(fast, "reuse_last_action_observation"))
        self.assertFalse(fast_eval_option_enabled(base, "skip_redundant_preflight"))


class EvaluationResultRecorderTest(unittest.TestCase):
    def test_policy_observation_digest_ignores_mapping_order_but_detects_values(self):
        base = {
            "pointcloud": np.arange(18, dtype=np.float32).reshape(3, 6),
            "joint_action": {"vector": np.arange(14, dtype=np.float32)},
            "object_pointcloud": {
                "{B}": np.ones((2, 6), dtype=np.float32),
                "{A}": np.zeros((2, 6), dtype=np.float32),
            },
        }
        reordered = dict(base)
        reordered["object_pointcloud"] = {
            "{A}": base["object_pointcloud"]["{A}"],
            "{B}": base["object_pointcloud"]["{B}"],
        }
        changed = dict(reordered)
        changed["pointcloud"] = base["pointcloud"].copy()
        changed["pointcloud"][0, 0] = -1
        self.assertEqual(policy_observation_digest(base), policy_observation_digest(reordered))
        self.assertNotEqual(policy_observation_digest(base), policy_observation_digest(changed))

    def test_jsonl_and_summary_preserve_evaluated_and_skipped_seeds(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            recorder = EvaluationResultRecorder(root / "episodes.jsonl")
            recorder.record_candidate(
                candidate_index=0,
                candidate_seed=100000,
                accepted=True,
                evaluated=True,
                success=True,
                status="evaluated_success",
                evaluation_index=0,
                policy_steps=12,
                elapsed_seconds=1.25,
            )
            recorder.record_candidate(
                candidate_index=1,
                candidate_seed=100001,
                accepted=False,
                evaluated=False,
                success=None,
                status="skipped_unstable",
                reason="candidate_setup_unstable",
                error_type="UnStableError",
                error_message="object moved",
            )
            recorder.record_candidate(
                candidate_index=2,
                candidate_seed=100002,
                accepted=True,
                evaluated=True,
                success=False,
                status="evaluated_failure",
                reason="policy_did_not_reach_success_before_step_limit",
                evaluation_index=1,
            )

            summary = recorder.write_summary(
                root / "summary.json",
                start_seed=100000,
                next_seed=100003,
                requested_test_num=3,
                fixed_seed_protocol=True,
                metadata={"task_name": "beat_block_hammer"},
            )

            lines = (root / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
            records = [json.loads(line) for line in lines]
            self.assertEqual(len(records), 3)
            self.assertTrue(all(record["schema"] == EPISODE_SCHEMA for record in records))
            self.assertEqual(records[1]["candidate_seed"], 100001)
            self.assertFalse(records[1]["accepted"])
            self.assertFalse(records[1]["evaluated"])
            self.assertIsNone(records[1]["success"])
            self.assertEqual(records[1]["reason"], "candidate_setup_unstable")
            self.assertEqual(records[0]["policy_steps"], 12)
            self.assertEqual(records[0]["elapsed_seconds"], 1.25)

            written_summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(written_summary, summary)
            self.assertEqual(summary["schema"], SUMMARY_SCHEMA)
            self.assertEqual(summary["actual_candidate_seeds"], [100000, 100001, 100002])
            self.assertEqual(summary["actual_accepted_seeds"], [100000, 100002])
            self.assertEqual(summary["actual_evaluated_seeds"], [100000, 100002])
            self.assertEqual(summary["success_count"], 1)
            self.assertEqual(summary["success_rate"], 1 / 3)
            self.assertEqual(summary["evaluated_success_rate"], 1 / 2)
            self.assertEqual(summary["skipped_candidates"][0]["candidate_seed"], 100001)

    def test_record_validation_rejects_ambiguous_success(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = EvaluationResultRecorder(Path(temp_dir) / "episodes.jsonl")
            with self.assertRaisesRegex(ValueError, "require a boolean success"):
                recorder.record_candidate(
                    candidate_index=0,
                    candidate_seed=100000,
                    accepted=True,
                    evaluated=True,
                    success=None,
                    status="bad",
                )


if __name__ == "__main__":
    unittest.main()
