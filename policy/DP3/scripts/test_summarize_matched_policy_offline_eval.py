import unittest

from summarize_matched_policy_offline_eval import aggregate_reports, summarize


def make_report(seed, route, raw, ema, validation_ids=(50, 67)):
    return {
        "route": route,
        "checkpoint_metadata": {"seed": seed},
        "dataset": {
            "validation_episode_ids": list(validation_ids),
            "validation_samples": 73,
            "n_episodes": 80,
        },
        "protocol": {
            "repeat_seeds": [1, 2],
            "repeats": 2,
            "paired_noise_across_raw_ema": True,
        },
        "results": {
            "raw": {"summary": {"mean": raw}},
            "ema": {"summary": {"mean": ema}},
        },
    }


class TestMatchedPolicySummary(unittest.TestCase):
    def test_summary_uses_independent_seed_means(self):
        routes = ["field", "xyz", "partprob"]
        reports = {
            10: {
                "field": make_report(10, "field", 1.0, 1.1),
                "xyz": make_report(10, "xyz", 2.0, 2.1),
                "partprob": make_report(10, "partprob", 0.5, 0.6),
            },
            11: {
                "field": make_report(11, "field", 3.0, 3.1),
                "xyz": make_report(11, "xyz", 2.0, 2.1),
                "partprob": make_report(11, "partprob", 1.5, 1.6),
            },
        }
        result = aggregate_reports(reports, routes)
        field = result["route_summaries"]["raw"]["field"]["across_training_seeds"]
        self.assertEqual(2, field["n"])
        self.assertAlmostEqual(2.0, field["mean"])
        self.assertAlmostEqual(2**0.5, field["std"])
        delta = result["pairwise"]["raw"]["field_minus_xyz"]
        self.assertEqual({"10": -1.0, "11": 1.0}, delta["per_training_seed_delta"])
        self.assertEqual(1, delta["left_lower_count"])
        self.assertEqual(1, delta["right_lower_count"])

    def test_mismatched_validation_split_fails(self):
        routes = ["field", "xyz"]
        reports = {
            10: {
                "field": make_report(10, "field", 1.0, 1.0),
                "xyz": make_report(10, "xyz", 2.0, 2.0, validation_ids=(50,)),
            },
            11: {
                "field": make_report(11, "field", 1.0, 1.0),
                "xyz": make_report(11, "xyz", 2.0, 2.0),
            },
        }
        with self.assertRaisesRegex(ValueError, "Validation split mismatch"):
            aggregate_reports(reports, routes)

    def test_empty_summary_fails(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            summarize([])


if __name__ == "__main__":
    unittest.main()
