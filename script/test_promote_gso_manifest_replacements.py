import unittest

from script.promote_gso_manifest_replacements import build_effective_manifest


class PromoteGsoManifestReplacementsTest(unittest.TestCase):
    def test_reserve_is_promoted_without_policy_outcomes(self) -> None:
        parent = {
            "schema_version": 1,
            "assets": [
                {"selection_rank": 3, "name": "bad", "partition": "development"},
                {"selection_rank": 20, "name": "reserve", "partition": "reserve"},
            ],
        }
        result = build_effective_manifest(parent, [(3, 20, "objective_gate")])
        self.assertEqual(len(result["assets"]), 1)
        promoted = result["assets"][0]
        self.assertEqual(promoted["selection_rank"], 20)
        self.assertEqual(promoted["partition"], "development")
        self.assertEqual(promoted["original_partition"], "reserve")
        self.assertFalse(result["effective_manifest"]["policy_outcomes_used"])

    def test_nonreserve_promotion_is_rejected(self) -> None:
        parent = {
            "assets": [
                {"selection_rank": 3, "name": "bad", "partition": "development"},
                {"selection_rank": 4, "name": "blind", "partition": "blind_test"},
            ]
        }
        with self.assertRaisesRegex(ValueError, "not frozen reserve"):
            build_effective_manifest(parent, [(3, 4, "objective_gate")])


if __name__ == "__main__":
    unittest.main()
