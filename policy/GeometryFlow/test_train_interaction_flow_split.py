import unittest

from .train_interaction_flow_tokens import resolve_policy_object_split


class TestExplicitPolicyObjectSplit(unittest.TestCase):
    def test_none_preserves_registered_folds(self):
        self.assertIsNone(resolve_policy_object_split(None, None, None))

    def test_explicit_split_is_preserved(self):
        self.assertEqual(
            resolve_policy_object_split([0, 1], [2], [3]),
            {"train": (0, 1), "validation": (2,), "test": (3,)},
        )

    def test_partial_split_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "supplied together"):
            resolve_policy_object_split([0], None, [2])

    def test_overlap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "disjoint"):
            resolve_policy_object_split([0, 1], [1], [2])

    def test_duplicates_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            resolve_policy_object_split([0, 0], [1], [2])


if __name__ == "__main__":
    unittest.main()
