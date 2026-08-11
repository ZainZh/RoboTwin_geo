import unittest
import json
import tempfile
from pathlib import Path

from script.eval_policy import (
    load_evaluation_seed_file,
    parse_bool,
    resolve_evaluation_output_root,
)


class TestEvalPolicyHelpers(unittest.TestCase):
    def test_evaluation_output_root_prefers_explicit_then_tagrt_root(self):
        self.assertEqual(
            resolve_evaluation_output_root(
                {"evaluation_output_root": "/tmp/explicit"},
                {"TAGRT_ARTIFACT_ROOT": "/tmp/tagrt"},
            ),
            Path("/tmp/explicit"),
        )
        self.assertEqual(
            resolve_evaluation_output_root({}, {"TAGRT_ARTIFACT_ROOT": "/tmp/tagrt"}),
            Path("/tmp/tagrt/experiment_outputs/robotwin_eval_result"),
        )
        self.assertEqual(resolve_evaluation_output_root({}, {}), Path("eval_result"))

    def test_parse_bool(self):
        for value in (True, "true", "YES", 1):
            self.assertTrue(parse_bool(value))
        for value in (False, "false", "Off", 0):
            self.assertFalse(parse_bool(value))

    def test_parse_bool_rejects_ambiguous_values(self):
        with self.assertRaises(ValueError):
            parse_bool("sometimes")

    def test_load_evaluation_seed_file_accepts_logged_format_and_slices(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "accepted.json"
            path.write_text(
                json.dumps({"start_seed": 100000, "accepted_seeds": [3, 7, 11]}),
                encoding="utf-8",
            )
            self.assertEqual(
                load_evaluation_seed_file(path, test_num=2),
                [3, 7],
            )

    def test_load_evaluation_seed_file_rejects_short_or_duplicate_lists(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "accepted.json"
            path.write_text(json.dumps([3]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fewer"):
                load_evaluation_seed_file(path, test_num=2)
            path.write_text(json.dumps([3, 3]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicates"):
                load_evaluation_seed_file(path, test_num=2)


if __name__ == "__main__":
    unittest.main()
