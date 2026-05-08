import json
import tempfile
import unittest
from pathlib import Path

from pointwise_preprocess_meta import (
    load_or_init_meta,
    reconcile_episode_stats,
    write_meta,
)


class PointwisePreprocessMetaTest(unittest.TestCase):
    def test_load_init_and_atomic_write(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_path = Path(tmpdir) / "meta.json"
            meta = load_or_init_meta(
                str(meta_path),
                task_name="hanging_mug",
                task_config="demo_clean_3d_object_pc",
                expert_data_num=50,
            )
            self.assertEqual(meta["task_name"], "hanging_mug")
            self.assertEqual(meta["episodes"], [])

            meta["episodes"].append({"episode": 0})
            write_meta(str(meta_path), meta)

            loaded = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["episodes"], [{"episode": 0}])

    def test_reconcile_episode_stats_truncates_and_backfills(self):
        stats = [{"episode": 0}, {"episode": 1}, {"episode": 2}]
        reconciled = reconcile_episode_stats(stats, start_episode=2)
        self.assertEqual(reconciled, [{"episode": 0}, {"episode": 1}])

        reconciled = reconcile_episode_stats([], start_episode=2)
        self.assertEqual(
            reconciled,
            [
                {"episode": 0, "recovered_without_stats": True},
                {"episode": 1, "recovered_without_stats": True},
            ],
        )


if __name__ == "__main__":
    unittest.main()
