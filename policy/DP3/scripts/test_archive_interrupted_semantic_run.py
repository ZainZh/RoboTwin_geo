from __future__ import annotations

import fcntl
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch

from archive_interrupted_semantic_run import archive_run, sha256


SCRIPT_DIR = Path(__file__).resolve().parent


class TestInterruptedArchive(unittest.TestCase):
    def _make_run(self, root: Path, run_name: str = "hammer_no_supcon_e4000_split42_seed7"):
        run_dir = root / run_name
        log_dir = root / "logs"
        run_dir.mkdir(parents=True)
        log_dir.mkdir()
        config = {
            "run_name": run_name,
            "seed": 7,
            "epochs": 4000,
            "sem_ce_weight": 1.0,
            "sem_contrastive_weight": 0.0,
            "sem_consistency_weight": 0.1,
        }
        (run_dir / "config.json").write_text(
            json.dumps(config, sort_keys=True), encoding="utf-8"
        )
        payload = {
            "format": "semantic_field_weights_only_v1",
            "training_resume_supported": False,
            "epoch": 7,
            "global_step": 56,
            "args": config,
            "projector_state_dict": {},
        }
        for name in ("last.pt", "best.pt", "best_sem.pt"):
            torch.save(payload, run_dir / name)
        log_path = log_dir / f"{run_name}.log"
        log_path.write_text("[train] epoch=7 loss=1.0\n", encoding="utf-8")
        proc_root = root / "proc"
        proc_root.mkdir()
        return run_name, run_dir, log_path, proc_root

    def test_archive_preserves_sha_epoch_config_and_vacates_standard_name(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_name, run_dir, log_path, proc_root = self._make_run(root)
            original_config_sha = sha256(run_dir / "config.json")
            original_log_sha = sha256(log_path)
            destination = archive_run(
                root,
                run_name,
                timestamp="20260806T120000Z",
                proc_root=proc_root,
            )
            self.assertFalse(run_dir.exists())
            self.assertFalse(log_path.exists())
            self.assertTrue((destination / "run" / "last.pt").is_file())
            self.assertTrue((destination / "log.log").is_file())
            manifest = json.loads(
                (destination / "archive_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(7, manifest["checkpoint_epoch"])
            self.assertEqual(4000, manifest["target_epoch"])
            self.assertEqual(original_config_sha, manifest["config_sha256"])
            self.assertEqual(original_log_sha, manifest["log_sha256"])
            self.assertEqual(0.0, manifest["config"]["sem_contrastive_weight"])

    def test_inner_completion_blocks_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_name, run_dir, log_path, proc_root = self._make_run(root)
            (run_dir / "completion.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "completed/finalized"):
                archive_run(root, run_name, proc_root=proc_root)
            self.assertTrue(run_dir.is_dir())
            self.assertTrue(log_path.is_file())

    def test_held_flock_blocks_archive_without_moving_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_name, run_dir, log_path, proc_root = self._make_run(root)
            lock_path = root / f".{run_name}.lock"
            with lock_path.open("a+", encoding="utf-8") as lock_handle:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                with self.assertRaisesRegex(RuntimeError, "run lock is held"):
                    archive_run(root, run_name, proc_root=proc_root)
            self.assertTrue(run_dir.is_dir())
            self.assertTrue(log_path.is_file())

    def test_active_training_cmdline_blocks_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_name, run_dir, log_path, proc_root = self._make_run(root)
            process = proc_root / "123"
            process.mkdir()
            command = (
                f"python\0train_semantic_field_loss_ablation_fast.py\0"
                f"--run-name\0{run_name}\0"
            )
            (process / "cmdline").write_bytes(command.encode())
            with self.assertRaisesRegex(RuntimeError, "active training PIDs"):
                archive_run(root, run_name, proc_root=proc_root)
            self.assertTrue(run_dir.is_dir())
            self.assertTrue(log_path.is_file())

    def test_shell_wrapper_has_valid_syntax(self):
        subprocess.run(
            [
                "bash",
                "-n",
                str(SCRIPT_DIR / "archive_interrupted_semantic_run.sh"),
            ],
            check=True,
        )


if __name__ == "__main__":
    unittest.main()
