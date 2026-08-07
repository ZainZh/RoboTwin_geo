from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import torch

import hammer_remote_slot_filler as legacy_remote
import hammer_remote_slot_filler_stage42 as remote_stage42
import hammer_stage42_local_queue as local_queue
from hammer_stage42_protocol import (
    EPOCHS,
    LOCAL_SEEDS,
    REMOTE_SEEDS,
    SPLIT_SEED,
    TEST_RATIO,
    VAL_RATIO,
    VARIANTS,
    Stage42Task,
    audit_completed_run,
    build_tasks,
    expected_config,
)


SCRIPT_DIR = Path(__file__).resolve().parent
LAUNCHER = SCRIPT_DIR / "run_hammer_stage42_independent_matrix.sh"
REMOTE_WRAPPER = SCRIPT_DIR / "run_remote_hammer_stage42_variant.sh"
REMOTE_CONF = SCRIPT_DIR / "supervisor_hammer_stage42_remote_tasks.conf"


class TestStage42ProtocolAndLauncher(unittest.TestCase):
    def _dry_run(self, role: str, **extra: str) -> list[str]:
        environment = os.environ.copy()
        environment.update(
            {
                "STAGE42_ROLE": role,
                "RUN_STAGE42": "0",
                # Conflicting legacy variables must not leak into stage42.
                "SPLIT_SEED": "42",
                "VAL_RATIO": "0.9",
                "TEST_RATIO": "0.0",
                **extra,
            }
        )
        completed = subprocess.run(
            ["bash", str(LAUNCHER)],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        return completed.stdout.splitlines()

    def test_shell_syntax_and_default_dry_run(self):
        for path in (LAUNCHER, REMOTE_WRAPPER):
            subprocess.run(["bash", "-n", str(path)], check=True)
        lines = self._dry_run("local")
        self.assertEqual(8, len(lines))
        for seed in LOCAL_SEEDS:
            for variant in VARIANTS:
                matches = [
                    line for line in lines
                    if f"split424242_seed{seed}" in line and f"hammer_{variant}_" in line
                ]
                self.assertEqual(1, len(matches), (seed, variant))
        for line in lines:
            self.assertIn("--split-seed 424242", line)
            self.assertIn("--val-ratio 0.1", line)
            self.assertIn("--test-ratio 0.2", line)
            self.assertIn("--epochs 1750", line)
            self.assertIn("--batch-size 6", line)
            self.assertIn("--num-workers 4", line)
            self.assertIn("--auto-resume", line)
            self.assertIn("train_semantic_field_loss_ablation_fast.py", line)

    def test_remote_assignment_is_exactly_seed06_four_variants(self):
        lines = self._dry_run("remote")
        self.assertEqual(4, len(lines))
        self.assertTrue(all("--seed 20260806" in line for line in lines))
        self.assertTrue(all("--num-workers 8" in line for line in lines))
        self.assertTrue(all("remote6000ada_seed20260806" in line for line in lines))

    def test_run_gate_and_machine_assignment_fail_closed(self):
        environment = os.environ.copy()
        environment.update({"STAGE42_ROLE": "local", "RUN_STAGE42": "maybe"})
        result = subprocess.run(
            ["bash", str(LAUNCHER)], capture_output=True, text=True, env=environment
        )
        self.assertNotEqual(0, result.returncode)
        environment.update(
            {
                "RUN_STAGE42": "0",
                "STAGE42_ROLE": "remote",
                "STAGE42_SEEDS": "20260805",
            }
        )
        result = subprocess.run(
            ["bash", str(LAUNCHER)], capture_output=True, text=True, env=environment
        )
        self.assertNotEqual(0, result.returncode)

    def test_unique_roots_and_frozen_split(self):
        local = build_tasks("local")
        remote = build_tasks("remote")
        self.assertEqual(8, len(local))
        self.assertEqual(4, len(remote))
        self.assertEqual(set(LOCAL_SEEDS), {task.seed for task in local})
        self.assertEqual(set(REMOTE_SEEDS), {task.seed for task in remote})
        self.assertTrue(all(f"split{SPLIT_SEED}" in task.run_name for task in local + remote))
        roots_by_seed = {(task.role, task.seed): task.output_root for task in local + remote}
        self.assertEqual(3, len(roots_by_seed))
        self.assertEqual(3, len({str(root) for root in roots_by_seed.values()}))
        self.assertEqual(VAL_RATIO, expected_config(local[0])["val_ratio"])
        self.assertEqual(TEST_RATIO, expected_config(local[0])["test_ratio"])

    def _make_complete(self, task: Stage42Task) -> None:
        task.run_dir.mkdir(parents=True)
        task.lock_path.touch()
        config = expected_config(task)
        config.update(
            {
                "dataset_root": "/fake/data",
                "alias_config": "/fake/hammer.json",
                "utonia_checkpoint": "/fake/utonia.pth",
                "device": "cuda:0",
            }
        )
        (task.run_dir / "config.json").write_text(
            json.dumps(config, sort_keys=True), encoding="utf-8"
        )
        for name, epoch in (("last.pt", EPOCHS), ("best.pt", 123), ("best_sem.pt", 321)):
            torch.save(
                {
                    "format": "semantic_field_weights_only_v1",
                    "training_resume_supported": False,
                    "epoch": epoch,
                    "args": config,
                },
                task.run_dir / name,
            )
        controls = {"resume", "resume_additional_epochs", "resume_from", "auto_resume"}
        identity_config = {key: value for key, value in config.items() if key not in controls}
        torch.save(
            {
                "format": "semantic_field_training_state_v1",
                "training_resume_supported": True,
                "resume_epoch_boundary": True,
                "epoch": EPOCHS,
                "args": config,
                "run_identity": {
                    "run_name": task.run_name,
                    "seed": int(task.seed),
                    "config": identity_config,
                    "config_sha256": "covered-by-training-state-unit-tests",
                },
            },
            task.run_dir / "resume.pt",
        )
        names = ("config.json", "last.pt", "best.pt", "best_sem.pt", "resume.pt")
        hashes = {}
        import hashlib
        for name in names:
            hashes[name] = hashlib.sha256((task.run_dir / name).read_bytes()).hexdigest()
        task.completion.write_text(
            json.dumps(
                {
                    "format": "semantic_field_completion_v1",
                    "run_name": task.run_name,
                    "target_epoch": EPOCHS,
                    "compute_elided_zero_weight_losses": True,
                    "sha256": hashes,
                }
            ),
            encoding="utf-8",
        )

    def test_weights_only_completion_identity_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            task = build_tasks(
                "local",
                output_template=str(Path(temporary) / "seed{seed}"),
                seeds=["20260805"],
                variants=["full_fixed"],
            )[0]
            self._make_complete(task)
            report = audit_completed_run(task)
            self.assertEqual("verified_complete", report["status"])
            config_path = task.run_dir / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["test_ratio"] = 0.0
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                audit_completed_run(task)


class TestLocalQueue(unittest.TestCase):
    def _fixture(self, root: Path):
        tasks = build_tasks(
            "local",
            output_template=str(root / "stage42-seed{seed}"),
        )
        guards = [
            local_queue.LegacyGuard(
                variant=variant,
                unit=f"legacy-{variant}.service",
                completion=root / "legacy" / variant / "completion.json",
            )
            for variant in VARIANTS
        ]
        units = {
            guard.unit: local_queue.UnitState("loaded", "active", "running", 10 + i)
            for i, guard in enumerate(guards)
        }
        units.update({task.program: local_queue.UnitState() for task in tasks})
        cuda = {7060, 101, 201, 301, 401}
        commands = {
            7060: ("/usr/libexec/gnome-remote-desktop-daemon",),
            101: ("python", "train_semantic_field_loss_ablation_fast.py", "--run-name", "legacy-a"),
            201: ("python", "train_semantic_field_loss_ablation_fast.py", "--run-name", "legacy-b"),
            301: ("python", "train_semantic_field_loss_ablation_fast.py", "--run-name", "legacy-c"),
            401: ("python", "train_semantic_field_loss_ablation_fast.py", "--run-name", "legacy-d"),
        }
        parents = {101: 10, 201: 11, 301: 12, 401: 13, 10: 1, 11: 1, 12: 1, 13: 1, 7060: 1}
        return tasks, guards, units, cuda, commands, parents

    def test_four_training_processes_and_desktop_never_start_fifth(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            tasks, guards, units, cuda, commands, parents = fixture
            with mock.patch.object(local_queue, "attempt_path", lambda task: Path(temporary) / "attempt" / task.run_name):
                snapshot = local_queue.fill_once(
                    tasks=tasks, guards=guards, max_active=4, allow_start=False,
                    cuda_pids=cuda, units=units, commands=commands, parents=parents,
                )
            self.assertEqual(4, snapshot["occupancy"])
            self.assertEqual([7060], snapshot["ignored_graphics_cuda_pids"])
            self.assertEqual([], snapshot["would_starts"])

    def test_successful_legacy_completion_releases_exactly_one_slot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tasks, guards, units, cuda, commands, parents = self._fixture(root)
            guard = guards[0]
            guard.completion.parent.mkdir(parents=True)
            guard.completion.touch()
            units[guard.unit] = local_queue.UnitState("loaded", "inactive", "dead", None)
            cuda.remove(101)
            commands.pop(101)
            started = []
            with mock.patch.object(local_queue, "attempt_path", lambda task: root / "attempt" / task.run_name):
                snapshot = local_queue.fill_once(
                    tasks=tasks, guards=guards, max_active=4, allow_start=True,
                    starter=started.append, cuda_pids=cuda, units=units,
                    commands=commands, parents=parents,
                )
            self.assertEqual([tasks[0]], started)
            self.assertEqual([tasks[0].program], snapshot["starts"])

    def test_inactive_legacy_without_completion_blocks_queue(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tasks, guards, units, cuda, commands, parents = self._fixture(root)
            guard = guards[0]
            units[guard.unit] = local_queue.UnitState("loaded", "failed", "failed", None)
            cuda.remove(101)
            commands.pop(101)
            with mock.patch.object(local_queue, "attempt_path", lambda task: root / "attempt" / task.run_name):
                snapshot = local_queue.fill_once(
                    tasks=tasks, guards=guards, max_active=4, allow_start=False,
                    cuda_pids=cuda, units=units, commands=commands, parents=parents,
                )
            self.assertTrue(snapshot["legacy_blockers"])
            self.assertEqual([], snapshot["would_starts"])

    def test_systemd_command_contains_run_gate_and_resource_limits(self):
        task = build_tasks("local", seeds=["20260805"], variants=["no_ce"])[0]
        command = local_queue.systemd_run_command(task)
        rendered = " ".join(command)
        self.assertIn("RUN_STAGE42=1", command)
        self.assertIn("STAGE42_ROLE=local", command)
        self.assertIn("STAGE42_SEEDS=20260805", command)
        self.assertIn("STAGE42_VARIANTS=no_ce", command)
        self.assertIn("--property=MemoryMax=12G", command)
        self.assertIn("run_hammer_stage42_independent_matrix.sh", rendered)


class TestRemoteSuccessor(unittest.TestCase):
    def _supervisor(self, tasks):
        return {
            task.program: legacy_remote.SupervisorProcess("STOPPED", None)
            for task in tasks
        }

    def test_stage42_tasks_are_seed06_split424242_and_disabled_conf(self):
        tasks = remote_stage42.build_stage42_tasks("/tmp/stage42-seed{seed}")
        self.assertEqual(4, len(tasks))
        self.assertTrue(all(task.seed == "20260806" for task in tasks))
        self.assertTrue(all("split424242" in task.run_name for task in tasks))
        text = REMOTE_CONF.read_text(encoding="utf-8")
        self.assertEqual(4, text.count("autostart=false"))
        self.assertEqual(4, text.count('RUN_STAGE42="1"'))

    def test_pending_legacy_has_priority_over_stage42(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy_tasks = legacy_remote.build_tasks(str(root / "legacy-seed{seed}"))
            stage_tasks = remote_stage42.build_stage42_tasks(str(root / "stage-seed{seed}"))
            supervisor = self._supervisor(legacy_tasks + stage_tasks)
            started = []
            snapshot = remote_stage42.fill_once(
                legacy_tasks=legacy_tasks, stage42_tasks=stage_tasks,
                include_stage42=True, max_active=1, supervisorctl="unused",
                starter=started.append, cuda_pids=set(), supervisor=supervisor,
                parents={}, commands={},
            )
            self.assertEqual("legacy", snapshot["active_phase"])
            self.assertEqual([legacy_tasks[0].program], started)
            self.assertTrue(all(task.program not in started for task in stage_tasks))

    def test_active_legacy_allows_stage42_to_backfill_free_slot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy_task = legacy_remote.build_tasks(
                str(root / "legacy-seed{seed}")
            )[0]
            stage_tasks = remote_stage42.build_stage42_tasks(
                str(root / "stage-seed{seed}")
            )
            supervisor = self._supervisor([legacy_task, *stage_tasks])
            command = (
                "python", "train_semantic_field_loss_ablation_fast.py",
                "--run-name", legacy_task.run_name,
            )
            started = []
            snapshot = remote_stage42.fill_once(
                legacy_tasks=[legacy_task], stage42_tasks=stage_tasks,
                include_stage42=True, max_active=2, supervisorctl="unused",
                starter=started.append, cuda_pids={101}, supervisor=supervisor,
                parents={101: 1}, commands={101: command},
            )
            self.assertFalse(snapshot["legacy_barrier_open"])
            self.assertEqual([stage_tasks[0].program], started)

    def test_stage42_starts_only_after_all_legacy_outer_markers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy_tasks = legacy_remote.build_tasks(str(root / "legacy-seed{seed}"))
            stage_tasks = remote_stage42.build_stage42_tasks(str(root / "stage-seed{seed}"))
            for task in legacy_tasks:
                task.outer_completion.parent.mkdir(parents=True, exist_ok=True)
                task.outer_completion.touch()
            supervisor = self._supervisor(legacy_tasks + stage_tasks)
            started = []
            snapshot = remote_stage42.fill_once(
                legacy_tasks=legacy_tasks, stage42_tasks=stage_tasks,
                include_stage42=True, max_active=1, supervisorctl="unused",
                starter=started.append, cuda_pids=set(), supervisor=supervisor,
                parents={}, commands={},
            )
            self.assertTrue(snapshot["legacy_barrier_open"])
            self.assertEqual("stage42", snapshot["active_phase"])
            self.assertEqual([stage_tasks[0].program], started)


if __name__ == "__main__":
    unittest.main()
