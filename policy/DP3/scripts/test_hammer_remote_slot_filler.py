from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import hammer_remote_slot_filler as slot


class TestRemoteSlotFiller(unittest.TestCase):
    def _tasks(self, root: Path) -> list[slot.FormalTask]:
        return slot.build_tasks(str(root / "seed{seed}"))

    def _stopped_supervisor(
        self, tasks: list[slot.FormalTask]
    ) -> dict[str, slot.SupervisorProcess]:
        return {
            task.program: slot.SupervisorProcess("STOPPED", None)
            for task in tasks
        }

    def test_parse_supervisor_status(self):
        parsed = slot.parse_supervisor_status(
            "a RUNNING pid 17, uptime 1:00\n"
            "b STARTING\n"
            "c STOPPED Aug 06 11:27 AM\n"
        )
        self.assertEqual(slot.SupervisorProcess("RUNNING", 17), parsed["a"])
        self.assertEqual(slot.SupervisorProcess("STARTING", None), parsed["b"])
        self.assertEqual(slot.SupervisorProcess("STOPPED", None), parsed["c"])

    def test_supervisor_status_exit_three_is_normal_with_stopped_tasks(self):
        completed = mock.Mock(
            returncode=3,
            stdout="task STOPPED Aug 06 11:27 AM\n",
            stderr="",
        )
        with mock.patch.object(slot.subprocess, "run", return_value=completed):
            parsed = slot.query_supervisor("/usr/local/bin/supervisorctl")
        self.assertEqual(
            slot.SupervisorProcess("STOPPED", None),
            parsed["task"],
        )

    def test_cuda_processes_and_transition_reservation_share_cap(self):
        supervisor = {
            "hammer_seed20260805": slot.SupervisorProcess("RUNNING", 10),
            "hammer_seed20260806": slot.SupervisorProcess("RUNNING", 20),
            "task_starting": slot.SupervisorProcess("STARTING", None),
        }
        occupancy, reservations = slot.calculate_occupancy(
            {101, 201},
            supervisor,
            {101: 10, 10: 1, 201: 99, 99: 1, 20: 1},
            supervisor,
        )
        # PID101 is under seed05. Seed06 has no CUDA child and reserves one
        # transition slot; task_starting reserves another.
        self.assertEqual(4, occupancy)
        self.assertEqual(
            {"hammer_seed20260806", "task_starting"}, set(reservations)
        )

    def test_four_cuda_main_processes_never_start_a_fifth(self):
        with tempfile.TemporaryDirectory() as temporary:
            tasks = self._tasks(Path(temporary))
            started: list[str] = []
            result = slot.fill_once(
                tasks=tasks,
                max_active=4,
                supervisorctl="unused",
                starter=started.append,
                cuda_pids={1, 2, 3, 4},
                supervisor=self._stopped_supervisor(tasks),
                parents={},
                commands={},
            )
            self.assertEqual([], started)
            self.assertEqual(4, result["occupancy"])

    def test_no_ce_pair_is_started_before_no_supcon(self):
        with tempfile.TemporaryDirectory() as temporary:
            tasks = self._tasks(Path(temporary))
            started: list[str] = []
            slot.fill_once(
                tasks=tasks,
                max_active=4,
                supervisorctl="unused",
                starter=started.append,
                cuda_pids={1, 2},
                supervisor=self._stopped_supervisor(tasks),
                parents={},
                commands={},
            )
            self.assertEqual(
                [
                    "hammer_slot_nce_seed20260805",
                    "hammer_slot_nce_seed20260806",
                ],
                started,
            )

    def test_incomplete_run_is_blocked_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            tasks = self._tasks(Path(temporary))
            blocked = tasks[0]
            blocked.run_dir.mkdir(parents=True)
            started: list[str] = []
            result = slot.fill_once(
                tasks=tasks,
                max_active=3,
                supervisorctl="unused",
                starter=started.append,
                cuda_pids={1, 2},
                supervisor=self._stopped_supervisor(tasks),
                parents={},
                commands={},
            )
            self.assertEqual(
                "blocked_incomplete",
                result["states"]["20260805/no_ce"],
            )
            self.assertNotIn(blocked.program, started)
            self.assertEqual(["hammer_slot_nce_seed20260806"], started)
            self.assertTrue(blocked.run_dir.is_dir())

    def test_filesystem_is_rechecked_at_toctou_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            tasks = self._tasks(Path(temporary))
            target = tasks[0]
            original = slot.classify_task
            calls = 0

            def changing_state(task, active, supervisor):
                nonlocal calls
                state = original(task, active, supervisor)
                if task == target:
                    calls += 1
                    if calls == 2:
                        task.run_dir.mkdir(parents=True)
                        return "blocked_incomplete"
                return state

            started: list[str] = []
            with mock.patch.object(slot, "classify_task", changing_state):
                slot.fill_once(
                    tasks=tasks,
                    max_active=1,
                    supervisorctl="unused",
                    starter=started.append,
                    cuda_pids=set(),
                    supervisor=self._stopped_supervisor(tasks),
                    parents={},
                    commands={},
                )
            self.assertNotIn(target.program, started)
            self.assertEqual(["hammer_slot_nce_seed20260806"], started)

    def test_completion_markers_prevent_start(self):
        with tempfile.TemporaryDirectory() as temporary:
            tasks = self._tasks(Path(temporary))
            task = tasks[0]
            task.outer_completion.parent.mkdir(parents=True)
            task.outer_completion.write_text("done", encoding="utf-8")
            self.assertEqual(
                "complete",
                slot.classify_task(
                    task, set(), self._stopped_supervisor(tasks)
                ),
            )


if __name__ == "__main__":
    unittest.main()
