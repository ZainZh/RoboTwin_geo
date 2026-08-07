from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from semantic_field_run_lock import (
    run_fresh_locked,
    validate_completed_run,
    write_completion_marker,
)


def _make_args(root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        output_dir=root,
        run_name="locked_run",
        epochs=3,
        resume=None,
        resume_additional_epochs=0,
        resume_from=None,
        auto_resume=False,
        seed=7,
        train_mode="semantic",
        compute_elided_zero_weight_losses=True,
    )


def _write_complete_run(root: Path, args: argparse.Namespace) -> Path:
    run_dir = root / args.run_name
    run_dir.mkdir(parents=True)
    primitive_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    (run_dir / "config.json").write_text(
        json.dumps(primitive_args, sort_keys=True),
        encoding="utf-8",
    )
    payload = {
        "format": "semantic_field_weights_only_v1",
        "training_resume_supported": False,
        "epoch": args.epochs,
        "args": primitive_args,
        "projector_state_dict": {},
    }
    for name in ("last.pt", "best.pt", "best_sem.pt"):
        torch.save(payload, run_dir / name)
    return run_dir


class _Parser:
    def __init__(self, args):
        self.args = args

    def parse_args(self):
        return self.args


class TestSemanticFieldRunLock(unittest.TestCase):
    def test_marker_hashes_detect_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = _make_args(root)
            run_dir = _write_complete_run(root, args)
            validate_completed_run(run_dir, args)
            marker = write_completion_marker(run_dir, args)
            self.assertTrue(marker.is_file())
            validate_completed_run(run_dir, args)

            with (run_dir / "config.json").open("a", encoding="utf-8") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                validate_completed_run(run_dir, args)

    def test_existing_complete_run_skips_callback_and_backfills_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = _make_args(root)
            run_dir = _write_complete_run(root, args)
            calls = []
            trainer = SimpleNamespace(
                build_argparser=lambda: _Parser(args),
                main=lambda: calls.append("called"),
            )
            run_fresh_locked(trainer)
            self.assertEqual([], calls)
            self.assertTrue((run_dir / "completion.json").is_file())
            validate_completed_run(run_dir, args)

    def test_existing_incomplete_run_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = _make_args(root)
            (root / args.run_name).mkdir()
            trainer = SimpleNamespace(
                build_argparser=lambda: _Parser(args),
                main=lambda: self.fail("incomplete run must not be overwritten"),
            )
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                run_fresh_locked(trainer)

    def test_auto_resume_rejects_legacy_partial_without_full_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = _make_args(root)
            args.auto_resume = True
            run_dir = root / args.run_name
            run_dir.mkdir()
            primitive_args = {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            }
            (run_dir / "config.json").write_text(
                json.dumps(primitive_args, sort_keys=True), encoding="utf-8"
            )
            torch.save(
                {
                    "format": "semantic_field_weights_only_v1",
                    "training_resume_supported": False,
                    "epoch": 1,
                    "args": primitive_args,
                    "projector_state_dict": {},
                },
                run_dir / "last.pt",
            )
            trainer = SimpleNamespace(build_argparser=lambda: _Parser(args))
            with self.assertRaisesRegex(RuntimeError, "resume checkpoint is unavailable"):
                run_fresh_locked(trainer, lambda: self.fail("legacy partial ran"))

if __name__ == "__main__":
    unittest.main()
