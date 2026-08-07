#!/usr/bin/env python3
"""Profile release semantic-field epochs without changing training behavior."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
RELEASE_ROOT = Path(
    os.environ.get(
        "SEMANTIC_FIELD_RELEASE_ROOT",
        REPO_ROOT / "include" / "3d_semantic_train" / "semantic_field_release",
    )
).expanduser().resolve()
UTONIA_ROOT = Path(os.environ["UTONIA_ROOT"]).expanduser().resolve()

sys.path.insert(0, str(UTONIA_ROOT))
sys.path.insert(0, str(RELEASE_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from train import train_utonia_universal_field as trainer  # noqa: E402
from train_semantic_field_loss_ablation_safe import main as safe_main  # noqa: E402


_run_epoch = trainer.run_epoch


def _timed_run_epoch(*args, **kwargs):
    optimizer = kwargs.get("optimizer")
    mode = "train" if optimizer is not None else "val"
    epoch = int(kwargs["epoch"])
    device = kwargs.get("device")
    is_cuda = str(device).startswith("cuda")
    if is_cuda:
        import torch

        torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    result = _run_epoch(*args, **kwargs)
    peak_mib = 0.0
    if is_cuda:
        torch.cuda.synchronize(device)
        peak_mib = torch.cuda.max_memory_allocated(device) / (1024**2)
    duration = time.perf_counter() - started
    print(
        f"[epoch-timing] mode={mode} epoch={epoch:03d} "
        f"seconds={duration:.6f} peak_allocated_mib={peak_mib:.3f}",
        flush=True,
    )
    return result


trainer.run_epoch = _timed_run_epoch


if __name__ == "__main__":
    safe_main()
