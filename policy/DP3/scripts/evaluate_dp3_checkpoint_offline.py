#!/usr/bin/env python3
"""Evaluate raw and EMA DP3 weights on a fixed held-out dataset split.

This evaluator intentionally accepts only the tensor-only companion format from
``safe_dp3_checkpoint.py``.  It never deserializes the legacy dill workspace
checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
DP3_ROOT = SCRIPT_DIR.parent
DP3_CODE_ROOT = DP3_ROOT / "3D-Diffusion-Policy"
if str(DP3_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(DP3_CODE_ROOT))
if str(DP3_ROOT) not in sys.path:
    sys.path.insert(0, str(DP3_ROOT))

from diffusion_policy_3d.common.pytorch_util import dict_apply  # noqa: E402
from diffusion_policy_3d.dataset.base_dataset import BaseDataset  # noqa: E402
from safe_dp3_checkpoint import load_weights_only_checkpoint  # noqa: E402


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_losses(losses: Iterable[float]) -> dict[str, float | int]:
    values = np.asarray(list(losses), dtype=np.float64)
    if values.size == 0:
        raise ValueError("Cannot summarize an empty loss sequence")
    std = float(values.std(ddof=1)) if values.size > 1 else 0.0
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": std,
        "sem": float(std / math.sqrt(values.size)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def validate_checkpoint_metadata(metadata: dict[str, Any], cfg: Any) -> None:
    expected_shape = OmegaConf.to_container(cfg.task.shape_meta, resolve=True)
    if metadata.get("shape_meta") != expected_shape:
        raise ValueError("Checkpoint/config shape_meta mismatch")
    checkpoint_zarr = Path(str(metadata.get("zarr_path", ""))).expanduser().resolve()
    config_zarr = Path(str(cfg.task.dataset.zarr_path)).expanduser().resolve()
    if checkpoint_zarr != config_zarr:
        raise ValueError(
            f"Checkpoint/config zarr mismatch: {checkpoint_zarr} != {config_zarr}"
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_state(
    *,
    model: torch.nn.Module,
    state_dict: dict[str, Any],
    val_dataloader: DataLoader,
    device: torch.device,
    repeat_seeds: list[int],
    max_batches: int | None,
) -> tuple[list[float], list[list[float]]]:
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    repeat_means: list[float] = []
    per_batch_losses: list[list[float]] = []
    with torch.no_grad():
        for repeat_seed in repeat_seeds:
            batch_losses: list[float] = []
            for batch_index, batch in enumerate(val_dataloader):
                if max_batches is not None and batch_index >= max_batches:
                    break
                batch = dict_apply(batch, lambda value: value.to(device, non_blocking=True))
                seed_everything(repeat_seed + batch_index)
                loss, _ = model.compute_loss(batch)
                batch_losses.append(float(loss.detach().cpu()))
            if not batch_losses:
                raise RuntimeError("Validation dataloader yielded no batches")
            repeat_means.append(float(np.mean(batch_losses)))
            per_batch_losses.append(batch_losses)
    return repeat_means, per_batch_losses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Trusted Hydra YAML from the matching run")
    parser.add_argument("--checkpoint", type=Path, required=True, help="weights-only DP3 companion")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--route", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--eval_seed", type=int, default=20260806)
    parser.add_argument("--max_batches", type=int, default=0, help="0 evaluates the full validation set")
    parser.add_argument("--skip_sha256", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats < 2:
        raise ValueError("--repeats must be at least 2 for uncertainty estimates")
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    config_path = args.config.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    cfg = OmegaConf.load(config_path)
    payload = load_weights_only_checkpoint(checkpoint_path)
    metadata = payload["metadata"]
    validate_checkpoint_metadata(metadata, cfg)

    seed_everything(int(cfg.training.seed))
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    if not isinstance(dataset, BaseDataset):
        raise TypeError(f"Expected BaseDataset, got {type(dataset)}")
    val_dataset = dataset.get_validation_dataset()
    if len(val_dataset) == 0:
        raise RuntimeError("Configured validation split is empty")
    normalizer = dataset.get_normalizer()
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=False,
    )

    model = hydra.utils.instantiate(cfg.policy)
    model.set_normalizer(normalizer)
    device = torch.device(args.device)
    repeat_seeds = [args.eval_seed + 1009 * index for index in range(args.repeats)]
    max_batches = None if args.max_batches == 0 else args.max_batches
    started = time.perf_counter()
    raw_losses, raw_batches = evaluate_state(
        model=model,
        state_dict=payload["model"],
        val_dataloader=val_dataloader,
        device=device,
        repeat_seeds=repeat_seeds,
        max_batches=max_batches,
    )
    results: dict[str, Any] = {
        "raw": {
            "summary": summarize_losses(raw_losses),
            "repeat_means": raw_losses,
            "per_batch_losses": raw_batches,
        }
    }
    ema_state = payload.get("ema_model")
    if ema_state is not None:
        ema_losses, ema_batches = evaluate_state(
            model=model,
            state_dict=ema_state,
            val_dataloader=val_dataloader,
            device=device,
            repeat_seeds=repeat_seeds,
            max_batches=max_batches,
        )
        results["ema"] = {
            "summary": summarize_losses(ema_losses),
            "repeat_means": ema_losses,
            "per_batch_losses": ema_batches,
        }
        paired_delta = np.asarray(ema_losses) - np.asarray(raw_losses)
        results["ema_minus_raw"] = {
            "summary": summarize_losses(paired_delta.tolist()),
            "repeat_deltas": paired_delta.tolist(),
        }

    val_episode_ids = np.flatnonzero(np.asarray(val_dataset.train_mask, dtype=bool)).tolist()
    if device.type == "cuda":
        peak_vram_bytes = int(torch.cuda.max_memory_allocated(device))
    else:
        peak_vram_bytes = 0
    report = {
        "schema_version": 1,
        "route": args.route,
        "evaluation_kind": "fixed_split_diffusion_imitation_loss",
        "paper_claim_eligible": False,
        "eligibility_note": (
            f"Single-seed checkpoint at epoch {metadata.get('epoch')}; offline imitation loss is "
            "diagnostic and is not a task success rate."
        ),
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": None if args.skip_sha256 else sha256_file(checkpoint_path),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "checkpoint_metadata": metadata,
        "dataset": {
            "zarr_path": str(Path(str(cfg.task.dataset.zarr_path)).expanduser().resolve()),
            "n_episodes": int(dataset.replay_buffer.n_episodes),
            "validation_episode_ids": val_episode_ids,
            "validation_samples": int(len(val_dataset)),
            "batch_size": args.batch_size,
            "evaluated_batches": min(len(val_dataloader), max_batches or len(val_dataloader)),
        },
        "protocol": {
            "repeat_seeds": repeat_seeds,
            "paired_noise_across_raw_ema": True,
            "repeats": args.repeats,
            "device": str(device),
        },
        "results": results,
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_vram_bytes": peak_vram_bytes,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "python_version": platform.python_version(),
            "hostname": platform.node(),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "results": results}, indent=2))


if __name__ == "__main__":
    main()
