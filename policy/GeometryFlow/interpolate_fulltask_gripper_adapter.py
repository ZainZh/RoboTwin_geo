#!/usr/bin/env python3
"""Interpolate two frozen-trunk FullTask gripper-head checkpoints."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch


TRAINABLE_PREFIXES = (
    "gripper_geometry_adapter.",
    "observation_gripper_head.",
    "observation_gripper_position",
)


def is_gripper_head_tensor(name: str) -> bool:
    return name.startswith(TRAINABLE_PREFIXES)


def interpolate_adapter_state(
    low: dict[str, torch.Tensor],
    high: dict[str, torch.Tensor],
    alpha: float,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must lie in [0,1]")
    if set(low) != set(high):
        raise ValueError("checkpoint state dictionaries have different keys")
    result = {}
    changed = []
    for name, low_value in low.items():
        high_value = high[name]
        if low_value.shape != high_value.shape:
            raise ValueError(f"shape mismatch for {name}")
        if is_gripper_head_tensor(name):
            result[name] = low_value + float(alpha) * (high_value - low_value)
            if not torch.equal(low_value, high_value):
                changed.append(name)
        else:
            # Independent bf16 evaluation passes can round frozen tensors by
            # sub-micro precision. Keep the low checkpoint exactly, but reject
            # any material non-adapter change.
            if not torch.allclose(
                low_value.float(), high_value.float(), atol=2.0e-6, rtol=1.0e-6
            ):
                raise ValueError(f"non-adapter tensor changed: {name}")
            result[name] = low_value.clone()
    return result, changed


def graft_exact_trunk(
    adapted: dict[str, torch.Tensor], trunk: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Restore every non-adapter tensor bit-for-bit from a deployed trunk."""
    result = {}
    for name, value in adapted.items():
        if is_gripper_head_tensor(name):
            result[name] = value.clone()
            continue
        if name not in trunk or trunk[name].shape != value.shape:
            raise ValueError(f"trunk lacks compatible tensor {name}")
        result[name] = trunk[name].clone()
    unexpected = sorted(set(trunk) - set(adapted))
    if unexpected:
        raise ValueError(f"trunk has unexpected tensors: {unexpected}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low", type=Path, required=True)
    parser.add_argument("--high", type=Path, required=True)
    parser.add_argument(
        "--trunk",
        type=Path,
        help="Optional original checkpoint supplying the exact frozen non-adapter trunk.",
    )
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    low = torch.load(args.low, map_location="cpu", weights_only=False)
    high = torch.load(args.high, map_location="cpu", weights_only=False)
    for key in ("condition", "model_config", "statistics", "action_representation"):
        if low[key] != high[key]:
            raise ValueError(f"checkpoint metadata differs for {key}")
    model, changed = interpolate_adapter_state(
        low["model"], high["model"], args.alpha
    )
    trunk_path = None
    if args.trunk is not None:
        trunk = torch.load(args.trunk, map_location="cpu", weights_only=False)
        for key in ("condition", "statistics", "action_representation"):
            if trunk[key] != high[key]:
                raise ValueError(f"trunk metadata differs for {key}")
        model = graft_exact_trunk(model, trunk["model"])
        trunk_path = str(args.trunk.resolve())
    output = copy.deepcopy(high)
    output["model"] = model
    output["gripper_adapter_interpolation"] = {
        "low": str(args.low.resolve()),
        "high": str(args.high.resolve()),
        "alpha": float(args.alpha),
        "changed_tensors": changed,
        "selection_role": "training-trajectory offline admission",
        "exact_frozen_trunk": trunk_path,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(
        {
            "output": str(args.output.resolve()),
            "alpha": float(args.alpha),
            "changed_tensors": len(changed),
        }
    )


if __name__ == "__main__":
    main()
