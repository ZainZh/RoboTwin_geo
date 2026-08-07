"""Safe, weights-only DP3 checkpoint helpers.

The legacy workspace checkpoint contains arbitrary pickle/dill objects.  Paper
experiments should export this simpler tensor-and-primitive payload so it can be
loaded with ``torch.load(..., weights_only=True)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch


FORMAT_NAME = "dp3_weights_only_v1"


def _state_dict_to_cpu(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
        for key, value in state_dict.items()
    }


def save_weights_only_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    ema_model: torch.nn.Module | None,
    metadata: Mapping[str, Any],
) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": FORMAT_NAME,
        "metadata": dict(metadata),
        "model": _state_dict_to_cpu(model.state_dict()),
        "ema_model": None if ema_model is None else _state_dict_to_cpu(ema_model.state_dict()),
    }
    torch.save(payload, destination)
    return destination


def load_weights_only_checkpoint(path: str | Path, *, mmap: bool = True) -> dict[str, Any]:
    payload = torch.load(
        Path(path).expanduser().resolve(),
        map_location="cpu",
        mmap=mmap,
        weights_only=True,
    )
    if not isinstance(payload, dict) or payload.get("format") != FORMAT_NAME:
        raise ValueError(f"Not a supported safe DP3 checkpoint: {path}")
    if not isinstance(payload.get("model"), dict):
        raise ValueError(f"Safe DP3 checkpoint has no model state_dict: {path}")
    return payload
