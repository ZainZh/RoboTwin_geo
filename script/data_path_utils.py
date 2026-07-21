from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


DEFAULT_SERVER_STORAGE_ROOT = Path("/shared2/sz")
DEFAULT_SERVER_DATA_DIRNAME = "robotwin_data"


@dataclass(frozen=True)
class DataPaths:
    mode: str
    repo_root: Path
    server_storage_root: Path
    raw_data_root: Path
    dp3_data_root: Path


def resolve_data_paths(
    *,
    repo_root: str | Path | None = None,
    server_storage_root: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> DataPaths:
    env = os.environ if environ is None else environ
    repo = Path(repo_root or Path(__file__).resolve().parents[1]).expanduser().resolve()
    server = Path(
        server_storage_root
        or env.get("ROBOTWIN_SERVER_STORAGE_ROOT", DEFAULT_SERVER_STORAGE_ROOT)
    ).expanduser().resolve()
    server_data_root = Path(
        env.get("ROBOTWIN_SERVER_DATA_ROOT", server / DEFAULT_SERVER_DATA_DIRNAME)
    ).expanduser().resolve()
    server_raw_root = Path(
        env.get("ROBOTWIN_SERVER_RAW_DATA_ROOT", server_data_root / "data")
    ).expanduser().resolve()
    mode = "server" if server_raw_root.is_dir() else "local"

    if mode == "server":
        default_raw = server_raw_root
        default_dp3 = server_data_root / "policy" / "DP3" / "data"
    else:
        default_raw = repo / "data"
        default_dp3 = repo / "policy" / "DP3" / "data"

    raw_root = Path(env.get("ROBOTWIN_RAW_DATA_ROOT", default_raw)).expanduser().resolve()
    dp3_root = Path(env.get("ROBOTWIN_DP3_DATA_ROOT", default_dp3)).expanduser().resolve()
    return DataPaths(
        mode=mode,
        repo_root=repo,
        server_storage_root=server,
        raw_data_root=raw_root,
        dp3_data_root=dp3_root,
    )


def raw_task_data_dir(task_name: str, task_config: str) -> Path:
    return resolve_data_paths().raw_data_root / str(task_name) / str(task_config)


def dp3_data_path(name: str) -> Path:
    root = resolve_data_paths().dp3_data_root
    root.mkdir(parents=True, exist_ok=True)
    return root / str(name)
