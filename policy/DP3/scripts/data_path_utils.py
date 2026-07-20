from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from script.data_path_utils import dp3_data_path, raw_task_data_dir, resolve_data_paths


__all__ = ("dp3_data_path", "raw_task_data_dir", "resolve_data_paths")
