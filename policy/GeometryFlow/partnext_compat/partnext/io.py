"""Small, dependency-free subset of ``partnext.io``."""

from __future__ import annotations

import ast
import json


def parse_str(value):
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except (TypeError, ValueError, json.JSONDecodeError):
        return ast.literal_eval(stripped)
