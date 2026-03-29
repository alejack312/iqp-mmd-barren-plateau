"""Config loading and validation.

Loads YAML experiment configs, merges with base.yaml defaults, and
returns a typed config dict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_BASE_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "base.yaml"


def load_config(path: str | Path, base_path: str | Path = _BASE_CONFIG_PATH) -> dict[str, Any]:
    """Load a YAML config, deep-merging over base.yaml defaults.

    Args:
        path: Path to the experiment config YAML.
        base_path: Path to the base config (default: configs/base.yaml).

    Returns:
        Merged config dict.
    """
    # TODO: Week 1 (D1.2) validate the merged config against the schema and persist
    # the fully resolved experiment grid for reproducible reruns and scope-lock review.
    base = _load_yaml(base_path)
    override = _load_yaml(path)
    return _deep_merge(base, override)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins on conflicts)."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result
