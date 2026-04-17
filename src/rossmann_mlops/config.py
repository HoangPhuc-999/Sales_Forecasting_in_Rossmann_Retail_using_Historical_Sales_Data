from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when configuration content is invalid."""


def project_root() -> Path:
    """Return the repository root from package-relative location."""
    return Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    """Load YAML config as dictionary with basic shape validation."""
    path = Path(config_path)
    if not path.is_absolute():
        path = project_root() / path

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ConfigError(f"Config file is empty: {path}")
    if not isinstance(config, dict):
        raise ConfigError("Config file must contain a top-level mapping")

    return config


def resolve_path(path_like: str | Path) -> Path:
    """Resolve relative paths from project root while preserving absolute paths."""
    path = Path(path_like)
    return path if path.is_absolute() else project_root() / path
