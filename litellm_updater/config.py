"""Configuration helpers for the LiteLLM updater."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from .models import AppConfig, LitellmTarget, SourceEndpoint, SourceType

DEFAULT_CONFIG_PATH = Path("data/config.json")

DEFAULT_CONFIG = AppConfig(
    litellm=LitellmTarget(base_url="http://localhost:4000"),
    sources=[],
    sync_interval_seconds=0,
)


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Load application config, writing defaults if the file does not exist."""

    if not path.exists():
        save_config(DEFAULT_CONFIG, path)
        return DEFAULT_CONFIG

    data = json.loads(path.read_text())
    try:
        return AppConfig.model_validate(data)
    except ValidationError as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Invalid config file {path}: {exc}") from exc


def save_config(config: AppConfig, path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Persist the app configuration to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.model_dump(), indent=2))


def add_source(endpoint: SourceEndpoint, path: Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Add a new source to the config file and return the updated config."""

    config = load_config(path)
    config.sources.append(endpoint)
    save_config(config, path)
    return config


def remove_source(name: str, path: Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Remove a source by name and return the updated config."""

    config = load_config(path)
    config.sources = [source for source in config.sources if source.name != name]
    save_config(config, path)
    return config


def update_litellm_target(target: LitellmTarget, path: Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Update LiteLLM target and persist configuration."""

    config = load_config(path)
    config.litellm = target
    save_config(config, path)
    return config


def set_sync_interval(seconds: int, path: Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Set the sync interval in seconds."""

    config = load_config(path)
    config.sync_interval_seconds = seconds
    save_config(config, path)
    return config

