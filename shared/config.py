"""Configuration helpers for the LiteLLM updater."""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from .models import AppConfig, LitellmDestination, SourceEndpoint, SourceType

DEFAULT_CONFIG_PATH = Path("data/config.json")

DEFAULT_CONFIG = AppConfig()


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Load application config, writing defaults if the file does not exist."""

    if not path.exists():
        save_config(DEFAULT_CONFIG, path)
        return DEFAULT_CONFIG

    try:
        config_text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"Failed to read config file {path}: {exc}") from exc

    try:
        data = json.loads(config_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in config file {path}: {exc}") from exc

    try:
        return AppConfig.model_validate(data)
    except ValidationError as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Invalid config file {path}: {exc}") from exc


def save_config(config: AppConfig, path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Persist the app configuration to disk."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"Failed to create directory {path.parent}: {exc}") from exc

    try:
        # Use JSON-compatible output so URL fields (HttpUrl) are serialized as strings
        config_json = json.dumps(config.model_dump(mode="json"), indent=2)
        path.write_text(config_json, encoding="utf-8")
    except (OSError, TypeError) as exc:
        raise RuntimeError(f"Failed to write config file {path}: {exc}") from exc


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


def update_litellm_target(target: LitellmDestination, path: Path = DEFAULT_CONFIG_PATH) -> AppConfig:
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

