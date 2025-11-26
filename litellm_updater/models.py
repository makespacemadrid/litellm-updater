"""Pydantic models and enums used across the LiteLLM updater service."""
from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class SourceType(str, Enum):
    """Supported upstream source types."""

    OLLAMA = "ollama"
    LITELLM = "litellm"


class SourceEndpoint(BaseModel):
    """Configuration for a single upstream source server."""

    name: str = Field(..., description="Display name for the source")
    base_url: HttpUrl = Field(..., description="Base URL for the upstream server")
    type: SourceType = Field(..., description="Type of the upstream server")
    api_key: Optional[str] = Field(
        None,
        description="Optional API key or bearer token used to authenticate against the source",
    )

    @property
    def normalized_base_url(self) -> str:
        """Return the base URL without a trailing slash for safe path joining."""

        return str(self.base_url).rstrip("/")


class LitellmTarget(BaseModel):
    """Target LiteLLM proxy configuration."""

    base_url: Optional[HttpUrl] = Field(
        None, description="LiteLLM base URL. Leave empty to disable synchronization."
    )
    api_key: Optional[str] = Field(None, description="API key to authenticate LiteLLM admin calls")

    @property
    def configured(self) -> bool:
        """Return True when a LiteLLM endpoint has been configured."""

        return self.base_url is not None

    @property
    def normalized_base_url(self) -> str:
        """Return the base URL without a trailing slash for safe path joining."""

        if self.base_url is None:
            raise ValueError("LiteLLM endpoint is not configured")
        return str(self.base_url).rstrip("/")


def _extract_numeric(raw: Dict, *keys: str) -> int | None:
    """Return the first numeric value found for the given keys in common sections."""

    def _search(mapping: Dict | None) -> int | None:
        if not isinstance(mapping, dict):
            return None

        for key in keys:
            value = mapping.get(key)
            if isinstance(value, (int, float)):
                return int(value)

        for key, value in mapping.items():
            if isinstance(value, (int, float)) and any(str(key).endswith(candidate) for candidate in keys):
                return int(value)

        return None

    for section in (
        raw,
        raw.get("metadata"),
        raw.get("details"),
        raw.get("model_info"),
        raw.get("summary"),
    ):
        found = _search(section)
        if found is not None:
            return found

    return None


def _dedupe(values: List[str]) -> List[str]:
    """Preserve order while removing duplicates."""

    seen = set()
    deduped: List[str] = []
    for value in values:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped


def _extract_capabilities(raw: Dict) -> List[str]:
    """Normalize capability-like fields into a readable list."""

    capabilities: List[str] = []

    raw_capabilities = raw.get("capabilities")
    if isinstance(raw_capabilities, dict):
        for name, enabled in raw_capabilities.items():
            if enabled:
                capabilities.append(str(name).replace("_", " "))
    elif isinstance(raw_capabilities, list):
        capabilities.extend(str(value) for value in raw_capabilities if value)

    for field in ("modalities", "tools"):
        values = raw.get(field)
        if isinstance(values, list):
            capabilities.extend(str(value) for value in values if value)

    return _dedupe(capabilities)


def _extract_model_type(model_id: str, raw: Dict, capabilities: List[str]) -> str | None:
    """Infer a model type such as embeddings or completion from known fields."""

    for key in ("type", "model_type", "task"):
        value = raw.get(key)
        if isinstance(value, str) and value:
            return value

    details = raw.get("details")
    if isinstance(details, dict):
        for key in ("type", "model_type", "family"):
            value = details.get(key)
            if isinstance(value, str) and value:
                return value

    if capabilities:
        return capabilities[0]

    normalized_id = model_id.lower()
    if "embed" in normalized_id:
        return "embedding"
    if any(token in normalized_id for token in ("gpt", "llama", "mixtral", "turbo", "chat")):
        return "completion"
    return None


def _ensure_capabilities(model_id: str, capabilities: List[str], model_type: str | None) -> List[str]:
    """Backfill capabilities using heuristics when upstream data is sparse."""

    normalized = list(capabilities)
    lowered_id = model_id.lower()

    if not normalized:
        if "vision" in lowered_id:
            normalized.append("vision")
        if "embed" in lowered_id:
            normalized.append("embedding")
        if not normalized:
            normalized.append("completion")

    if model_type and model_type not in normalized:
        normalized.append(model_type)

    return _dedupe(normalized)


def _fallback_context_window(model_id: str, context_window: int | None) -> int | None:
    """Infer a reasonable context window when upstream metadata omits it."""

    if context_window is not None:
        return context_window

    lowered_id = model_id.lower()
    match = re.search(r"(\d+)(?:k)", lowered_id)
    if match:
        try:
            return int(match.group(1)) * 1000
        except ValueError:
            pass

    if "gpt-3.5" in lowered_id:
        return 16385
    if "gpt-4" in lowered_id or "gpt4" in lowered_id:
        return 128000

    return None


class ModelMetadata(BaseModel):
    """Normalized model description."""

    id: str
    model_type: str | None = Field(None, description="Model type such as embeddings or completion")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    context_window: int | None = Field(
        None, description="Maximum input context window (tokens) when provided by the source"
    )
    max_output_tokens: int | None = Field(
        None, description="Maximum output tokens supported when provided by the source"
    )
    capabilities: List[str] = Field(
        default_factory=list, description="Normalized list of model capabilities or modalities"
    )
    raw: Dict = Field(default_factory=dict, description="Raw metadata returned from the source")

    @classmethod
    def from_raw(cls, model_id: str, raw: Dict) -> "ModelMetadata":
        """Construct a metadata object with normalized context and capability details."""

        context_window = _extract_numeric(raw, "context_length", "context_window", "max_context")
        max_output_tokens = _extract_numeric(raw, "max_output_tokens", "max_output_length")
        capabilities = _extract_capabilities(raw)
        model_type = _extract_model_type(model_id, raw, capabilities)
        capabilities = _ensure_capabilities(model_id, capabilities, model_type)
        context_window = _fallback_context_window(model_id, context_window)

        return cls(
            id=model_id,
            model_type=model_type,
            context_window=context_window,
            max_output_tokens=max_output_tokens,
            capabilities=capabilities,
            raw=raw,
        )


class SourceModels(BaseModel):
    """Collection of models for a source."""

    source: SourceEndpoint
    models: List[ModelMetadata] = Field(default_factory=list)
    fetched_at: Optional[datetime] = None


class AppConfig(BaseModel):
    """Application configuration for the updater service."""

    model_config = ConfigDict(validate_assignment=True)

    litellm: LitellmTarget = Field(default_factory=LitellmTarget)
    sources: List[SourceEndpoint] = Field(default_factory=list)
    sync_interval_seconds: int = Field(
        0,
        ge=0,
        description="Sync interval in seconds (0 disables automatic synchronization)",
    )

    @model_validator(mode="after")
    def _validate_sync_interval(self) -> "AppConfig":
        """Allow disabling sync with 0 while enforcing sane intervals when enabled."""

        if 0 < self.sync_interval_seconds < 30:
            raise ValueError("Sync interval must be at least 30 seconds or 0 to disable")
        return self
