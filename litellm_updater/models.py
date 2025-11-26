"""Pydantic models and enums used across the LiteLLM updater service."""
from __future__ import annotations

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

    for key in keys:
        value = raw.get(key)
        if isinstance(value, (int, float)):
            return int(value)

    metadata = raw.get("metadata")
    if isinstance(metadata, dict):
        for key in keys:
            value = metadata.get(key)
            if isinstance(value, (int, float)):
                return int(value)

    details = raw.get("details")
    if isinstance(details, dict):
        for key in keys:
            value = details.get(key)
            if isinstance(value, (int, float)):
                return int(value)

    return None


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

    seen = set()
    deduped: List[str] = []
    for capability in capabilities:
        if capability not in seen:
            deduped.append(capability)
            seen.add(capability)
    return deduped


class ModelMetadata(BaseModel):
    """Normalized model description."""

    id: str
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

        return cls(
            id=model_id,
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

