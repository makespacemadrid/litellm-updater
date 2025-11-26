"""Upstream source clients for Ollama and LiteLLM-compatible endpoints."""
from __future__ import annotations

from datetime import datetime
from typing import List

import httpx

from .models import LitellmTarget, ModelMetadata, SourceEndpoint, SourceModels, SourceType


async def fetch_ollama_models(client: httpx.AsyncClient, source: SourceEndpoint) -> List[ModelMetadata]:
    """Fetch models from an Ollama server."""

    url = f"{source.normalized_base_url}/api/tags"
    headers = {"Authorization": f"Bearer {source.api_key}"} if source.api_key else {}
    response = await client.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    payload = response.json()
    models = payload.get("models", [])
    return [ModelMetadata.from_raw(model.get("name", "unknown"), model) for model in models]


async def fetch_litellm_models(client: httpx.AsyncClient, source: SourceEndpoint) -> List[ModelMetadata]:
    """Fetch models from a LiteLLM / OpenAI-compatible endpoint."""

    url = f"{source.normalized_base_url}/v1/models"
    headers = {"Authorization": f"Bearer {source.api_key}"} if source.api_key else {}
    response = await client.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    payload = response.json()
    models = payload.get("data", [])
    return [ModelMetadata.from_raw(model.get("id", "unknown"), model) for model in models]


async def fetch_litellm_target_models(target: LitellmTarget) -> List[ModelMetadata]:
    """Fetch models directly from the configured LiteLLM endpoint."""

    if not target.base_url:
        raise ValueError("LiteLLM endpoint is not configured")

    source = SourceEndpoint(
        name="LiteLLM",
        base_url=target.normalized_base_url,
        type=SourceType.LITELLM,
        api_key=target.api_key,
    )
    async with httpx.AsyncClient() as client:
        return await fetch_litellm_models(client, source)


async def fetch_source_models(source: SourceEndpoint) -> SourceModels:
    """Dispatch to the correct fetcher based on the source type."""

    async with httpx.AsyncClient() as client:
        if source.type is SourceType.OLLAMA:
            models = await fetch_ollama_models(client, source)
        elif source.type is SourceType.LITELLM:
            models = await fetch_litellm_models(client, source)
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported source type: {source.type}")

    return SourceModels(source=source, models=models, fetched_at=datetime.utcnow())

