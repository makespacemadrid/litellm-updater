"""Upstream source clients for Ollama and LiteLLM-compatible endpoints."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List

import httpx

from .models import LitellmTarget, ModelMetadata, SourceEndpoint, SourceModels, SourceType

logger = logging.getLogger(__name__)


async def fetch_ollama_models(client: httpx.AsyncClient, source: SourceEndpoint) -> List[ModelMetadata]:
    """Fetch models from an Ollama server."""

    url = f"{source.normalized_base_url}/api/tags"
    headers = {"Authorization": f"Bearer {source.api_key}"} if source.api_key else {}
    timeout = httpx.Timeout(30.0, connect=10.0)

    response = await client.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    models = payload.get("models", [])
    results: List[ModelMetadata] = []
    for model in models:
        model_id = model.get("name", "unknown")
        combined_raw = model

        try:
            show_response = await client.post(
                f"{source.normalized_base_url}/api/show",
                json={"model": model_id, "verbose": True},
                headers=headers,
                timeout=timeout,
            )
            show_response.raise_for_status()
            show_payload = show_response.json()
            if isinstance(show_payload, dict):
                cleaned_payload = _clean_ollama_payload(show_payload)
                combined_raw = {**cleaned_payload, "tag": model}
        except Exception as exc:  # pragma: no cover - debug aid when upstream data is missing
            logger.debug("Failed fetching Ollama model info for %s: %s", model_id, exc)

        results.append(ModelMetadata.from_raw(model_id, combined_raw))

    return results


async def fetch_litellm_models(client: httpx.AsyncClient, source: SourceEndpoint) -> List[ModelMetadata]:
    """Fetch models from a LiteLLM / OpenAI-compatible endpoint."""

    url = f"{source.normalized_base_url}/v1/models"
    headers = {"Authorization": f"Bearer {source.api_key}"} if source.api_key else {}
    timeout = httpx.Timeout(30.0, connect=10.0)

    response = await client.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    models = payload.get("data", [])
    results: List[ModelMetadata] = []
    for model in models:
        model_id = model.get("id", "unknown")
        raw_model = model
        detail_url = f"{source.normalized_base_url}/v1/models/{model_id}"

        try:
            detail_response = await client.get(detail_url, headers=headers, timeout=timeout)
            detail_response.raise_for_status()
            detail_payload = detail_response.json()
            if isinstance(detail_payload, dict):
                detail_payload.setdefault("summary", model)
                raw_model = detail_payload
        except Exception as exc:  # pragma: no cover - upstream compatibility varies
            logger.debug("Failed fetching details for %s from %s: %s", model_id, source.base_url, exc)

        results.append(ModelMetadata.from_raw(model_id, raw_model))

    return results


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
def _clean_ollama_payload(payload: Dict) -> Dict:
    """Remove extremely large or redundant fields from Ollama responses.

    The `/api/show` endpoint can include tensors and the full modelfile content,
    which are not required for synchronization and can bloat the stored
    metadata. This helper returns a sanitized copy while leaving the original
    payload untouched.
    """

    if not isinstance(payload, dict):
        return payload

    cleaned = {**payload}
    for field in ("tensors", "license", "licence", "modelfile"):
        cleaned.pop(field, None)

    model_info = cleaned.get("model_info")
    if isinstance(model_info, dict):
        model_info = {**model_info}
        model_info.pop("tensors", None)
        cleaned["model_info"] = model_info

    return cleaned

