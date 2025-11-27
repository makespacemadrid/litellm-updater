"""Upstream source clients for Ollama and LiteLLM-compatible endpoints."""
from __future__ import annotations

import logging
from datetime import UTC, datetime

import httpx

from .models import LitellmDestination, ModelMetadata, SourceEndpoint, SourceModels, SourceType

logger = logging.getLogger(__name__)

# Default timeout configuration for HTTP requests
DEFAULT_TIMEOUT = httpx.Timeout(30.0, connect=10.0)


def _make_auth_headers(api_key: str | None) -> dict[str, str]:
    """Create authorization headers if an API key is provided."""
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def _clean_ollama_payload(payload: dict) -> dict:
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


async def fetch_ollama_models(client: httpx.AsyncClient, source: SourceEndpoint) -> list[ModelMetadata]:
    """Fetch models from an Ollama server using the lightweight tags endpoint."""

    url = f"{source.normalized_base_url}/api/tags"
    headers = _make_auth_headers(source.api_key)

    response = await client.get(url, headers=headers, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    try:
        payload = response.json()
    except Exception as exc:
        raise ValueError(f"Invalid JSON response from Ollama server: {exc}") from exc
    models = payload.get("models", [])
    results: list[ModelMetadata] = []
    for model in models:
        model_id = model.get("name", "unknown")
        results.append(ModelMetadata.from_raw(model_id, model))

    return results


async def _fetch_ollama_model_details(
    client: httpx.AsyncClient, source: SourceEndpoint, model_id: str
) -> dict:
    """Fetch extended model details from Ollama's /api/show endpoint."""

    url = f"{source.normalized_base_url}/api/show"
    headers = _make_auth_headers(source.api_key)

    response = await client.post(
        url,
        json={"model": model_id, "verbose": True},
        headers=headers,
        timeout=DEFAULT_TIMEOUT,
    )
    response.raise_for_status()
    try:
        payload = response.json()
    except Exception as exc:
        raise ValueError(f"Invalid JSON response from Ollama server: {exc}") from exc
    return _clean_ollama_payload(payload) if isinstance(payload, dict) else payload


async def fetch_ollama_model_details(source: SourceEndpoint, model_id: str) -> dict:
    """Convenience wrapper to fetch Ollama model details with a managed client."""

    async with httpx.AsyncClient() as client:
        return await _fetch_ollama_model_details(client, source, model_id)


async def fetch_litellm_models(client: httpx.AsyncClient, source: SourceEndpoint) -> list[ModelMetadata]:
    """Fetch models from a LiteLLM / OpenAI-compatible endpoint."""

    url = f"{source.normalized_base_url}/v1/models"
    headers = _make_auth_headers(source.api_key)

    response = await client.get(url, headers=headers, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    try:
        payload = response.json()
    except Exception as exc:
        raise ValueError(f"Invalid JSON response from LiteLLM server: {exc}") from exc
    models = payload.get("data", [])
    results: list[ModelMetadata] = []
    for model in models:
        model_id = model.get("id", "unknown")
        raw_model = model
        detail_url = f"{source.normalized_base_url}/v1/models/{model_id}"

        try:
            detail_response = await client.get(detail_url, headers=headers, timeout=DEFAULT_TIMEOUT)
            detail_response.raise_for_status()
            try:
                detail_payload = detail_response.json()
            except Exception:
                # Skip this model detail if JSON parsing fails
                logger.debug("Invalid JSON response for model %s from %s", model_id, source.base_url)
                continue
            if isinstance(detail_payload, dict):
                detail_payload.setdefault("summary", model)
                raw_model = detail_payload
        except (httpx.HTTPStatusError, httpx.RequestError) as exc:  # pragma: no cover - upstream compatibility varies
            logger.debug("Failed fetching details for %s from %s: %s", model_id, source.base_url, exc)

        # For LiteLLM sources, extract database ID from model_info
        database_id = None
        if isinstance(raw_model, dict):
            model_info = raw_model.get("model_info", {})
            if isinstance(model_info, dict):
                database_id = model_info.get("id")

        results.append(ModelMetadata.from_raw(model_id, raw_model, database_id=database_id))

    return results


async def fetch_litellm_target_models(target: LitellmDestination) -> list[ModelMetadata]:
    """Fetch models directly from the configured LiteLLM endpoint using /model/info."""

    if not target.base_url:
        raise ValueError("LiteLLM endpoint is not configured")

    # Use /model/info endpoint which returns complete model data including UUIDs
    url = f"{target.normalized_base_url}/model/info"
    headers = _make_auth_headers(target.api_key)

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        try:
            payload = response.json()
        except Exception as exc:
            raise ValueError(f"Invalid JSON response from LiteLLM server: {exc}") from exc

        models = payload.get("data", [])
        results: list[ModelMetadata] = []

        for model in models:
            # Use model_name as the display ID
            model_name = model.get("model_name", "unknown")

            # Extract UUID from model_info.id for deletion operations
            database_id = None
            model_info = model.get("model_info", {})
            if isinstance(model_info, dict):
                database_id = model_info.get("id")

            # Use the complete model object as raw data
            results.append(ModelMetadata.from_raw(model_name, model, database_id=database_id))

        return results


async def fetch_source_models(source: SourceEndpoint) -> SourceModels:
    """Dispatch to the correct fetcher based on the source type."""

    async with httpx.AsyncClient() as client:
        if source.type is SourceType.OLLAMA:
            models = await fetch_ollama_models(client, source)
        elif source.type is SourceType.LITELLM:
            models = await fetch_litellm_models(client, source)
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported source type: {source.type}")

    return SourceModels(source=source, models=models, fetched_at=datetime.now(UTC))

