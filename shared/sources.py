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
    for field in (
        "tensors",
        "tensor_data",
        "license",
        "licence",
        "modelfile",
        "projector_info",
        "model_info",
        "template",
        "system",
    ):
        cleaned.pop(field, None)

    model_info = cleaned.get("model_info")
    if isinstance(model_info, dict):
        model_info = {**model_info}
        model_info.pop("tensors", None)
        cleaned["model_info"] = model_info

    return cleaned


def _slim_ollama_payload(payload: dict) -> dict:
    """Keep only small, useful Ollama fields after parsing."""

    if not isinstance(payload, dict):
        return {}

    slim: dict = {}
    for key in ("name", "model", "digest"):
        if key in payload:
            slim[key] = payload[key]

    if isinstance(payload.get("parameters"), str):
        slim["parameters"] = payload["parameters"]

    if isinstance(payload.get("capabilities"), list):
        slim["capabilities"] = payload["capabilities"]

    details = payload.get("details")
    if isinstance(details, dict):
        keep_keys = {
            "family",
            "parameter_size",
            "quantization_level",
            "format",
            "architecture",
            "context_length",
            "context_window",
            "max_context",
            "max_tokens",
            "max_input_tokens",
            "max_output_tokens",
            "embedding_length",
            "embedding_size",
            "output_vector_size",
        }
        slim_details = {k: v for k, v in details.items() if k in keep_keys}
        if slim_details:
            slim["details"] = slim_details

    return slim


async def fetch_ollama_models(client: httpx.AsyncClient, source: SourceEndpoint) -> list[ModelMetadata]:
    """Fetch models from an Ollama server and enrich with detailed metadata."""

    url = f"{source.normalized_base_url}/api/tags"
    headers = _make_auth_headers(source.api_key)

    response = await client.get(url, headers=headers, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    try:
        payload = response.json()
    except Exception as exc:
        raise ValueError(f"Invalid JSON response from Ollama server: {exc}") from exc

    # Support both {"models": [...]} and bare list payloads
    models = []
    if isinstance(payload, list):
        models = payload
    elif isinstance(payload, dict):
        models = payload.get("models", [])

    results: list[ModelMetadata] = []

    for model in models:
        model_id = model.get("name", "unknown")

        # Always fetch detailed information for each model (streamed one-by-one to limit memory)
        try:
            detailed = await _fetch_ollama_model_details(client, source, model_id)
            merged = {**model, **_clean_ollama_payload(detailed)}
            logger.debug(f"Merged keys for {model_id}: {list(merged.keys())}")
            logger.debug(f"Has model_info: {'model_info' in merged}")
            metadata = ModelMetadata.from_raw(model_id, merged)
            metadata.raw = _slim_ollama_payload(merged)
            results.append(metadata)
            continue
        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            logger.warning("Failed to fetch details for %s: %s. Using basic info.", model_id, exc)
        except Exception as exc:
            logger.error("Unexpected error fetching details for %s: %s. Skipping.", model_id, exc)
            continue

        # Fallback to basic info if details fetch is disabled or failed
        cleaned_basic = _clean_ollama_payload(model)
        metadata = ModelMetadata.from_raw(model_id, cleaned_basic)
        metadata.raw = _slim_ollama_payload(cleaned_basic)
        results.append(metadata)

    return results


async def _fetch_ollama_model_details(
    client: httpx.AsyncClient, source: SourceEndpoint, model_id: str
) -> dict:
    """Fetch extended model details from Ollama's /api/show endpoint."""

    url = f"{source.normalized_base_url}/api/show"
    headers = _make_auth_headers(source.api_key)

    response = await client.post(
        url,
        json={"name": model_id, "verbose": True},
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


async def fetch_openai_models(client: httpx.AsyncClient, source: SourceEndpoint) -> list[ModelMetadata]:
    """Fetch models from an OpenAI-compatible endpoint."""

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
        elif source.type is SourceType.OPENAI:
            models = await fetch_openai_models(client, source)
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported source type: {source.type}")

    return SourceModels(source=source, models=models, fetched_at=datetime.now(UTC))
