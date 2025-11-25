"""Synchronization worker that pushes upstream models into LiteLLM."""
from __future__ import annotations

import asyncio
import logging
from typing import Callable, Dict

import httpx

from .models import AppConfig, ModelMetadata, SourceEndpoint, SourceModels
from .sources import fetch_source_models

logger = logging.getLogger(__name__)


async def _register_model_with_litellm(
    client: httpx.AsyncClient, litellm_base_url: str, api_key: str | None, model: ModelMetadata
) -> None:
    """Attempt to register a model with LiteLLM."""

    url = f"{litellm_base_url}/router/model/add"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    payload = {"model_name": model.id}
    response = await client.post(url, json=payload, headers=headers, timeout=30)
    response.raise_for_status()


async def sync_once(config: AppConfig) -> Dict[str, SourceModels]:
    """Run a single synchronization loop.

    Returns mapping from source name to fetched models.
    """

    results: Dict[str, SourceModels] = {}
    async with httpx.AsyncClient() as client:
        for source in config.sources:
            try:
                source_models = await fetch_source_models(source)
                results[source.name] = source_models
                for model in source_models.models:
                    await _register_model_with_litellm(client, config.litellm.base_url, config.litellm.api_key, model)
            except httpx.RequestError as exc:  # pragma: no cover - runtime logging
                logger.warning(
                    "Failed reaching source %s at %s: %s",
                    source.name,
                    source.base_url,
                    exc,
                )
            except Exception as exc:  # pragma: no cover - runtime logging
                logger.exception("Failed syncing source %s: %s", source.name, exc)
    return results


async def start_scheduler(config_loader: Callable[[], AppConfig], store_callback) -> None:
    """Continuously run sync at the configured interval and push results to the store callback."""

    logger.info("Starting scheduler")
    while True:
        config = config_loader()
        if config.sync_interval_seconds <= 0:
            logger.info("Automatic synchronization disabled; skipping run")
            await asyncio.sleep(60)
            continue

        results = await sync_once(config)
        store_callback(results)
        await asyncio.sleep(config.sync_interval_seconds)

