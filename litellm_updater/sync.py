"""Synchronization worker that pushes upstream models into LiteLLM."""
from __future__ import annotations

import asyncio
import logging
from typing import Callable

import httpx

from .models import AppConfig, ModelMetadata, SourceEndpoint, SourceModels
from .sources import DEFAULT_TIMEOUT, _make_auth_headers, fetch_source_models

logger = logging.getLogger(__name__)


async def _register_model_with_litellm(
    client: httpx.AsyncClient, litellm_base_url: str, api_key: str | None, model: ModelMetadata
) -> None:
    """Attempt to register a model with LiteLLM."""

    url = f"{litellm_base_url}/router/model/add"
    headers = _make_auth_headers(api_key)
    payload = {"model_name": model.id}
    response = await client.post(url, json=payload, headers=headers, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()


async def sync_once(config: AppConfig) -> dict[str, SourceModels]:
    """Run a single synchronization loop.

    Returns mapping from source name to fetched models.
    """

    results: dict[str, SourceModels] = {}
    async with httpx.AsyncClient() as client:
        for source in config.sources:
            try:
                source_models = await fetch_source_models(source)
                results[source.name] = source_models
            except httpx.RequestError as exc:  # pragma: no cover - runtime logging
                logger.warning(
                    "Failed reaching source %s at %s: %s",
                    source.name,
                    source.base_url,
                    exc,
                )
                continue
            except ValueError as exc:  # pragma: no cover - invalid JSON or data
                logger.warning("Invalid response from source %s: %s", source.name, exc)
                continue
            except Exception:  # pragma: no cover - unexpected errors
                logger.exception("Unexpected error syncing source %s", source.name)
                continue

            if not config.litellm.configured:
                logger.info("LiteLLM target not configured; skipping registration for %s", source.name)
                continue

            for model in source_models.models:
                try:
                    await _register_model_with_litellm(
                        client, config.litellm.base_url, config.litellm.api_key, model
                    )
                except httpx.HTTPStatusError as exc:  # pragma: no cover - runtime logging
                    logger.warning(
                        "LiteLLM rejected model %s from %s: %s", model.id, source.name, exc.response.text
                    )
                except httpx.RequestError as exc:  # pragma: no cover - runtime logging
                    logger.warning(
                        "Failed reaching LiteLLM at %s: %s", config.litellm.base_url, exc
                    )
                except Exception:  # pragma: no cover - unexpected errors
                    logger.exception("Unexpected error registering model %s from %s", model.id, source.name)
    return results


async def start_scheduler(config_loader: Callable[[], AppConfig], store_callback) -> None:
    """Continuously run sync at the configured interval and push results to the store callback."""

    logger.info("Starting scheduler")
    while True:
        try:
            config = config_loader()
        except (OSError, ValueError, RuntimeError) as exc:  # pragma: no cover - config errors
            logger.error("Failed loading config: %s", exc)
            await asyncio.sleep(60)
            continue
        except Exception:  # pragma: no cover - unexpected errors
            logger.exception("Unexpected error loading config")
            await asyncio.sleep(60)
            continue

        if config.sync_interval_seconds <= 0:
            logger.info("Automatic synchronization disabled; skipping run")
            await asyncio.sleep(60)
            continue

        try:
            results = await sync_once(config)
            await store_callback(results)
        except Exception:  # pragma: no cover - unexpected errors
            logger.exception("Sync loop failed")

        await asyncio.sleep(config.sync_interval_seconds)

