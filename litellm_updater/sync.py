"""Synchronization worker that pushes upstream models into LiteLLM."""
from __future__ import annotations

import asyncio
import logging
from typing import Callable

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from .models import AppConfig, ModelMetadata, SourceEndpoint, SourceModels
from .sources import DEFAULT_TIMEOUT, _make_auth_headers, fetch_litellm_target_models, fetch_source_models
from .tags import generate_model_tags

DEFAULT_PROVIDER_API_KEY = "sk-1234"

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


async def _add_model_to_litellm(
    client: httpx.AsyncClient,
    litellm_base_url: str,
    api_key: str | None,
    model_name: str,
    litellm_params: dict,
    model_info: dict | None = None,
) -> None:
    """Add model to LiteLLM using /model/new."""
    url = f"{litellm_base_url}/model/new"
    headers = _make_auth_headers(api_key)
    payload: dict = {
        "model_name": model_name,
        "litellm_params": litellm_params,
    }
    if model_info:
        payload["model_info"] = model_info
    response = await client.post(url, json=payload, headers=headers, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()


async def _delete_model_from_litellm(
    client: httpx.AsyncClient, litellm_base_url: str, api_key: str | None, model_id: str
) -> None:
    """Delete a model from LiteLLM using /model/delete endpoint."""
    url = f"{litellm_base_url}/model/delete"
    headers = _make_auth_headers(api_key)
    payload = {"id": model_id}
    response = await client.post(url, json=payload, headers=headers, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()


def _build_connection_params(provider, model, mode: str | None) -> dict:
    """Build litellm_params connection config for a model.

    Args:
        provider: Provider database model
        model: Model database model (used for compat mapping)
        mode: Ollama mode (for Ollama providers)

    Returns:
        Dictionary with litellm_params
    """
    litellm_params: dict = {}
    model_id = model.model_id if hasattr(model, 'model_id') else str(model)

    if provider.type == "openai":
        # OpenAI-compatible providers always use openai/ prefix and /v1 endpoint
        litellm_params["model"] = f"openai/{model_id}"
        litellm_params["api_base"] = provider.base_url
        litellm_params["api_key"] = provider.api_key or DEFAULT_PROVIDER_API_KEY
    elif provider.type == "ollama":
        if mode == "openai":
            litellm_params["model"] = f"openai/{model_id}"
            api_base = provider.base_url.rstrip("/")
            litellm_params["api_base"] = f"{api_base}/v1"
            litellm_params["api_key"] = provider.api_key or DEFAULT_PROVIDER_API_KEY
        else:
            litellm_params["model"] = f"ollama/{model_id}"
            litellm_params["api_base"] = provider.base_url
    elif provider.type == "compat":
        # Compat models: if mapped, use the mapped model's connection params
        # Otherwise just use the model_id as-is (direct LiteLLM model reference)
        if hasattr(model, 'mapped_provider_id') and model.mapped_provider_id:
            # For compat models with mapping, we'll handle this in the caller
            # by fetching the mapped provider and using its connection params
            pass
        # For now, just store the compat model name
        litellm_params["model"] = model_id
    return litellm_params


async def _reconcile_litellm_for_provider(
    client: httpx.AsyncClient,
    config: AppConfig,
    provider,
    provider_models,
    litellm_models: list[ModelMetadata],
    session: "AsyncSession | None" = None,
) -> None:
    """Ensure LiteLLM has the expected models for a provider and remove stale ones."""
    if not config.litellm.configured:
        return

    # Map existing LiteLLM models by unique_id tag to prevent duplicates across providers
    litellm_index: dict[str, ModelMetadata] = {}
    for m in litellm_models:
        tags = set(m.tags or [])
        if isinstance(m.raw, dict):
            tags.update(m.raw.get("litellm_params", {}).get("tags", []) or [])
            tags.update(m.raw.get("model_info", {}).get("tags", []) or [])
        unique_id_tag = next((t for t in tags if t.startswith("unique_id:")), None)
        if unique_id_tag:
            litellm_index[unique_id_tag.lower()] = m

    active_ids = [m.model_id for m in provider_models if not m.is_orphaned]

    # Add missing models
    for model in provider_models:
        if model.is_orphaned:
            continue

        # Skip models with sync disabled
        if not model.sync_enabled:
            logger.debug("Skipping sync-disabled model: %s", model.model_id)
            continue

        ollama_mode = model.ollama_mode or provider.default_ollama_mode or "ollama"
        unique_id_tag = f"unique_id:{provider.name}/{model.model_id}"
        if unique_id_tag in litellm_index:
            continue

        display_name = model.get_display_name(apply_prefix=True)

        # For compat models, inherit properties from mapped model
        if provider.type == "compat" and model.mapped_provider_id and model.mapped_model_id:
            from .crud import get_provider_by_id, get_model_by_provider_and_name
            from .db_models import Model as DbModel

            try:
                # Get the mapped provider and model
                if not session:
                    logger.warning("Compat models require database session; skipping %s", model.model_id)
                    continue

                mapped_provider = await get_provider_by_id(session, model.mapped_provider_id)
                mapped_model: DbModel | None = await get_model_by_provider_and_name(
                    session, model.mapped_provider_id, model.mapped_model_id
                )

                if mapped_provider and mapped_model:
                    # Use mapped model's connection params and metadata
                    mapped_ollama_mode = mapped_model.ollama_mode or mapped_provider.default_ollama_mode or "ollama"
                    litellm_params = _build_connection_params(mapped_provider, mapped_model, mapped_ollama_mode)

                    # Use mapped model's metadata but keep compat model's tags and access_groups
                    model_info = mapped_model.effective_params.copy()

                    # Set litellm_provider based on mapped provider
                    if mapped_provider.type == "openai":
                        model_info["litellm_provider"] = "openai"
                    elif mapped_provider.type == "ollama":
                        model_info["mode"] = mapped_ollama_mode
                        model_info["litellm_provider"] = "openai" if mapped_ollama_mode == "openai" else "ollama"
                else:
                    logger.warning("Mapped model not found for compat model %s", model.model_id)
                    continue
            except Exception as exc:
                logger.error("Failed to get mapped model for compat %s: %s", model.model_id, exc)
                continue
        else:
            # Regular model (non-compat)
            litellm_params = _build_connection_params(provider, model, ollama_mode)
            model_info = model.effective_params.copy()

            if provider.type == "openai":
                model_info["litellm_provider"] = "openai"
            elif provider.type == "ollama":
                model_info["mode"] = ollama_mode
                model_info["litellm_provider"] = "openai" if ollama_mode == "openai" else "ollama"

        auto_tags = model.all_tags or generate_model_tags(
            provider_name=provider.name,
            provider_type=provider.type,
            metadata=ModelMetadata.from_raw(model.model_id, model.raw_metadata_dict),
            provider_tags=provider.tags_list,
            mode=ollama_mode if provider.type != "compat" else None,
        )
        litellm_params["tags"] = auto_tags
        model_info["tags"] = auto_tags

        # Add access_groups if configured (model overrides provider)
        effective_access_groups = model.get_effective_access_groups()
        if effective_access_groups:
            model_info["access_groups"] = effective_access_groups

        try:
            await _add_model_to_litellm(
                client,
                config.litellm.normalized_base_url,
                config.litellm.api_key,
                display_name,
                litellm_params,
                model_info,
            )
            logger.info("Synced missing model %s to LiteLLM", display_name)
        except httpx.HTTPStatusError as exc:
            logger.warning("LiteLLM rejected model %s: %s", display_name, exc.response.text)
        except httpx.RequestError as exc:
            logger.warning("Failed reaching LiteLLM for %s: %s", display_name, exc)

    # Remove models from this provider that no longer exist
    provider_tag = f"provider:{provider.name}"
    for unique_id_tag, litellm_model in litellm_index.items():
        # Only remove models from this provider
        if provider_tag not in (litellm_model.tags or []):
            continue

        # Extract model_id from unique_id:provider/model format
        if unique_id_tag.startswith("unique_id:"):
            parts = unique_id_tag.split("unique_id:", 1)[1].split("/", 1)
            if len(parts) == 2:
                tag_provider, model_id = parts
                # Only process models from the current provider
                if tag_provider == provider.name and model_id not in active_ids:
                    delete_id = litellm_model.database_id or litellm_model.id
                    try:
                        await _delete_model_from_litellm(
                            client, config.litellm.normalized_base_url, config.litellm.api_key, delete_id
                        )
                        logger.info("Removed stale model %s from LiteLLM", litellm_model.id)
                    except httpx.HTTPStatusError as exc:
                        logger.warning("LiteLLM rejected delete for %s: %s", litellm_model.id, exc.response.text)
                    except httpx.RequestError as exc:
                        logger.warning("Failed reaching LiteLLM for delete %s: %s", litellm_model.id, exc)


async def sync_once(config: AppConfig, session: AsyncSession | None = None) -> tuple[dict[str, SourceModels], dict]:
    """Run a single synchronization loop.

    If a database session is provided, models are persisted to the database.
    Returns tuple of (mapping from source name to fetched models, sync statistics dict).
    """

    results: dict[str, SourceModels] = {}
    stats = {
        "sources_synced": 0,
        "models_fetched": 0,
        "models_added": 0,
        "models_updated": 0,
        "models_orphaned": 0,
    }
    litellm_models: list[ModelMetadata] = []
    if config.litellm.configured:
        try:
            litellm_models = await fetch_litellm_target_models(config.litellm)
        except Exception as exc:  # pragma: no cover - network/runtime
            logger.warning("Failed to fetch LiteLLM models: %s", exc)
    async with httpx.AsyncClient() as client:
        for source in config.sources:
            # Check if provider has sync enabled (if using database session)
            if session:
                from .crud import get_provider_by_name

                provider = await get_provider_by_name(session, source.name)
                if provider and not provider.sync_enabled:
                    logger.info("Skipping provider %s (sync disabled)", source.name)
                    continue

                # Skip fetch for compat providers - they don't have upstream models
                if provider and provider.type == "compat":
                    logger.info("Skipping fetch for compat provider %s (models are manually created)", source.name)
                    # Still process LiteLLM reconciliation for compat models below
                    stats["sources_synced"] += 1

                    # Reconcile compat models with LiteLLM
                    if config.litellm.configured:
                        try:
                            from .crud import get_models_by_provider

                            db_models = await get_models_by_provider(session, provider.id, include_orphaned=False)
                            await _reconcile_litellm_for_provider(
                                client, config, provider, db_models, litellm_models, session
                            )
                        except Exception:
                            logger.exception("Failed reconciling LiteLLM models for compat provider %s", source.name)
                    continue

            try:
                source_models = await fetch_source_models(source)
                results[source.name] = source_models
                stats["sources_synced"] += 1
                stats["models_fetched"] += len(source_models.models)
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

            # Persist models to database if session provided
            if session:
                try:
                    from .crud import get_provider_by_name, mark_orphaned_models, upsert_model

                    # Get provider from database
                    provider = await get_provider_by_name(session, source.name)
                    if not provider:
                        logger.warning("Provider %s not found in database; skipping persistence", source.name)
                    else:
                        # Upsert all fetched models
                        active_model_ids = set()
                        for model in source_models.models:
                            try:
                                _, was_created = await upsert_model(session, provider, model, full_update=True)
                                active_model_ids.add(model.id)
                                if was_created:
                                    stats["models_added"] += 1
                                else:
                                    stats["models_updated"] += 1
                            except Exception as exc:
                                logger.error("Failed to persist model %s from %s: %s", model.id, source.name, exc)

                        # Mark models not in this fetch as orphaned
                        orphaned_count = await mark_orphaned_models(session, provider, active_model_ids)
                        stats["models_orphaned"] += orphaned_count
                        if orphaned_count > 0:
                            logger.info("Marked %d models as orphaned for provider %s", orphaned_count, source.name)

                        # Commit the changes for this provider
                        await session.commit()
                        logger.info("Persisted %d models for provider %s", len(active_model_ids), source.name)
                except Exception:  # pragma: no cover - unexpected errors
                    logger.exception("Failed to persist models for source %s", source.name)
                    await session.rollback()

            if not config.litellm.configured:
                logger.info("LiteLLM target not configured; skipping registration for %s", source.name)
                continue
            # Reconcile LiteLLM entries for this provider using DB models if session available
            if session:
                try:
                    from .crud import get_models_by_provider, get_provider_by_name

                    provider = await get_provider_by_name(session, source.name)
                    if provider:
                        db_models = await get_models_by_provider(session, provider.id, include_orphaned=True)
                        await _reconcile_litellm_for_provider(
                            client, config, provider, db_models, litellm_models, session
                        )
                except Exception:
                    logger.exception("Failed reconciling LiteLLM models for provider %s", source.name)
            else:
                # Fallback: best-effort registration
                for model in source_models.models:
                    try:
                        await _register_model_with_litellm(
                            client, config.litellm.normalized_base_url, config.litellm.api_key, model
                        )
                    except httpx.HTTPStatusError as exc:  # pragma: no cover - runtime logging
                        logger.warning(
                            "LiteLLM rejected model %s from %s: %s", model.id, source.name, exc.response.text
                        )
                    except httpx.RequestError as exc:  # pragma: no cover - runtime logging
                        logger.warning(
                            "Failed reaching LiteLLM at %s: %s", config.litellm.normalized_base_url, exc
                        )
                    except Exception:  # pragma: no cover - unexpected errors
                        logger.exception("Unexpected error registering model %s from %s", model.id, source.name)

    logger.info(
        "Sync complete: %d sources, %d models fetched, %d added, %d updated, %d orphaned",
        stats["sources_synced"],
        stats["models_fetched"],
        stats["models_added"],
        stats["models_updated"],
        stats["models_orphaned"],
    )
    return (results, stats)


async def start_scheduler(
    config_loader: Callable[[], AppConfig],
    store_callback,
    session_maker: Callable | None = None,
) -> None:
    """Continuously run sync at the configured interval and push results to the store callback.

    Args:
        config_loader: Function to load the application config
        store_callback: Callback to store sync results (for SyncState)
        session_maker: Optional async session maker for database persistence
    """

    logger.info("Starting scheduler")
    while True:
        try:
            # If session_maker provided, load providers from database
            if session_maker:
                from .config_db import load_config_with_db_providers

                async with session_maker() as session:
                    config = await load_config_with_db_providers(session)
            else:
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
            # Create database session if session_maker provided
            if session_maker:
                async with session_maker() as session:
                    try:
                        results, stats = await sync_once(config, session)
                        await store_callback(results)
                        logger.info("Sync statistics: %s", stats)
                    except Exception:  # pragma: no cover - unexpected errors
                        logger.exception("Sync loop failed")
                        await session.rollback()
            else:
                results, stats = await sync_once(config)
                await store_callback(results)
                logger.info("Sync statistics: %s", stats)
        except Exception:  # pragma: no cover - unexpected errors
            logger.exception("Sync loop failed")

        await asyncio.sleep(config.sync_interval_seconds)
