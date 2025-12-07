"""Configuration helpers that use the database."""
from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from .crud import get_all_providers, get_config
from .models import AppConfig, LitellmDestination, SourceEndpoint, SourceType

logger = logging.getLogger(__name__)


async def load_providers_from_db(session: AsyncSession, only_sync_enabled: bool = True) -> list[SourceEndpoint]:
    """Load all providers from database as SourceEndpoint objects.

    Args:
        session: Database session
        only_sync_enabled: If True, only load providers with sync_enabled=True
    """
    providers = await get_all_providers(session)

    sources = []
    for provider in providers:
        # Skip disabled providers if filtering is enabled
        if only_sync_enabled and not provider.sync_enabled:
            logger.debug("Skipping disabled provider: %s", provider.name)
            continue

        try:
            source = SourceEndpoint(
                name=provider.name,
                base_url=provider.base_url,
                type=SourceType(provider.type),
                api_key=provider.api_key,
                prefix=provider.prefix,
                default_ollama_mode=provider.default_ollama_mode or (
                    "ollama" if provider.type == "ollama" else None
                ),
                tags=provider.tags_list,
            )
            sources.append(source)
        except Exception as exc:
            logger.error("Failed to load provider %s from DB: %s", provider.name, exc)
            continue

    return sources


async def load_config_with_db_providers(session: AsyncSession) -> AppConfig:
    """Load configuration entirely from database."""
    # Load config from database
    db_config = await get_config(session)

    # Build LiteLLM target from database config
    litellm_target = LitellmDestination(
        base_url=db_config.litellm_base_url,
        api_key=db_config.litellm_api_key,
    )

    # Load providers from database
    sources = await load_providers_from_db(session)

    # Build AppConfig from database
    return AppConfig(
        litellm=litellm_target,
        sources=sources,
        sync_interval_seconds=db_config.sync_interval_seconds,
    )
