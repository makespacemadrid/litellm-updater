"""Configuration helpers that use the database."""
from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from .config import load_config
from .crud import create_provider_from_source, get_all_providers
from .db_models import Provider
from .models import AppConfig, SourceEndpoint, SourceType

logger = logging.getLogger(__name__)


async def load_providers_from_db(session: AsyncSession) -> list[SourceEndpoint]:
    """Load all providers from database as SourceEndpoint objects."""
    providers = await get_all_providers(session)

    sources = []
    for provider in providers:
        try:
            source = SourceEndpoint(
                name=provider.name,
                base_url=provider.base_url,
                type=SourceType(provider.type),
                api_key=provider.api_key,
                prefix=provider.prefix,
                default_ollama_mode=provider.default_ollama_mode,
            )
            sources.append(source)
        except Exception as exc:
            logger.error("Failed to load provider %s from DB: %s", provider.name, exc)
            continue

    return sources


async def load_config_with_db_providers(session: AsyncSession) -> AppConfig:
    """Load config with providers from database instead of config.json."""
    # Load base config (litellm target, sync interval)
    config = load_config()

    # Replace sources with providers from database
    config.sources = await load_providers_from_db(session)

    return config


async def migrate_sources_to_db(session: AsyncSession) -> int:
    """
    Migrate sources from config.json to database.

    Returns the number of sources migrated.
    """
    # Load sources from config.json
    config = load_config()

    if not config.sources:
        logger.info("No sources in config.json to migrate")
        return 0

    # Check if migration already happened (DB has providers)
    existing_providers = await get_all_providers(session)
    if existing_providers:
        logger.info("Database already has %d providers, skipping migration", len(existing_providers))
        return 0

    # Migrate each source
    migrated = 0
    for source in config.sources:
        try:
            await create_provider_from_source(session, source)
            migrated += 1
            logger.info("Migrated source to DB: %s", source.name)
        except Exception as exc:
            logger.error("Failed to migrate source %s: %s", source.name, exc)
            continue

    if migrated > 0:
        await session.commit()
        logger.info("Migration complete: %d sources migrated to database", migrated)

    return migrated
