"""CRUD operations for database models."""
import json
from datetime import UTC, datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from .db_models import Model, Provider
from .models import ModelMetadata, SourceEndpoint


# Provider CRUD Operations


async def get_all_providers(session: AsyncSession) -> list[Provider]:
    """Get all providers."""
    result = await session.execute(select(Provider).order_by(Provider.name))
    return list(result.scalars().all())


async def get_provider_by_id(session: AsyncSession, provider_id: int) -> Provider | None:
    """Get provider by ID."""
    result = await session.execute(select(Provider).where(Provider.id == provider_id))
    return result.scalar_one_or_none()


async def get_provider_by_name(session: AsyncSession, name: str) -> Provider | None:
    """Get provider by name."""
    result = await session.execute(select(Provider).where(Provider.name == name))
    return result.scalar_one_or_none()


async def create_provider(
    session: AsyncSession,
    name: str,
    base_url: str,
    type_: str,
    api_key: str | None = None,
    prefix: str | None = None,
    default_ollama_mode: str | None = None,
) -> Provider:
    """Create a new provider."""
    provider = Provider(
        name=name,
        base_url=base_url,
        type=type_,
        api_key=api_key,
        prefix=prefix,
        default_ollama_mode=default_ollama_mode,
    )
    session.add(provider)
    await session.flush()
    return provider


async def create_provider_from_source(session: AsyncSession, source: SourceEndpoint) -> Provider:
    """Create provider from SourceEndpoint."""
    return await create_provider(
        session,
        name=source.name,
        base_url=str(source.base_url),
        type_=source.type.value,
        api_key=source.api_key,
        prefix=getattr(source, "prefix", None),
        default_ollama_mode=getattr(source, "default_ollama_mode", None),
    )


async def update_provider(
    session: AsyncSession,
    provider: Provider,
    name: str | None = None,
    base_url: str | None = None,
    type_: str | None = None,
    api_key: str | None = None,
    prefix: str | None = None,
    default_ollama_mode: str | None = None,
) -> Provider:
    """Update existing provider."""
    if name is not None:
        provider.name = name
    if base_url is not None:
        provider.base_url = base_url
    if type_ is not None:
        provider.type = type_
    if api_key is not None:
        provider.api_key = api_key
    if prefix is not None:
        provider.prefix = prefix
    if default_ollama_mode is not None:
        provider.default_ollama_mode = default_ollama_mode

    provider.updated_at = datetime.now(UTC)
    return provider


async def update_provider_from_source(
    session: AsyncSession, provider: Provider, source: SourceEndpoint
) -> Provider:
    """Update existing provider from SourceEndpoint."""
    provider.base_url = str(source.base_url)
    provider.type = source.type.value
    provider.api_key = source.api_key
    provider.prefix = getattr(source, "prefix", None)
    provider.default_ollama_mode = getattr(source, "default_ollama_mode", None)
    provider.updated_at = datetime.now(UTC)
    return provider


async def delete_provider(session: AsyncSession, provider: Provider) -> None:
    """Delete provider (cascades to models)."""
    await session.delete(provider)


# Model CRUD Operations


async def get_model_by_id(session: AsyncSession, model_id: int) -> Model | None:
    """Get model by ID with provider relationship loaded."""
    result = await session.execute(
        select(Model).where(Model.id == model_id).options(selectinload(Model.provider))
    )
    return result.scalar_one_or_none()


async def get_models_by_provider(
    session: AsyncSession, provider_id: int, include_orphaned: bool = True
) -> list[Model]:
    """Get all models for a provider."""
    query = select(Model).where(Model.provider_id == provider_id)

    if not include_orphaned:
        query = query.where(Model.is_orphaned == False)  # noqa: E712

    query = query.order_by(Model.model_id)
    result = await session.execute(query)
    return list(result.scalars().all())


async def get_model_by_provider_and_name(
    session: AsyncSession, provider_id: int, model_id: str
) -> Model | None:
    """Get model by provider and model_id."""
    result = await session.execute(
        select(Model)
        .where(Model.provider_id == provider_id, Model.model_id == model_id)
        .options(selectinload(Model.provider))
    )
    return result.scalar_one_or_none()


async def create_model(
    session: AsyncSession,
    provider: Provider,
    model_id: str,
    litellm_params: dict,
    raw_metadata: dict,
    model_type: str | None = None,
    context_window: int | None = None,
    max_input_tokens: int | None = None,
    max_output_tokens: int | None = None,
    max_tokens: int | None = None,
    capabilities: list[str] | None = None,
    ollama_mode: str | None = None,
) -> Model:
    """Create a new model."""
    now = datetime.now(UTC)
    model = Model(
        provider_id=provider.id,
        model_id=model_id,
        model_type=model_type,
        context_window=context_window,
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
        max_tokens=max_tokens,
        capabilities=json.dumps(capabilities) if capabilities else None,
        litellm_params=json.dumps(litellm_params),
        raw_metadata=json.dumps(raw_metadata),
        ollama_mode=ollama_mode,
        first_seen=now,
        last_seen=now,
    )
    session.add(model)
    await session.flush()
    return model


async def upsert_model(
    session: AsyncSession, provider: Provider, metadata: ModelMetadata
) -> Model:
    """Create or update model from ModelMetadata."""
    existing = await get_model_by_provider_and_name(session, provider.id, metadata.id)
    now = datetime.now(UTC)

    if existing:
        # Update existing model
        existing.model_type = metadata.model_type
        existing.context_window = metadata.context_window
        existing.max_input_tokens = metadata.max_input_tokens
        existing.max_output_tokens = metadata.max_output_tokens
        existing.max_tokens = metadata.max_tokens
        existing.capabilities = json.dumps(metadata.capabilities)

        # Only update litellm_params if user hasn't modified
        if not existing.user_modified:
            existing.litellm_params = json.dumps(metadata.litellm_fields)

        existing.raw_metadata = json.dumps(metadata.raw)
        existing.last_seen = now

        # Un-orphan if previously orphaned
        if existing.is_orphaned:
            existing.is_orphaned = False
            existing.orphaned_at = None

        existing.updated_at = now
        return existing
    else:
        # Create new model
        return await create_model(
            session,
            provider=provider,
            model_id=metadata.id,
            model_type=metadata.model_type,
            context_window=metadata.context_window,
            max_input_tokens=metadata.max_input_tokens,
            max_output_tokens=metadata.max_output_tokens,
            max_tokens=metadata.max_tokens,
            capabilities=metadata.capabilities,
            litellm_params=metadata.litellm_fields,
            raw_metadata=metadata.raw,
        )


async def mark_orphaned_models(
    session: AsyncSession, provider: Provider, active_model_ids: set[str]
) -> int:
    """
    Mark models as orphaned if they weren't in the latest sync.

    Returns the number of models marked as orphaned.
    """
    now = datetime.now(UTC)

    # Find models that are not in active set and not already orphaned
    result = await session.execute(
        select(Model).where(
            Model.provider_id == provider.id,
            Model.model_id.notin_(active_model_ids),
            Model.is_orphaned == False,  # noqa: E712
        )
    )
    models_to_orphan = list(result.scalars().all())

    for model in models_to_orphan:
        model.is_orphaned = True
        model.orphaned_at = now
        model.updated_at = now

    return len(models_to_orphan)


async def update_model_params(
    session: AsyncSession, model: Model, user_params: dict
) -> Model:
    """Update model with user-edited parameters."""
    model.user_params = json.dumps(user_params)
    model.user_modified = True
    model.updated_at = datetime.now(UTC)
    return model


async def reset_model_params(session: AsyncSession, model: Model) -> Model:
    """Reset model to provider defaults (clear user edits)."""
    model.user_params = None
    model.user_modified = False
    model.updated_at = datetime.now(UTC)
    return model


async def delete_model(session: AsyncSession, model: Model) -> None:
    """Delete model from database."""
    await session.delete(model)


async def get_all_orphaned_models(session: AsyncSession) -> list[Model]:
    """Get all orphaned models across all providers."""
    result = await session.execute(
        select(Model)
        .where(Model.is_orphaned == True)  # noqa: E712
        .options(selectinload(Model.provider))
        .order_by(Model.provider_id, Model.model_id)
    )
    return list(result.scalars().all())
