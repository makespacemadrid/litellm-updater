"""CRUD operations for database models."""
import json
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from .db_models import (
    Config,
    Model,
    Provider,
    RoutingGroup,
    RoutingTarget,
    RoutingProviderLimit,
)
from .models import ModelMetadata, SourceEndpoint
from .pricing_profiles import apply_pricing_overrides
from .tags import generate_model_tags, normalize_tags


def _normalize_provider_base_url(base_url: str | None, type_: str | None) -> str | None:
    """Normalize provider base_url for known API patterns."""
    if base_url is None:
        return None
    normalized = base_url.strip()
    if not normalized:
        return None
    normalized = normalized.rstrip("/")
    if type_ == "openai":
        if normalized.endswith("/openai/v1"):
            normalized = normalized[: -len("/openai/v1")] + "/v1/openai"
        elif normalized.endswith("/v1"):
            normalized = normalized[: -len("/v1")]
    return normalized


# Config CRUD Operations


async def get_config(session: AsyncSession) -> Config:
    """Get global configuration (creates default if not exists)."""
    result = await session.execute(select(Config).limit(1))
    config = result.scalar_one_or_none()

    if config is None:
        # Create default config
        config = Config(
            litellm_base_url=None,
            litellm_api_key=None,
            sync_interval_seconds=300,
            default_pricing_profile=None,
            default_pricing_override=None,
        )
        session.add(config)
        await session.flush()

    return config


async def update_config(
    session: AsyncSession,
    litellm_base_url: str | None = None,
    litellm_api_key: str | None = None,
    sync_interval_seconds: int | None = None,
    default_pricing_profile: str | None = None,
    default_pricing_override: dict | None = None,
) -> Config:
    """Update global configuration."""
    config = await get_config(session)

    if litellm_base_url is not None:
        config.litellm_base_url = litellm_base_url or None
    if litellm_api_key is not None:
        config.litellm_api_key = litellm_api_key or None
    if sync_interval_seconds is not None:
        config.sync_interval_seconds = sync_interval_seconds
    if default_pricing_profile is not None:
        config.default_pricing_profile = default_pricing_profile or None
    if default_pricing_override is not None:
        config.default_pricing_override_dict = default_pricing_override

    config.updated_at = datetime.now(UTC)
    return config


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
    tags: list[str] | None = None,
    access_groups: list[str] | None = None,
    sync_enabled: bool = True,
    sync_interval_seconds: int | None = None,
    pricing_profile: str | None = None,
    pricing_override: dict | None = None,
    auto_detect_fim: bool = True,
    model_filter: str | None = None,
    model_filter_exclude: str | None = None,
) -> Provider:
    """Create a new provider."""
    if type_ == "ollama" and default_ollama_mode is None:
        default_ollama_mode = "ollama_chat"
    base_url = _normalize_provider_base_url(base_url, type_)
    provider = Provider(
        name=name,
        base_url=base_url,
        type=type_,
        api_key=api_key,
        prefix=prefix,
        default_ollama_mode=default_ollama_mode,
        sync_enabled=sync_enabled,
        sync_interval_seconds=sync_interval_seconds,
        pricing_profile=pricing_profile,
        auto_detect_fim=auto_detect_fim,
        model_filter=model_filter,
        model_filter_exclude=model_filter_exclude,
    )
    provider.tags_list = normalize_tags(tags)
    provider.access_groups_list = normalize_tags(access_groups)
    provider.pricing_override_dict = pricing_override
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
        tags=normalize_tags(getattr(source, "tags", [])),
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
    model_filter: str | None = None,
    tags: list[str] | None = None,
    access_groups: list[str] | None = None,
    sync_enabled: bool | None = None,
    sync_interval_seconds: int | None = None,
    pricing_profile: str | None = None,
    pricing_override: dict | None = None,
    auto_detect_fim: bool | None = None,
    model_filter_exclude: str | None = None,
) -> Provider:
    """Update existing provider."""
    if name is not None:
        provider.name = name
    if base_url is not None:
        provider.base_url = _normalize_provider_base_url(base_url, type_ or provider.type)
    if type_ is not None:
        provider.type = type_
    if api_key is not None:
        provider.api_key = api_key
    if prefix is not None:
        provider.prefix = prefix
    if default_ollama_mode is not None:
        provider.default_ollama_mode = default_ollama_mode
    elif provider.type == "ollama" and provider.default_ollama_mode is None:
        provider.default_ollama_mode = "ollama_chat"
    if model_filter is not None:
        provider.model_filter = model_filter
    if model_filter_exclude is not None:
        provider.model_filter_exclude = model_filter_exclude
    if tags is not None:
        provider.tags_list = normalize_tags(tags)
    if access_groups is not None:
        provider.access_groups_list = normalize_tags(access_groups)
    if sync_enabled is not None:
        provider.sync_enabled = sync_enabled
    if sync_interval_seconds is not None:
        provider.sync_interval_seconds = sync_interval_seconds
    if pricing_profile is not None:
        provider.pricing_profile = pricing_profile
    if pricing_override is not None:
        provider.pricing_override_dict = pricing_override
    if auto_detect_fim is not None:
        provider.auto_detect_fim = auto_detect_fim

    provider.updated_at = datetime.now(UTC)
    return provider


async def update_provider_from_source(
    session: AsyncSession, provider: Provider, source: SourceEndpoint
) -> Provider:
    """Update existing provider from SourceEndpoint."""
    provider.base_url = _normalize_provider_base_url(str(source.base_url), source.type.value)
    provider.type = source.type.value
    provider.api_key = source.api_key
    provider.prefix = getattr(source, "prefix", None)
    provider.default_ollama_mode = getattr(source, "default_ollama_mode", None)
    provider.tags_list = normalize_tags(getattr(source, "tags", []))
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
    query = select(Model).where(Model.provider_id == provider_id).options(selectinload(Model.provider))

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
    system_tags: list[str] | None = None,
    user_tags: list[str] | None = None,
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
        system_tags=json.dumps(system_tags or []),
        user_tags=json.dumps(user_tags) if user_tags else None,
        ollama_mode=ollama_mode,
        first_seen=now,
        last_seen=now,
    )
    session.add(model)
    await session.flush()
    return model


async def upsert_model(
    session: AsyncSession,
    provider: Provider,
    metadata: ModelMetadata,
    full_update: bool = False,
    config: Config | None = None,
) -> tuple[Model, bool]:
    """Create or update model from ModelMetadata.

    Args:
        session: Database session
        provider: Provider instance
        metadata: Model metadata from source
        full_update: If True, updates all fields. If False (default), only touches last_seen.
                     During regular sync, full_update=False to preserve database state.
                     During refresh, full_update=True to fetch latest from provider.

    Returns:
        Tuple of (model, was_created) where was_created is True if model was newly created
    """
    existing = await get_model_by_provider_and_name(session, provider.id, metadata.id)
    now = datetime.now(UTC)
    ollama_mode = metadata.litellm_mode or provider.default_ollama_mode
    system_tags = generate_model_tags(
        provider.name,
        provider.type,
        metadata,
        provider_tags=provider.tags_list,
        mode=ollama_mode,
    )
    litellm_fields = apply_pricing_overrides(
        metadata.litellm_fields.copy(), config=config, provider=provider
    )
    if system_tags:
        litellm_fields["tags"] = system_tags

    if existing:
        if full_update:
            # Full update: Update all model fields (used by Refresh action)
            existing.model_type = metadata.model_type
            existing.context_window = metadata.context_window
            existing.max_input_tokens = metadata.max_input_tokens
            existing.max_output_tokens = metadata.max_output_tokens
            existing.max_tokens = metadata.max_tokens
            existing.capabilities = json.dumps(metadata.capabilities)

            existing.raw_metadata = json.dumps(metadata.raw)

        # Always refresh LiteLLM-mappable fields and system tags
        existing.litellm_params = json.dumps(litellm_fields)
        existing.system_tags_list = system_tags

        # Always update last_seen and un-orphan
        existing.last_seen = now

        # Un-orphan if previously orphaned
        if existing.is_orphaned:
            existing.is_orphaned = False
            existing.orphaned_at = None

        existing.updated_at = now
        return (existing, False)
    else:
        # Create new model
        new_model = await create_model(
            session,
            provider=provider,
            model_id=metadata.id,
            model_type=metadata.model_type,
            context_window=metadata.context_window,
            max_input_tokens=metadata.max_input_tokens,
            max_output_tokens=metadata.max_output_tokens,
            max_tokens=metadata.max_tokens,
            capabilities=metadata.capabilities,
            litellm_params=litellm_fields,
            raw_metadata=metadata.raw,
            system_tags=system_tags,
        )
        return (new_model, True)


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
    session: AsyncSession,
    model: Model,
    user_params: dict | None = None,
    user_tags: list[str] | None = None,
    access_groups: list[str] | None = None,
    sync_enabled: bool | None = None,
    pricing_profile: str | None = None,
    pricing_override: dict | None = None,
    ollama_mode: str | None = None,
    ollama_mode_provided: bool = False,
    config: Config | None = None,
) -> Model:
    """Update model with user-edited parameters, tags, access_groups, and sync settings."""
    if user_params is not None:
        model.user_params = json.dumps(user_params)
    if user_tags is not None:
        model.user_tags_list = normalize_tags(user_tags)
    if access_groups is not None:
        model.access_groups_list = normalize_tags(access_groups)
    if sync_enabled is not None:
        model.sync_enabled = sync_enabled
    if pricing_profile is not None:
        model.pricing_profile = pricing_profile
    if pricing_override is not None:
        model.pricing_override_dict = pricing_override
    if ollama_mode_provided:
        model.ollama_mode = ollama_mode

    if user_params is not None or user_tags is not None or access_groups is not None or sync_enabled is not None or pricing_profile is not None or pricing_override is not None or ollama_mode_provided:
        if user_params is not None or user_tags is not None or access_groups is not None or pricing_profile is not None or pricing_override is not None or ollama_mode_provided:
            model.user_modified = True
        model.updated_at = datetime.now(UTC)

    # Recompute pricing onto litellm_params if overrides changed
    if pricing_profile is not None or pricing_override is not None:
        model.litellm_params_dict = apply_pricing_overrides(
            model.litellm_params_dict,
            config=config,
            provider=model.provider,
            model=model,
        )
    return model


async def reset_model_params(session: AsyncSession, model: Model) -> Model:
    """Reset model to provider defaults (clear user edits)."""
    model.user_params = None
    model.user_tags = None
    model.access_groups = None
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


# Compat Models CRUD Operations


async def get_or_create_compat_provider(session: AsyncSession) -> Provider:
    """Get or create the special 'compat_models' provider."""
    provider = await get_provider_by_name(session, "compat_models")
    if provider is None:
        # Create compat provider with access_group set to 'compat'
        provider = await create_provider(
            session,
            name="compat_models",
            base_url="http://localhost",  # Dummy URL, not used
            type_="compat",
            prefix=None,
            access_groups=["compat"],
            sync_enabled=False,  # Compat models are manually managed
        )
        await session.flush()
    return provider


async def get_all_compat_models(session: AsyncSession) -> list[Model]:
    """Get all compat models."""
    provider = await get_or_create_compat_provider(session)
    models = await get_models_by_provider(session, provider.id, include_orphaned=False)
    return models


async def create_compat_model(
    session: AsyncSession,
    model_name: str,
    mapped_provider_id: int | None = None,
    mapped_model_id: str | None = None,
    user_params: dict | None = None,
    mode: str | None = None,
    ollama_mode: str | None = None,
    access_groups: list[str] | None = None,
) -> Model:
    """Create a new compat model."""
    provider = await get_or_create_compat_provider(session)

    # Default access_groups to ['compat'] if not provided
    if access_groups is None:
        access_groups = ["compat"]

    params = dict(user_params or {})
    if mode is not None and mode != "default":
        params["mode"] = mode

    # Create minimal metadata for compat model
    model = Model(
        provider_id=provider.id,
        model_id=model_name,
        model_type="compat",
        litellm_params=json.dumps({}),
        raw_metadata=json.dumps({"type": "compat"}),
        mapped_provider_id=mapped_provider_id,
        mapped_model_id=mapped_model_id,
        ollama_mode=ollama_mode,
        first_seen=datetime.now(UTC),
        last_seen=datetime.now(UTC),
        is_orphaned=False,
        user_modified=False,  # Only set to True when user actually modifies the model
        sync_enabled=True,
    )

    if params:
        model.user_params = json.dumps(params)

    model.access_groups_list = normalize_tags(access_groups)

    session.add(model)
    await session.flush()
    return model


async def update_compat_model(
    session: AsyncSession,
    model: Model,
    mapped_provider_id: int | None = None,
    mapped_model_id: str | None = None,
    user_params: dict | None = None,
    mode: str | None = None,
    ollama_mode: str | None = None,
    ollama_mode_provided: bool = False,
    access_groups: list[str] | None = None,
) -> Model:
    """Update a compat model's mapping and parameters."""
    if mapped_provider_id is not None:
        model.mapped_provider_id = mapped_provider_id
    if mapped_model_id is not None:
        model.mapped_model_id = mapped_model_id
    params = None
    if user_params is not None:
        params = dict(user_params)
    elif model.user_params_dict:
        params = dict(model.user_params_dict)

    if mode is not None:
        params = params or {}
        if mode == "default":
            params.pop("mode", None)
        else:
            params["mode"] = mode

    if params is not None:
        model.user_params = json.dumps(params) if params else None
    if access_groups is not None:
        model.access_groups_list = normalize_tags(access_groups)
    if ollama_mode_provided:
        model.ollama_mode = ollama_mode

    # Mark as user modified if any changes were made
    if (mapped_provider_id is not None or mapped_model_id is not None or
        params is not None or access_groups is not None or ollama_mode_provided):
        model.user_modified = True

    model.updated_at = datetime.now(UTC)
    return model


# Routing Group CRUD Operations


def _normalize_capabilities(capabilities: list[str] | None) -> list[str]:
    """Normalize capability names for routing filters."""
    return normalize_tags(capabilities or [])


async def get_routing_groups(session: AsyncSession) -> list[RoutingGroup]:
    """Return all routing groups."""
    result = await session.execute(select(RoutingGroup).order_by(RoutingGroup.name))
    return list(result.scalars().all())


async def get_routing_group(
    session: AsyncSession, group_id: int, include_children: bool = True
) -> RoutingGroup | None:
    """Return a routing group by id."""
    query = select(RoutingGroup).where(RoutingGroup.id == group_id)
    if include_children:
        query = query.options(
            selectinload(RoutingGroup.targets).selectinload(RoutingTarget.provider),
            selectinload(RoutingGroup.provider_limits).selectinload(RoutingProviderLimit.provider),
        )
    result = await session.execute(query)
    return result.scalar_one_or_none()


async def create_routing_group(
    session: AsyncSession,
    name: str,
    description: str | None = None,
    capabilities: list[str] | None = None,
) -> RoutingGroup:
    """Create a new routing group."""
    group = RoutingGroup(name=name, description=description)
    group.capabilities_list = _normalize_capabilities(capabilities)
    session.add(group)
    await session.flush()
    return group


async def update_routing_group(
    session: AsyncSession,
    group: RoutingGroup,
    name: str | None = None,
    description: str | None = None,
    capabilities: list[str] | None = None,
) -> RoutingGroup:
    """Update routing group attributes."""
    if name is not None:
        group.name = name
    if description is not None:
        group.description = description or None
    if capabilities is not None:
        group.capabilities_list = _normalize_capabilities(capabilities)
    group.updated_at = datetime.now(UTC)
    return group


async def delete_routing_group(session: AsyncSession, group: RoutingGroup) -> None:
    """Delete a routing group."""
    await session.delete(group)


def _build_routing_targets(
    group_id: int, targets: list[dict]
) -> list[RoutingTarget]:
    """Build RoutingTarget rows from payload."""
    built: list[RoutingTarget] = []
    for target in targets:
        built.append(
            RoutingTarget(
                group_id=group_id,
                provider_id=int(target["provider_id"]),
                model_id=str(target["model_id"]),
                weight=int(target.get("weight", 1)),
                priority=int(target.get("priority", 0)),
                enabled=bool(target.get("enabled", True)),
            )
        )
    return built


def _build_provider_limits(
    group_id: int, limits: list[dict]
) -> list[RoutingProviderLimit]:
    """Build RoutingProviderLimit rows from payload."""
    built: list[RoutingProviderLimit] = []
    for limit in limits:
        raw_limit = limit.get("max_requests_per_hour")
        max_requests = int(raw_limit) if raw_limit not in (None, "") else None
        built.append(
            RoutingProviderLimit(
                group_id=group_id,
                provider_id=int(limit["provider_id"]),
                max_requests_per_hour=max_requests,
            )
        )
    return built


async def replace_routing_targets(
    session: AsyncSession, group: RoutingGroup, targets: list[dict]
) -> RoutingGroup:
    """Replace routing targets for a group."""
    group.targets = _build_routing_targets(group.id, targets)
    group.updated_at = datetime.now(UTC)
    return group


async def replace_provider_limits(
    session: AsyncSession, group: RoutingGroup, limits: list[dict]
) -> RoutingGroup:
    """Replace provider limits for a group."""
    group.provider_limits = _build_provider_limits(group.id, limits)
    group.updated_at = datetime.now(UTC)
    return group


async def list_routing_candidates(
    session: AsyncSession,
    capabilities: list[str] | None = None,
    query: str | None = None,
) -> list[dict]:
    """List models eligible for routing groups with optional filters."""
    normalized_caps = _normalize_capabilities(capabilities)
    search = query.strip().lower() if query else None

    result = await session.execute(
        select(Model)
        .join(Provider)
        .where(
            Model.is_orphaned == False,  # noqa: E712
            Provider.type != "compat",
        )
        .options(selectinload(Model.provider))
    )
    models = result.scalars().all()
    candidates: list[dict] = []
    for model in models:
        provider = model.provider
        model_caps = _normalize_capabilities(model.capabilities_list)
        if normalized_caps and not set(normalized_caps).issubset(model_caps):
            continue
        if search:
            haystack = f"{provider.name} {model.model_id}".lower()
            if search not in haystack:
                continue
        candidates.append(
            {
                "provider_id": provider.id,
                "provider_name": provider.name,
                "provider_type": provider.type,
                "model_id": model.model_id,
                "model_type": model.model_type,
                "capabilities": model_caps,
                "display_name": f"{provider.name}/{model.model_id}",
            }
        )
    candidates.sort(key=lambda item: (item["provider_name"].lower(), item["model_id"].lower()))
    return candidates


async def compile_routing_groups(session: AsyncSession) -> list[dict]:
    """Compile routing groups into a proxy-friendly payload."""
    result = await session.execute(
        select(RoutingGroup)
        .options(
            selectinload(RoutingGroup.targets).selectinload(RoutingTarget.provider),
            selectinload(RoutingGroup.provider_limits).selectinload(RoutingProviderLimit.provider),
        )
        .order_by(RoutingGroup.name)
    )
    groups = result.scalars().all()
    compiled: list[dict] = []
    for group in groups:
        targets = [
            {
                "provider_id": target.provider_id,
                "provider_name": target.provider.name if target.provider else None,
                "provider_type": target.provider.type if target.provider else None,
                "base_url": target.provider.base_url if target.provider else None,
                "model_id": target.model_id,
                "weight": target.weight,
                "priority": target.priority,
                "enabled": target.enabled,
            }
            for target in sorted(group.targets, key=lambda t: (t.priority, t.id))
        ]
        limits = [
            {
                "provider_id": limit.provider_id,
                "provider_name": limit.provider.name if limit.provider else None,
                "max_requests_per_hour": limit.max_requests_per_hour,
            }
            for limit in sorted(group.provider_limits, key=lambda p: p.provider_id)
        ]
        compiled.append(
            {
                "id": group.id,
                "name": group.name,
                "description": group.description,
                "capabilities": group.capabilities_list,
                "targets": targets,
                "provider_limits": limits,
            }
        )
    return compiled
