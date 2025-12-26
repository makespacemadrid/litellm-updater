"""Provider management API routes."""
from fastapi import APIRouter, Depends, HTTPException, Form, Body
from sqlalchemy.ext.asyncio import AsyncSession

from shared.database import get_session
from shared.crud import (
    get_all_providers,
    get_provider_by_id,
    create_provider,
    update_provider,
    delete_provider,
    get_models_by_provider,
    get_config,
)
from shared.categorization import get_category_stats
from backend.provider_sync import sync_provider

router = APIRouter()


def _parse_csv_list(raw: str | None) -> list[str] | None:
    """Convert comma separated input into a normalized list."""
    if raw is None:
        return None
    parts = [item.strip() for item in raw.split(",")]
    return [p for p in parts if p]


def _parse_bool(raw) -> bool | None:
    """Best-effort bool parsing for form/query values."""
    if raw is None:
        return None
    if isinstance(raw, bool):
        return raw
    return str(raw).lower() in {"true", "1", "yes", "on"}


def _normalize_optional_str(raw: str | None) -> str | None:
    """Return None when string is empty/whitespace."""
    if raw is None:
        return None
    raw = raw.strip()
    return raw or None


def _parse_pricing_override(input_cost: str | None, output_cost: str | None) -> dict | None:
    """Parse numeric pricing overrides."""
    pricing: dict[str, float] = {}
    if input_cost not in (None, ""):
        try:
            pricing["input_cost_per_token"] = float(input_cost)
        except ValueError:
            pass
    if output_cost not in (None, ""):
        try:
            pricing["output_cost_per_token"] = float(output_cost)
        except ValueError:
            pass
    return pricing or None


@router.get("")
@router.get("/")
async def list_providers(session: AsyncSession = Depends(get_session)):
    """List all providers."""
    providers = await get_all_providers(session)
    return [
        {
            "id": p.id,
            "name": p.name,
            "base_url": p.base_url,
            "type": p.type,
            "api_key": p.api_key,
            "prefix": p.prefix,
            "default_ollama_mode": p.default_ollama_mode,
            "auto_detect_fim": p.auto_detect_fim,
            "tags": p.tags_list,
            "access_groups": p.access_groups_list,
            "sync_enabled": p.sync_enabled,
            "pricing_profile": p.pricing_profile,
            "pricing_override": p.pricing_override_dict,
            "created_at": p.created_at.isoformat(),
            "updated_at": p.updated_at.isoformat()
        }
        for p in providers
    ]


@router.get("/stats/all")
async def get_all_stats(session: AsyncSession = Depends(get_session)):
    """Get category statistics for all providers combined."""
    from sqlalchemy import select
    from shared.db_models import Model, Provider

    # Get all non-orphaned models from non-compat providers
    result = await session.execute(
        select(Model)
        .join(Provider)
        .where(
            Model.is_orphaned == False,
                Provider.type.notin_(["compat", "completion"])
        )
    )
    models = result.scalars().all()

    # Convert to dict format for categorization
    models_data = [
        {
            "capabilities": m.capabilities,
            "system_tags": m.system_tags
        }
        for m in models
    ]

    stats = get_category_stats(models_data)

    return {
        "total_models": len(models),
        "categories": stats
    }


@router.get("/{provider_id}/models")
async def list_provider_models(
    provider_id: int,
    include_orphaned: bool = False,
    session: AsyncSession = Depends(get_session)
):
    """List all models for a provider."""
    # Get provider first to access prefix
    provider = await get_provider_by_id(session, provider_id)
    if not provider:
        raise HTTPException(404, "Provider not found")

    # Fetch models
    models = await get_models_by_provider(session, provider_id, include_orphaned=include_orphaned)

    # Build response list
    result = []
    for m in models:
        display_name = m.get_display_name(apply_prefix=True)
        result.append({
            "id": m.id,
            "model_id": m.model_id,
            "display_name": display_name,
            "model_type": m.model_type,
            "context_window": m.context_window,
            "capabilities": m.capabilities_list,
            "tags": m.system_tags_list,
            "user_tags": m.user_tags_list,
            "access_groups": m.access_groups_list or provider.access_groups_list,
            "is_orphaned": m.is_orphaned,
            "sync_enabled": m.sync_enabled,
            "user_modified": m.user_modified,
            "first_seen": m.first_seen.isoformat(),
            "last_seen": m.last_seen.isoformat()
        })

    return {"provider": {
        "id": provider.id,
        "name": provider.name,
        "type": provider.type,
        "base_url": provider.base_url,
    }, "models": result}


@router.get("/{provider_id}/stats")
async def get_provider_stats(
    provider_id: int,
    session: AsyncSession = Depends(get_session)
):
    """Get category statistics for a provider's models."""
    provider = await get_provider_by_id(session, provider_id)
    if not provider:
        raise HTTPException(404, "Provider not found")

    # Get all non-orphaned models
    models = await get_models_by_provider(session, provider_id, include_orphaned=False)

    # Convert to dict format for categorization
    models_data = [
        {
            "capabilities": m.capabilities,
            "system_tags": m.system_tags
        }
        for m in models
    ]

    stats = get_category_stats(models_data)

    return {
        "provider_id": provider_id,
        "provider_name": provider.name,
        "total_models": len(models),
        "categories": stats
    }


@router.get("/{provider_id}")
async def get_provider(provider_id: int, session: AsyncSession = Depends(get_session)):
    """Get provider by ID."""
    provider = await get_provider_by_id(session, provider_id)
    if not provider:
        raise HTTPException(404, "Provider not found")

    return {
        "id": provider.id,
        "name": provider.name,
        "base_url": provider.base_url,
        "type": provider.type,
        "api_key": provider.api_key,
        "prefix": provider.prefix,
        "default_ollama_mode": provider.default_ollama_mode,
        "auto_detect_fim": provider.auto_detect_fim,
        "tags": provider.tags_list,
        "access_groups": provider.access_groups_list,
        "sync_enabled": provider.sync_enabled,
        "pricing_profile": provider.pricing_profile,
        "pricing_override": provider.pricing_override_dict,
    }


@router.post("")
@router.post("/")
async def add_provider(
    name: str = Form(...),
    base_url: str = Form(...),
    type: str = Form(...),
    api_key: str | None = Form(None),
    prefix: str | None = Form(None),
    default_ollama_mode: str | None = Form(None),
    tags: str | None = Form(None),
    access_groups: str | None = Form(None),
    sync_enabled: bool | None = Form(True),
    auto_detect_fim: bool | None = Form(True),
    pricing_profile: str | None = Form(None),
    pricing_input_cost_per_token: str | None = Form(None),
    pricing_output_cost_per_token: str | None = Form(None),
    session: AsyncSession = Depends(get_session)
):
    """Create new provider."""
    sync_enabled_val = _parse_bool(sync_enabled)
    if sync_enabled_val is None:
        sync_enabled_val = True
    auto_detect_fim_val = _parse_bool(auto_detect_fim)
    if auto_detect_fim_val is None:
        auto_detect_fim_val = True
    provider = await create_provider(
        session,
        name=name,
        base_url=base_url,
        type_=type,
        api_key=_normalize_optional_str(api_key),
        prefix=_normalize_optional_str(prefix),
        default_ollama_mode=_normalize_optional_str(default_ollama_mode),
        tags=_parse_csv_list(tags),
        access_groups=_parse_csv_list(access_groups),
        sync_enabled=sync_enabled_val,
        auto_detect_fim=auto_detect_fim_val,
        pricing_profile=_normalize_optional_str(pricing_profile),
        pricing_override=_parse_pricing_override(
            pricing_input_cost_per_token, pricing_output_cost_per_token
        ),
    )
    return {"id": provider.id, "name": provider.name}


@router.post("/sync-all")
async def sync_all_providers(session: AsyncSession = Depends(get_session)):
    """Sync all enabled providers now (fetch + push)."""
    providers = await get_all_providers(session)
    config = await get_config(session)

    results = []
    for provider in providers:
        if not provider.sync_enabled:
            continue
        if provider.type in ("compat", "completion"):
            continue
        try:
            stats = await sync_provider(session, config, provider, push_to_litellm=True)
            results.append({"provider": provider.name, "stats": stats})
        except Exception as exc:  # pragma: no cover - network dependent
            results.append({"provider": provider.name, "error": str(exc)})

    return {"status": "completed", "results": results}


@router.post("/fetch-all")
async def fetch_all_providers(session: AsyncSession = Depends(get_session)):
    """Fetch all enabled providers without pushing to LiteLLM."""
    providers = await get_all_providers(session)
    config = await get_config(session)

    results = []
    for provider in providers:
        if not provider.sync_enabled:
            continue
        if provider.type in ("compat", "completion"):
            continue
        try:
            stats = await sync_provider(session, config, provider, push_to_litellm=False)
            results.append({"provider": provider.name, "stats": stats})
        except Exception as exc:  # pragma: no cover - network dependent
            results.append({"provider": provider.name, "error": str(exc)})

    return {"status": "completed", "results": results}


@router.patch("/{provider_id}")
@router.post("/{provider_id}")
async def update_provider_endpoint(
    provider_id: int,
    name: str | None = Form(None),
    base_url: str | None = Form(None),
    type: str | None = Form(None),
    api_key: str | None = Form(None),
    prefix: str | None = Form(None),
    default_ollama_mode: str | None = Form(None),
    tags: str | None = Form(None),
    access_groups: str | None = Form(None),
    sync_enabled: bool | None = Form(None),
    auto_detect_fim: bool | None = Form(None),
    pricing_profile: str | None = Form(None),
    pricing_input_cost_per_token: str | None = Form(None),
    pricing_output_cost_per_token: str | None = Form(None),
    session: AsyncSession = Depends(get_session)
):
    """Update provider."""
    provider_obj = await get_provider_by_id(session, provider_id)
    if not provider_obj:
        raise HTTPException(404, "Provider not found")

    await update_provider(
        session,
        provider_obj,
        name=name,
        base_url=base_url,
        type_=type,
        api_key=_normalize_optional_str(api_key),
        prefix=_normalize_optional_str(prefix),
        default_ollama_mode=_normalize_optional_str(default_ollama_mode),
        tags=_parse_csv_list(tags),
        access_groups=_parse_csv_list(access_groups),
        sync_enabled=_parse_bool(sync_enabled),
        auto_detect_fim=_parse_bool(auto_detect_fim),
        pricing_profile=_normalize_optional_str(pricing_profile),
        pricing_override=_parse_pricing_override(
            pricing_input_cost_per_token, pricing_output_cost_per_token
        ),
    )

    return {"status": "updated"}


@router.delete("/{provider_id}")
async def remove_provider(provider_id: int, session: AsyncSession = Depends(get_session)):
    """Delete provider and all its models."""
    provider = await get_provider_by_id(session, provider_id)
    if not provider:
        raise HTTPException(404, "Provider not found")
    await delete_provider(session, provider)
    return {"status": "deleted"}


@router.patch("/{provider_id}/sync")
async def sync_provider_endpoint(
    provider_id: int,
    payload: dict = Body(...),
    session: AsyncSession = Depends(get_session)
):
    """Toggle provider sync flag (backend handles actual sync scheduling)."""
    provider = await get_provider_by_id(session, provider_id)
    if not provider:
        raise HTTPException(404, "Provider not found")

    if "sync_enabled" not in payload:
        raise HTTPException(400, "sync_enabled is required")

    await update_provider(session, provider, sync_enabled=_parse_bool(payload["sync_enabled"]))

    return {
        "status": "updated",
        "provider_id": provider.id,
        "sync_enabled": provider.sync_enabled
    }


@router.post("/{provider_id}/sync-now")
async def sync_provider_now(provider_id: int, session: AsyncSession = Depends(get_session)):
    """Trigger an immediate sync for a single provider in the background."""
    provider = await get_provider_by_id(session, provider_id)
    if not provider:
        raise HTTPException(404, "Provider not found")
    if provider.type in ("compat", "completion"):
        raise HTTPException(400, "Managed providers cannot be synced")

    # Snapshot config and provider id for background task
    config = await get_config(session)
    from shared.database import async_session_maker
    import asyncio

    async def _run():
        if async_session_maker is None:
            return
        async with async_session_maker() as bg_session:
            prov = await get_provider_by_id(bg_session, provider_id)
            if not prov:
                return
            try:
                # Full sync: fetch and push to LiteLLM
                await sync_provider(bg_session, config, prov, push_to_litellm=True)
                await bg_session.commit()
            except Exception:
                await bg_session.rollback()

    asyncio.create_task(_run())
    return {"status": "started", "provider": provider.name}


@router.post("/{provider_id}/fetch-now")
async def fetch_provider_now(provider_id: int, session: AsyncSession = Depends(get_session)):
    """Fetch a single provider without pushing to LiteLLM."""
    provider = await get_provider_by_id(session, provider_id)
    if not provider:
        raise HTTPException(404, "Provider not found")
    if provider.type in ("compat", "completion"):
        raise HTTPException(400, "Managed providers cannot be fetched")

    # Snapshot config and provider id for background task
    config = await get_config(session)
    from shared.database import async_session_maker
    import asyncio

    async def _run():
        if async_session_maker is None:
            return
        async with async_session_maker() as bg_session:
            prov = await get_provider_by_id(bg_session, provider_id)
            if not prov:
                return
            try:
                await sync_provider(bg_session, config, prov, push_to_litellm=False)
                await bg_session.commit()
            except Exception:
                await bg_session.rollback()

    asyncio.create_task(_run())
    return {"status": "started", "provider": provider.name}


@router.post("/{provider_id}/push")
async def push_provider_to_litellm(provider_id: int, session: AsyncSession = Depends(get_session)):
    """Push all non-orphaned, sync-enabled models for a single provider to LiteLLM."""
    from backend.litellm_client import reconcile_litellm_models

    provider = await get_provider_by_id(session, provider_id)
    if not provider:
        raise HTTPException(404, "Provider not found")

    config = await get_config(session)
    if not config.litellm_base_url:
        raise HTTPException(400, "LiteLLM destination not configured")

    models = await get_models_by_provider(session, provider_id, include_orphaned=False)
    try:
        stats = await reconcile_litellm_models(
            session,
            config,
            provider,
            [m for m in models if m.sync_enabled and not m.is_orphaned],
            remove_orphaned=False,
        )
        return {"status": "pushed", "provider": provider.name, "stats": stats}
    except Exception as exc:  # pragma: no cover - network dependent
        raise HTTPException(502, f"Failed to push models: {exc}") from exc
