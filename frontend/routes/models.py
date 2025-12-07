"""Model management API routes."""
import httpx
from fastapi import APIRouter, Body, Depends, HTTPException
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.litellm_client import push_model_to_litellm
from backend.provider_sync import _clean_ollama_payload
from shared.crud import (
    get_config,
    get_model_by_id,
    update_model_params,
    delete_model,
    upsert_model,
)
from shared.database import get_session
from shared.db_models import Model
from shared.models import SourceEndpoint, SourceType
from shared.sources import fetch_source_models

router = APIRouter()


def _parse_bool(raw) -> bool | None:
    """Best-effort boolean parsing for payload values."""
    if raw is None:
        return None
    if isinstance(raw, bool):
        return raw
    return str(raw).lower() in {"true", "1", "yes", "on"}


def _normalize_list(raw) -> list | None:
    """Convert comma-separated strings to lists; pass through lists."""
    if raw is None:
        return None
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        parts = [item.strip() for item in raw.split(",")]
        return [p for p in parts if p]
    return None


def _model_response(model: Model) -> dict:
    """Serialize a model for API responses."""
    provider = model.provider
    display_name = model.get_display_name(apply_prefix=True)
    return {
        "id": model.id,
        "model_id": model.model_id,
        "provider_id": model.provider_id,
        "provider": {
            "id": provider.id if provider else None,
            "name": provider.name if provider else None,
            "type": provider.type if provider else None,
            "prefix": provider.prefix if provider else None,
        },
        "display_name": display_name,
        "litellm_params": model.litellm_params_dict,
        "user_params": model.user_params_dict,
        "effective_params": model.effective_params,
        "is_orphaned": model.is_orphaned,
        "sync_enabled": model.sync_enabled,
        "user_modified": model.user_modified,
        "ollama_mode": model.ollama_mode,
        "capabilities": model.capabilities_list,
        "tags": model.system_tags_list,
        "user_tags": model.user_tags_list,
        "access_groups": model.access_groups_list,
    }


@router.get("/{model_id}")
async def get_model(model_id: int, session: AsyncSession = Depends(get_session)):
    """Get model details."""
    model = await get_model_by_id(session, model_id)
    if not model:
        raise HTTPException(404, "Model not found")

    return _model_response(model)


@router.patch("/{model_id}/params")
@router.post("/{model_id}/params")
async def update_model_parameters(
    model_id: int,
    payload: dict = Body(...),
    session: AsyncSession = Depends(get_session),
):
    """Update model user parameters."""
    model = await get_model_by_id(session, model_id)
    if not model:
        raise HTTPException(404, "Model not found")

    params = payload.get("params")
    tags = _normalize_list(payload.get("tags"))
    access_groups = _normalize_list(payload.get("access_groups"))
    sync_enabled = _parse_bool(payload.get("sync_enabled"))

    await update_model_params(
        session,
        model,
        user_params=params,
        user_tags=tags,
        access_groups=access_groups,
        sync_enabled=sync_enabled,
    )

    return {"status": "updated", "message": "Model parameters saved"}


@router.delete("/{model_id}")
async def remove_model(model_id: int, session: AsyncSession = Depends(get_session)):
    """Delete model from database."""
    model = await get_model_by_id(session, model_id)
    if not model:
        raise HTTPException(404, "Model not found")
    await delete_model(session, model)
    return {"status": "deleted", "message": f"Deleted {model.model_id}"}


@router.post("/push-all")
async def push_all_models(session: AsyncSession = Depends(get_session)):
    """Push all non-orphaned, sync-enabled models to LiteLLM using reconciliation logic."""
    from backend.litellm_client import reconcile_litellm_models

    config = await get_config(session)
    if not config.litellm_base_url:
        raise HTTPException(400, "LiteLLM destination not configured")

    result = {"success": 0, "failed": 0, "errors": []}

    # Group models by provider
    stmt = select(Model).options(selectinload(Model.provider)).where(Model.is_orphaned == False)  # noqa: E712
    all_models = list((await session.execute(stmt)).scalars().all())

    # Group by provider
    from collections import defaultdict
    models_by_provider = defaultdict(list)
    for model in all_models:
        if model.sync_enabled and model.provider:
            models_by_provider[model.provider].append(model)

    # Reconcile each provider (no orphan removal for manual push)
    for provider, models in models_by_provider.items():
        try:
            stats = await reconcile_litellm_models(
                session,
                config,
                provider,
                models,
                remove_orphaned=False  # Don't remove orphans on manual push
            )
            result["success"] += stats.get("added", 0) + stats.get("updated", 0)
            result["failed"] += stats.get("duplicates_failed", 0)  # placeholder for future
            result["duplicates_removed"] = result.get("duplicates_removed", 0) + stats.get("duplicates_removed", 0)
        except Exception as exc:  # pragma: no cover - network dependent
            result["failed"] += len(models)
            result["errors"].append(f"{provider.name}: {exc}")

    return {"message": "Push complete", "results": result}


@router.post("/db/reset-all")
async def reset_all_models(session: AsyncSession = Depends(get_session)):
    """Delete all models from the database."""
    await session.execute(delete(Model))
    return {"status": "reset", "message": "All models removed"}


@router.post("/{model_id}/refresh")
async def refresh_model(model_id: int, session: AsyncSession = Depends(get_session)):
    """Refresh a single model from its provider."""
    model = await get_model_by_id(session, model_id)
    if not model:
        raise HTTPException(404, "Model not found")

    provider = model.provider
    if provider is None:
        raise HTTPException(400, "Model provider missing")

    if provider.type == "compat":
        raise HTTPException(400, "Compat models cannot be refreshed from provider")

    source = SourceEndpoint(
        name=provider.name,
        base_url=provider.base_url,
        type=SourceType(provider.type),
        api_key=provider.api_key,
        prefix=provider.prefix,
        default_ollama_mode=provider.default_ollama_mode,
    )

    source_models = await fetch_source_models(source)
    metadata = next((m for m in source_models.models if m.id == model.model_id), None)
    if not metadata:
        raise HTTPException(404, "Model not found in provider response")

    if provider.type == "ollama":
        metadata.raw = _clean_ollama_payload(metadata.raw)

    await upsert_model(session, provider, metadata, full_update=True)
    return {"status": "refreshed", "message": f"Updated from {provider.name}"}


@router.post("/{model_id}/push")
async def push_model_to_litellm_endpoint(model_id: int, session: AsyncSession = Depends(get_session)):
    """Push a single model to LiteLLM using reconciliation logic."""
    from backend.litellm_client import reconcile_litellm_models

    model = await get_model_by_id(session, model_id)
    if not model:
        raise HTTPException(404, "Model not found")

    config = await get_config(session)
    if not config.litellm_base_url:
        raise HTTPException(400, "LiteLLM destination not configured")

    # Use reconciliation logic (no orphan removal for manual push)
    await reconcile_litellm_models(
        session,
        config,
        model.provider,
        [model],
        remove_orphaned=False
    )

    return {"status": "pushed", "message": f"Pushed {model.get_display_name()}"}
