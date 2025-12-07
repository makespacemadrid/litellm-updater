"""Compat models API routes - OpenAI-compatible aliases."""
from fastapi import APIRouter, Depends, Form, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from shared.crud import (
    create_compat_model,
    delete_model,
    get_all_compat_models,
    get_model_by_id,
    get_provider_by_id,
    update_compat_model,
    get_or_create_compat_provider,
)
from shared.database import get_session
from shared.db_models import Model, Provider
from sqlalchemy import select

router = APIRouter()


def _parse_csv_list(raw: str | None) -> list[str] | None:
    """Convert comma separated string to list, removing empties."""
    if raw is None:
        return None
    parts = [p.strip() for p in raw.split(",")]
    return [p for p in parts if p]


@router.get("/models")
async def list_compat_models(session: AsyncSession = Depends(get_session)):
    """List all compat models."""
    models = await get_all_compat_models(session)

    # Cache providers we look up for mappings
    provider_cache: dict[int, dict] = {}

    result = []
    for model in models:
        mapped_provider = None
        if model.mapped_provider_id:
            if model.mapped_provider_id in provider_cache:
                mapped_provider = provider_cache[model.mapped_provider_id]
            else:
                provider = await get_provider_by_id(session, model.mapped_provider_id)
                if provider:
                    mapped_provider = {
                        "id": provider.id,
                        "name": provider.name,
                        "type": provider.type,
                        "base_url": provider.base_url,
                    }
                    provider_cache[provider.id] = mapped_provider

        result.append(
            {
                "id": model.id,
                "model_name": model.model_id,
                "mapped_model_id": model.mapped_model_id,
                "mapped_provider": mapped_provider,
                "access_groups": model.get_effective_access_groups(),
            }
        )

    return result


@router.post("/models")
async def create_compat_model_endpoint(
    model_name: str = Form(...),
    mapped_provider_id: int | None = Form(None),
    mapped_model_id: str | None = Form(None),
    access_groups: str | None = Form(None),
    session: AsyncSession = Depends(get_session),
):
    """Create a compat mapping."""
    model = await create_compat_model(
        session,
        model_name=model_name,
        mapped_provider_id=mapped_provider_id,
        mapped_model_id=mapped_model_id,
        access_groups=_parse_csv_list(access_groups),
    )
    return {"id": model.id, "message": "Compat model created"}


@router.put("/models/{model_id}")
async def update_compat_model_endpoint(
    model_id: int,
    mapped_provider_id: int | None = Form(None),
    mapped_model_id: str | None = Form(None),
    access_groups: str | None = Form(None),
    session: AsyncSession = Depends(get_session),
):
    """Update an existing compat mapping."""
    model = await get_model_by_id(session, model_id)
    if not model or not model.provider or model.provider.type != "compat":
        raise HTTPException(404, "Compat model not found")

    await update_compat_model(
        session,
        model,
        mapped_provider_id=mapped_provider_id,
        mapped_model_id=mapped_model_id,
        access_groups=_parse_csv_list(access_groups),
    )
    return {"message": "Compat model updated"}


@router.delete("/models/{model_id}")
async def delete_compat_model(model_id: int, session: AsyncSession = Depends(get_session)):
    """Delete compat model."""
    model = await get_model_by_id(session, model_id)
    if not model or not model.provider or model.provider.type != "compat":
        raise HTTPException(404, "Compat model not found")

    await delete_model(session, model)
    return {"message": "Compat model deleted"}


@router.post("/register-defaults")
async def register_default_models(session: AsyncSession = Depends(get_session)):
    """
    Create compat aliases by matching DEFAULT_COMPAT_MODELS against existing models in the database.

    For each default OpenAI name (like "gpt-4o", "gpt-3.5-turbo"), extracts the actual Ollama model
    (like "qwen3:32b", "qwen3:4b") from the litellm_params and searches for it in the database.
    If found, creates the compat alias mapping.
    """
    from shared.default_compat_models import DEFAULT_COMPAT_MODELS

    compat_provider = await get_or_create_compat_provider(session)

    # Existing compat names to avoid duplicates
    existing_compat = {m.model_id for m in await get_all_compat_models(session)}

    created = 0
    skipped = []
    errors: list[str] = []
    summary = {"chat": 0, "vision": 0, "embedding": 0, "reasoning": 0, "code": 0}

    for compat_def in DEFAULT_COMPAT_MODELS:
        openai_name = compat_def["model_name"]

        # Skip if already exists
        if openai_name in existing_compat:
            skipped.append(openai_name)
            continue

        # Extract the actual model from litellm_params.model
        # Format is "ollama/qwen3:8b" -> extract "qwen3:8b"
        litellm_model = compat_def["litellm_params"]["model"]
        if "/" in litellm_model:
            _, actual_model_id = litellm_model.split("/", 1)
        else:
            actual_model_id = litellm_model

        # Find a model with matching model_id in any provider (except compat)
        result = await session.execute(
            select(Model)
            .join(Provider)
            .where(
                Model.model_id == actual_model_id,
                Provider.type != "compat",
            )
        )
        target = result.scalar_one_or_none()

        if not target:
            skipped.append(f"{openai_name} (missing {actual_model_id})")
            continue

        # Determine category from tags
        tags = compat_def["litellm_params"]["tags"]
        if "vision" in tags:
            category = "vision"
        elif "embedding" in tags:
            category = "embedding"
        elif "reasoning" in tags:
            category = "reasoning"
        elif "code" in tags:
            category = "code"
        else:
            category = "chat"

        try:
            await create_compat_model(
                session,
                model_name=openai_name,
                mapped_provider_id=target.provider_id,
                mapped_model_id=target.model_id,
                access_groups=["compat"],
            )
            created += 1
            summary[category] += 1
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(f"{openai_name}: {exc}")

    return {
        "message": "Compat defaults processed",
        "created": created,
        "skipped": skipped,
        "errors": errors,
        "summary": summary,
        "results": {"errors": errors},
        "provider": {"id": compat_provider.id, "name": compat_provider.name},
    }
