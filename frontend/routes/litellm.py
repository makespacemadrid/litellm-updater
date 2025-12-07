"""LiteLLM models API routes - for viewing and managing LiteLLM models."""
from fastapi import APIRouter, Body, Depends, Form, Request
from sqlalchemy.ext.asyncio import AsyncSession
import httpx

from shared.database import get_session
from shared.crud import get_config

router = APIRouter()


def _normalize_base_url(raw: str | None) -> str | None:
    """Strip trailing slash to avoid double slashes."""
    if not raw:
        return None
    return raw.rstrip("/")


@router.get("/models")
async def list_litellm_models(session: AsyncSession = Depends(get_session)):
    """Fetch models from LiteLLM server."""
    config = await get_config(session)

    base_url = _normalize_base_url(config.litellm_base_url)
    if not base_url:
        return {"models": [], "error": "LiteLLM not configured"}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{base_url}/model/info",
                headers={"Authorization": f"Bearer {config.litellm_api_key}"} if config.litellm_api_key else {},
                timeout=10.0
            )
            response.raise_for_status()
            data = response.json()
            return {"models": data.get("data", [])}
    except Exception as e:
        return {"models": [], "error": str(e)}


@router.post("/models/delete")
async def delete_litellm_model(
    model_id: str = Form(...),
    session: AsyncSession = Depends(get_session)
):
    """Delete a model from LiteLLM."""
    config = await get_config(session)

    base_url = _normalize_base_url(config.litellm_base_url)
    if not base_url:
        return {"status": "error", "message": "LiteLLM not configured"}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{base_url}/model/delete",
                json={"id": model_id},
                headers={"Authorization": f"Bearer {config.litellm_api_key}"} if config.litellm_api_key else {},
                timeout=10.0
            )
            response.raise_for_status()
            return {"status": "deleted"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/models/delete/bulk")
async def delete_litellm_models_bulk(
    request: Request,
    model_ids: list[str] | None = Form(None),
    payload: dict | None = Body(None),
    session: AsyncSession = Depends(get_session)
):
    """Delete multiple models from LiteLLM."""
    config = await get_config(session)

    base_url = _normalize_base_url(config.litellm_base_url)
    if not base_url:
        return {"status": "error", "message": "LiteLLM not configured"}

    # Support JSON body without form encoding
    if not model_ids:
        try:
            body = payload or await request.json()
            model_ids = body.get("model_ids")
        except Exception:
            model_ids = None

    if not model_ids:
        return {"status": "error", "message": "No model IDs provided"}

    deleted = []
    errors = []

    async with httpx.AsyncClient() as client:
        for model_id in model_ids:
            try:
                response = await client.post(
                    f"{base_url}/model/delete",
                    json={"id": model_id},
                    headers={"Authorization": f"Bearer {config.litellm_api_key}"} if config.litellm_api_key else {},
                    timeout=10.0
                )
                response.raise_for_status()
                deleted.append(model_id)
            except httpx.HTTPStatusError as e:  # propagate HTTP errors clearly
                errors.append({"id": model_id, "error": f"{e.response.status_code}: {e.response.text}"})
            except Exception as e:
                errors.append({"id": model_id, "error": str(e)})

    return {
        "status": "completed",
        "deleted": len(deleted),
        "errors": errors,
        "model_ids": model_ids,
    }
