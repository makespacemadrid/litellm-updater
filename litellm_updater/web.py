"""FastAPI application exposing UI and APIs for syncing models."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime, timedelta

from fastapi import FastAPI, Form, HTTPException, Request, status, Depends
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession

import httpx
from pydantic import BaseModel

from .config import (
    add_source as save_source,
    load_config,
    remove_source as delete_source_from_config,
    set_sync_interval,
    update_litellm_target,
)
from .database import create_engine, get_session, init_session_maker
from .db_models import Base
from .models import (
    AppConfig,
    LitellmDestination,
    ModelMetadata,
    SourceEndpoint,
    SourceModels,
    SourceType,
)
from .sources import (
    fetch_litellm_target_models,
    fetch_ollama_model_details,
    fetch_source_models,
)
from .sync import start_scheduler, sync_once

logger = logging.getLogger(__name__)


def _human_source_type(source_type: SourceType) -> str:
    """Helper function for template compatibility."""
    return source_type.display_name()


class SyncState:
    """In-memory store for the latest synchronization results."""

    def __init__(self) -> None:
        self.models: dict[str, SourceModels] = {}
        self.last_synced: datetime | None = None
        self._lock = asyncio.Lock()

    async def update(self, results: dict[str, SourceModels]) -> None:
        async with self._lock:
            self.models = results
            self.last_synced = datetime.now(UTC)

    async def update_source(self, source_name: str, source_models: SourceModels) -> None:
        async with self._lock:
            self.models[source_name] = source_models
            self.last_synced = datetime.now(UTC)

    async def get_models(self) -> dict[str, SourceModels]:
        async with self._lock:
            return self.models.copy()

    async def get_last_synced(self) -> datetime | None:
        async with self._lock:
            return self.last_synced


sync_state = SyncState()


class ModelDetailsCache:
    """In-memory cache for Ollama model details to avoid repeated /api/show calls."""

    def __init__(self, ttl_seconds: int = 600, max_size: int = 1000) -> None:
        self.ttl = timedelta(seconds=ttl_seconds)
        self.max_size = max_size
        self._entries: dict[tuple[str, str], tuple[datetime, dict]] = {}
        self._lock = asyncio.Lock()

    async def get(self, source_name: str, model_id: str) -> dict | None:
        async with self._lock:
            key = (source_name, model_id)
            cached = self._entries.get(key)
            if not cached:
                return None

            stored_at, payload = cached
            if datetime.now(UTC) - stored_at > self.ttl:
                self._entries.pop(key, None)
                return None

            return payload

    async def set(self, source_name: str, model_id: str, payload: dict) -> None:
        async with self._lock:
            key = (source_name, model_id)
            self._entries[key] = (datetime.now(UTC), payload)

            # Evict oldest entries if cache exceeds max size
            if len(self._entries) > self.max_size:
                # Remove the oldest 10% of entries
                sorted_entries = sorted(self._entries.items(), key=lambda x: x[1][0])
                entries_to_remove = int(self.max_size * 0.1)
                for entry_key, _ in sorted_entries[:entries_to_remove]:
                    self._entries.pop(entry_key, None)

    async def update(self, source_name: str, model_id: str, payload: dict) -> dict:
        """Replace the cached payload while resetting the TTL."""

        await self.set(source_name, model_id, payload)
        return payload


model_details_cache = ModelDetailsCache()


class CacheUpdateRequest(BaseModel):
    """Payload used when overwriting cached model details from the UI."""

    source: str
    model: str
    litellm_model: dict


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database
    logger.info("Initializing database...")
    engine = create_engine()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_maker = init_session_maker(engine)
    logger.info("Database initialized successfully")

    # Start scheduler with database session maker
    task = asyncio.create_task(start_scheduler(load_config, sync_state.update, session_maker))

    try:
        yield
    finally:
        logger.info("Shutting down...")

        # Cancel scheduler
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=10.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            logger.info("Scheduler shutdown complete")
        except Exception:
            logger.exception("Unexpected error during scheduler shutdown")

        # Dispose database engine
        await engine.dispose()
        logger.info("Shutdown complete")


async def _delete_model_from_litellm(
    client: httpx.AsyncClient, litellm_base_url: str, api_key: str | None, model_id: str
) -> None:
    """Delete a model from LiteLLM using /model/delete endpoint."""
    url = f"{litellm_base_url}/model/delete"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {"id": model_id}
    response = await client.post(url, json=payload, headers=headers, timeout=30.0)
    response.raise_for_status()


async def _add_model_to_litellm(
    litellm_base_url: str, api_key: str | None, model_name: str, model_params: dict | None = None,
    source_base_url: str | None = None
) -> dict:
    """Add a single model to LiteLLM using /model/new endpoint.

    Args:
        litellm_base_url: Base URL of LiteLLM instance
        api_key: API key for authentication
        model_name: Name of the model (e.g., "qwen3:8b")
        model_params: Additional LiteLLM parameters (optional)
        source_base_url: Base URL of the source (for Ollama models)
    """
    url = f"{litellm_base_url}/model/new"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Construct payload for LiteLLM's /model/new endpoint
    # For Ollama models, prefix with "ollama/" and set api_base
    litellm_model_name = f"ollama/{model_name}"

    # litellm_params: Only configuration for connecting to the model
    litellm_params = {
        "model": litellm_model_name,
    }

    # Add source base URL if provided (for Ollama)
    if source_base_url:
        litellm_params["api_base"] = source_base_url

    # model_info: Metadata about the model (capabilities, pricing, limits, etc.)
    model_info = {}

    # Separate model_params into litellm_params and model_info
    if model_params:
        # Fields that go in litellm_params (connection config)
        connection_fields = ["model", "api_base", "api_key", "custom_llm_provider"]

        # Fields that go in model_info (metadata)
        metadata_fields = [
            "max_input_tokens", "max_output_tokens", "max_tokens",
            "input_cost_per_token", "output_cost_per_token",
            "input_cost_per_second", "output_cost_per_second",
            "output_cost_per_image", "litellm_provider",
            "supported_openai_params"
        ]

        for key, value in model_params.items():
            if value is None:  # Skip None values
                continue

            # Fields starting with "supports_" go to model_info
            if key.startswith("supports_"):
                model_info[key] = value
            # Known metadata fields go to model_info
            elif key in metadata_fields:
                model_info[key] = value
            # Connection fields go to litellm_params (if not already set)
            elif key in connection_fields and key not in litellm_params:
                litellm_params[key] = value

    payload = {
        "model_name": litellm_model_name,
        "litellm_params": litellm_params
    }

    # Only include model_info if it has content
    if model_info:
        payload["model_info"] = model_info

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, headers=headers, timeout=30.0)
        response.raise_for_status()
        return response.json()


def create_app() -> FastAPI:
    app = FastAPI(title="LiteLLM Updater", description="Sync models into LiteLLM", lifespan=lifespan)
    templates = Jinja2Templates(directory="litellm_updater/templates")
    app.mount("/static", StaticFiles(directory="litellm_updater/static"), name="static")

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle validation errors with user-friendly messages."""
        errors = exc.errors()
        error_messages = []
        for error in errors:
            field = " -> ".join(str(loc) for loc in error["loc"])
            message = error["msg"]
            error_messages.append(f"{field}: {message}")

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": error_messages}
        )

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        config = load_config()
        last_synced = await sync_state.get_last_synced()
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "config": config,
                "last_synced": last_synced,
                "human_source_type": _human_source_type,
            },
        )

    @app.get("/sources", response_class=HTMLResponse)
    async def sources(request: Request):
        """Database-driven providers and models page."""
        return templates.TemplateResponse(
            "sources_db.html",
            {
                "request": request,
            },
        )

    @app.get("/models")
    async def models_redirect(request: Request, source: str | None = None):
        """Expose the latest synced models or redirect browsers to the UI.

        When the request prefers HTML (e.g. a user navigating directly in a
        browser), redirect to the sources page to keep the existing UX. API
        consumers can request JSON to retrieve the in-memory models, optionally
        filtered by source name.
        """

        accepts = request.headers.get("accept", "")
        prefers_json = "application/json" in accepts or "*/*" == accepts

        if not prefers_json:
            return RedirectResponse(url="/sources", status_code=308)

        models = await sync_state.get_models()
        if source:
            return models.get(source) or {}

        return models

    @app.get("/models/show")
    async def model_details(source: str, model: str):
        """Fetch extended model details on demand.

        For Ollama sources, fetches detailed information via `/api/show` endpoint.
        For LiteLLM sources, returns the model metadata from sync state.
        """

        config = load_config()
        source_endpoint = next((item for item in config.sources if item.name == source), None)
        if not source_endpoint:
            raise HTTPException(status_code=404, detail="Source not found")

        # Check cache first
        cached = await model_details_cache.get(source_endpoint.name, model)
        if cached:
            return cached

        # Handle Ollama sources with detailed fetch
        if source_endpoint.type is SourceType.OLLAMA:
            try:
                raw_details = await fetch_ollama_model_details(source_endpoint, model)
                litellm_model = ModelMetadata.from_raw(model, raw_details)
                payload = {
                    "source": source_endpoint.name,
                    "model": model,
                    "fetched_at": datetime.now(UTC),
                    "litellm_model": litellm_model.litellm_fields,
                    "raw": raw_details,
                }
                await model_details_cache.set(source_endpoint.name, model, payload)
                return payload
            except httpx.HTTPStatusError as exc:
                logger.error("HTTP error fetching Ollama model details for %s: %s - %s", model, exc.response.status_code, exc.response.text)
                raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
            except httpx.RequestError as exc:
                logger.error("Network error fetching Ollama model details for %s: %s", model, exc)
                raise HTTPException(status_code=502, detail=f"Network error: {str(exc)}")
            except Exception as exc:
                logger.exception("Unexpected error fetching Ollama model details for %s", model)
                raise HTTPException(status_code=500, detail=f"Internal error: {type(exc).__name__}: {str(exc)}")

        # Handle LiteLLM sources from sync state
        else:
            models_dict = await sync_state.get_models()
            source_models = models_dict.get(source)
            if not source_models:
                raise HTTPException(status_code=404, detail=f"No models found for source '{source}'. Try refreshing first.")

            model_metadata = next((m for m in source_models.models if m.id == model), None)
            if not model_metadata:
                raise HTTPException(status_code=404, detail=f"Model '{model}' not found in source '{source}'")

            payload = {
                "source": source_endpoint.name,
                "model": model,
                "fetched_at": source_models.fetched_at or datetime.now(UTC),
                "litellm_model": model_metadata.litellm_fields,
                "raw": model_metadata.raw,
            }
            await model_details_cache.set(source_endpoint.name, model, payload)
            return payload

    @app.post("/models/cache")
    async def update_model_cache(payload: CacheUpdateRequest):
        """Allow overriding cached model details from the UI."""

        config = load_config()
        source_endpoint = next((item for item in config.sources if item.name == payload.source), None)
        if not source_endpoint:
            raise HTTPException(status_code=404, detail="Source not found")

        cached = await model_details_cache.get(source_endpoint.name, payload.model)
        if not cached:
            # For Ollama sources, fetch detailed information
            if source_endpoint.type is SourceType.OLLAMA:
                try:
                    raw_details = await fetch_ollama_model_details(source_endpoint, payload.model)
                    litellm_model = ModelMetadata.from_raw(payload.model, raw_details).litellm_fields
                    cached = {
                        "source": source_endpoint.name,
                        "model": payload.model,
                        "fetched_at": datetime.now(UTC),
                        "litellm_model": litellm_model,
                        "raw": raw_details,
                    }
                except httpx.HTTPStatusError as exc:
                    raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
                except httpx.RequestError as exc:
                    raise HTTPException(status_code=502, detail=str(exc))
            # For LiteLLM sources, get from sync state
            else:
                models_dict = await sync_state.get_models()
                source_models = models_dict.get(payload.source)
                if not source_models:
                    raise HTTPException(status_code=404, detail=f"No models found for source '{payload.source}'. Try refreshing first.")

                model_metadata = next((m for m in source_models.models if m.id == payload.model), None)
                if not model_metadata:
                    raise HTTPException(status_code=404, detail=f"Model '{payload.model}' not found in source '{payload.source}'")

                cached = {
                    "source": source_endpoint.name,
                    "model": payload.model,
                    "fetched_at": source_models.fetched_at or datetime.now(UTC),
                    "litellm_model": model_metadata.litellm_fields,
                    "raw": model_metadata.raw,
                }

        updated_payload = {**cached, "litellm_model": payload.litellm_model, "fetched_at": datetime.now(UTC)}
        await model_details_cache.update(source_endpoint.name, payload.model, updated_payload)
        return updated_payload

    @app.get("/admin", response_class=HTMLResponse)
    async def admin(request: Request):
        config = load_config()
        return templates.TemplateResponse(
            "admin.html",
            {"request": request, "config": config, "human_source_type": _human_source_type},
        )

    @app.get("/litellm", response_class=HTMLResponse)
    async def litellm(request: Request):
        config = load_config()
        litellm_models = []
        litellm_error: str | None = None
        fetched_at: datetime | None = None

        if config.litellm.configured:
            try:
                litellm_models = await fetch_litellm_target_models(config.litellm)
                fetched_at = datetime.now(UTC)
            except (httpx.HTTPError, ValueError) as exc:  # pragma: no cover - runtime logging
                logger.exception("Failed fetching LiteLLM models")
                litellm_error = str(exc)
            except Exception as exc:  # pragma: no cover - unexpected errors
                logger.exception("Unexpected error fetching LiteLLM models")
                litellm_error = f"Unexpected error: {type(exc).__name__}"
        else:
            litellm_error = "LiteLLM target is not configured."

        return templates.TemplateResponse(
            "litellm.html",
            {
                "request": request,
                "config": config,
                "litellm_models": litellm_models,
                "litellm_error": litellm_error,
                "fetched_at": fetched_at,
            },
        )

    @app.post("/admin/sources")
    async def add_source(
        name: str = Form(...),
        base_url: str = Form(...),
        source_type: SourceType = Form(...),
        api_key: str | None = Form(None),
    ):
        # Validate name is not empty
        if not name or not name.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Source name cannot be empty"
            )

        # Check for duplicate source names
        config = load_config()
        if any(s.name == name for s in config.sources):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Source with name '{name}' already exists"
            )

        try:
            endpoint = SourceEndpoint(name=name, base_url=base_url, type=source_type, api_key=api_key or None)
            save_source(endpoint)
            logger.info("Added source: %s (%s) at %s", name, source_type, base_url)
        except ValueError as exc:
            logger.error("Invalid source configuration: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid source configuration: {str(exc)}"
            )
        except (OSError, RuntimeError) as exc:
            logger.error("Failed to save source: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save source: {str(exc)}"
            )

        return RedirectResponse(url="/admin", status_code=303)

    @app.post("/admin/sources/delete")
    async def delete_source(name: str = Form(...)):
        # Verify source exists
        config = load_config()
        if not any(s.name == name for s in config.sources):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Source '{name}' not found"
            )

        try:
            delete_source_from_config(name)
            logger.info("Deleted source: %s", name)
        except (OSError, RuntimeError) as exc:
            logger.error("Failed to delete source: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete source: {str(exc)}"
            )

        return RedirectResponse(url="/admin", status_code=303)

    @app.post("/admin/litellm")
    async def update_litellm(base_url: str = Form(""), api_key: str | None = Form(None)):
        try:
            target = LitellmDestination(
                base_url=base_url or None,
                api_key=api_key or None,
            )
            update_litellm_target(target)
            logger.info("Updated LiteLLM target: %s", base_url or "disabled")
        except ValueError as exc:
            logger.error("Invalid LiteLLM configuration: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid LiteLLM configuration: {str(exc)}"
            )
        except (OSError, RuntimeError) as exc:
            logger.error("Failed to update LiteLLM target: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to update LiteLLM target: {str(exc)}"
            )

        return RedirectResponse(url="/admin", status_code=303)

    @app.post("/admin/interval")
    async def update_interval(sync_interval_seconds: int = Form(...)):
        try:
            set_sync_interval(sync_interval_seconds)
            logger.info("Updated sync interval: %d seconds", sync_interval_seconds)
        except ValueError as exc:
            logger.error("Invalid sync interval: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid sync interval: {str(exc)}"
            )
        except (OSError, RuntimeError) as exc:
            logger.error("Failed to update sync interval: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to update sync interval: {str(exc)}"
            )

        return RedirectResponse(url="/admin", status_code=303)

    # Provider CRUD endpoints (database-first)

    @app.post("/admin/providers")
    async def create_provider_endpoint(
        session: AsyncSession = Depends(get_session),
        name: str = Form(...),
        base_url: str = Form(...),
        type: str = Form(...),
        api_key: str | None = Form(None),
        prefix: str | None = Form(None),
        default_ollama_mode: str | None = Form(None),
    ):
        """Create a new provider in the database."""
        from .crud import create_provider, get_provider_by_name

        # Check if provider already exists
        existing = await get_provider_by_name(session, name)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Provider with name '{name}' already exists",
            )

        # Validate type
        if type not in ("ollama", "litellm"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Type must be 'ollama' or 'litellm'"
            )

        # Validate ollama_mode
        if default_ollama_mode and type != "ollama":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="default_ollama_mode is only valid for Ollama providers",
            )

        if default_ollama_mode and default_ollama_mode not in ("ollama", "openai"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="default_ollama_mode must be 'ollama' or 'openai'",
            )

        try:
            await create_provider(
                session,
                name=name,
                base_url=base_url,
                type_=type,
                api_key=api_key or None,
                prefix=prefix or None,
                default_ollama_mode=default_ollama_mode or None,
            )
            logger.info("Created provider: %s (%s) at %s", name, type, base_url)
        except Exception as exc:
            logger.exception("Failed to create provider")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create provider: {str(exc)}",
            )

        return RedirectResponse(url="/admin", status_code=303)

    @app.post("/admin/providers/{provider_id}")
    async def update_provider_endpoint(
        provider_id: int,
        session: AsyncSession = Depends(get_session),
        name: str | None = Form(None),
        base_url: str | None = Form(None),
        type: str | None = Form(None),
        api_key: str | None = Form(None),
        prefix: str | None = Form(None),
        default_ollama_mode: str | None = Form(None),
    ):
        """Update an existing provider."""
        from .crud import get_provider_by_id, update_provider

        provider = await get_provider_by_id(session, provider_id)
        if not provider:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Provider not found")

        # Validate type if provided
        if type and type not in ("ollama", "litellm"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Type must be 'ollama' or 'litellm'"
            )

        # Validate ollama_mode
        final_type = type or provider.type
        if default_ollama_mode and final_type != "ollama":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="default_ollama_mode is only valid for Ollama providers",
            )

        if default_ollama_mode and default_ollama_mode not in ("ollama", "openai"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="default_ollama_mode must be 'ollama' or 'openai'",
            )

        try:
            await update_provider(
                session,
                provider,
                name=name,
                base_url=base_url,
                type_=type,
                api_key=api_key,
                prefix=prefix,
                default_ollama_mode=default_ollama_mode,
            )
            logger.info("Updated provider: %s", provider.name)
        except Exception as exc:
            logger.exception("Failed to update provider")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to update provider: {str(exc)}",
            )

        return RedirectResponse(url="/admin", status_code=303)

    @app.delete("/admin/providers/{provider_id}")
    async def delete_provider_endpoint(
        provider_id: int, session: AsyncSession = Depends(get_session)
    ):
        """Delete a provider (cascades to models)."""
        from .crud import delete_provider, get_provider_by_id

        provider = await get_provider_by_id(session, provider_id)
        if not provider:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Provider not found")

        try:
            await delete_provider(session, provider)
            logger.info("Deleted provider: %s", provider.name)
        except Exception as exc:
            logger.exception("Failed to delete provider")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete provider: {str(exc)}",
            )

        return JSONResponse(content={"status": "deleted"}, status_code=200)

    @app.post("/admin/migrate-to-db")
    async def migrate_to_db_endpoint(session: AsyncSession = Depends(get_session)):
        """Migrate sources from config.json to database."""
        from .config_db import migrate_sources_to_db

        try:
            migrated = await migrate_sources_to_db(session)

            if migrated == 0:
                # Check if there were no sources or if migration already happened
                from .crud import get_all_providers
                existing_providers = await get_all_providers(session)

                if existing_providers:
                    logger.info("Migration skipped: database already has %d providers", len(existing_providers))
                    return JSONResponse(
                        content={
                            "status": "skipped",
                            "message": f"Database already has {len(existing_providers)} providers. Migration not needed.",
                            "migrated": 0
                        },
                        status_code=200
                    )
                else:
                    logger.info("Migration skipped: no sources in config.json")
                    return JSONResponse(
                        content={
                            "status": "skipped",
                            "message": "No sources in config.json to migrate.",
                            "migrated": 0
                        },
                        status_code=200
                    )

            logger.info("Migration completed: %d sources migrated to database", migrated)
            return JSONResponse(
                content={
                    "status": "success",
                    "message": f"Successfully migrated {migrated} source(s) to database.",
                    "migrated": migrated
                },
                status_code=200
            )
        except Exception as exc:
            logger.exception("Failed to migrate sources to database")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Migration failed: {str(exc)}"
            )

    @app.post("/sync")
    async def run_sync(session: AsyncSession = Depends(get_session)):
        config = load_config()
        try:
            results = await sync_once(config, session)
            await sync_state.update(results)
        except (httpx.HTTPError, ValueError) as exc:  # pragma: no cover - expected errors
            logger.warning("Manual sync failed: %s", exc)
        except Exception:  # pragma: no cover - unexpected errors
            logger.exception("Unexpected error during manual sync")
        return RedirectResponse(url="/sources", status_code=303)

    @app.post("/sources/refresh")
    async def refresh_source(name: str = Form(...)):
        config = load_config()
        source = next((source for source in config.sources if source.name == name), None)
        if not source:
            logger.warning("Attempted to refresh unknown source %s", name)
            return RedirectResponse(url="/sources", status_code=303)

        try:
            models = await fetch_source_models(source)
            await sync_state.update_source(name, models)
        except httpx.RequestError as exc:  # pragma: no cover - runtime logging
            logger.warning(
                "Failed refreshing models for %s at %s: %s", name, source.base_url, exc
            )
        except ValueError as exc:  # pragma: no cover - invalid JSON or data
            logger.warning("Invalid response from %s: %s", name, exc)
        except Exception:  # pragma: no cover - unexpected errors
            logger.exception("Unexpected error refreshing models for %s", name)

        return RedirectResponse(url="/sources", status_code=303)

    @app.post("/models/add")
    async def add_model_to_litellm(
        source: str = Form(...),
        model: str = Form(...),
    ):
        """Add a single model from a source to LiteLLM."""
        config = load_config()

        if not config.litellm.configured:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="LiteLLM target is not configured"
            )

        # Find the source
        source_endpoint = next((s for s in config.sources if s.name == source), None)
        if not source_endpoint:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Source '{source}' not found"
            )

        # Get model metadata from sync state
        models_dict = await sync_state.get_models()
        source_models = models_dict.get(source)
        if not source_models:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No models found for source '{source}'. Try refreshing first."
            )

        # Find the specific model
        model_metadata = next((m for m in source_models.models if m.id == model), None)
        if not model_metadata:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model}' not found in source '{source}'"
            )

        try:
            # Add model to LiteLLM
            await _add_model_to_litellm(
                config.litellm.normalized_base_url,
                config.litellm.api_key,
                model_metadata.id,
                model_metadata.litellm_fields,
                source_endpoint.normalized_base_url
            )
            logger.info("Successfully added model %s from %s to LiteLLM", model, source)
        except httpx.HTTPStatusError as exc:
            logger.error("LiteLLM rejected model %s: %s", model, exc.response.text)
            raise HTTPException(
                status_code=exc.response.status_code,
                detail=f"LiteLLM rejected the model: {exc.response.text}"
            )
        except httpx.RequestError as exc:
            logger.error("Failed to reach LiteLLM: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Failed to reach LiteLLM: {str(exc)}"
            )

        return RedirectResponse(url="/sources", status_code=303)

    @app.post("/litellm/models/delete")
    async def delete_model_from_litellm(model_id: str = Form(...)):
        """Delete a single model from LiteLLM."""
        config = load_config()

        if not config.litellm.configured:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="LiteLLM target is not configured"
            )

        try:
            async with httpx.AsyncClient() as client:
                await _delete_model_from_litellm(
                    client,
                    config.litellm.normalized_base_url,
                    config.litellm.api_key,
                    model_id
                )
            logger.info("Successfully deleted model %s from LiteLLM", model_id)
        except httpx.HTTPStatusError as exc:
            logger.error("LiteLLM rejected delete for %s: %s", model_id, exc.response.text)
            raise HTTPException(
                status_code=exc.response.status_code,
                detail=f"LiteLLM rejected the delete request: {exc.response.text}"
            )
        except httpx.RequestError as exc:
            logger.error("Failed to reach LiteLLM: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Failed to reach LiteLLM: {str(exc)}"
            )

        return RedirectResponse(url="/litellm", status_code=303)

    @app.post("/litellm/models/delete/bulk")
    async def delete_models_bulk(request: Request):
        """Delete multiple models from LiteLLM."""
        config = load_config()

        if not config.litellm.configured:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="LiteLLM target is not configured"
            )

        form_data = await request.form()
        model_ids = form_data.getlist("model_ids")

        if not model_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No model IDs provided"
            )

        deleted = []
        failed = []

        async with httpx.AsyncClient() as client:
            for model_id in model_ids:
                try:
                    await _delete_model_from_litellm(
                        client,
                        config.litellm.normalized_base_url,
                        config.litellm.api_key,
                        model_id
                    )
                    deleted.append(model_id)
                    logger.info("Successfully deleted model %s from LiteLLM", model_id)
                except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                    failed.append({"model_id": model_id, "error": str(exc)})
                    logger.error("Failed to delete model %s: %s", model_id, exc)

        if failed:
            logger.warning("Bulk delete completed with errors. Deleted: %d, Failed: %d",
                         len(deleted), len(failed))

        return RedirectResponse(url="/litellm", status_code=303)

    @app.get("/api/sources")
    async def api_sources() -> AppConfig:
        return load_config()

    @app.get("/api/providers")
    async def api_get_providers(session: AsyncSession = Depends(get_session)):
        """Get all providers from database."""
        from .crud import get_all_providers

        providers = await get_all_providers(session)
        return [
            {
                "id": p.id,
                "name": p.name,
                "base_url": p.base_url,
                "type": p.type,
                "prefix": p.prefix,
                "default_ollama_mode": p.default_ollama_mode,
                "created_at": p.created_at.isoformat(),
                "updated_at": p.updated_at.isoformat(),
            }
            for p in providers
        ]

    @app.get("/api/providers/{provider_id}/models")
    async def api_get_provider_models(
        provider_id: int,
        include_orphaned: bool = True,
        session: AsyncSession = Depends(get_session),
    ):
        """Get all models for a provider from database."""
        from .crud import get_models_by_provider, get_provider_by_id

        provider = await get_provider_by_id(session, provider_id)
        if not provider:
            raise HTTPException(status_code=404, detail="Provider not found")

        models = await get_models_by_provider(session, provider_id, include_orphaned)

        # Format models with prefix applied
        result = []
        for model in models:
            # Apply prefix to model name for display
            display_name = model.model_id
            if provider.prefix:
                display_name = f"{provider.prefix}/{model.model_id}"

            result.append(
                {
                    "id": model.id,
                    "model_id": model.model_id,  # Original name without prefix
                    "display_name": display_name,  # Name with prefix
                    "model_type": model.model_type,
                    "context_window": model.context_window,
                    "max_input_tokens": model.max_input_tokens,
                    "max_output_tokens": model.max_output_tokens,
                    "max_tokens": model.max_tokens,
                    "capabilities": model.capabilities_list,
                    "litellm_params": model.litellm_params_dict,
                    "user_params": model.user_params_dict,
                    "effective_params": model.effective_params,
                    "ollama_mode": model.ollama_mode,
                    "is_orphaned": model.is_orphaned,
                    "orphaned_at": model.orphaned_at.isoformat() if model.orphaned_at else None,
                    "user_modified": model.user_modified,
                    "first_seen": model.first_seen.isoformat(),
                    "last_seen": model.last_seen.isoformat(),
                }
            )

        return {
            "provider": {
                "id": provider.id,
                "name": provider.name,
                "prefix": provider.prefix,
            },
            "models": result,
        }

    @app.get("/api/models/db/{model_id}")
    async def api_get_model_by_id(model_id: int, session: AsyncSession = Depends(get_session)):
        """Get a specific model by database ID."""
        from .crud import get_model_by_id

        model = await get_model_by_id(session, model_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")

        provider = model.provider

        # Apply prefix
        display_name = model.model_id
        if provider.prefix:
            display_name = f"{provider.prefix}/{model.model_id}"

        return {
            "id": model.id,
            "model_id": model.model_id,
            "display_name": display_name,
            "model_type": model.model_type,
            "context_window": model.context_window,
            "max_input_tokens": model.max_input_tokens,
            "max_output_tokens": model.max_output_tokens,
            "max_tokens": model.max_tokens,
            "capabilities": model.capabilities_list,
            "litellm_params": model.litellm_params_dict,
            "user_params": model.user_params_dict,
            "effective_params": model.effective_params,
            "raw_metadata": model.raw_metadata_dict,
            "ollama_mode": model.ollama_mode,
            "is_orphaned": model.is_orphaned,
            "orphaned_at": model.orphaned_at.isoformat() if model.orphaned_at else None,
            "user_modified": model.user_modified,
            "first_seen": model.first_seen.isoformat(),
            "last_seen": model.last_seen.isoformat(),
            "provider": {
                "id": provider.id,
                "name": provider.name,
                "prefix": provider.prefix,
                "base_url": provider.base_url,
                "type": provider.type,
            },
        }

    @app.post("/api/models/db/{model_id}/params")
    async def api_update_model_params(
        model_id: int,
        params: dict,
        session: AsyncSession = Depends(get_session),
    ):
        """Update model parameters with user edits."""
        from .crud import get_model_by_id, update_model_params

        model = await get_model_by_id(session, model_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")

        try:
            await update_model_params(session, model, params)
            await session.commit()
            logger.info("Updated parameters for model %s (ID: %d)", model.model_id, model_id)

            return {
                "status": "success",
                "message": f"Parameters updated for model {model.model_id}",
                "model_id": model_id,
                "user_params": model.user_params_dict,
            }
        except Exception as exc:
            logger.exception("Failed to update model parameters")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to update parameters: {str(exc)}",
            )

    @app.delete("/api/models/db/{model_id}/params")
    async def api_reset_model_params(
        model_id: int,
        session: AsyncSession = Depends(get_session),
    ):
        """Reset model parameters to provider defaults."""
        from .crud import get_model_by_id, reset_model_params

        model = await get_model_by_id(session, model_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")

        try:
            await reset_model_params(session, model)
            await session.commit()
            logger.info("Reset parameters for model %s (ID: %d)", model.model_id, model_id)

            return {
                "status": "success",
                "message": f"Parameters reset to defaults for model {model.model_id}",
                "model_id": model_id,
                "litellm_params": model.litellm_params_dict,
            }
        except Exception as exc:
            logger.exception("Failed to reset model parameters")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to reset parameters: {str(exc)}",
            )

    @app.post("/api/models/db/{model_id}/refresh")
    async def api_refresh_model(
        model_id: int,
        session: AsyncSession = Depends(get_session),
    ):
        """Refresh a single model from its provider."""
        from .crud import get_model_by_id, upsert_model

        model = await get_model_by_id(session, model_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")

        provider = model.provider

        # Create SourceEndpoint from provider
        try:
            source = SourceEndpoint(
                name=provider.name,
                base_url=provider.base_url,
                type=SourceType(provider.type),
                api_key=provider.api_key,
                prefix=provider.prefix,
                default_ollama_mode=provider.default_ollama_mode,
            )
        except Exception as exc:
            logger.error("Failed to create SourceEndpoint from provider: %s", exc)
            raise HTTPException(status_code=500, detail="Invalid provider configuration")

        # Fetch models from provider
        try:
            source_models = await fetch_source_models(source)
        except httpx.RequestError as exc:
            logger.error("Failed to fetch models from provider %s: %s", provider.name, exc)
            raise HTTPException(
                status_code=502,
                detail=f"Failed to reach provider: {str(exc)}",
            )
        except ValueError as exc:
            logger.error("Invalid response from provider %s: %s", provider.name, exc)
            raise HTTPException(
                status_code=502,
                detail=f"Invalid provider response: {str(exc)}",
            )

        # Find the specific model
        model_metadata = next((m for m in source_models.models if m.id == model.model_id), None)
        if not model_metadata:
            # Model no longer exists in provider - mark as orphaned
            model.is_orphaned = True
            model.orphaned_at = datetime.now(UTC)
            await session.commit()

            logger.info("Model %s no longer exists in provider %s, marked as orphaned", model.model_id, provider.name)
            raise HTTPException(
                status_code=404,
                detail=f"Model {model.model_id} not found in provider {provider.name}",
            )

        # Update the model in database
        try:
            await upsert_model(session, provider, model_metadata)
            await session.commit()
            logger.info("Refreshed model %s from provider %s", model.model_id, provider.name)

            return {
                "status": "success",
                "message": f"Model {model.model_id} refreshed from provider",
                "model_id": model_id,
                "is_orphaned": False,
            }
        except Exception as exc:
            logger.exception("Failed to update model in database")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to update model: {str(exc)}",
            )

    @app.post("/api/models/db/{model_id}/push")
    async def api_push_model_to_litellm(
        model_id: int,
        session: AsyncSession = Depends(get_session),
    ):
        """Push a model to LiteLLM with its current effective parameters."""
        from .crud import get_model_by_id

        model = await get_model_by_id(session, model_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")

        config = load_config()
        if not config.litellm.configured:
            raise HTTPException(
                status_code=400,
                detail="LiteLLM target is not configured",
            )

        provider = model.provider

        # Get effective parameters (user_params if available, else litellm_params)
        effective_params = model.effective_params

        # Build model name with prefix
        model_name = model.model_id
        if provider.prefix:
            model_name = f"{provider.prefix}/{model.model_id}"

        # Determine ollama_mode (model override or provider default)
        ollama_mode = model.ollama_mode or provider.default_ollama_mode or "ollama"

        # Build litellm_params for the request
        litellm_params = {
            "model": f"ollama/{model.model_id}",  # Original name without prefix
            "api_base": provider.base_url,
        }

        # Add ollama_mode to model_info
        model_info = effective_params.copy()
        if provider.type == "ollama":
            model_info["mode"] = ollama_mode

        try:
            # Push to LiteLLM
            result = await _add_model_to_litellm(
                config.litellm.normalized_base_url,
                config.litellm.api_key,
                model_name,  # Use prefixed name
                model_info,
                provider.base_url,
            )
            logger.info("Pushed model %s to LiteLLM", model_name)

            return {
                "status": "success",
                "message": f"Model {model_name} pushed to LiteLLM",
                "model_id": model_id,
                "litellm_response": result,
            }
        except httpx.HTTPStatusError as exc:
            logger.error("LiteLLM rejected model %s: %s", model_name, exc.response.text)
            raise HTTPException(
                status_code=exc.response.status_code,
                detail=f"LiteLLM rejected the model: {exc.response.text}",
            )
        except httpx.RequestError as exc:
            logger.error("Failed to reach LiteLLM: %s", exc)
            raise HTTPException(
                status_code=502,
                detail=f"Failed to reach LiteLLM: {str(exc)}",
            )

    @app.post("/api/sources")
    async def api_add_source(endpoint: SourceEndpoint):
        """Add a new source via API."""
        # Check for duplicate source names
        config = load_config()
        if any(s.name == endpoint.name for s in config.sources):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Source with name '{endpoint.name}' already exists"
            )

        try:
            save_source(endpoint)
            logger.info("Added source via API: %s (%s) at %s", endpoint.name, endpoint.type, endpoint.base_url)
            return {"status": "success", "message": f"Source '{endpoint.name}' added successfully"}
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid source configuration: {str(exc)}"
            )
        except (OSError, RuntimeError) as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save source: {str(exc)}"
            )

    @app.delete("/api/sources/{name}")
    async def api_delete_source(name: str):
        """Delete a source via API."""
        config = load_config()
        if not any(s.name == name for s in config.sources):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Source '{name}' not found"
            )

        try:
            delete_source_from_config(name)
            logger.info("Deleted source via API: %s", name)
            return {"status": "success", "message": f"Source '{name}' deleted successfully"}
        except (OSError, RuntimeError) as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete source: {str(exc)}"
            )

    @app.get("/api/models")
    async def api_models() -> dict[str, SourceModels]:
        return await sync_state.get_models()

    @app.post("/api/models")
    async def api_add_model(source: str, model: str):
        """Add a model to LiteLLM via API."""
        config = load_config()

        if not config.litellm.configured:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="LiteLLM target is not configured"
            )

        # Find the source
        source_endpoint = next((s for s in config.sources if s.name == source), None)
        if not source_endpoint:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Source '{source}' not found"
            )

        # Get model metadata from sync state
        models_dict = await sync_state.get_models()
        source_models = models_dict.get(source)
        if not source_models:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No models found for source '{source}'. Try syncing first."
            )

        # Find the specific model
        model_metadata = next((m for m in source_models.models if m.id == model), None)
        if not model_metadata:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model}' not found in source '{source}'"
            )

        try:
            result = await _add_model_to_litellm(
                config.litellm.normalized_base_url,
                config.litellm.api_key,
                model_metadata.id,
                model_metadata.litellm_fields,
                source_endpoint.normalized_base_url
            )
            logger.info("Successfully added model %s from %s to LiteLLM via API", model, source)
            return {"status": "success", "message": f"Model '{model}' added to LiteLLM", "result": result}
        except httpx.HTTPStatusError as exc:
            logger.error("LiteLLM rejected model %s: %s", model, exc.response.text)
            raise HTTPException(
                status_code=exc.response.status_code,
                detail=f"LiteLLM rejected the model: {exc.response.text}"
            )
        except httpx.RequestError as exc:
            logger.error("Failed to reach LiteLLM: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Failed to reach LiteLLM: {str(exc)}"
            )

    @app.delete("/api/models/{model_id}")
    async def api_delete_model(model_id: str):
        """Delete a model from LiteLLM via API."""
        config = load_config()

        if not config.litellm.configured:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="LiteLLM target is not configured"
            )

        try:
            async with httpx.AsyncClient() as client:
                await _delete_model_from_litellm(
                    client,
                    config.litellm.normalized_base_url,
                    config.litellm.api_key,
                    model_id
                )
            logger.info("Successfully deleted model %s from LiteLLM via API", model_id)
            return {"status": "success", "message": f"Model '{model_id}' deleted from LiteLLM"}
        except httpx.HTTPStatusError as exc:
            logger.error("LiteLLM rejected delete for %s: %s", model_id, exc.response.text)
            raise HTTPException(
                status_code=exc.response.status_code,
                detail=f"LiteLLM rejected the delete request: {exc.response.text}"
            )
        except httpx.RequestError as exc:
            logger.error("Failed to reach LiteLLM: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Failed to reach LiteLLM: {str(exc)}"
            )

    @app.post("/api/sync")
    async def api_sync(session: AsyncSession = Depends(get_session)):
        """Trigger a manual sync via API."""
        config = load_config()
        try:
            results = await sync_once(config, session)
            await sync_state.update(results)
            model_count = sum(len(source_models.models) for source_models in results.values())
            return {
                "status": "success",
                "message": "Sync completed successfully",
                "sources_synced": len(results),
                "total_models": model_count
            }
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("API sync failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Sync failed: {str(exc)}"
            )
        except Exception as exc:
            logger.exception("Unexpected error during API sync")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Unexpected error during sync: {str(exc)}"
            )

    return app


