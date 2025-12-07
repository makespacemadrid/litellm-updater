"""FastAPI application exposing UI and APIs for syncing models."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime, timedelta

from fastapi import Body, Depends, FastAPI, Form, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession

import httpx
from alembic.util.exc import CommandError
from pydantic import BaseModel

from .config import (
    add_source as save_source,
    load_config,
    remove_source as delete_source_from_config,
    set_sync_interval,
    update_litellm_target,
)
from .database import (
    create_engine,
    ensure_minimum_schema,
    get_database_url,
    get_session,
    async_session_maker,
    init_session_maker,
    run_migrations,
)
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
from .tags import generate_model_tags, normalize_tags, parse_tags_input

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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

DEFAULT_LITELLM_API_KEY = "sk-1234"


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
    db_url = get_database_url()
    engine = create_engine(db_url)
    try:
        # Run migrations in executor to avoid nested event loop conflict
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, run_migrations, db_url)
        logger.info("Database migrations applied")
    except CommandError as exc:
        logger.warning("Alembic migration scripts missing (%s); applying safety schema fixups", exc)
        await ensure_minimum_schema(engine)
    except Exception:
        logger.exception("Database migration failed; falling back to create_all + safety fixups")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await ensure_minimum_schema(engine)
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
    litellm_base_url: str,
    api_key: str | None,
    model_name: str,
    litellm_params: dict,
    model_info: dict | None = None,
) -> dict:
    """Add a single model to LiteLLM using /model/new endpoint.

    Args:
        litellm_base_url: Base URL of LiteLLM instance
        api_key: API key for authentication
        model_name: Display name for the model in LiteLLM (e.g., "mks-ollama/qwen3:8b")
        litellm_params: Complete litellm_params dict (model, api_base, tags, etc.)
        model_info: Metadata about the model (capabilities, pricing, limits, etc.)
    """
    url = f"{litellm_base_url}/model/new"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Build payload for LiteLLM's /model/new endpoint
    payload = {
        "model_name": model_name,
        "litellm_params": litellm_params,
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
    async def index(request: Request, session: AsyncSession = Depends(get_session)):
        from .config_db import load_config_with_db_providers

        config = await load_config_with_db_providers(session)
        last_synced = await sync_state.get_last_synced()
        # Stats for overview page
        stats = {"providers": 0, "models": 0, "orphaned": 0, "modified": 0, "litellm_models": 0}

        try:
            from .crud import get_all_providers, get_models_by_provider

            providers = await get_all_providers(session)
            stats["providers"] = len(providers)
            for provider in providers:
                models = await get_models_by_provider(session, provider.id, include_orphaned=True)
                stats["models"] += len(models)
                stats["orphaned"] += len([m for m in models if m.is_orphaned])
                stats["modified"] += len([m for m in models if m.user_modified])

            # Fetch LiteLLM model count
            if config.litellm.configured:
                try:
                    litellm_models = await fetch_litellm_target_models(
                        LitellmDestination(base_url=config.litellm.base_url, api_key=config.litellm.api_key)
                    )
                    # Count only lupdater-managed models
                    stats["litellm_models"] = len([m for m in litellm_models if "lupdater" in (m.tags or [])])
                except Exception as exc:
                    logger.warning("Failed to fetch LiteLLM model count: %s", exc)
        except Exception:
            logger.exception("Failed to load stats for overview")

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "config": config,
                "last_synced": last_synced,
                "stats": stats,
                "human_source_type": _human_source_type,
            },
        )

    @app.get("/sources", response_class=HTMLResponse)
    async def sources(request: Request):
        """Database-driven providers and models page."""
        return templates.TemplateResponse(
            "sources.html",
            {
                "request": request,
            },
        )

    @app.get("/compat", response_class=HTMLResponse)
    async def compat(request: Request):
        """Compatibility models management page."""
        return templates.TemplateResponse(
            "compat.html",
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
    async def admin(request: Request, session: AsyncSession = Depends(get_session)):
        from .config_db import load_config_with_db_providers

        config = await load_config_with_db_providers(session)
        return templates.TemplateResponse(
            "admin.html",
            {"request": request, "config": config, "human_source_type": _human_source_type},
        )

    @app.get("/litellm", response_class=HTMLResponse)
    async def litellm(request: Request, session: AsyncSession = Depends(get_session)):
        from .config_db import load_config_with_db_providers

        config = await load_config_with_db_providers(session)
        litellm_models = []
        litellm_error: str | None = None
        fetched_at: datetime | None = None

        logger.info(f"LiteLLM config: base_url={config.litellm.base_url}, configured={config.litellm.configured}")

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
    async def update_litellm(
        base_url: str = Form(""),
        api_key: str | None = Form(None),
        session: AsyncSession = Depends(get_session),
    ):
        """Update LiteLLM destination in database."""
        from .crud import update_config

        try:
            await update_config(
                session,
                litellm_base_url=base_url or None,
                litellm_api_key=api_key or None,
            )
            await session.commit()
            logger.info("Updated LiteLLM target: %s", base_url or "disabled")
        except Exception as exc:
            await session.rollback()
            logger.error("Failed to update LiteLLM target: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to update LiteLLM target: {str(exc)}",
            )

        return RedirectResponse(url="/admin", status_code=303)

    @app.post("/admin/interval")
    async def update_interval(
        sync_interval_seconds: int = Form(...),
        session: AsyncSession = Depends(get_session),
    ):
        """Update sync interval in database."""
        from .crud import update_config

        if sync_interval_seconds < 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Sync interval must be >= 0",
            )

        if sync_interval_seconds > 0 and sync_interval_seconds < 30:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Sync interval must be at least 30 seconds when enabled",
            )

        try:
            await update_config(session, sync_interval_seconds=sync_interval_seconds)
            await session.commit()
            logger.info("Updated sync interval: %d seconds", sync_interval_seconds)
        except Exception as exc:
            await session.rollback()
            logger.error("Failed to update sync interval: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to update sync interval: {str(exc)}",
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
        tags: str | None = Form(None),
        access_groups: str | None = Form(None),
        sync_enabled: bool = Form(True),
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
        if type not in ("ollama", "openai", "compat"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Type must be 'ollama', 'openai', or 'compat'"
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
            parsed_tags = parse_tags_input(tags)
            parsed_access_groups = parse_tags_input(access_groups)
            await create_provider(
                session,
                name=name,
                base_url=base_url,
                type_=type,
                api_key=api_key or None,
                prefix=prefix or None,
                default_ollama_mode=default_ollama_mode or None,
                tags=parsed_tags,
                access_groups=parsed_access_groups,
                sync_enabled=sync_enabled,
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
        tags: str | None = Form(None),
        access_groups: str | None = Form(None),
        sync_enabled: bool | None = Form(None),
    ):
        """Update an existing provider."""
        from .crud import get_provider_by_id, update_provider

        provider = await get_provider_by_id(session, provider_id)
        if not provider:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Provider not found")

        # Validate type if provided
        if type and type not in ("ollama", "openai", "compat"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Type must be 'ollama', 'openai', or 'compat'"
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
            parsed_tags = parse_tags_input(tags)
            parsed_access_groups = parse_tags_input(access_groups)
            await update_provider(
                session,
                provider,
                name=name,
                base_url=base_url,
                type_=type,
                api_key=api_key,
                prefix=prefix,
                default_ollama_mode=default_ollama_mode,
                tags=parsed_tags if tags is not None else None,
                access_groups=parsed_access_groups if access_groups is not None else None,
                sync_enabled=sync_enabled,
            )
            logger.info("Updated provider: %s", provider.name)
        except Exception as exc:
            logger.exception("Failed to update provider")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to update provider: {str(exc)}",
            )

        return RedirectResponse(url="/admin", status_code=303)

    @app.patch("/api/providers/{provider_id}/sync")
    async def toggle_provider_sync(
        provider_id: int,
        payload: dict = Body(...),
        session: AsyncSession = Depends(get_session),
    ):
        """Toggle sync_enabled for a provider."""
        from .crud import get_provider_by_id, update_provider

        provider = await get_provider_by_id(session, provider_id)
        if not provider:
            raise HTTPException(status_code=404, detail="Provider not found")

        sync_enabled = payload.get("sync_enabled")
        if sync_enabled is None:
            raise HTTPException(status_code=400, detail="sync_enabled is required")

        try:
            await update_provider(session, provider, sync_enabled=sync_enabled)
            await session.commit()
            logger.info(
                "Updated sync_enabled=%s for provider %s (ID: %d)",
                sync_enabled,
                provider.name,
                provider_id,
            )
            return {
                "status": "success",
                "provider_id": provider_id,
                "sync_enabled": sync_enabled,
            }
        except Exception as exc:
            logger.exception("Failed to update provider sync setting")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to update sync setting: {str(exc)}",
            )

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

    @app.post("/sync")
    async def run_sync(session: AsyncSession = Depends(get_session)):
        from .config_db import load_config_with_db_providers

        config = await load_config_with_db_providers(session)
        try:
            results, stats = await sync_once(config, session)
            await sync_state.update(results)
            logger.info("Manual sync completed: %s", stats)
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
        session: AsyncSession = Depends(get_session),
    ):
        """Add a single model from a source to LiteLLM."""
        from .config_db import load_config_with_db_providers

        config = await load_config_with_db_providers(session)

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
            # Build display name (use model_metadata.id directly for legacy endpoint)
            display_name = model_metadata.id

            # Determine ollama_mode
            ollama_mode = source_endpoint.default_ollama_mode or "ollama"

            # Build litellm_params based on source type and mode
            litellm_params = {}

            if source_endpoint.type.value == "openai":
                # OpenAI-compatible provider
                litellm_params["model"] = f"openai/{model_metadata.id}"
                litellm_params["api_base"] = source_endpoint.normalized_base_url
                litellm_params.setdefault("api_key", source_endpoint.api_key or DEFAULT_LITELLM_API_KEY)
            elif source_endpoint.type.value == "ollama":
                # Ollama provider: set model prefix and api_base based on mode
                if ollama_mode == "openai":
                    litellm_params["model"] = f"openai/{model_metadata.id}"
                    # OpenAI mode uses /v1 endpoint
                    api_base = source_endpoint.normalized_base_url.rstrip("/")
                    litellm_params["api_base"] = f"{api_base}/v1"
                    litellm_params.setdefault("api_key", source_endpoint.api_key or DEFAULT_LITELLM_API_KEY)
                else:
                    litellm_params["model"] = f"ollama/{model_metadata.id}"
                    litellm_params["api_base"] = source_endpoint.normalized_base_url

            auto_tags = generate_model_tags(
                provider_name=source,
                provider_type=source_endpoint.type.value,
                metadata=model_metadata,
                provider_tags=source_endpoint.tags,
                mode=ollama_mode,
            )
            # Add tags inside litellm_params
            litellm_params["tags"] = auto_tags

            # Build model_info with metadata
            model_info = model_metadata.litellm_fields.copy()
            model_info["tags"] = auto_tags

            # Set correct litellm_provider based on source type and mode
            if source_endpoint.type.value == "openai":
                model_info["litellm_provider"] = "openai"
            elif source_endpoint.type.value == "ollama":
                model_info["mode"] = ollama_mode
                if ollama_mode == "openai":
                    model_info["litellm_provider"] = "openai"
                else:
                    model_info["litellm_provider"] = "ollama"

            # Add model to LiteLLM
            await _add_model_to_litellm(
                config.litellm.normalized_base_url,
                config.litellm.api_key,
                display_name,
                litellm_params,
                model_info,
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
    async def delete_model_from_litellm(
        model_id: str = Form(...),
        session: AsyncSession = Depends(get_session),
    ):
        """Delete a single model from LiteLLM."""
        from .config_db import load_config_with_db_providers

        config = await load_config_with_db_providers(session)

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
    async def delete_models_bulk(request: Request, session: AsyncSession = Depends(get_session)):
        """Delete multiple models from LiteLLM."""
        from .config_db import load_config_with_db_providers

        config = await load_config_with_db_providers(session)

        if not config.litellm.configured:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="LiteLLM target is not configured"
            )

        # Accept JSON to avoid form field limit (1000 fields max)
        try:
            body = await request.json()
            model_ids = body.get("model_ids", [])
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid JSON body: {exc}"
            )

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
                "tags": p.tags_list,
                "access_groups": p.access_groups_list,
                "sync_enabled": p.sync_enabled,
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
                    "system_tags": model.system_tags_list,
                    "user_tags": model.user_tags_list,
                    "tags": model.all_tags,
                    "access_groups": model.access_groups_list,
                    "ollama_mode": model.ollama_mode,
                    "is_orphaned": model.is_orphaned,
                    "orphaned_at": model.orphaned_at.isoformat() if model.orphaned_at else None,
                    "user_modified": model.user_modified,
                    "sync_enabled": model.sync_enabled,
                    "first_seen": model.first_seen.isoformat(),
                    "last_seen": model.last_seen.isoformat(),
                }
            )

        return {
            "provider": {
                "id": provider.id,
                "name": provider.name,
                "prefix": provider.prefix,
                "tags": provider.tags_list,
                "access_groups": provider.access_groups_list,
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
            "system_tags": model.system_tags_list,
            "user_tags": model.user_tags_list,
            "tags": model.all_tags,
            "access_groups": model.access_groups_list,
            "raw_metadata": model.raw_metadata_dict,
            "ollama_mode": model.ollama_mode,
            "is_orphaned": model.is_orphaned,
            "orphaned_at": model.orphaned_at.isoformat() if model.orphaned_at else None,
            "user_modified": model.user_modified,
            "sync_enabled": model.sync_enabled,
            "first_seen": model.first_seen.isoformat(),
            "last_seen": model.last_seen.isoformat(),
            "provider": {
                "id": provider.id,
                "name": provider.name,
                "prefix": provider.prefix,
                "base_url": provider.base_url,
                "type": provider.type,
                "tags": provider.tags_list,
                "access_groups": provider.access_groups_list,
            },
        }

    @app.post("/api/models/db/{model_id}/params")
    async def api_update_model_params(
        model_id: int,
        payload: dict = Body(...),
        session: AsyncSession = Depends(get_session),
    ):
        """Update model parameters with user edits."""
        from .crud import get_model_by_id, update_model_params

        model = await get_model_by_id(session, model_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")

        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Body must be a JSON object")

        is_dict_payload = isinstance(payload, dict)
        params = payload.get("params") if is_dict_payload and "params" in payload else payload
        tags = payload.get("tags") if is_dict_payload else None
        access_groups = payload.get("access_groups") if is_dict_payload else None
        sync_enabled = payload.get("sync_enabled") if is_dict_payload else None

        if isinstance(tags, str):
            normalized_tags = parse_tags_input(tags)
        else:
            normalized_tags = normalize_tags(tags) if tags is not None else None

        if isinstance(access_groups, str):
            normalized_access_groups = parse_tags_input(access_groups)
        else:
            normalized_access_groups = normalize_tags(access_groups) if access_groups is not None else None

        if is_dict_payload and "params" not in payload and set(payload.keys()) <= {"tags", "access_groups", "sync_enabled"}:
            params = None

        try:
            await update_model_params(session, model, params, normalized_tags, normalized_access_groups, sync_enabled)
            await session.commit()
            logger.info("Updated parameters for model %s (ID: %d)", model.model_id, model_id)

            return {
                "status": "success",
                "message": f"Parameters updated for model {model.model_id}",
                "model_id": model_id,
                "user_params": model.user_params_dict,
                "user_tags": model.user_tags_list,
                "access_groups": model.access_groups_list,
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
                "user_tags": model.user_tags_list,
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

        # Update the model in database with full_update=True to refresh all fields
        try:
            _, _ = await upsert_model(session, provider, model_metadata, full_update=True)
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

    @app.post("/api/models/db/{model_id}/sync-toggle")
    async def api_toggle_model_sync(
        model_id: int,
        payload: dict = Body(...),
        session: AsyncSession = Depends(get_session),
    ):
        """Toggle sync_enabled for a model."""
        from .crud import get_model_by_id

        model = await get_model_by_id(session, model_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")

        sync_enabled = payload.get("sync_enabled")
        if sync_enabled is None:
            raise HTTPException(status_code=400, detail="sync_enabled field is required")

        try:
            model.sync_enabled = bool(sync_enabled)
            await session.commit()
            logger.info("Toggled sync_enabled=%s for model: %s", sync_enabled, model.model_id)
            return {"status": "success", "sync_enabled": model.sync_enabled}
        except Exception as exc:
            await session.rollback()
            logger.exception("Failed to toggle sync_enabled for model")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to toggle sync: {str(exc)}",
            )

    @app.delete("/api/models/db/{model_id}")
    async def api_delete_model_from_db(
        model_id: int,
        session: AsyncSession = Depends(get_session),
    ):
        """Delete a model from the database."""
        from .crud import get_model_by_id, delete_model

        model = await get_model_by_id(session, model_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")

        model_name = model.model_id
        provider_name = model.provider.name

        try:
            await delete_model(session, model)
            await session.commit()
            logger.info("Deleted model %s (ID: %d) from database", model_name, model_id)
            return {
                "status": "success",
                "message": f"Model '{model_name}' deleted from database",
                "model_id": model_id,
            }
        except Exception as exc:
            await session.rollback()
            logger.exception("Failed to delete model from database")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete model: {str(exc)}",
            )

    @app.post("/api/models/db/delete-bulk")
    async def api_delete_models_bulk(
        payload: dict = Body(...),
        session: AsyncSession = Depends(get_session),
    ):
        """Delete multiple models from the database."""
        from .crud import get_model_by_id, delete_model

        model_ids = payload.get("model_ids", [])
        if not model_ids:
            raise HTTPException(status_code=400, detail="model_ids array is required")

        deleted = []
        failed = []

        for model_id in model_ids:
            try:
                model = await get_model_by_id(session, model_id)
                if model:
                    model_name = model.model_id
                    await delete_model(session, model)
                    deleted.append({"id": model_id, "name": model_name})
                else:
                    failed.append({"id": model_id, "error": "Model not found"})
            except Exception as exc:
                logger.error("Failed to delete model ID %d: %s", model_id, exc)
                failed.append({"id": model_id, "error": str(exc)})

        try:
            await session.commit()
            logger.info("Bulk delete: %d models deleted, %d failed", len(deleted), len(failed))
            return {
                "status": "success",
                "deleted": deleted,
                "failed": failed,
                "total_deleted": len(deleted),
                "total_failed": len(failed),
            }
        except Exception as exc:
            await session.rollback()
            logger.exception("Failed to commit bulk delete")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to commit bulk delete: {str(exc)}",
            )

    @app.post("/api/models/db/reset-all")
    async def api_reset_all_models(
        session: AsyncSession = Depends(get_session),
    ):
        """Delete all models from the database. WARNING: This cannot be undone!"""
        from sqlalchemy import delete, select, func
        from .db_models import Model

        try:
            # Count models before deletion
            result = await session.execute(select(func.count()).select_from(Model))
            models_before = result.scalar()

            # Delete all models
            await session.execute(delete(Model))
            await session.commit()

            logger.warning("RESET: Deleted all %d models from database", models_before)
            return {
                "status": "success",
                "message": f"All {models_before} models deleted from database",
                "total_deleted": models_before,
            }
        except Exception as exc:
            await session.rollback()
            logger.exception("Failed to reset all models")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to reset models: {str(exc)}",
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

        from .config_db import load_config_with_db_providers

        config = await load_config_with_db_providers(session)
        if not config.litellm.configured:
            raise HTTPException(
                status_code=400,
                detail="LiteLLM target is not configured",
            )

        provider = model.provider

        # For compat models, inherit properties from mapped model
        if provider.type == "compat" and model.mapped_provider_id and model.mapped_model_id:
            from .crud import get_provider_by_id, get_model_by_provider_and_name

            mapped_provider = await get_provider_by_id(session, model.mapped_provider_id)
            mapped_model = await get_model_by_provider_and_name(
                session, model.mapped_provider_id, model.mapped_model_id
            )

            if not mapped_provider or not mapped_model:
                raise HTTPException(
                    status_code=400,
                    detail=f"Mapped model not found for compat model {model.model_id}",
                )

            # Use mapped model's connection params and metadata
            mapped_ollama_mode = mapped_model.ollama_mode or mapped_provider.default_ollama_mode or "ollama"

            # Build litellm_params using mapped model
            litellm_params = {}
            if mapped_provider.type == "openai":
                litellm_params["model"] = f"openai/{mapped_model.model_id}"
                litellm_params["api_base"] = mapped_provider.base_url
                litellm_params.setdefault("api_key", mapped_provider.api_key or DEFAULT_LITELLM_API_KEY)
            elif mapped_provider.type == "ollama":
                if mapped_ollama_mode == "openai":
                    litellm_params["model"] = f"openai/{mapped_model.model_id}"
                    api_base = mapped_provider.base_url.rstrip("/")
                    litellm_params["api_base"] = f"{api_base}/v1"
                    litellm_params.setdefault("api_key", mapped_provider.api_key or DEFAULT_LITELLM_API_KEY)
                else:
                    litellm_params["model"] = f"ollama/{mapped_model.model_id}"
                    litellm_params["api_base"] = mapped_provider.base_url

            # Use compat model's display name
            display_name = model.model_id
            if provider.prefix:
                display_name = f"{provider.prefix}/{model.model_id}"

            # Use compat model's tags and access_groups, but mapped model's metadata
            combined_tags = model.all_tags or ["lupdater", f"provider:{provider.name}", "type:compat"]
            litellm_params["tags"] = combined_tags

            # Use mapped model's metadata (effective_params)
            model_info = mapped_model.effective_params.copy()
            model_info["tags"] = combined_tags

            # Set litellm_provider based on mapped provider
            if mapped_provider.type == "openai":
                model_info["litellm_provider"] = "openai"
            elif mapped_provider.type == "ollama":
                model_info["mode"] = mapped_ollama_mode
                model_info["litellm_provider"] = "openai" if mapped_ollama_mode == "openai" else "ollama"

            # Use compat model's access_groups (overrides mapped model)
            effective_access_groups = model.get_effective_access_groups()
            if effective_access_groups:
                model_info["access_groups"] = effective_access_groups

        else:
            # Regular model (non-compat)
            # Get effective parameters (user_params if available, else litellm_params)
            effective_params = model.effective_params

            # Build display name with prefix (shown in LiteLLM UI)
            display_name = model.model_id
            if provider.prefix:
                display_name = f"{provider.prefix}/{model.model_id}"

            # Determine ollama_mode (model override or provider default)
            ollama_mode = model.ollama_mode or provider.default_ollama_mode or "ollama"

            # Build litellm_params based on provider type and mode
            litellm_params = {}

            if provider.type == "openai":
                # OpenAI-compatible provider
                litellm_params["model"] = f"openai/{model.model_id}"
                litellm_params["api_base"] = provider.base_url
                litellm_params.setdefault("api_key", provider.api_key or DEFAULT_LITELLM_API_KEY)
            elif provider.type == "ollama":
                # Ollama provider: set model prefix and api_base based on mode
                if ollama_mode == "openai":
                    litellm_params["model"] = f"openai/{model.model_id}"
                    # OpenAI mode uses /v1 endpoint
                    api_base = provider.base_url.rstrip("/")
                    litellm_params["api_base"] = f"{api_base}/v1"
                    litellm_params.setdefault("api_key", provider.api_key or DEFAULT_LITELLM_API_KEY)
                else:
                    litellm_params["model"] = f"ollama/{model.model_id}"
                    litellm_params["api_base"] = provider.base_url

            combined_tags = model.all_tags or ["lupdater", f"provider:{provider.name}", f"type:{provider.type}"]
            litellm_params["tags"] = combined_tags

            # Build model_info with metadata
            model_info = effective_params.copy()
            model_info["tags"] = combined_tags

            # Set correct litellm_provider based on provider type and mode
            if provider.type == "openai":
                model_info["litellm_provider"] = "openai"
            elif provider.type == "ollama":
                model_info["mode"] = ollama_mode
                if ollama_mode == "openai":
                    model_info["litellm_provider"] = "openai"
                else:
                    model_info["litellm_provider"] = "ollama"

            # Add access_groups if configured (model overrides provider)
            effective_access_groups = model.get_effective_access_groups()
            if effective_access_groups:
                model_info["access_groups"] = effective_access_groups

        try:
            # Push to LiteLLM
            result = await _add_model_to_litellm(
                config.litellm.normalized_base_url,
                config.litellm.api_key,
                display_name,  # Display name with prefix
                litellm_params,  # Complete connection config with tags
                model_info,  # Metadata
            )
            logger.info("Pushed model %s to LiteLLM", display_name)

            return {
                "status": "success",
                "message": f"Model {display_name} pushed to LiteLLM",
                "model_id": model_id,
                "litellm_response": result,
            }
        except httpx.HTTPStatusError as exc:
            logger.error("LiteLLM rejected model %s: %s", display_name, exc.response.text)
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

    @app.post("/api/models/push-all")
    async def api_push_all_models_to_litellm(
        session: AsyncSession = Depends(get_session),
    ):
        """Push all non-orphaned models from all providers to LiteLLM."""
        from .config_db import load_config_with_db_providers
        from .crud import get_all_providers, get_models_by_provider

        config = await load_config_with_db_providers(session)
        if not config.litellm.configured:
            raise HTTPException(
                status_code=400,
                detail="LiteLLM target is not configured",
            )

        # Fetch existing models from LiteLLM to avoid duplicates
        existing_unique_ids = set()
        try:
            litellm_models = await fetch_litellm_target_models(
                LitellmDestination(base_url=config.litellm.base_url, api_key=config.litellm.api_key)
            )
            for m in litellm_models:
                combined_tags = set(m.tags or [])
                if isinstance(m.raw, dict):
                    combined_tags.update(m.raw.get("litellm_params", {}).get("tags", []) or [])
                    combined_tags.update(m.raw.get("model_info", {}).get("tags", []) or [])
                unique_id_tag = next((t for t in combined_tags if isinstance(t, str) and t.startswith("unique_id:")), None)
                if unique_id_tag:
                    # Normalize to lowercase for case-insensitive comparison
                    existing_unique_ids.add(unique_id_tag.lower())
            logger.info("Found %d existing models in LiteLLM with unique_id tags", len(existing_unique_ids))
        except Exception as exc:
            logger.warning("Failed to fetch existing LiteLLM models: %s", exc)

        providers = await get_all_providers(session)

        results = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "errors": []
        }

        for provider in providers:
            # Skip providers with sync disabled
            if not provider.sync_enabled:
                logger.info("Skipping provider %s (sync disabled)", provider.name)
                continue

            # Get all non-orphaned models for this provider
            models = await get_models_by_provider(session, provider.id, include_orphaned=False)

            for model in models:
                # Skip models with sync disabled
                if not model.sync_enabled:
                    logger.info("Skipping model %s (sync disabled)", model.model_id)
                    continue
                results["total"] += 1

                # Check if model already exists in LiteLLM using unique_id tag (case-insensitive)
                unique_id_tag = f"unique_id:{provider.name}/{model.model_id}".lower()
                if unique_id_tag in existing_unique_ids:
                    results["skipped"] += 1
                    logger.info("Skipping duplicate model %s (already in LiteLLM)", model.model_id)
                    continue

                # For compat models, inherit properties from mapped model
                if provider.type == "compat" and model.mapped_provider_id and model.mapped_model_id:
                    from .crud import get_provider_by_id, get_model_by_provider_and_name

                    try:
                        mapped_provider = await get_provider_by_id(session, model.mapped_provider_id)
                        mapped_model = await get_model_by_provider_and_name(
                            session, model.mapped_provider_id, model.mapped_model_id
                        )

                        if not mapped_provider or not mapped_model:
                            results["failed"] += 1
                            results["errors"].append(f"{model.model_id}: Mapped model not found")
                            logger.warning("Mapped model not found for compat model %s", model.model_id)
                            continue

                        # Use mapped model's connection params and metadata
                        mapped_ollama_mode = mapped_model.ollama_mode or mapped_provider.default_ollama_mode or "ollama"

                        # Build litellm_params using mapped model
                        litellm_params = {}
                        if mapped_provider.type == "openai":
                            litellm_params["model"] = f"openai/{mapped_model.model_id}"
                            litellm_params["api_base"] = mapped_provider.base_url
                            litellm_params.setdefault("api_key", mapped_provider.api_key or DEFAULT_LITELLM_API_KEY)
                        elif mapped_provider.type == "ollama":
                            if mapped_ollama_mode == "openai":
                                litellm_params["model"] = f"openai/{mapped_model.model_id}"
                                api_base = mapped_provider.base_url.rstrip("/")
                                litellm_params["api_base"] = f"{api_base}/v1"
                                litellm_params.setdefault("api_key", mapped_provider.api_key or DEFAULT_LITELLM_API_KEY)
                            else:
                                litellm_params["model"] = f"ollama/{mapped_model.model_id}"
                                litellm_params["api_base"] = mapped_provider.base_url

                        # Use compat model's display name
                        display_name = model.model_id
                        if provider.prefix:
                            display_name = f"{provider.prefix}/{model.model_id}"

                        # Use compat model's tags
                        combined_tags = model.all_tags or ["lupdater", f"provider:{provider.name}", "type:compat"]
                        litellm_params["tags"] = combined_tags

                        # Use mapped model's metadata (effective_params)
                        model_info = mapped_model.effective_params.copy()
                        model_info["tags"] = combined_tags

                        # Set litellm_provider based on mapped provider
                        if mapped_provider.type == "openai":
                            model_info["litellm_provider"] = "openai"
                        elif mapped_provider.type == "ollama":
                            model_info["mode"] = mapped_ollama_mode
                            model_info["litellm_provider"] = "openai" if mapped_ollama_mode == "openai" else "ollama"

                        # Use compat model's access_groups (overrides mapped model)
                        effective_access_groups = model.get_effective_access_groups()
                        if effective_access_groups:
                            model_info["access_groups"] = effective_access_groups

                    except Exception as exc:
                        results["failed"] += 1
                        results["errors"].append(f"{model.model_id}: {str(exc)}")
                        logger.error("Failed to process compat model %s: %s", model.model_id, exc)
                        continue

                else:
                    # Regular model (non-compat)
                    # Get effective parameters
                    effective_params = model.effective_params

                    # Build display name with prefix (shown in LiteLLM UI)
                    display_name = model.model_id
                    if provider.prefix:
                        display_name = f"{provider.prefix}/{model.model_id}"

                    # Determine ollama_mode
                    ollama_mode = model.ollama_mode or provider.default_ollama_mode or "ollama"

                    # Build litellm_params based on provider type and mode
                    litellm_params = {}

                    if provider.type == "openai":
                        # OpenAI-compatible provider
                        litellm_params["model"] = f"openai/{model.model_id}"
                        litellm_params["api_base"] = provider.base_url
                        litellm_params.setdefault("api_key", provider.api_key or DEFAULT_LITELLM_API_KEY)
                    elif provider.type == "ollama":
                        # Ollama provider: set model prefix and api_base based on mode
                        if ollama_mode == "openai":
                            litellm_params["model"] = f"openai/{model.model_id}"
                            # OpenAI mode uses /v1 endpoint
                            api_base = provider.base_url.rstrip("/")
                            litellm_params["api_base"] = f"{api_base}/v1"
                            litellm_params.setdefault("api_key", provider.api_key or DEFAULT_LITELLM_API_KEY)
                        else:
                            litellm_params["model"] = f"ollama/{model.model_id}"
                            litellm_params["api_base"] = provider.base_url

                    # Generate tags using the tag generator to ensure unique_id is included
                    if model.all_tags:
                        combined_tags = model.all_tags
                    else:
                        # Fallback: generate tags from model metadata
                        from .models import ModelMetadata
                        from .tags import generate_model_tags

                        metadata = ModelMetadata.from_raw(model.model_id, model.raw_metadata_dict)
                        combined_tags = generate_model_tags(
                            provider_name=provider.name,
                            provider_type=provider.type,
                            metadata=metadata,
                            provider_tags=provider.tags_list,
                            mode=ollama_mode,
                        )

                    litellm_params["tags"] = combined_tags

                    # Build model_info with metadata
                    model_info = effective_params.copy()
                    model_info["tags"] = combined_tags

                    # Set correct litellm_provider based on provider type and mode
                    if provider.type == "openai":
                        model_info["litellm_provider"] = "openai"
                    elif provider.type == "ollama":
                        model_info["mode"] = ollama_mode
                        if ollama_mode == "openai":
                            model_info["litellm_provider"] = "openai"
                        else:
                            model_info["litellm_provider"] = "ollama"

                    # Add access_groups if configured (model overrides provider)
                    effective_access_groups = model.get_effective_access_groups()
                    if effective_access_groups:
                        model_info["access_groups"] = effective_access_groups

                try:
                    # Push to LiteLLM
                    await _add_model_to_litellm(
                        config.litellm.normalized_base_url,
                        config.litellm.api_key,
                        display_name,  # Display name with prefix
                        litellm_params,  # Complete connection config with tags
                        model_info,  # Metadata
                    )
                    results["success"] += 1
                    logger.info("Pushed model %s to LiteLLM", display_name)
                except httpx.HTTPStatusError as exc:
                    results["failed"] += 1
                    error_msg = f"{display_name}: {exc.response.text}"
                    results["errors"].append(error_msg)
                    logger.warning("LiteLLM rejected model %s: %s", display_name, exc.response.text)
                except httpx.RequestError as exc:
                    results["failed"] += 1
                    error_msg = f"{display_name}: Failed to reach LiteLLM"
                    results["errors"].append(error_msg)
                    logger.warning("Failed to push model %s: %s", display_name, exc)
                except Exception as exc:
                    results["failed"] += 1
                    error_msg = f"{display_name}: {str(exc)}"
                    results["errors"].append(error_msg)
                    logger.exception("Unexpected error pushing model %s", display_name)

        return {
            "status": "completed",
            "message": f"Pushed {results['success']}/{results['total']} models to LiteLLM",
            "results": results,
        }

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
            # Build display name (use model_metadata.id directly for legacy endpoint)
            display_name = model_metadata.id

            # Determine ollama_mode
            ollama_mode = source_endpoint.default_ollama_mode or "ollama"

            # Build litellm_params based on source type and mode
            litellm_params = {}

            if source_endpoint.type.value == "litellm":
                # OpenAI-compatible provider
                litellm_params["model"] = f"openai/{model_metadata.id}"
                litellm_params["api_base"] = source_endpoint.normalized_base_url
                litellm_params.setdefault("api_key", source_endpoint.api_key or DEFAULT_LITELLM_API_KEY)
            elif source_endpoint.type.value == "ollama":
                # Ollama provider: set model prefix and api_base based on mode
                if ollama_mode == "openai":
                    litellm_params["model"] = f"openai/{model_metadata.id}"
                    # OpenAI mode uses /v1 endpoint
                    api_base = source_endpoint.normalized_base_url.rstrip("/")
                    litellm_params["api_base"] = f"{api_base}/v1"
                    litellm_params.setdefault("api_key", source_endpoint.api_key or DEFAULT_LITELLM_API_KEY)
                else:
                    litellm_params["model"] = f"ollama/{model_metadata.id}"
                    litellm_params["api_base"] = source_endpoint.normalized_base_url

            auto_tags = generate_model_tags(
                provider_name=source,
                provider_type=source_endpoint.type.value,
                metadata=model_metadata,
                provider_tags=source_endpoint.tags,
                mode=ollama_mode,
            )
            # Add tags inside litellm_params
            litellm_params["tags"] = auto_tags

            # Build model_info with metadata
            model_info = model_metadata.litellm_fields.copy()
            model_info["tags"] = auto_tags

            # Set correct litellm_provider based on source type and mode
            if source_endpoint.type.value == "openai":
                model_info["litellm_provider"] = "openai"
            elif source_endpoint.type.value == "ollama":
                model_info["mode"] = ollama_mode
                if ollama_mode == "openai":
                    model_info["litellm_provider"] = "openai"
                else:
                    model_info["litellm_provider"] = "ollama"

            result = await _add_model_to_litellm(
                config.litellm.normalized_base_url,
                config.litellm.api_key,
                display_name,
                litellm_params,
                model_info,
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
        from .config_db import load_config_with_db_providers

        config = await load_config_with_db_providers(session)
        try:
            results, stats = await sync_once(config, session)
            await sync_state.update(results)
            return {
                "status": "success",
                "message": "Sync completed successfully",
                "statistics": stats
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

    # Compat Models API Endpoints

    @app.get("/api/compat/models")
    async def api_get_compat_models(session: AsyncSession = Depends(get_session)):
        """Get all compat models."""
        from .crud import get_all_compat_models, get_provider_by_id

        models = await get_all_compat_models(session)
        result = []

        for model in models:
            mapped_provider = None
            mapped_model = None
            if model.mapped_provider_id:
                mapped_prov = await get_provider_by_id(session, model.mapped_provider_id)
                if mapped_prov:
                    mapped_provider = {
                        "id": mapped_prov.id,
                        "name": mapped_prov.name,
                        "type": mapped_prov.type,
                    }
            if model.mapped_model_id:
                mapped_model = model.mapped_model_id

            result.append({
                "id": model.id,
                "model_name": model.model_id,
                "mapped_provider": mapped_provider,
                "mapped_model_id": mapped_model,
                "user_params": model.user_params_dict,
                "access_groups": model.access_groups_list,
                "sync_enabled": model.sync_enabled,
                "created_at": model.created_at.isoformat(),
                "updated_at": model.updated_at.isoformat(),
            })

        return result

    @app.post("/api/compat/models")
    async def api_create_compat_model(
        model_name: str = Form(...),
        mapped_provider_id: int = Form(None),
        mapped_model_id: str = Form(None),
        access_groups: str = Form(None),
        session: AsyncSession = Depends(get_session),
    ):
        """Create a new compat model."""
        from .crud import create_compat_model, get_provider_by_id
        from .tags import parse_tags_input

        # Validate mapped provider if provided
        if mapped_provider_id:
            mapped_provider = await get_provider_by_id(session, mapped_provider_id)
            if not mapped_provider:
                raise HTTPException(status_code=404, detail="Mapped provider not found")

        # Parse access_groups
        access_groups_list = parse_tags_input(access_groups) if access_groups else None

        # Create compat model
        model = await create_compat_model(
            session,
            model_name=model_name,
            mapped_provider_id=mapped_provider_id,
            mapped_model_id=mapped_model_id,
            access_groups=access_groups_list,
        )

        await session.commit()
        return {
            "status": "success",
            "message": f"Compat model '{model_name}' created",
            "model_id": model.id,
        }

    @app.put("/api/compat/models/{model_id}")
    async def api_update_compat_model(
        model_id: int,
        mapped_provider_id: int = Form(None),
        mapped_model_id: str = Form(None),
        access_groups: str = Form(None),
        session: AsyncSession = Depends(get_session),
    ):
        """Update a compat model's mapping."""
        from .crud import get_model_by_id, update_compat_model, get_provider_by_id
        from .tags import parse_tags_input

        model = await get_model_by_id(session, model_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")

        # Validate it's a compat model
        if model.provider.type != "compat":
            raise HTTPException(status_code=400, detail="Model is not a compat model")

        # Validate mapped provider if provided
        if mapped_provider_id:
            mapped_provider = await get_provider_by_id(session, mapped_provider_id)
            if not mapped_provider:
                raise HTTPException(status_code=404, detail="Mapped provider not found")

        # Parse access_groups
        access_groups_list = parse_tags_input(access_groups) if access_groups else None

        # Update compat model
        await update_compat_model(
            session,
            model=model,
            mapped_provider_id=mapped_provider_id,
            mapped_model_id=mapped_model_id,
            access_groups=access_groups_list,
        )

        await session.commit()
        return {
            "status": "success",
            "message": f"Compat model '{model.model_id}' updated",
        }

    @app.delete("/api/compat/models/{model_id}")
    async def api_delete_compat_model(
        model_id: int,
        session: AsyncSession = Depends(get_session),
    ):
        """Delete a compat model."""
        from .crud import get_model_by_id, delete_model

        model = await get_model_by_id(session, model_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")

        # Validate it's a compat model
        if model.provider.type != "compat":
            raise HTTPException(status_code=400, detail="Model is not a compat model")

        model_name = model.model_id
        await delete_model(session, model)
        await session.commit()

        return {
            "status": "success",
            "message": f"Compat model '{model_name}' deleted",
        }

    @app.post("/api/compat/register-defaults")
    async def api_register_default_compat_models(
        session: AsyncSession = Depends(get_session),
    ):
        """Register all default OpenAI-compatible models to the compat provider."""
        from .crud import create_compat_model, get_provider_by_name, get_model_by_provider_and_name
        from .default_compat_models import DEFAULT_COMPAT_MODELS, DEFAULT_OLLAMA_BASE, get_model_count_summary
        import re

        # Find the Ollama provider that matches the default base URL
        from .crud import get_all_providers
        providers = await get_all_providers(session)

        ollama_provider = None
        for provider in providers:
            if provider.type == "ollama" and provider.base_url.rstrip("/") == DEFAULT_OLLAMA_BASE.rstrip("/"):
                ollama_provider = provider
                break

        if not ollama_provider:
            raise HTTPException(
                status_code=400,
                detail=f"No Ollama provider found for {DEFAULT_OLLAMA_BASE}. Please create one first.",
            )

        # Track results
        results = {
            "total": len(DEFAULT_COMPAT_MODELS),
            "success": 0,
            "skipped": 0,
            "failed": 0,
            "errors": [],
        }

        # Register each model to the compat provider in database
        for model_def in DEFAULT_COMPAT_MODELS:
            model_name = model_def["model_name"]

            try:
                # Extract Ollama model name from litellm_params.model
                # e.g., "ollama/gpt-oss:20b" or "ollama/qwen3:4b"
                litellm_model = model_def["litellm_params"]["model"]
                match = re.match(r"^ollama/(.+)$", litellm_model)
                if not match:
                    results["failed"] += 1
                    results["errors"].append(f"{model_name}: Invalid model format '{litellm_model}'")
                    continue

                ollama_model_name = match.group(1)  # e.g., "gpt-oss:20b"

                # Extract access_groups from tags (use first tag that looks like an access group)
                tags = model_def["litellm_params"].get("tags", [])
                access_groups = ["compat"]  # Default
                if "compat" in tags:
                    access_groups = ["compat"]

                # Check if compat model already exists
                compat_provider = await get_provider_by_name(session, "compat_models")
                if compat_provider:
                    existing_model = await get_model_by_provider_and_name(
                        session, compat_provider.id, model_name
                    )
                    if existing_model:
                        results["skipped"] += 1
                        logger.info(f"Skipped existing compat model: {model_name}")
                        continue

                # Create compat model with mapping to Ollama provider
                model = await create_compat_model(
                    session,
                    model_name=model_name,
                    mapped_provider_id=ollama_provider.id,
                    mapped_model_id=ollama_model_name,
                    access_groups=access_groups,
                )

                results["success"] += 1
                logger.info(f"Created compat model: {model_name} → {ollama_model_name}")

            except Exception as exc:
                results["failed"] += 1
                error_msg = f"{model_name}: {str(exc)}"
                results["errors"].append(error_msg)
                logger.exception(f"Error creating compat model {model_name}")

        # Commit all changes
        await session.commit()

        # Get category summary
        summary = get_model_count_summary()

        return {
            "status": "completed",
            "message": f"Created {results['success']} compat models, skipped {results['skipped']} existing, {results['failed']} failed",
            "results": results,
            "summary": summary,
            "provider": {
                "id": ollama_provider.id,
                "name": ollama_provider.name,
                "base_url": ollama_provider.base_url,
            },
        }

    return app
