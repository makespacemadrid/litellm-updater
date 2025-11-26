"""FastAPI application exposing UI and APIs for syncing models."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager, suppress
from datetime import datetime
from typing import Dict

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import httpx

from .config import (
    add_source,
    load_config,
    remove_source,
    set_sync_interval,
    update_litellm_target,
)
from .models import AppConfig, LitellmTarget, SourceEndpoint, SourceModels, SourceType
from .sources import fetch_litellm_target_models, fetch_source_models
from .sync import start_scheduler, sync_once

logger = logging.getLogger(__name__)


class SyncState:
    """In-memory store for the latest synchronization results."""

    def __init__(self) -> None:
        self.models: Dict[str, SourceModels] = {}
        self.last_synced: datetime | None = None

    def update(self, results: Dict[str, SourceModels]) -> None:
        self.models = results
        self.last_synced = datetime.utcnow()

    def update_source(self, source_name: str, source_models: SourceModels) -> None:
        self.models[source_name] = source_models
        self.last_synced = datetime.utcnow()


sync_state = SyncState()


def _human_source_type(source_type: SourceType) -> str:
    return "Ollama" if source_type is SourceType.OLLAMA else "LiteLLM / OpenAI"


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(start_scheduler(load_config, sync_state.update))
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


def create_app() -> FastAPI:
    app = FastAPI(title="LiteLLM Updater", description="Sync models into LiteLLM", lifespan=lifespan)
    templates = Jinja2Templates(directory="litellm_updater/templates")
    app.mount("/static", StaticFiles(directory="litellm_updater/static"), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        config = load_config()
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "config": config,
                "sync_state": sync_state,
                "human_source_type": _human_source_type,
            },
        )

    @app.get("/providers", response_class=HTMLResponse)
    async def providers_page(request: Request):
        config = load_config()
        models = sync_state.models
        return templates.TemplateResponse(
            "providers.html",
            {
                "request": request,
                "config": config,
                "models": models,
                "last_synced": sync_state.last_synced,
                "human_source_type": _human_source_type,
            },
        )

    @app.get("/models")
    async def models_endpoint(request: Request, source: str | None = None):
        """Expose the latest synced models or redirect browsers to the UI.

        When the request prefers HTML (e.g. a user navigating directly in a
        browser), redirect to the providers page to keep the existing UX. API
        consumers can request JSON to retrieve the in-memory models, optionally
        filtered by source name.
        """

        accepts = request.headers.get("accept", "")
        prefers_json = "application/json" in accepts or "*/*" == accepts

        if not prefers_json:
            return RedirectResponse(url="/providers", status_code=308)

        if source:
            return sync_state.models.get(source) or {}

        return sync_state.models

    @app.get("/admin", response_class=HTMLResponse)
    async def admin_page(request: Request):
        config = load_config()
        return templates.TemplateResponse(
            "admin.html",
            {"request": request, "config": config, "human_source_type": _human_source_type},
        )

    @app.get("/litellm", response_class=HTMLResponse)
    async def litellm_page(request: Request):
        config = load_config()
        litellm_models = []
        litellm_error: str | None = None
        fetched_at: datetime | None = None

        if config.litellm.configured:
            try:
                litellm_models = await fetch_litellm_target_models(config.litellm)
                fetched_at = datetime.utcnow()
            except Exception as exc:  # pragma: no cover - runtime logging
                logger.exception("Failed fetching LiteLLM models: %s", exc)
                litellm_error = str(exc)
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
    async def add_source_form(
        name: str = Form(...),
        base_url: str = Form(...),
        source_type: SourceType = Form(...),
        api_key: str | None = Form(None),
    ):
        endpoint = SourceEndpoint(name=name, base_url=base_url, type=source_type, api_key=api_key or None)
        add_source(endpoint)
        return RedirectResponse(url="/admin", status_code=303)

    @app.post("/admin/sources/delete")
    async def delete_source_form(name: str = Form(...)):
        remove_source(name)
        return RedirectResponse(url="/admin", status_code=303)

    @app.post("/admin/litellm")
    async def update_litellm(base_url: str = Form(""), api_key: str | None = Form(None)):
        target = LitellmTarget(
            base_url=base_url or None,
            api_key=api_key or None,
        )
        update_litellm_target(target)
        return RedirectResponse(url="/admin", status_code=303)

    @app.post("/admin/interval")
    async def update_interval(sync_interval_seconds: int = Form(...)):
        set_sync_interval(sync_interval_seconds)
        return RedirectResponse(url="/admin", status_code=303)

    @app.post("/sync")
    async def manual_sync():
        config = load_config()
        try:
            results = await sync_once(config)
            sync_state.update(results)
        except Exception:  # pragma: no cover - defensive logging for manual runs
            logger.exception("Manual sync failed")
        return RedirectResponse(url="/providers", status_code=303)

    @app.post("/providers/refresh")
    async def refresh_provider_models(name: str = Form(...)):
        config = load_config()
        source = next((source for source in config.sources if source.name == name), None)
        if not source:
            logger.warning("Attempted to refresh unknown source %s", name)
            return RedirectResponse(url="/providers", status_code=303)

        try:
            models = await fetch_source_models(source)
            sync_state.update_source(name, models)
        except httpx.RequestError as exc:  # pragma: no cover - runtime logging
            logger.warning(
                "Failed refreshing models for %s at %s: %s", name, source.base_url, exc
            )
        except Exception:  # pragma: no cover - runtime logging for diagnostics
            logger.exception("Failed refreshing models for %s", name)

        return RedirectResponse(url="/providers", status_code=303)

    @app.get("/api/sources")
    async def api_sources() -> AppConfig:
        return load_config()

    @app.get("/api/models")
    async def api_models() -> Dict[str, SourceModels]:
        return sync_state.models

    return app


