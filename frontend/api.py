"""Frontend FastAPI application - UI and API only, no sync logic."""
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Depends, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.database import create_engine, init_session_maker, get_session, ensure_minimum_schema
from shared.crud import get_all_providers, get_config, get_provider_by_id
from frontend.routes import providers, models, admin, compat, litellm
from backend import provider_sync
from sqlalchemy import select, func
from shared.db_models import Model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    logger.info("Frontend service starting...")

    # Create engine (backend handles migrations)
    engine = create_engine()
    init_session_maker(engine)
    await ensure_minimum_schema(engine)

    logger.info("Frontend service ready")

    yield

    logger.info("Frontend service shutting down...")

    # Cleanup
    await engine.dispose()


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="LiteLLM Updater",
        description="Model synchronization UI and API",
        version="0.2.0",
        lifespan=lifespan
    )

    # Mount static files
    static_path = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

    # Initialize templates
    templates_path = Path(__file__).parent / "templates"
    templates = Jinja2Templates(directory=str(templates_path))

    # Helper function for templates
    def _human_source_type(source_type: str) -> str:
        """Display name for source type."""
        type_names = {
            "ollama": "Ollama",
            "openai": "OpenAI-compatible",
            "compat": "Compat"
        }
        return type_names.get(source_type, source_type)

    templates.env.globals["human_source_type"] = _human_source_type

    # Include API routers
    app.include_router(providers.router, prefix="/api/providers", tags=["providers"])
    app.include_router(models.router, prefix="/api/models", tags=["models"])
    app.include_router(admin.router, prefix="/api/admin", tags=["admin"])
    app.include_router(compat.router, prefix="/api/compat", tags=["compat"])
    app.include_router(litellm.router, prefix="/litellm", tags=["litellm"])

    # HTML Routes
    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request, session = Depends(get_session)):
        """Dashboard page."""
        providers_list = await get_all_providers(session)
        config = await get_config(session)
        litellm_models = 0
        try:
            litellm_data = await litellm.list_litellm_models(session)
            if isinstance(litellm_data, dict):
                litellm_models = len(litellm_data.get("models", []) or [])
        except Exception:
            litellm_models = 0

        # Calculate stats
        total_models = await session.scalar(select(func.count()).select_from(Model))
        orphaned_models = await session.scalar(select(func.count()).select_from(Model).where(Model.is_orphaned == True))
        modified_models = await session.scalar(select(func.count()).select_from(Model).where(Model.user_modified == True))

        stats = {
            "providers": len(providers_list),
            "models": total_models or 0,
            "orphaned": orphaned_models or 0,
            "modified": modified_models or 0,
            "litellm_models": litellm_models
        }

        # Convert config to template-compatible format
        config_dict = {
            "litellm": {
                "configured": bool(config.litellm_base_url),
                "base_url": config.litellm_base_url or "",
                "api_key": config.litellm_api_key or ""
            },
            "sync_interval_seconds": config.sync_interval_seconds
        }
        config_dict["sources"] = [
            {
                "name": p.name,
                "type": p.type,
                "base_url": p.base_url,
                "prefix": p.prefix,
                "tags": p.tags_list,
            }
            for p in providers_list
        ]

        return templates.TemplateResponse("index.html", {
            "request": request,
            "providers": providers_list,
            "config": config_dict,
            "stats": stats,
            "last_synced": None
        })

    @app.get("/sources", response_class=HTMLResponse)
    async def sources_page(request: Request, session = Depends(get_session)):
        """Providers page."""
        config = await get_config(session)
        config_dict = {
            "litellm": {
                "configured": bool(config.litellm_base_url),
                "base_url": config.litellm_base_url or "",
                "api_key": config.litellm_api_key or ""
            },
            "sync_interval_seconds": config.sync_interval_seconds
        }
        return templates.TemplateResponse("sources.html", {
            "request": request,
            "config": config_dict
        })

    @app.get("/compat", response_class=HTMLResponse)
    async def compat_page(request: Request, session = Depends(get_session)):
        """Compat models page."""
        config = await get_config(session)
        config_dict = {
            "litellm": {
                "configured": bool(config.litellm_base_url),
                "base_url": config.litellm_base_url or "",
                "api_key": config.litellm_api_key or ""
            },
            "sync_interval_seconds": config.sync_interval_seconds
        }
        return templates.TemplateResponse("compat.html", {
            "request": request,
            "config": config_dict
        })

    @app.get("/litellm", response_class=HTMLResponse)
    async def litellm_page(request: Request, session = Depends(get_session)):
        """LiteLLM models page."""
        config = await get_config(session)
        litellm_data = await litellm.list_litellm_models(session)
        litellm_models = []
        litellm_error = None
        if isinstance(litellm_data, dict):
            litellm_models = litellm_data.get("models", [])
            litellm_error = litellm_data.get("error")

        # Normalize models for template (ensure id/name fields exist)
        normalized_models = []
        for m in litellm_models or []:
            # LiteLLM may return different keys; create consistent shape
            model_info = m.get("model_info") or {}
            litellm_params = m.get("litellm_params") or {}
            model_id = m.get("model_name") or m.get("id") or m.get("name") or model_info.get("id") or "unknown"
            database_id = m.get("database_id") or model_info.get("id")
            tags = (
                m.get("tags")
                or litellm_params.get("tags")
                or model_info.get("tags")
                or []
            )
            capabilities = model_info.get("capabilities") or []
            normalized_models.append({
                "id": model_id,
                "database_id": database_id,
                "litellm_fields": m.get("litellm_fields") or model_info or {},
                "tags": tags,
                "capabilities": capabilities,
                "raw": m or {},
            })

        config_dict = {
            "litellm": {
                "configured": bool(config.litellm_base_url),
                "base_url": config.litellm_base_url or "",
                "api_key": config.litellm_api_key or ""
            },
            "sync_interval_seconds": config.sync_interval_seconds
        }
        return templates.TemplateResponse("litellm.html", {
            "request": request,
            "config": config_dict,
            "litellm_models": normalized_models,
            "litellm_error": litellm_error,
            "fetched_at": datetime.now(timezone.utc).isoformat()
        })

    @app.get("/admin", response_class=HTMLResponse)
    async def admin_page(request: Request, session = Depends(get_session)):
        """Admin configuration page."""
        providers_list = await get_all_providers(session)
        config = await get_config(session)
        config_dict = {
            "litellm": {
                "configured": bool(config.litellm_base_url),
                "base_url": config.litellm_base_url or "",
                "api_key": config.litellm_api_key or ""
            },
            "sync_interval_seconds": config.sync_interval_seconds
        }
        return templates.TemplateResponse("admin.html", {
            "request": request,
            "providers": providers_list,
            "config": config_dict
        })

    @app.post("/sync")
    async def manual_sync(request: Request, session = Depends(get_session)):
        """
        Manual sync trigger - spawns a background task so the UI does not time out.
        Sync = fetch from providers and push to LiteLLM.
        """
        from shared.database import async_session_maker  # use global session maker set in lifespan

        # Snapshot providers/config first
        providers_list = await get_all_providers(session)
        config = await get_config(session)
        provider_ids = [p.id for p in providers_list if p.sync_enabled and p.type != "compat"]
        provider_names = [p.name for p in providers_list if p.id in provider_ids]

        async def _run_sync():
            if async_session_maker is None:
                return
            for pid in provider_ids:
                async with async_session_maker() as sync_session:
                    provider = await get_provider_by_id(sync_session, pid)
                    if not provider:
                        continue
                    try:
                        cfg = await get_config(sync_session)
                        # Full sync: fetch then push to LiteLLM
                        await provider_sync.sync_provider(sync_session, cfg, provider, push_to_litellm=True)  # type: ignore[arg-type]
                        await sync_session.commit()
                    except Exception:
                        await sync_session.rollback()

        import asyncio

        asyncio.create_task(_run_sync())
        return {"status": "started", "providers": provider_names}

    @app.post("/admin/providers")
    async def create_provider_legacy(
        name: str = Form(...),
        base_url: str = Form(...),
        type: str = Form(...),
        api_key: str | None = Form(None),
        prefix: str | None = Form(None),
        default_ollama_mode: str | None = Form(None),
        session: AsyncSession = Depends(get_session)
    ):
        """Legacy endpoint for creating providers - redirects to API."""
        from shared.crud import create_provider
        provider = await create_provider(
            session,
            name=name,
            base_url=base_url,
            type_=type,
            api_key=api_key,
            prefix=prefix,
            default_ollama_mode=default_ollama_mode
        )
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/admin", status_code=303)

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy", "service": "frontend"}

    return app


# For development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(create_app(), host="0.0.0.0", port=8000)
