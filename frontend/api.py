"""Frontend FastAPI application - UI and API only, no sync logic."""
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Depends, Form, HTTPException
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
from shared import __version__ as APP_VERSION

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
        title="LiteLLM Companion",
        description="Model synchronization UI and API",
        version=APP_VERSION,
        lifespan=lifespan
    )

    # Mount static files
    static_path = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

    # Initialize templates
    templates_path = Path(__file__).parent / "templates"
    templates = Jinja2Templates(directory=str(templates_path))
    templates.env.globals["APP_VERSION"] = APP_VERSION

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

        # Get category stats
        from shared.db_models import Provider as ProviderModel
        from shared.categorization import get_category_stats
        result = await session.execute(
            select(Model)
            .join(ProviderModel)
            .where(
                Model.is_orphaned == False,
                ProviderModel.type.notin_(["compat", "completion"])
            )
        )
        all_models = result.scalars().all()
        models_data = [
            {"capabilities": m.capabilities, "system_tags": m.system_tags}
            for m in all_models
        ]
        category_stats = get_category_stats(models_data)

        stats = {
            "providers": len(providers_list),
            "models": total_models or 0,
            "orphaned": orphaned_models or 0,
            "modified": modified_models or 0,
            "litellm_models": litellm_models,
            "categories": category_stats
        }

        # Convert config to template-compatible format
        config_dict = {
            "litellm": {
                "configured": bool(config.litellm_base_url),
                "base_url": config.litellm_base_url or "",
                "api_key": config.litellm_api_key or ""
            },
            "sync_interval_seconds": config.sync_interval_seconds,
            "default_pricing_profile": config.default_pricing_profile,
            "default_pricing_override": config.default_pricing_override_dict,
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

        # Prepare last sync info
        last_sync_info = None
        if config.last_sync_at:
            last_sync_info = {
                "timestamp": config.last_sync_at,
                "results": config.last_sync_results_dict
            }

        return templates.TemplateResponse("index.html", {
            "request": request,
            "providers": providers_list,
            "config": config_dict,
            "stats": stats,
            "last_sync": last_sync_info
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
            "sync_interval_seconds": config.sync_interval_seconds,
            "default_pricing_profile": config.default_pricing_profile,
            "default_pricing_override": config.default_pricing_override_dict,
        }
        return templates.TemplateResponse("sources.html", {
            "request": request,
            "config": config_dict
        })

    @app.get("/models", response_class=HTMLResponse)
    async def models_page(request: Request, session = Depends(get_session)):
        """Models browser page."""
        config = await get_config(session)
        config_dict = {
            "litellm": {
                "configured": bool(config.litellm_base_url),
                "base_url": config.litellm_base_url or "",
                "api_key": config.litellm_api_key or ""
            },
            "sync_interval_seconds": config.sync_interval_seconds,
            "default_pricing_profile": config.default_pricing_profile,
            "default_pricing_override": config.default_pricing_override_dict,
        }
        return templates.TemplateResponse("models.html", {
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
            "sync_interval_seconds": config.sync_interval_seconds,
            "default_pricing_profile": config.default_pricing_profile,
            "default_pricing_override": config.default_pricing_override_dict,
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

            # Extract capabilities from multiple sources
            capabilities = []

            # Check for explicit capabilities array
            if model_info.get("capabilities"):
                capabilities = model_info.get("capabilities")
            else:
                # Extract from supports_* fields
                if model_info.get("supports_vision") or model_info.get("supports_image"):
                    capabilities.append("Vision")
                if model_info.get("supports_function_calling"):
                    capabilities.append("Function Calling")
                if model_info.get("supports_tool_choice"):
                    capabilities.append("Tools")
                if model_info.get("supports_audio"):
                    capabilities.append("Audio")
                if model_info.get("supports_embedding"):
                    capabilities.append("Embedding")

            normalized_models.append({
                "id": model_id,
                "database_id": database_id,
                "litellm_fields": m.get("litellm_fields") or model_info or {},
                "tags": tags,
                "capabilities": capabilities,
                "max_input_tokens": model_info.get("max_input_tokens"),
                "context_window": model_info.get("max_tokens") or model_info.get("context_window"),
                "raw": m or {},
            })

        config_dict = {
            "litellm": {
                "configured": bool(config.litellm_base_url),
                "base_url": config.litellm_base_url or "",
                "api_key": config.litellm_api_key or ""
            },
            "sync_interval_seconds": config.sync_interval_seconds,
            "default_pricing_profile": config.default_pricing_profile,
            "default_pricing_override": config.default_pricing_override_dict,
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
            "sync_interval_seconds": config.sync_interval_seconds,
            "default_pricing_profile": config.default_pricing_profile,
            "default_pricing_override": config.default_pricing_override_dict,
        }
        return templates.TemplateResponse("admin.html", {
            "request": request,
            "providers": providers_list,
            "config": config_dict
        })

    @app.post("/sync")
    async def manual_sync(request: Request, session = Depends(get_session)):
        """
        Manual sync trigger - runs synchronously and returns detailed results.
        Sync = fetch from providers and push to LiteLLM.
        """
        # Get providers and config
        providers_list = await get_all_providers(session)
        config = await get_config(session)
        enabled_providers = [
            p for p in providers_list if p.sync_enabled and p.type not in ("compat", "completion")
        ]

        # Track results
        results = {
            "total_providers": len(enabled_providers),
            "providers": [],
            "total_models_fetched": 0,
            "errors": []
        }

        # Sync each provider
        for provider in enabled_providers:
            provider_result = {
                "name": provider.name,
                "status": "success",
                "models_fetched": 0,
                "models_added": 0,
                "models_updated": 0,
                "models_orphaned": 0,
                "error": None
            }

            try:
                # Run sync for this provider
                sync_result = await provider_sync.sync_provider(
                    session, config, provider, push_to_litellm=True
                )

                # Extract results if available
                if isinstance(sync_result, dict):
                    added = sync_result.get("added", 0)
                    updated = sync_result.get("updated", 0)
                    orphaned = sync_result.get("orphaned", 0)
                    provider_result["models_fetched"] = added + updated
                    provider_result["models_added"] = added
                    provider_result["models_updated"] = updated
                    provider_result["models_orphaned"] = orphaned
                    results["total_models_fetched"] += provider_result["models_fetched"]

                await session.commit()
            except Exception as e:
                await session.rollback()
                provider_result["status"] = "error"
                provider_result["error"] = str(e)
                results["errors"].append(f"{provider_result['name']}: {str(e)}")
                logger.exception(f"Error syncing provider {provider_result['name']}")

            results["providers"].append(provider_result)

        results["success"] = len(results["errors"]) == 0

        # Save sync results to config
        config.last_sync_at = datetime.now(timezone.utc)
        config.last_sync_results_dict = results
        await session.commit()

        return results

    @app.post("/admin/providers")
    async def create_provider_legacy(
        name: str = Form(...),
        base_url: str = Form(...),
        type: str = Form(...),
        api_key: str | None = Form(None),
        prefix: str | None = Form(None),
        default_ollama_mode: str | None = Form(None),
        model_filter: str | None = Form(None),
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
            default_ollama_mode=default_ollama_mode,
            model_filter=model_filter
        )
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/admin", status_code=303)

    @app.post("/api/providers/test-connection")
    async def test_provider_connection(
        base_url: str = Form(...),
        type: str = Form(...),
        api_key: str | None = Form(None),
    ):
        """
        Test connection to a provider without saving it.
        Returns connection status and any error messages.
        """
        from shared.models import SourceEndpoint, SourceType
        from shared.sources import fetch_source_models
        import httpx

        try:
            # Create a test source endpoint
            source = SourceEndpoint(
                name="test",
                base_url=base_url,
                type=SourceType(type),
                api_key=api_key,
            )

            # Try to fetch models (this will test the connection)
            source_models = await fetch_source_models(source)

            return {
                "success": True,
                "message": f"Successfully connected! Found {len(source_models.models)} models.",
                "model_count": len(source_models.models)
            }

        except ConnectionError as e:
            return {
                "success": False,
                "message": str(e),
                "error_type": "connection_error"
            }
        except TimeoutError as e:
            return {
                "success": False,
                "message": str(e),
                "error_type": "timeout_error"
            }
        except ValueError as e:
            return {
                "success": False,
                "message": str(e),
                "error_type": "validation_error"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Unexpected error: {str(e)}",
                "error_type": "unknown_error"
            }

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy", "service": "frontend"}

    return app


# For development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(create_app(), host="0.0.0.0", port=8000)
