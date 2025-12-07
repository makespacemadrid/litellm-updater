# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LiteLLM Updater now runs as two FastAPI services built from shared code:
- `backend/` → headless sync worker (`backend/sync_worker.py`) that fetches provider models and can push them into LiteLLM on a schedule.
- `frontend/` → UI + API (`frontend/api.py`) for manual fetch/push/sync and CRUD over providers/models.

Data lives in `./data/models.db` (SQLite) mounted into both services. Docker Compose also brings up LiteLLM (`http://localhost:4000`, API key `sk-1234` by default) and the UI on `http://localhost:4001`.

## Terminology (keep consistent in UI + API)
- Fetch: only pull models from providers into the local database.
- Push: send database models to LiteLLM (deduped, no new fetch).
- Sync: fetch + push in one operation.

## Current layout
- `shared/`: database models/CRUD, source fetchers, normalization, tags, and config helpers shared by both services.
- `backend/`: provider sync pipeline, LiteLLM client, and the scheduler entrypoint (`python -m backend.sync_worker`).
- `frontend/`: FastAPI UI, templates, and the provider/model routes that call into `backend.provider_sync`.
- `litellm_updater/`: legacy entrypoint kept for compatibility; most logic now lives under `shared/` + `backend/`.

## Development Commands

### Setup
```bash
# Install dependencies
pip install -e .

# Install with dev dependencies (for testing and linting)
pip install -e ".[dev]"
```

### Running the service
```bash
# Using the CLI entrypoint
PORT=8000 litellm-updater

# Or using uvicorn directly
PORT=8000 uvicorn litellm_updater.web:create_app --host 0.0.0.0 --port $PORT
```

### Testing
```bash
# Run all tests
pytest

# Run integration tests (requires live endpoints configured in tests/.env)
pytest tests/test_sources_integration.py -q

# Setup integration tests
cp tests/example.env tests/.env
# Edit tests/.env with TEST_OLLAMA_URL, TEST_OPENAI_URL and optional API keys
```

### Docker
```bash

# Using docker-compose
cp example.env .env
docker compose --env-file .env build --no-cache model-updater-backend model-updater-web
docker compose --env-file .env up -d
```

### Linting
```bash
# Run ruff for linting/formatting
ruff check .
ruff format .
```

## Deployment & Live Testing

Compose brings up:
- `model-updater-backend`: sync worker (no HTTP).
- `model-updater-web`: UI/API on `http://localhost:4001`.
- `litellm`: target proxy on `http://localhost:4000` (`Authorization: Bearer sk-1234`).
- `db`: Postgres backing LiteLLM.
- `watchtower`: optional image updater (labelled).

Rebuild + relaunch after code changes:
```bash
docker compose build --no-cache model-updater-backend model-updater-web
docker compose up -d
```

Quick checks:
```bash
docker compose ps
curl -s http://localhost:4001/health
curl -s -H "Authorization: Bearer sk-1234" http://localhost:4000/health/liveliness
docker compose logs --tail=50 model-updater-web
docker compose logs --tail=50 model-updater-backend
```

## Operational notes
- Fetch = load models from providers into the database, Push = register existing DB models into LiteLLM, Sync = fetch + push. UI buttons and routes follow this naming (`/api/providers/fetch-all`, `/api/providers/sync-all`, per-provider Fetch/Sync/Push).
- LiteLLM pushes dedupe by lowercasing `unique_id` and pruning duplicates before registration; per-provider Push and Push All avoid re-adding existing models.
- Ollama details: by default only `/api/tags` is fetched. Set `FETCH_OLLAMA_DETAILS=true` to pull `/api/show` per model; heavy fields (tensors/modelfile/license/etc.) are stripped before storing to keep memory usage low.

## Architecture

### Core Components

**Database Layer** (`shared/database.py`, `shared/db_models.py`, `shared/crud.py`)
- SQLite database (`data/models.db`) with async SQLAlchemy 2.0+
- **Providers table**: Stores provider configuration with prefix and default_ollama_mode
- **Models table**: Persists model metadata, tracks orphan status, preserves user edits
- Alembic migrations for schema versioning
- CRUD operations for all database entities
- Session management with FastAPI dependency injection

**Configuration Layer** (`shared/config.py`, `shared/config_db.py`)
- **Providers managed in database** - use `config_db.py` helpers to load from DB
- Default config auto-created on first run with LiteLLM at `http://localhost:4000`
- Uses Pydantic validation with `AppConfig` model

**Source Fetchers** (`shared/sources.py`)
- Supports two source types: Ollama (`/api/tags`) and LiteLLM/OpenAI (`/v1/models`)
- `fetch_source_models()` dispatches to the correct fetcher based on `SourceType`
- Ollama fetcher includes `_clean_ollama_payload()` to strip large/redundant fields (tensors, modelfile, license). Optional deep fetch via `FETCH_OLLAMA_DETAILS=true` pulls `/api/show` per model and still strips heavy fields.
- For LiteLLM sources, fetches list then individual model details from `/v1/models/{id}` (when supported)
- `fetch_litellm_target_models()` uses LiteLLM's `/model/info` endpoint which includes model UUIDs and complete metadata

**Model Normalization** (`shared/models.py`)
- `ModelMetadata.from_raw()` normalizes diverse upstream formats into consistent structure
- Supports `database_id` parameter for LiteLLM models to separate display name (`id`) from database UUID (`database_id`)
  - Display name used in UI (e.g., "ollama/qwen3:8b")
  - Database UUID used for deletion operations (e.g., "3dbd2639-ccf1-4628-86ff-60a8e9d93fce")
- Extraction functions (`_extract_numeric`, `_extract_text`, `_extract_capabilities`) search multiple nested sections (metadata, details, model_info, summary)
- `litellm_fields` property maps normalized metadata to LiteLLM-compatible fields including:
  - Context window and token limits
  - Capabilities → supports_* boolean fields (vision, function_calling, etc.)
  - Default pricing based on GPT-4/Whisper/DALL-E 3 rates
  - Ollama parameters → OpenAI-compatible parameters mapping
- Always sets `litellm_provider: "ollama"` for Ollama models

**Synchronization** (`backend/provider_sync.py`, `backend/sync_worker.py`)
- `sync_provider()` handles fetch + DB upsert + optional LiteLLM push. Uses `_clean_ollama_payload` for heavy models and honors `push_to_litellm` flag.
- `sync_worker.py` schedules periodic syncs using provider defaults (`sync_enabled` flag + interval from config).
- Manual UI endpoints call into the same `sync_provider` with explicit fetch/push/sync semantics.

**LiteLLM Integration** (`backend/litellm_client.py`)
- Model registration: `POST /model/new` with `{model_name, litellm_params, model_info}`
  - `litellm_params`: Connection configuration (model, api_base)
  - `model_info`: Metadata fields (max_tokens, capabilities, pricing, etc.)
- Model deletion: `POST /model/delete` with `{id: <database_uuid>}`
- Model listing: `GET /model/info` returns complete model data including database UUIDs

**Web Layer** (`frontend/api.py` + `frontend/templates/`)
- FastAPI UI surfaced on `:4001` via Docker.
- Database initialization in lifespan context manager (uses `shared/database.init_session_maker` + migrations).
- Provider/model routes wrap `backend.provider_sync` for fetch/push/sync actions and expose per-provider + global buttons in `/sources`.
- Admin page uses modal dialogs for add/edit provider.

**Provider Management API:**
  - `GET /api/providers` - List all providers from database
  - `POST /api/providers/fetch-all` - Fetch enabled providers into DB (no LiteLLM push)
  - `POST /api/providers/sync-all` - Fetch + push enabled providers
  - `POST /api/providers/{id}/fetch-now` - Fetch one provider into DB
  - `POST /api/providers/{id}/sync-now` - Fetch + push one provider
  - `POST /api/providers/{id}/push` - Push one provider's sync-enabled, non-orphaned models
  - `POST /admin/providers` - Create provider (with prefix, default_ollama_mode)
  - `POST /admin/providers/{id}` - Update provider
  - `DELETE /admin/providers/{id}` - Delete provider (cascades to models)

**Model Management API:**
  - `GET /api/providers/{id}/models` - Get models for provider (with orphan filtering)
  - `GET /api/models/db/{id}` - Get specific model by database ID
  - `POST /api/models/db/{id}/params` - Update model user parameters
  - `DELETE /api/models/db/{id}/params` - Reset to provider defaults
  - `POST /api/models/db/{id}/refresh` - Refresh single model from provider
  - `POST /api/models/db/{id}/push` - Push single model to LiteLLM with effective params
  - `POST /api/models/push-all` - Push all non-orphaned models to LiteLLM

**Legacy/Compatibility API:**
  - `POST /sync` - Manual sync trigger (uses database session)
  - `/models/show?source=X&model=Y` - Fetch Ollama model details on demand
  - `/api/sources`, `/api/models` - JSON APIs (SyncState-based)

> The flows below describe the legacy `litellm_updater` entrypoint. The Docker services now route through `backend/provider_sync.py` and `frontend/api.py`, but the database behaviors (upserts, orphan handling, effective params) remain the same.

### Key Data Flow

**Initial Setup:**
1. User adds providers in `/admin` (stored in database)

**Synchronization Flow:**
1. Scheduler (or manual `/sync` trigger) calls `sync_once()` with database session
2. For each provider:
   - `fetch_source_models()` retrieves raw model list from provider
   - Each raw model is normalized via `ModelMetadata.from_raw()`
   - `upsert_model()` creates or updates model in database
   - User-edited parameters (`user_params`) are preserved during update
   - Models not in fetch are marked as `is_orphaned = True`
   - If LiteLLM configured, models are POSTed to `/model/new`
3. Results also stored in `SyncState` for backward compatibility

**Model Management Flow:**
1. User views providers/models at `/sources` (loads from database via API)
2. Orphaned models displayed in RED, modified models in BLUE
3. Per-model actions:
   - **Refresh**: Fetches latest data from provider, updates database with `full_update=True`
   - **Edit Params**: Updates `user_params` (preserved across syncs), sets `user_modified=True`
   - **Push to LiteLLM**: Sends single model with `effective_params` and proper tags
4. Bulk actions:
   - **Push All to LiteLLM**: Pushes all non-orphaned models with tags (`lupdater`, `provider:*`, `type:*`)
   - **Sync All Providers**: Fetches models from all providers, updates database with `full_update=False`
5. LiteLLM page at `/litellm` shows models with tag filtering:
   - Click tag buttons to filter models by tags (OR logic for multiple tags)
   - Tags include: `lupdater`, `provider:<name>`, `type:<ollama|litellm>`

**Database Schema:**
- **Providers**: id, name, base_url, type, prefix, default_ollama_mode, api_key
- **Models**: id, provider_id, model_id, litellm_params, user_params, is_orphaned, user_modified, first_seen, last_seen

### Important Patterns

**Database Operations**
- Use `Depends(get_session)` for FastAPI dependency injection
- Sessions auto-commit on success, auto-rollback on exceptions
- CRUD functions handle model relationships (e.g., `selectinload(Model.provider)`)
- JSON fields: Use `_dict` properties (e.g., `model.litellm_params_dict`) for parsed data
- Orphan detection: Compare active_model_ids set with existing database models
- User edits: `user_modified` flag prevents automatic overwrites during sync

**Prefix Application and Model Naming**
- Store original model name in `model_id` field (NO prefix) in database
- Apply prefix only for display: `display_name = f"{prefix}/{model_id}"`
- LiteLLM registration uses two different names:
  - `model_name`: Display name with prefix (e.g., `mks-ollama/qwen3:8b`) - shown in LiteLLM UI
  - `litellm_params.model`: Connection string with provider prefix (e.g., `ollama/qwen3:8b` or `openai/qwen3:8b`)
- Example flow for model `qwen3:8b` with provider prefix `mks-ollama`:
  - Database `model_id`: `qwen3:8b` (original name)
  - Display `model_name`: `mks-ollama/qwen3:8b` (prefix for UI)
  - `litellm_params.model`: `ollama/qwen3:8b` or `openai/qwen3:8b` (based on mode)

**Effective Parameters**
- `litellm_params`: Auto-updated from provider during sync
- `user_params`: Manual edits via UI, preserved during sync
- `effective_params`: Returns `user_params` if set, else `litellm_params`
- Always use `effective_params` when pushing to LiteLLM

**Ollama Mode Configuration**
- Provider-level: `default_ollama_mode` (applies to all models)
- Model-level: `ollama_mode` (overrides provider default)
- Effective mode: `model.ollama_mode or provider.default_ollama_mode or "ollama"`
- Values: "ollama" (native format) or "openai" (OpenAI-compatible format)

**LiteLLM Integration Pattern**

When pushing models to LiteLLM, the payload is constructed based on provider type and Ollama mode:

**For LiteLLM/OpenAI-compatible providers (`provider.type == "litellm"`):**
```python
litellm_params = {
    "model": f"openai/{model.model_id}",  # e.g., "openai/gpt-4"
    "api_base": provider.base_url,        # e.g., "http://localai:8080"
    "tags": ["lupdater", f"provider:{provider.name}", f"type:{provider.type}"]
}
model_info = {
    "litellm_provider": "openai",
    # ... other metadata from effective_params ...
}
```

**For Ollama providers in native mode (`provider.type == "ollama"` and `mode == "ollama"`):**
```python
litellm_params = {
    "model": f"ollama/{model.model_id}",  # e.g., "ollama/qwen3:8b"
    "api_base": provider.base_url,        # e.g., "http://ollama:11434"
    "tags": ["lupdater", f"provider:{provider.name}", "type:ollama"]
}
model_info = {
    "litellm_provider": "ollama",
    "mode": "ollama",
    # ... other metadata from effective_params ...
}
```

**For Ollama providers in OpenAI mode (`provider.type == "ollama"` and `mode == "openai"`):**
```python
api_base = provider.base_url.rstrip("/") + "/v1"  # Add /v1 endpoint
litellm_params = {
    "model": f"openai/{model.model_id}",  # e.g., "openai/qwen3:8b"
    "api_base": api_base,                 # e.g., "http://ollama:11434/v1"
    "tags": ["lupdater", f"provider:{provider.name}", "type:ollama"]
}
model_info = {
    "litellm_provider": "openai",
    "mode": "openai",
    # ... other metadata from effective_params ...
}
```

**Key Points:**
- `litellm_params` contains connection configuration (model, api_base, tags)
- `model_info` contains metadata (capabilities, limits, pricing, litellm_provider)
- Tags are always placed inside `litellm_params`, not at the top level
- OpenAI mode for Ollama uses the `/v1` endpoint suffix
- The `model` field in `litellm_params` always uses the original `model_id` without display prefix
- The `model_name` in the payload uses the display prefix (e.g., `mks-ollama/qwen3:8b`)

**Error Handling**
- Network errors (httpx.RequestError) are logged but don't crash the sync
- HTTP errors (httpx.HTTPStatusError) are logged with response text
- Validation errors use Pydantic's ValidationError
- Config errors raise RuntimeError with descriptive messages
- Database errors rollback transaction automatically

**Ollama Payload Cleaning**
- The `/api/show` endpoint returns very large responses (tensors, full modelfile)
- Always use `_clean_ollama_payload()` before storing/caching Ollama responses
- Cleaned payload is used in `ModelDetailsCache` and returned by `/models/show`

**URL Normalization**
- All URLs stored as Pydantic `HttpUrl` type
- Use `normalized_base_url` property to get string without trailing slash for path joining
- Don't manually strip slashes; use the property

**Thread Safety**
- `SyncState` and `ModelDetailsCache` use asyncio locks (`asyncio.Lock()`)
- Always use `async with self._lock` pattern when accessing/modifying shared state
- Database sessions are async-safe via SQLAlchemy async engine

## Configuration Notes

**NEW: Providers are now in the database!**

The `data/config.json` schema (reduced):
```json
{
  "litellm": {"base_url": "http://localhost:4000", "api_key": null},
  "sources": [],
  "sync_interval_seconds": 300
}
```

- `sync_interval_seconds`: 0 = disabled, minimum 30 when enabled
- `litellm.base_url`: Can be null to disable LiteLLM registration (still fetches models)
- **`sources` array is legacy** - providers are now managed in database

**Database Schema (`data/models.db`):**

Providers table:
```sql
CREATE TABLE providers (
    id INTEGER PRIMARY KEY,
    name VARCHAR UNIQUE NOT NULL,
    base_url VARCHAR NOT NULL,
    type VARCHAR NOT NULL,  -- 'ollama' or 'litellm'
    api_key VARCHAR,
    prefix VARCHAR,  -- e.g., 'mks-ollama'
    default_ollama_mode VARCHAR,  -- 'ollama' or 'openai'
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL
);
```

Models table:
```sql
CREATE TABLE models (
    id INTEGER PRIMARY KEY,
    provider_id INTEGER NOT NULL REFERENCES providers(id) ON DELETE CASCADE,
    model_id VARCHAR NOT NULL,  -- Original name WITHOUT prefix
    model_type VARCHAR,
    context_window INTEGER,
    max_input_tokens INTEGER,
    max_output_tokens INTEGER,
    max_tokens INTEGER,
    capabilities TEXT,  -- JSON array
    litellm_params TEXT NOT NULL,  -- JSON object (provider defaults)
    raw_metadata TEXT NOT NULL,  -- JSON object (full raw response)
    user_params TEXT,  -- JSON object (user edits)
    ollama_mode VARCHAR,  -- Per-model override
    first_seen DATETIME NOT NULL,
    last_seen DATETIME NOT NULL,
    is_orphaned BOOLEAN NOT NULL DEFAULT FALSE,
    orphaned_at DATETIME,
    user_modified BOOLEAN NOT NULL DEFAULT FALSE,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    UNIQUE(provider_id, model_id)
);
```

## Provider Management

### Adding New Providers

**Via Admin UI:**
1. Go to `/admin`
2. Click "Add Provider" to open the modal
3. Fill in: Name, Base URL, Type, Optional: API Key, Prefix, Ollama Mode
4. Save to create the provider

**Via API:**
```bash
curl -X POST http://localhost:8000/admin/providers \
  -F "name=my-ollama" \
  -F "base_url=http://localhost:11434" \
  -F "type=ollama" \
  -F "prefix=local" \
  -F "default_ollama_mode=ollama"
```

### Managing Models

**Refresh Single Model:**
```bash
curl -X POST http://localhost:8000/api/models/db/123/refresh
```

**Edit Model Parameters:**
```bash
curl -X POST http://localhost:8000/api/models/db/123/params \
  -H "Content-Type: application/json" \
  -d '{"max_tokens": 4096, "temperature": 0.7}'
```

**Push to LiteLLM:**
```bash
curl -X POST http://localhost:8000/api/models/db/123/push
```

**Reset to Defaults:**
```bash
curl -X DELETE http://localhost:8000/api/models/db/123/params
```

## Testing Strategy

**Unit Tests:**
- `tests/test_ollama_payload_cleaning.py` - Ollama payload cleaning
- `tests/test_model_details_cache.py` - Model details caching

**Integration Tests:**
- `tests/test_sources_integration.py` - Live endpoint tests (requires `.env` configuration)
- Uses `pytest-asyncio` for async test support
- Tests skip when endpoints not configured (graceful degradation)

**Database Testing:**
- All new database functionality has been manually tested
- Tested: Provider CRUD, model persistence, orphan detection
- See commit history for test results

**Manual Testing Workflow:**
```bash
# 1. Install dependencies
pip install -e .

# 2. Run unit tests
pytest tests/test_model_details_cache.py tests/test_ollama_payload_cleaning.py -v

# 3. Test API endpoints
curl http://localhost:8000/api/providers
curl http://localhost:8000/api/providers/1/models

# 4. Test model management
# Use UI at /sources to refresh, edit, and push models
```

## Recent Changes & Gotchas
- Ollama `/api/tags` responses that return a bare list (instead of `{ "models": [...] }`) are now parsed correctly; this fixes empty syncs from some servers.
- `mode:*` tags are only generated for Ollama providers. OpenAI/compat providers should no longer get `mode:ollama` attached to their models.
- Duplicate detection when pushing to LiteLLM now reads tags from both `litellm_params` and `model_info`, so older LiteLLM entries without top-level tags are still de-duped.
- The Providers page uses a **Fetch** button that runs even if sync is disabled for that provider; the sync flag only controls scheduled syncs.
- On the Admin page, adding and editing providers happen in modals; the inline add form is gone.
