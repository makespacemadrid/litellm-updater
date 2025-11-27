# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LiteLLM Updater is a FastAPI service that synchronizes models from Ollama or other LiteLLM/OpenAI-compatible servers into a LiteLLM proxy. It periodically fetches models from upstream sources and registers them with LiteLLM's admin API.

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
# Build and run directly
docker build -t litellm-updater .
docker run --rm -e PORT=8000 -p 8000:8000 -v $(pwd)/data:/app/data litellm-updater

# Using docker-compose
cp example.env .env
docker-compose --env-file .env up --build
```

### Linting
```bash
# Run ruff for linting/formatting
ruff check .
ruff format .
```

## Deployment & Live Testing

### Detecting Docker Deployment

**IMPORTANT**: If a `.env` file exists in the project root directory, the service is running in Docker Compose. This means:
- Changes to Python code require rebuilding the Docker image
- The service is accessible at the configured PORT (check `docker-compose.yml` or `.env`)
- Data is persisted in `./data` volume mount
- Live testing can be performed against the running instance

### Making Code Changes in Docker Deployment

When the service is running via Docker Compose (`.env` exists):

```bash
# 1. Make your code changes to Python files

# 2. REBUILD the Docker image (required for code changes to take effect)
docker compose build litellm-updater

# 3. Restart the service with the new image
docker compose restart litellm-updater

# 4. Verify the service is running
docker compose logs --tail=20 litellm-updater

# Alternative: Build and restart in one command
docker compose up --build -d litellm-updater
```

**Why rebuild is necessary**: The Docker image copies the Python code during build. Simply restarting won't pick up code changes - you must rebuild the image first.

### Live Testing

When deployed via Docker Compose, the service is typically accessible at:
- Web UI: `http://localhost:<PORT>` (check `.env` or `docker-compose.yml` for PORT)
- Common routes for testing:
  - `/` - Dashboard
  - `/sources` - View source models
  - `/litellm` - View LiteLLM destination models
  - `/admin` - Configure sources and sync

Example workflow for testing a fix:
```bash
# 1. Make code changes
# 2. Rebuild and restart
docker compose build litellm-updater && docker compose restart litellm-updater

# 3. Test in browser or via curl
curl http://localhost:8005/

# 4. Check logs for errors
docker compose logs -f litellm-updater
```

### Checking Service Status

```bash
# View running containers
docker compose ps

# Follow logs in real-time
docker compose logs -f litellm-updater

# Check last N log lines
docker compose logs --tail=50 litellm-updater

# Restart all services
docker compose restart

# Stop all services
docker compose down
```

## Architecture

### Core Components

**Configuration Layer** (`config.py`)
- Manages `data/config.json` containing LiteLLM destination, sources, and sync interval
- Default config is auto-created on first run with empty sources, sync disabled, and LiteLLM at `http://localhost:4000`
- Uses Pydantic validation with `AppConfig` model

**Source Fetchers** (`sources.py`)
- Supports two source types: Ollama (`/api/tags`) and LiteLLM/OpenAI (`/v1/models`)
- `fetch_source_models()` dispatches to the correct fetcher based on `SourceType`
- Ollama fetcher includes `_clean_ollama_payload()` to strip large/redundant fields (tensors, modelfile, license)
- For LiteLLM sources, fetches list then individual model details from `/v1/models/{id}`
- `fetch_litellm_target_models()` uses LiteLLM's `/model/info` endpoint which includes model UUIDs and complete metadata

**Model Normalization** (`models.py`)
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

**Synchronization** (`sync.py`)
- `sync_once()` fetches models from all sources and registers them with LiteLLM via `/model/new`
- `start_scheduler()` runs sync in a loop at configured interval (disabled if interval ≤ 0)
- Registration only happens when LiteLLM is configured; fetching always occurs
- Errors are logged but don't stop sync for other sources/models

**LiteLLM Integration** (`web.py`)
- Model registration: `POST /model/new` with `{model_name, litellm_params, model_info}`
  - `litellm_params`: Connection configuration (model, api_base)
  - `model_info`: Metadata fields (max_tokens, capabilities, pricing, etc.)
- Model deletion: `POST /model/delete` with `{id: <database_uuid>}`
- Model listing: `GET /model/info` returns complete model data including database UUIDs

**Web Layer** (`web.py`)
- FastAPI app with Jinja2 templates and static files
- `SyncState` class maintains in-memory cache of last sync results (not persisted)
- `ModelDetailsCache` provides TTL-based caching (600s default) for Ollama `/api/show` calls
- Lifespan context manager handles scheduler startup/shutdown
- Routes:
  - `/` - Overview dashboard
  - `/sources` - Browse models by source
  - `/models` - JSON API endpoint (redirects HTML requests to `/sources`)
  - `/models/show?source=X&model=Y` - Fetch Ollama model details on demand
  - `/admin` - Configure sources, LiteLLM target, sync interval
  - `/litellm` - View models in LiteLLM destination
  - `/sync` (POST) - Manual sync trigger
  - `/sources/refresh` (POST) - Refresh single source
  - `/api/sources`, `/api/models` - JSON APIs

### Key Data Flow

1. User configures sources and LiteLLM destination in `/admin`
2. Scheduler (or manual trigger) calls `sync_once()`
3. For each source, `fetch_source_models()` retrieves raw model list
4. Each raw model is normalized via `ModelMetadata.from_raw()`
5. If LiteLLM configured, each model is POSTed to `/model/new` with separated `litellm_params` and `model_info`
6. Results are stored in `SyncState` for UI display
7. UI can fetch extended Ollama details via `/models/show` which uses `ModelDetailsCache`
8. `/litellm` page fetches models via `/model/info` endpoint to get database UUIDs for deletion
9. Delete operations use database UUID from `model_info.id`, not the model name

### Important Patterns

**Error Handling**
- Network errors (httpx.RequestError) are logged but don't crash the sync
- HTTP errors (httpx.HTTPStatusError) are logged with response text
- Validation errors use Pydantic's ValidationError
- Config errors raise RuntimeError with descriptive messages

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

## Configuration Notes

The `data/config.json` schema:
```json
{
  "litellm": {"base_url": "http://localhost:4000", "api_key": null},
  "sources": [
    {"name": "MyOllama", "base_url": "http://localhost:11434", "type": "ollama", "api_key": null}
  ],
  "sync_interval_seconds": 300
}
```

- `sync_interval_seconds`: 0 = disabled, minimum 30 when enabled (validated in `AppConfig`)
- `litellm.base_url`: Can be null to disable LiteLLM registration (still fetches models)
- Sources are added/removed via admin UI or API calls

## Testing Strategy

- Integration tests in `tests/test_sources_integration.py` require live endpoints
- Uses `pytest-asyncio` for async test support
- Tests skip when endpoints not configured (graceful degradation)
- Unit tests for payload cleaning in `tests/test_ollama_payload_cleaning.py`
