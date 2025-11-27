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

**Model Normalization** (`models.py`)
- `ModelMetadata.from_raw()` normalizes diverse upstream formats into consistent structure
- Extraction functions (`_extract_numeric`, `_extract_text`, `_extract_capabilities`) search multiple nested sections (metadata, details, model_info, summary)
- `litellm_fields` property maps normalized metadata to LiteLLM-compatible fields including:
  - Context window and token limits
  - Capabilities → supports_* boolean fields (vision, function_calling, etc.)
  - Default pricing based on GPT-4/Whisper/DALL-E 3 rates
  - Ollama parameters → OpenAI-compatible parameters mapping
- Always sets `litellm_provider: "ollama"` for Ollama models

**Synchronization** (`sync.py`)
- `sync_once()` fetches models from all sources and registers them with LiteLLM via `/router/model/add`
- `start_scheduler()` runs sync in a loop at configured interval (disabled if interval ≤ 0)
- Registration only happens when LiteLLM is configured; fetching always occurs
- Errors are logged but don't stop sync for other sources/models

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
5. If LiteLLM configured, each model is POSTed to `/router/model/add`
6. Results are stored in `SyncState` for UI display
7. UI can fetch extended Ollama details via `/models/show` which uses `ModelDetailsCache`

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
