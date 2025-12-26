# litellm-companion

A FastAPI service that synchronizes models from Ollama or other LiteLLM/OpenAI-compatible servers into a LiteLLM proxy. It periodically scans upstream sources for models, persists them to a database, and registers them with LiteLLM using the admin API. Includes a web UI for provider management, model editing, and monitoring.

## Features
- **Database-driven provider management** with SQLite persistence
- **Provider prefixes** (e.g., `mks-ollama/qwen3:8b`) for namespace organization
- **Ollama mode configuration** (native ollama format vs OpenAI-compatible)
- **Model parameter editing** with user override preservation across syncs
- **Orphaned model detection** - highlights models no longer available in source
- **Per-model actions**: Refresh from source, Push to LiteLLM, Edit parameters
- **Full OpenAI API compatibility** - 30+ OpenAI parameters supported across 85+ models
- **Multi-capability models** - Vision (19), Function calling (34), Embeddings (9), Audio (4)
- Configurable sources (Ollama or LiteLLM/OpenAI compatible)
- Periodic sync job that registers upstream models with LiteLLM
- Manual sync trigger from the UI
- Web UI for browsing models with database persistence

> **📊 Model Statistics**: See [MODEL_STATISTICS.md](MODEL_STATISTICS.md) for detailed statistics on available models and OpenAI API coverage.

## Getting started
1. **Install dependencies**
   ```bash
   pip install -e .
   ```

2. **Run the server**
   ```bash
   PORT=8000 litellm-companion
   # or
   PORT=8000 uvicorn frontend.api:create_app --factory --port $PORT
   ```

   The server defaults to `http://0.0.0.0:8000`.

3. **Configure providers and LiteLLM destination**
   - Navigate to `http://localhost:8000/admin` to set the LiteLLM base URL, update the sync interval, or manage providers.
   - **NEW**: If you have existing sources in `config.json`, use the "Migrate from config.json" button to move them to the database.
   - Add new providers with optional prefix and Ollama mode configuration.
   - A default `data/config.json` is generated on first run with automatic sync disabled and LiteLLM destination at `http://localhost:4000`.

4. **Trigger sync**
   - The scheduler runs automatically only when the interval is greater than zero.
   - Use the "Run sync now" button on the overview or models page to trigger a manual sync.

## Running with Docker
- Build the image directly:
  ```bash
  docker build -t litellm-companion .
  docker run --rm -e PORT=8000 -p 8000:8000 -v $(pwd)/data:/app/data litellm-companion
  ```

- Or use Docker Compose with the provided `example.env` (copy or override values as needed):
  ```bash
  cp example.env .env
  docker-compose --env-file .env up --build
  ```
  The compose file binds the UI to `${PORT:-8000}` for both the host and container, and mounts the local `data/` directory so configuration persists across restarts.
  An `env-sync` helper service runs before the app to append any new variables from the container's `env.example` into your local `.env` without overwriting existing values.
  The stack also includes a `watchtower` container that checks for image updates every `${WATCHTOWER_POLL_INTERVAL:-60}` seconds (configurable via `.env`) and only acts on services labeled for updates.

## Running integration tests
- Create a virtual environment if you do not already have one:
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  ```
- Install dev dependencies so `pytest-asyncio` is available:
  ```bash
  pip install -e ".[dev]"
  ```
- Copy `tests/example.env` to `tests/.env` and point the values at live, reachable endpoints (URLs must include the scheme):
  ```bash
  cp tests/example.env tests/.env
  # edit tests/.env to include TEST_OLLAMA_URL / TEST_OPENAI_URL and optional *_KEY values
  ```
- Run the integration suite; it will automatically load `tests/.env` and skip live checks if no endpoints are configured:
  ```bash
  pytest tests/test_sources_integration.py -q
  ```

## Configuration

**Database (`data/models.db`):**
- Providers (formerly "sources") are now stored in a SQLite database
- Model metadata, user edits, and orphan status are persisted
- Use `/admin` UI to manage providers or migrate from config.json

**Config file (`data/config.json`):**
- Now only contains LiteLLM destination and sync interval
- Providers are managed through the database (not config.json)
```json
{
  "litellm": {"base_url": "http://localhost:4000", "api_key": null},
  "sources": [],  // Legacy - use database instead
  "sync_interval_seconds": 0
}
```

## Database Schema

**Providers table:**
- id, name, base_url, type, api_key, **prefix**, **default_ollama_mode**

**Models table:**
- id, provider_id, model_id, litellm_params, **user_params**, **is_orphaned**, **user_modified**
- Tracks first_seen, last_seen, orphaned_at timestamps
- JSON fields for capabilities, raw_metadata

## Notes
- **LiteLLM registration** is performed via `/model/new` endpoint with model name discovered from upstream source
- **Prefixes are applied to display names** (e.g., `mks-ollama/qwen3:8b`) but not to internal model paths
- **User-edited parameters are preserved** across syncs (stored in `user_params` field)
- **Orphaned models** (no longer in provider) are highlighted in red in the UI
- **Database migrations** are handled by Alembic (auto-run on startup)

## Documentation

- [MODEL_STATISTICS.md](MODEL_STATISTICS.md) - Model statistics and OpenAI API coverage
- [MIGRATION.md](MIGRATION.md) - Migration guide from previous versions
- [CLAUDE.md](CLAUDE.md) - Development guide for contributors
