# litellm-updater

A small FastAPI service for synchronizing models from Ollama or other LiteLLM/OpenAI-compatible servers into a LiteLLM proxy. It periodically scans upstream sources for models, registers them with LiteLLM using the admin API, and provides a minimal web UI for monitoring and configuration.

## Features
- Configurable sources (Ollama or LiteLLM/OpenAI compatible)
- Periodic sync job that registers upstream models with LiteLLM
- Manual sync trigger from the UI
- Admin view for managing sources, LiteLLM target, and sync interval
- Simple dashboard to browse models fetched from each source

## Getting started
1. **Install dependencies**
   ```bash
   pip install -e .
   ```

2. **Run the server**
   ```bash
   PORT=8000 litellm-updater
   # or
   PORT=8000 uvicorn litellm_updater.web:create_app --port $PORT
   ```

   The server defaults to `http://0.0.0.0:8000`.

3. **Configure sources and LiteLLM target**
   - Navigate to `http://localhost:8000/admin` to set the LiteLLM base URL, update the sync interval, or add/remove sources.
   - A default `data/config.json` is generated on first run with an empty provider list, automatic sync disabled, and a LiteLLM target at `http://localhost:4000`.

4. **Trigger sync**
   - The scheduler runs automatically only when the interval is greater than zero.
   - Use the "Run sync now" button on the overview or models page to trigger a manual sync.

## Running with Docker
- Build the image directly:
  ```bash
  docker build -t litellm-updater .
  docker run --rm -e PORT=8000 -p 8000:8000 -v $(pwd)/data:/app/data litellm-updater
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
Configuration is stored in `data/config.json` and uses the following shape:
```json
{
  "litellm": {"base_url": "http://localhost:4000", "api_key": null},
  "sources": [],
  "sync_interval_seconds": 0
}
```

## Notes
- LiteLLM registration is performed via the `/router/model/add` endpoint with the model name discovered from the upstream source.
- The service uses in-memory state for recent sync results; persistent history is not yet tracked.
