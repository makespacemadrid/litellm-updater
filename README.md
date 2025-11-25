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
   litellm-updater
   # or
   uvicorn litellm_updater.web:create_app
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
  docker run --rm -p 8000:8000 -v $(pwd)/data:/app/data litellm-updater
  ```

- Or use Docker Compose with the provided `example.env` (copy or override values as needed):
  ```bash
  cp example.env .env
  docker-compose --env-file .env up --build
  ```
  The compose file binds the UI to `${PORT:-8000}` and mounts the local `data/` directory so configuration persists across restarts.

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

