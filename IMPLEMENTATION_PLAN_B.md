# Implementation Plan B: Complete Separation with Local DB

**Approach:** Separate backend sync service and frontend API/UI service, keeping local SQLite database.

---

## 🎯 Architecture Overview

```
┌─────────────────────────────────────┐
│   BACKEND (Sync Worker)             │
│   - Standalone Python service       │
│   - Periodic sync scheduler          │
│   - Provider fetching                │
│   - Database writes                  │
│   - LiteLLM API integration          │
│   - NO HTTP server                   │
│   - NO templates/static files        │
└─────────────────────────────────────┘
         ↓
         Writes to
         ↓
┌─────────────────────────────────────┐
│   SQLite Database (Shared Volume)   │
│   - providers                        │
│   - models                           │
│   - config                           │
│   - (mounted at /app/data)          │
└─────────────────────────────────────┘
         ↑
         Reads from
         ↑
┌─────────────────────────────────────┐
│   FRONTEND (API + UI)               │
│   - FastAPI HTTP server              │
│   - REST API endpoints              │
│   - Jinja2 templates                │
│   - Static files (CSS/JS)           │
│   - Database reads (mostly)         │
│   - Manual sync triggers            │
│   - NO automatic sync loop          │
└─────────────────────────────────────┘
```

---

## 📂 New Project Structure

```
litellm-updater/
├── backend/                       # NEW: Sync worker service
│   ├── __init__.py
│   ├── sync_worker.py            # Main sync loop
│   ├── provider_sync.py          # Provider-specific sync logic
│   ├── litellm_client.py         # LiteLLM API client
│   ├── Dockerfile                # Lightweight backend image
│   └── requirements.txt          # Minimal deps (no FastAPI)
│
├── frontend/                      # NEW: API + UI service
│   ├── __init__.py
│   ├── api.py                    # FastAPI app (no sync logic)
│   ├── routes/
│   │   ├── providers.py          # Provider management
│   │   ├── models.py             # Model management
│   │   ├── admin.py              # Config endpoints
│   │   └── diagnostics.py        # Health/status
│   ├── templates/                # Moved from litellm_updater/
│   ├── static/                   # Moved from litellm_updater/
│   ├── Dockerfile                # Frontend image
│   └── requirements.txt          # FastAPI, Jinja2, etc.
│
├── shared/                        # NEW: Shared code
│   ├── __init__.py
│   ├── database.py               # DB engine, session maker
│   ├── db_models.py              # SQLAlchemy models
│   ├── crud.py                   # CRUD operations
│   ├── models.py                 # Pydantic models
│   ├── sources.py                # Provider fetching
│   ├── tags.py                   # Tag generation
│   └── config.py                 # Config helpers
│
├── alembic/                       # Keep migrations
├── docker-compose.yml             # Updated for 2 services
├── pyproject.toml                 # Updated structure
└── README.md
```

---

## 🔧 Component Breakdown

### Backend Service (Sync Worker)

**File: `backend/sync_worker.py`**

**Purpose:** Autonomous service that runs sync loop indefinitely.

**Key Features:**
- ✅ Runs as background daemon
- ✅ Reads config from shared database
- ✅ Writes models to shared database
- ✅ Pushes to LiteLLM API
- ✅ Graceful shutdown (SIGTERM)
- ✅ Health status in database

**Core Logic:**
```python
import asyncio
import logging
import signal
from datetime import datetime, UTC

from shared.database import create_engine, init_session_maker
from shared.crud import get_config, get_all_providers, upsert_model
from shared.sources import fetch_source_models
from backend.litellm_client import push_models_to_litellm

logger = logging.getLogger(__name__)

class SyncWorker:
    def __init__(self):
        self.running = True
        self.engine = create_engine()
        self.session_maker = init_session_maker(self.engine)

    async def run(self):
        """Main sync loop."""
        logger.info("Sync worker started")

        # Setup graceful shutdown
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        signal.signal(signal.SIGINT, self.handle_shutdown)

        while self.running:
            try:
                async with self.session_maker() as session:
                    config = await get_config(session)

                    # Check if sync is enabled
                    if config.sync_interval_seconds == 0:
                        logger.info("Sync disabled (interval=0), sleeping...")
                        await asyncio.sleep(60)
                        continue

                    # Sync all enabled providers
                    await self.sync_all_providers(session, config)

                    # Wait for next interval
                    await asyncio.sleep(config.sync_interval_seconds)

            except Exception as e:
                logger.exception("Error in sync loop: %s", e)
                await asyncio.sleep(60)  # Retry after error

        logger.info("Sync worker stopped")

    async def sync_all_providers(self, session, config):
        """Sync models from all enabled providers."""
        providers = await get_all_providers(session)

        for provider in providers:
            if not provider.sync_enabled:
                logger.debug("Skipping disabled provider: %s", provider.name)
                continue

            try:
                logger.info("Syncing provider: %s", provider.name)

                # Fetch models from provider
                models = await fetch_source_models(
                    provider.name,
                    provider.base_url,
                    provider.type,
                    provider.api_key
                )

                # Update database
                active_model_ids = set()
                for model in models:
                    await upsert_model(session, provider, model)
                    active_model_ids.add(model.id)

                # Mark orphans
                await mark_orphaned_models(session, provider.id, active_model_ids)

                # Push to LiteLLM if configured
                if config.litellm_base_url:
                    await push_models_to_litellm(
                        session,
                        config.litellm_base_url,
                        config.litellm_api_key,
                        provider
                    )

                await session.commit()
                logger.info("Synced provider %s: %d models", provider.name, len(models))

            except Exception as e:
                logger.error("Failed to sync provider %s: %s", provider.name, e)
                await session.rollback()

    def handle_shutdown(self, signum, frame):
        """Handle graceful shutdown."""
        logger.info("Received signal %d, shutting down...", signum)
        self.running = False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    worker = SyncWorker()
    asyncio.run(worker.run())
```

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install only backend dependencies (no FastAPI, Jinja2)
COPY backend/requirements.txt /app/backend/
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy shared code
COPY shared/ /app/shared/
COPY alembic/ /app/alembic/
COPY alembic.ini /app/

# Copy backend code
COPY backend/ /app/backend/

# Run sync worker
CMD ["python", "-m", "backend.sync_worker"]
```

**requirements.txt:**
```
sqlalchemy[asyncio]>=2.0.0
aiosqlite>=0.19.0
httpx>=0.27.0
pydantic>=2.7.0
```

**Estimated Memory:** 80-150 MB

---

### Frontend Service (API + UI)

**File: `frontend/api.py`**

**Purpose:** HTTP server for user interface and API access.

**Key Features:**
- ✅ FastAPI REST API
- ✅ Jinja2 HTML templates
- ✅ Static file serving
- ✅ Database reads (mostly)
- ✅ Trigger manual syncs (writes flag to DB)
- ❌ NO sync loop
- ❌ NO provider fetching

**Core Structure:**
```python
from fastapi import FastAPI, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from shared.database import create_engine, init_session_maker, get_session
from frontend.routes import providers, models, admin, diagnostics

def create_app() -> FastAPI:
    app = FastAPI(title="LiteLLM Updater")

    # Initialize database (no migrations here, backend handles it)
    engine = create_engine()
    session_maker = init_session_maker(engine)

    # Mount static files
    app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

    # Include routers
    app.include_router(providers.router, prefix="/api/providers", tags=["providers"])
    app.include_router(models.router, prefix="/api/models", tags=["models"])
    app.include_router(admin.router, prefix="/api/admin", tags=["admin"])
    app.include_router(diagnostics.router, prefix="/api/diagnostics", tags=["diagnostics"])

    # HTML routes
    templates = Jinja2Templates(directory="frontend/templates")

    @app.get("/")
    async def index(request: Request, session: AsyncSession = Depends(get_session)):
        providers = await get_all_providers(session)
        return templates.TemplateResponse("index.html", {
            "request": request,
            "providers": providers
        })

    # ... other HTML routes

    return app
```

**Route Example: `frontend/routes/providers.py`**
```python
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from shared.database import get_session
from shared.crud import get_all_providers, create_provider, delete_provider

router = APIRouter()

@router.get("/")
async def list_providers(session: AsyncSession = Depends(get_session)):
    """List all providers."""
    providers = await get_all_providers(session)
    return [p.to_dict() for p in providers]

@router.post("/")
async def add_provider(
    name: str,
    base_url: str,
    type: str,
    session: AsyncSession = Depends(get_session)
):
    """Add new provider."""
    provider = await create_provider(session, name, base_url, type)
    await session.commit()
    return provider.to_dict()

@router.delete("/{provider_id}")
async def remove_provider(provider_id: int, session: AsyncSession = Depends(get_session)):
    """Delete provider."""
    await delete_provider(session, provider_id)
    await session.commit()
    return {"status": "deleted"}

@router.post("/{provider_id}/sync")
async def trigger_sync(provider_id: int, session: AsyncSession = Depends(get_session)):
    """Trigger immediate sync for provider (sets flag for backend)."""
    # Option 1: Set sync_pending flag in database
    # Option 2: Send signal to backend service
    # Option 3: Call backend HTTP endpoint (if we add one)

    provider = await get_provider_by_id(session, provider_id)
    if not provider:
        raise HTTPException(404, "Provider not found")

    # For now, just enable sync and reduce interval temporarily
    provider.sync_enabled = True
    await session.commit()

    return {"status": "sync_requested", "message": "Backend will sync on next cycle"}
```

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install frontend dependencies
COPY frontend/requirements.txt /app/frontend/
RUN pip install --no-cache-dir -r frontend/requirements.txt

# Copy shared code
COPY shared/ /app/shared/

# Copy frontend code
COPY frontend/ /app/frontend/

EXPOSE 8000

CMD ["uvicorn", "frontend.api:create_app", "--host", "0.0.0.0", "--port", "8000"]
```

**requirements.txt:**
```
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
jinja2>=3.1.3
python-multipart>=0.0.9
sqlalchemy[asyncio]>=2.0.0
aiosqlite>=0.19.0
httpx>=0.27.0
pydantic>=2.7.0
```

**Estimated Memory:** 100-150 MB

---

## 🐳 Docker Compose Configuration

**Updated `docker-compose.yml`:**
```yaml
services:
  # Backend sync worker
  litellm-sync:
    build:
      context: .
      dockerfile: backend/Dockerfile
    container_name: litellm-sync
    volumes:
      - ./data:/app/data  # Shared database volume
    environment:
      - DATABASE_PATH=/app/data/models.db
      - LOG_LEVEL=info
    restart: unless-stopped
    networks:
      - litellm
    deploy:
      resources:
        limits:
          memory: 256M
        reservations:
          memory: 128M

  # Frontend API + UI
  litellm-frontend:
    build:
      context: .
      dockerfile: frontend/Dockerfile
    container_name: litellm-frontend
    ports:
      - "${PORT:-4001}:8000"
    volumes:
      - ./data:/app/data  # Shared database volume (read-mostly)
    environment:
      - DATABASE_PATH=/app/data/models.db
      - LOG_LEVEL=info
    restart: unless-stopped
    depends_on:
      - litellm-sync
    networks:
      - litellm
    deploy:
      resources:
        limits:
          memory: 256M
        reservations:
          memory: 128M

  # LiteLLM server (unchanged)
  litellm:
    image: ghcr.io/berriai/litellm:main-stable
    ports:
      - "4000:4000"
    env_file:
      - .env
    depends_on:
      db:
        condition: service_healthy
    networks:
      - litellm

  # PostgreSQL for LiteLLM (unchanged)
  db:
    image: postgres:16
    restart: always
    container_name: litellm_db
    env_file:
      - .env
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -d ${POSTGRES_DB} -U ${POSTGRES_USER}"]
      interval: 1s
      timeout: 5s
      retries: 10
    networks:
      - litellm

networks:
  litellm:

volumes:
  postgres_data:
```

---

## 🔄 Migration Strategy

### Step 1: Code Refactoring

**1.1 Extract Shared Code**
```bash
mkdir -p shared
mv litellm_updater/{database,db_models,crud,models,sources,tags,config}.py shared/
```

**1.2 Create Backend**
```bash
mkdir -p backend
# Create sync_worker.py with logic from web.py lifespan + sync.py
# Create litellm_client.py with push logic
# Create Dockerfile and requirements.txt
```

**1.3 Create Frontend**
```bash
mkdir -p frontend/routes
# Split web.py into api.py + routes
mv litellm_updater/templates frontend/
mv litellm_updater/static frontend/
# Create Dockerfile and requirements.txt
```

### Step 2: Remove Memory Hogs

**2.1 Eliminate SyncState**
```python
# OLD (web.py):
sync_state = SyncState()  # In-memory cache

# NEW (frontend/api.py):
# Read directly from database when needed
@router.get("/api/models")
async def get_models(session: AsyncSession = Depends(get_session)):
    return await get_all_models(session)
```

**2.2 Eliminate ModelDetailsCache**
```python
# OLD (web.py):
model_details_cache = ModelDetailsCache()  # TTL cache

# NEW (backend/provider_sync.py):
# Fetch on-demand during sync only, don't cache
async def fetch_model_details(provider, model_id):
    # Fetch, use immediately, discard
    details = await fetch_ollama_model_details(provider.base_url, model_id)
    return clean_ollama_payload(details)
```

**2.3 Clean Raw Metadata**
```python
# Ensure _clean_ollama_payload() is called before storing
def clean_ollama_payload(payload: dict) -> dict:
    """Remove heavy fields from Ollama payload."""
    cleaned = payload.copy()
    # Remove tensors, modelfile, projector, etc.
    for key in ['model_info', 'projector_info', 'modelfile']:
        cleaned.pop(key, None)
    return cleaned
```

### Step 3: Database Handling

**3.1 SQLite WAL Mode**
```python
# shared/database.py
def create_engine(db_url: str | None = None) -> AsyncEngine:
    if db_url is None:
        db_url = get_database_url()

    # Enable WAL mode for better concurrent access
    connect_args = {}
    if db_url.startswith('sqlite'):
        connect_args = {
            "check_same_thread": False,
            "timeout": 30.0,  # Wait up to 30s for locks
        }

    engine = create_async_engine(
        db_url,
        echo=False,
        future=True,
        connect_args=connect_args
    )

    # Enable WAL mode
    if db_url.startswith('sqlite'):
        @event.listens_for(engine.sync_engine, "connect")
        def set_sqlite_pragma(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.close()

    return engine
```

### Step 4: Testing

**4.1 Build Images**
```bash
docker compose build litellm-sync litellm-frontend
```

**4.2 Test Backend Standalone**
```bash
docker compose up litellm-sync
# Check logs for sync activity
docker compose logs -f litellm-sync
```

**4.3 Test Frontend Standalone**
```bash
docker compose up litellm-frontend
# Access UI at http://localhost:4001
```

**4.4 Integration Test**
```bash
docker compose up -d
# Add provider via UI
# Watch backend logs for sync
# Verify models appear in UI
```

### Step 5: Deployment

**5.1 Stop Old Service**
```bash
docker compose stop litellm-updater
docker compose rm litellm-updater
```

**5.2 Start New Services**
```bash
docker compose up -d litellm-sync litellm-frontend
```

**5.3 Monitor**
```bash
docker stats litellm-sync litellm-frontend
# Verify memory < 300 MB total
```

---

## 📊 Expected Benefits

### Memory Usage

| Component | Before | After | Savings |
|-----------|--------|-------|---------|
| Sync Worker | - | 100-150 MB | - |
| Frontend | - | 100-150 MB | - |
| Monolith | 2.9 GB | - | - |
| **Total** | **2.9 GB** | **~250 MB** | **91%** |

### Other Improvements

- ✅ **Failure Isolation:** Frontend stays up if sync fails
- ✅ **Independent Scaling:** Can run multiple frontends
- ✅ **Clearer Responsibilities:** Sync vs. UI separation
- ✅ **Easier Development:** Can work on UI without sync
- ✅ **Faster Deployments:** Update UI without restarting sync
- ✅ **Resource Control:** Memory limits per service

---

## 🚨 Potential Issues & Solutions

### Issue 1: SQLite Write Contention
**Problem:** Backend writes, frontend reads → locks

**Solutions:**
- Use WAL mode (already planned)
- Increase timeout to 30s
- Backend commits in batches
- Frontend uses read-only connections where possible

### Issue 2: Stale Data in UI
**Problem:** User sees old data if sync just ran

**Solutions:**
- Poll database every 5s for updates
- WebSocket push notifications (future)
- "Refresh" button in UI
- Show last_sync_time prominently

### Issue 3: Manual Sync Trigger
**Problem:** User clicks "Sync Now" but backend on timer

**Solutions:**
- **Option A:** Add HTTP endpoint to backend (simple ping)
- **Option B:** Set `sync_pending` flag in database, backend checks every 5s
- **Option C:** Shared Redis/message queue (overkill for this)

**Recommended: Option B**
```python
# Frontend: Set flag
await set_sync_pending(session, provider_id=123, pending=True)

# Backend: Check flag
if await has_pending_sync(session):
    await sync_now()
    await clear_pending_flags(session)
```

---

## ✅ Implementation Checklist

- [ ] Create `shared/` directory with common code
- [ ] Create `backend/` with sync_worker.py
- [ ] Create `frontend/` with api.py + routes
- [ ] Move templates and static files to frontend
- [ ] Update database.py with WAL mode
- [ ] Write backend Dockerfile
- [ ] Write frontend Dockerfile
- [ ] Update docker-compose.yml
- [ ] Add sync_pending flag to database
- [ ] Test backend standalone
- [ ] Test frontend standalone
- [ ] Integration test both services
- [ ] Document new architecture
- [ ] Update README.md
- [ ] Deploy and monitor

---

## 🎯 Success Criteria

1. ✅ Total memory < 300 MB (both services combined)
2. ✅ Sync continues to work (same frequency)
3. ✅ UI remains responsive
4. ✅ No data loss during migration
5. ✅ All existing features work
6. ✅ Services can restart independently
7. ✅ Clear separation of concerns

---

## 📝 Notes

- Keep SQLite for simplicity (no PostgreSQL dependency for updater)
- Backend can run on separate machine if needed (shared volume)
- Frontend is stateless, can run multiple instances
- Both services share same database schema (via alembic)
- Clean upgrade path: stop old, start new, same data
