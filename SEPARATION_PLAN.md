# Plan: Backend/Frontend Separation & Memory Optimization

**Current State:** Monolithic FastAPI app consuming ~2.9 GiB memory
**Goal:** Separate into lightweight backend sync service + frontend API/UI, optimize memory usage

---

## 📊 Current Architecture Analysis

### Memory Usage Issues
- **Current:** 2.9 GiB for single process
- **Root Causes:**
  1. `SyncState` class: In-memory cache of all models from all providers
  2. `ModelDetailsCache`: In-memory TTL cache (max 1000 entries) for Ollama model details
  3. Large JSON payloads in `raw_metadata` field (includes modelfiles, tensors data)
  4. All FastAPI routes + templates + static files loaded in memory
  5. Jinja2 template compilation overhead

### Database Duplication Analysis

**Current Setup:**
- **Local SQLite DB** (`litellm_updater`):
  - Providers table
  - Models table (with raw_metadata, litellm_params, user_params)
  - Config table

- **LiteLLM PostgreSQL DB:**
  - `LiteLLM_ProxyModelTable` (model_id, model_name, litellm_params, model_info)
  - 20+ other tables for auth, teams, budgets, etc.

**Key Insight:** LiteLLM's database is **richer and authoritative** - it already stores all model configuration we need!

### Data Flow
```
Providers → Fetch Models → Normalize → Local DB → Push to LiteLLM → LiteLLM DB
                                ↓
                         SyncState (memory)
                                ↓
                         Frontend reads from memory
```

---

## 🎯 Proposed Architecture

### Option A: **Hybrid Approach** (RECOMMENDED)
Use LiteLLM database as primary source, keep lightweight local metadata.

**Architecture:**
```
┌─────────────────────────────────────┐
│   BACKEND (Sync Worker)             │
│   - Python service (100-200 MB)     │
│   - Periodic sync scheduler          │
│   - Direct provider fetching         │
│   - Direct LiteLLM DB writes         │
│   - NO HTTP server                   │
│   - NO in-memory caches              │
└─────────────────────────────────────┘
         │
         │ Writes to
         ↓
┌─────────────────────────────────────┐
│   PostgreSQL (LiteLLM DB)           │
│   - LiteLLM_ProxyModelTable         │
│   - Custom: provider_metadata table │
│   - Custom: sync_status table       │
└─────────────────────────────────────┘
         ↑
         │ Reads from
┌─────────────────────────────────────┐
│   FRONTEND (API + UI)               │
│   - FastAPI (100-150 MB)            │
│   - REST API endpoints              │
│   - Static HTML/CSS/JS              │
│   - NO sync logic                   │
│   - NO provider fetching            │
└─────────────────────────────────────┘
```

**Benefits:**
- ✅ Eliminates duplicate data storage
- ✅ Single source of truth (LiteLLM DB)
- ✅ Frontend can query LiteLLM models directly
- ✅ No in-memory state in either service
- ✅ Backend can be very lightweight
- ✅ Easy to scale horizontally (multiple frontends)

**Tradeoffs:**
- ⚠️ Requires LiteLLM DB schema extensions
- ⚠️ Need to maintain compatibility with LiteLLM updates
- ⚠️ Provider metadata needs custom tables


### Option B: **Complete Separation with Local DB**
Keep local SQLite, separate services.

**Architecture:**
```
BACKEND (Sync) → Local SQLite DB ← FRONTEND (API/UI)
      │                                  ↑
      └──────> LiteLLM API ──────────────┘
```

**Benefits:**
- ✅ Simpler migration (less DB work)
- ✅ Independent of LiteLLM DB schema
- ✅ Can work with any LiteLLM deployment

**Tradeoffs:**
- ❌ Keeps data duplication
- ❌ SQLite can be bottleneck with concurrent reads/writes
- ❌ Still need to maintain local model state


### Option C: **Stateless Backend + LiteLLM API Only**
Backend writes to LiteLLM, frontend reads via LiteLLM API.

**Tradeoffs:**
- ❌ Can't store provider metadata (base_url, api_key, sync_enabled)
- ❌ Can't track orphaned models
- ❌ Can't store user customizations (ollama_mode, user_params)
- ❌ LiteLLM API doesn't expose all needed metadata

---

## 📋 Recommended Implementation: Option A (Hybrid)

### Phase 1: Database Schema Extension

**New tables in LiteLLM PostgreSQL:**

```sql
-- Provider configuration (replaces local providers table)
CREATE TABLE litellm_updater_providers (
    id SERIAL PRIMARY KEY,
    name VARCHAR UNIQUE NOT NULL,
    base_url VARCHAR NOT NULL,
    type VARCHAR NOT NULL CHECK (type IN ('ollama', 'openai', 'compat')),
    api_key VARCHAR,
    prefix VARCHAR,
    default_ollama_mode VARCHAR CHECK (default_ollama_mode IN ('ollama', 'openai')),
    tags JSONB DEFAULT '[]',
    access_groups JSONB DEFAULT '[]',
    sync_enabled BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Model metadata extensions (complements LiteLLM_ProxyModelTable)
CREATE TABLE litellm_updater_model_metadata (
    model_id VARCHAR PRIMARY KEY,  -- References LiteLLM_ProxyModelTable.model_id
    provider_id INTEGER REFERENCES litellm_updater_providers(id) ON DELETE CASCADE,
    raw_metadata JSONB NOT NULL,  -- Full upstream payload (for diagnostics)
    user_params JSONB,  -- User overrides
    ollama_mode VARCHAR CHECK (ollama_mode IN ('ollama', 'openai')),
    mapped_provider_id INTEGER,  -- For compat models
    mapped_model_id VARCHAR,
    is_orphaned BOOLEAN DEFAULT FALSE,
    orphaned_at TIMESTAMP,
    user_modified BOOLEAN DEFAULT FALSE,
    sync_enabled BOOLEAN DEFAULT TRUE,
    first_seen TIMESTAMP DEFAULT NOW(),
    last_seen TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Sync status tracking
CREATE TABLE litellm_updater_sync_status (
    id SERIAL PRIMARY KEY,
    provider_id INTEGER REFERENCES litellm_updater_providers(id) ON DELETE CASCADE,
    last_sync_at TIMESTAMP,
    last_sync_status VARCHAR,  -- 'success', 'error', 'in_progress'
    last_sync_error TEXT,
    models_added INTEGER DEFAULT 0,
    models_updated INTEGER DEFAULT 0,
    models_removed INTEGER DEFAULT 0
);

-- Global config
CREATE TABLE litellm_updater_config (
    id INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),  -- Singleton
    sync_interval_seconds INTEGER DEFAULT 300,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### Phase 2: Backend Service (Sync Worker)

**File: `backend/sync_worker.py`**

**Key Responsibilities:**
1. Load provider configs from `litellm_updater_providers`
2. Fetch models from each provider (Ollama, OpenAI-compatible)
3. Normalize to LiteLLM format
4. Write to `LiteLLM_ProxyModelTable` (main model data)
5. Write to `litellm_updater_model_metadata` (provider tracking)
6. Detect and mark orphaned models
7. Update `litellm_updater_sync_status`

**Memory Optimizations:**
- ❌ Remove `SyncState` class (no in-memory cache)
- ❌ Remove `ModelDetailsCache` (query on-demand)
- ✅ Stream processing (fetch → normalize → write → release)
- ✅ Process one provider at a time
- ✅ Use database as state store

**Estimated Memory:** 80-150 MB (down from 2.9 GB)

**Core Loop:**
```python
async def sync_loop():
    while True:
        providers = await get_enabled_providers(db_session)

        for provider in providers:
            try:
                # Fetch models from provider
                models = await fetch_provider_models(provider)

                # Process in batches to avoid memory spike
                for batch in chunk(models, 50):
                    async with db_session.begin():
                        for model in batch:
                            # Normalize
                            normalized = normalize_model(model, provider)

                            # Write to LiteLLM main table
                            await upsert_litellm_model(db_session, normalized)

                            # Write metadata
                            await upsert_model_metadata(db_session, model, provider)

                # Mark orphans
                await mark_orphaned_models(db_session, provider, model_ids)

                # Update sync status
                await update_sync_status(db_session, provider, success=True)

            except Exception as e:
                await update_sync_status(db_session, provider, success=False, error=str(e))

        # Sleep until next interval
        config = await get_config(db_session)
        await asyncio.sleep(config.sync_interval_seconds)
```

**Deployment:**
```dockerfile
FROM python:3.11-slim
RUN pip install sqlalchemy asyncpg httpx
COPY backend/sync_worker.py /app/
CMD ["python", "/app/sync_worker.py"]
```

### Phase 3: Frontend Service (API + UI)

**File: `frontend/api.py`**

**Key Responsibilities:**
1. Serve HTML/CSS/JS UI
2. REST API for provider CRUD
3. REST API for model management
4. Read from LiteLLM database (no sync logic)
5. Proxy to LiteLLM API when needed

**Endpoints:**
- `GET /` - Dashboard HTML
- `GET /providers` - List providers (from `litellm_updater_providers`)
- `POST /providers` - Create provider
- `GET /providers/{id}/models` - List models (JOIN with `LiteLLM_ProxyModelTable`)
- `PATCH /models/{id}` - Update model metadata
- `POST /models/{id}/refresh` - Trigger backend sync (via database flag)
- `GET /litellm/models` - Proxy to LiteLLM API

**Memory Optimizations:**
- ✅ Lazy template loading
- ✅ No caching (database is cache)
- ✅ Stateless (can run multiple instances)
- ✅ Minimal dependencies

**Estimated Memory:** 80-120 MB

**Example Query:**
```python
async def get_provider_models(provider_id: int):
    query = """
        SELECT
            m.model_id,
            m.model_name,
            m.litellm_params,
            m.model_info,
            mm.is_orphaned,
            mm.user_modified,
            mm.sync_enabled
        FROM LiteLLM_ProxyModelTable m
        JOIN litellm_updater_model_metadata mm ON m.model_id = mm.model_id
        WHERE mm.provider_id = $1
        ORDER BY m.model_name
    """
    return await db.fetch_all(query, provider_id)
```

### Phase 4: Migration Path

**Step 1: Extend LiteLLM Database**
```bash
# Run migration script
docker exec litellm_db psql -U llmproxy -d litellm -f migrations/001_add_updater_tables.sql
```

**Step 2: Migrate Data**
```bash
# Export current SQLite data
python scripts/export_sqlite.py > data/export.json

# Import to PostgreSQL
python scripts/import_to_postgres.py data/export.json
```

**Step 3: Deploy New Services**
```yaml
# docker-compose.yml
services:
  litellm-sync:
    build: ./backend
    environment:
      DATABASE_URL: postgresql://llmproxy:dbpassword9090@db:5432/litellm
      SYNC_INTERVAL: 300
    depends_on:
      - db
    networks:
      - litellm
    deploy:
      resources:
        limits:
          memory: 256M

  litellm-frontend:
    build: ./frontend
    environment:
      DATABASE_URL: postgresql://llmproxy:dbpassword9090@db:5432/litellm
      LITELLM_API_URL: http://litellm:4000
    ports:
      - "4001:8000"
    depends_on:
      - db
      - litellm
    networks:
      - litellm
    deploy:
      resources:
        limits:
          memory: 256M
```

**Step 4: Cutover**
```bash
# Stop old service
docker compose stop litellm-updater

# Start new services
docker compose up -d litellm-sync litellm-frontend

# Monitor logs
docker compose logs -f litellm-sync litellm-frontend
```

---

## 🚀 Alternative: Lightweight Optimization (Quick Win)

If full separation is too complex initially, quick wins:

### 1. Remove In-Memory Caches
```python
# Replace SyncState with database queries
# Replace ModelDetailsCache with TTL in database or remove entirely
```

### 2. Lazy Load Templates
```python
templates = Jinja2Templates(directory="templates", auto_reload=False)
# Add cache_size parameter to limit compiled templates
```

### 3. Stream Large Payloads
```python
# Don't load all models into memory
async for model in stream_models_from_db():
    yield model
```

### 4. Clean Ollama Raw Metadata
```python
# Already implemented _clean_ollama_payload() but ensure it's used everywhere
# Strip tensors, modelfile, projector_info, etc.
```

### 5. Use Database for State
```python
# Store last_sync_time in database, not memory
# Store provider status in database, not memory
```

**Estimated savings:** 2.9 GB → 500-800 MB (still single process)

---

## 📊 Expected Results

### Memory Comparison

| Component | Current | Option A (Hybrid) | Option B (Separated) | Quick Win |
|-----------|---------|-------------------|----------------------|-----------|
| Sync Worker | - | 80-150 MB | 100-200 MB | - |
| Frontend | - | 80-120 MB | 100-150 MB | - |
| Monolith | 2.9 GB | - | - | 500-800 MB |
| **Total** | **2.9 GB** | **~200 MB** | **~300 MB** | **~650 MB** |
| **Savings** | - | **93%** | **90%** | **77%** |

### Scalability Comparison

| Metric | Current | Separated |
|--------|---------|-----------|
| Horizontal scaling | ❌ | ✅ (multiple frontends) |
| Zero-downtime updates | ❌ | ✅ (update sync without downtime) |
| Resource allocation | Fixed | Per-component |
| Failure isolation | ❌ | ✅ (sync fails ≠ UI down) |

---

## 🎯 Recommendation

**Start with Option A (Hybrid)** for maximum benefit:

1. **Week 1:** Create PostgreSQL schema extensions
2. **Week 2:** Build sync worker (backend)
3. **Week 3:** Build frontend API
4. **Week 4:** Migration + testing
5. **Week 5:** Deploy + monitor

**Fallback:** If timeline too aggressive, start with Quick Win optimizations first, then migrate gradually.

**Key Success Metrics:**
- Memory usage < 300 MB total (both services)
- Sync latency < 30s per provider
- UI response time < 200ms
- Zero data loss during migration
