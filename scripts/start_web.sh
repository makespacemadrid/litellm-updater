#!/bin/bash
set -e

# Check if database exists
if [ -f "/app/data/models.db" ]; then
    echo "Existing database found, running migrations..."
    python -m alembic upgrade head || {
        echo "Migration failed, trying fallback initialization..."
        python /app/scripts/init_database.py
        python -m alembic stamp head || echo "Alembic stamp failed"
    }
else
    echo "No database found, initializing fresh database..."
    python /app/scripts/init_database.py
    python -m alembic stamp head || echo "Alembic stamp failed"
fi

echo "Starting web service..."
exec uvicorn frontend.api:create_app --factory --host 0.0.0.0 --port 8000
