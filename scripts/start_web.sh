#!/bin/bash
set -e

echo "Initializing database schema..."
python /app/scripts/init_database.py

echo "Stamping database as up-to-date..."
python -m alembic stamp head || echo "Alembic stamp failed or not needed"

echo "Starting web service..."
exec uvicorn frontend.api:create_app --factory --host 0.0.0.0 --port 8000
