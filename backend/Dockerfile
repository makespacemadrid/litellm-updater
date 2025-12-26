FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install backend dependencies (no FastAPI, Jinja2, uvicorn)
COPY backend/requirements.txt /app/backend/
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy shared code
COPY shared/ /app/shared/
COPY alembic/ /app/alembic/
COPY alembic.ini /app/alembic.ini

# Copy backend code
COPY backend/ /app/backend/
COPY scripts/ /app/scripts/

# Run sync worker
CMD ["python", "-m", "backend.sync_worker"]
