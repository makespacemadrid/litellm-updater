FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Install sqlite3 and other system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends sqlite3 && \
    rm -rf /var/lib/apt/lists/*

# Install application
COPY pyproject.toml README.md /app/
COPY litellm_updater /app/litellm_updater
COPY example.env /app/env.example
COPY scripts /app/scripts

RUN pip install --no-cache-dir .

EXPOSE ${PORT}

CMD ["sh", "-c", "uvicorn litellm_updater.web:create_app --host 0.0.0.0 --port ${PORT}"]
