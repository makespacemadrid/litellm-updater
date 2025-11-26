FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Install application
COPY pyproject.toml README.md /app/
COPY litellm_updater /app/litellm_updater
COPY data /app/data
COPY example.env /app/env.example
COPY scripts /app/scripts

RUN pip install --no-cache-dir .

EXPOSE ${PORT}

CMD ["sh", "-c", "uvicorn litellm_updater.web:create_app --host 0.0.0.0 --port ${PORT}"]
