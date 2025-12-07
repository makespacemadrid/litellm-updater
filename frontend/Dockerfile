FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install frontend dependencies
COPY frontend/requirements.txt /app/frontend/
RUN pip install --no-cache-dir -r frontend/requirements.txt

# Copy shared code
COPY shared/ /app/shared/

# Copy backend helpers used by frontend sync endpoints
COPY backend/ /app/backend/

# Copy frontend code
COPY frontend/ /app/frontend/

EXPOSE 8000

CMD ["uvicorn", "frontend.api:create_app", "--host", "0.0.0.0", "--port", "8000", "--factory"]
