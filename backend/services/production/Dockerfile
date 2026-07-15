FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install shared dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY . /app

ENV PYTHONPATH=/app:/app/backend:/app/backend/services/production

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "backend.services.production.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
