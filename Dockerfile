FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY python/ ./python/
COPY demo_api/ ./demo_api/
COPY config/ ./config/

# Set Python path
ENV PYTHONPATH=/app

# Expose API port
EXPOSE 8000

# Run the API
CMD ["python", "-m", "uvicorn", "demo_api.app:app", "--host", "0.0.0.0", "--port", "8000"]

