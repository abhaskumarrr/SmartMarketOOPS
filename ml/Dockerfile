# syntax=docker/dockerfile:1
FROM python:3.10-slim AS base

WORKDIR /app

# Install dependencies only when needed
FROM base AS deps

# Install required packages for build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base AS dev

# Copy dependencies from deps stage
COPY --from=deps /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Set Python to run in unbuffered mode
ENV PYTHONUNBUFFERED=1

# Copy application code
COPY . .

# Expose port
EXPOSE 3002

# Start the ML service
CMD ["python", "-m", "ml.backend.src.scripts.server"]
