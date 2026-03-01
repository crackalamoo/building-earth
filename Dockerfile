# Stage 1: Build frontend
FROM node:22-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Python runtime
FROM python:3.13-slim
WORKDIR /app

# Install system dependencies for rasterio/GDAL
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev gdal-bin gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install Python dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --no-install-project

# Copy backend source
COPY backend/ backend/

# Copy built frontend
COPY --from=frontend-build /app/frontend/dist frontend/dist

# Copy climate data
COPY data/main.npz data/main.npz

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "backend.server.main:app", "--host", "0.0.0.0", "--port", "8000"]
