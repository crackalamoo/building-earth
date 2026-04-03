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

# Copy climate data
COPY data/main.npz data/main.npz

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "backend.server.main:app", "--host", "0.0.0.0", "--port", "8000"]
