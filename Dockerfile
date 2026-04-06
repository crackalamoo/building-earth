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
COPY data/processed/ref_climatology_1deg_1981-2010.nc data/processed/ref_climatology_1deg_1981-2010.nc
COPY data/processed/ref_humidity_precip_1deg_1981-2010.nc data/processed/ref_humidity_precip_1deg_1981-2010.nc

# Copy frontend binary export (used by ClimateDataStore for high-res sampling)
COPY frontend/public/main.bin.gz frontend/public/main.bin.gz
COPY frontend/public/main.manifest.json frontend/public/main.manifest.json

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "backend.server.main:app", "--host", "0.0.0.0", "--port", "8000"]
