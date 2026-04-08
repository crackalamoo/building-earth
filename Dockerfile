FROM python:3.13-slim
WORKDIR /app

# Install system dependencies for rasterio/GDAL, plus curl for fetching data
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev gdal-bin gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install Python dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --no-install-project

# Copy backend source
COPY backend/ backend/

# Fetch climate data from Cloudflare R2. Build arg lets the URL be
# overridden at build time, but defaults to the public R2 bucket.
ARG DATA_BASE=https://pub-9a1d53d2ac6f42a8b83952d8fab2e668.r2.dev
RUN mkdir -p data/processed frontend/public && \
    curl -fsSL "$DATA_BASE/main.npz" -o data/main.npz && \
    curl -fsSL "$DATA_BASE/main.bin.gz" -o frontend/public/main.bin.gz && \
    curl -fsSL "$DATA_BASE/main.manifest.json" -o frontend/public/main.manifest.json && \
    for i in 1 2 3 4; do \
      curl -fsSL "$DATA_BASE/stage$i.npz" -o "data/stage$i.npz" && \
      curl -fsSL "$DATA_BASE/stage$i.bin.gz" -o "frontend/public/stage$i.bin.gz" && \
      curl -fsSL "$DATA_BASE/stage$i.manifest.json" -o "frontend/public/stage$i.manifest.json"; \
    done && \
    curl -fsSL "$DATA_BASE/ref_climatology_1deg_1981-2010.nc" -o data/processed/ref_climatology_1deg_1981-2010.nc && \
    curl -fsSL "$DATA_BASE/ref_humidity_precip_1deg_1981-2010.nc" -o data/processed/ref_humidity_precip_1deg_1981-2010.nc

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "backend.server.main:app", "--host", "0.0.0.0", "--port", "8000"]
