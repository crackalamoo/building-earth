.PHONY: frontend backend sim docker docker-stop \
        export export-main export-stages downsample-mobile \
        upload upload-main upload-stages upload-mobile upload-obs \
        deploy

R2_BUCKET = climate-sim-data

# ── Dev ─────────────────────────────────────────────────────────────────
frontend:
	cd frontend && npm run dev

backend:
	uv run uvicorn backend.server.main:app --port 8000 --reload

sim:
	uv run python backend/main.py --resolution 5

docker:
	docker compose up --build

docker-stop:
	docker compose down

# ── Export binaries ─────────────────────────────────────────────────────
# Produces frontend/public/main.bin.gz + manifest, and stage1-4 bins.
# Uses --cache so main.npz from the last sim run is reused.
export: export-main export-stages downsample-mobile

export-main:
	PYTHONPATH=backend uv run python -m export_frontend_data --cache --resolution 5 --interpolate

export-stages:
	PYTHONPATH=backend uv run python -m export_frontend_data --onboarding --cache --resolution 5 --interpolate --stages 1,2,3,4

# Build mobile-resolution variants from the freshly exported high-res files.
# Output is frontend/public/{main,stage1..4}_mobile.bin.gz + manifests.
downsample-mobile:
	PYTHONPATH=backend uv run python -m export_frontend_data.downsample_for_mobile --input-dir frontend/public --factor 4

# ── Upload to R2 ────────────────────────────────────────────────────────
# Requires wrangler CLI + `wrangler login` (or R2_ACCESS_KEY_ID/R2_SECRET_ACCESS_KEY env vars).
upload: upload-main upload-stages upload-mobile upload-obs

upload-main:
	wrangler r2 object put $(R2_BUCKET)/main.npz --file data/main.npz --remote
	wrangler r2 object put $(R2_BUCKET)/main.bin.gz --file frontend/public/main.bin.gz --remote
	wrangler r2 object put $(R2_BUCKET)/main.manifest.json --file frontend/public/main.manifest.json --remote
	wrangler r2 object put $(R2_BUCKET)/landmask1deg.bin --file frontend/public/landmask1deg.bin --remote

upload-stages:
	for i in 1 2 3 4; do \
		wrangler r2 object put $(R2_BUCKET)/stage$$i.bin.gz --file frontend/public/stage$$i.bin.gz --remote; \
		wrangler r2 object put $(R2_BUCKET)/stage$$i.manifest.json --file frontend/public/stage$$i.manifest.json --remote; \
	done

upload-mobile:
	wrangler r2 object put $(R2_BUCKET)/main_mobile.bin.gz --file frontend/public/main_mobile.bin.gz --remote
	wrangler r2 object put $(R2_BUCKET)/main_mobile.manifest.json --file frontend/public/main_mobile.manifest.json --remote
	for i in 1 2 3 4; do \
		wrangler r2 object put $(R2_BUCKET)/stage$${i}_mobile.bin.gz --file frontend/public/stage$${i}_mobile.bin.gz --remote; \
		wrangler r2 object put $(R2_BUCKET)/stage$${i}_mobile.manifest.json --file frontend/public/stage$${i}_mobile.manifest.json --remote; \
	done

upload-obs:
	wrangler r2 object put $(R2_BUCKET)/ref_climatology_1deg_1981-2010.nc --file data/processed/ref_climatology_1deg_1981-2010.nc --remote
	wrangler r2 object put $(R2_BUCKET)/ref_humidity_precip_1deg_1981-2010.nc --file data/processed/ref_humidity_precip_1deg_1981-2010.nc --remote

# ── Full deploy: sim → export → upload → push ──────────────────────────
# After push, Cloudflare Pages and Railway auto-rebuild from main.
deploy: sim export upload
	git push
