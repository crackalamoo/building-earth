.PHONY: frontend backend sim

frontend:
	cd frontend && npm run dev

backend:
	uv run uvicorn backend.server.main:app --port 8000 --reload

sim:
	uv run python backend/main.py --resolution 5
