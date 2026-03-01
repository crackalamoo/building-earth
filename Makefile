.PHONY: frontend backend sim docker docker-stop

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
