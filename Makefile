.PHONY: install test lint format serve ask docker

install:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

serve:
	uv run uvicorn agent.api:app --reload --port 8000

ask:
	uv run python -m agent "$(Q)"

docker:
	docker build -t langgraph-research-agent .
	docker run --rm -p 8000:8000 --env-file .env langgraph-research-agent
