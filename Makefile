.PHONY: setup test lint typecheck dev bench up

setup:
	uv sync --extra dev

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy

bench:
	uv run python -m inference_demo.bench.static_vs_continuous

# Placeholders wired up in later phases.
dev:
	@echo "dev: implemented in phase 4 (gateway/app.py)"

up:
	@echo "up: implemented in phase 7 (docker compose)"
