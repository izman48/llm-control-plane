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

# Placeholders wired up in later phases.
dev:
	@echo "dev: implemented in phase 4 (gateway/app.py)"

bench:
	@echo "bench: implemented in phase 1 (bench/static_vs_continuous.py)"

up:
	@echo "up: implemented in phase 7 (docker compose)"
