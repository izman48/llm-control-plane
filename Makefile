.PHONY: setup test lint typecheck dev bench bench-real up ui-install ui-dev ui-build ui-test

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

# Real-model benchmark (host-native; needs the realmodel extra).
bench-real:
	uv run python -m inference_demo.bench.real_static_vs_continuous

dev:
	uv run uvicorn inference_demo.gateway.app:app --host 127.0.0.1 --port 8000

# ---- React control console (ui/) ----
ui-install:
	npm --prefix ui install

ui-dev:
	npm --prefix ui run dev

ui-build:
	npm --prefix ui run build

ui-test:
	npm --prefix ui test

# Full sim stack (control plane + console) via Docker. Picks a free port, starts
# detached, waits until ready, and opens the console in your browser.
up:
	./deploy/up.sh sim

# Hybrid: dockerized control plane + a REAL model served by host-native Ollama
# (Docker on macOS can't reach the GPU). Auto-starts Ollama + pulls the model if
# needed, picks a free port, and opens the console. MODEL_NAME overrides the tag.
up-ollama:
	./deploy/up.sh ollama
