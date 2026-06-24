#!/usr/bin/env bash
# Turnkey launcher for the local demo. One command, no copy-paste:
#   - picks a free host port (so a port already in use never blocks you)
#   - (ollama mode) makes sure Ollama is installed, running, and the model is pulled
#   - rebuilds + starts the stack detached, waits until it answers, opens the browser
#
# Usage:  deploy/up.sh sim        (control plane + sim backend)
#         deploy/up.sh ollama     (control plane in Docker -> host-native Ollama)
set -euo pipefail

MODE="${1:-sim}"
MODEL_NAME="${MODEL_NAME:-qwen2.5:0.5b}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

COMPOSE=(-f docker-compose.yml)
[ "$MODE" = "ollama" ] && COMPOSE+=(-f docker-compose.ollama.yml)

pick_free_port() {
  python3 - <<'PY'
import socket
for p in range(8088, 8121):
    s = socket.socket()
    try:
        s.bind(("0.0.0.0", p)); s.close(); print(p); break
    except OSError:
        continue
else:
    print(8088)
PY
}

ollama_ready() { curl -fsS --max-time 2 http://localhost:11434/api/tags >/dev/null 2>&1; }

if [ "$MODE" = "ollama" ]; then
  if ! command -v ollama >/dev/null 2>&1; then
    echo "Ollama isn't installed. Install it once:  brew install ollama" >&2
    echo "(or grab it from https://ollama.com), then re-run 'make up-ollama'." >&2
    exit 1
  fi
  if ! ollama_ready; then
    echo "Starting Ollama in the background..."
    nohup ollama serve >/tmp/ollama-demo.log 2>&1 &
    for _ in $(seq 1 30); do ollama_ready && break; sleep 1; done
  fi
  if ! ollama list 2>/dev/null | grep -q "$MODEL_NAME"; then
    echo "Pulling model $MODEL_NAME (first run only)..."
    ollama pull "$MODEL_NAME"
  fi
fi

# Drop any previous instance of THIS stack so re-running never stacks duplicates.
docker compose "${COMPOSE[@]}" down --remove-orphans >/dev/null 2>&1 || true

PORT="$(pick_free_port)"
echo "Building + starting the control plane on http://localhost:$PORT ..."
HTTP_PORT="$PORT" MODEL_NAME="$MODEL_NAME" docker compose "${COMPOSE[@]}" up -d --build

URL="http://localhost:$PORT"
for _ in $(seq 1 30); do
  curl -fsS --max-time 2 "$URL/api/snapshot" >/dev/null 2>&1 && break
  sleep 1
done
command -v open >/dev/null 2>&1 && open "$URL" >/dev/null 2>&1 || true

echo
echo "  ▶ Console:  $URL  (opened in your browser)"
[ "$MODE" = "ollama" ] && echo "  ▶ Model:    $MODEL_NAME via host Ollama"
echo "  ▶ Logs:     docker compose ${COMPOSE[*]} logs -f"
echo "  ▶ Stop:     docker compose ${COMPOSE[*]} down"
