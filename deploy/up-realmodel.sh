#!/usr/bin/env bash
# Turnkey HOST-NATIVE real-model run — our continuous batching on a real model
# (Apple MPS). Not Docker: a container on macOS can't reach the GPU, and torch's
# MPS device isn't available inside it, so this runs straight on the host.
#
#   - installs the heavy deps (torch/transformers) via the `realmodel` extra
#   - builds the console and serves it + the API from one process
#   - opens the browser; the model (~1 GB) downloads on first run
#
# Usage:  make up-realmodel        (MODEL_NAME overrides the default HF model id)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
MODEL_NAME="${MODEL_NAME:-}"   # empty -> RealModelWorker's default (Qwen2.5-0.5B-Instruct)

echo "1/3  Installing real-model deps (torch/transformers) — first run is a big download..."
uv sync --extra dev --extra realmodel

echo "2/3  Building the console..."
npm --prefix ui install >/dev/null 2>&1
npm --prefix ui run build >/dev/null

PORT="$(python3 - <<'PY'
import socket
for p in range(8000, 8030):
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", p)); s.close(); print(p); break
    except OSError:
        continue
else:
    print(8000)
PY
)"
URL="http://127.0.0.1:$PORT"

# Open the browser once the server answers (the first request also triggers the
# model download, so give startup a generous window).
( for _ in $(seq 1 120); do
    curl -fsS --max-time 2 "$URL/api/snapshot" >/dev/null 2>&1 && { command -v open >/dev/null 2>&1 && open "$URL"; break; }
    sleep 2
  done ) &

echo "3/3  Starting host-native real-model server on $URL  (model downloads on first run)"
echo "     Backend: realmodel  ·  Stop: Ctrl-C"
exec env WORKER_BACKEND=realmodel UI_DIST="$ROOT/ui/dist" ${MODEL_NAME:+MODEL_NAME="$MODEL_NAME"} \
  uv run uvicorn inference_demo.gateway.app:app --host 127.0.0.1 --port "$PORT"
