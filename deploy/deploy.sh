#!/usr/bin/env bash
# Sync the repo to the VPS and (re)build/restart the co-host demo stack.
# Mirrors finance-tracker's deploy.sh: rsync the working tree to the box, then
# rebuild the single co-host container joined to the existing proxy network.
# Usage: ./deploy/deploy.sh <ssh-host>   (or set DEPLOY_HOST)
set -euo pipefail

HOST="${1:-${DEPLOY_HOST:-}}"
if [ -z "$HOST" ]; then
  echo "Usage: ./deploy/deploy.sh <ssh-host>   (an ~/.ssh/config alias or user@ip)" >&2
  echo "Or set DEPLOY_HOST in your environment." >&2
  exit 1
fi

# The existing reverse-proxy network the demo container joins (finance's Caddy).
NET="${NET:-finance-tracker_default}"
# Baked into the UI bundle at build time AND read by the backend — must match.
: "${CONTROL_TOKEN:?set CONTROL_TOKEN (stable across deploys)}"

REMOTE_DIR="llm-control-plane"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

rsync -az --delete \
  --exclude '.git' \
  --exclude '.claude' \
  --exclude '.venv' \
  --exclude 'ui/node_modules' \
  --exclude 'ui/dist' \
  --exclude '__pycache__' \
  "$REPO_ROOT/" "$HOST:$REMOTE_DIR/"

ssh "$HOST" "cd $REMOTE_DIR && \
  NET='$NET' CONTROL_TOKEN='$CONTROL_TOKEN' \
  docker compose -f deploy/docker-compose.cohost.yml up -d --build && \
  docker image prune -f"

echo "Deployed. Check status with:"
echo "  ssh $HOST 'docker ps --filter name=llm-dashboard'"
