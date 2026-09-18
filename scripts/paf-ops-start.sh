#!/usr/bin/env bash
# Start the paf ops transport (systemd control for paf-sse) as a foreground
# process. Used by systemd (paf-ops.service) and for manual/diagnostic runs.
#
#   scripts/paf-ops-start.sh [port]
#
# Sources the project .env (exports OPENROUTER_API_KEY etc.) then execs the
# venv python so the service holds the real PID. The .env also carries
# PAF_OPS_TOKEN (the auth token for /api/restart) and optionally PAF_OPS_HOST.
#
# systemd command:
# systemctl --user daemon-reload
# systemctl --user enable --now paf-ops.service
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# Export .env keys into the environment (plain `source` doesn't export).
set -a
# shellcheck disable=SC1091
source .env
set +a

# Expand relative venv python path to absolute so exec works from any cwd.
PYTHON="$PROJECT_DIR/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  echo "error: venv python not found at $PYTHON" >&2
  exit 1
fi

# Default port 8091 (keeps clear of SSE :8090 and Langfuse :3000/:3001).
PORT="${1:-8091}"

exec "$PYTHON" -m uvicorn ui.ops:app --host 0.0.0.0 --port "$PORT" --log-level info
