#!/usr/bin/env bash
# Start the paf SSE transport as a foreground process.
# Used both by systemd (paf-sse.service) and for manual/diagnostic runs.
#
#   scripts/paf-sse-start.sh [port]
#
# Sources the project .env (exports OPENROUTER_API_KEY / OPENAI_API_KEY etc.)
# and execs the venv python so the service holds the real PID.
#
# systemd command:
# systemctl --user daemon-reload
# systemctl --user enable --now paf-sse.service
# systemctl --user restart paf-sse.service && sleep 4 && systemctl --user is-active paf-sse.service
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

# Default port 8090 (keeps clear of Gradio's :8080; overridable via $1).
PORT="${1:-8090}"

exec "$PYTHON" -m uvicorn ui.sse:app --host 0.0.0.0 --port "$PORT" --log-level info