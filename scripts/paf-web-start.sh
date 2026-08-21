#!/usr/bin/env bash
# Start the paf Gradio web UI as a foreground process.
# Used both by systemd (paf-web.service) and for manual/diagnostic runs.
#
#   scripts/paf-web-start.sh [agent_name] [port]
#
# Sources the project .env (exports OPENROUTER_API_KEY / OPENAI_API_KEY etc.)
# and execs the venv python so the service holds the real PID.
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

exec "$PYTHON" "$PROJECT_DIR/ui/web.py" "$@"
