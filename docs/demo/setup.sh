#!/usr/bin/env bash
# Build a throwaway PAF_HOME for the demo recording and print its path.
# Usage: export PAF_HOME=$(bash docs/demo/setup.sh)
set -euo pipefail
repo="$(cd "$(dirname "$0")/../.." && pwd)"
home="$(mktemp -d -t paf-demo.XXXXXX)"
cp -r "$repo/specs" "$home/specs"
rm -f "$home/specs/assistant.yaml"
cp -r "$repo/src" "$home/src"
find "$home/src" -name __pycache__ -prune -exec rm -rf {} +
mkdir -p "$home/build-cache" && echo x > "$home/build-cache/main.o"
[ -f "$repo/.env" ] && cp "$repo/.env" "$home/.env"
# Warm the codebase map so the recording doesn't pause on a first-time index.
py="$repo/.venv/bin/python"; [ -x "$py" ] || py=python3
PAF_HOME="$home" "$py" -m paf.capabilities.codebase_map.cli --root "$home" >/dev/null 2>&1 || true
echo "$home"
