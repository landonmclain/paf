#!/usr/bin/env bash
# Shrink the vhs output in place: 1000px wide, 12 fps, 64 colors (~2 MB -> ~1.2 MB).
set -euo pipefail
gif="$(cd "$(dirname "$0")" && pwd)/demo.gif"
tmp="$(mktemp --suffix=.gif)"
ffmpeg -v error -y -i "$gif" -vf "fps=12,scale=1000:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle" "$tmp"
mv "$tmp" "$gif"
