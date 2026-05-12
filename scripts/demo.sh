#!/usr/bin/env bash
# Demo driver: download a sample phone video and run the full pipeline.
#
# This script uses the Modal execution path (Path A in the README). If you'd
# rather run on your own GPU without a Modal account, use
# `scripts/run_local_gpu.py` instead — see the README's "Path B" section.
#
# Usage:  bash scripts/demo.sh
#
# Prereqs (one-time setup, see README):
#   - Modal CLI authenticated (`modal token new`)
#   - Modal apps deployed (`modal deploy backend/modal/inference.py`
#     and `modal deploy backend/modal/segmentation.py`)
#   - Hugging Face and Pydantic AI Gateway Modal Secrets populated
#   - ffmpeg + ffprobe on PATH
#
# What it does:
#   1. Downloads a small sample video into backend/data/inputs/demo/source.mp4
#   2. Invokes scripts/run_pipeline_cli.py, which mirrors what the FastAPI
#      orchestrator does on every web upload (ffmpeg → push → inference →
#      pull poses → segmentation + Stage 5 → pull all).
#   3. Leaves all artefacts under backend/data/outputs/demo/ so `pnpm dev`
#      in web/ can render the scene at /scenes/demo.

set -euo pipefail

SCENE_ID="${SCENE_ID:-demo}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INPUT_DIR="$REPO_ROOT/backend/data/inputs/$SCENE_ID"
VIDEO_PATH="$INPUT_DIR/source.mp4"

# Public URL of the sample capture. Replace with your own R2 / S3 link before
# publishing. Keep clips under ~30 MB for a fast first run.
#
# TODO(maintainer): host docs/sample_scene/source.mp4 (or pick any short
# handheld interior clip) and paste the link here. The clip in our test
# rotation is a ~20 s walk-through of a small office, ~25 MB at 1080p30.
SAMPLE_URL="${SAMPLE_URL:-PLACEHOLDER_REPLACE_BEFORE_PUBLISH}"

if [[ "$SAMPLE_URL" == "PLACEHOLDER_REPLACE_BEFORE_PUBLISH" ]]; then
  cat <<'EOF' >&2
demo.sh: SAMPLE_URL is unset. Either:
  - set SAMPLE_URL to a public-readable .mp4 (R2 / S3 / Cloudflare R2)
    and re-run:    SAMPLE_URL="https://…/source.mp4" bash scripts/demo.sh
  - or drop your own video at  backend/data/inputs/demo/source.mp4
    and re-run this script — it will skip the download step.

This placeholder exists so the script itself is committable without bundling
a 25 MB video into the git repo.
EOF
  if [[ ! -f "$VIDEO_PATH" ]]; then
    exit 2
  fi
fi

mkdir -p "$INPUT_DIR"
if [[ ! -f "$VIDEO_PATH" ]]; then
  echo "[demo] downloading sample video → $VIDEO_PATH"
  curl -fL --retry 3 -o "$VIDEO_PATH" "$SAMPLE_URL"
else
  echo "[demo] reusing existing $VIDEO_PATH"
fi

echo "[demo] running pipeline for scene_id=$SCENE_ID"
cd "$REPO_ROOT"
python scripts/run_pipeline_cli.py "$SCENE_ID" "$@"

echo
echo "[demo] done. Artefacts in backend/data/outputs/$SCENE_ID/:"
ls -lh "backend/data/outputs/$SCENE_ID/" || true
echo
echo "[demo] open the viewer:"
echo "  cd web && pnpm dev   # then visit http://localhost:3000/scenes/$SCENE_ID"
