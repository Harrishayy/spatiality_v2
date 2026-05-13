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
#      pull poses → segmentation + Stage 4 → pull all).
#   3. Leaves all artefacts under backend/data/outputs/demo/ so `pnpm dev`
#      in web/ can render the scene at /scenes/demo.

set -euo pipefail

SCENE_ID="${SCENE_ID:-demo}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INPUT_DIR="$REPO_ROOT/backend/data/inputs/$SCENE_ID"
VIDEO_PATH="$INPUT_DIR/source.mp4"

# Bring your own video. Drop a phone-captured .mp4 at
#   backend/data/inputs/demo/source.mp4
# (override the path by setting SCENE_ID=foo → backend/data/inputs/foo/...).
# Keep clips under ~30 MB / ~30 s for a fast first run; a short handheld
# walk-through of an interior space works best.
#
# If SAMPLE_URL is set, the script will fetch from there instead — handy for
# CI or sharing a canonical clip with reviewers.
mkdir -p "$INPUT_DIR"
if [[ ! -f "$VIDEO_PATH" ]]; then
  if [[ -n "${SAMPLE_URL:-}" ]]; then
    echo "[demo] downloading sample video → $VIDEO_PATH"
    curl -fL --retry 3 -o "$VIDEO_PATH" "$SAMPLE_URL"
  else
    cat <<EOF >&2
demo.sh: no input video found at $VIDEO_PATH.
Drop your own .mp4 there (or pass SAMPLE_URL=https://… to fetch one) and
re-run this script.
EOF
    exit 2
  fi
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
