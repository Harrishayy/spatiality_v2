#!/usr/bin/env bash
# Run the full pipeline on a phone video and open the result in the viewer.
#
# This script uses the Modal execution path (Path A in the README). If you'd
# rather run on your own GPU without a Modal account, use
# `scripts/run_local_gpu.py` instead — see the README's "Path B" section.
#
# Usage:
#   # Drop your own .mp4 at backend/data/inputs/<scene_id>/source.mp4, then:
#   SCENE_ID=my_room bash scripts/run_scene.sh
#
#   # Or fetch a remote clip:
#   SCENE_ID=my_room SAMPLE_URL=https://... bash scripts/run_scene.sh
#
# Prereqs (one-time setup, see README):
#   - Modal CLI authenticated (`modal token new`)
#   - Modal apps deployed (`modal deploy backend/modal/inference.py`
#     and `modal deploy backend/modal/segmentation.py`)
#   - Hugging Face and Pydantic AI Gateway Modal Secrets populated
#   - ffmpeg + ffprobe on PATH
#
# What it does:
#   1. Expects a video at backend/data/inputs/$SCENE_ID/source.mp4
#      (downloads it if SAMPLE_URL is set).
#   2. Invokes scripts/run_pipeline_cli.py, which mirrors what the FastAPI
#      orchestrator does on every web upload (ffmpeg → push → inference →
#      pull poses → segmentation + Stage 4 → pull all).
#   3. Leaves all artefacts under backend/data/outputs/$SCENE_ID/ so `pnpm dev`
#      in web/ can render the scene at /scenes/$SCENE_ID.

set -euo pipefail

SCENE_ID="${SCENE_ID:-my_scene}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INPUT_DIR="$REPO_ROOT/backend/data/inputs/$SCENE_ID"
VIDEO_PATH="$INPUT_DIR/source.mp4"

mkdir -p "$INPUT_DIR"
if [[ ! -f "$VIDEO_PATH" ]]; then
  if [[ -n "${SAMPLE_URL:-}" ]]; then
    echo "[run_scene] downloading sample video → $VIDEO_PATH"
    curl -fL --retry 3 -o "$VIDEO_PATH" "$SAMPLE_URL"
  else
    cat <<EOF >&2
run_scene.sh: no input video found at $VIDEO_PATH.
Drop your own .mp4 there (or pass SAMPLE_URL=https://… to fetch one) and
re-run this script. Override the scene id with SCENE_ID=foo.
EOF
    exit 2
  fi
else
  echo "[run_scene] reusing existing $VIDEO_PATH"
fi

echo "[run_scene] running pipeline for scene_id=$SCENE_ID"
cd "$REPO_ROOT"
python scripts/run_pipeline_cli.py "$SCENE_ID" "$@"

echo
echo "[run_scene] done. Artefacts in backend/data/outputs/$SCENE_ID/:"
ls -lh "backend/data/outputs/$SCENE_ID/" || true
echo
echo "[run_scene] open the viewer:"
echo "  cd web && pnpm dev   # then visit http://localhost:3000/scenes/$SCENE_ID"
