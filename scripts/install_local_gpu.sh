#!/usr/bin/env bash
# Install spatiality_v2's GPU dependencies into the current Python environment.
#
# ⚠️  EXPERIMENTAL / UNTESTED ⚠️
# Authored on macOS — never executed end-to-end on a real CUDA host. Each
# step here mirrors a specific call in backend/modal/{inference,segmentation}.py;
# diff against those two files if anything errors.
#
# Usage:
#   # in your activated venv / conda env, with a CUDA-capable GPU + driver
#   bash scripts/install_local_gpu.sh
#
# What it does:
#   1. pip install -r backend/requirements-local-gpu.txt
#   2. git clone FlashVGGT, swap in our patched pyproject.toml, pip install it.
#
# After this, you can run:
#   python scripts/run_local_gpu.py <scene_id>

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REQ_FILE="$REPO_ROOT/backend/requirements-local-gpu.txt"
PATCH_FILE="$REPO_ROOT/patches/flashvggt_pyproject.toml"

if [[ ! -f "$REQ_FILE" ]]; then
  echo "install_local_gpu: missing $REQ_FILE" >&2
  exit 2
fi
if [[ ! -f "$PATCH_FILE" ]]; then
  echo "install_local_gpu: missing $PATCH_FILE" >&2
  exit 2
fi

# Step 1 — base requirements (torch + everything else except FlashVGGT itself).
echo "[install_local_gpu] pip install -r $REQ_FILE"
python -m pip install -r "$REQ_FILE"

# Step 2 — FlashVGGT with the patched pyproject. Upstream's pyproject is
# broken (see DESIGN_DECISIONS.md → FlashVGGT pyproject patch). We mirror
# the same three commands the Modal inference image runs at build time.
WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

echo "[install_local_gpu] cloning FlashVGGT into $WORKDIR/flashvggt"
git clone --depth 1 https://github.com/wzpscott/FlashVGGT.git "$WORKDIR/flashvggt"
cp "$PATCH_FILE" "$WORKDIR/flashvggt/pyproject.toml"

echo "[install_local_gpu] pip install $WORKDIR/flashvggt"
python -m pip install "$WORKDIR/flashvggt"

echo
echo "[install_local_gpu] DONE. Sanity-check the install:"
echo "    python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"
echo "    python -c 'import flashvggt.models.flashvggt; print(\"flashvggt OK\")'"
echo "    python -c 'from transformers import AutoModelForZeroShotObjectDetection; print(\"gdino OK\")'"
echo "    python -c 'from sam2.sam2_image_predictor import SAM2ImagePredictor; print(\"sam2 OK\")'"
echo
echo "Then run the pipeline:"
echo "    python scripts/run_local_gpu.py <scene_id>"
