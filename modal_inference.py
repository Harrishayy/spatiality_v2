"""Modal container: FlashVGGT geometry inference for spatiality_v2.

Stage 1 of the pipeline. Reads frames from the inputs volume, runs FlashVGGT
(with a base VGGT fallback) to produce dense per-pixel depth + per-pixel
confidence + per-frame camera intrinsics/extrinsics, and emits:

    /outputs/<input_id>/
        points.ply          # confidence-gated dense point cloud (XYZ+RGB+conf)
        cameras.json        # K, R, t per frame
        depth/<frame>.npy
        depth_conf/<frame>.npy
        manifest.json       # poses stage entry

Coordinate convention on disk: OpenCV (+y down, +z forward). The web viewer
already negates y/z while parsing, so do not pre-flip here.

Run: ``modal run modal_inference.py::main --input-id <id>``
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

# ---------------------------------------------------------------------------- paths

REPO = Path(__file__).resolve().parent
ENV_FILE = REPO / ".env"
SRC_DIR = REPO / "backend" / "src"

INPUTS_VOLUME = "spatiality-inputs"
OUTPUTS_VOLUME = "spatiality-outputs"

# Modal Secret name. Create once with the keys you have on hand:
#   modal secret create spatiality HF_TOKEN=hf_xxx GEMINI_API_KEY=AIzaxxx
# Override the default name by setting SPATIALITY_MODAL_SECRET in your shell.
MODAL_SECRET_NAME = os.environ.get("SPATIALITY_MODAL_SECRET", "spatiality")


# ---------------------------------------------------------------------------- image
#
# We try FlashVGGT first (Dec 2025, compressed-descriptor attention; ~10× faster
# at scale, recovers fine detail base VGGT misses on long sequences). Base VGGT
# is installed alongside as a fallback for short clips and as a sanity check.

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "git",
        "libgl1",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
    )
    .pip_install(
        "numpy>=1.24,<2",
        "Pillow",
        "opencv-python-headless",
        "scipy",
        "einops",
        "huggingface_hub[hf_transfer]",
        "torch==2.4.0",
        "torchvision==0.19.0",
        "plyfile",
        "trimesh",
        "tqdm",
    )
    # FlashVGGT (preferred) and base VGGT (fallback). Pulled directly from the
    # canonical repos so we always get the latest published weights/configs.
    .pip_install(
        "git+https://github.com/wzpscott/FlashVGGT.git@main",
        # Fallback. Same package surface as FlashVGGT's; both expose `vggt.models.vggt.VGGT`.
        "git+https://github.com/facebookresearch/vggt.git@main",
    )
    .env(
        {
            "PYTHONPATH": "/root/src",
            "SPATIALITY_DATA_ROOT": "/inputs",
            "SPATIALITY_ARTEFACTS_ROOT": "/outputs",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
    .add_local_dir(str(SRC_DIR), remote_path="/root/src")
)


# ---------------------------------------------------------------------------- volumes + secrets

inputs_vol = modal.Volume.from_name(INPUTS_VOLUME, create_if_missing=True)
outputs_vol = modal.Volume.from_name(OUTPUTS_VOLUME, create_if_missing=True)

secret = (
    modal.Secret.from_dotenv(path=ENV_FILE) if ENV_FILE.exists() else None
)


# ---------------------------------------------------------------------------- app + resources

app = modal.App("spatiality-inference")

# FlashVGGT scales to 1k+ frames on A100-80GB. A10G works for short clips
# (≤200 frames) at reduced chunk size; falls back automatically inside the runner.
GPU_KIND = "A100-80GB"
GPU_CPU = 8
GPU_MEMORY_MB = 64 * 1024
TIMEOUT = 60 * 60  # 1 h — long captures with FlashVGGT chunked inference

_FN_KW = dict(
    image=image,
    gpu=GPU_KIND,
    volumes={"/inputs": inputs_vol, "/outputs": outputs_vol},
    secrets=[secret] if secret else [],
    cpu=GPU_CPU,
    memory=GPU_MEMORY_MB,
    timeout=TIMEOUT,
)


# ---------------------------------------------------------------------------- remote functions


@app.function(**_FN_KW)
def run_inference_one(input_id: str, **kwargs) -> dict:
    """Run inference on a single input. Delegates to ``spatiality.inference.run``."""
    inputs_vol.reload()
    outputs_vol.reload()

    from spatiality.inference import run as run_inference

    result = run_inference(input_id, **kwargs)

    outputs_vol.commit()
    return result


# ---------------------------------------------------------------------------- local entrypoints


@app.local_entrypoint()
def main(input_id: str = "", all: bool = False) -> None:
    """``modal run modal_inference.py::main --input-id <id>`` or ``--all``."""
    if all:
        raise SystemExit("--all not implemented yet; pass --input-id <id>")
    if not input_id:
        raise SystemExit("usage: modal run modal_inference.py::main --input-id <id>")

    result = run_inference_one.remote(input_id)
    print(result)
