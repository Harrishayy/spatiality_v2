"""Modal container: SAM 3.1 + lifting + 3 labeling lanes (B/E/F).

Stage 2+ of the pipeline. Reads geometry artefacts from Stage 1 (points.ply,
cameras.json, depth/, depth_conf/, frames/) and runs:

    - SAM 3.1 detection + video tracker (Object Multiplex)
    - 3D pinning per track (confidence-gated unprojection + median centroid)
    - Lane B  : VLM-verified labels via orbital novel-view renders + Claude
    - Lane E  : ConceptGraphs-style scene graph (objects + relations)
    - Lane F  : SpatialLM layout (walls, doors, windows)

Outputs three independent annotations.json variants (annotations.b.json,
annotations.e.json, annotations.f.json) that the web UI can switch between.

Requires the ``HF_TOKEN`` and ``ANTHROPIC_API_KEY`` env vars (from .env).

Run: ``modal run modal_segmentation.py::main --input-id <id>``
"""

from __future__ import annotations

from pathlib import Path

import modal

# ---------------------------------------------------------------------------- paths

REPO = Path(__file__).resolve().parent
ENV_FILE = REPO / ".env"
SRC_DIR = REPO / "backend" / "src"

INPUTS_VOLUME = "spatiality-inputs"
OUTPUTS_VOLUME = "spatiality-outputs"


# ---------------------------------------------------------------------------- image

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "git",
        "libgl1",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "ffmpeg",
    )
    .pip_install(
        "numpy>=1.24,<2",
        "Pillow",
        "opencv-python-headless",
        "scipy",
        "scikit-learn",
        "huggingface_hub[hf_transfer]",
        "torch==2.4.0",
        "torchvision==0.19.0",
        "transformers>=4.45",
        "plyfile",
        "trimesh",
        "tqdm",
        # Lanes B + E use PydanticAI with Gemini 2.5 Flash / Flash-Lite for
        # structured-output VLM calls. The `[google]` extra pulls the
        # google-genai client. Auth via GEMINI_API_KEY in .env.
        "pydantic-ai-slim[google]>=0.0.20",
        # Open-vocab SigLIP for cross-frame stitching (matches
        # ConceptGraphs-style merge_visual_sim_thresh=0.8 semantics).
        "open_clip_torch",
    )
    # SAM 3.1 (Object Multiplex, 7× speedup at 128 objects on H100).
    # Drop-in over SAM 3 — same predictor classes, new weights via HF.
    .pip_install("git+https://github.com/facebookresearch/sam3.git@main")
    # SpatialLM (NeurIPS'25) for Lane F layout (walls/doors/windows).
    .pip_install("git+https://github.com/manycore-research/SpatialLM.git@main")
    # Point-cloud rendering for orbital novel views in Lane B.
    .pip_install("open3d>=0.18", "pyrender>=0.1.45")
    .env(
        {
            "PYTHONPATH": "/root/src",
            "SPATIALITY_DATA_ROOT": "/inputs",
            "SPATIALITY_ARTEFACTS_ROOT": "/outputs",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYOPENGL_PLATFORM": "egl",  # offscreen render for orbital views
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

app = modal.App("spatiality-segmentation")

# A100-80GB headroom for SAM 3.1 (Object Multiplex on long clips) + SpatialLM
# inference + concurrent VLM I/O. SAM 3.1 reports best on H100/H200; A100 is
# the practical Modal default and still much faster than SAM 3.
GPU_KIND = "A100-80GB"
GPU_CPU = 8
GPU_MEMORY_MB = 64 * 1024
TIMEOUT = 60 * 90

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
def run_segmentation_one(input_id: str, **kwargs) -> dict:
    """Run the full segmentation + 3-lane labeling pipeline on a single input."""
    inputs_vol.reload()
    outputs_vol.reload()

    from spatiality.segmentation import run as run_segmentation

    result = run_segmentation(input_id, **kwargs)

    outputs_vol.commit()
    return result


# ---------------------------------------------------------------------------- local entrypoints


@app.local_entrypoint()
def main(input_id: str = "", all: bool = False, **kwargs) -> None:
    """``modal run modal_segmentation.py::main --input-id <id>`` or ``--all``."""
    if all:
        raise SystemExit("--all not implemented yet; pass --input-id <id>")
    if not input_id:
        raise SystemExit("usage: modal run modal_segmentation.py::main --input-id <id>")

    result = run_segmentation_one.remote(input_id, **kwargs)
    print(result)
