"""Modal scaffold: segmentation container for spatiality_v2.

Template only — segmentation model, dependencies, and the actual logic are
deliberately left as TODOs. Fill them in once the model is picked.

Kept separate from ``inference.py`` so the segmentation image (typically heavy:
torch + vision encoder + checkpoints) only spins up when needed, and the two
paths can scale independently.

Setup (one-time)
----------------
1. ``pip install modal`` (or ``uv pip install modal``)
2. ``modal token set``
3. Volumes auto-create on first ``modal run``; or pre-create explicitly:
       modal volume create spatiality-inputs
       modal volume create spatiality-outputs
4. If the model is gated on Hugging Face, drop ``HF_TOKEN=hf_...`` into ``.env``
   — it'll be picked up by the secret below.

Running
-------
    modal run modal_segmentation.py::main --input-id <id>
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

# TODO: pin the segmentation-model dependencies here once chosen.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "libgl1",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
    )
    .pip_install(
        "numpy",
        "Pillow",
        # TODO: torch / torchvision / transformers / model checkpoints
    )
    .env(
        {
            "PYTHONPATH": "/root/src",
            "SPATIALITY_DATA_ROOT": "/inputs",
            "SPATIALITY_ARTEFACTS_ROOT": "/outputs",
            # Faster HF downloads on cold start (no-op if hf_transfer not installed).
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
    # Mount last so code edits don't bust the image cache.
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

# TODO: revisit once the model + batch size is known.
GPU_KIND = "A10G"
GPU_CPU = 8
GPU_MEMORY_MB = 32 * 1024
TIMEOUT = 60 * 60

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
    """Run segmentation on a single input. Delegates to ``spatiality.segmentation.run``."""
    inputs_vol.reload()
    outputs_vol.reload()

    from spatiality.segmentation import run as run_segmentation

    result = run_segmentation(input_id, **kwargs)

    outputs_vol.commit()
    return result


# ---------------------------------------------------------------------------- local entrypoints


@app.local_entrypoint()
def main(input_id: str = "", all: bool = False) -> None:
    """``modal run modal_segmentation.py::main --input-id <id>`` or ``--all``."""
    if all:
        # TODO: enumerate inputs and fan out via .spawn().
        raise SystemExit("--all not implemented yet; pass --input-id <id>")
    if not input_id:
        raise SystemExit("usage: modal run modal_segmentation.py::main --input-id <id>")

    result = run_segmentation_one.remote(input_id)
    print(result)
