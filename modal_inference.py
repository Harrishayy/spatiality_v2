"""Modal scaffold: inference container for spatiality_v2.

Template only — model choice, dependencies, and the actual inference logic
are deliberately left as TODOs. Fill them in once the model is picked.

Setup (one-time)
----------------
1. ``pip install modal`` (or ``uv pip install modal``)
2. ``modal token set`` — authenticate the CLI against your account.
3. Volumes auto-create on first ``modal run``; or pre-create explicitly:
       modal volume create spatiality-inputs
       modal volume create spatiality-outputs
4. Upload inputs once they exist:
       modal volume put spatiality-inputs ./data /

Running
-------
    modal run modal_inference.py::main --input-id <id>
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

# TODO: pin the model-specific dependencies here once chosen.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "libgl1",
        "libglib2.0-0",
    )
    .pip_install(
        "numpy",
        "Pillow",
        # TODO: torch / transformers / model-specific libs
    )
    .env(
        {
            "PYTHONPATH": "/root/src",
            "SPATIALITY_DATA_ROOT": "/inputs",
            "SPATIALITY_ARTEFACTS_ROOT": "/outputs",
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

app = modal.App("spatiality-inference")

# TODO: revisit once the model + memory footprint is known.
GPU_KIND = "A10G"
GPU_CPU = 4
GPU_MEMORY_MB = 16 * 1024
TIMEOUT = 60 * 30

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
        # TODO: enumerate inputs locally (or in-container) and fan out via .spawn().
        raise SystemExit("--all not implemented yet; pass --input-id <id>")
    if not input_id:
        raise SystemExit("usage: modal run modal_inference.py::main --input-id <id>")

    result = run_inference_one.remote(input_id)
    print(result)
