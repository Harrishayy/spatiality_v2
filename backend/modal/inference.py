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

Run: ``modal run backend/modal/inference.py::main --input-id <id>``
"""

from __future__ import annotations

from pathlib import Path

import modal

# ---------------------------------------------------------------------------- paths

# This file lives at <repo>/backend/modal/inference.py — parents[2] is the
# repo root that contains backend/, patches/, web/, …
REPO = Path(__file__).resolve().parents[2]
SRC_DIR = REPO / "backend" / "src"

INPUTS_VOLUME = "spatiality-inputs"
OUTPUTS_VOLUME = "spatiality-outputs"


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
    # Torch first (CUDA wheels), then the slim base, then FlashVGGT/VGGT which
    # re-use the already-installed torch and pull only their loose deps.
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
    )
    .pip_install(
        # SAM 2.1 (in segmentation image) requires numpy>=1.26 — keeping the
        # same floor here so artefacts produced by inference round-trip cleanly.
        "numpy>=1.26,<2",
        "Pillow",
        "opencv-python-headless",
        "scipy",
        "einops",
        "huggingface_hub[hf_transfer]",
        "tqdm",
    )
    # Base VGGT — clean pyproject, installs straight from git.
    .pip_install("git+https://github.com/facebookresearch/vggt.git@main")
    # FlashVGGT runtime deps that upstream's pyproject.toml omits.
    # Audited from the actual source tree:
    #   - torch_kmeans     module-level import in flashvggt/models/aggregator.py
    #   - kornia, iopath,  listed in requirements.txt; imported by
    #     fvcore, wcmatch  flashvggt/dependency/* modules (not always in our
    #                      import chain, but cheap insurance)
    #   - pycolmap         pose-refinement helpers in dependency/
    #   - lightglue        feature matching; not on PyPI, install from git
    .pip_install(
        "torch_kmeans",
        "kornia",
        "iopath",
        "fvcore",
        "wcmatch",
        "pycolmap",
    )
    .pip_install("git+https://github.com/cvg/LightGlue.git@main")
    # FlashVGGT — upstream pyproject is broken: `include` uses non-glob
    # names so `flashvggt.models`, `flashvggt.utils`, etc. never get
    # installed, AND there are no `__init__.py` files so namespace-package
    # discovery is required. We swap in a corrected pyproject.toml from
    # ./patches/ and install from the patched source tree.
    .add_local_file(
        str(REPO / "patches" / "flashvggt_pyproject.toml"),
        remote_path="/tmp/flashvggt_pyproject.toml",
        copy=True,
    )
    .run_commands(
        "git clone --depth 1 https://github.com/wzpscott/FlashVGGT.git /tmp/flashvggt",
        "cp /tmp/flashvggt_pyproject.toml /tmp/flashvggt/pyproject.toml",
        "pip install /tmp/flashvggt",
    )
    .env(
        {
            "PYTHONPATH": "/root/src",
            "SPATIALITY_DATA_ROOT": "/inputs",
            "SPATIALITY_ARTEFACTS_ROOT": "/outputs",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            # FlashVGGT pulls wandb[media] transitively; disable so a missing
            # WANDB_API_KEY never blocks startup.
            "WANDB_DISABLED": "true",
            "WANDB_MODE": "disabled",
        }
    )
    # Pre-download model weights at image build time so cold starts skip
    # the FlashVGGT (~5 GB) + base VGGT (~3 GB) HF fetches. The `huggingface`
    # secret provides HF_TOKEN for facebook/VGGT-1B (gated). ZipW/FlashVGGT
    # is public so it'd download without the secret too.
    .run_commands(
        "python -c 'from huggingface_hub import hf_hub_download; hf_hub_download(repo_id=\"ZipW/FlashVGGT\", filename=\"flashvggt.pt\")'",
        "python -c 'from vggt.models.vggt import VGGT; VGGT.from_pretrained(\"facebook/VGGT-1B\")'",
        secrets=[modal.Secret.from_name("huggingface")],
    )
    .add_local_dir(str(SRC_DIR), remote_path="/root/src")
)


# ---------------------------------------------------------------------------- volumes + secrets

inputs_vol = modal.Volume.from_name(INPUTS_VOLUME, create_if_missing=True)
outputs_vol = modal.Volume.from_name(OUTPUTS_VOLUME, create_if_missing=True)

# Existing Modal Secrets in this workspace. `huggingface` exposes HF_TOKEN
# (gated SAM3.1 / VGGT weights). Inference doesn't need pydantic-gateway.
secrets = [modal.Secret.from_name("huggingface")]


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
    secrets=secrets,
    cpu=GPU_CPU,
    memory=GPU_MEMORY_MB,
    timeout=TIMEOUT,
)


# ---------------------------------------------------------------------------- remote functions


@app.function(**_FN_KW)
def run_inference_one(input_id: str, **kwargs) -> dict:
    """Run inference on a single input. Delegates to ``spatiality.inference.run``.

    Always runs a SINGLE forward pass over the full sequence — no chunking.
    Chunked VGGT/FlashVGGT solves are chunk-local (each chunk's first frame
    is pinned at the world origin), and naive concatenation of those windows
    produces N disjoint reconstructions overlapping at the origin. FlashVGGT
    handles 500+ frames at 518×518 in a single forward on A100-80GB
    (~245s observed on IMG_7531). If a sequence is too large for one
    forward, raise the GPU class — don't chunk.
    """
    inputs_vol.reload()
    outputs_vol.reload()

    from spatiality.inference import run as run_inference

    result = run_inference(input_id, **kwargs)

    outputs_vol.commit()
    return result


# ---------------------------------------------------------------------------- local entrypoints


# Where the FastAPI server (backend/main.py) reads scene artifacts from. We
# mirror the Modal outputs volume here so /scenes/<id> works the same whether
# the run was kicked off via POST /api/jobs (which already pulls) or via
# `modal run backend/modal/inference.py::main` (which previously left the
# data on the remote volume and required a manual `modal volume get`).
_LOCAL_OUTPUTS_ROOT = REPO / "backend" / "data" / "outputs"


def _fresh_local_dir(input_id: str) -> Path:
    """Return a brand-new local sibling directory for this run's pull.

    Format: ``backend/data/outputs/<input_id>_<YYYY-MM-DD_HH-MM-SS>/``.
    Each pull creates its own dir so prior data is never touched.
    """
    from datetime import datetime
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    candidate = _LOCAL_OUTPUTS_ROOT / f"{input_id}_{stamp}"
    suffix = 1
    while candidate.exists():
        candidate = _LOCAL_OUTPUTS_ROOT / f"{input_id}_{stamp}_{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def _pull_outputs_to_local(input_id: str) -> int:
    """Mirror the remote scene outputs to a fresh sibling directory.

    Shells out to ``modal volume get``, which downloads in parallel with
    checksums. The previous serial loop over ``volume.read_file`` was
    bottlenecked on per-file roundtrip latency for the 1500+ small depth /
    frame artefacts produced by Stage 1.
    """
    import shutil
    import subprocess

    dst_root = _fresh_local_dir(input_id)
    cmd = [
        "modal", "volume", "get", "--force",
        OUTPUTS_VOLUME, f"/{input_id}", str(dst_root),
    ]
    print(f"[pull] $ {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"[pull] modal volume get exited rc={rc}", flush=True)
        return 0

    # `modal volume get` puts contents under <dst_root>/<input_id>/ — flatten
    # so the local layout matches the remote one (frames/, depth/, etc. live
    # directly under dst_root, the way callers expect).
    nested = dst_root / input_id
    if nested.is_dir():
        for child in nested.iterdir():
            shutil.move(str(child), str(dst_root / child.name))
        nested.rmdir()

    written = sum(1 for _ in dst_root.rglob("*") if _.is_file())
    print(f"[pull] mirrored {written} file(s) → {dst_root}", flush=True)
    return written


@app.local_entrypoint()
def main(input_id: str = "", all: bool = False) -> None:
    """``modal run backend/modal/inference.py::main --input-id <id>`` or ``--all``."""
    if all:
        raise SystemExit("--all not implemented yet; pass --input-id <id>")
    if not input_id:
        raise SystemExit("usage: modal run backend/modal/inference.py::main --input-id <id>")

    result = run_inference_one.remote(input_id)
    print(result)
    _pull_outputs_to_local(input_id)
