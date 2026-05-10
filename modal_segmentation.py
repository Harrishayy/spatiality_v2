"""Modal container: Grounding DINO + bbox-depth lift + 3 labeling lanes.

Stage 2+ of the pipeline. Reads geometry artefacts from Stage 1 (points.ply,
cameras.json, depth/, depth_conf/, frames/) and runs:

    - Grounding DINO (`IDEA-Research/grounding-dino-base`) per-frame
      open-vocabulary detection over scout-discovered phrases
    - IoU tracklet linking — each tracklet becomes a Track directly (no
      mask propagation; the lift consumes bboxes, not masks)
    - 3D pinning per track via bbox-depth unprojection (5×5 grid sampling
      inside each frame's bbox, confidence-gated, PCA-OBB)
    - Lane B  : VLM-verified labels via orbital novel-view renders + Gemini
                (asyncio.gather, 16-way concurrent, per-track flush)
    - Lane E  : ConceptGraphs-style scene graph (objects + relations) via Gemini
    - Lane F  : SpatialLM layout (walls, doors, windows) — degrades to empty
                layout if SpatialLM is unavailable in the image.

Why no SAM 2: SAM 2's mask-propagation outputs were only consumed by an
old mask-pixel lift. The web UI never rendered masks, and bbox-center
depth unprojection gives ~5–10 cm centroid accuracy — well within the
"geometrically coherent" target. Dropping SAM 2 saves ~10 min wall-clock
per scene and an entire git+ install in the image.

Outputs three independent annotations.json variants (annotations.b.json,
annotations.e.json, annotations.f.json) that the web UI can switch between.

Auth via existing Modal Secrets:
    - `huggingface`     → HF_TOKEN (only needed if you swap to a gated model)
    - `pydantic-gateway`→ PYDANTIC_GATEWAY_KEY / PYDANTIC_GATEWAY_URL,
                          aliased at runtime onto PYDANTIC_AI_GATEWAY_API_KEY
                          for scout + Lanes B/E (model id `gateway/gemini:...`).

Run: ``modal run modal_segmentation.py::main --input-id <id>``
"""

from __future__ import annotations

from pathlib import Path

import modal

# ---------------------------------------------------------------------------- paths

REPO = Path(__file__).resolve().parent
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
    )
    # Stable torch 2.5.1 + cu124. No flash-attn-3, no nightly torch — SAM
    # 2.1-hiera-tiny works fine on vanilla scaled-dot-product attention at
    # the modest mask-call rate this pipeline uses.
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        # Pillow / opencv / scipy / scikit-* are used across the lift +
        # render pipeline. numpy<2 is forced because some transitive deps
        # still expect the 1.x ABI.
        "numpy>=1.26,<2",
        "Pillow",
        "opencv-python-headless",
        "scipy",
        # scikit-learn needed for lift's GMM front-surface filter
        # (GaussianMixture) and 3D coherence filter (DBSCAN). Both are
        # pure-Python wrappers over compiled C — fast, lightweight.
        "scikit-learn>=1.4",
        "huggingface_hub[hf_transfer]",
        # transformers >=4.50 ships AutoModelForZeroShotObjectDetection
        # (Grounding DINO) and DINOv2 image-encoder weights via AutoModel.
        "transformers>=4.50,<5",
        "tqdm",
        "regex",
        "psutil",
        "pandas",
        # PydanticAI with Gemini 2.5 Flash for scout + Lanes B/C. Auth via
        # the gateway path: PYDANTIC_AI_GATEWAY_API_KEY (aliased from the
        # `pydantic-gateway` Modal Secret at function entry).
        "pydantic-ai-slim[google]>=0.0.40",
    )
    # SAM 2.1-hiera-tiny powers the lift's mask-grade pixel grounding.
    # Installing from a pinned commit because (a) there's no PyPI release
    # with the HF `from_pretrained` API, and (b) `@main` is unstable.
    # Distribution name is `sam-2` (hyphen), the importable module is
    # `sam2` (no hyphen) — pip rejects the spec if you mix these up.
    # ~150 MB weights are pre-cached in the next run_commands step.
    .pip_install(
        "sam-2 @ git+https://github.com/facebookresearch/sam2.git"
        "@2b90b9f5ceec907a1c18123530e92e794ad901a4",
    )
    .env(
        {
            "PYTHONPATH": "/root/src",
            "SPATIALITY_DATA_ROOT": "/inputs",
            "SPATIALITY_ARTEFACTS_ROOT": "/outputs",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            # Route VLM calls (scout + Lanes B/C) through Pydantic AI Gateway
            # (the `pydantic-gateway` Modal Secret holds the gateway key, not
            # a raw Google API key). pydantic-ai's gateway accepts the
            # upstream provider name `gemini` (NOT `google-gla` — that's the
            # direct-API provider name). vlm.py honours SPATIALITY_VLM_MODEL.
            "SPATIALITY_VLM_MODEL": "gateway/gemini:gemini-2.5-flash",
        }
    )
    # Pre-download model weights at image build so cold starts skip the
    # network fetch. All public weights — no HF token required at build.
    #   GDINO base: ~700 MB
    #   SAM 2.1-hiera-tiny: ~150 MB (lift mask predictor)
    #   DINOv2-small: ~85 MB (re-ID encoder)
    .run_commands(
        "python -c 'from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, AutoModel, AutoImageProcessor; "
        "AutoProcessor.from_pretrained(\"IDEA-Research/grounding-dino-base\"); "
        "AutoModelForZeroShotObjectDetection.from_pretrained(\"IDEA-Research/grounding-dino-base\"); "
        "AutoImageProcessor.from_pretrained(\"facebook/dinov2-small\"); "
        "AutoModel.from_pretrained(\"facebook/dinov2-small\")'",
        # SAM 2.1-hiera-tiny is fetched by the sam2 package's HF integration
        # on first use; trigger that now so it's baked into the image.
        # device="cpu" because Modal image builds run on CPU — the default
        # cuda placement raises "Found no NVIDIA driver" at build time.
        # Runtime ImagePredictor construction (mask.py) still defaults to
        # cuda, so this only affects the build-time weight fetch.
        "python -c 'from sam2.sam2_image_predictor import SAM2ImagePredictor; "
        "SAM2ImagePredictor.from_pretrained(\"facebook/sam2.1-hiera-tiny\", device=\"cpu\")'",
    )
    .add_local_dir(str(SRC_DIR), remote_path="/root/src")
)


# ---------------------------------------------------------------------------- volumes + secrets

inputs_vol = modal.Volume.from_name(INPUTS_VOLUME, create_if_missing=True)
outputs_vol = modal.Volume.from_name(OUTPUTS_VOLUME, create_if_missing=True)

# Existing Modal Secrets in this workspace.
secrets = [
    # Defensive — left wired in case a future swap brings back gated weights.
    modal.Secret.from_name("huggingface"),
    modal.Secret.from_name("pydantic-gateway"),  # Gemini gateway — scout + Lanes B/C
]


# ---------------------------------------------------------------------------- app + resources

app = modal.App("spatiality-segmentation")

# A100-40GB: GDINO (Swin-B) peaks ~18 GB at batch 8 / 1024px. DINOv2-small
# adds ~0.6 GB during the re-ID embedding pass; SAM 2.1-hiera-tiny adds
# ~0.7 GB during the lift's mask predictor. The three are loaded/freed
# sequentially (not all resident at once), so peak VRAM stays bounded by
# GDINO's ~18 GB. A100-40GB is comfortable; the legacy A100-80GB is no
# longer required since SAM 2's per-frame video state was retired.
GPU_KIND = "A100-40GB"
GPU_CPU = 8
GPU_MEMORY_MB = 64 * 1024
TIMEOUT = 60 * 90

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
def run_segmentation_one(input_id: str, **kwargs) -> dict:
    """Run the full segmentation + 3-lane labeling pipeline on a single input."""
    import os

    # Bridge the `pydantic-gateway` Modal Secret's env var names onto what
    # pydantic-ai expects.
    if "PYDANTIC_GATEWAY_KEY" in os.environ and "PYDANTIC_AI_GATEWAY_API_KEY" not in os.environ:
        os.environ["PYDANTIC_AI_GATEWAY_API_KEY"] = os.environ["PYDANTIC_GATEWAY_KEY"]

    inputs_vol.reload()
    outputs_vol.reload()

    from spatiality.segmentation import run as run_segmentation

    result = run_segmentation(input_id, **kwargs)

    outputs_vol.commit()
    return result


# ---------------------------------------------------------------------------- local entrypoints


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

    Each pull lands in
    ``backend/data/outputs/<input_id>_<timestamp>/`` so prior runs at
    ``<input_id>/`` (or earlier timestamped dirs) are never overwritten.
    """
    dst_root = _fresh_local_dir(input_id)

    def _walk(remote_dir: str):
        for entry in outputs_vol.iterdir(remote_dir):
            if getattr(entry, "type", None) and int(entry.type) == 2:
                yield from _walk(entry.path)
            else:
                yield entry.path

    try:
        files = list(_walk(f"/{input_id}"))
    except FileNotFoundError:
        print(f"[pull] no remote dir /{input_id} on outputs volume", flush=True)
        return 0

    written = 0
    for remote_path in files:
        rel = remote_path.lstrip("/")[len(input_id) + 1:]
        local_path = dst_root / rel
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with local_path.open("wb") as f:
            for chunk in outputs_vol.read_file(remote_path):
                f.write(chunk)
        written += 1
    print(f"[pull] mirrored {written} file(s) → {dst_root}", flush=True)
    return written


@app.local_entrypoint()
def main(input_id: str = "", all: bool = False, lanes: str = "") -> None:
    """``modal run modal_segmentation.py::main --input-id <id>`` or ``--all``.

    ``--lanes`` accepts a comma-separated subset of ``b,e,f`` (default: all).
    """
    if all:
        raise SystemExit("--all not implemented yet; pass --input-id <id>")
    if not input_id:
        raise SystemExit("usage: modal run modal_segmentation.py::main --input-id <id>")

    kwargs: dict = {}
    if lanes:
        kwargs["lanes"] = [s.strip() for s in lanes.split(",") if s.strip()]

    result = run_segmentation_one.remote(input_id, **kwargs)
    print(result)
    _pull_outputs_to_local(input_id)
