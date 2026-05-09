"""Modal container: SAM 3.1 + lifting + 3 labeling lanes (B/E/F).

Stage 2+ of the pipeline. Reads geometry artefacts from Stage 1 (points.ply,
cameras.json, depth/, depth_conf/, frames/) and runs:

    - SAM 3.1 detection + video tracker (Object Multiplex)
    - 3D pinning per track (confidence-gated unprojection + median centroid)
    - Lane B  : VLM-verified labels via orbital novel-view renders + Gemini
    - Lane E  : ConceptGraphs-style scene graph (objects + relations) via Gemini
    - Lane F  : SpatialLM layout (walls, doors, windows) — degrades to empty
                layout if SpatialLM is unavailable in the image. SpatialLM is
                NOT installed in the default image because its poetry
                pyproject pins torch ^2.4.1+cu124 and requires manual builds
                of flash-attn / torchsparse / torch-scatter / spconv-cu120
                that don't fit a single `pip install git+...`. Lane F's
                fallback emits an empty layout payload so the rest of the
                pipeline runs to completion.

Outputs three independent annotations.json variants (annotations.b.json,
annotations.e.json, annotations.f.json) that the web UI can switch between.

Auth via existing Modal Secrets:
    - `huggingface`     → HF_TOKEN (gated SAM 3.1 weights)
    - `pydantic-gateway`→ PYDANTIC_GATEWAY_KEY / PYDANTIC_GATEWAY_URL,
                          aliased at runtime onto PYDANTIC_AI_GATEWAY_API_KEY
                          for Lanes B/E (model id `gateway/google-gla:...`).

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
    # Torch first (CUDA wheels), so SAM 3 doesn't pull a different torch.
    # SAM 3.1 imports `flash_attn_interface` (flash-attn v3) unconditionally
    # at inference time. The prebuilt flash-attn-3 wheel on PyTorch's cu128
    # index links against very-recent torch C symbols (e.g.
    # `aoti_torch_create_device_guard`) that aren't in stable torch 2.7 OR
    # 2.8 — both produced `undefined symbol` ImportErrors on every
    # `add_prompt`. Use the cu128 NIGHTLY index for torch + torchvision so
    # the C++ ABI matches what flash-attn-3 was compiled against.
    .pip_install(
        "torch",
        "torchvision",
        pre=True,
        index_url="https://download.pytorch.org/whl/nightly/cu128",
    )
    .pip_install(
        # SAM 3 pyproject requires numpy>=1.26,<2.
        "numpy>=1.26,<2",
        "Pillow",
        "opencv-python-headless",
        "scipy",
        "scikit-learn",
        "huggingface_hub[hf_transfer]",
        # Cap below 4.47 — SAM 3 was tested against 4.45/4.46 and SpatialLM
        # (if added later) caps at 4.46.1.
        "transformers>=4.45,<4.47",
        "tqdm",
        # PydanticAI with Gemini 2.5 Flash / Flash-Lite for Lanes B + E.
        # The `[google]` extra pulls google-genai. Auth via the gateway path:
        # PYDANTIC_AI_GATEWAY_API_KEY (aliased from the pydantic-gateway secret).
        "pydantic-ai-slim[google]>=0.0.40",
        # Open-vocab SigLIP for cross-frame stitching (ConceptGraphs-style
        # merge_visual_sim_thresh=0.8 semantics).
        "open_clip_torch",
    )
    # SAM 3 runtime deps that upstream's pyproject.toml omits. Audited from
    # `find sam3 -name "*.py" -exec cat {} + | grep -E "^(import|from)"` —
    # extracted unique top-level packages, dropped what's already in the
    # base image. setuptools<81 keeps legacy `pkg_resources`.
    .pip_install(
        "setuptools<81",     # provides pkg_resources for sam3/model_builder.py
        "einops",            # sam3/sam/rope.py:15 — module-level import
        "hydra-core",        # sam3/model_builder.py — instantiate(), compose()
        "omegaconf",         # transitive via hydra-core, also direct imports
        "matplotlib",        # sam3/visualization_utils.py
        "scikit-image",      # sam3/agent/* color conversions
        "ftfy",              # sam3 text encoder
        "regex",             # tokenizer
        "pycocotools",       # train/data/coco_json_loaders.py — pulled by tracker_base
        "submitit",          # train/* — also pulled by tracker_base transitively
        "psutil",            # system info, used by some sam3 utils
        "pandas",            # used by sam3 eval / agent paths
    )
    # SAM 3.1 (Object Multiplex, 7× speedup at 128 objects on H100).
    .pip_install("git+https://github.com/facebookresearch/sam3.git@main")
    # flash-attn v3 — required at inference time. SAM 3.1's tracker imports
    # `flash_attn_interface` unconditionally (every `add_prompt` raises
    # `ModuleNotFoundError: flash_attn_interface` without it, so all text
    # prompts return 0 detections). Install the prebuilt cu128 wheel per
    # SAM 3's README rather than building from source. `--no-deps` skips a
    # transitive torch reinstall that would clobber our pinned 2.7.0.
    .pip_install(
        "flash-attn-3",
        index_url="https://download.pytorch.org/whl/cu128",
        extra_options="--no-deps",
    )
    # NOTE: SpatialLM (Lane F) is intentionally NOT pip-installed here.
    # Its poetry pyproject pins torch ^2.4.1+cu124 from a custom index AND
    # requires manual builds of flash-attn / torchsparse / torch-scatter /
    # spconv-cu120 via poetry poe-tasks. None of that fits a single
    # `pip install git+...`. Lane F's import is wrapped in try/except and
    # emits an empty layout payload when unavailable — see lane_f.py:86-90.
    # If you want SpatialLM later, build a separate image stage with the
    # custom CUDA wheels.
    .env(
        {
            "PYTHONPATH": "/root/src",
            "SPATIALITY_DATA_ROOT": "/inputs",
            "SPATIALITY_ARTEFACTS_ROOT": "/outputs",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            # Route Lanes B/E through Pydantic AI Gateway (the `pydantic-gateway`
            # Modal Secret holds the gateway key, not a raw Google API key).
            # vlm.py honours SPATIALITY_VLM_MODEL — the `gateway/` prefix tells
            # pydantic-ai to proxy through gateway-us.pydantic.dev.
            "SPATIALITY_VLM_MODEL": "gateway/google-gla:gemini-2.5-flash",
        }
    )
    # Pre-download model weights at image build time so cold starts skip
    # the ~1 GB SAM 3.1 + ~360 MB SigLIP fetches. Lives in the image layer,
    # cached forever until the corresponding pip_install line changes. The
    # huggingface secret provides HF_TOKEN for the gated SAM 3.1 repo.
    #
    # `gpu=` is required because `build_sam3_predictor` calls
    # `torch.cuda.get_device_properties(0)` at import (sam3_multiplex_base.py
    # line 36) — the build container needs a real CUDA device just to get
    # past the import. open_clip is happy on CPU. The GPU cost is ~$0.05
    # for this one-shot step and the resulting layer is cached forever.
    .run_commands(
        "python -c 'from sam3.model_builder import build_sam3_predictor; build_sam3_predictor(version=\"sam3.1\")'",
        "python -c 'import open_clip; open_clip.create_model_and_transforms(\"ViT-B-16-SigLIP\", pretrained=\"webli\")'",
        secrets=[modal.Secret.from_name("huggingface")],
        gpu="A10G",
    )
    .add_local_dir(str(SRC_DIR), remote_path="/root/src")
)


# ---------------------------------------------------------------------------- volumes + secrets

inputs_vol = modal.Volume.from_name(INPUTS_VOLUME, create_if_missing=True)
outputs_vol = modal.Volume.from_name(OUTPUTS_VOLUME, create_if_missing=True)

# Existing Modal Secrets in this workspace.
secrets = [
    modal.Secret.from_name("huggingface"),       # HF_TOKEN — gated SAM 3.1 weights
    modal.Secret.from_name("pydantic-gateway"),  # GEMINI_API_KEY — Lanes B/E
]


# ---------------------------------------------------------------------------- app + resources

app = modal.App("spatiality-segmentation")

# A100-80GB. SAM 3.1's `sam3/perflib/fa3.py` upcasts q/k/v to FP8
# (Hopper-only) before flash-attn-3, which is the real cause of the
# "Ampere only supports fp16/bf16" wall. We monkey-patch
# `sam3.perflib.fa3.flash_attn_func` in `sam3.py::_patch_fa3_for_ampere`
# to cast to bf16 on Ampere, so A100 works without a fork. H100 stays
# untouched (keeps FP8) — set GPU_KIND = "H100" if you want the official
# Hopper path back.
GPU_KIND = "A100-80GB"
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
def probe_pydantic_ai() -> dict:
    """Inspect installed pydantic-ai version + gateway-routing API surface."""
    import os
    import importlib.metadata as _md
    import pkgutil
    import traceback

    out: dict = {}

    # 1. Versions.
    for pkg in ("pydantic-ai", "pydantic-ai-slim", "google-genai"):
        try:
            out[f"version_{pkg}"] = _md.version(pkg)
        except Exception as e:  # noqa: BLE001
            out[f"version_{pkg}"] = f"missing: {e}"

    # 2. Env: do the gateway secrets actually land?
    out["env_PYDANTIC_GATEWAY_KEY"] = "set" if os.environ.get("PYDANTIC_GATEWAY_KEY") else "MISSING"
    out["env_PYDANTIC_AI_GATEWAY_API_KEY"] = "set" if os.environ.get("PYDANTIC_AI_GATEWAY_API_KEY") else "MISSING"
    out["env_PYDANTIC_GATEWAY_URL"] = os.environ.get("PYDANTIC_GATEWAY_URL", "MISSING")
    out["env_PYDANTIC_AI_GATEWAY_BASE_URL"] = os.environ.get("PYDANTIC_AI_GATEWAY_BASE_URL", "MISSING")
    out["env_GOOGLE_API_KEY"] = "set" if os.environ.get("GOOGLE_API_KEY") else "MISSING"
    out["env_GEMINI_API_KEY"] = "set" if os.environ.get("GEMINI_API_KEY") else "MISSING"
    out["env_SPATIALITY_VLM_MODEL"] = os.environ.get("SPATIALITY_VLM_MODEL", "MISSING")

    # 3. Bridge step (mirrors run_segmentation_one) — does it land the alias?
    if "PYDANTIC_GATEWAY_KEY" in os.environ and "PYDANTIC_AI_GATEWAY_API_KEY" not in os.environ:
        os.environ["PYDANTIC_AI_GATEWAY_API_KEY"] = os.environ["PYDANTIC_GATEWAY_KEY"]
    out["env_after_bridge_PYDANTIC_AI_GATEWAY_API_KEY"] = (
        "set" if os.environ.get("PYDANTIC_AI_GATEWAY_API_KEY") else "MISSING"
    )

    # 4. Walk pydantic_ai for anything Gateway-related.
    try:
        import pydantic_ai
        out["pydantic_ai_path"] = pydantic_ai.__path__[0]
        gateway_hits: list[str] = []
        for finder, modname, _ in pkgutil.walk_packages(pydantic_ai.__path__, prefix="pydantic_ai."):
            low = modname.lower()
            if "gateway" in low or "openrouter" in low or "router" in low:
                gateway_hits.append(modname)
        out["pydantic_ai_gateway_modules"] = gateway_hits
    except Exception:
        out["pydantic_ai_walk_error"] = traceback.format_exc()

    # 5. providers + models top-level surface.
    for sub in ("providers", "models"):
        try:
            mod = importlib.import_module(f"pydantic_ai.{sub}")
            out[f"pydantic_ai.{sub}_dir"] = sorted(
                a for a in dir(mod) if not a.startswith("_")
            )
        except Exception as e:  # noqa: BLE001
            out[f"pydantic_ai.{sub}_dir_error"] = f"{type(e).__name__}: {e}"

    # 6. Try each plausible gateway invocation and record outcome.
    from pydantic import BaseModel
    import numpy as np
    from PIL import Image
    import io as _io

    class _Tiny(BaseModel):
        echo: str

    def _png():
        b = _io.BytesIO()
        Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(b, format="PNG")
        return b.getvalue()

    async def _try(model_arg, **agent_kw):
        from pydantic_ai import Agent, BinaryContent
        agent = Agent(model_arg, output_type=_Tiny, **agent_kw)
        result = await agent.run([BinaryContent(data=_png(), media_type="image/png"), "say hi"])
        return result.output.echo

    import asyncio

    async def _probe():
        attempts: dict[str, str] = {}
        cases = [
            ("model=gateway/google-gla:gemini-2.5-flash (current)", "gateway/google-gla:gemini-2.5-flash", {}),
            ("model=gateway:google-gla:gemini-2.5-flash", "gateway:google-gla:gemini-2.5-flash", {}),
            ("model=google-gla:gemini-2.5-flash (no gateway)", "google-gla:gemini-2.5-flash", {}),
        ]
        for label, m, kw in cases:
            try:
                attempts[label] = f"OK: {await _try(m, **kw)}"
            except Exception as e:  # noqa: BLE001
                attempts[label] = f"{type(e).__name__}: {str(e)[:300]}"
        return attempts

    try:
        out["gateway_attempts"] = asyncio.run(_probe())
    except Exception:
        out["gateway_attempts_error"] = traceback.format_exc()

    return out


@app.function(**_FN_KW)
def probe_fa3() -> dict:
    """Temporary: reproduce the FA3 import failure SAM 3.1 hits during add_prompt."""
    import subprocess
    import traceback

    out: dict = {}

    # 1. Where does sam3 source reference `flash_attn_interface`?
    try:
        grep = subprocess.run(
            ["grep", "-rnE", r"flash_attn_interface|from flash_attn|import flash_attn",
             "/usr/local/lib/python3.12/site-packages/sam3/"],
            capture_output=True, text=True, timeout=10,
        )
        out["sam3_flash_attn_refs"] = grep.stdout.splitlines()[:40]
    except Exception as e:  # noqa: BLE001
        out["sam3_flash_attn_refs_error"] = f"{type(e).__name__}: {e}"

    # 2. Plain top-level import (this passed last time).
    try:
        import flash_attn_interface  # noqa: F401
        out["flash_attn_interface_top_level_import"] = "ok"
    except Exception:
        out["flash_attn_interface_top_level_import"] = traceback.format_exc()

    # 3. Try importing `sam3.perflib.fa3` (which our patch targets).
    try:
        import sam3.perflib.fa3 as _fa3
        out["sam3_perflib_fa3_import"] = "ok"
        out["sam3_perflib_fa3_attrs"] = sorted(
            a for a in dir(_fa3) if not a.startswith("_")
        )
        if hasattr(_fa3, "flash_attn_func"):
            out["sam3_perflib_fa3_flash_attn_func"] = repr(_fa3.flash_attn_func)[:200]
    except Exception:
        out["sam3_perflib_fa3_import"] = traceback.format_exc()

    # 4. Actually try to *call* flash_attn_func on dummy GPU tensors. If the
    #    wheel's torch ABI is broken, this is where it surfaces — possibly
    #    raising something Python re-labels as ModuleNotFoundError.
    try:
        import torch
        from flash_attn_interface import flash_attn_func  # type: ignore[import-not-found]
        q = torch.randn(1, 16, 4, 64, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, 16, 4, 64, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1, 16, 4, 64, device="cuda", dtype=torch.bfloat16)
        result = flash_attn_func(q, k, v)
        # API has changed in some FA3 versions to return (out, lse).
        if isinstance(result, tuple):
            out["flash_attn_func_call"] = f"ok (tuple of {len(result)})"
        else:
            out["flash_attn_func_call"] = f"ok (shape={tuple(result.shape)}, dtype={result.dtype})"
    except Exception:
        out["flash_attn_func_call"] = traceback.format_exc()

    # 5. Dump fa3.py source so we can see what line raises ModuleNotFoundError.
    try:
        with open("/usr/local/lib/python3.12/site-packages/sam3/perflib/fa3.py") as f:
            out["sam3_perflib_fa3_source"] = f.read()
    except Exception as e:  # noqa: BLE001
        out["sam3_perflib_fa3_source_error"] = f"{type(e).__name__}: {e}"

    # 6. Reproduce SAM 3.1's failing call exactly as run_sam3 does it.
    try:
        import sys, contextlib, tempfile
        sys.path.insert(0, "/root/src")
        import torch
        from PIL import Image
        import numpy as np
        from spatiality.segmentation.sam3 import _patch_fa3_for_ampere, _amp_ctx
        out["patch_returned_in_repro"] = _patch_fa3_for_ampere()
        from sam3.model_builder import build_sam3_predictor  # type: ignore[import-not-found]

        td = tempfile.mkdtemp()
        for i in range(3):
            Image.fromarray(
                np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            ).save(f"{td}/{i:04d}.jpg")

        pred = build_sam3_predictor(version="sam3.1")
        # Match the real init_state filter our code installs.
        import inspect as _inspect
        _orig = pred.model.init_state
        _accepted = set(_inspect.signature(_orig).parameters.keys())
        def _filtered(*a, **kw):
            kw = {k: v for k, v in kw.items() if k in _accepted}
            return _orig(*a, **kw)
        pred.model.init_state = _filtered

        with _amp_ctx():
            sess = pred.handle_request({"type": "start_session", "resource_path": td})
            sid = sess["session_id"]
            out["session_id_ok"] = bool(sid)
            try:
                resp = pred.handle_request({
                    "type": "add_prompt",
                    "session_id": sid,
                    "frame_index": 0,
                    "text": "chair",
                })
                out["add_prompt"] = "ok"
                out["add_prompt_keys"] = (
                    list(resp.keys()) if isinstance(resp, dict) else type(resp).__name__
                )
            except Exception:
                out["add_prompt"] = traceback.format_exc()
    except Exception:
        out["repro_setup"] = traceback.format_exc()

    return out


@app.function(**_FN_KW)
def run_segmentation_one(input_id: str, **kwargs) -> dict:
    """Run the full segmentation + 3-lane labeling pipeline on a single input."""
    import os

    # Bridge the `pydantic-gateway` Modal Secret's env var names onto what
    # pydantic-ai expects. Done at runtime because the secret is only injected
    # when the function executes.
    if "PYDANTIC_GATEWAY_KEY" in os.environ and "PYDANTIC_AI_GATEWAY_API_KEY" not in os.environ:
        os.environ["PYDANTIC_AI_GATEWAY_API_KEY"] = os.environ["PYDANTIC_GATEWAY_KEY"]

    inputs_vol.reload()
    outputs_vol.reload()

    from spatiality.segmentation import run as run_segmentation

    result = run_segmentation(input_id, **kwargs)

    outputs_vol.commit()
    return result


# ---------------------------------------------------------------------------- local entrypoints


# Mirror of `backend.main._pull_outputs_from_modal` / the helper in
# modal_inference.py. Pulls Modal-side artefacts into the FastAPI server's
# expected location so /scenes/<id> works whether the run was kicked off
# via the HTTP pipeline or via `modal run modal_segmentation.py::main`.
_LOCAL_OUTPUTS_ROOT = REPO / "backend" / "data" / "outputs"


def _pull_outputs_to_local(input_id: str) -> int:
    dst_root = _LOCAL_OUTPUTS_ROOT / input_id
    dst_root.mkdir(parents=True, exist_ok=True)

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
    Modal's local_entrypoint doesn't support ``**kwargs`` from the CLI, so
    each tunable kwarg is exposed as an explicit option here.
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


@app.local_entrypoint()
def probe() -> None:
    """Temporary: ``modal run modal_segmentation.py::probe`` to inspect FA3 install."""
    import json

    result = probe_fa3.remote()
    print(json.dumps(result, indent=2, default=str))
