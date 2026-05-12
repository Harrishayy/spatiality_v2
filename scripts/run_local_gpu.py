"""Run the spatiality_v2 pipeline on a local CUDA GPU — no Modal.

⚠️  EXPERIMENTAL / UNTESTED ⚠️

This script is the "bring-your-own-GPU" entry point. It was authored on macOS,
where the GPU stages physically cannot run, so it has not been smoke-tested
end-to-end on real CUDA hardware. Every choice in here — the env-var
plumbing, the per-stage import order, the way scout/Lane B keys are
bridged — is *inferred from* the working Modal setup in
``backend/modal/inference.py`` and ``backend/modal/segmentation.py``.

If something errors at runtime, treat it as config drift between your local
venv and what the Modal images install. Diff your ``pip list`` against the
``.pip_install(...)`` blocks in the two modal/*.py files; that's the
authoritative dependency manifest. The patched FlashVGGT pyproject under
``patches/`` also has to be applied during install (see
``scripts/install_local_gpu.sh``).

What this script does
---------------------
1. Points the inference + segmentation modules at local ``backend/data/``
   directories instead of Modal's volume mounts (``/inputs`` / ``/outputs``).
2. Runs ffmpeg frame extraction (re-uses the orchestrator's helper so the
   blur-prefilter oversample factor stays identical).
3. Calls ``spatiality.inference.run`` in-process (FlashVGGT geometry).
4. Calls ``spatiality.segmentation.run`` in-process (GDINO → re-ID → lift →
   Lane B → Lane C → Stage 5 free-space).
5. Writes ``manifest.json`` plus all artefacts under
   ``backend/data/outputs/<scene_id>/``, exactly where the FastAPI viewer
   reads them from.

Prerequisites — verify these yourself first
-------------------------------------------
- CUDA-capable GPU with ≥24 GB VRAM. The Modal path uses A100-80GB for
  FlashVGGT; smaller GPUs may OOM on 500-frame captures.
- ``ffmpeg`` and ``ffprobe`` on PATH (frame extraction).
- Python deps installed via ``scripts/install_local_gpu.sh`` (or by hand
  from ``backend/requirements-local-gpu.txt`` + the FlashVGGT patched
  install).
- At least one of the following env vars set for the VLM lanes:
    * ``PYDANTIC_AI_GATEWAY_API_KEY`` (recommended — single key, multiple
      providers via PydanticAI Gateway), OR
    * ``GOOGLE_API_KEY`` / ``GEMINI_API_KEY`` for the direct API path.
  The Modal setup uses Pydantic AI Gateway under the alias
  ``PYDANTIC_GATEWAY_KEY``; this script bridges that to the var
  PydanticAI itself reads.
- Optional: ``HF_TOKEN`` if you want the VGGT-1B fallback (``facebook/VGGT-1B``
  is a gated repo on Hugging Face).

Usage
-----
::

    # one-shot end-to-end
    python scripts/run_local_gpu.py <scene_id> [--frames 500]

    # re-run just segmentation (inference already done; geometry artefacts
    # exist at backend/data/outputs/<scene_id>/)
    python scripts/run_local_gpu.py <scene_id> --skip-extract --skip-inference

Inputs expected at::

    backend/data/inputs/<scene_id>/source.<mp4|mov|webm|mkv|m4v>

Outputs written to::

    backend/data/outputs/<scene_id>/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
INPUTS_ROOT = REPO / "backend" / "data" / "inputs"
OUTPUTS_ROOT = REPO / "backend" / "data" / "outputs"


def _bridge_env() -> None:
    """Replicate the env-var wiring the Modal images bake in.

    Each line here mirrors a specific decision in ``backend/modal/{inference,
    segmentation}.py``. Keep them in sync if you tune the Modal images.
    """
    # Where ``spatiality.inference.run`` and ``spatiality.segmentation.run``
    # look for input frames and write artefacts. On Modal these are the
    # mount points of the two named volumes; locally we collapse them onto
    # the backend/data tree the FastAPI orchestrator already uses.
    os.environ.setdefault("SPATIALITY_DATA_ROOT", str(INPUTS_ROOT))
    os.environ.setdefault("SPATIALITY_ARTEFACTS_ROOT", str(OUTPUTS_ROOT))

    # Speedup for big HF downloads (FlashVGGT ~5 GB, VGGT-1B ~3 GB, GDINO
    # ~700 MB). Harmless if hf_transfer isn't installed; HF falls back.
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    # FlashVGGT pulls wandb[media] transitively. Mirroring the Modal image:
    # disable wandb so a missing WANDB_API_KEY never blocks startup.
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("WANDB_MODE", "disabled")

    # PydanticAI looks for PYDANTIC_AI_GATEWAY_API_KEY when routing via the
    # gateway model id `gateway/gemini:gemini-2.5-flash`. The Modal Secret
    # in our workspace stores the same value under PYDANTIC_GATEWAY_KEY, so
    # we bridge whichever shape the user has in their shell.
    if (
        "PYDANTIC_GATEWAY_KEY" in os.environ
        and "PYDANTIC_AI_GATEWAY_API_KEY" not in os.environ
    ):
        os.environ["PYDANTIC_AI_GATEWAY_API_KEY"] = os.environ["PYDANTIC_GATEWAY_KEY"]

    # Pick the same default VLM model id the Modal segmentation image uses.
    # Users can override with their own SPATIALITY_VLM_MODEL.
    os.environ.setdefault("SPATIALITY_VLM_MODEL", "gateway/gemini:gemini-2.5-flash")


def _seed_manifest(scene_id: str) -> Path:
    """Write the initial manifest the FastAPI viewer polls.

    Identical schema to ``backend.main._seed_manifest``. We duplicate it
    here so the local-GPU path doesn't depend on FastAPI being importable
    (which it always is in this repo, but importing it just to seed a
    JSON file is heavier than needed).
    """
    from datetime import datetime, timezone

    out_dir = OUTPUTS_ROOT / scene_id
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = {
        "scene_id": scene_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "processing",
        "stages": {
            "capture": {"status": "complete"},
            "poses": {"status": "pending"},
            "splat": {"status": "complete"},
            "segmentation": {"status": "pending"},
        },
        "artifacts": {},
        "stats": {"frame_count": 0, "object_count": 0, "splat_size_mb": 0.0},
    }
    mpath = out_dir / "manifest.json"
    mpath.write_text(json.dumps(seed, indent=2))
    return mpath


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the spatiality_v2 pipeline on a local CUDA GPU "
        "(no Modal). Experimental — see the module docstring.",
    )
    parser.add_argument("scene_id", help="Scene id; videos at backend/data/inputs/<id>/source.<ext>")
    parser.add_argument("--frames", type=int, default=500,
                        help="Target frame count POST blur-filter (default 500).")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip ffmpeg frame extraction (frames/*.png already on disk).")
    parser.add_argument("--skip-inference", action="store_true",
                        help="Skip Stage 1; assumes points.ply + depth/ already exist.")
    parser.add_argument("--skip-segmentation", action="store_true",
                        help="Skip Stages 2–5 (geometry only).")
    args = parser.parse_args()

    scene_id: str = args.scene_id
    in_dir = INPUTS_ROOT / scene_id

    if not in_dir.exists():
        print(f"error: no input directory at {in_dir}", file=sys.stderr)
        print(
            f"hint: drop your video at {in_dir}/source.mp4 and re-run.",
            file=sys.stderr,
        )
        return 2

    if not args.skip_extract:
        sources = list(in_dir.glob("source.*"))
        sources = [p for p in sources if p.is_file() and not p.name.endswith(".part")]
        if not sources:
            print(f"error: no source.* file in {in_dir}", file=sys.stderr)
            return 2

    _bridge_env()

    # Make ``spatiality.*`` importable. Mirrors what
    # ``add_local_dir(SRC_DIR, remote_path='/root/src')`` + ``PYTHONPATH=/root/src``
    # achieves inside the Modal container.
    sys.path.insert(0, str(REPO / "backend" / "src"))
    sys.path.insert(0, str(REPO))

    t0 = time.time()
    print(f"[local-gpu] scene_id={scene_id} frames={args.frames}", flush=True)
    print(f"[local-gpu] SPATIALITY_DATA_ROOT={os.environ['SPATIALITY_DATA_ROOT']}", flush=True)
    print(f"[local-gpu] SPATIALITY_ARTEFACTS_ROOT={os.environ['SPATIALITY_ARTEFACTS_ROOT']}", flush=True)

    _seed_manifest(scene_id)

    if not args.skip_extract:
        # Re-use the orchestrator's helper so the 1.30× oversample factor
        # for the blur prefilter is identical to the Modal path.
        from backend.main import _extract_frames  # noqa: E402

        print(f"[local-gpu] extracting frames (target {args.frames}) …", flush=True)
        info = _extract_frames(scene_id, args.frames)
        print(
            f"[local-gpu]   frames={info['frame_count']} "
            f"duration={info['duration_s']:.1f}s",
            flush=True,
        )

    if not args.skip_inference:
        print("[local-gpu] === Stage 1: FlashVGGT geometry ===", flush=True)
        t_inf = time.time()
        # In-process call — same entry point Modal's run_inference_one uses
        # via ``Function.from_name``.
        from spatiality.inference import run as run_inference  # noqa: E402

        res = run_inference(scene_id)
        print(
            f"[local-gpu] inference done in {time.time()-t_inf:.1f}s: {res}",
            flush=True,
        )
    else:
        print("[local-gpu] (Stage 1 skipped)", flush=True)

    if not args.skip_segmentation:
        print("[local-gpu] === Stages 2–5: segmentation + free-space ===", flush=True)
        t_seg = time.time()
        from spatiality.segmentation import run as run_segmentation  # noqa: E402

        res = run_segmentation(scene_id)
        print(
            f"[local-gpu] segmentation done in {time.time()-t_seg:.1f}s: {res}",
            flush=True,
        )
    else:
        print("[local-gpu] (segmentation skipped)", flush=True)

    out_dir = OUTPUTS_ROOT / scene_id
    print(f"[local-gpu] DONE in {time.time()-t0:.1f}s. Artefacts under {out_dir}:", flush=True)
    for p in sorted(out_dir.iterdir()):
        if p.is_file():
            print(f"  {p.name:32s} {p.stat().st_size/1e6:8.2f} MB", flush=True)
        else:
            print(f"  {p.name}/", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
