"""Local FastAPI backend for spatiality_v2.

Replaces the previous Modal-hosted ``spatiality-api`` app. Everything that
isn't GPU work (HTTP endpoints, manifest writes, ffmpeg frame extraction,
artifact streaming) runs on the laptop. Only the two GPU stages —
``spatiality-inference`` and ``spatiality-segmentation`` — still execute on
Modal, invoked cross-app via ``Function.from_name(...).remote(...)``.

URL shapes are unchanged from the old Modal API, so the existing Next.js
client in ``web/app/lib/api.ts`` works as-is once
``next.config.mjs`` rewrites point at ``http://localhost:8765``.

Layout on disk (sibling to this file):

    backend/data/
        inputs/<scene_id>/
            source.<ext>            uploaded video
            frames/0001.png …       ffmpeg-extracted frames
        outputs/<scene_id>/
            manifest.json           pipeline state
            points.ply, frames/, masks/, annotations.*.json, …

Run:
    uvicorn backend.main:app --host 0.0.0.0 --port 8765 --reload
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse


# ---------------------------------------------------------------------------- paths

BACKEND_DIR = Path(__file__).resolve().parent
DATA_ROOT = BACKEND_DIR / "data"
INPUTS_ROOT = DATA_ROOT / "inputs"
OUTPUTS_ROOT = DATA_ROOT / "outputs"

INPUTS_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)

DEFAULT_FRAMES = 500  # target frames AFTER preprocessing — what FlashVGGT actually sees

# Inference's blur filter (inference/run.py) drops the bottom ~20% by Laplacian
# variance before FlashVGGT. To land at exactly DEFAULT_FRAMES going into the
# pose head, ffmpeg oversamples by this factor; the even-spacing cap inside
# select_frames trims any surplus to DEFAULT_FRAMES exactly. 1.30 = 1/0.8 + 5%
# safety so a slightly under-shooting ffmpeg still leaves ≥DEFAULT_FRAMES.
_EXTRACT_OVERSAMPLE = 1.30

# Modal volume names — must match modal_inference.py / modal_segmentation.py.
INPUTS_VOLUME = "spatiality-inputs"
OUTPUTS_VOLUME = "spatiality-outputs"


# ---------------------------------------------------------------------------- helpers

def _scene_input_dir(scene_id: str) -> Path:
    if not scene_id or "/" in scene_id or ".." in scene_id:
        raise ValueError(f"invalid scene_id: {scene_id!r}")
    return INPUTS_ROOT / scene_id


def _scene_output_dir(scene_id: str) -> Path:
    if not scene_id or "/" in scene_id or ".." in scene_id:
        raise ValueError(f"invalid scene_id: {scene_id!r}")
    return OUTPUTS_ROOT / scene_id


def _safe_artifact_path(scene_id: str, rel_path: str) -> Path:
    base = _scene_output_dir(scene_id).resolve()
    target = (base / rel_path).resolve()
    if base != target and base not in target.parents:
        raise ValueError(f"path escapes scene dir: {rel_path!r}")
    return target


def _ffprobe_duration_seconds(video: Path) -> float:
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video),
        ],
        text=True,
    ).strip()
    return float(out) if out else 0.0


def _extract_frames(scene_id: str, n_frames: int = DEFAULT_FRAMES) -> dict:
    """Even-cadence ffmpeg split. Mirrors the old modal_api.extract_frames."""
    scene_dir = _scene_input_dir(scene_id)
    if not scene_dir.exists():
        raise FileNotFoundError(f"no scene dir at {scene_dir}")

    sources = sorted(scene_dir.glob("source.*"))
    sources = [p for p in sources if p.is_file() and not p.name.endswith(".part")]
    if not sources:
        raise FileNotFoundError(f"no source.* file in {scene_dir}")
    video = sources[0]

    frames_dir = scene_dir / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)

    duration = _ffprobe_duration_seconds(video)
    if duration <= 0:
        raise RuntimeError(f"ffprobe returned non-positive duration: {duration}")
    extract_count = int(round(n_frames * _EXTRACT_OVERSAMPLE))
    target_fps = float(extract_count) / duration

    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error",
        "-i", str(video),
        "-vf", f"fps={target_fps:.6f}",
        "-vsync", "vfr",
        "-frame_pts", "0",
        "-q:v", "2",
        f"{frames_dir}/%04d.png",
    ]
    subprocess.run(cmd, check=True)

    frame_count = sum(1 for _ in frames_dir.glob("*.png"))
    if frame_count == 0:
        raise RuntimeError("ffmpeg produced 0 frames")
    if frame_count > extract_count:
        for extra in sorted(frames_dir.glob("*.png"))[extract_count:]:
            extra.unlink()
        frame_count = extract_count

    return {
        "scene_id": scene_id,
        "frame_count": frame_count,
        "duration_s": duration,
        "target_fps": target_fps,
        "video_basename": video.name,
    }


# ---------------------------------------------------------------------------- modal volume sync

def _push_inputs_to_modal(scene_id: str) -> None:
    """Mirror local data/inputs/<id>/ into the Modal `spatiality-inputs` volume.

    The GPU containers read frames from ``/inputs/<id>/frames`` so we need
    them on the volume before invoking the Modal function. We use
    ``Volume.batch_upload`` in force-replace mode so re-runs always reflect
    the current local state.
    """
    import modal

    src = _scene_input_dir(scene_id)
    if not src.exists():
        raise FileNotFoundError(f"no local input dir for {scene_id}")

    vol = modal.Volume.from_name(INPUTS_VOLUME, create_if_missing=True)
    with vol.batch_upload(force=True) as batch:
        batch.put_directory(str(src), f"/{scene_id}")


# Path prefixes that are pipeline-internal (only consumed on Modal during
# segmentation) and have no local consumer once the run completes. Skipping
# them shrinks a typical pull from ~200 MB to ~10–20 MB. Re-add a prefix
# here if a future local feature needs it.
#
#   frames/                — replaced by per-(track, frame) evidence crops
#                            under evidence/<id>/<frame>.jpg
#   depth/, depth_conf/    — VGGT depth maps; only used by Stage 3 lift
#   world_points*/         — VGGT point-head outputs; same
#   _forward_preds.pt      — Stage 1 crash-safety checkpoint
#   _lifted_tracks_v2.pkl  — Stage 2/3 crash-safety checkpoint
_PULL_SKIP_PREFIXES: tuple[str, ...] = (
    "frames/",
    "depth/",
    "depth_conf/",
    "world_points/",
    "world_points_conf/",
    "_forward_preds.pt",
    "_lifted_tracks_v2.pkl",
)


def _pull_outputs_from_modal(scene_id: str, exclude: set[str] | None = None) -> None:
    """Mirror Modal `spatiality-outputs`/<id>/ back to local data/outputs/<id>/.

    Walks the remote tree with ``Volume.iterdir`` and streams each file via
    ``Volume.read_file``. Done after every Modal stage so the local disk is
    always the source of truth the FastAPI artifact endpoint serves from.

    Skips pipeline-internal artefacts (see ``_PULL_SKIP_PREFIXES``) that
    have no local consumer — these stay on the Modal volume where the next
    stage / a re-run can still reach them.

    ``exclude`` is a set of scene-relative paths (e.g. ``{"manifest.json"}``)
    to skip. The poses→segmentation overlap pulls with manifest excluded so
    the orchestrator's ``_bump_manifest`` writes during segmentation don't
    race with Modal's manifest writes streaming back from the inference run.
    """
    import modal

    skip = exclude or set()
    vol = modal.Volume.from_name(OUTPUTS_VOLUME, create_if_missing=True)
    dst_root = _scene_output_dir(scene_id)
    dst_root.mkdir(parents=True, exist_ok=True)

    def _walk(remote_dir: str) -> Iterator[str]:
        for entry in vol.iterdir(remote_dir):
            # FileEntryType.DIRECTORY == 2 in the Modal SDK.
            if getattr(entry, "type", None) and int(entry.type) == 2:
                yield from _walk(entry.path)
            else:
                yield entry.path

    remote_root = f"/{scene_id}"
    try:
        files = list(_walk(remote_root))
    except FileNotFoundError:
        return

    for remote_path in files:
        rel = remote_path.lstrip("/")[len(scene_id) + 1:]  # strip "<id>/"
        if rel in skip:
            continue
        if any(rel == p or rel.startswith(p) for p in _PULL_SKIP_PREFIXES):
            continue
        local_path = dst_root / rel
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with local_path.open("wb") as f:
            for chunk in vol.read_file(remote_path):
                f.write(chunk)


# ---------------------------------------------------------------------------- manifest

def _seed_manifest(scene_id: str) -> Path:
    out_dir = _scene_output_dir(scene_id)
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


def _bump_manifest(
    scene_id: str,
    stage: str | None,
    status: str,
    *,
    top: str | None = None,
    error: str | None = None,
) -> None:
    mpath = _scene_output_dir(scene_id) / "manifest.json"
    if not mpath.exists():
        return
    m = json.loads(mpath.read_text())
    if stage and stage in m.get("stages", {}):
        m["stages"][stage]["status"] = status
    if top is not None:
        m["status"] = top
    if error:
        m.setdefault("errors", []).append(error)
    mpath.write_text(json.dumps(m, indent=2))


def _recompute_stats(scene_id: str) -> None:
    """Refresh manifest.stats from what actually landed on disk.

    The seed manifest writes zeros for splat_size_mb / frame_count /
    object_count; the GPU-side modules update stage status but don't know the
    final on-disk sizes. Without this, the frontend treats the splat as empty
    (splat_size_mb <= 0.001 in scenes/[id]/page.tsx) and never mounts the
    viewer. Called after each _pull_outputs_from_modal so the stats track the
    artifacts the FastAPI server is about to serve.
    """
    out_dir = _scene_output_dir(scene_id)
    mpath = out_dir / "manifest.json"
    if not mpath.exists():
        return
    m = json.loads(mpath.read_text())
    stats = m.setdefault("stats", {})

    ply = out_dir / "points.ply"
    if ply.exists():
        stats["splat_size_mb"] = round(ply.stat().st_size / 1e6, 3)

    frames_dir = out_dir / "frames"
    if frames_dir.is_dir():
        stats["frame_count"] = sum(1 for _ in frames_dir.glob("*.png"))

    # Lane B is the canonical annotation set the UI defaults to.
    anno = out_dir / "annotations.b.json"
    if anno.exists():
        try:
            body = json.loads(anno.read_text())
            stats["object_count"] = len(body) if isinstance(body, list) else len(body.get("annotations", []))
        except Exception:
            pass

    mpath.write_text(json.dumps(m, indent=2))


# ---------------------------------------------------------------------------- pipeline

def _run_pipeline(scene_id: str, n_frames: int, infer_kwargs: dict | None = None) -> None:
    """Run extract → inference → segmentation locally-driven.

    Each Modal stage is followed by a pull-from-volume so the local disk
    reflects what the GPU stage wrote. Failures flip the manifest so the
    UI poll terminates instead of spinning forever.

    `infer_kwargs` is the per-job override dict from POST /api/jobs
    `settings` (e.g. {"target_count": 100_000_000} for a demo capture).
    Forwarded straight through into `inference.run.run`.
    """
    import modal

    infer_kwargs = dict(infer_kwargs or {})

    try:
        _bump_manifest(scene_id, "poses", "running")
        _extract_frames(scene_id, n_frames)
        _push_inputs_to_modal(scene_id)
    except BaseException as exc:
        _bump_manifest(
            scene_id, "poses", "failed",
            top="failed", error=f"{type(exc).__name__}: {exc}",
        )
        return

    try:
        infer_fn = modal.Function.from_name("spatiality-inference", "run_inference_one")
        infer_fn.remote(scene_id, frames_max=n_frames, **infer_kwargs)
    except BaseException as exc:
        _bump_manifest(
            scene_id, "poses", "failed",
            top="failed", error=f"{type(exc).__name__}: {exc}",
        )
        return

    # Pull poses artifacts and run segmentation concurrently. Segmentation reads
    # its inputs from the Modal volume, not from the local disk, so the pull is
    # only for the laptop's artifact-serving role and can overlap with the seg
    # call. Saves the wallclock cost of the ~5–8 min PLY+depth-map pull.
    #
    # The pull excludes manifest.json so it doesn't race with the orchestrator's
    # `_bump_manifest("segmentation", "running")` below; the final pull at the
    # end of the function picks up the canonical manifest from Modal.
    pull_outcome: dict = {}

    def _bg_pull_poses() -> None:
        try:
            _pull_outputs_from_modal(scene_id, exclude={"manifest.json"})
            _recompute_stats(scene_id)
        except BaseException as exc:
            pull_outcome["error"] = exc

    pull_thread = threading.Thread(target=_bg_pull_poses, daemon=True)
    pull_thread.start()

    seg_error: BaseException | None = None
    try:
        _bump_manifest(scene_id, "segmentation", "running")
        seg_fn = modal.Function.from_name("spatiality-segmentation", "run_segmentation_one")
        seg_fn.remote(scene_id)
    except BaseException as exc:
        seg_error = exc

    # Always wait for the poses pull to finish before pulling segmentation
    # outputs (avoids two threads writing the same files) and before reporting
    # any failure.
    pull_thread.join()

    if seg_error is not None:
        _bump_manifest(
            scene_id, "segmentation", "failed",
            top="failed", error=f"{type(seg_error).__name__}: {seg_error}",
        )
        return

    if "error" in pull_outcome:
        exc = pull_outcome["error"]
        _bump_manifest(
            scene_id, "poses", "failed",
            top="failed", error=f"{type(exc).__name__}: {exc}",
        )
        return

    try:
        _pull_outputs_from_modal(scene_id)
        _recompute_stats(scene_id)
    except BaseException as exc:
        _bump_manifest(
            scene_id, "segmentation", "failed",
            top="failed", error=f"{type(exc).__name__}: {exc}",
        )
        return


# ---------------------------------------------------------------------------- app

app = FastAPI(
    title="spatiality-local",
    version="0.2.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.post("/api/uploads/local")
async def upload_local(request: Request):
    form = await request.form()
    upload = form.get("video")
    if upload is None or not hasattr(upload, "read") or not hasattr(upload, "filename"):
        raise HTTPException(400, "expected multipart field 'video' with a file")
    if not upload.filename:
        raise HTTPException(400, "missing filename")
    ext = Path(upload.filename).suffix.lower() or ".mp4"
    if ext not in (".mp4", ".mov", ".webm", ".mkv", ".m4v"):
        raise HTTPException(400, f"unsupported extension: {ext}")

    scene_id = uuid.uuid4().hex[:12]
    scene_dir = _scene_input_dir(scene_id)
    scene_dir.mkdir(parents=True, exist_ok=True)
    target = scene_dir / f"source{ext}"

    bytes_written = 0
    chunk_size = 8 * 1024 * 1024
    with target.open("wb") as f:
        while True:
            buf = await upload.read(chunk_size)
            if not buf:
                break
            f.write(buf)
            bytes_written += len(buf)

    return {
        "scene_id": scene_id,
        "upload_path": str(target),
        "bytes": bytes_written,
        "original_filename": upload.filename,
    }


@app.post("/api/jobs")
async def submit_job(request: Request):
    body = await request.json()
    scene_id = body.get("scene_id")
    if not scene_id:
        raise HTTPException(400, "scene_id required")
    settings = body.get("settings") or {}
    n_frames = int(settings.get("max_frames") or DEFAULT_FRAMES)

    # Pass-through of inference filter knobs (forwarded into
    # `inference.run.run` via run_inference_one's **kwargs). Lets demo runs
    # bump the cloud cap to 100 M etc. without a redeploy.
    infer_kwargs: dict = {}
    for key in ("target_count", "conf_min", "pixel_stride",
                "depth_grad_max", "depth_far_pct", "depth_far_mult",
                "blur_drop_pct"):
        if key in settings and settings[key] is not None:
            infer_kwargs[key] = settings[key]

    _seed_manifest(scene_id)

    # Background thread keeps the request lightweight and the Modal calls
    # off the event loop. We don't need durability across server restarts —
    # this is the local-orchestrator format on purpose.
    threading.Thread(
        target=_run_pipeline,
        args=(scene_id, n_frames, infer_kwargs),
        daemon=True,
    ).start()

    return {"status": "queued", "scene_id": scene_id}


@app.get("/api/jobs/{scene_id}")
def get_job(scene_id: str):
    try:
        path = _scene_output_dir(scene_id) / "manifest.json"
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    if not path.exists():
        raise HTTPException(404, "manifest not found")
    # Self-heal stale stats: if the splat finished but stats still report
    # zero size (e.g. a downstream stage crashed before _recompute_stats
    # ran, or the scene was produced by an older pipeline), recompute from
    # disk so the frontend's emptySplat gate flips and the viewer mounts.
    m = json.loads(path.read_text())
    splat_done = m.get("stages", {}).get("splat", {}).get("status") == "complete"
    splat_mb = m.get("stats", {}).get("splat_size_mb", 0.0) or 0.0
    if splat_done and splat_mb <= 0.001:
        _recompute_stats(scene_id)
        m = json.loads(path.read_text())
    return JSONResponse(m)


@app.get("/api/scenes")
def list_scenes():
    if not OUTPUTS_ROOT.exists():
        return []
    out: list[dict] = []
    for d in sorted(OUTPUTS_ROOT.iterdir()):
        if not d.is_dir():
            continue
        mpath = d / "manifest.json"
        if not mpath.exists():
            continue
        try:
            m = json.loads(mpath.read_text())
        except Exception:
            continue
        out.append({
            "scene_id": d.name,
            "status": m.get("status"),
            "created_at": m.get("created_at"),
            "stats": m.get("stats", {}),
        })
    out.sort(key=lambda s: s.get("created_at") or "", reverse=True)
    return out


@app.get("/api/gateway/health")
def gateway_health():
    return {
        "ok": True,
        "key_set": True,
        "region": "us",
        "probe_status": 200,
        "latency_ms": 0,
    }


@app.get("/api/trace/{scene_id}")
def trace(scene_id: str):
    return {
        "scene_id": scene_id,
        "span_count": 0,
        "tree": [],
        "cost": {"total_usd": 0.0, "call_count": 0, "by_span": []},
    }


@app.get("/api/trace/{scene_id}/cost")
def trace_cost(scene_id: str):
    return {"total_usd": 0.0, "call_count": 0, "by_span": []}


@app.post("/api/agent/locate")
def locate():
    raise HTTPException(501, "agent locate not wired in this demo")


@app.post("/api/agent/chat")
def chat():
    raise HTTPException(501, "agent chat not wired in this demo")


_MEDIA_TYPES = {
    ".json": "application/json",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".ply": "application/octet-stream",
    ".npy": "application/octet-stream",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
}


@app.get("/artifacts/scenes/{scene_id}/{rel_path:path}")
def get_artifact(scene_id: str, rel_path: str):
    try:
        target = _safe_artifact_path(scene_id, rel_path)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    if not target.exists() or not target.is_file():
        raise HTTPException(404, f"artifact not found: {rel_path}")

    media_type = _MEDIA_TYPES.get(target.suffix.lower(), "application/octet-stream")
    size = target.stat().st_size

    def _iter():
        with target.open("rb") as f:
            while True:
                buf = f.read(1024 * 1024)
                if not buf:
                    break
                yield buf

    headers = {
        "cache-control": "public, max-age=60",
        "content-length": str(size),
    }
    return StreamingResponse(_iter(), media_type=media_type, headers=headers)


@app.get("/health")
def health():
    return {"ok": True, "ts": time.time()}
