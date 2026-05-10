"""Segmentation orchestrator.

Reads geometry artefacts from Stage 1 and runs the rest of the pipeline:

  - Stage 2  : Grounding DINO detection + IoU tracklet linking
  - Stage 3  : per-track 3D pinning (bbox-depth unprojection)
  - Stage 4B : VLM-verified labels  → annotations.b.json (async, 16-way)

Lane E (scene-graph relations) and Lane F (SpatialLM layout) were removed
2026-05-10 — they were not contributing to the VLM-labelling story we
currently care about. Their modules / Modal deps are gone; if you bring
relations back later, reintroduce as a separate stage.

Updates manifest.json so the frontend's stage waterfall reflects progress.

The orchestrator is sync; Lane B runs inside its own asyncio event loop
via ``asyncio.run(...)``. Stage 1.5 (`scene_scout`) uses its own
``asyncio.run`` for scout's internal slice fan-out, so we keep it as a
plain sync call from this module — nesting would raise
``RuntimeError: cannot be called from a running event loop``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def _artefact_root() -> Path:
    return Path(os.environ.get("SPATIALITY_ARTEFACTS_ROOT", "/outputs"))


def _read_manifest(scene_dir: Path) -> dict:
    path = scene_dir / "manifest.json"
    if path.exists():
        return json.loads(path.read_text())
    return {
        "scene_id": scene_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "processing",
        "stages": {
            "capture": {"status": "complete"},
            "poses": {"status": "complete"},
            "splat": {"status": "complete"},
            "segmentation": {"status": "running"},
        },
        "artifacts": {},
        "stats": {"frame_count": 0, "object_count": 0, "splat_size_mb": 0.0},
    }


def _write_manifest(scene_dir: Path, manifest: dict) -> None:
    (scene_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def _set_segmentation_status(
    scene_dir: Path,
    status: str,
    duration_s: float | None = None,
    object_count: int | None = None,
    error: str | None = None,
) -> None:
    m = _read_manifest(scene_dir)
    seg = m["stages"]["segmentation"]
    seg["status"] = status
    if duration_s is not None:
        seg["duration_s"] = duration_s
    if object_count is not None:
        seg["object_count"] = object_count
        m["stats"]["object_count"] = object_count

    artifacts = m.setdefault("artifacts", {})
    if (scene_dir / "annotations.b.json").exists():
        artifacts["annotations_b_json"] = "annotations.b.json"
    if (scene_dir / "annotations.b.discarded.json").exists():
        artifacts["annotations_b_discarded_json"] = "annotations.b.discarded.json"
    if (scene_dir / "annotations.c.json").exists():
        artifacts["annotations_c_json"] = "annotations.c.json"
    # Frontend reads `annotations_json` — prefer Lane C (whole-scene
    # coherence-reviewed) when available, fall back to Lane B raw labels.
    if (scene_dir / "annotations.c.json").exists():
        artifacts["annotations_json"] = "annotations.c.json"
    else:
        artifacts.setdefault("annotations_json", "annotations.b.json")

    if status == "complete":
        m["status"] = "ready"
    elif status == "failed":
        m["status"] = "failed"
        errors = m.setdefault("errors", [])
        if error:
            errors.append(error)

    _write_manifest(scene_dir, m)


def run(input_id: str, **kwargs) -> dict:
    """Entry point called from ``modal_segmentation.py::run_segmentation_one``.

    Accepted kwargs:
      lanes (list[str]): subset of {"b", "e", "f"} to run (default all).
      vlm_model (str): Gemini model id used by scout + lanes B/E.
      extra_text_prompts (list[str]): optional taxonomy lane in GDINO.
      use_scout (bool): default True; set False to skip scout and use the
        fallback vocabulary baked into gdino.py.
      gdino_score_threshold (float): GDINO confidence threshold (default 0.20).
      min_track_frames (int): drop tracks shorter than this (default 5).
      scout_n_frames (int): keyframes per slice the scout sees (default 6).
    """
    scene_dir = _artefact_root() / input_id
    if not scene_dir.exists():
        raise SystemExit(f"missing geometry stage outputs at {scene_dir}; run inference first")

    # Default lanes: B (per-track labels) + C (whole-scene coherence).
    # Lane C is cheap (~15s) and always idempotent, so it's safe to leave on.
    lanes = kwargs.get("lanes") or ["b", "c"]
    # Default Gemini 2.5 Flash; override via kwargs or SPATIALITY_VLM_MODEL.
    vlm_model = kwargs.get("vlm_model") or os.environ.get(
        "SPATIALITY_VLM_MODEL", "gemini-2.5-flash"
    )

    _set_segmentation_status(scene_dir, "running")
    t0 = time.time()
    print(f"[stage:segmentation] input_id={input_id} lanes={lanes} vlm_model={vlm_model}", flush=True)

    # Crash-safety: pickle the lifted tracks once they're built. The `_v2`
    # suffix ensures stale pickles from the SAM-2-era schema (which carried
    # siglip_feat) are never silently loaded into the new dataclass.
    import pickle as _pickle
    lifted_ckpt = scene_dir / "_lifted_tracks_v2.pkl"
    sam_tracks: list = []
    lifted: list = []

    try:
        if lifted_ckpt.exists():
            t_resume = time.time()
            with lifted_ckpt.open("rb") as f:
                _maybe_lifted = _pickle.load(f)
            if _maybe_lifted:
                lifted = _maybe_lifted
                print(f"[stage:segmentation] resuming from lifted-tracks checkpoint: "
                      f"{lifted_ckpt.name} ({len(lifted)} tracks, "
                      f"{time.time()-t_resume:.1f}s)", flush=True)
            else:
                print(f"[stage:segmentation] ignoring empty checkpoint "
                      f"{lifted_ckpt.name} — re-running detection + lift",
                      flush=True)
                lifted_ckpt.unlink(missing_ok=True)
        if not lifted:
            # Stage 1.5 — VLM scene scout (sync; uses its own asyncio.run).
            scout_prompts = kwargs.get("text_prompts")
            use_scout = bool(kwargs.get("use_scout", True))
            if scout_prompts is None and use_scout:
                from .scene_scout import discover_scene_prompts

                t_scout = time.time()
                print("[stage:segmentation] === Stage 1.5: VLM scene scout ===", flush=True)
                try:
                    scout_prompts = discover_scene_prompts(
                        frames_dir=scene_dir / "frames",
                        vlm_model=vlm_model,
                        n_frames=int(kwargs.get("scout_n_frames", 6)),
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"[stage:segmentation] scout FAILED ({type(e).__name__}: {e}) — "
                          f"GDINO will use the fallback vocabulary", flush=True)
                    scout_prompts = None
                print(f"[stage:segmentation] scout done in {time.time()-t_scout:.1f}s", flush=True)

            # Stage 2 — Grounding DINO + IoU tracklet linking.
            from .gdino import run_gdino

            t_sam = time.time()
            print("[stage:segmentation] === Stage 2: GDINO detect + IoU-link ===", flush=True)
            sam_tracks = run_gdino(
                frames_dir=scene_dir / "frames",
                out_dir=scene_dir,
                text_prompts=scout_prompts,
                extra_text_prompts=kwargs.get("extra_text_prompts"),
                score_threshold=float(kwargs.get("gdino_score_threshold", 0.20)),
                min_track_frames=int(kwargs.get("min_track_frames", 5)),
            )
            print(f"[stage:segmentation] GDINO done in {time.time()-t_sam:.1f}s — "
                  f"{len(sam_tracks)} tracks", flush=True)

            # Stage 3 — bbox-depth lifting.
            from .lift import run_lifting

            t_lift = time.time()
            print(f"[stage:segmentation] === Stage 3: 3D lifting on {len(sam_tracks)} tracks ===", flush=True)
            lifted = run_lifting(sam_tracks, scene_dir)
            print(f"[stage:segmentation] lifting done in {time.time()-t_lift:.1f}s — "
                  f"{len(lifted)} lifted tracks", flush=True)

            if lifted:
                t_save = time.time()
                with lifted_ckpt.open("wb") as f:
                    _pickle.dump(lifted, f)
                size_mb = lifted_ckpt.stat().st_size / 1e6
                print(f"[stage:segmentation] lifted-tracks checkpoint saved → "
                      f"{lifted_ckpt.name} ({size_mb:.1f} MB, {time.time()-t_save:.1f}s) — "
                      f"lanes are now crash-safe", flush=True)
            else:
                print(f"[stage:segmentation] 0 lifted tracks — skipping checkpoint "
                      f"so the next run re-runs detection instead of resuming nothing",
                      flush=True)

        # Stage 4 — labeling lanes.
        lane_b_anns: list[dict] = []
        if "b" in lanes:
            from .lane_b import run_lane_b

            t_b = time.time()
            print(f"[stage:segmentation] === Stage 4B: VLM labels on {len(lifted)} tracks ===", flush=True)
            # Lane B is the only lane that needs the asyncio loop (it fans
            # out 16-way under asyncio.gather). Scout + Lane C use sync
            # wrappers internally, so we keep them on the main thread.
            lane_b_anns = asyncio.run(run_lane_b(lifted, scene_dir, vlm_model=vlm_model))
            print(f"[stage:segmentation] Lane B done in {time.time()-t_b:.1f}s — "
                  f"{len(lane_b_anns)} annotations", flush=True)
        else:
            lane_b_anns = []
            print("[stage:segmentation] Lane B skipped", flush=True)

        # Stage 4C — whole-scene coherence review (one Gemini call).
        # Idempotent: if annotations.c.json already exists this returns
        # instantly without invoking the VLM, so a downstream failure
        # never re-pays the call cost.
        lane_c_anns: list[dict] = lane_b_anns
        if "c" in lanes and lane_b_anns:
            from .lane_c import run_lane_c

            t_c = time.time()
            print(f"[stage:segmentation] === Stage 4C: whole-scene coherence on "
                  f"{len(lane_b_anns)} annotations ===", flush=True)
            lane_c_anns = run_lane_c(lane_b_anns, scene_dir, vlm_model=vlm_model)
            print(f"[stage:segmentation] Lane C done in {time.time()-t_c:.1f}s — "
                  f"{len(lane_c_anns)} annotations", flush=True)

        duration = time.time() - t0
        _set_segmentation_status(
            scene_dir, "complete",
            duration_s=duration,
            object_count=len(lifted),
        )
        if lifted_ckpt.exists():
            try:
                lifted_ckpt.unlink()
                print(f"[stage:segmentation] cleaned up lifted-tracks checkpoint", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[stage:segmentation] WARN: could not delete {lifted_ckpt}: {e}", flush=True)

        print(f"[stage:segmentation] DONE in {duration:.1f}s — "
              f"{len(sam_tracks)} tracks, {len(lifted)} lifted, "
              f"{len(lane_b_anns)} labelled, {len(lane_c_anns)} after coherence",
              flush=True)
        return {
            "input_id": input_id,
            "status": "complete",
            "track_count": len(sam_tracks),
            "annotation_count": len(lane_c_anns),
            "duration_s": duration,
            "lanes": lanes,
        }
    except Exception as e:  # noqa: BLE001
        print(f"[stage:segmentation] FAILED after {time.time()-t0:.1f}s: "
              f"{type(e).__name__}: {e}", flush=True)
        logger.exception("segmentation failed")
        _set_segmentation_status(
            scene_dir, "failed", duration_s=time.time() - t0, error=str(e),
        )
        raise
