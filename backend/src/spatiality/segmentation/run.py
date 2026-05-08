"""Segmentation orchestrator.

Reads geometry artefacts from Stage 1 and runs the rest of the pipeline:

  - Stage 2  : SAM 3.1 detection + tracking
  - Stage 3  : per-track 3D pinning + cross-frame stitch
  - Stage 4B : VLM-verified labels  → annotations.b.json
  - Stage 4E : scene graph (relations) → annotations.e.json
  - Stage 4F : SpatialLM layout       → annotations.f.json

Updates manifest.json so the frontend's stage waterfall reflects progress.
"""

from __future__ import annotations

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
    if (scene_dir / "annotations.e.json").exists():
        artifacts["annotations_e_json"] = "annotations.e.json"
    if (scene_dir / "annotations.f.json").exists():
        artifacts["annotations_f_json"] = "annotations.f.json"
    artifacts.setdefault("annotations_json", "annotations.b.json")  # default lane the legacy UI reads

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
      vlm_model (str): Claude model id used by lanes B + E.
      seed_stride (int), reprompt_stride (int): SAM 3.1 cadence.
      extra_text_prompts (list[str]): optional taxonomy lane in SAM 3.1.
    """
    scene_dir = _artefact_root() / input_id
    if not scene_dir.exists():
        raise SystemExit(f"missing geometry stage outputs at {scene_dir}; run inference first")

    lanes = kwargs.get("lanes") or ["b", "e", "f"]
    # Default Gemini 2.5 Flash; override via kwargs or SPATIALITY_VLM_MODEL.
    # gemini-2.5-flash-lite is the cheaper / faster option for large clips.
    vlm_model = kwargs.get("vlm_model", "gemini-2.5-flash")

    _set_segmentation_status(scene_dir, "running")
    t0 = time.time()

    try:
        # Stage 2 — SAM 3.1.
        from .sam3 import run_sam3

        sam_tracks = run_sam3(
            frames_dir=scene_dir / "frames",
            out_dir=scene_dir,
            seed_stride=int(kwargs.get("seed_stride", 25)),
            reprompt_stride=int(kwargs.get("reprompt_stride", 100)),
            extra_text_prompts=kwargs.get("extra_text_prompts"),
        )

        # Stage 3 — lifting + safety-net merge.
        from .lift import run_lifting

        lifted = run_lifting(sam_tracks, scene_dir)

        # Stage 4 — labeling lanes.
        lane_b_anns: list[dict] = []
        if "b" in lanes:
            from .lane_b import run_lane_b

            lane_b_anns = run_lane_b(lifted, scene_dir, vlm_model=vlm_model)
        else:
            lane_b_anns = []

        if "e" in lanes:
            from .lane_e import run_lane_e

            run_lane_e(lifted, lane_b_anns, scene_dir, vlm_model=vlm_model)

        if "f" in lanes:
            from .lane_f import run_lane_f

            run_lane_f(scene_dir)

        duration = time.time() - t0
        _set_segmentation_status(
            scene_dir, "complete",
            duration_s=duration,
            object_count=len(lifted),
        )
        return {
            "input_id": input_id,
            "status": "complete",
            "track_count": len(sam_tracks),
            "annotation_count": len(lifted),
            "duration_s": duration,
            "lanes": lanes,
        }
    except Exception as e:  # noqa: BLE001
        logger.exception("segmentation failed")
        _set_segmentation_status(
            scene_dir, "failed", duration_s=time.time() - t0, error=str(e),
        )
        raise
