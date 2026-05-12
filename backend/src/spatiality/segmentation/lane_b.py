"""Lane B — VLM-verified labels (async, 16-way concurrent).

For each lifted track:
  1. render 6 orbital novel views of the splat focused on its OBB
  2. crop 3 anchor RGB views — score-weighted pick across early/mid/late
     frames so the VLM gets multiple real-photo angles, not one possibly
     awkward frame
  3. pass the original detector phrase as a candidate-label hint
  4. send the 9-image grid to Gemini 2.5 Flash via PydanticAI with a
     structured-output schema → {label, alternatives, confidence, reasoning}

Why 3 anchors and a phrase hint:
- The orbital views are stylised point-cloud renders; Gemini guesses
  shape from blurry blobs, leading to mistakes like "stroller" for what's
  actually a metal gymnastics bar.
- A single anchor crop can show an awkward partial view (half a bed
  framed as if it were a sofa). Three frames give Gemini temporal +
  angular variety so it can disambiguate.
- The detector phrase grounds Gemini in the original detection ("the
  detector flagged this via 'stroller' — confirm or correct").

Output: outputs/<id>/annotations.b.json with the standard `Annotation` shape.
Calls fan out via asyncio.gather under a Semaphore(16). Per-track flush
to disk after every completion; resume on retry skips already-labelled tracks.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time as _time
from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from .lift import LiftedTrack, _absorb_evidence
from .postprocess import cleanup_lane_b_annotations
from .render import composite_grid, crop_anchor, render_track_orbit
from .vlm import call_vlm_async

logger = logging.getLogger(__name__)


# Concurrency cap. Gemini Flash tolerates ~16 concurrent calls per project
# under default per-minute rate limits; higher than that we start seeing
# 429s without backoff hardening.
_LANE_B_CONCURRENCY = 16


class LabelOutput(BaseModel):
    """Structured response from the labeling VLM (Gemini 2.5 Flash)."""

    label: str = Field(description="A concrete noun phrase, or 'unknown'.")
    alternatives: list[str] = Field(default_factory=list, description="Up to 3 alternative noun phrases.")
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(default="", description="One short sentence.")


_PROMPT_TEMPLATE = """\
Images 1-6: orbital point-cloud renders of one 3D region (use for shape/scale only).
Images {anchor_start}-{anchor_end}: {n_anchors} real photographs of that same region.
Trust photographs over renders. Detector candidate: "{detector_phrase}" — verify or override.

Identify the SINGLE physical object. Use a concrete noun phrase, specific when supported \
("office chair", "ceramic mug", "table lamp", "houseplant in a pot"). Avoid bare single \
words when a qualifier is visible.

Return label="unknown" with confidence ≤ 0.3 if ANY of:
- No photograph shows the whole object (only cropped, occluded, or partial views) — do not \
guess what is off-frame.
- The region is a wall, floor, ceiling, room, scene, or empty space, not a discrete object.
- Different photographs show clearly different objects (tracker drift).
- Views are too ambiguous to commit.

Forbidden labels: "room", "wall", "floor", "ceiling", "scene", "area", "space", "background", \
"interior", "environment".

Output: label, up to 3 alternatives ranked by plausibility, calibrated confidence in [0, 1], \
and one short sentence of reasoning grounded in what the photographs show.\
"""


_N_ANCHORS = 3


def _pick_anchor_frames(
    track_payload: dict, out_dir: Path, n: int = _N_ANCHORS
) -> list[tuple[str, np.ndarray, tuple[int, int, int, int]]]:
    """Pick up to ``n`` high-evidence anchor frames spanning the track's lifetime.

    Strategy: compute an evidence score per frame = bbox_area × detector_score,
    then split the track temporally into ``n`` equal windows and pick the
    highest-evidence frame from each. This gives the VLM:
      - the BEST view per temporal segment (not just the largest bbox overall)
      - temporal diversity (early/mid/late) so it can spot tracker drift,
        pose changes, and partial-view artefacts.
    Returns at most ``n`` (frame_id, crop, bbox) tuples; fewer if the track
    has fewer frames or some PNGs are missing.
    """
    frames = list(track_payload.get("frames", []))
    if not frames:
        return []
    frames_sorted = sorted(frames, key=lambda f: f["frame_id"])
    n_frames = len(frames_sorted)
    n = min(n, n_frames)
    if n == 0:
        return []

    # Split temporally into n windows; pick highest-evidence frame in each.
    boundaries = [int(round(i * n_frames / n)) for i in range(n + 1)]
    picks: list[dict] = []
    seen_frame_ids: set[str] = set()
    for w in range(n):
        lo, hi = boundaries[w], max(boundaries[w] + 1, boundaries[w + 1])
        window = frames_sorted[lo:hi]
        if not window:
            continue
        best_in_window = max(
            window,
            key=lambda f: (
                max(0, f["bbox_2d"][2] - f["bbox_2d"][0])
                * max(0, f["bbox_2d"][3] - f["bbox_2d"][1])
                * float(f.get("score", 0.5))
            ),
        )
        if best_in_window["frame_id"] not in seen_frame_ids:
            picks.append(best_in_window)
            seen_frame_ids.add(best_in_window["frame_id"])

    out: list[tuple[str, np.ndarray, tuple[int, int, int, int]]] = []
    for p in picks:
        frame_path = out_dir / "frames" / f"{p['frame_id']}.png"
        if not frame_path.exists():
            continue
        bbox = tuple(int(v) for v in p["bbox_2d"])
        try:
            crop = crop_anchor(frame_path, bbox)
        except Exception as e:  # noqa: BLE001
            logger.warning("anchor crop failed for %s frame %s: %s",
                           track_payload.get("track_id", "?"), p["frame_id"], e)
            continue
        out.append((p["frame_id"], crop, bbox))
    return out


def _color_for(track_id: str) -> str:
    """Stable hex colour from the track id (purely cosmetic for the UI legend).

    Uses MD5 rather than ``hash()`` because Python salts ``hash()`` per
    process, which made the same scene's per-track colours flip between
    cold-start and resumed runs.
    """
    digest = hashlib.md5(track_id.encode("utf-8")).hexdigest()  # noqa: S324
    return f"#{digest[:6]}"


def _bbox_lo_hi(corners: np.ndarray) -> tuple[list[float], list[float]]:
    return corners.min(axis=0).tolist(), corners.max(axis=0).tolist()


def _render_track_inputs(
    track: LiftedTrack, track_payload: dict, out_dir: Path, points_path: Path
) -> tuple[np.ndarray, int]:
    """Build the composite (orbital + N anchor) grid for one track.

    Pure CPU work (numpy + PIL); kept synchronous and run inside a worker
    thread via asyncio.to_thread so the asyncio loop stays responsive.

    Returns ``(grid_image, n_anchors_actually_used)`` so the caller can
    parameterise the prompt accurately when fewer than the target number
    of anchor frames were available.
    """
    orbital = render_track_orbit(
        points_path,
        track.centroid,
        track.obb_corners,
        n_views=6,
    )
    target_size = orbital[0].shape[:2]  # (H, W)
    anchors = _pick_anchor_frames(track_payload, out_dir, n=_N_ANCHORS)
    anchor_crops: list[np.ndarray] = []
    for _, crop, _ in anchors:
        resized = np.asarray(
            Image.fromarray(crop).resize((target_size[1], target_size[0]))
        )
        anchor_crops.append(resized)

    images = orbital + anchor_crops
    return composite_grid(images, cols=3), len(anchor_crops)


async def _label_one(
    ti: int,
    n_total: int,
    track: LiftedTrack,
    track_payload: dict,
    out_dir: Path,
    points_path: Path,
    vlm_model: str,
    sem: asyncio.Semaphore,
    on_done,  # callable(annotation_dict) — invoked under caller's lock
) -> None:
    """Label one track and hand the annotation to ``on_done`` for flushing."""
    async with sem:
        _t_start = _time.time()
        n_anchors = 0
        grid: np.ndarray | None = None
        try:
            grid, n_anchors = await asyncio.to_thread(
                _render_track_inputs, track, track_payload, out_dir, points_path
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("render failed for %s: %s", track.track_id, e)

        if grid is None:
            reply = {"label": "unknown", "alternatives": [], "confidence": 0.1, "reasoning": ""}
        else:
            detector_phrase = (track.text_prompt or "unknown").strip()
            anchor_start = 7  # orbital views are 1-6
            anchor_end = 6 + n_anchors if n_anchors > 0 else 6
            prompt = _PROMPT_TEMPLATE.format(
                n_anchors=n_anchors,
                anchor_start=anchor_start,
                anchor_end=anchor_end,
                detector_phrase=detector_phrase,
            )
            try:
                model_output = await call_vlm_async(prompt, [grid], LabelOutput, model=vlm_model)
                reply = model_output.model_dump()
            except Exception as e:  # noqa: BLE001
                logger.warning("VLM call failed for %s: %s", track.track_id, e)
                reply = {"label": "unknown", "alternatives": [], "confidence": 0.1, "reasoning": ""}

        lo, hi = _bbox_lo_hi(track.obb_corners)
        # Calibrate confidence by combining the VLM's self-reported number
        # with two cheap corroborating signals: temporal support
        # (longer tracks are more reliable up to a 30-frame ceiling) and
        # depth-confidence mean from the lift. The VLM remains primary —
        # we only down-weight when corroboration is weak.
        vlm_conf = float(reply.get("confidence", 0.5))
        track_len_factor = min(1.0, len(track.frame_ids) / 30.0)
        depth_factor = float(np.clip(track.mean_conf, 0.0, 1.0))
        corroboration = 0.5 + 0.5 * (0.5 * track_len_factor + 0.5 * depth_factor)
        calibrated = float(np.clip(vlm_conf * corroboration, 0.0, 1.0))
        annotation = {
            "id": track.track_id,
            "label": reply.get("label", "unknown"),
            "centroid": track.centroid.tolist(),
            "bbox": [lo, hi],
            # Real PCA OBB for downstream consumers that want orientation.
            # Frontend still reads `bbox` as AABB; this is purely additive.
            "obb": {
                "centroid": track.centroid.tolist(),
                "axes": track.obb_axes.tolist(),
                "extents": track.obb_extents.tolist(),
                "corners": track.obb_corners.tolist(),
            },
            "color": _color_for(track.track_id),
            "confidence": calibrated,
            "confidence_components": {
                "vlm": vlm_conf,
                "track_length": track_len_factor,
                "depth_mean": depth_factor,
                "n_frames": len(track.frame_ids),
            },
            "alternatives": reply.get("alternatives", []),
            # Surface only the frames for which the lift wrote per-(track,
            # frame) evidence (`evidence/<id>/<frame>.jpg` +
            # `masks/<id>/<frame>.png`). The UI's evidence panel only ever
            # requests these stems, so it never sees a 404 on a frame the
            # lift didn't consider. Falls back to the full track frame
            # list for legacy lifted-track pickles missing the field.
            "frame_ids": [
                f"{fid}.png"
                for fid in (getattr(track, "evidence_frame_ids", None) or track.frame_ids)
            ],
            "provenance": [
                f"gdino:{track.source}",
                f"vlm:{vlm_model}",
            ],
        }
        await on_done(annotation)
        print(f"[lane_b]   {ti}/{n_total} {track.track_id} → "
              f"label='{reply.get('label','?')}' "
              f"conf={float(reply.get('confidence',0)):.2f} "
              f"({_time.time()-_t_start:.1f}s)", flush=True)


# ---------------------------------------------------------------------------- entry point


async def run_lane_b(
    lifted_tracks: list[LiftedTrack],
    out_dir: Path,
    vlm_model: str = "gemini-2.5-flash",
) -> list[dict]:
    """Produce VLM-verified annotations for every lifted track (async, 16-way)."""
    tracks_payload = json.loads((out_dir / "tracks.json").read_text())
    tracks_by_id = {t["track_id"]: t for t in tracks_payload["tracks"]}

    # Per-track flushes target the *raw* file so resume sees every track,
    # even those the cleanup pass would later drop (scene labels, low-conf,
    # oversize). The cleaned file (`annotations.b.json`) is rewritten once
    # at the end and is what the frontend reads.
    raw_path = out_dir / "annotations.b.raw.json"
    out_path = out_dir / "annotations.b.json"
    done: dict[str, dict] = {}
    # Prefer the raw file for resume; fall back to the cleaned file for
    # backward compatibility with runs from before this split existed.
    resume_source = raw_path if raw_path.exists() else (out_path if out_path.exists() else None)
    if resume_source is not None:
        try:
            existing = json.loads(resume_source.read_text())
            done = {a["id"]: a for a in existing if isinstance(a, dict) and "id" in a}
        except Exception as e:  # noqa: BLE001
            logger.warning("could not parse existing %s (%s); starting fresh",
                           resume_source.name, e)
            done = {}

    # Drop resumed entries whose track_id no longer appears in the current
    # lifted_tracks. This prevents a stale annotation set (e.g. from
    # before the reprojection guard or the merge changes) from leaking
    # tracks that the new lift would have rejected.
    valid_ids = {t.track_id for t in lifted_tracks}
    n_stale = sum(1 for tid in done if tid not in valid_ids)
    if n_stale:
        done = {tid: a for tid, a in done.items() if tid in valid_ids}
        print(f"[lane_b] dropped {n_stale} stale resumed annotation(s) — "
              f"track_ids no longer in current lifted set", flush=True)

    points_path = out_dir / "points.ply"
    pending = [t for t in lifted_tracks if t.track_id not in done]
    print(f"[lane_b] {len(lifted_tracks)} tracks total, {len(done)} resumed, "
          f"{len(pending)} pending; vlm_model={vlm_model}, "
          f"concurrency={_LANE_B_CONCURRENCY}", flush=True)

    discarded_path = out_dir / "annotations.b.discarded.json"

    if not pending:
        print(f"[lane_b] nothing to do — all {len(done)} tracks already labelled", flush=True)
        return _finalise(list(done.values()), raw_path, out_path, discarded_path)

    sem = asyncio.Semaphore(_LANE_B_CONCURRENCY)
    write_lock = asyncio.Lock()

    async def _flush(ann: dict) -> None:
        async with write_lock:
            done[ann["id"]] = ann
            raw_path.write_text(json.dumps(list(done.values()), indent=2))

    n_total = len(lifted_tracks)
    coros = []
    for ti, tr in enumerate(lifted_tracks, start=1):
        if tr.track_id in done:
            continue
        payload = tracks_by_id.get(tr.track_id, {"frames": []})
        coros.append(_label_one(
            ti, n_total, tr, payload, out_dir, points_path, vlm_model, sem, _flush,
        ))

    _t_lane = _time.time()
    await asyncio.gather(*coros)
    print(f"[lane_b] all tracks labelled — {len(done)} raw annotations in "
          f"{raw_path.name} ({_time.time()-_t_lane:.1f}s)", flush=True)
    return _finalise(list(done.values()), raw_path, out_path, discarded_path)


def _finalise(
    raw: list[dict],
    raw_path: Path,
    out_path: Path,
    discarded_path: Path,
) -> list[dict]:
    """Run the postprocess cleanup and write the final annotations file.

    Keeps the raw flush file on disk for debugging / re-cleanup; writes
    the cleaned, deduped, scene-label-free list to ``out_path`` (which is
    what the frontend reads). Also writes a unified
    ``annotations.b.discarded.json`` that merges drops from every
    pipeline stage:
      - GDINO short-tracklet drops (``_gdino_discards.json``)
      - 3D-lift drops & merge-losers (``_lift_discards.json``)
      - Lane B postprocess drops (computed here)
    Each entry carries ``stage`` + ``discard_reason`` so the UI can
    group by stage without duplicating cleanup logic client-side.
    """
    cleaned, postprocess_discarded, stats = cleanup_lane_b_annotations(raw)
    print(
        f"[lane_b] postprocess: {stats['n_in']} → {stats['n_out']} "
        f"(scene-labels={stats['dropped_scene_label']}, "
        f"low-conf={stats['dropped_low_conf']}, "
        f"oversize={stats['dropped_oversize']}, "
        f"merged-duplicates={stats['merged_duplicates']}, "
        f"scene_diag={stats['scene_diag_m']}m)",
        flush=True,
    )

    out_dir = discarded_path.parent

    # Relocate evidence/masks crops from merged-loser dirs into the survivor's
    # dir so the viewer can resolve `obj_<survivor>/<frame>.{jpg,png}` for
    # every frame_id listed on the merged annotation. Postprocess dedup only
    # rewrites the JSON; without this step the loser's per-frame files stay
    # under their original track_id and 404 in the UI.
    for entry in cleaned:
        for loser_id in entry.get("merged_from", []) or []:
            _absorb_evidence(out_dir, entry["id"], loser_id)

    out_path.write_text(json.dumps(cleaned, indent=2))
    upstream: list[dict] = []
    for fname in ("_gdino_discards.json", "_lift_discards.json"):
        fp = out_dir / fname
        if not fp.exists():
            continue
        try:
            upstream.extend(json.loads(fp.read_text()))
        except Exception as e:  # noqa: BLE001
            logger.warning("could not parse %s: %s", fp.name, e)

    all_discarded = upstream + postprocess_discarded
    discarded_path.write_text(json.dumps(all_discarded, indent=2))
    print(
        f"[lane_b] discards: {len(all_discarded)} total "
        f"(gdino+lift={len(upstream)}, postprocess={len(postprocess_discarded)})",
        flush=True,
    )
    return cleaned
