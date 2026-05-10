"""Grounding DINO open-vocab detection + IoU tracklet linking.

Stage 2 of the segmentation pipeline. Replaces the prior GDINO + SAM 2.1 stack:
SAM 2 mask propagation is gone, so each IoU-linked tracklet becomes a
``Track`` directly with per-frame bboxes. The web UI never rendered masks
and the lift stage now reads bboxes (not masks), so SAM 2 was paying for
precision nobody consumed.

Pipeline:

  1. **Grounding DINO** (`IDEA-Research/grounding-dino-base`, HF transformers
     `AutoModelForZeroShotObjectDetection`) runs once over every frame with
     a single dot-separated multi-phrase query. Returns per-frame bboxes
     with text labels routed back to scout phrases.

  2. **Detection-grouping** turns the per-frame bbox lists into per-instance
     "tracklets" via SORT-style greedy IoU linking (``_group_into_tracklets``).
     Each tracklet emits one ``Track`` whose ``frames`` carry the per-frame
     bbox + GDINO score.

Public entrypoint :func:`run_gdino` writes ``outputs/<id>/tracks.json`` with:

    {
      "n_input_frames": ...,
      "n_tracks": ...,
      "tracks": [{"track_id", "text_prompt", "source", "frames": [
          {"frame_id", "score", "bbox_2d": [x0, y0, x1, y1]}, ...
      ]}, ...]
    }

Upgrade path: swap ``IDEA-Research/grounding-dino-base`` for
MM-Grounding-DINO if/when it lands as a registered HF architecture
(currently only available via OpenMMLab `mmdet`).
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFile

from ._track_types import Track, TrackFrame
from .reid import cosine_similarity, embed_detections
from .scene_scout import ScopedPrompt

logger = logging.getLogger(__name__)


# At least one frame in the inputs volume has a truncated PNG tail (likely
# from an incomplete upload). Keep the global PIL toggle the HF GDINO
# processor relies on internally.
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ---------------------------------------------------------------------------- model id

_GDINO_MODEL_ID = "IDEA-Research/grounding-dino-base"


# Per-phrase tracklet cap. GDINO often returns slightly-shifted bboxes for
# the same object across frames, which our IoU linker may split into
# multiple short tracklets. Capping per-phrase to the top-K longest
# tracklets prevents one noisy phrase ("upholstered furniture": 24
# tracklets) from monopolising the track budget. 6 (was 3) admits busier
# scenes — a conference room legitimately has 6+ chairs. Real cluttered
# scenes still rely on 3D AABB merge in lift.py to collapse synonyms.
_MAX_TRACKLETS_PER_PHRASE = 6


# Enable Ampere/Hopper TF32 fast paths once at module import. Idempotent
# and harmless on older GPUs.
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# Fallback vocabulary used only when no `text_prompts` are passed AND the
# VLM scene-scout pass is bypassed. The scout replaces it for real runs.
_FALLBACK_PROMPTS: list[str] = [
    "chair", "table", "sofa", "bed", "desk",
    "person", "door", "window",
]


# ---------------------------------------------------------------------------- types


@dataclass
class _Detection:
    """Single GDINO detection at a frame for one phrase.

    ``embed`` is an L2-normalised DINOv2 appearance vector populated by
    :mod:`reid` when re-ID is enabled. The linker uses it to disambiguate
    SORT continuations under fast camera motion.
    """
    bbox_xyxy: tuple[float, float, float, float]
    score: float
    embed: np.ndarray | None = None


@dataclass
class _Tracklet:
    """A single physical instance linked across consecutive frames."""
    frames: dict[int, _Detection] = field(default_factory=dict)  # frame_idx → detection

    def add(self, frame_idx: int, det: _Detection) -> None:
        self.frames[frame_idx] = det

    @property
    def length(self) -> int:
        return len(self.frames)


# ---------------------------------------------------------------------------- helpers


def _iou_xyxy(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    a_area = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    b_area = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = a_area + b_area - inter
    return float(inter / union) if union > 0 else 0.0


# ---------------------------------------------------------------------------- prompt normalization


def _normalize_prompts(
    text_prompts: list[ScopedPrompt] | list[str] | None,
    extra_text_prompts: list[str] | None,
) -> list[ScopedPrompt]:
    """Coerce mixed legacy / typed prompt inputs into a uniform ScopedPrompt list."""
    out: list[ScopedPrompt] = []
    if text_prompts:
        for p in text_prompts:
            if isinstance(p, ScopedPrompt):
                out.append(p)
            elif isinstance(p, str) and p.strip():
                out.append(ScopedPrompt(phrase=p.strip(), frame_range=None))
    else:
        for p in _FALLBACK_PROMPTS:
            out.append(ScopedPrompt(phrase=p, frame_range=None))

    if extra_text_prompts:
        seen = {p.phrase.lower().strip() for p in out}
        for p in extra_text_prompts:
            key = (p or "").lower().strip()
            if key and key not in seen:
                out.append(ScopedPrompt(phrase=p.strip(), frame_range=None))
                seen.add(key)
    return out


# ---------------------------------------------------------------------------- model builder


def _build_gdino():
    """Construct the Grounding DINO model + processor on CUDA, ready for inference."""
    from transformers import (  # noqa: PLC0415
        AutoModelForZeroShotObjectDetection,
        AutoProcessor,
    )

    print(f"[gdino] loading {_GDINO_MODEL_ID} …", flush=True)
    t = time.time()
    processor = AutoProcessor.from_pretrained(_GDINO_MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(_GDINO_MODEL_ID)
    model = model.to("cuda").eval()
    print(f"[gdino] GDINO loaded in {time.time()-t:.1f}s", flush=True)
    return processor, model


# ---------------------------------------------------------------------------- stage 1: GDINO detection


def _canonicalize_label(raw_label: str, phrases_lower: list[str]) -> str | None:
    """Map a raw GDINO `text_labels` entry to the closest scout phrase."""
    raw = raw_label.lower().strip()
    if not raw or raw.startswith("##"):
        return None
    candidates = [p for p in phrases_lower if p in raw]
    if not candidates:
        return None
    return max(candidates, key=len)


def _detect_all_phrases(
    processor,
    model,
    frame_paths: list[Path],
    phrases: list[str],
    score_threshold: float,
    batch_size: int,
    verbose: bool = True,
    log_every_n_batches: int = 4,
) -> dict[str, dict[int, list[_Detection]]]:
    """Run GDINO over every frame for ALL phrases in a single dot-separated query.

    Returns ``{phrase_label: {frame_idx: [Detection, ...]}}``.
    """
    phrases_clean = [p.lower().rstrip(".").strip() for p in phrases]
    phrases_lower_by_len = sorted({p for p in phrases_clean if p}, key=len, reverse=True)
    text_query = ". ".join(phrases_clean) + "."
    out: dict[str, dict[int, list[_Detection]]] = {}
    dropped_unknown_label = 0
    total_batches = (len(frame_paths) + batch_size - 1) // batch_size
    t_start = time.time()
    cum_dets = 0

    dropped_unreadable = 0
    for bi, batch_start in enumerate(range(0, len(frame_paths), batch_size)):
        batch_paths = frame_paths[batch_start:batch_start + batch_size]
        # Defense-in-depth: a 0-byte filter runs upstream, but PIL can also
        # raise on truncated/zip-bombed PNGs that pass the size check. Skip
        # individual unreadable frames rather than crashing the whole sweep.
        images: list = []
        local_indices: list[int] = []  # index within this batch (i in enumerate(results))
        for li, p in enumerate(batch_paths):
            try:
                images.append(Image.open(p).convert("RGB"))
                local_indices.append(li)
            except Exception as e:  # noqa: BLE001
                dropped_unreadable += 1
                logger.warning("skipping unreadable frame %s: %s", p, e)
        if not images:
            continue
        inputs = processor(
            images=images,
            text=[text_query] * len(images),
            return_tensors="pt",
        ).to("cuda")
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = torch.tensor([img.size[::-1] for img in images])
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=score_threshold,
            text_threshold=score_threshold,
            target_sizes=target_sizes,
        )
        for i, res in enumerate(results):
            # Route back to the original frame index (skipping failed loads).
            fidx = batch_start + local_indices[i]
            boxes = res.get("boxes", torch.empty((0, 4))).cpu().tolist()
            scores = res.get("scores", torch.empty((0,))).cpu().tolist()
            labels = res.get("text_labels")
            if labels is None:
                labels = res.get("labels", [None] * len(boxes))
            if not boxes:
                continue
            for box, score, label in zip(boxes, scores, labels):
                if label is None:
                    continue
                key = _canonicalize_label(str(label), phrases_lower_by_len)
                if key is None:
                    dropped_unknown_label += 1
                    continue
                phrase_buckets = out.setdefault(key, {})
                phrase_buckets.setdefault(fidx, []).append(
                    _Detection(bbox_xyxy=tuple(box), score=float(score))
                )
                cum_dets += 1
        if verbose and ((bi + 1) % log_every_n_batches == 0 or (bi + 1) == total_batches):
            frames_done = min(batch_start + batch_size, len(frame_paths))
            elapsed = time.time() - t_start
            it_per_s = frames_done / elapsed if elapsed > 0 else 0.0
            print(
                f"[gdino]     {frames_done}/{len(frame_paths)} frames "
                f"({it_per_s:.1f} it/s, cum_dets={cum_dets}, "
                f"phrases_kept={len(out)}, dropped={dropped_unknown_label})",
                flush=True,
            )
    if verbose and dropped_unknown_label:
        print(f"[gdino] dropped {dropped_unknown_label} detections with "
              f"unrecognized text_labels (cross-phrase token spans)", flush=True)
    if verbose and dropped_unreadable:
        print(f"[gdino] dropped {dropped_unreadable} unreadable frame PNGs "
              f"during batch decode", flush=True)
    return out


# ---------------------------------------------------------------------------- cross-phrase NMS


# Two detections from different phrases at IoU above this threshold are
# treated as the same physical bbox — only the higher-scoring one survives.
# 0.7 is conservative: distinct nearby objects rarely share that much
# bbox overlap, but synonym phrases ("chair" + "office chair") of the same
# chair almost always do.
_CROSS_PHRASE_NMS_IOU = 0.7


def _apply_cross_phrase_nms(
    phrase_to_dets: dict[str, dict[int, list[_Detection]]],
    iou_thresh: float = _CROSS_PHRASE_NMS_IOU,
) -> tuple[dict[str, dict[int, list[_Detection]]], int]:
    """Per-frame NMS across phrases.

    GDINO's multi-phrase query produces one detection per (phrase, bbox).
    A single physical chair often gets detected under several scout
    phrases ("chair", "office chair", "swivel chair") at almost the same
    bbox in the same frame. Without cross-phrase NMS those fan out into
    parallel tracklets after the linker stage, multiplying redundancy.

    For each frame: sort all detections by (score, phrase_length) descending,
    walk in order, and suppress any later detection whose bbox IoU with a
    *different-phrase* survivor exceeds ``iou_thresh``. Same-phrase
    detections at the same frame are left alone — those are handled by
    the per-phrase IoU linker downstream.

    Returns (filtered_phrase_to_dets, n_suppressed).
    """
    by_frame: dict[int, list[tuple[str, _Detection]]] = {}
    for phrase, frame_buckets in phrase_to_dets.items():
        for fidx, dets in frame_buckets.items():
            for d in dets:
                by_frame.setdefault(fidx, []).append((phrase, d))

    out: dict[str, dict[int, list[_Detection]]] = {}
    n_suppressed = 0
    for fidx, items in by_frame.items():
        # Sort by score descending; tie-break by phrase length descending
        # (longer phrase = more specific → preferred winner).
        items.sort(key=lambda x: (x[1].score, len(x[0])), reverse=True)
        keepers: list[tuple[str, _Detection]] = []
        for ph, det in items:
            suppressed = False
            for kp, kd in keepers:
                if kp == ph:
                    # Same phrase — let the per-phrase linker dedupe.
                    continue
                if _iou_xyxy(det.bbox_xyxy, kd.bbox_xyxy) >= iou_thresh:
                    suppressed = True
                    break
            if suppressed:
                n_suppressed += 1
                continue
            keepers.append((ph, det))
        for ph, det in keepers:
            out.setdefault(ph, {}).setdefault(fidx, []).append(det)

    return out, n_suppressed


# ---------------------------------------------------------------------------- scope filter


def _filter_by_scope(
    phrase_to_dets: dict[str, dict[int, list[_Detection]]],
    scoped: list[ScopedPrompt],
    filtered_to_absolute: list[int],
) -> tuple[dict[str, dict[int, list[_Detection]]], int]:
    """Drop per-frame detections whose absolute frame index falls outside the phrase's scout-assigned range.

    ``phrase_to_dets`` keys are positions in gdino's *filtered* frame list
    (Stage-1 frame-selection drops unposed frames). Scout computed
    ``frame_range`` against the *unfiltered* frame list. We map filtered
    fidx → unfiltered idx via ``filtered_to_absolute`` so the comparison
    is apples-to-apples; without this the scope filter mis-aligns by
    however many frames Stage 1 dropped.

    Phrases with frame_range=None (the global safety net like "person")
    pass through untouched. Phrases that don't appear in ``scoped`` (e.g.
    fallback vocabulary) also pass through.
    """
    range_by_phrase: dict[str, tuple[int, int] | None] = {}
    for sp in scoped:
        key = sp.phrase.lower().strip()
        if key:
            range_by_phrase[key] = sp.frame_range

    n_dropped = 0
    out: dict[str, dict[int, list[_Detection]]] = {}
    for phrase, frame_buckets in phrase_to_dets.items():
        scope = range_by_phrase.get(phrase.lower().strip(), None)
        for fidx, dets in frame_buckets.items():
            if scope is not None:
                # fidx is into the filtered list; scout's range is into the
                # unfiltered list. Map via the parallel array.
                if 0 <= fidx < len(filtered_to_absolute):
                    abs_idx = filtered_to_absolute[fidx]
                else:
                    abs_idx = fidx  # defensive — shouldn't happen
                start, end = scope
                if not (start <= abs_idx < end):
                    n_dropped += len(dets)
                    continue
            out.setdefault(phrase, {})[fidx] = dets
    return out, n_dropped


# ---------------------------------------------------------------------------- stage 2: tracklet linking


# Linker scoring weights when DINOv2 embeddings are available. α = IoU
# weight; (1-α) = appearance cosine weight. 0.6 IoU keeps the linker
# geometry-led (fast frames where bboxes overlap heavily are still the
# common case) while letting appearance rescue tracklets that IoU alone
# would split (fast motion, partial occlusion).
_LINKER_IOU_WEIGHT = 0.6


def _group_into_tracklets(
    per_frame_dets: dict[int, list[_Detection]],
    iou_thresh: float,
    gap_tolerance: int,
    min_run_frames: int,
) -> list[_Tracklet]:
    """SORT-style greedy linker. When detections carry DINOv2 embeddings,
    the score becomes ``α · IoU + (1-α) · cosine``; otherwise IoU-only.

    Combined score is compared against ``iou_thresh`` directly — the
    weighted sum is bounded above by 1.0 just like raw IoU, so the same
    threshold acts as a calibrated decision boundary in both modes.
    """
    if not per_frame_dets:
        return []

    sorted_frames = sorted(per_frame_dets.keys())
    closed: list[_Tracklet] = []
    active: list[tuple[int, _Tracklet]] = []  # (last_seen_frame, tracklet)

    for fidx in sorted_frames:
        still_active: list[tuple[int, _Tracklet]] = []
        for last_seen, tr in active:
            if fidx - last_seen > gap_tolerance:
                closed.append(tr)
            else:
                still_active.append((last_seen, tr))
        active = still_active

        for det in per_frame_dets[fidx]:
            best_score, best_idx = 0.0, -1
            for j, (_, tr) in enumerate(active):
                last_fidx = max(tr.frames)
                last_det = tr.frames[last_fidx]
                iou = _iou_xyxy(det.bbox_xyxy, last_det.bbox_xyxy)
                if det.embed is not None and last_det.embed is not None:
                    cos = cosine_similarity(det.embed, last_det.embed)
                    # Cosine ranges roughly [0, 1] for crops of the same
                    # object class; clamp to non-negative to avoid the
                    # rare anti-correlated pair pulling the score below
                    # IoU-only (which would mis-link a worse candidate).
                    cos = max(0.0, cos)
                    score = _LINKER_IOU_WEIGHT * iou + (1.0 - _LINKER_IOU_WEIGHT) * cos
                else:
                    score = iou
                if score > best_score:
                    best_score, best_idx = score, j
            if best_idx >= 0 and best_score >= iou_thresh:
                active[best_idx][1].add(fidx, det)
                active[best_idx] = (fidx, active[best_idx][1])
            else:
                tr = _Tracklet()
                tr.add(fidx, det)
                active.append((fidx, tr))

    closed.extend(tr for _, tr in active)
    return [tr for tr in closed if tr.length >= min_run_frames]


def _tracklets_for_phrase(
    per_frame_dets: dict[int, list[_Detection]],
    *,
    iou_thresh: float = 0.3,
    gap_tolerance: int = 3,
    min_run_frames: int = 8,
    min_bbox_side: int = 16,
    max_tracklets_per_phrase: int = _MAX_TRACKLETS_PER_PHRASE,
) -> list[_Tracklet]:
    """One phrase's per-frame detections → filtered tracklets ready for emission."""
    tracklets = _group_into_tracklets(
        per_frame_dets,
        iou_thresh=iou_thresh,
        gap_tolerance=gap_tolerance,
        min_run_frames=min_run_frames,
    )

    # Drop tracklets whose detections are uniformly tiny — they're usually
    # noise on textured background. We keep a tracklet if its peak-confidence
    # detection has a decent side; smaller frames inside it are fine.
    survivors: list[_Tracklet] = []
    for tr in tracklets:
        peak = max(tr.frames.values(), key=lambda d: d.score)
        x0, y0, x1, y1 = peak.bbox_xyxy
        if min(x1 - x0, y1 - y0) < min_bbox_side:
            continue
        survivors.append(tr)

    # Per-phrase top-K cap: keep the longest tracklets (most temporally
    # supported, lowest false-positive risk).
    if len(survivors) > max_tracklets_per_phrase:
        survivors = sorted(survivors, key=lambda t: t.length, reverse=True)
        survivors = survivors[:max_tracklets_per_phrase]
    return survivors


# ---------------------------------------------------------------------------- stage 3: assembly + writeout


def _tracklet_to_track(
    tracklet: _Tracklet,
    track_id: str,
    phrase: str,
    frame_paths: list[Path],
) -> Track:
    """Convert a tracklet's per-frame detections into a Track of TrackFrames."""
    frames: list[TrackFrame] = []
    for fidx in sorted(tracklet.frames.keys()):
        if fidx >= len(frame_paths):
            continue
        det = tracklet.frames[fidx]
        x0, y0, x1, y1 = det.bbox_xyxy
        frames.append(TrackFrame(
            frame_id=frame_paths[fidx].stem,
            score=float(det.score),
            bbox_2d=(int(x0), int(y0), int(x1), int(y1)),
        ))
    return Track(track_id=track_id, frames=frames, text_prompt=phrase, source="text")


def _write_tracks_json(out_dir: Path, tracks: list[Track], n_input_frames: int) -> None:
    payload = {
        "n_input_frames": n_input_frames,
        "n_tracks": len(tracks),
        "tracks": [
            {
                "track_id": t.track_id,
                "source": t.source,
                "text_prompt": t.text_prompt,
                "frames": [
                    {
                        "frame_id": f.frame_id,
                        "score": f.score,
                        "bbox_2d": list(f.bbox_2d),
                    }
                    for f in t.frames
                ],
            }
            for t in tracks
        ],
    }
    (out_dir / "tracks.json").write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------- public entrypoint


def run_gdino(
    frames_dir: Path,
    out_dir: Path,
    text_prompts: list[ScopedPrompt] | list[str] | None = None,
    extra_text_prompts: list[str] | None = None,
    # ByteTrack-style threshold + temporal calibration (Zhang et al.,
    # ECCV 2022). Lower per-frame threshold admits more borderline true
    # positives (laptops half-occluded, monitors with screen reflections);
    # the bumped `min_track_frames` is the temporal validation that keeps
    # noise out. Approx FP-survival rate: 0.20^5 ≈ 0.0003 vs the prior
    # 0.25^3 ≈ 0.0156 — ~50× lower under the (rough) frame-independence
    # assumption.
    score_threshold: float = 0.20,
    min_track_frames: int = 5,
    gdino_batch_size: int = 8,
    **_unused: Any,  # absorbs legacy seed_stride/reprompt_stride/min_track_score kwargs
) -> list[Track]:
    """Detect with Grounding DINO and emit IoU-linked Tracks (no mask propagation).

    Args:
      frames_dir: directory of ordered frame PNGs (sorted = time order).
      out_dir:   where to write tracks.json.
      text_prompts: scout output. Only the phrase strings are used.
      extra_text_prompts: appended to the prompt list (debug / CLI override).
      score_threshold: GDINO confidence threshold (both box and text head).
      min_track_frames: drop tracks shorter than this after assembly.
      gdino_batch_size: frames per GDINO forward call.
    """
    frame_paths = sorted(frames_dir.iterdir())
    frame_paths = [p for p in frame_paths if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    if not frame_paths:
        raise SystemExit(f"no frames under {frames_dir}")

    # Snapshot the unfiltered frame list — scout's ScopedPrompt.frame_range
    # is indexed against this list (sorted, png/jpg only, but BEFORE any
    # camera/depth/0-byte filtering). The scope filter later uses this to
    # map filtered fidx → absolute idx.
    unfiltered_frame_paths = list(frame_paths)
    unfiltered_idx_by_path: dict[Path, int] = {
        p: i for i, p in enumerate(unfiltered_frame_paths)
    }

    # Resume shortcut: if a prior Stage 2 already wrote tracks.json, parse
    # it back into Track objects and skip GDINO entirely.
    tracks_path = out_dir / "tracks.json"
    if tracks_path.exists():
        try:
            payload = json.loads(tracks_path.read_text())
            existing = payload.get("tracks", [])
        except Exception as e:  # noqa: BLE001
            logger.warning("could not parse existing tracks.json: %s", e)
            existing = []
        if existing:
            print(f"[gdino] resuming from existing tracks.json "
                  f"({len(existing)} tracks; skipping GDINO)", flush=True)
            tracks: list[Track] = []
            for t in existing:
                frames = [
                    TrackFrame(
                        frame_id=f["frame_id"],
                        score=float(f.get("score", 1.0)),
                        bbox_2d=tuple(f["bbox_2d"]),
                    )
                    for f in t.get("frames", [])
                ]
                tracks.append(Track(
                    track_id=t["track_id"],
                    frames=frames,
                    text_prompt=t.get("text_prompt"),
                    source=t.get("source", "text"),
                ))
            return tracks

    # Filter to frames that have everything the lifter will need: a camera
    # pose (cameras.json), a depth map, and a depth-confidence map.
    cameras_path = out_dir / "cameras.json"
    depth_dir = out_dir / "depth"
    conf_dir = out_dir / "depth_conf"
    if cameras_path.exists() and depth_dir.exists() and conf_dir.exists():
        cam_ids = {
            f["frame_id"] for f in json.loads(cameras_path.read_text()).get("frames", [])
        }
        depth_ids = {p.stem for p in depth_dir.iterdir() if p.suffix == ".npy"}
        conf_ids = {p.stem for p in conf_dir.iterdir() if p.suffix == ".npy"}
        valid_ids = cam_ids & depth_ids & conf_ids
        before = len(frame_paths)
        frame_paths = [p for p in frame_paths if p.stem in valid_ids]
        dropped = before - len(frame_paths)
        if dropped:
            print(f"[gdino] filtered {before} → {len(frame_paths)} frames "
                  f"(dropped {dropped} without camera/depth — Stage 1 frame-selection)",
                  flush=True)
        if not frame_paths:
            raise SystemExit(
                f"no frames in {frames_dir} have camera/depth/conf — "
                f"run Stage 1 inference first"
            )

    # Drop 0-byte / unreadable PNGs. Inference's frame writer can leave
    # empty files behind on partial-disk failures; PIL.Image.open raises
    # UnidentifiedImageError on those mid-batch and crashes the entire
    # GDINO sweep. Pre-filter is O(N) stat() calls — negligible.
    before = len(frame_paths)
    frame_paths = [p for p in frame_paths if p.stat().st_size > 0]
    dropped = before - len(frame_paths)
    if dropped:
        print(f"[gdino] dropped {dropped} 0-byte / corrupt frame PNGs", flush=True)

    # Stage-1 presence check (separate from the 0-byte filter above; the
    # previous version raised here as a side-effect of the `else` branch
    # whenever no PNGs were dropped, even when Stage 1 was healthy).
    missing = [p for p in (cameras_path, depth_dir, conf_dir) if not p.exists()]
    if missing:
        raise SystemExit(
            f"Stage 1 outputs missing: {[str(m) for m in missing]} — "
            f"run inference first"
        )

    n_frames = len(frame_paths)
    scoped = _normalize_prompts(text_prompts, extra_text_prompts)
    print(f"[gdino] {n_frames} frames; {len(scoped)} prompts; threshold={score_threshold}",
          flush=True)

    # Wipe any stale outputs from a prior SAM 2 run on the same scene.
    legacy_dirs = [
        out_dir / "masks",
        out_dir / "sam3",
        out_dir / "_sam2_frames",
        out_dir / "_sam3_slice_tmp",
    ]
    for stale in legacy_dirs:
        if stale.exists():
            shutil.rmtree(stale, ignore_errors=True)

    # ── Stage 1: single-pass multi-phrase detection ───────────────────────
    proc, gdino = _build_gdino()
    phrases = [prompt.phrase for prompt in scoped]
    print(f"[gdino] running multi-phrase GDINO sweep over {n_frames} frames "
          f"({len(phrases)} phrases in one query)…", flush=True)
    t_detect = time.time()
    phrase_to_dets = _detect_all_phrases(
        proc, gdino, frame_paths, phrases, score_threshold, gdino_batch_size,
    )
    print(f"[gdino] sweep done in {time.time()-t_detect:.1f}s — "
          f"{sum(len(b) for b in phrase_to_dets.values())} mask-frame entries across "
          f"{len(phrase_to_dets)} distinct phrase labels", flush=True)

    # Honour scout's per-slice frame ranges — drops out-of-scope detections
    # that GDINO over-fires on (visually similar but in a different slice
    # of the walkthrough). No-op for phrases with frame_range=None.
    filtered_to_absolute = [
        unfiltered_idx_by_path.get(p, i) for i, p in enumerate(frame_paths)
    ]
    n_in_scope = sum(len(b) for d in phrase_to_dets.values() for b in d.values())
    phrase_to_dets, n_scope_dropped = _filter_by_scope(
        phrase_to_dets, scoped, filtered_to_absolute,
    )
    n_after_scope = sum(len(b) for d in phrase_to_dets.values() for b in d.values())
    print(f"[gdino] scope filter: {n_in_scope} → {n_after_scope} detections "
          f"(dropped {n_scope_dropped} outside scout's frame ranges)", flush=True)

    # Cross-phrase NMS: same physical bbox detected under multiple synonym
    # phrases would otherwise produce parallel tracklets. Suppress before
    # the linker runs.
    n_before = sum(len(b) for d in phrase_to_dets.values() for b in d.values())
    phrase_to_dets, n_suppressed = _apply_cross_phrase_nms(phrase_to_dets)
    n_after = sum(len(b) for d in phrase_to_dets.values() for b in d.values())
    print(f"[gdino] cross-phrase NMS @ IoU≥{_CROSS_PHRASE_NMS_IOU}: "
          f"{n_before} → {n_after} detections "
          f"(suppressed {n_suppressed} duplicates from synonym phrases)",
          flush=True)

    # Free GDINO before any other GPU work.
    del gdino
    del proc
    torch.cuda.empty_cache()

    # ── Re-ID: per-detection appearance embeddings (DINOv2-small) ─────────
    # Populates _Detection.embed in-place so the linker scores combined
    # IoU + cosine similarity. Skipped silently if reid is disabled or
    # the model fails to load — linker then falls back to pure IoU.
    detection_index: dict[int, list[tuple[str, tuple[float, float, float, float], int]]] = {}
    # Build a stable per-(frame, phrase) ordering so the embedding lookup
    # key matches the detection at the same position.
    for phrase, frame_buckets in phrase_to_dets.items():
        for fidx, dets in frame_buckets.items():
            for di, det in enumerate(dets):
                detection_index.setdefault(fidx, []).append(
                    (phrase, det.bbox_xyxy, di)
                )
    embeds = embed_detections(frame_paths, detection_index)
    if embeds:
        # Annotate the detections in place. The (fidx, phrase, di) key is
        # stable because we only append to the buckets — order is
        # preserved between detection_index construction and the
        # subsequent linker walk.
        for phrase, frame_buckets in phrase_to_dets.items():
            for fidx, dets in frame_buckets.items():
                for di, det in enumerate(dets):
                    e = embeds.get((fidx, phrase, di))
                    if e is not None:
                        det.embed = e

    # ── Stage 2: per-phrase tracklet linking → Track objects ──────────────
    tracks: list[Track] = []
    next_obj_id = 1
    for label, per_frame_dets in phrase_to_dets.items():
        tracklets = _tracklets_for_phrase(per_frame_dets)
        n_dets = sum(len(d) for d in per_frame_dets.values())
        n_frames_with_dets = len(per_frame_dets)
        print(f"[gdino]   '{label}': {n_dets} detections in "
              f"{n_frames_with_dets} frames → {len(tracklets)} tracklets",
              flush=True)
        for tr in tracklets:
            track_id = f"obj_{next_obj_id:04d}"
            next_obj_id += 1
            tracks.append(_tracklet_to_track(tr, track_id, label, frame_paths))

    # Final length filter (defensive — _tracklets_for_phrase already drops
    # below min_run_frames=8, but min_track_frames may be tighter or looser).
    short_drops = [t for t in tracks if len(t.frames) < min_track_frames]
    tracks = [t for t in tracks if len(t.frames) >= min_track_frames]
    tracks.sort(key=lambda t: t.track_id)
    print(f"[gdino] {len(tracks)} tracks after min_frames={min_track_frames} filter "
          f"(dropped {len(short_drops)} short tracklets)", flush=True)

    # Persist short-tracklet drops so the UI's Discarded tab can show them
    # alongside lift-stage and postprocess-stage drops. Each record carries
    # `stage` + `discard_reason` so the frontend can group them. Geometry
    # fields are absent here (no 3D yet — these never made it past Stage 2).
    short_records = [
        {
            "id": t.track_id,
            "label": t.text_prompt or t.track_id,
            "stage": "gdino",
            "discard_reason": "short_tracklet",
            "discard_detail": (
                f"tracklet survived only {len(t.frames)} frame(s); "
                f"min_track_frames={min_track_frames}."
            ),
            "n_frames": len(t.frames),
            "frame_ids": [f"{tf.frame_id}.png" for tf in t.frames],
            "source": t.source,
        }
        for t in short_drops
    ]
    (out_dir / "_gdino_discards.json").write_text(json.dumps(short_records, indent=2))

    _write_tracks_json(out_dir, tracks, n_input_frames=n_frames)
    print(f"[gdino] wrote tracks.json ({len(tracks)} tracks, {n_frames} frames)",
          flush=True)
    return tracks
