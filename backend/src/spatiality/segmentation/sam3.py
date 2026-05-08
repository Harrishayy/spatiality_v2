"""SAM 3.1 detection + video tracking with Object Multiplex.

The "same object 3×" bug in the original spatiality came from running per-frame
detection without cross-frame association. SAM 3.1 has a memory-based video
tracker that shares the backbone with the detector — once an object gets a
``track_id``, it keeps it across all the frames it appears in.

This module implements the robust multi-frame strategy from the plan:
  1. seed pass: detector on every Kth frame in "everything" mode
  2. mask-prompt each detection into a video session
  3. bidirectional propagation (forward + backward from each seed frame)
  4. re-prompt cadence to catch objects that appear mid-clip
  5. Object Multiplex bundles up to 16 obj_ids per forward pass

Outputs:
  outputs/<id>/sam3/tracks.json
  outputs/<id>/masks/<track_id>/<frame>.png
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------- types


@dataclass
class TrackFrame:
    frame_id: str
    mask_path: str          # relative to the artefacts root for this scene
    score: float
    bbox_2d: tuple[int, int, int, int]  # (x0, y0, x1, y1)


@dataclass
class Track:
    track_id: str
    frames: list[TrackFrame] = field(default_factory=list)
    text_prompt: str | None = None
    source: str = "open_set"  # "open_set" | "text" | "reprompt"


# ---------------------------------------------------------------------------- model builders


def _build_image_predictor():
    """Image-mode detector for the seed-pass and re-prompt steps."""
    from sam3.model_builder import build_sam3_image_predictor  # type: ignore[import-not-found]

    return build_sam3_image_predictor(model_id="facebook/sam3.1")


def _build_video_predictor():
    """Video tracker with Object Multiplex enabled — 7× speedup at scale."""
    from sam3.model_builder import build_sam3_video_predictor  # type: ignore[import-not-found]

    try:
        return build_sam3_video_predictor(
            model_id="facebook/sam3.1",
            enable_object_multiplex=True,
        )
    except TypeError:
        # Older SAM 3 (pre-3.1) signature: kwargs differ. Fall back without OM.
        logger.warning("Object Multiplex unavailable; falling back to per-object passes")
        return build_sam3_video_predictor(model_id="facebook/sam3")


# ---------------------------------------------------------------------------- core helpers


def _save_mask(mask: np.ndarray, out_path: Path) -> None:
    """Persist a binary mask as a single-channel PNG (255 inside, 0 outside)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(out_path)


def _bbox_of_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _iou_2d(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


# ---------------------------------------------------------------------------- main


def run_sam3(
    frames_dir: Path,
    out_dir: Path,
    seed_stride: int = 25,
    reprompt_stride: int = 100,
    min_track_frames: int = 3,
    min_track_score: float = 0.4,
    extra_text_prompts: list[str] | None = None,
) -> list[Track]:
    """Run the SAM 3.1 detection + tracking pipeline over the frames directory.

    Args:
      frames_dir: directory of ordered frame PNGs (named so sort gives time order).
      out_dir: where to write masks and tracks.json.
      seed_stride: detector cadence in frames (every Kth frame for the seed pass).
      reprompt_stride: detector cadence for catching mid-clip new objects.
      min_track_frames: drop tracks shorter than this.
      min_track_score: drop tracks with mean score below this.
      extra_text_prompts: optional list of text phrases for the taxonomy lane.

    Returns:
      List of Track objects (filtered, persistent obj_ids).
    """
    frame_paths = sorted(frames_dir.iterdir())
    frame_paths = [p for p in frame_paths if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    if not frame_paths:
        raise SystemExit(f"no frames under {frames_dir}")

    masks_dir = out_dir / "masks"
    sam_dir = out_dir / "sam3"
    sam_dir.mkdir(parents=True, exist_ok=True)

    image_predictor = _build_image_predictor()
    video_predictor = _build_video_predictor()

    # 1. Start a video session over the frames sequence.
    session_resp = video_predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=str(frames_dir),
        )
    )
    session_id = session_resp["session_id"]

    obj_id_counter = 0
    obj_id_to_label: dict[int, str | None] = {}

    # 2. Seed pass: open-set detector on every seed_stride-th frame.
    seed_indices = list(range(0, len(frame_paths), seed_stride))
    logger.info("SAM 3.1 seed pass: %d frames at stride %d", len(seed_indices), seed_stride)

    for fidx in seed_indices:
        path = frame_paths[fidx]
        image_np = np.array(Image.open(path).convert("RGB"))

        det = image_predictor.predict_everything(image_np)
        # det: list of {"mask": HxW bool, "score": float, "bbox": (x0,y0,x1,y1)}
        for d in det:
            obj_id_counter += 1
            video_predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=fidx,
                    obj_id=obj_id_counter,
                    mask=d["mask"].astype(bool),
                )
            )
            obj_id_to_label[obj_id_counter] = None

    # 3. Optional taxonomy lane: text-prompted obj_ids at frame 0.
    if extra_text_prompts:
        for phrase in extra_text_prompts:
            obj_id_counter += 1
            video_predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=0,
                    obj_id=obj_id_counter,
                    text=phrase,
                )
            )
            obj_id_to_label[obj_id_counter] = phrase

    # 4. Bidirectional propagation across the full clip. Object Multiplex
    # bundles up to 16 obj_ids per forward pass under the hood.
    video_predictor.handle_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
            direction="both",
        )
    )

    # 5. Re-prompt cadence: catch objects that appear mid-clip after the seed pass.
    reprompt_indices = list(range(seed_stride // 2, len(frame_paths), reprompt_stride))
    for fidx in reprompt_indices:
        if fidx in seed_indices:
            continue
        path = frame_paths[fidx]
        image_np = np.array(Image.open(path).convert("RGB"))
        det = image_predictor.predict_everything(image_np)

        # Gather propagated masks at this frame so we can IoU-match against existing tracks.
        cur_resp = video_predictor.handle_request(
            request=dict(
                type="get_outputs",
                session_id=session_id,
                frame_indices=[fidx],
            )
        )
        existing = cur_resp.get("outputs", {}).get(fidx, {})  # obj_id -> mask
        existing_masks = {oid: np.asarray(m, dtype=bool) for oid, m in existing.items()}

        for d in det:
            new_mask = d["mask"].astype(bool)
            best_iou = max(
                (_iou_2d(new_mask, m) for m in existing_masks.values()),
                default=0.0,
            )
            if best_iou >= 0.3:
                continue  # already covered by a propagated track

            obj_id_counter += 1
            video_predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=fidx,
                    obj_id=obj_id_counter,
                    mask=new_mask,
                )
            )
            obj_id_to_label[obj_id_counter] = None

    # If new prompts were added, propagate again so the new obj_ids extend
    # forward + backward across the clip.
    if obj_id_counter > len(seed_indices):
        video_predictor.handle_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
                direction="both",
            )
        )

    # 6. Pull final outputs across all frames and write masks + tracks.json.
    out_resp = video_predictor.handle_request(
        request=dict(
            type="get_outputs",
            session_id=session_id,
            frame_indices=list(range(len(frame_paths))),
        )
    )
    outputs = out_resp.get("outputs", {})  # frame_idx -> {obj_id: mask}

    tracks: dict[int, Track] = {}
    for fidx, masks in outputs.items():
        frame_path = frame_paths[fidx]
        frame_id = frame_path.stem
        for obj_id, mask in masks.items():
            mask = np.asarray(mask, dtype=bool)
            if not mask.any():
                continue
            score = float(masks.get(f"{obj_id}_score", 1.0)) if isinstance(masks, dict) else 1.0
            bbox = _bbox_of_mask(mask)
            if bbox is None:
                continue

            track_id = f"obj_{obj_id:04d}"
            mask_path = masks_dir / track_id / f"{frame_id}.png"
            _save_mask(mask, mask_path)

            tf = TrackFrame(
                frame_id=frame_id,
                mask_path=str(mask_path.relative_to(out_dir)),
                score=score,
                bbox_2d=bbox,
            )
            track = tracks.setdefault(
                obj_id,
                Track(
                    track_id=track_id,
                    text_prompt=obj_id_to_label.get(obj_id),
                    source="text" if obj_id_to_label.get(obj_id) else "open_set",
                ),
            )
            track.frames.append(tf)

    # Filtering.
    final: list[Track] = []
    for tr in tracks.values():
        if len(tr.frames) < min_track_frames:
            continue
        mean_score = sum(f.score for f in tr.frames) / len(tr.frames)
        if mean_score < min_track_score:
            continue
        tr.frames.sort(key=lambda f: f.frame_id)
        final.append(tr)

    final.sort(key=lambda t: t.track_id)

    tracks_path = sam_dir / "tracks.json"
    tracks_path.write_text(
        json.dumps(
            {
                "tracks": [
                    {
                        "track_id": t.track_id,
                        "source": t.source,
                        "text_prompt": t.text_prompt,
                        "frames": [
                            {
                                "frame_id": f.frame_id,
                                "mask_path": f.mask_path,
                                "score": f.score,
                                "bbox_2d": list(f.bbox_2d),
                            }
                            for f in t.frames
                        ],
                    }
                    for t in final
                ],
                "n_input_frames": len(frame_paths),
                "n_tracks": len(final),
            },
            indent=2,
        )
    )

    # Cleanup (best-effort).
    try:
        video_predictor.handle_request(
            request=dict(type="close_session", session_id=session_id)
        )
    except Exception:  # noqa: BLE001
        pass

    logger.info("SAM 3.1 produced %d tracks across %d frames", len(final), len(frame_paths))
    return final
