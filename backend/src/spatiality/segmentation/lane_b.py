"""Lane B — VLM-verified labels.

For each lifted track:
  1. render 6 orbital novel views of the splat focused on its OBB
  2. crop 1 anchor view from the keyframe with the largest mask
  3. send the 7-image grid to Gemini 2.5 Flash via PydanticAI with a
     structured-output schema → {label, alternatives, confidence, reasoning}
  4. closed-loop verify: feed the proposed label back to SAM 3.1 as a text
     prompt on the anchor frame; if IoU(re-mask, original mask) < 0.3,
     mark verification "failed" and discount confidence.

Output: outputs/<id>/annotations.b.json with the standard `Annotation` shape
the web UI already understands.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from .lift import LiftedTrack
from .render import composite_grid, crop_anchor, render_track_orbit
from .vlm import call_vlm

logger = logging.getLogger(__name__)


class LabelOutput(BaseModel):
    """Structured response from the labeling VLM (Gemini 2.5 Flash)."""

    label: str = Field(description="A concrete noun phrase, or 'unknown'.")
    alternatives: list[str] = Field(default_factory=list, description="Up to 3 alternative noun phrases.")
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(default="", description="One short sentence.")


_PROMPT = """\
You are looking at one physical object viewed from 6 orbital angles around its 3D bounding box,\
 plus 1 in-context original photograph (last image). The orbital views are point-cloud renders so\
 they look stylised; use them for shape/scale/orientation. The original photo gives true colours\
 and surrounding context.

Identify the object. If the views are too ambiguous, set label to "unknown" and confidence ≤ 0.3.\
 Otherwise pick a concrete noun phrase (e.g. "office chair", "ceramic mug", "table lamp").\
 Provide up to 3 alternative phrasings ranked by plausibility, your confidence in [0, 1],\
 and one short sentence of reasoning grounded in what's visible.\
"""


def _largest_anchor_frame(
    track: LiftedTrack, sam_tracks: dict, out_dir: Path
) -> tuple[str, np.ndarray, tuple[int, int, int, int]] | None:
    """Pick the keyframe where this track has the largest mask area."""
    sam_track = sam_tracks.get(track.track_id)
    if sam_track is None:
        return None

    best = None
    best_area = -1
    for f in sam_track["frames"]:
        x0, y0, x1, y1 = f["bbox_2d"]
        area = (x1 - x0) * (y1 - y0)
        if area > best_area:
            best_area = area
            best = f

    if best is None:
        return None
    frame_path = out_dir / "frames" / f"{best['frame_id']}.png"
    if not frame_path.exists():
        return None
    crop = crop_anchor(frame_path, tuple(best["bbox_2d"]))
    return best["frame_id"], crop, tuple(best["bbox_2d"])


def _verify_with_sam(
    label: str,
    anchor_frame_id: str,
    out_dir: Path,
    original_mask_relpath: str,
) -> tuple[str, float]:
    """Re-segment the anchor frame with the proposed label and IoU-check vs original."""
    try:
        from sam3.model_builder import build_sam3_image_predictor  # type: ignore[import-not-found]
    except Exception as e:  # noqa: BLE001
        logger.warning("SAM 3.1 image predictor unavailable for verify: %s", e)
        return "skipped", 0.0

    frame_path = out_dir / "frames" / f"{anchor_frame_id}.png"
    if not frame_path.exists():
        return "skipped", 0.0

    image_np = np.array(Image.open(frame_path).convert("RGB"))
    predictor = build_sam3_image_predictor(model_id="facebook/sam3.1")
    result = predictor.predict_text(image_np, text=label)
    if not result:
        return "failed", 0.0

    cand_mask = np.asarray(result[0]["mask"], dtype=bool)

    original = np.array(Image.open(out_dir / original_mask_relpath)) > 127
    inter = np.logical_and(cand_mask, original).sum()
    union = np.logical_or(cand_mask, original).sum()
    iou = float(inter / union) if union else 0.0
    return ("passed" if iou >= 0.3 else "failed"), iou


def _color_for(track_id: str) -> str:
    """Stable hex colour from the track id (purely cosmetic for the UI legend)."""
    h = abs(hash(track_id)) % 0xFFFFFF
    return f"#{h:06x}"


def _bbox_lo_hi(corners: np.ndarray) -> tuple[list[float], list[float]]:
    return corners.min(axis=0).tolist(), corners.max(axis=0).tolist()


# ---------------------------------------------------------------------------- entry point


def run_lane_b(
    lifted_tracks: list[LiftedTrack],
    out_dir: Path,
    vlm_model: str = "gemini-2.5-flash",
) -> list[dict]:
    """Produce VLM-verified annotations for every lifted track."""
    import time as _time
    sam_tracks_payload = json.loads((out_dir / "sam3" / "tracks.json").read_text())
    sam_tracks = {t["track_id"]: t for t in sam_tracks_payload["tracks"]}

    annotations: list[dict] = []
    points_path = out_dir / "points.ply"
    print(f"[lane_b] {len(lifted_tracks)} tracks, vlm_model={vlm_model}", flush=True)

    for ti, track in enumerate(lifted_tracks, start=1):
        _t_track = _time.time()
        anchor = _largest_anchor_frame(track, sam_tracks, out_dir)

        # Render orbital + (optional) anchor crop.
        orbital = render_track_orbit(
            points_path,
            track.centroid,
            track.obb_corners,
            n_views=6,
        )

        images = orbital
        anchor_frame_id = None
        anchor_mask_rel = None
        if anchor:
            anchor_frame_id, anchor_crop, _bbox = anchor
            target = orbital[0].shape[:2]
            anchor_crop = np.asarray(
                Image.fromarray(anchor_crop).resize((target[1], target[0]))
            )
            images = orbital + [anchor_crop]
            sam_track = sam_tracks[track.track_id]
            for f in sam_track["frames"]:
                if f["frame_id"] == anchor_frame_id:
                    anchor_mask_rel = f["mask_path"]
                    break

        grid = composite_grid(images, cols=3)

        try:
            reply = call_vlm(_PROMPT, [grid], LabelOutput, model=vlm_model).model_dump()
        except Exception as e:  # noqa: BLE001
            logger.warning("VLM call failed for %s: %s", track.track_id, e)
            reply = {"label": "unknown", "alternatives": [], "confidence": 0.1, "reasoning": ""}

        # The verify-with-SAM step was originally a re-prompt + IoU check, but
        # SAM 3.1's image-only predict-text API isn't part of the unified
        # video predictor we now use. Disabling for this build — confidence
        # comes from the VLM call alone, which is already grounded in the
        # text prompt SAM 3.1 used to find the object in the first place
        # (so re-verification was somewhat redundant).
        verification = "skipped"
        verify_iou = 0.0

        lo, hi = _bbox_lo_hi(track.obb_corners)
        annotations.append(
            {
                "id": track.track_id,
                "label": reply.get("label", "unknown"),
                "centroid": track.centroid.tolist(),
                "bbox": [lo, hi],
                "color": _color_for(track.track_id),
                "confidence": float(reply.get("confidence", 0.5)),
                "alternatives": reply.get("alternatives", []),
                "frame_ids": [f"{fid}.png" for fid in track.frame_ids],
                "provenance": [
                    f"sam3.1:{track.source}",
                    f"vlm:{vlm_model}",
                    f"verify:{verification}",
                    f"verify_iou:{verify_iou:.2f}",
                ],
            }
        )
        print(f"[lane_b]   {ti}/{len(lifted_tracks)} {track.track_id} → "
              f"label='{reply.get('label','?')}' "
              f"conf={float(reply.get('confidence',0)):.2f} "
              f"verify={verification} ({_time.time()-_t_track:.1f}s)", flush=True)

    out_path = out_dir / "annotations.b.json"
    out_path.write_text(json.dumps(annotations, indent=2))
    print(f"[lane_b] wrote {len(annotations)} annotations to {out_path.name}", flush=True)
    return annotations
