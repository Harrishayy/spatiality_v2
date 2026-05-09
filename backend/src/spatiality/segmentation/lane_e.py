"""Lane E — ConceptGraphs-style scene graph (objects + relations).

Reuses Lane B's verified labels and adds spatial-relationship edges between
nearby objects, predicted by Gemini 2.5 Flash via PydanticAI from a top-down
render + per-pair close-up crops. Output is `annotations.e.json` with the
same Annotation shape as Lane B plus an `edges` array.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from .lift import LiftedTrack
from .render import load_points_ply, orbital_poses, render_view
from .vlm import call_vlm

logger = logging.getLogger(__name__)


Relation = Literal[
    "on", "under", "next-to", "contains", "supports",
    "behind", "in-front-of", "none",
]


class RelationOutput(BaseModel):
    """Structured response from the relation VLM."""

    relation: Relation = Field(description="Spatial relationship of A to B.")
    confidence: float = Field(ge=0.0, le=1.0)


_PROMPT = """\
Two distinct objects are visible in the views: A (label='{label_a}') and B (label='{label_b}').

The first image is a top-down render of the room with both objects in their\
 actual 3D positions. The next two images are orbital close-ups of A and B individually.

What is the spatial relationship of A relative to B? Choose exactly one of:\
 on, under, next-to, contains, supports, behind, in-front-of, none.\

Return your confidence in [0, 1].\
"""


def _topdown_render(
    points_path: Path, image_size: tuple[int, int] = (640, 640)
) -> np.ndarray:
    """Single top-down (looking +y) render of the whole cloud."""
    xyz, rgb, _ = load_points_ply(points_path)
    if not len(xyz):
        return np.full((*image_size, 3), 24, dtype=np.uint8)

    centre = xyz.mean(axis=0)
    extent = np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0)) / 2.0
    eye = centre + np.array([0.0, -extent * 1.8, 0.0], dtype=np.float32)
    target = centre.astype(np.float32)
    up_world = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    fwd = target - eye; fwd /= max(1e-8, np.linalg.norm(fwd))
    right = np.cross(fwd, up_world); right /= max(1e-8, np.linalg.norm(right))
    new_up = np.cross(right, fwd)
    R = np.stack([right, -new_up, fwd], axis=0)
    t = -R @ eye
    extrinsic = np.concatenate([R, t.reshape(3, 1)], axis=1)
    return render_view(xyz, rgb, extrinsic, image_size=image_size, fov_deg=70.0)


def _track_closeup(
    points_path: Path, track: LiftedTrack, image_size: tuple[int, int] = (320, 320)
) -> np.ndarray:
    xyz, rgb, _ = load_points_ply(points_path)
    aabb_min = track.obb_corners.min(axis=0)
    aabb_max = track.obb_corners.max(axis=0)
    half_extent = (aabb_max - aabb_min) / 2.0 * 1.4
    centre = (aabb_min + aabb_max) / 2.0
    lo = centre - half_extent
    hi = centre + half_extent

    inside = (
        (xyz[:, 0] >= lo[0]) & (xyz[:, 0] <= hi[0]) &
        (xyz[:, 1] >= lo[1]) & (xyz[:, 1] <= hi[1]) &
        (xyz[:, 2] >= lo[2]) & (xyz[:, 2] <= hi[2])
    )
    if inside.sum() < 50:
        focus = xyz; focus_rgb = rgb
    else:
        focus = xyz[inside]; focus_rgb = rgb[inside]

    pose = orbital_poses(track.centroid.astype(np.float32),
                         radius=float(np.linalg.norm(half_extent)) * 1.6, n=1)[0]
    return render_view(focus, focus_rgb, pose, image_size=image_size, fov_deg=50.0)


# ---------------------------------------------------------------------------- entry point


def run_lane_e(
    lifted_tracks: list[LiftedTrack],
    lane_b_annotations: list[dict],
    out_dir: Path,
    vlm_model: str = "gemini-2.5-flash",
    distance_threshold: float = 2.0,
    confidence_threshold: float = 0.5,
) -> dict:
    """Produce per-pair relation edges, layered over Lane B's annotations."""
    import time as _time
    points_path = out_dir / "points.ply"
    topdown = _topdown_render(points_path)

    annotations_by_id = {a["id"]: a for a in lane_b_annotations}

    closeups: dict[str, np.ndarray] = {}

    edges: list[dict] = []
    n = len(lifted_tracks)
    n_pairs_total = n * (n - 1)
    n_evaluated = 0
    n_calls = 0
    print(f"[lane_e] {n} lifted tracks → up to {n_pairs_total} directed pairs "
          f"(filter: dist≤{distance_threshold}m, both labelled, conf≥{confidence_threshold})", flush=True)
    _t = _time.time()

    for i in range(n):
        ti = lifted_tracks[i]
        for j in range(n):
            if i == j:
                continue
            n_evaluated += 1
            tj = lifted_tracks[j]
            dist = float(np.linalg.norm(ti.centroid - tj.centroid))
            if dist > distance_threshold:
                continue

            label_a = annotations_by_id.get(ti.track_id, {}).get("label", "unknown")
            label_b = annotations_by_id.get(tj.track_id, {}).get("label", "unknown")
            if "unknown" in (label_a, label_b):
                continue

            if ti.track_id not in closeups:
                closeups[ti.track_id] = _track_closeup(points_path, ti)
            if tj.track_id not in closeups:
                closeups[tj.track_id] = _track_closeup(points_path, tj)

            n_calls += 1
            _t_call = _time.time()
            try:
                reply = call_vlm(
                    _PROMPT.format(label_a=label_a, label_b=label_b),
                    [topdown, closeups[ti.track_id], closeups[tj.track_id]],
                    RelationOutput,
                    model=vlm_model,
                )
            except Exception as e:  # noqa: BLE001
                print(f"[lane_e]   call {n_calls}: ({ti.track_id}->{tj.track_id}) "
                      f"VLM FAILED: {e}", flush=True)
                logger.warning("relation VLM failed for (%s,%s): %s",
                               ti.track_id, tj.track_id, e)
                continue

            kept = reply.relation != "none" and reply.confidence >= confidence_threshold
            print(f"[lane_e]   call {n_calls}: {label_a}({ti.track_id})→{label_b}({tj.track_id}) "
                  f"= {reply.relation} conf={reply.confidence:.2f} "
                  f"{'KEPT' if kept else 'dropped'} ({_time.time()-_t_call:.1f}s)", flush=True)
            if not kept:
                continue

            edges.append(
                {
                    "from": ti.track_id,
                    "to": tj.track_id,
                    "relation": reply.relation,
                    "confidence": reply.confidence,
                }
            )

    print(f"[lane_e] evaluated {n_evaluated} pairs, {n_calls} VLM calls, "
          f"{len(edges)} kept edges ({_time.time()-_t:.1f}s)", flush=True)

    payload = {
        "annotations": lane_b_annotations,
        "edges": edges,
    }
    (out_dir / "annotations.e.json").write_text(json.dumps(payload, indent=2))
    print(f"[lane_e] wrote annotations.e.json ({len(edges)} edges over "
          f"{len(lane_b_annotations)} annotations)", flush=True)
    return payload
