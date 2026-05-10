"""Lane C — whole-scene coherence review.

Lane B labels each track in isolation. Lane C does a second VLM pass on
the *full annotated scene* so Gemini can:

  - flag implausible labels given the spatial neighbours
    ("a 'laptop' floating in mid-air next to a 'ceiling fan' is suspect")
  - merge duplicates that survived per-class clustering
  - drop tracks that the whole-scene context reveals as background
  - propose parent-child relations ("monitor on desk")

It's one Gemini Flash call per scene. Inputs:
  - top-down render of the point cloud
  - JSON list of (id, label, centroid, extents) for every Lane B annotation

Output is a structured ``LaneCCorrections`` object whose corrections are
applied to the Lane B annotations to produce ``annotations.c.json``.

Checkpointed: if ``annotations.c.json`` already exists, this stage is
skipped on resume. The Lane B output (``annotations.b.json``) remains the
fallback if Lane C fails or is disabled.
"""

from __future__ import annotations

import json
import logging
import time as _time
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from .postprocess import _labels_compatible
from .render import load_points_ply, render_view
from .vlm import call_vlm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------- structured output


class _Relabel(BaseModel):
    id: str
    new_label: str = Field(description="A concrete, instance-level noun phrase.")
    reason: str = Field(default="", description="One short sentence.")


class _Drop(BaseModel):
    id: str
    reason: str = Field(default="", description="One short sentence.")


class _Merge(BaseModel):
    keep_id: str = Field(description="The annotation id whose label/colour survive.")
    drop_ids: list[str] = Field(description="Ids whose data is folded into keep_id.")
    reason: str = Field(default="", description="One short sentence.")


class _Relation(BaseModel):
    subject_id: str
    relation: str = Field(
        description="One of: on, under, contains, supports, next-to, behind, in-front-of."
    )
    object_id: str


class LaneCCorrections(BaseModel):
    """All edits Gemini wants applied to the Lane B output."""

    relabels: list[_Relabel] = Field(default_factory=list)
    drops: list[_Drop] = Field(default_factory=list)
    merges: list[_Merge] = Field(default_factory=list)
    relations: list[_Relation] = Field(default_factory=list)
    summary: str = Field(default="", description="One sentence describing the scene.")


# ---------------------------------------------------------------------------- prompt


_PROMPT = """\
You are reviewing a labelled 3D scan of an indoor scene. Image: a top-down \
render of the full point cloud. JSON below: every object my detector found \
in this scene, each with id, label, centroid (x,y,z metres), and bbox extents.

Your job is to spot mistakes only the *whole scene* makes obvious. Apply \
the lightest edit that fixes each issue. Be strict — if nothing is wrong, \
return empty lists. The scene labels are usually right; you are a reviewer, \
not a re-labeller.

Issue types:

1. relabels — the label is implausible given neighbours (e.g. a "stroller" \
inside a closed bedroom). Suggest the corrected label.
2. drops — the annotation is background, architecture, or a tracker drift \
artefact. Common cases: items at scene boundaries with no visible support, \
labels for room surfaces that slipped through.
3. merges — two annotations describe the SAME physical object viewed \
from different tracks. They MUST share an object identity, not just \
spatial proximity. ALLOWED: "office chair" + "swivel chair" (synonyms \
for the same chair). FORBIDDEN: "desk" + "chest of drawers" (distinct \
furniture types regardless of how close they sit). FORBIDDEN: any merge \
where the two labels would not be interchangeable English nouns for the \
same physical thing. When in doubt, do NOT merge — emit zero merges \
rather than a wrong one. Pick the higher-confidence id as keep_id.
4. relations — propose at most 8 high-confidence parent-child relations \
between surviving objects. Allowed verbs: on, under, contains, supports, \
next-to, behind, in-front-of.

Annotations:
{annotations_json}

Summarise the scene type in one sentence (e.g. "small home office with \
desk, chair, and two monitors").\
"""


# ---------------------------------------------------------------------------- top-down render


def _topdown_render(
    points_path: Path,
    centroids: np.ndarray,
    image_size: tuple[int, int] = (768, 768),
    margin: float = 1.2,
) -> np.ndarray:
    """Render the cloud from above, framed to enclose every annotation centroid.

    Camera looks straight down at the cloud's mean height; framing tight
    enough that all annotations sit in-frame with a small margin.
    """
    xyz, rgb, _ = load_points_ply(points_path)
    if not len(centroids):
        centroids = xyz[::1000]  # safety net — frame on a sub-sampled cloud
    lo = centroids.min(axis=0)
    hi = centroids.max(axis=0)
    centre = (lo + hi) / 2.0
    extent = float(np.linalg.norm(hi - lo)) * margin
    # Place the camera high above the centre, looking straight down.
    # Inline the (right-handed, OpenCV) look-at math here so we don't
    # reach for render's private helper. Identical to render._look_at.
    eye = (centre + np.array([0, -extent, 0], dtype=np.float32)).astype(np.float32)
    target = centre.astype(np.float32)
    up = np.array([0, 0, 1], dtype=np.float32)  # cloud forward → screen-up
    fwd = target - eye
    fwd /= max(1e-8, float(np.linalg.norm(fwd)))
    right = np.cross(fwd, up)
    right /= max(1e-8, float(np.linalg.norm(right)))
    new_up = np.cross(right, fwd)
    R = np.stack([right, -new_up, fwd], axis=0)  # OpenCV: x right, y down, z fwd
    t = -R @ eye
    extrinsic = np.concatenate([R, t.reshape(3, 1)], axis=1)
    return render_view(xyz, rgb, extrinsic, image_size=image_size, fov_deg=60.0)


# ---------------------------------------------------------------------------- application


def _apply_corrections(
    annotations: list[dict], corr: LaneCCorrections
) -> list[dict]:
    """Apply Gemini's relabel / drop / merge edits to the Lane B output.

    Order matters: drop → merge → relabel. Relations are attached to the
    surviving annotations as a 'relations' field. All edits are applied
    by-id; unknown ids are ignored with a warning.
    """
    by_id: dict[str, dict] = {a["id"]: dict(a) for a in annotations}

    # 1. drops
    drop_ids: set[str] = set()
    for d in corr.drops:
        if d.id not in by_id:
            logger.warning("lane_c drop refers to unknown id %s", d.id)
            continue
        drop_ids.add(d.id)

    # 2. merges (skip any merge whose keep_id was already dropped, AND
    #    block cross-class merges via a programmatic guard — the prompt
    #    instructs against them but Gemini occasionally emits one anyway).
    for m in corr.merges:
        if m.keep_id in drop_ids or m.keep_id not in by_id:
            continue
        survivor = by_id[m.keep_id]
        keep_label = survivor.get("label", "")
        # Inspect every drop-id and reject the whole merge if any drop's
        # label fails the class-equivalence check against keep's label.
        # Engineering safety, not theoretical: the heuristic is
        # last-noun match OR substring (e.g., "chair" / "office chair").
        rejects = []
        for did in m.drop_ids:
            d_ann = by_id.get(did)
            if d_ann is None:
                continue
            d_label = d_ann.get("label", "")
            if not _labels_compatible(keep_label, d_label):
                rejects.append((did, d_label))
        if rejects:
            logger.warning(
                "lane_c: REJECTING cross-class merge — keep=%s '%s' would absorb %s",
                m.keep_id, keep_label,
                [f"{did} '{lb}'" for did, lb in rejects],
            )
            continue
        survivor.setdefault("merged_from_lane_c", []).extend(m.drop_ids)
        survivor.setdefault("provenance", []).append(f"lane_c:merged-{len(m.drop_ids)}")
        for did in m.drop_ids:
            if did in by_id:
                drop_ids.add(did)

    # 3. relabels
    for r in corr.relabels:
        if r.id in drop_ids or r.id not in by_id:
            continue
        existing = by_id[r.id]
        existing.setdefault("alternatives", []).insert(0, existing.get("label", ""))
        existing["alternatives"] = existing["alternatives"][:5]
        existing["label"] = r.new_label
        existing.setdefault("provenance", []).append("lane_c:relabel")

    # 4. relations — bind to surviving annotations
    relations_payload: list[dict] = []
    for rel in corr.relations:
        if rel.subject_id in drop_ids or rel.object_id in drop_ids:
            continue
        if rel.subject_id not in by_id or rel.object_id not in by_id:
            continue
        relations_payload.append({
            "from": rel.subject_id,
            "to": rel.object_id,
            "relation": rel.relation,
        })

    survivors = [by_id[i] for i in by_id if i not in drop_ids]
    # Stash relations on the first survivor as scene-level metadata so the
    # JSON file remains a flat list (frontend reads bare arrays for Lane B).
    if survivors and relations_payload:
        survivors[0].setdefault("scene_relations", []).extend(relations_payload)
    if survivors and corr.summary:
        survivors[0]["scene_summary"] = corr.summary
    return survivors


# ---------------------------------------------------------------------------- entry point


def run_lane_c(
    annotations: list[dict],
    out_dir: Path,
    vlm_model: str = "gemini-2.5-flash",
) -> list[dict]:
    """Run the whole-scene coherence pass on Lane B annotations.

    Writes ``annotations.c.json``. Idempotent — if the file exists, returns
    its contents directly without invoking the VLM. Non-fatal: on any
    error, returns the input annotations unchanged so the pipeline still
    ships a usable scene.
    """
    out_path = out_dir / "annotations.c.json"
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
            print(f"[lane_c] resuming from {out_path.name} ({len(existing)} annotations)",
                  flush=True)
            return existing
        except Exception as e:  # noqa: BLE001
            logger.warning("could not parse %s (%s); re-running", out_path.name, e)

    if len(annotations) < 2:
        print(f"[lane_c] only {len(annotations)} annotation(s) — skipping coherence pass",
              flush=True)
        out_path.write_text(json.dumps(annotations, indent=2))
        return annotations

    points_path = out_dir / "points.ply"
    if not points_path.exists():
        logger.warning("no points.ply — skipping Lane C")
        out_path.write_text(json.dumps(annotations, indent=2))
        return annotations

    _t = _time.time()
    centroids = np.asarray([a["centroid"] for a in annotations], dtype=np.float32)
    try:
        topdown = _topdown_render(points_path, centroids)
    except Exception as e:  # noqa: BLE001
        logger.warning("top-down render failed: %s — skipping Lane C", e)
        out_path.write_text(json.dumps(annotations, indent=2))
        return annotations

    # Compact JSON view of the annotations — only fields that help Gemini
    # reason about coherence. Confidence + extents tell it which entries
    # are weakest.
    summary_payload = []
    for a in annotations:
        bbox = a.get("bbox") or [[0, 0, 0], [0, 0, 0]]
        lo, hi = np.asarray(bbox[0]), np.asarray(bbox[1])
        extent = (hi - lo).tolist()
        summary_payload.append({
            "id": a["id"],
            "label": a.get("label", "unknown"),
            "centroid": [round(float(x), 2) for x in a["centroid"]],
            "extents_xyz": [round(float(x), 2) for x in extent],
            "confidence": round(float(a.get("confidence", 0.0)), 2),
        })

    prompt = _PROMPT.format(annotations_json=json.dumps(summary_payload, indent=2))

    print(f"[lane_c] dispatching coherence call: {len(annotations)} annotations, "
          f"top-down {topdown.shape[1]}×{topdown.shape[0]}", flush=True)
    try:
        corr = call_vlm(prompt, [topdown], LaneCCorrections, model=vlm_model)
    except Exception as e:  # noqa: BLE001
        logger.warning("Lane C VLM call failed (%s) — keeping Lane B output unchanged", e)
        out_path.write_text(json.dumps(annotations, indent=2))
        return annotations

    revised = _apply_corrections(annotations, corr)
    out_path.write_text(json.dumps(revised, indent=2))
    print(f"[lane_c] applied {len(corr.relabels)} relabels, {len(corr.drops)} drops, "
          f"{len(corr.merges)} merges, {len(corr.relations)} relations "
          f"({len(annotations)} → {len(revised)} annotations, "
          f"{_time.time()-_t:.1f}s)", flush=True)
    if corr.summary:
        print(f"[lane_c]   summary: {corr.summary}", flush=True)
    return revised
