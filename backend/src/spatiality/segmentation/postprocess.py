"""Post-processing for Lane B annotations.

Called from ``lane_b._finalise`` after the per-track flush completes.
Reads the raw VLM output and produces:
  - ``annotations.b.json``         — kept (cleaned, deduped) tracks
  - ``annotations.b.discarded.json`` — dropped tracks tagged with
    ``discard_reason`` so the UI can render a Discarded tab.

Lane B emits one annotation per LiftedTrack. The pre-Lane-B pipeline
intentionally over-detects:
  - Multiple scout phrases ("plush toy", "stuffed animal", "Stitch plush toy")
    can all match the same physical object → multiple tracks at the same 3D
    location.
  - The GDINO IoU linker has gap_tolerance=3 frames; a brief look-away
    splits a single physical instance into multiple tracklets.
  - The _MAX_TRACKLETS_PER_PHRASE cap can let three "cap-3" phrases each
    keep distinct slices of the same object.

Without a SAM 2 propagation step (which provided implicit cross-frame
identity), these all survive as independent annotations. This module
collapses the noise into a clean object list:

  1. Filter scene-level / non-object labels via a deny-list ("room", "scene",
     "scan", "workspace", …) plus low-confidence / unknown drops.
  2. Filter implausibly large OBBs (anything > ROOM_VOLUME_FRACTION of the
     scene AABB is almost certainly the room/wall/floor mislabelled).
  3. Cluster surviving annotations by 3D centroid distance + label
     similarity (last-noun match) and keep the highest-confidence
     representative per cluster.
"""

from __future__ import annotations

import logging
import re

import numpy as np

logger = logging.getLogger(__name__)


# Labels that aren't physical objects. Case-insensitive substring match.
# Built from observed Lane B outputs that turned out to be scene-level
# rather than object-level (room mislabelled by Gemini when an OBB happens
# to cover a wall, ceiling, or large floor patch).
_SCENE_DENY_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\broom\b",        # "room", "bedroom", "living room", "A room"
        r"\bscene\b",
        r"\bscan\b",        # "3D room scan"
        r"\bworkspace\b",
        r"\bbackground\b",
        r"\benvironment\b",
        r"\barea\b",
        r"\bspace\b",
        r"\binterior\b",
        r"\bunknown\b",
    ]
]

# Confidence floor — everything below is dropped before clustering. Lane B
# already biases low-confidence outputs toward "unknown" but the explicit
# floor handles edge cases where Gemini hedged with a specific label at
# low confidence.
_MIN_CONFIDENCE = 0.30

# Class-conditional 3D size priors (in metres, OBB diagonal). Replaces
# the historical scene-relative `_MAX_OBB_DIAG_FRACTION` cap with
# real-world dimension bounds per object class. Ranges drawn from
# IKEA / ANSI office-furniture / retail dimension tables.
#
# Reference: nuScenes (Caesar et al. CVPR 2020) — class-conditional 3D
# box priors used both for proposal regression and post-processing NMS.
# A "chair" with OBB diagonal of 6.5m fails its prior regardless of
# scene size; a "bed" at 2.5m passes its prior regardless of scene size.
_CLASS_OBB_RANGES_M: dict[str, tuple[float, float]] = {
    # Chairs / seating
    "chair":           (0.4, 1.5),
    "armchair":        (0.6, 1.5),
    "ottoman":         (0.3, 1.0),
    "stool":           (0.3, 1.0),
    # Tables / desks
    "desk":            (0.8, 2.5),
    "table":           (0.4, 3.5),
    "nightstand":      (0.3, 1.0),
    "dresser":         (0.8, 2.5),
    "drawers":         (0.5, 2.5),
    # Beds / sofas (large fabric)
    "bed":             (1.5, 3.5),
    "sofa":            (1.5, 3.5),
    "couch":           (1.5, 3.5),
    "mattress":        (1.5, 3.0),
    # Storage
    "wardrobe":        (1.0, 3.0),
    "closet":          (1.0, 3.0),
    "cabinet":         (0.5, 3.0),
    "bookshelf":       (0.5, 3.0),
    "shelf":           (0.3, 3.0),
    # Architecture
    "door":            (1.0, 2.5),
    "window":          (0.5, 3.0),
    "curtain":         (0.5, 3.5),
    # Electronics
    "monitor":         (0.3, 1.0),
    "screen":          (0.3, 2.0),
    "tv":              (0.5, 2.0),
    "laptop":          (0.2, 0.6),
    "computer":        (0.2, 0.8),
    "keyboard":        (0.2, 0.6),
    # Lighting
    "lamp":            (0.2, 1.5),
    # Soft furnishings
    "blanket":         (0.3, 3.0),
    "pillow":          (0.2, 0.8),
    "cushion":         (0.2, 0.8),
    "rug":             (0.5, 4.0),
    # Stuff
    "book":            (0.1, 0.4),
    "mug":             (0.05, 0.2),
    "bottle":          (0.05, 0.4),
    "cable":           (0.1, 3.0),
    # Wall hardware (small)
    "outlet":          (0.05, 0.3),
    "socket":          (0.05, 0.3),
    "switch":          (0.05, 0.3),
    # Power
    "strip":           (0.1, 0.5),  # power strip
    # Plushies / toys
    "toy":             (0.1, 1.0),
    "plushie":         (0.1, 0.8),
    "animal":          (0.1, 0.8),  # stuffed animal
    # Apparel / bags
    "backpack":        (0.3, 1.2),
    "bag":             (0.2, 1.5),
    "boot":            (0.2, 0.6),
    "shoe":            (0.2, 0.6),
    # Storage / hardware
    "fan":             (0.3, 1.5),
    "vent":            (0.1, 1.0),
}

# Fallback fraction of scene diagonal for classes not in the table. Used
# only when last-noun lookup misses; ensures the postprocess always has
# a usable upper bound.
_MAX_OBB_DIAG_FALLBACK_FRACTION = 0.85


def _obb_diag_range_for(label: str, scene_diag: float) -> tuple[float, float]:
    """Resolve the (min, max) OBB-diagonal range in metres for ``label``.

    Priority: exact last-noun match in the prior table; else any other
    word in the label that matches a key (handles "stuffed animal" →
    "animal", "computer monitor" → "monitor"); else the scene-relative
    fallback (0, 0.85 × scene_diag).
    """
    if not label:
        return 0.0, scene_diag * _MAX_OBB_DIAG_FALLBACK_FRACTION
    last = _last_noun(label)
    if last in _CLASS_OBB_RANGES_M:
        return _CLASS_OBB_RANGES_M[last]
    # Try other words (catches "stuffed animal" → "animal").
    for word in label.lower().split():
        if word in _CLASS_OBB_RANGES_M:
            return _CLASS_OBB_RANGES_M[word]
    return 0.0, scene_diag * _MAX_OBB_DIAG_FALLBACK_FRACTION

# Centroid distance for an instance-aware merge: within a single class,
# only annotations whose centroids are closer than this DBSCAN-eps are
# treated as duplicates. 0.5m is the practical "same physical object"
# distance: typical tracker-fragmentation spread sits at 0.2-0.4m, while
# distinct chairs around a table sit ≥ 0.7m apart. Down from 1.5m (which
# was tuned for *cross-class* merging — far too permissive once we group
# by class first).
_CLUSTER_DIST_THRESHOLD_M = 0.5

# Alternative merge signal — if two annotations have AABB-IoU above this,
# they're likely the same physical thing even if centroids are far apart.
# Useful when depth-bleed shifts centroids while leaving the bounding
# region overlapping.
_CLUSTER_AABB_IOU_THRESHOLD = 0.3

# Disjoint-OBB guard. Two annotations whose AABBs do not overlap at all
# (intersection volume = 0) are kept separate even if centroids fall
# inside the eps ball — the geometry rules out a single-object merge.
# Catches the failure mode where two shelves stacked vertically have
# close centroids but disjoint bounding regions.


def _is_scene_label(label: str) -> bool:
    if not label:
        return True
    return any(p.search(label) for p in _SCENE_DENY_PATTERNS)


def _last_noun(label: str) -> str:
    """Best-effort last-noun extractor for label similarity."""
    return (label or "").strip().lower().split()[-1] if label else ""


def _labels_compatible(a: str, b: str) -> bool:
    """Return True if two labels are similar enough to merge.

    Rules (in order):
      1. case-insensitive equality
      2. one is a substring of the other (handles "chair" vs "office chair")
      3. shared last word (handles "Stitch plush toy" vs "plush toy")
    """
    al, bl = a.lower().strip(), b.lower().strip()
    if al == bl:
        return True
    if al in bl or bl in al:
        return True
    return _last_noun(al) == _last_noun(bl)


def _scene_diagonal(annotations: list[dict]) -> float:
    """Diagonal of the AABB enclosing every annotation's centroid."""
    if not annotations:
        return 0.0
    cs = np.asarray([a["centroid"] for a in annotations], dtype=np.float32)
    return float(np.linalg.norm(cs.max(axis=0) - cs.min(axis=0)))


def _obb_diagonal(ann: dict) -> float:
    """Diagonal of the annotation's axis-aligned bbox (lo / hi corners)."""
    bbox = ann.get("bbox")
    if not bbox or len(bbox) != 2:
        return 0.0
    lo, hi = np.asarray(bbox[0]), np.asarray(bbox[1])
    return float(np.linalg.norm(hi - lo))


def _filter_drop_scene_and_outsized(
    annotations: list[dict], scene_diag: float
) -> tuple[list[dict], list[dict], dict[str, int]]:
    """Drop scene-level labels, low confidence, and class-prior-violating OBBs.

    Size check uses :func:`_obb_diag_range_for` — a class-conditional
    [min, max] in metres rather than a single scene-relative fraction.
    For unrecognised classes the function returns (0, 0.85 × scene_diag),
    matching the previous global-cap behaviour as a fallback.

    Returns ``(kept, discarded, counts)``. Each entry in ``discarded`` is
    a copy of the original annotation augmented with ``discard_reason``
    and a human-readable ``discard_detail`` so the UI can surface why it
    was dropped without re-running the pipeline logic client-side.
    """
    kept: list[dict] = []
    discarded: list[dict] = []
    counts = {"scene_label": 0, "low_conf": 0, "oversize": 0, "undersize": 0}
    for a in annotations:
        label = a.get("label", "")
        if _is_scene_label(label):
            counts["scene_label"] += 1
            discarded.append({
                **a,
                "stage": "postprocess",
                "discard_reason": "scene_label",
                "discard_detail": f"VLM returned scene-level label '{label}', not an object.",
            })
            continue
        if float(a.get("confidence", 0.0)) < _MIN_CONFIDENCE:
            counts["low_conf"] += 1
            discarded.append({
                **a,
                "stage": "postprocess",
                "discard_reason": "low_confidence",
                "discard_detail": (
                    f"calibrated confidence {float(a.get('confidence', 0.0)):.2f} "
                    f"below floor {_MIN_CONFIDENCE:.2f}."
                ),
            })
            continue
        diag = _obb_diagonal(a)
        d_min, d_max = _obb_diag_range_for(label, scene_diag)
        if d_max > 0 and diag > d_max:
            logger.info(
                "drop %s: %s OBB diag %.2fm exceeds class prior max %.2fm",
                a.get("id", "?"), label, diag, d_max,
            )
            counts["oversize"] += 1
            discarded.append({
                **a,
                "stage": "postprocess",
                "discard_reason": "oversize",
                "discard_detail": (
                    f"OBB diagonal {diag:.2f}m exceeds class prior max "
                    f"{d_max:.2f}m for '{label}'."
                ),
            })
            continue
        # Undersize is usually fine (sub-mm noise tracks already filtered
        # upstream). We log but don't drop here — small OBBs aren't slop,
        # they're just under-sampled.
        if d_min > 0 and diag < d_min and diag > 0:
            logger.info(
                "small %s: %s OBB diag %.2fm below class prior min %.2fm — keeping",
                a.get("id", "?"), label, diag, d_min,
            )
        kept.append(a)
    return kept, discarded, counts


def _aabb_iou_3d(a: dict, b: dict) -> float:
    """3D AABB IoU between two annotations' bbox lo/hi extents."""
    abb, bbb = a.get("bbox"), b.get("bbox")
    if not abb or not bbb or len(abb) != 2 or len(bbb) != 2:
        return 0.0
    a_lo, a_hi = np.asarray(abb[0]), np.asarray(abb[1])
    b_lo, b_hi = np.asarray(bbb[0]), np.asarray(bbb[1])
    inter_lo = np.maximum(a_lo, b_lo)
    inter_hi = np.minimum(a_hi, b_hi)
    inter_extent = np.maximum(inter_hi - inter_lo, 0)
    inter_vol = float(inter_extent.prod())
    a_vol = float(np.maximum(a_hi - a_lo, 0).prod())
    b_vol = float(np.maximum(b_hi - b_lo, 0).prod())
    union_vol = a_vol + b_vol - inter_vol
    return inter_vol / union_vol if union_vol > 0 else 0.0


def _aabbs_disjoint(a: dict, b: dict) -> bool:
    """True if a's and b's AABBs share zero volume — geometric guarantee they're distinct objects."""
    abb, bbb = a.get("bbox"), b.get("bbox")
    if not abb or not bbb or len(abb) != 2 or len(bbb) != 2:
        return False
    a_lo, a_hi = np.asarray(abb[0]), np.asarray(abb[1])
    b_lo, b_hi = np.asarray(bbb[0]), np.asarray(bbb[1])
    inter_lo = np.maximum(a_lo, b_lo)
    inter_hi = np.minimum(a_hi, b_hi)
    return bool(np.any(inter_hi <= inter_lo))


def _cluster_and_dedupe(
    annotations: list[dict],
) -> tuple[list[dict], list[dict], int]:
    """Instance-aware merge: group by class first, then DBSCAN within each class.

    Algorithm:
      1. Bucket annotations by class key (`_last_noun(label)`) so distinct
         classes never merge.
      2. Within each class bucket, single-link cluster on (centroid eps OR
         AABB IoU) — but skip pairs whose AABBs are entirely disjoint
         (geometric guarantee they're distinct instances).

    The eps is now 0.5m (down from 1.5m) because we no longer need a
    permissive distance to swallow cross-class label noise — we group on
    class first. With the disjoint-AABB guard, three chairs around a
    table at ~0.8m spacing stay three chairs.

    Returns ``(deduped, merged_losers, n_merged)``. ``merged_losers``
    contains every non-primary annotation in a multi-member cluster,
    tagged with ``discard_reason='merged_duplicate'`` so the UI can show
    them under the Discarded tab.
    """
    n = len(annotations)
    if n < 2:
        return list(annotations), [], 0

    # Step 1 — bucket by class. Empty / unlabelled go to a single bucket
    # so they can still merge with each other.
    buckets: dict[str, list[int]] = {}
    for i, a in enumerate(annotations):
        cls = _last_noun(a.get("label", ""))
        buckets.setdefault(cls, []).append(i)

    cs = np.asarray([a["centroid"] for a in annotations], dtype=np.float32)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        a, b = find(i), find(j)
        if a != b:
            parent[a] = b

    # Step 2 — single-link within each class bucket. Pairs across classes
    # are not even considered, so distinct classes can never merge.
    for indices in buckets.values():
        m = len(indices)
        if m < 2:
            continue
        for ii in range(m):
            for jj in range(ii + 1, m):
                i, j = indices[ii], indices[jj]
                # Belt-and-braces label compatibility (e.g. catches
                # "ceramic mug" vs "coffee mug" sharing last-noun
                # 'mug' — already true by bucket key, but explicit).
                if not _labels_compatible(annotations[i]["label"], annotations[j]["label"]):
                    continue
                # Geometric disjointness vetoes the merge regardless of
                # centroid distance — stacked shelves, neighbouring chairs.
                if _aabbs_disjoint(annotations[i], annotations[j]):
                    continue
                d = float(np.linalg.norm(cs[i] - cs[j]))
                iou = _aabb_iou_3d(annotations[i], annotations[j])
                if d > _CLUSTER_DIST_THRESHOLD_M and iou < _CLUSTER_AABB_IOU_THRESHOLD:
                    continue
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    survivors: list[dict] = []
    losers: list[dict] = []
    for members in groups.values():
        if len(members) == 1:
            survivors.append(annotations[members[0]])
            continue
        # Keep highest-confidence; merge frame_ids + alternatives from siblings.
        members_sorted = sorted(
            members, key=lambda i: float(annotations[i].get("confidence", 0.0)), reverse=True
        )
        primary = dict(annotations[members_sorted[0]])
        all_frame_ids: set[str] = set()
        all_alternatives: list[str] = list(primary.get("alternatives", []))
        merged_track_ids = [annotations[m]["id"] for m in members_sorted]
        for m in members_sorted:
            all_frame_ids.update(annotations[m].get("frame_ids", []))
            for alt in annotations[m].get("alternatives", []):
                if alt not in all_alternatives:
                    all_alternatives.append(alt)
        primary["frame_ids"] = sorted(all_frame_ids)
        primary["alternatives"] = all_alternatives[:5]  # cap
        primary["merged_from"] = merged_track_ids[1:]   # provenance
        primary.setdefault("provenance", []).append(
            f"dedup:merged-{len(merged_track_ids)}-tracks"
        )
        survivors.append(primary)
        for m in members_sorted[1:]:
            loser = dict(annotations[m])
            loser["stage"] = "postprocess"
            loser["discard_reason"] = "merged_duplicate"
            loser["discard_detail"] = (
                f"merged into '{primary.get('label', '?')}' "
                f"({primary.get('id', '?')}) — same physical object."
            )
            loser["merged_into"] = primary.get("id")
            losers.append(loser)

    return survivors, losers, n - len(survivors)


def cleanup_lane_b_annotations(
    annotations: list[dict],
) -> tuple[list[dict], list[dict], dict]:
    """Run the full post-process: drop scene labels → drop oversize → cluster.

    Returns ``(cleaned_list, discarded_list, stats_dict)``. Each entry in
    ``discarded_list`` carries a ``discard_reason`` /
    ``discard_detail`` pair so the UI can render a "Discarded" tab
    without re-running cleanup logic on the client. Stats include each
    filter's drop count and the dedup count.
    """
    n_in = len(annotations)
    scene_diag = _scene_diagonal(annotations)
    after_filter, dropped_filtered, drop_counts = _filter_drop_scene_and_outsized(
        annotations, scene_diag
    )
    after_dedup, dropped_merged, n_merged = _cluster_and_dedupe(after_filter)
    discarded = dropped_filtered + dropped_merged
    return after_dedup, discarded, {
        "n_in": n_in,
        "n_out": len(after_dedup),
        "n_discarded": len(discarded),
        "scene_diag_m": round(scene_diag, 2),
        "fallback_max_obb_diag_m": round(
            scene_diag * _MAX_OBB_DIAG_FALLBACK_FRACTION, 2
        ),
        "dropped_scene_label": drop_counts["scene_label"],
        "dropped_low_conf": drop_counts["low_conf"],
        "dropped_oversize": drop_counts["oversize"],
        "merged_duplicates": n_merged,
    }
