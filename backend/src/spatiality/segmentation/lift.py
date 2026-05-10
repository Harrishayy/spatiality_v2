"""3D pinning per track via bbox-depth unprojection.

Each tracklet from Stage 2 has per-frame bboxes only (no SAM 2 masks). We
lift to 3D by:

  1. For each frame in the track: sample a 5×5 grid of pixels inside the
     bbox interior, gate by depth confidence (drop conf < 0.1), and
     unproject through VGGT depth + camera params.
  2. Concatenate the resulting per-frame point lists into one cloud per
     track (~hundreds of points).
  3. Centroid = median(P, axis=0); OBB = PCA on P.

We don't need SOR or DBSCAN here: bbox-interior pixels are far cleaner
than mask-edge pixels (no per-pixel mask error), the depth-confidence
gate already throws out the worst outliers, and the 5×5 grid never blows
past a few thousand points so PCA/median runs in milliseconds.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from ._track_types import Track, TrackFrame
from .postprocess import _labels_compatible

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------- types


@dataclass
class LiftedTrack:
    track_id: str
    centroid: np.ndarray         # (3,) world coords (OpenCV: y-down, z-fwd)
    obb_corners: np.ndarray      # (8, 3) oriented bounding-box corners
    obb_axes: np.ndarray         # (3, 3) PCA axes (rows = axis vectors)
    obb_extents: np.ndarray      # (3,) half-extents along each axis
    point_count: int
    mean_conf: float
    frame_ids: list[str]
    text_prompt: str | None = None
    source: str = "open_set"


# ---------------------------------------------------------------------------- helpers


def _build_camera_lookup(
    cameras: dict,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Index cameras by frame_id once for O(1) lookup."""
    return {
        f["frame_id"]: (
            np.asarray(f["K"], dtype=np.float32),
            np.asarray(f["R"], dtype=np.float32),
            np.asarray(f["t"], dtype=np.float32),
        )
        for f in cameras.get("frames", [])
    }


def _load_camera(
    camera_lookup: dict, frame_id: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return (K, R, t) for ``frame_id`` or None if Stage 1 frame-selected it out."""
    return camera_lookup.get(frame_id)


# Percentile band for OBB extents. Even with SAM masks, depth has noise
# and a few seam pixels project to far-away free space. PCA is sensitive
# to outliers — one stray point 5m off-object stretches the major-axis
# half-extent by metres. We compute extents from the 5th-95th percentile
# of the per-axis projections (i.e., a 90% middle band), which is
# robust to ~5% outliers per side without rejecting legitimate object
# shape. The centroid is unchanged (median already robust).
_OBB_PERCENTILE_LO = 5.0
_OBB_PERCENTILE_HI = 95.0


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Per-axis weighted median: smallest v with cumulative weight ≥ ½ total.

    When weights are uniform, this returns the unweighted median (modulo
    tie-handling). When weights vary, the result moves toward the values
    of the higher-weight samples — which is what we want when per-pixel
    depth confidence varies across a track's mask.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    if v.size == 0:
        return float("nan")
    if v.size != w.size:
        # Defensive: if weights misaligned, fall back to unweighted.
        return float(np.median(v))
    total = float(w.sum())
    if total <= 0:
        return float(np.median(v))
    order = np.argsort(v)
    cumw = np.cumsum(w[order])
    idx = int(np.searchsorted(cumw, total / 2.0, side="left"))
    idx = min(idx, len(order) - 1)
    return float(v[order[idx]])


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Weighted ``q``-th percentile (q ∈ [0, 100]) with linear interpolation.

    Matches ``np.percentile`` (linear interpolation method) under uniform
    weights — within float precision. Under non-uniform weights, returns
    the value at fractional cumulative weight ``q/100`` interpolated
    linearly between the two adjacent sorted samples, which is the
    natural sample-weight generalisation of numpy's default.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    if v.size == 0:
        return float("nan")
    if v.size != w.size:
        return float(np.percentile(v, q))
    total = float(w.sum())
    if total <= 0:
        return float(np.percentile(v, q))
    order = np.argsort(v)
    v_sorted = v[order]
    w_sorted = w[order]
    # Use plotting-position convention matching numpy's "linear" method:
    # for uniform weights, sample i (0-indexed) sits at position i/(n-1)
    # along the [0, 1] cumulative axis. For weighted samples we use the
    # midpoint of each sample's weight contribution.
    cumw = np.cumsum(w_sorted)
    # Plotting positions = (cumw - w/2) / total, so each sample's pos
    # is the centre of its weight slab. With uniform weights this gives
    # (i + 0.5)/n, which under linear interpolation reproduces numpy's
    # behaviour at all q (modulo edge handling).
    pos = (cumw - w_sorted / 2.0) / total
    target = q / 100.0
    if target <= pos[0]:
        return float(v_sorted[0])
    if target >= pos[-1]:
        return float(v_sorted[-1])
    # Linear interp between the two adjacent samples whose pos brackets target.
    upper = int(np.searchsorted(pos, target, side="left"))
    lower = max(0, upper - 1)
    p_lo, p_hi = pos[lower], pos[upper]
    if p_hi <= p_lo:
        return float(v_sorted[upper])
    frac = (target - p_lo) / (p_hi - p_lo)
    return float(v_sorted[lower] * (1.0 - frac) + v_sorted[upper] * frac)


def _pca_obb(
    points: np.ndarray, weights: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (axes 3×3, half_extents 3, corners 8×3) from PCA on the cluster.

    When ``weights`` is None, behaves identically to the previous
    implementation: simple mean-centred PCA + 5/95 percentile extents.

    When ``weights`` is supplied (per-point ∈ [0, 1] confidence), uses:
      - weighted mean for centring,
      - weighted covariance for the PCA eigenproblem,
      - weighted 5/95 percentiles for axis-aligned extents.

    These are the standard sample-weight generalisations of the
    unweighted operations. With uniform weights the result is identical.
    """
    use_weights = weights is not None and len(weights) == len(points) and float(np.asarray(weights).sum()) > 0
    if use_weights:
        w = np.asarray(weights, dtype=np.float64).ravel()
        ws = float(w.sum())
        centroid = (points.astype(np.float64) * w[:, None]).sum(axis=0) / ws
        centred = points.astype(np.float64) - centroid
        # Weighted covariance: Σ = (Σ w_i (x - μ)(x - μ)^T) / Σ w_i
        cov = (centred.T * w) @ centred / ws
    else:
        centroid = points.mean(axis=0)
        centred = points - centroid
        cov = np.cov(centred.T)

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    axes = eigvecs[:, order].T  # rows = axes

    proj = centred @ axes.T

    if use_weights:
        mins = np.array([
            _weighted_percentile(proj[:, k], w, _OBB_PERCENTILE_LO)
            for k in range(3)
        ])
        maxs = np.array([
            _weighted_percentile(proj[:, k], w, _OBB_PERCENTILE_HI)
            for k in range(3)
        ])
    else:
        mins = np.percentile(proj, _OBB_PERCENTILE_LO, axis=0)
        maxs = np.percentile(proj, _OBB_PERCENTILE_HI, axis=0)

    half = (maxs - mins) / 2.0
    centre_local = (maxs + mins) / 2.0
    centre_world = centre_local @ axes + centroid

    signs = np.array(
        [[s0, s1, s2] for s0 in (-1, 1) for s1 in (-1, 1) for s2 in (-1, 1)],
        dtype=np.float32,
    )
    corners = centre_world + (signs * half) @ axes
    return axes.astype(np.float32), half.astype(np.float32), corners.astype(np.float32)


# ---------------------------------------------------------------------------- per-track lift


# Frame-stride within a track. Bbox-depth lift is so cheap per frame that
# stride only saves I/O — but the depth_conf .npy fetch over Modal's FUSE
# volume is 50–200 ms per file, so striding 8× cuts the lift's wall-clock
# from ~5 s/track to ~0.5 s on long tracks.
LIFT_FRAME_STRIDE = 8
LIFT_MIN_FRAMES = 8

# Parallel I/O for loading depth/conf .npy files. Modal volumes are
# network-mounted; threading lets the kernel saturate the FUSE fetch
# pipeline.
LIFT_IO_WORKERS = 16

# Pixels sampled inside each bbox per frame. A 5×5 grid is plenty — the
# median centroid is dominated by depth, not pixel count.
_GRID_SIDE = 5

# Fraction to inset the bbox on each side before sampling. GDINO bboxes
# are loose around the object and usually include a margin of background
# pixels (the bed, wall, or floor visible behind the object in that
# frame). The 5×5 grid corners would otherwise sample those background
# pixels, pull their depth, and unproject points to wherever the
# background sits in 3D — that's the failure mode behind "the laptop is
# placed near the bed". Inset by 25% per side → the grid samples only the
# inner 50%×50% area where pixels are far more likely to actually be on
# the object. Trade-off: very small objects whose bbox is already tight
# may lose 1-2 valid samples, but for a 5×5 grid that's still plenty.
_BBOX_INSET_FRACTION = 0.25


def _sample_grid_pixels(
    bbox: tuple[int, int, int, int],
    img_h: int,
    img_w: int,
    side: int = _GRID_SIDE,
    inset_fraction: float = _BBOX_INSET_FRACTION,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (ys, xs) of grid sample points clamped inside the bbox + image.

    The grid covers the inner ``(1 - 2 * inset_fraction)`` of the bbox.
    """
    x0, y0, x1, y1 = bbox
    x0 = max(0, min(img_w - 1, int(x0)))
    y0 = max(0, min(img_h - 1, int(y0)))
    x1 = max(x0 + 1, min(img_w, int(x1)))
    y1 = max(y0 + 1, min(img_h, int(y1)))

    # Inset the bbox to its central core, where pixels are most likely
    # to actually be on the object (not background bleed at edges).
    bw = x1 - x0
    bh = y1 - y0
    dx = int(round(bw * inset_fraction))
    dy = int(round(bh * inset_fraction))
    xi0 = x0 + dx
    yi0 = y0 + dy
    xi1 = max(xi0 + 1, x1 - dx)
    yi1 = max(yi0 + 1, y1 - dy)

    xs = np.linspace(xi0, xi1 - 1, side).round().astype(np.int32)
    ys = np.linspace(yi0, yi1 - 1, side).round().astype(np.int32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return yy.ravel(), xx.ravel()


def _load_frame_depth(
    tf: TrackFrame, out_dir: Path
) -> tuple[TrackFrame, np.ndarray, np.ndarray] | None:
    """Worker: load (depth, conf) for one TrackFrame in parallel.

    Returns None if any required file is missing (Stage 1 frame-selection).
    """
    depth_path = out_dir / "depth" / f"{tf.frame_id}.npy"
    conf_path = out_dir / "depth_conf" / f"{tf.frame_id}.npy"
    if not (depth_path.exists() and conf_path.exists()):
        return None
    try:
        depth = np.load(depth_path)
        conf = np.load(conf_path)
    except Exception as e:  # noqa: BLE001
        logger.warning("frame depth load failed for %s: %s", tf.frame_id, e)
        return None
    return tf, depth, conf


# Stride at which the inference stage saves world_points (must match
# inference/run.py::_WP_STRIDE). Lift looks up at full-res (ys, xs) by
# integer-dividing the indices.
_WORLD_POINTS_STRIDE = 2


def _load_world_points(
    tf: TrackFrame, out_dir: Path
) -> tuple[np.ndarray, np.ndarray | None] | None:
    """Load VGGT's pre-computed world_points + (optional) conf for a frame.

    Returns ``(world_points HxWx3, world_points_conf HxW | None)`` at the
    saved stride, or ``None`` if the optional files don't exist (older
    inference runs that pre-date this change).
    """
    wp_path = out_dir / "world_points" / f"{tf.frame_id}.npy"
    if not wp_path.exists():
        return None
    try:
        wp = np.load(wp_path)
    except Exception as e:  # noqa: BLE001
        logger.warning("world_points load failed for %s: %s", tf.frame_id, e)
        return None
    wpc_path = out_dir / "world_points_conf" / f"{tf.frame_id}.npy"
    wpc: np.ndarray | None = None
    if wpc_path.exists():
        try:
            wpc = np.load(wpc_path)
        except Exception as e:  # noqa: BLE001
            logger.warning("world_points_conf load failed for %s: %s", tf.frame_id, e)
            wpc = None
    return wp, wpc


def _lookup_world_points(
    wp: np.ndarray, ys: np.ndarray, xs: np.ndarray
) -> np.ndarray:
    """Sample stride-saved world_points at full-res (ys, xs)."""
    s = _WORLD_POINTS_STRIDE
    h, w = wp.shape[:2]
    yi = np.clip(ys // s, 0, h - 1).astype(np.int32)
    xi = np.clip(xs // s, 0, w - 1).astype(np.int32)
    return wp[yi, xi].astype(np.float32)  # (N, 3)


# Drop a track if its lifted centroid projects inside the source bbox in
# fewer than this fraction of frames. Catches the bbox-interior failure
# mode where the 5×5 grid lands on background that happens to be far in
# front of / behind the real object — the resulting "centroid" sits in
# free space and orbital renders show the wrong region. 50% is permissive:
# half the frames can have one bad sample without the track dying.
_REPROJ_INLIER_THRESHOLD = 0.5


# Minimum sample count required before fitting a 2-component GMM for
# bimodal depth filtering (Newcombe et al. 2011 / KinectFusion style).
# Below this, EM is statistically unstable; we keep the median fallback.
_GMM_MIN_SAMPLES = 20


# 3D-coherence filter parameters. After accumulating world-space points
# across a track, DBSCAN-cluster them with eps=0.3m (typical indoor
# object inter-pixel spacing after stride/conf gating). A coherent
# single-object track has ≥70% of points in its largest cluster — when
# this fails the track has drifted between distinct physical objects
# (e.g. fragmenting between two curtains on opposite walls).
#
# Reference: Schönberger et al. ECCV 2016 (COLMAP MVS); Caesar et al.
# CVPR 2020 (nuScenes — spatial coherence as 3D-MOT validation).
_COHERENCE_DBSCAN_EPS_M = 0.3
_COHERENCE_DBSCAN_MIN_PTS = 5
_COHERENCE_LARGEST_FRACTION = 0.7


# Multi-view geometric consistency filter parameters. For each lifted
# pixel's world point, we project it back into every OTHER frame in the
# tracklet and check whether the projected (u', v') lands inside that
# frame's SAM mask. A pixel is kept if (a) it's in-frustum in at least
# `_MULTIVIEW_MIN_OTHER_FRAMES` other frames AND (b) the majority of
# those frames have the mask covering its projected location.
#
# Why a sample-decision rule and not a population CI (Wilson etc.):
# we're filtering pixels per-track at the per-pixel level, not making
# a claim about a true population rate. Wilson and similar CI bounds
# are appropriate when you want to assert population properties; for
# sample-level operational decisions, a simple ratio test plus a
# minimum-sample-size floor is more interpretable and avoids
# calibration choices that don't reflect a real underlying claim.
#
# The min_other_frames=3 floor rejects the n=2 k=1 ambiguous-evidence
# case (one agree, one disagree) — the failure mode that motivated this
# whole exercise. We lose unanimous-on-2-frames cases but those are
# rare in practice (the source frame is usually a third confirming view
# in spirit; tracks with < 3 in-frustum other frames have effectively
# no cross-view evidence).
#
# Reference: Schönberger et al. ECCV 2016 — multi-view photometric/
# geometric consistency, COLMAP's MVS view-selection algorithm.
_MULTIVIEW_MIN_OTHER_FRAMES = 3  # bumped from 2: rejects n=2 k=1 ambiguous case
_MULTIVIEW_CONSISTENCY_THRESHOLD = 0.5  # naive sample-majority rule
_MULTIVIEW_MASK_DILATE_PX = 2  # dilate target masks by this many px to absorb SAM boundary jitter


def _largest_coherent_cluster(
    P: np.ndarray,
) -> tuple[np.ndarray | None, float, np.ndarray | None]:
    """Return ``(largest-cluster points, largest-fraction, keep_mask)`` or ``(None, frac, None)``.

    Runs DBSCAN on ``P`` (Nx3 world-space points). If the largest cluster
    holds at least ``_COHERENCE_LARGEST_FRACTION`` of the total points,
    returns those points, the fraction, AND the boolean mask into ``P``
    that selected them — so callers can apply the same mask to parallel
    arrays (per-pixel confidence, etc.) without re-deriving membership.

    Returns ``(None, fraction, None)`` when the cluster fails. Returns
    ``(P, 1.0, all-True mask)`` when the input is too small to cluster.
    """
    n = len(P)
    if n < _COHERENCE_DBSCAN_MIN_PTS * 2:
        # Too few points for meaningful clustering; defer to caller's
        # downstream guards (reprojection check, etc.).
        return P, 1.0, np.ones(n, dtype=bool)
    try:
        from sklearn.cluster import DBSCAN  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return P, 1.0, np.ones(n, dtype=bool)

    labels = DBSCAN(
        eps=_COHERENCE_DBSCAN_EPS_M,
        min_samples=_COHERENCE_DBSCAN_MIN_PTS,
    ).fit_predict(P)

    if labels.max() < 0:
        return None, 0.0, None

    cluster_ids, counts = np.unique(labels[labels >= 0], return_counts=True)
    largest_cid = cluster_ids[int(np.argmax(counts))]
    largest_count = int(counts.max())
    fraction = largest_count / n
    if fraction < _COHERENCE_LARGEST_FRACTION:
        return None, fraction, None
    keep_mask = labels == largest_cid
    return P[keep_mask], fraction, keep_mask


def _front_surface_mask(depths: np.ndarray) -> np.ndarray | None:
    """Return a boolean mask selecting the front-surface depth mode.

    Implementation per the plan:
      1. Fit a 2-component Gaussian Mixture (EM) to ``depths``.
      2. BIC-gate against a 1-component fit. If 1-component wins, the
         distribution is unimodal — return None and the caller falls back
         to the full sample.
      3. Otherwise return ``P(z = front | d_i) > 0.5`` for each pixel,
         where ``front`` is the component with smaller mean.

    Reference: KinectFusion (Newcombe et al., ISMAR 2011); COLMAP MVS
    photometric consistency (Schönberger et al., ECCV 2016).
    """
    n = len(depths)
    if n < _GMM_MIN_SAMPLES:
        return None
    try:
        from sklearn.mixture import GaussianMixture  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return None

    d = depths.reshape(-1, 1).astype(np.float64)
    try:
        gmm2 = GaussianMixture(n_components=2, covariance_type="full",
                               random_state=0, n_init=1, max_iter=50).fit(d)
        gmm1 = GaussianMixture(n_components=1, covariance_type="full",
                               random_state=0, n_init=1, max_iter=50).fit(d)
    except Exception:  # noqa: BLE001 — EM occasionally fails on degenerate inputs
        return None

    # BIC: lower is better. If unimodal wins, no bimodality to exploit.
    if gmm1.bic(d) <= gmm2.bic(d):
        return None

    # Front mode = the component with smaller mean.
    means = gmm2.means_.ravel()
    front_idx = int(np.argmin(means))
    posterior = gmm2.predict_proba(d)[:, front_idx]  # P(z = front | d_i)
    keep = posterior > 0.5

    # Defensive: if for some reason every sample is assigned to back (e.g.
    # near-equal means), don't return an empty mask — return None so the
    # caller falls back.
    if int(keep.sum()) < max(_GMM_MIN_SAMPLES // 2, 5):
        return None
    return keep


def _reprojection_inlier_fraction(
    centroid: np.ndarray,
    frames: list[TrackFrame],
    camera_lookup: dict,
) -> float:
    """Fraction of frames where ``centroid`` projects inside that frame's bbox.

    Used as a sanity check on the lift: a centroid that drifted onto the
    wall behind the object will project outside most source bboxes.

    Bbox-size-aware tolerance was considered (precision-variance
    propagation suggests small bboxes need more slack), but the only
    mechanism that actually delivered slack — an absolute pixel floor —
    was uncalibrated. Reverted to the strict-inside test; small-object
    recall is an out-of-scope problem better handled at GDINO /
    postprocess level.
    """
    inside = 0
    total = 0
    for tf in frames:
        cam = camera_lookup.get(tf.frame_id)
        if cam is None:
            continue
        K, R, t = cam
        # world → camera. unprojection used `world = R.T @ (cam - t)`, so
        # forward projection is `cam = R @ world + t`.
        cam_pt = R @ centroid + t
        z = float(cam_pt[2])
        if z <= 1e-3:
            continue
        u = float(cam_pt[0]) / z * float(K[0, 0]) + float(K[0, 2])
        v = float(cam_pt[1]) / z * float(K[1, 1]) + float(K[1, 2])
        x0, y0, x1, y1 = tf.bbox_2d
        total += 1
        if x0 <= u <= x1 and y0 <= v <= y1:
            inside += 1
    return inside / total if total else 0.0


def _dilate_mask(mask: np.ndarray, px: int) -> np.ndarray:
    """Binary-dilate ``mask`` by ``px`` pixels. No-op on px=0.

    Uses cv2.dilate (SIMD + threaded C) — ~15× faster than scipy on a
    1.5k×1.5k mask. Falls back to scipy.ndimage.binary_dilation when cv2
    isn't available. Both produce identical output for binary masks.

    Used to absorb SAM boundary jitter when the target frame's mask
    crops a few pixels tighter than the source frame's. 2 px dilation
    rescues legitimate object pixels without admitting floor-bleed:
    floor pixels project tens of pixels off the real object, not 1-2.
    """
    if px <= 0 or not mask.any():
        return mask
    try:
        import cv2  # noqa: PLC0415
        kernel = np.ones((2 * px + 1, 2 * px + 1), dtype=np.uint8)
        dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        return dilated.astype(bool)
    except Exception:  # noqa: BLE001
        try:
            from scipy.ndimage import binary_dilation  # noqa: PLC0415
            structure = np.ones((2 * px + 1, 2 * px + 1), dtype=bool)
            return binary_dilation(mask, structure=structure)
        except Exception:  # noqa: BLE001
            return mask


def _multiview_visibility_keep(
    per_frame_data: list[tuple[TrackFrame, np.ndarray]],
    masks_by_frame_id: dict[str, np.ndarray],
    camera_lookup: dict,
    *,
    min_other_frames: int = _MULTIVIEW_MIN_OTHER_FRAMES,
    threshold: float = _MULTIVIEW_CONSISTENCY_THRESHOLD,
) -> list[np.ndarray]:
    """Cross-frame mask-consistency filter for one track.

    For every frame's world points, project each point into every *other*
    frame, look up that frame's (dilated) mask, and tally hits / total
    in-frustum frames. A point is kept iff
        (# other frames in frustum) ≥ ``min_other_frames``
        AND (mask hits) / (in-frustum frames) ≥ ``threshold``.

    Args:
      per_frame_data: list of ``(track_frame, world_points_Nx3)`` per frame
        — the world points that survived per-frame conf+finite gates.
      masks_by_frame_id: source-frame SAM masks keyed by frame_id, already
        at the depth-map's resolution. Will be dilated internally per the
        module-level constant.
      camera_lookup: frame_id → (K, R, t).

    Returns:
      A list parallel to ``per_frame_data`` where the i-th element is a
      bool array of shape ``(N_i,)`` — True for "keep this pixel".
    """
    n_frames = len(per_frame_data)
    if n_frames < 2:
        return [np.ones(len(P), dtype=bool) for _, P in per_frame_data]

    # Pre-dilate every mask once. ~5 ms each on a 1.5k×1.5k mask.
    dilated_masks: dict[str, np.ndarray] = {
        fid: _dilate_mask(m, _MULTIVIEW_MASK_DILATE_PX)
        for fid, m in masks_by_frame_id.items()
    }

    keep_per_frame: list[np.ndarray] = []
    for i, (tf_i, P_i) in enumerate(per_frame_data):
        n = len(P_i)
        if n == 0:
            keep_per_frame.append(np.zeros(0, dtype=bool))
            continue

        in_frustum = np.zeros(n, dtype=np.int32)
        in_mask = np.zeros(n, dtype=np.int32)

        for j, (tf_j, _) in enumerate(per_frame_data):
            if j == i:
                continue
            cam = camera_lookup.get(tf_j.frame_id)
            if cam is None:
                continue
            mask_j = dilated_masks.get(tf_j.frame_id)
            if mask_j is None:
                continue
            K, R, t = cam

            # cam = R @ world + t  (mirrors _reprojection_inlier_fraction)
            cam_pts = (R @ P_i.T).T + t  # (n, 3)
            z = cam_pts[:, 2]
            in_front = z > 1e-3
            with np.errstate(invalid="ignore", divide="ignore"):
                u = cam_pts[:, 0] / z * K[0, 0] + K[0, 2]
                v = cam_pts[:, 1] / z * K[1, 1] + K[1, 2]

            h, w = mask_j.shape[:2]
            u_int = np.clip(u, -1, w).astype(np.int32)
            v_int = np.clip(v, -1, h).astype(np.int32)
            in_bounds = (
                in_front
                & (u_int >= 0) & (u_int < w)
                & (v_int >= 0) & (v_int < h)
            )
            in_frustum[in_bounds] += 1

            if in_bounds.any():
                idxs = np.where(in_bounds)[0]
                hits = mask_j[v_int[idxs], u_int[idxs]]
                in_mask[idxs[hits]] += 1

        # Sample-majority rule, with a min-frames floor to reject the
        # ambiguous n=2 k=1 case. Floor was raised from 2 to 3 in
        # preference to a Wilson-LB rewrite — Wilson makes a population
        # claim we don't actually need and its calibration didn't yield
        # a clean improvement over the floor + ratio combination.
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(in_frustum > 0, in_mask / in_frustum, 0.0)
        keep = (in_frustum >= min_other_frames) & (ratio >= threshold)
        keep_per_frame.append(keep)

    return keep_per_frame


def _lift_discard_record(track: Track, *, reason: str, detail: str) -> dict:
    """Build a discard record for a track that failed 3D lifting.

    Geometry is absent (centroid/bbox/etc) — these tracks never produced a
    LiftedTrack, so they have no 3D pose to expose. The frontend renders
    them under the Discarded tab grouped by `stage="lift"`.
    """
    return {
        "id": track.track_id,
        "label": track.text_prompt or track.track_id,
        "stage": "lift",
        "discard_reason": reason,
        "discard_detail": detail,
        "n_frames": len(track.frames),
        "frame_ids": [f"{tf.frame_id}.png" for tf in track.frames],
        "source": track.source,
    }


def lift_track(
    track: Track,
    out_dir: Path,
    camera_lookup: dict,
    conf_threshold: float = 0.5,
    mask_predictor=None,  # SamMaskPredictor | None
    discards: list[dict] | None = None,
) -> LiftedTrack | None:
    """Confidence-gated bbox-depth unprojection for one tracklet.

    Pipeline per frame:
      1. SAM-mask-sample (or 5×5 inset grid fallback)
      2. Confidence gate (default 0.5; adaptive 0.3 fallback if too few survive)
      3. Finite-depth gate
      4. Unproject to world

    Then ONCE per track:
      5. Multi-view consistency filter — drop pixels whose world point
         doesn't reproject inside the SAM mask in a majority of other
         frames. Replaces the per-frame GMM filter for tracks with ≥ 3
         posed frames; the GMM falls back in for shorter tracks.
      6. DBSCAN 3D-coherence filter (catastrophic-drift safeguard).
      7. Reprojection sanity (centroid must project inside source bboxes).
      8. PCA OBB.

    When ``mask_predictor`` is None, the 5×5 inset grid takes over and the
    multi-view filter is skipped (no real masks to test against).
    """
    from .mask import sample_mask_pixels  # noqa: PLC0415
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    # Subsample frames within long tracks.
    frames = track.frames
    if len(frames) >= LIFT_MIN_FRAMES * LIFT_FRAME_STRIDE:
        frames = frames[::LIFT_FRAME_STRIDE]

    # Hard cap on per-track frames. SAM cost scales O(N) and the
    # multi-view consistency filter scales O(N²); without a cap, a 50-
    # frame track that fell below the stride threshold blows up to
    # ~30-50s of work. 16 is enough viewpoints for robust DBSCAN
    # coherence + multi-view (and median-based aggregation is
    # insensitive to count beyond ~10), while bounding worst-case
    # per-track work at ~3-4s. Even sub-sampling preserves temporal
    # diversity (linspace) so we don't lose start/end views.
    _MAX_FRAMES_PER_TRACK = 16
    if len(frames) > _MAX_FRAMES_PER_TRACK:
        idx = np.linspace(0, len(frames) - 1, _MAX_FRAMES_PER_TRACK).round().astype(int)
        frames = [frames[int(i)] for i in idx]

    posed_frames = [tf for tf in frames if _load_camera(camera_lookup, tf.frame_id) is not None]

    with ThreadPoolExecutor(max_workers=LIFT_IO_WORKERS) as pool:
        bundles = list(pool.map(lambda tf: _load_frame_depth(tf, out_dir), posed_frames))

    # Fast-path map: frame_id → (world_points, world_points_conf|None).
    # If the inference stage saved VGGT's point head, we'll use direct
    # XYZ lookup instead of manual unprojection — eliminates convention
    # risk and exposes a separate confidence channel. Empty when the
    # scene was inferred before that change landed.
    wp_by_frame: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
    if (out_dir / "world_points").exists():
        for tf in posed_frames:
            loaded = _load_world_points(tf, out_dir)
            if loaded is not None:
                wp_by_frame[tf.frame_id] = loaded

    frames_dir = out_dir / "frames"
    # Per-frame collected data — drives the multi-view filter below.
    # Each entry: (track_frame, world_points (N,3), confidences (N,), source_mask (H,W) | None)
    per_frame: list[tuple[TrackFrame, np.ndarray, np.ndarray, np.ndarray | None]] = []
    n_mask_used = 0
    n_grid_fallback = 0
    for bundle in bundles:
        if bundle is None:
            continue
        tf, depth, conf = bundle

        cam = _load_camera(camera_lookup, tf.frame_id)
        if cam is None:
            continue
        K, R, t = cam

        # Sample pixels via SAM mask if available, else 5×5 inset grid.
        # SAM now runs at its native 1024×1024 — masks come back at the
        # resized resolution and we upsample (nearest) to depth res.
        h, w = depth.shape[:2]
        ys = xs = None
        source_mask: np.ndarray | None = None
        if mask_predictor is not None:
            frame_png = frames_dir / f"{tf.frame_id}.png"
            if frame_png.exists():
                mask = mask_predictor.predict(frame_png, tf.bbox_2d)
                if mask is not None:
                    if mask.shape[:2] != (h, w):
                        # cv2.resize is ~10× faster than the PIL
                        # roundtrip we used previously; nearest-neighbor
                        # preserves the boolean nature of the mask.
                        try:
                            import cv2  # noqa: PLC0415
                            mask = cv2.resize(
                                mask.astype(np.uint8), (w, h),
                                interpolation=cv2.INTER_NEAREST,
                            ).astype(bool)
                        except Exception:  # noqa: BLE001
                            mask = np.asarray(
                                Image.fromarray(mask.astype(np.uint8) * 255)
                                .resize((w, h), Image.NEAREST)
                            ).astype(bool)
                    ys, xs = sample_mask_pixels(mask)
                    if len(ys):
                        n_mask_used += 1
                        source_mask = mask
        if ys is None or not len(ys):
            ys, xs = _sample_grid_pixels(tf.bbox_2d, h, w)
            n_grid_fallback += 1
        if not len(ys):
            continue

        # Confidence gate with adaptive fallback. VGGT's depth_conf is
        # calibrated; values < 0.5 cluster at depth discontinuities (the
        # boundaries where bleed-through lives). Tighten to 0.5 by default,
        # but if that strips a frame to fewer than 50 samples, retry at 0.3
        # for that frame only — preserves coverage on small / distant
        # objects whose entire mask is at lower confidence.
        cs_all = conf[ys, xs]
        valid = cs_all > conf_threshold
        if int(valid.sum()) < 50 and conf_threshold > 0.3:
            valid = cs_all > 0.3
        if not valid.any():
            continue
        ys = ys[valid]; xs = xs[valid]
        ds = depth[ys, xs].astype(np.float32)
        finite = np.isfinite(ds) & (ds > 1e-3)
        if not finite.any():
            continue
        ys = ys[finite]; xs = xs[finite]; ds = ds[finite]
        cs = cs_all[valid][finite].astype(np.float32)

        # GMM front-surface filter — only for short tracks where the
        # multi-view filter below can't run effectively. For tracks with
        # ≥ 3 posed frames, multi-view consistency does a strictly better
        # job and we skip the GMM to avoid double-filtering.
        if len(posed_frames) < 3:
            front_keep = _front_surface_mask(ds)
            if front_keep is not None:
                ys = ys[front_keep]; xs = xs[front_keep]
                ds = ds[front_keep]; cs = cs[front_keep]

        # World coordinates: prefer VGGT's point-head output when present
        # (avoids convention risk in our manual R, t math), else compute
        # via standard unprojection: world = R.T @ (cam - t).
        wp_pair = wp_by_frame.get(tf.frame_id)
        if wp_pair is not None:
            wp_map, wpc_map = wp_pair
            world = _lookup_world_points(wp_map, ys, xs)
            # Combine VGGT's separate world_points_conf with depth_conf so
            # downstream weighting reflects both signals. Multiply (both
            # are in [0, 1]); fall back to depth_conf alone when wpc is
            # missing.
            if wpc_map is not None:
                wpc_lookup = _lookup_world_points(
                    wpc_map[..., None], ys, xs
                ).reshape(-1)
                cs = (cs * np.clip(wpc_lookup, 0.0, 1.0)).astype(np.float32)
        else:
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            x_cam = (xs.astype(np.float32) - cx) * ds / fx
            y_cam = (ys.astype(np.float32) - cy) * ds / fy
            z_cam = ds
            cam_pts = np.stack([x_cam, y_cam, z_cam], axis=1)
            world = (R.T @ (cam_pts - t).T).T.astype(np.float32)

        per_frame.append((tf, world, cs, source_mask))

    if not per_frame:
        return None

    # Multi-view geometric consistency filter — the actual fix for the
    # floor-bleed failure mode. Skip when SAM masks weren't available
    # (no source masks to test against) or when fewer than 3 frames
    # survived (the test degenerates with very few views).
    have_masks = sum(1 for _, _, _, m in per_frame if m is not None)
    if have_masks >= 3:
        masks_by_frame_id = {tf.frame_id: m for tf, _, _, m in per_frame if m is not None}
        framed_data = [(tf, P) for tf, P, _, _ in per_frame]
        keeps = _multiview_visibility_keep(framed_data, masks_by_frame_id, camera_lookup)
        n_in = sum(len(P) for _, P, _, _ in per_frame)
        n_out = sum(int(k.sum()) for k in keeps)
        per_frame = [
            (tf, P[k], C[k], m)
            for (tf, P, C, m), k in zip(per_frame, keeps, strict=True)
        ]
        per_frame = [(tf, P, C, m) for tf, P, C, m in per_frame if len(P) > 0]
        if not per_frame:
            logger.warning(
                "drop %s: multi-view filter rejected all pixels (%d → 0)",
                track.track_id, n_in,
            )
            if discards is not None:
                discards.append(_lift_discard_record(
                    track,
                    reason="multiview_filter",
                    detail=f"multi-view consistency rejected all {n_in} pixels.",
                ))
            return None
        if n_in:
            logger.info(
                "track %s multi-view filter: %d/%d pixels kept (%.0f%%)",
                track.track_id, n_out, n_in, 100.0 * n_out / max(1, n_in),
            )

    pts_world = [P for _, P, _, _ in per_frame]
    confs = [C for _, _, C, _ in per_frame]
    if not pts_world:
        return None

    P = np.concatenate(pts_world, axis=0)
    C = np.concatenate(confs, axis=0) if confs else np.array([])
    if len(P) < 4:
        return None

    # 3D-coherence filter (per the plan). Catches the "wrong side of the
    # room" failure mode where the tracker bridged two physically distinct
    # objects (e.g. curtains on opposite walls fused into one tracklet).
    largest_pts, frac, keep_mask = _largest_coherent_cluster(P)
    if largest_pts is None:
        logger.warning(
            "drop %s: 3D coherence — largest cluster only %.0f%% of points",
            track.track_id, frac * 100.0,
        )
        if discards is not None:
            discards.append(_lift_discard_record(
                track,
                reason="3d_coherence",
                detail=(
                    f"largest 3D cluster only {frac * 100.0:.0f}% of points — "
                    f"tracker likely bridged two physically distinct objects."
                ),
            ))
        return None
    # Apply the same mask to the parallel confidence array so weighted
    # statistics downstream stay aligned. (Previously we dropped C when
    # sizes diverged — that disabled confidence-weighted aggregation.)
    if keep_mask is not None and not keep_mask.all():
        P = largest_pts
        if len(C):
            C = C[keep_mask]

    # Confidence-weighted centroid: per-axis weighted median. When all
    # cs are equal (or empty) this collapses to the unweighted median —
    # behaviour identical to before. When per-pixel conf varies, the
    # centroid biases toward the high-conf samples (which is what we
    # want when noisy boundary pixels exist alongside clean object
    # pixels).
    if len(C) == len(P) and float(C.sum()) > 0:
        centroid = np.array([
            _weighted_median(P[:, k], C) for k in range(3)
        ], dtype=np.float32)
    else:
        centroid = np.median(P, axis=0).astype(np.float32)

    # Reprojection sanity: the lifted centroid must land inside the source
    # bbox in a reasonable fraction of frames. If it doesn't, the lift
    # latched onto background depth and the resulting OBB is not actually
    # this object — drop rather than mislead Lane B.
    inlier_frac = _reprojection_inlier_fraction(
        centroid.astype(np.float32), track.frames, camera_lookup
    )
    if inlier_frac < _REPROJ_INLIER_THRESHOLD:
        logger.warning(
            "drop %s: centroid projects inside bbox in only %.0f%% of frames",
            track.track_id, inlier_frac * 100.0,
        )
        if discards is not None:
            discards.append(_lift_discard_record(
                track,
                reason="reprojection",
                detail=(
                    f"lifted centroid projects inside bbox in only "
                    f"{inlier_frac * 100.0:.0f}% of frames — likely background-depth latch."
                ),
            ))
        return None

    # Weighted PCA OBB when confidence is aligned. Same fallback logic.
    weights_for_obb = C if (len(C) == len(P) and float(C.sum()) > 0) else None
    axes, extents, corners = _pca_obb(P, weights=weights_for_obb)

    return LiftedTrack(
        track_id=track.track_id,
        centroid=centroid.astype(np.float32),
        obb_corners=corners,
        obb_axes=axes,
        obb_extents=extents,
        point_count=int(P.shape[0]),
        mean_conf=float(C.mean()) if len(C) else 0.0,
        frame_ids=[f.frame_id for f in track.frames],
        text_prompt=track.text_prompt,
        source=track.source,
    )


# ---------------------------------------------------------------------------- 3D OBB merge


# Two LiftedTracks whose AABBs share more than this fraction (3D IoU) are
# treated as the same physical instance and merged. 0.5 is a tight
# threshold: distinct neighbouring objects (a lamp on a desk) rarely share
# that much volume, while synonym-phrase or gap-split duplicates of the
# same physical object almost always do.
_MERGE_AABB_IOU_THRESHOLD = 0.5

# Fallback merge cue: very-close centroids (< this fraction of the *smaller*
# OBB diagonal) merge regardless of AABB IoU. Catches the rare case where
# the lift's depth-bleed shifts two tracks of the same object enough that
# their AABBs no longer overlap heavily, but their centroids are still
# clearly inside each other's volume.
_MERGE_CENTROID_FRACTION = 0.5


def _aabb_from_corners_minmax(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (lo, hi) axis-aligned extents."""
    return corners.min(axis=0), corners.max(axis=0)


def _aabb_iou_3d(
    a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]
) -> float:
    a_lo, a_hi = a
    b_lo, b_hi = b
    inter_lo = np.maximum(a_lo, b_lo)
    inter_hi = np.minimum(a_hi, b_hi)
    inter_extent = np.maximum(inter_hi - inter_lo, 0)
    inter_vol = float(inter_extent.prod())
    a_vol = float(np.maximum(a_hi - a_lo, 0).prod())
    b_vol = float(np.maximum(b_hi - b_lo, 0).prod())
    union_vol = a_vol + b_vol - inter_vol
    return inter_vol / union_vol if union_vol > 0 else 0.0


def _obb_diagonal(corners: np.ndarray) -> float:
    lo, hi = _aabb_from_corners_minmax(corners)
    return float(np.linalg.norm(hi - lo))


def merge_lifted_tracks(
    tracks: list[LiftedTrack],
    iou_threshold: float = _MERGE_AABB_IOU_THRESHOLD,
    centroid_fraction: float = _MERGE_CENTROID_FRACTION,
) -> tuple[list[LiftedTrack], dict]:
    """Cluster LiftedTracks by 3D AABB IoU + centroid proximity.

    Single-link clustering: tracks i,j merge if either:
      - AABB IoU ≥ ``iou_threshold`` (high-overlap → almost certainly
        the same physical object regardless of detector phrase), OR
      - centroid distance < ``centroid_fraction`` × min(diag_i, diag_j)
        AND detector phrases are compatible (synonym / substring / shared
        last noun) — the label guard prevents distinct neighbouring
        objects with overlapping diagonals from collapsing.

    Within each cluster the highest-mean_conf track is canonical (its
    track_id, text_prompt, source survive). The cluster's OBB is re-fit
    on the union of all members' corner points; centroid is the
    point-count-weighted mean of member centroids; frame_ids and point
    counts are unioned/summed.

    Returns (merged_tracks, stats_dict).
    """
    n = len(tracks)
    if n < 2:
        return list(tracks), {"n_in": n, "n_out": n, "merged": 0, "merged_losers": []}

    aabbs = [_aabb_from_corners_minmax(t.obb_corners) for t in tracks]
    diags = [_obb_diagonal(t.obb_corners) for t in tracks]

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

    for i in range(n):
        for j in range(i + 1, n):
            iou = _aabb_iou_3d(aabbs[i], aabbs[j])
            cd = float(np.linalg.norm(tracks[i].centroid - tracks[j].centroid))
            cd_thresh = centroid_fraction * min(diags[i], diags[j])
            high_overlap = iou >= iou_threshold
            close_and_compatible = (
                cd < cd_thresh
                and _labels_compatible(tracks[i].text_prompt or "", tracks[j].text_prompt or "")
            )
            if high_overlap or close_and_compatible:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    merged: list[LiftedTrack] = []
    merged_losers: list[dict] = []
    n_merged = 0
    for members in groups.values():
        if len(members) == 1:
            merged.append(tracks[members[0]])
            continue
        n_merged += len(members) - 1

        # Canonical = highest mean_conf member.
        canonical_idx = max(members, key=lambda m: tracks[m].mean_conf)
        canonical = tracks[canonical_idx]
        for m in members:
            if m == canonical_idx:
                continue
            t = tracks[m]
            merged_losers.append({
                "id": t.track_id,
                "label": t.text_prompt or t.track_id,
                "stage": "lift",
                "discard_reason": "merged_3d",
                "discard_detail": (
                    f"merged into '{canonical.text_prompt or canonical.track_id}' "
                    f"({canonical.track_id}) — same 3D region "
                    f"(AABB IoU ≥ {iou_threshold} or centroid distance < "
                    f"{centroid_fraction}×min_diag)."
                ),
                "merged_into": canonical.track_id,
                "n_frames": len(t.frame_ids),
                "frame_ids": list(t.frame_ids),
                "source": t.source,
            })

        # Re-fit OBB on the union of member corners (8 × N points). Cheap
        # PCA approximation that's accurate enough for downstream rendering.
        all_corners = np.concatenate([tracks[m].obb_corners for m in members])
        axes, extents, corners = _pca_obb(all_corners)

        # Point-count-weighted centroid.
        weights = np.asarray([tracks[m].point_count for m in members], dtype=np.float32)
        if weights.sum() <= 0:
            weights = np.ones(len(members), dtype=np.float32)
        weights = weights / weights.sum()
        centroid = np.sum(
            np.stack([tracks[m].centroid * w for m, w in zip(members, weights)]),
            axis=0,
        )

        all_frame_ids = sorted(set().union(*[set(tracks[m].frame_ids) for m in members]))
        merged.append(LiftedTrack(
            track_id=canonical.track_id,
            centroid=centroid.astype(np.float32),
            obb_corners=corners,
            obb_axes=axes,
            obb_extents=extents,
            point_count=int(sum(tracks[m].point_count for m in members)),
            mean_conf=float(np.mean([tracks[m].mean_conf for m in members])),
            frame_ids=all_frame_ids,
            text_prompt=canonical.text_prompt,
            source="merged",
        ))

    stats = {
        "n_in": n,
        "n_out": len(merged),
        "merged": n_merged,
        "iou_threshold": iou_threshold,
        "centroid_fraction": centroid_fraction,
        "merged_losers": merged_losers,
    }
    return merged, stats


# ---------------------------------------------------------------------------- entry point


def run_lifting(
    tracks: list[Track],
    out_dir: Path,
) -> list[LiftedTrack]:
    """Lift every tracklet to a ``LiftedTrack`` and merge 3D duplicates.

    Uses SAM 2.1-hiera-tiny for mask-grade pixel selection per (track,
    frame) when the predictor builds successfully. Falls back to the
    bbox-interior 5×5 grid otherwise — sets the env var
    ``SPATIALITY_DISABLE_SAM=1`` to force the fallback for cost / debug.
    """
    import time as _time
    from .mask import build_predictor  # noqa: PLC0415

    cameras = json.loads((out_dir / "cameras.json").read_text())
    camera_lookup = _build_camera_lookup(cameras)
    print(f"[lift] {len(tracks)} tracks, {len(camera_lookup)} cameras loaded "
          f"(stride={LIFT_FRAME_STRIDE})", flush=True)

    lifted: list[LiftedTrack] = []
    discards: list[dict] = []
    with build_predictor() as mask_predictor:
        if mask_predictor is None:
            print("[lift] mask predictor unavailable — using bbox-interior grid", flush=True)
        else:
            print("[lift] mask-grade unprojection via SAM 2.1-hiera-tiny", flush=True)

        _t = _time.time()
        for i, tr in enumerate(tracks, start=1):
            _t_tr = _time.time()
            result = lift_track(
                tr, out_dir, camera_lookup,
                mask_predictor=mask_predictor,
                discards=discards,
            )
            ok = result is not None
            if ok:
                lifted.append(result)
            # Drop the per-frame encoder cache between tracks so VRAM stays
            # bounded — the cache is keyed by image, but tracks may share
            # frames; release here is conservative (re-encodes on next hit
            # for the same frame, but only if the next track's first frame
            # collides with the prior track's last cached one).
            if mask_predictor is not None:
                mask_predictor.release()
            print(f"[lift]   {i}/{len(tracks)} '{tr.text_prompt or tr.track_id}' "
                  f"{'lifted' if ok else 'dropped'} "
                  f"(track={_time.time()-_t_tr:.2f}s, total={_time.time()-_t:.1f}s, "
                  f"running={len(lifted)}/{i})", flush=True)
    print(f"[lift] before merge: {len(lifted)} / {len(tracks)} lifted "
          f"({_time.time()-_t:.1f}s)", flush=True)

    _t_merge = _time.time()
    merged, merge_stats = merge_lifted_tracks(lifted)
    print(f"[lift] 3D OBB merge: {merge_stats['n_in']} → {merge_stats['n_out']} "
          f"(merged {merge_stats['merged']} duplicates @ IoU≥"
          f"{merge_stats['iou_threshold']} OR centroid_dist < "
          f"{merge_stats['centroid_fraction']}×min_diag, "
          f"{_time.time()-_t_merge:.2f}s)", flush=True)

    # Merge-loser discards — non-canonical members of each merge cluster.
    discards.extend(merge_stats.get("merged_losers", []))

    # Persist all lift-stage discards (multi-view, 3D coherence,
    # reprojection, merge-loser). Read by lane_b._finalise into the
    # unified annotations.b.discarded.json.
    (out_dir / "_lift_discards.json").write_text(json.dumps(discards, indent=2))
    print(f"[lift] {len(discards)} lift-stage discards written → _lift_discards.json", flush=True)

    return merged
