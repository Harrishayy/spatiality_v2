"""3D pinning per track + cross-frame stitching.

This is the fix for the "annotation on the wrong side of the room" bug. The
old pipeline lifted a 2D mask centroid to 3D using a single mean depth — any
background bleed dragged the centroid onto the far wall. We do this instead:

  1. confidence-gate per-pixel depth (drop conf < 0.1)
  2. unproject every mask pixel through VGGT depth + camera params
  3. statistical outlier removal across the multi-view-fused cloud
  4. DBSCAN; keep only the largest cluster (drops surviving background bleed)
  5. median centroid (robust to remaining outliers in a way mean is not)
  6. PCA-OBB for the oriented bbox

After per-track lifting, run a final 3D-IoU + SigLIP merge (parameters
borrowed from ConceptGraphs: vis_sim > 0.8, IoU3d > 0.5 OR centroid_dist <
0.3 m) to mop up any tracks SAM 3.1 split apart.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .sam3 import Track

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
    siglip_feat: np.ndarray      # (D,) mean SigLIP embedding over mask regions
    text_prompt: str | None = None
    source: str = "open_set"


# ---------------------------------------------------------------------------- helpers


def _load_camera(cameras: dict, frame_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    for cam in cameras["frames"]:
        if cam["frame_id"] == frame_id:
            return (
                np.asarray(cam["K"], dtype=np.float32),
                np.asarray(cam["R"], dtype=np.float32),
                np.asarray(cam["t"], dtype=np.float32),
            )
    raise KeyError(f"camera for {frame_id} not found")


def _statistical_outlier_removal(
    points: np.ndarray, k: int = 20, std_ratio: float = 2.0
) -> np.ndarray:
    """Open3D-style SOR: drop points whose mean kNN distance is >μ+σ·std_ratio."""
    if len(points) < k + 1:
        return points

    from sklearn.neighbors import NearestNeighbors  # noqa: PLC0415

    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(points)
    dists, _ = nbrs.kneighbors(points)
    mean_d = dists[:, 1:].mean(axis=1)
    threshold = mean_d.mean() + std_ratio * mean_d.std()
    return points[mean_d <= threshold]


def _largest_dbscan_cluster(points: np.ndarray, eps: float | None = None) -> np.ndarray:
    """Run DBSCAN, return only the largest cluster's points."""
    if len(points) < 10:
        return points

    from sklearn.cluster import DBSCAN  # noqa: PLC0415

    if eps is None:
        # Adaptive eps from the median nearest-neighbor distance, scaled up
        # to bridge typical object-scale gaps.
        from sklearn.neighbors import NearestNeighbors  # noqa: PLC0415

        nbrs = NearestNeighbors(n_neighbors=2).fit(points)
        d, _ = nbrs.kneighbors(points)
        eps = float(np.median(d[:, 1])) * 4.0

    labels = DBSCAN(eps=eps, min_samples=10).fit_predict(points)
    if (labels == -1).all():
        return points

    # Largest non-noise cluster.
    counts = np.bincount(labels[labels >= 0])
    if not len(counts):
        return points
    keep = labels == int(np.argmax(counts))
    return points[keep]


def _pca_obb(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (axes 3×3, half_extents 3, corners 8×3) from PCA on the cluster."""
    centroid = points.mean(axis=0)
    centred = points - centroid
    cov = np.cov(centred.T)
    eigvals, eigvecs = np.linalg.eigh(cov)

    order = np.argsort(eigvals)[::-1]
    axes = eigvecs[:, order].T  # rows = axes

    proj = centred @ axes.T
    mins = proj.min(axis=0)
    maxs = proj.max(axis=0)
    half = (maxs - mins) / 2.0
    centre_local = (maxs + mins) / 2.0
    centre_world = centre_local @ axes + centroid

    signs = np.array(
        [[s0, s1, s2] for s0 in (-1, 1) for s1 in (-1, 1) for s2 in (-1, 1)],
        dtype=np.float32,
    )
    corners = centre_world + (signs * half) @ axes
    return axes.astype(np.float32), half.astype(np.float32), corners.astype(np.float32)


def _aabb_from_corners(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return corners.min(axis=0), corners.max(axis=0)


# ---------------------------------------------------------------------------- SigLIP loader (cached)


_SIGLIP = None


def _siglip():
    """Load OpenCLIP SigLIP once per process."""
    global _SIGLIP
    if _SIGLIP is not None:
        return _SIGLIP
    import torch  # noqa: PLC0415
    import open_clip  # noqa: PLC0415

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16-SigLIP", pretrained="webli"
    )
    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    _SIGLIP = (model, preprocess)
    return _SIGLIP


def _siglip_feature(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean SigLIP embedding over the mask region, as a (D,) numpy vector."""
    import torch  # noqa: PLC0415

    model, preprocess = _siglip()
    bbox_y, bbox_x = np.where(mask)
    if not len(bbox_x):
        return np.zeros(model.visual.output_dim, dtype=np.float32)
    y0, y1 = bbox_y.min(), bbox_y.max() + 1
    x0, x1 = bbox_x.min(), bbox_x.max() + 1

    crop = image_rgb[y0:y1, x0:x1].copy()
    crop_mask = mask[y0:y1, x0:x1]
    # Black out non-masked area to focus the encoder.
    crop[~crop_mask] = 0
    pil = Image.fromarray(crop)
    tensor = preprocess(pil).unsqueeze(0)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    with torch.no_grad():
        feat = model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0).cpu().numpy()


# ---------------------------------------------------------------------------- per-track lift


def lift_track(
    track: Track,
    out_dir: Path,
    cameras: dict,
    conf_threshold: float = 0.1,
) -> LiftedTrack | None:
    """Confidence-gated multi-view fusion for one SAM 3.1 track."""
    pts_world: list[np.ndarray] = []
    confs: list[np.ndarray] = []
    siglip_feats: list[np.ndarray] = []

    for tf in track.frames:
        depth_path = out_dir / "depth" / f"{tf.frame_id}.npy"
        conf_path = out_dir / "depth_conf" / f"{tf.frame_id}.npy"
        mask_path = out_dir / tf.mask_path
        frame_path = out_dir / "frames" / f"{tf.frame_id}.png"
        if not (depth_path.exists() and conf_path.exists() and mask_path.exists()):
            continue

        depth = np.load(depth_path)        # (H, W)
        conf = np.load(conf_path)          # (H, W)
        mask = np.array(Image.open(mask_path)) > 127
        rgb = np.array(Image.open(frame_path).convert("RGB"))
        K, R, t = _load_camera(cameras, tf.frame_id)

        valid = mask & (conf > conf_threshold)
        if not valid.any():
            continue

        ys, xs = np.where(valid)
        ds = depth[ys, xs]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        x_cam = (xs.astype(np.float32) - cx) * ds / fx
        y_cam = (ys.astype(np.float32) - cy) * ds / fy
        z_cam = ds.astype(np.float32)
        cam = np.stack([x_cam, y_cam, z_cam], axis=1)
        world = (R.T @ (cam - t).T).T

        pts_world.append(world.astype(np.float32))
        confs.append(conf[ys, xs].astype(np.float32))

        try:
            siglip_feats.append(_siglip_feature(rgb, mask))
        except Exception as e:  # noqa: BLE001
            logger.warning("SigLIP failed for %s/%s: %s", track.track_id, tf.frame_id, e)

    if not pts_world:
        return None

    P = np.concatenate(pts_world, axis=0)
    C = np.concatenate(confs, axis=0) if confs else np.array([])

    # Multi-stage outlier rejection.
    P = _statistical_outlier_removal(P, k=20, std_ratio=2.0)
    P = _largest_dbscan_cluster(P)
    if len(P) < 10:
        return None

    centroid = np.median(P, axis=0)
    axes, extents, corners = _pca_obb(P)

    feat = (
        np.mean(siglip_feats, axis=0)
        if siglip_feats
        else np.zeros(384, dtype=np.float32)
    )

    return LiftedTrack(
        track_id=track.track_id,
        centroid=centroid.astype(np.float32),
        obb_corners=corners,
        obb_axes=axes,
        obb_extents=extents,
        point_count=int(P.shape[0]),
        mean_conf=float(C.mean()) if len(C) else 0.0,
        frame_ids=[f.frame_id for f in track.frames],
        siglip_feat=feat.astype(np.float32),
        text_prompt=track.text_prompt,
        source=track.source,
    )


# ---------------------------------------------------------------------------- cross-frame stitch


def merge_tracks(tracks: list[LiftedTrack]) -> list[LiftedTrack]:
    """Final 3D-IoU + SigLIP merge pass.

    Catches cases where SAM 3.1 split a single physical object into multiple
    tracks (e.g. brief occlusion → re-detection). We use the ConceptGraphs
    criterion: agree geometrically AND semantically.
    """
    n = len(tracks)
    if n < 2:
        return tracks

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
        ai = _aabb_from_corners(tracks[i].obb_corners)
        for j in range(i + 1, n):
            aj = _aabb_from_corners(tracks[j].obb_corners)
            iou = _aabb_iou(ai, aj)
            cd = float(np.linalg.norm(tracks[i].centroid - tracks[j].centroid))
            sim = float(np.dot(tracks[i].siglip_feat, tracks[j].siglip_feat))
            if (iou > 0.5 or cd < 0.3) and sim > 0.8:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    merged: list[LiftedTrack] = []
    for root, members in groups.items():
        if len(members) == 1:
            merged.append(tracks[members[0]])
            continue
        combined_corners = np.concatenate([tracks[m].obb_corners for m in members])
        combined_centroid = np.mean([tracks[m].centroid for m in members], axis=0)
        combined_feat = np.mean([tracks[m].siglip_feat for m in members], axis=0)
        combined_feat /= max(1e-8, np.linalg.norm(combined_feat))

        # Re-fit OBB on union of corners as a fast proxy for re-fitting on all points.
        axes, extents, corners = _pca_obb(combined_corners)

        primary = tracks[members[0]]
        merged.append(
            LiftedTrack(
                track_id=primary.track_id,
                centroid=combined_centroid.astype(np.float32),
                obb_corners=corners,
                obb_axes=axes,
                obb_extents=extents,
                point_count=sum(tracks[m].point_count for m in members),
                mean_conf=float(np.mean([tracks[m].mean_conf for m in members])),
                frame_ids=sorted(
                    set().union(*[set(tracks[m].frame_ids) for m in members])
                ),
                siglip_feat=combined_feat.astype(np.float32),
                text_prompt=primary.text_prompt,
                source="merged",
            )
        )
    return merged


def _aabb_iou(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> float:
    a_min, a_max = a
    b_min, b_max = b
    lo = np.maximum(a_min, b_min)
    hi = np.minimum(a_max, b_max)
    inter = np.maximum(hi - lo, 0).prod()
    a_vol = (a_max - a_min).prod()
    b_vol = (b_max - b_min).prod()
    union = a_vol + b_vol - inter
    return float(inter / union) if union > 0 else 0.0


# ---------------------------------------------------------------------------- entry point


def run_lifting(
    tracks: list[Track],
    out_dir: Path,
) -> list[LiftedTrack]:
    """Lift every SAM 3.1 track to a `LiftedTrack`, then run the safety-net merge."""
    import time as _time
    cameras = json.loads((out_dir / "cameras.json").read_text())
    print(f"[lift] {len(tracks)} tracks, {len(cameras.get('frames', []))} cameras loaded", flush=True)

    lifted: list[LiftedTrack] = []
    _t = _time.time()
    for i, tr in enumerate(tracks, start=1):
        result = lift_track(tr, out_dir, cameras)
        if result is not None:
            lifted.append(result)
        if i % 10 == 0 or i == len(tracks):
            print(f"[lift]   {i}/{len(tracks)} tracks processed, "
                  f"{len(lifted)} lifted ({_time.time()-_t:.1f}s)", flush=True)

    print(f"[lift] before stitch: {len(lifted)} / {len(tracks)} lifted", flush=True)
    _t_merge = _time.time()
    merged = merge_tracks(lifted)
    print(f"[lift] after stitch: {len(merged)} tracks "
          f"({_time.time()-_t_merge:.1f}s)", flush=True)
    return merged
