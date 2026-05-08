"""Lane F — SpatialLM layout (walls, doors, windows).

SpatialLM-Llama-1B (NeurIPS'25) takes an axis-aligned z-up point cloud and
outputs structured architectural elements (walls, doors, windows). Our pipeline
runs in OpenCV convention (+y down, +z fwd, arbitrary scale), so we:

  1. Re-orient: estimate the floor plane via RANSAC, rotate so floor normal = +z.
  2. Skip metric scale (per the design decision — use ratios/topology only).
  3. Run SpatialLM, parse walls/doors/windows.
  4. Translate back to OpenCV coords so the frontend's existing y/z flip works.
  5. Write annotations.f.json with a `layout` object.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .render import load_points_ply

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------- floor plane


def _ransac_floor(points: np.ndarray, n_iter: int = 1000, threshold: float = 0.05) -> np.ndarray:
    """Returns a unit normal pointing 'up' (away from the floor)."""
    rng = np.random.default_rng(0)
    best_inliers = -1
    best_normal = np.array([0.0, -1.0, 0.0], dtype=np.float32)  # OpenCV: -y is up

    for _ in range(n_iter):
        idx = rng.choice(len(points), size=3, replace=False)
        p1, p2, p3 = points[idx]
        v1, v2 = p2 - p1, p3 - p1
        normal = np.cross(v1, v2)
        norm_len = float(np.linalg.norm(normal))
        if norm_len < 1e-6:
            continue
        normal /= norm_len

        d = -np.dot(normal, p1)
        distances = np.abs(points @ normal + d)
        inliers = int((distances < threshold).sum())
        if inliers > best_inliers:
            best_inliers = inliers
            best_normal = normal

    # Floor normal should point upward (against gravity). In OpenCV, "up" is -y.
    if best_normal[1] > 0:
        best_normal = -best_normal
    return best_normal.astype(np.float32)


def _rotation_to_z_up(up_normal: np.ndarray) -> np.ndarray:
    """Rodrigues rotation that aligns `up_normal` to (0, 0, 1)."""
    target = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    v = np.cross(up_normal, target)
    s = float(np.linalg.norm(v))
    c = float(np.dot(up_normal, target))
    if s < 1e-6:
        return np.eye(3, dtype=np.float32) * (1.0 if c > 0 else -1.0)
    v_skew = np.array(
        [[0, -v[2], v[1]],
         [v[2], 0, -v[0]],
         [-v[1], v[0], 0]],
        dtype=np.float32,
    )
    R = np.eye(3, dtype=np.float32) + v_skew + v_skew @ v_skew * ((1 - c) / (s * s))
    return R


# ---------------------------------------------------------------------------- SpatialLM call


def _run_spatiallm(points_zup: np.ndarray) -> dict:
    """Invoke SpatialLM-Llama-1B on an axis-aligned z-up point cloud.

    Returns a dict with shape `{"walls": [...], "doors": [...], "windows": [...]}`
    where each element has world-space coordinates in the rotated z-up frame.
    """
    try:
        from spatiallm import SpatialLM  # type: ignore[import-not-found]
    except Exception as e:  # noqa: BLE001
        logger.warning("SpatialLM unavailable (%s); returning empty layout", e)
        return {"walls": [], "doors": [], "windows": []}

    model = SpatialLM.from_pretrained("manycore-research/SpatialLM-Llama-1B")
    out = model.predict(points_zup)
    return {
        "walls": [_to_segment(w) for w in out.walls],
        "doors": [_to_quad(d) for d in out.doors],
        "windows": [_to_quad(w) for w in out.windows],
    }


def _to_segment(w) -> dict:
    """Each wall in SpatialLM is described as a line segment (a, b) with a height."""
    return {
        "a": list(map(float, w.a)),
        "b": list(map(float, w.b)),
        "height": float(getattr(w, "height", 0.0)),
    }


def _to_quad(q) -> dict:
    """Doors and windows: 3D quadrilateral with center + extent."""
    return {
        "center": list(map(float, q.center)),
        "extent": list(map(float, q.extent)),
    }


# ---------------------------------------------------------------------------- entry point


def run_lane_f(out_dir: Path) -> dict:
    """Run SpatialLM layout extraction on points.ply for this scene."""
    xyz, _, _ = load_points_ply(out_dir / "points.ply")
    if not len(xyz):
        payload = {"layout": {"walls": [], "doors": [], "windows": []}, "annotations": []}
        (out_dir / "annotations.f.json").write_text(json.dumps(payload, indent=2))
        return payload

    sample = xyz if len(xyz) <= 2_000_000 else xyz[
        np.random.default_rng(0).choice(len(xyz), 2_000_000, replace=False)
    ]
    up = _ransac_floor(sample.astype(np.float32))
    R_to_zup = _rotation_to_z_up(up)
    R_back = R_to_zup.T

    points_zup = (R_to_zup @ sample.T).T

    layout_zup = _run_spatiallm(points_zup)

    layout_cv: dict = {"walls": [], "doors": [], "windows": []}
    for w in layout_zup.get("walls", []):
        a_cv = (R_back @ np.asarray(w["a"], dtype=np.float32)).tolist()
        b_cv = (R_back @ np.asarray(w["b"], dtype=np.float32)).tolist()
        layout_cv["walls"].append({"a": a_cv, "b": b_cv, "height": w.get("height", 0.0)})
    for q in layout_zup.get("doors", []):
        c_cv = (R_back @ np.asarray(q["center"], dtype=np.float32)).tolist()
        layout_cv["doors"].append({"center": c_cv, "extent": q["extent"]})
    for q in layout_zup.get("windows", []):
        c_cv = (R_back @ np.asarray(q["center"], dtype=np.float32)).tolist()
        layout_cv["windows"].append({"center": c_cv, "extent": q["extent"]})

    payload = {
        "layout": layout_cv,
        "annotations": [],
    }
    (out_dir / "annotations.f.json").write_text(json.dumps(payload, indent=2))
    logger.info(
        "Lane F wrote layout: %d walls, %d doors, %d windows",
        len(layout_cv["walls"]), len(layout_cv["doors"]), len(layout_cv["windows"]),
    )
    return payload
