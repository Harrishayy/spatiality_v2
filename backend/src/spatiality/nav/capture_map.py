"""Stage 4: capture map — top-down 2D footprint of the captured scene.

Reframes what used to be a humanoid free-space / traversability layer
into a more honest artefact: a top-down density map of the captured
cloud. It answers "what's in the room and how much of it did we cover?"
without trying to predict where a humanoid could stand — handheld
captures rarely contain enough floor pixels to support that inference,
and the previous algorithm was returning 0 m² traversable on desk-
centric scenes.

Inputs (all from ``<scene_dir>/``):
    points.ply       confidence-gated dense colour cloud (xyz+rgb+conf)
    cameras.json     per-frame K, R, t in OpenCV convention (+y down)

Outputs (written into ``<scene_dir>/``):
    capture_map.json  density grid (uint8 log-normalised) + ground-plane
                      basis for the 3D overlay
    capture_map.png   top-down preview: amber density heatmap

The ground-plane basis (``up_axis_world``, ``u_axis_world``,
``v_axis_world``, ``floor_height_world``) is preserved verbatim from the
previous schema so the web viewer's leveling code keeps working without
changes.

CPU only (numpy + Pillow). Wall-clock ~5–15 s on a 50 M-point cloud.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------- PLY reader

_PLY_DTYPE = np.dtype(
    [
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("r", "u1"), ("g", "u1"), ("b", "u1"),
        ("c", "<f4"),
    ]
)


def _read_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(xyz, conf)`` from the pipeline's binary PLY. RGB is unused here."""
    with path.open("rb") as f:
        header_lines: list[str] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"unexpected EOF reading PLY header in {path}")
            header_lines.append(line.decode("ascii", errors="replace").strip())
            if header_lines[-1] == "end_header":
                break
        n = -1
        for h in header_lines:
            if h.startswith("element vertex"):
                n = int(h.split()[-1])
                break
        if n < 0:
            raise ValueError(f"PLY {path} missing 'element vertex N' line")
        rec = np.frombuffer(f.read(n * _PLY_DTYPE.itemsize), dtype=_PLY_DTYPE)
    xyz = np.stack([rec["x"], rec["y"], rec["z"]], axis=1).astype(np.float32)
    conf = rec["c"].astype(np.float32)
    return xyz, conf


# ---------------------------------------------------------------------------- ground frame

def _camera_positions_and_up(cameras_json: dict) -> tuple[np.ndarray, np.ndarray]:
    """Recover world-space camera centres and a robust scene-up direction.

    OpenCV cameras store ``[R | t]`` with ``X_cam = R @ X_world + t``, so the
    camera centre in world is ``C = -R.T @ t`` and the image-y-down axis
    ``[0, -1, 0]`` maps to world via ``R.T``. Averaging the per-frame ups
    is more stable than either "−y in world" (only true if frame 0 is level)
    or cloud PCA (gets confused by tall obstacles).
    """
    centers: list[np.ndarray] = []
    ups: list[np.ndarray] = []
    for f in cameras_json.get("frames", []):
        R = np.asarray(f["R"], dtype=np.float32)
        t = np.asarray(f["t"], dtype=np.float32).reshape(3)
        centers.append(-R.T @ t)
        ups.append(R.T @ np.array([0.0, -1.0, 0.0], dtype=np.float32))
    if not centers:
        raise ValueError("cameras.json has no frames")
    cam_centers = np.stack(centers, axis=0)
    up = np.mean(np.stack(ups, axis=0), axis=0)
    norm = float(np.linalg.norm(up))
    if norm < 1e-6:
        up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    else:
        up = up / norm
    return cam_centers, up


def _ground_plane_basis(up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal axes spanning the floor plane (perpendicular to up).

    First axis is world-x projected onto the plane, so the grid stays
    approximately aligned with the captured space's natural orientation.
    Falls back to world-z when up is nearly parallel to x.
    """
    seed = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(seed, up))) > 0.9:
        seed = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    u = seed - up * float(np.dot(seed, up))
    u = u / max(float(np.linalg.norm(u)), 1e-6)
    v = np.cross(up, u)
    v = v / max(float(np.linalg.norm(v)), 1e-6)
    return u, v


def _estimate_floor_height(heights: np.ndarray) -> float:
    """Floor = densest 5 cm bin near the 2nd-percentile height.

    Robust to single-point under-floor outliers (which would dominate
    a naive ``min``) and to tall obstacles (which would tug PCA off).
    """
    p_low = np.percentile(heights, 2.0)
    p_high = np.percentile(heights, 60.0)
    mask = heights <= p_high
    bins = np.arange(p_low - 0.20, p_low + 0.40 + 0.05, 0.05, dtype=np.float32)
    if bins.size < 2:
        return float(p_low)
    hist, edges = np.histogram(heights[mask], bins=bins)
    if hist.sum() == 0:
        return float(p_low)
    i = int(np.argmax(hist))
    return float(0.5 * (edges[i] + edges[i + 1]))


# ---------------------------------------------------------------------------- compute

def compute_capture_map(
    scene_dir: Path | str,
    *,
    cell_size_m: float = 0.05,
    conf_floor: float = 0.20,
    max_extent_m: float = 12.0,
    min_height_above_floor_m: float = 0.05,
) -> dict:
    """Compute and persist the capture-map artefacts for one scene.

    Returns a small summary dict (no grid payload) for logs / manifest.

    Parameters
    ----------
    cell_size_m
        Side length of one grid cell. 5 cm matches the underlying depth-map
        resolution for indoor captures.
    conf_floor
        Drop PLY points with depth-confidence below this. ``points.ply``
        already had a floor applied at write time (0.15 default).
    max_extent_m
        Hard cap on grid side length before the bbox crop runs. Stops a
        single far-away outlier point from blowing the grid to 100 m.
    min_height_above_floor_m
        Drop points within this height of the estimated floor. They're
        either the floor itself (uninteresting for the "what's in the
        room" map) or under-floor noise. 5 cm matches one cell at default
        resolution, so we lose nothing visible.
    """
    scene_dir = Path(scene_dir)
    ply_path = scene_dir / "points.ply"
    cam_path = scene_dir / "cameras.json"
    if not ply_path.exists():
        raise FileNotFoundError(f"points.ply not found in {scene_dir}")
    if not cam_path.exists():
        raise FileNotFoundError(f"cameras.json not found in {scene_dir}")

    cams = json.loads(cam_path.read_text())
    cam_centers, up = _camera_positions_and_up(cams)
    u_axis, v_axis = _ground_plane_basis(up)

    xyz, conf = _read_ply(ply_path)
    if xyz.size == 0:
        raise ValueError(f"points.ply at {ply_path} has 0 points")
    if conf_floor > 0:
        keep = conf >= conf_floor
        xyz = xyz[keep]
        if xyz.shape[0] == 0:
            raise ValueError(
                f"all points dropped at conf_floor={conf_floor}; "
                f"upstream PLY may already be over-pruned"
            )

    # Project everything into (u, v, up). Recentre on the camera-path mean
    # so the origin is the captured space rather than the world frame.
    cam_uv = np.stack(
        [cam_centers @ u_axis, cam_centers @ v_axis],
        axis=1,
    )
    cam_center_uv = cam_uv.mean(axis=0)
    pts_height = xyz @ up
    pts_uv = np.stack(
        [xyz @ u_axis, xyz @ v_axis],
        axis=1,
    ) - cam_center_uv

    in_window = (
        (np.abs(pts_uv[:, 0]) <= max_extent_m)
        & (np.abs(pts_uv[:, 1]) <= max_extent_m)
    )
    pts_uv = pts_uv[in_window]
    pts_height = pts_height[in_window]
    if pts_uv.shape[0] == 0:
        raise ValueError("no points survived the max_extent_m window — bad scale?")

    floor_h = _estimate_floor_height(pts_height)
    rel_h = pts_height - floor_h

    above_floor = rel_h >= min_height_above_floor_m
    pts_uv = pts_uv[above_floor]
    if pts_uv.shape[0] == 0:
        raise ValueError("no above-floor points survived — floor estimate too high?")

    # Rasterise above-floor points into a grid_n × grid_n density grid.
    half_extent = max_extent_m
    grid_n = int(np.ceil(2 * half_extent / cell_size_m))
    cell_i = np.floor((pts_uv[:, 0] + half_extent) / cell_size_m).astype(np.int64)
    cell_j = np.floor((pts_uv[:, 1] + half_extent) / cell_size_m).astype(np.int64)
    valid = (cell_i >= 0) & (cell_i < grid_n) & (cell_j >= 0) & (cell_j < grid_n)
    flat = cell_i[valid] * grid_n + cell_j[valid]
    counts = np.bincount(flat, minlength=grid_n * grid_n).astype(np.int64)
    density = counts.reshape(grid_n, grid_n)

    # Tighten to the bbox of *well-supported* cells. The 5th percentile of
    # non-empty densities filters out long-tail single-point cells at the
    # periphery; typical cells hold hundreds of points, so the threshold
    # only catches outliers. `tight_*` are the no-margin dimensions used
    # for the "this is what we covered" extent displayed in the card; the
    # saved grid (`density`) keeps a small breathing-room margin so the
    # PNG isn't visually cramped.
    nz_vals = density[density > 0]
    if nz_vals.size > 0:
        bbox_thresh = max(1.0, float(np.percentile(nz_vals, 5)))
    else:
        bbox_thresh = 1.0
    nz = np.argwhere(density >= bbox_thresh)
    if nz.size:
        tight_ymin = int(nz[:, 0].min())
        tight_ymax = int(nz[:, 0].max()) + 1
        tight_xmin = int(nz[:, 1].min())
        tight_xmax = int(nz[:, 1].max()) + 1
        margin = 3
        ymin = max(0, tight_ymin - margin)
        ymax = min(grid_n, tight_ymax + margin)
        xmin = max(0, tight_xmin - margin)
        xmax = min(grid_n, tight_xmax + margin)
        density = density[ymin:ymax, xmin:xmax]
        crop_origin_uv = (
            -half_extent + xmin * cell_size_m,
            -half_extent + ymin * cell_size_m,
        )
        tight_extent_m = (
            float((tight_xmax - tight_xmin) * cell_size_m),
            float((tight_ymax - tight_ymin) * cell_size_m),
        )
    else:
        crop_origin_uv = (-half_extent, -half_extent)
        tight_extent_m = (0.0, 0.0)

    # Log-bin density into uint8. Raw point counts are heavy-tailed (a
    # single high-coverage shelf can have 100× the points of a typical
    # cell) — a linear ramp washes everything else into the background.
    log_d = np.log1p(density.astype(np.float64))
    dmax = float(log_d.max()) if log_d.size else 0.0
    if dmax > 0:
        density_u8 = (log_d / dmax * 255.0).clip(0, 255).astype(np.uint8)
    else:
        density_u8 = np.zeros_like(density, dtype=np.uint8)

    h_cells, w_cells = density.shape
    coverage_cells = int((density > 0).sum())
    n_frames = int(cam_centers.shape[0])

    meta = {
        "version": 2,
        "kind": "capture_map",
        "convention": "opencv",
        "cell_size_m": float(cell_size_m),
        "grid_shape": [int(h_cells), int(w_cells)],
        "tight_extent_m": [tight_extent_m[0], tight_extent_m[1]],
        "origin_uv_m": [float(crop_origin_uv[0]), float(crop_origin_uv[1])],
        "floor_height_world": float(floor_h),
        "up_axis_world": [float(up[0]), float(up[1]), float(up[2])],
        "u_axis_world": [float(u_axis[0]), float(u_axis[1]), float(u_axis[2])],
        "v_axis_world": [float(v_axis[0]), float(v_axis[1]), float(v_axis[2])],
        "camera_center_uv_m": [float(cam_center_uv[0]), float(cam_center_uv[1])],
        "stats": {
            "coverage_cells": coverage_cells,
            "coverage_m2": float(coverage_cells * cell_size_m * cell_size_m),
            "n_frames": n_frames,
        },
        "density_b64": base64.b64encode(density_u8.tobytes(order="C")).decode("ascii"),
    }

    json_path = scene_dir / "capture_map.json"
    json_path.write_text(json.dumps(meta))

    png_path = scene_dir / "capture_map.png"
    _render_preview_png(
        density_uint8=density_u8,
        out=png_path,
    )

    return {
        "shape": [int(h_cells), int(w_cells)],
        "cell_size_m": cell_size_m,
        "floor_height_world": float(floor_h),
        "stats": meta["stats"],
        "json_path": json_path.name,
        "png_path": png_path.name,
    }


# ---------------------------------------------------------------------------- preview

# Sunset palette — matches the rest of the viewer's colour language so the
# preview reads as "same artefact, different lens" rather than a separate UI.
_COLOUR_BG = np.array([24, 22, 26], dtype=np.float32)
_COLOUR_DENSITY_LO = np.array([60, 38, 32], dtype=np.float32)
_COLOUR_DENSITY_HI = np.array([255, 196, 132], dtype=np.float32)


def _render_preview_png(
    *,
    density_uint8: np.ndarray,
    out: Path,
    upscale: int = 6,
) -> None:
    """Top-down PNG: amber density heatmap."""
    h, w = density_uint8.shape
    t = density_uint8.astype(np.float32)[..., None] / 255.0
    ramp = (1 - t) * _COLOUR_DENSITY_LO + t * _COLOUR_DENSITY_HI
    nonzero = (density_uint8 > 0)[..., None]
    rgb = np.where(nonzero, ramp, _COLOUR_BG[None, None, :]).astype(np.uint8)

    img = Image.fromarray(rgb, mode="RGB").resize(
        (w * upscale, h * upscale),
        resample=Image.NEAREST,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, format="PNG", optimize=True)
