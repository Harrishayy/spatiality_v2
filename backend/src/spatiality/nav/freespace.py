"""Free-space / traversability map for a humanoid-sized robot.

Stage 5 of the pipeline. Takes the dense point cloud + camera poses produced
by inference (Stage 1) and the labelled tracks from segmentation (Stage 3+),
and emits a top-down 2D occupancy grid telling a robot which floor cells it
can stand on without colliding with anything between ankle and head height.

This is the layer that turns "a labelled 3D scene" into "something a
locomotion planner can consume." The motivation is deliberately concrete: a
humanoid walking through the captured room needs to know (a) where the floor
is, (b) which floor cells have a clear vertical column up to head height, and
(c) which cells are too close to an obstacle to step on.

Inputs (all from ``<scene_dir>/``):
    points.ply       confidence-gated dense colour cloud (xyz+rgb+conf)
    cameras.json     per-frame K, R, t in OpenCV convention (+y down)

Outputs (written into ``<scene_dir>/``):
    traversability.json  grid metadata + raw cell labels (uint8 encoded)
    traversability.png   top-down preview (green=traversable, red=obstacle,
                         grey=unknown, blue dots=camera path)

Coordinate convention on disk follows the rest of the pipeline: OpenCV world
frame, +y down, +z forward in the first camera. Up in world is therefore
"the −y direction of the camera's local frame, rotated into world." We do
NOT pre-flip y/z here — the viewer applies the flip at render time, and the
traversability grid is generated in the same frame as ``points.ply`` so the
overlay co-registers without any extra transform.

This stage runs on CPU inside the segmentation Modal container (numpy +
Pillow already installed there). Wall-clock is ~5–15 s on a 50 M-point cloud.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Iterable

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


def _read_ply(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the binary little-endian PLY that ``inference._write_ply`` emits.

    Returns ``(xyz, rgb, conf)`` as a tuple of numpy arrays. Trusts the
    pipeline's own writer for the schema — if you point this at a foreign
    PLY it will raise.
    """
    with path.open("rb") as f:
        header_lines: list[str] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"unexpected EOF reading PLY header in {path}")
            header_lines.append(line.decode("ascii", errors="replace").strip())
            if header_lines[-1] == "end_header":
                break
        # Parse vertex count from "element vertex N".
        n = -1
        for h in header_lines:
            if h.startswith("element vertex"):
                n = int(h.split()[-1])
                break
        if n < 0:
            raise ValueError(f"PLY {path} missing 'element vertex N' line")

        rec = np.frombuffer(f.read(n * _PLY_DTYPE.itemsize), dtype=_PLY_DTYPE)

    xyz = np.stack([rec["x"], rec["y"], rec["z"]], axis=1).astype(np.float32)
    rgb = np.stack([rec["r"], rec["g"], rec["b"]], axis=1)
    conf = rec["c"].astype(np.float32)
    return xyz, rgb, conf


# ---------------------------------------------------------------------------- scene-up estimation

def _camera_positions_and_up(cameras_json: dict) -> tuple[np.ndarray, np.ndarray]:
    """Recover world-space camera positions and a robust scene-up direction.

    In OpenCV convention each camera stores ``[R | t]`` such that
    ``X_cam = R @ X_world + t``. So the camera center in world is
    ``C = -R.T @ t`` and the camera-local "up" axis ``[0, -1, 0]`` (because
    image y points down) maps to world as ``R.T @ [0, -1, 0]``.

    Averaging the per-camera up vectors gives a far more stable estimate
    of the scene's gravity axis than either (a) "−y in world" (which is
    only true if frame 0 happens to be level) or (b) PCA of the point
    cloud (which gets confused by tall obstacles).
    """
    centers: list[np.ndarray] = []
    ups: list[np.ndarray] = []
    for f in cameras_json.get("frames", []):
        R = np.asarray(f["R"], dtype=np.float32)
        t = np.asarray(f["t"], dtype=np.float32).reshape(3)
        C = -R.T @ t
        up_world = R.T @ np.array([0.0, -1.0, 0.0], dtype=np.float32)
        centers.append(C)
        ups.append(up_world)
    if not centers:
        raise ValueError("cameras.json has no frames")
    cam_centers = np.stack(centers, axis=0)
    up = np.mean(np.stack(ups, axis=0), axis=0)
    norm = np.linalg.norm(up)
    if norm < 1e-6:
        # Fallback — orthogonal-to-camera-track approximation. Almost never
        # triggered in practice, but keeps the function total.
        up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    else:
        up = up / norm
    return cam_centers, up


def _ground_plane_basis(up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pick two orthonormal axes spanning the floor plane (perpendicular to up).

    Chooses the first basis vector to be the world-x direction projected
    onto the plane, so the grid's "u" axis stays approximately aligned with
    the captured space's natural orientation. Falls back to world-z when
    up is nearly parallel to x.
    """
    seed = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(seed, up))) > 0.9:
        seed = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    u = seed - up * float(np.dot(seed, up))
    u = u / max(float(np.linalg.norm(u)), 1e-6)
    v = np.cross(up, u)
    v = v / max(float(np.linalg.norm(v)), 1e-6)
    return u, v


# ---------------------------------------------------------------------------- core occupancy grid

def _estimate_floor_height(heights_above_camera_track: np.ndarray) -> float:
    """Pick the floor as the densest low-percentile band.

    We can't just take min — that would pick a single outlier point under
    the floor (or a stray reconstruction artefact below the room). We take
    the 2nd percentile as a starting point, then refine by finding the
    densest 10 cm band around it (mode of the lower tail). This is the same
    trick robust SLAM packages use for "ground plane elevation."
    """
    p_low = np.percentile(heights_above_camera_track, 2.0)
    p_high = np.percentile(heights_above_camera_track, 60.0)
    # Histogram the bottom 60% at 5 cm bins; pick the densest bin near p_low.
    mask = heights_above_camera_track <= p_high
    bins = np.arange(p_low - 0.20, p_low + 0.40 + 0.05, 0.05, dtype=np.float32)
    if bins.size < 2:
        return float(p_low)
    hist, edges = np.histogram(heights_above_camera_track[mask], bins=bins)
    if hist.sum() == 0:
        return float(p_low)
    densest = int(np.argmax(hist))
    return float(0.5 * (edges[densest] + edges[densest + 1]))


def _dilate_bool(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    """Binary dilation by a circular structuring element of integer radius.

    Pure numpy implementation so we don't add scipy as a dep just for one
    morphology call. ``radius_cells`` typically maps to a robot's body
    radius (e.g. 0.25 m / 0.05 m cell = 5 cells).
    """
    if radius_cells <= 0:
        return mask.copy()
    r = int(radius_cells)
    yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
    kernel = (xx * xx + yy * yy) <= (r * r)
    H, W = mask.shape
    pad = np.zeros((H + 2 * r, W + 2 * r), dtype=bool)
    pad[r : H + r, r : W + r] = mask
    out = np.zeros_like(pad)
    # Slide the kernel — vectorised over kernel offsets, not over pixels.
    for dy in range(2 * r + 1):
        for dx in range(2 * r + 1):
            if not kernel[dy, dx]:
                continue
            out[dy : dy + H, dx : dx + W] |= mask
    return out[r : H + r, r : W + r]


# ---------------------------------------------------------------------------- compute

# Cell labels emitted in traversability.json. Keep these stable — the frontend
# reads them directly.
CELL_UNKNOWN = 0
CELL_TRAVERSABLE = 1
CELL_OBSTACLE = 2

# Robot body envelope. Tuned for a generic ~1.7 m humanoid:
#   ankle  : low support band — what tells us "there's floor here"
#   knee   : early-obstacle band — catches low furniture (sofas, coffee tables)
#   hip    : main obstacle band — chairs, desks, kitchen counters
#   head   : tall obstacle band — hanging fixtures, low ceilings, shelving
# Heights are *above the estimated floor plane*, not raw world coordinates.
_BANDS_M: tuple[tuple[float, float], ...] = (
    (0.05, 0.20),   # ankle
    (0.20, 0.60),   # knee
    (0.60, 1.20),   # hip
    (1.20, 1.80),   # head
)


def compute_freespace(
    scene_dir: Path | str,
    *,
    cell_size_m: float = 0.05,
    robot_radius_m: float = 0.25,
    floor_support_min_points: int = 3,
    conf_floor: float = 0.20,
    max_extent_m: float = 12.0,
) -> dict:
    """Compute and persist the traversability grid for one scene.

    Returns a small metadata dict (NOT the full grid) so callers can log it
    without ballooning memory. The full grid lives in
    ``traversability.json``.

    Parameters
    ----------
    cell_size_m
        Side length of one grid cell. 5 cm matches the resolution of the
        underlying depth maps for indoor captures — finer than that and the
        grid is dominated by per-pixel noise, coarser and a humanoid foot
        straddles cells.
    robot_radius_m
        Inflate obstacles by this radius before computing traversable. 0.25
        m approximates the half-width of HMND-01-class humanoids' stance.
    floor_support_min_points
        Cells need at least this many points in the ankle band to count as
        "the floor exists here." Filters out columns where the cloud is too
        thin to trust.
    conf_floor
        Drop PLY points with depth-confidence below this. ``points.ply``
        already had a floor applied at write time (0.15 default), but this
        gives us a second pass aligned with the navigation-quality bar.
    max_extent_m
        Hard cap on grid side length, just to keep the rasteriser from
        allocating gigabyte grids if the cloud has an outlier point in
        another room. The grid is centred on the camera track and clipped
        to this radius before rasterisation.
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

    xyz, _rgb, conf = _read_ply(ply_path)
    if xyz.size == 0:
        raise ValueError(f"points.ply at {ply_path} has 0 points")

    if conf_floor > 0:
        keep = conf >= conf_floor
        xyz = xyz[keep]
        conf = conf[keep]
        if xyz.shape[0] == 0:
            raise ValueError(
                f"all {keep.size} points dropped at conf_floor={conf_floor}; "
                f"upstream PLY may already be over-pruned"
            )

    # Project everything onto (u, v, up). Height (along up) is what tells us
    # floor vs ceiling; (u, v) is the floor-plane footprint.
    cam_uv = np.stack(
        [cam_centers @ u_axis, cam_centers @ v_axis],
        axis=1,
    )
    cam_center_uv = cam_uv.mean(axis=0)

    pts_height = xyz @ up
    pts_uv = np.stack(
        [xyz @ u_axis, xyz @ v_axis],
        axis=1,
    ) - cam_center_uv  # recentre on camera track for a stable origin

    # Clip to a sane window around the camera track. Stops a single outlier
    # point in another room from inflating the grid to 100 m.
    in_window = (
        (np.abs(pts_uv[:, 0]) <= max_extent_m)
        & (np.abs(pts_uv[:, 1]) <= max_extent_m)
    )
    pts_uv = pts_uv[in_window]
    pts_height = pts_height[in_window]
    if pts_uv.shape[0] == 0:
        raise ValueError("no points survived the max_extent_m window — bad scale?")

    # Floor plane: heights of points *near the camera track* dominate the
    # lower percentile cleanly. Use the full filtered cloud — the histogram
    # trick in _estimate_floor_height handles outliers.
    floor_h = _estimate_floor_height(pts_height)
    rel_h = pts_height - floor_h  # height above estimated floor

    # Grid extents in metres → cells. Anchor the grid on (0, 0) in
    # camera-recentred coordinates so the origin is roughly the centre of
    # the captured space, regardless of the underlying world frame.
    half_extent = max_extent_m
    grid_n = int(np.ceil((2 * half_extent) / cell_size_m))
    # Build per-band occupancy masks at the chosen resolution.
    cell_i = np.floor((pts_uv[:, 0] + half_extent) / cell_size_m).astype(np.int64)
    cell_j = np.floor((pts_uv[:, 1] + half_extent) / cell_size_m).astype(np.int64)
    valid = (cell_i >= 0) & (cell_i < grid_n) & (cell_j >= 0) & (cell_j < grid_n)
    cell_i = cell_i[valid]
    cell_j = cell_j[valid]
    rel_h = rel_h[valid]

    # Count points per cell per band — done with bincount on a flat (i*N+j) key.
    flat = cell_i * grid_n + cell_j

    def _count_in_band(lo: float, hi: float) -> np.ndarray:
        m = (rel_h >= lo) & (rel_h < hi)
        if not m.any():
            return np.zeros(grid_n * grid_n, dtype=np.int32)
        counts = np.bincount(flat[m], minlength=grid_n * grid_n).astype(np.int32)
        return counts

    ankle_counts = _count_in_band(*_BANDS_M[0])
    obstacle_counts = np.zeros(grid_n * grid_n, dtype=np.int32)
    for lo, hi in _BANDS_M[1:]:  # knee, hip, head
        obstacle_counts += _count_in_band(lo, hi)

    ankle_grid = ankle_counts.reshape(grid_n, grid_n)
    obstacle_grid = obstacle_counts.reshape(grid_n, grid_n)

    floor_support = ankle_grid >= floor_support_min_points
    is_obstacle = obstacle_grid > 0

    # Inflate obstacles by the robot radius. The traversable set is then
    # "floor support AND not an inflated obstacle."
    inflated = _dilate_bool(is_obstacle, int(round(robot_radius_m / cell_size_m)))
    traversable = floor_support & (~inflated)

    labels = np.full((grid_n, grid_n), CELL_UNKNOWN, dtype=np.uint8)
    labels[is_obstacle] = CELL_OBSTACLE
    labels[traversable] = CELL_TRAVERSABLE

    # Tighten the grid to the bounding box of any-non-unknown cells +
    # 10-cell margin. The 12-m window kept the math simple; cropping makes
    # the JSON payload an order of magnitude smaller for typical rooms.
    nonzero = np.argwhere(labels != CELL_UNKNOWN)
    if nonzero.size:
        ymin = max(0, nonzero[:, 0].min() - 10)
        ymax = min(grid_n, nonzero[:, 0].max() + 11)
        xmin = max(0, nonzero[:, 1].min() - 10)
        xmax = min(grid_n, nonzero[:, 1].max() + 11)
        labels = labels[ymin:ymax, xmin:xmax]
        crop_origin_uv = (
            -half_extent + xmin * cell_size_m,
            -half_extent + ymin * cell_size_m,
        )
    else:
        crop_origin_uv = (-half_extent, -half_extent)

    # Serialisation. The full label grid is small (≲ 50 KB for a typical
    # room at 5 cm) so we just base64-encode the raw uint8 buffer rather
    # than RLE — keeps the reader trivial on the frontend.
    h_cells, w_cells = labels.shape
    n_trav = int((labels == CELL_TRAVERSABLE).sum())
    n_obst = int((labels == CELL_OBSTACLE).sum())
    n_unkn = int((labels == CELL_UNKNOWN).sum())

    meta = {
        "version": 1,
        "convention": "opencv",
        "cell_size_m": float(cell_size_m),
        "grid_shape": [int(h_cells), int(w_cells)],
        "origin_uv_m": [float(crop_origin_uv[0]), float(crop_origin_uv[1])],
        "floor_height_world": float(floor_h),
        "up_axis_world": [float(up[0]), float(up[1]), float(up[2])],
        "u_axis_world": [float(u_axis[0]), float(u_axis[1]), float(u_axis[2])],
        "v_axis_world": [float(v_axis[0]), float(v_axis[1]), float(v_axis[2])],
        "camera_center_uv_m": [float(cam_center_uv[0]), float(cam_center_uv[1])],
        "robot_radius_m": float(robot_radius_m),
        "bands_m": [list(b) for b in _BANDS_M],
        "cell_labels": {
            "unknown": CELL_UNKNOWN,
            "traversable": CELL_TRAVERSABLE,
            "obstacle": CELL_OBSTACLE,
        },
        "stats": {
            "traversable_cells": n_trav,
            "obstacle_cells": n_obst,
            "unknown_cells": n_unkn,
            "traversable_m2": float(n_trav * cell_size_m * cell_size_m),
            "obstacle_m2": float(n_obst * cell_size_m * cell_size_m),
        },
        "cells_b64": base64.b64encode(labels.tobytes(order="C")).decode("ascii"),
    }

    json_path = scene_dir / "traversability.json"
    json_path.write_text(json.dumps(meta))

    png_path = scene_dir / "traversability.png"
    _render_preview_png(
        labels=labels,
        cam_uv=cam_uv - np.array(crop_origin_uv, dtype=np.float32),
        cell_size_m=cell_size_m,
        out=png_path,
    )

    # Return a tiny summary (no cells) for logs / manifest.
    return {
        "shape": [int(h_cells), int(w_cells)],
        "cell_size_m": cell_size_m,
        "floor_height_world": float(floor_h),
        "stats": meta["stats"],
        "json_path": json_path.name,
        "png_path": png_path.name,
    }


# ---------------------------------------------------------------------------- preview rendering

# Sunset-palette greens / corals so the preview matches the web viewer's
# colour language without having to import any styling.
_COLOUR_TRAV = (96, 192, 128)
_COLOUR_OBST = (255, 107, 74)
_COLOUR_UNKN = (32, 26, 24)
_COLOUR_CAM_LINE = (255, 210, 156)


def _render_preview_png(
    *,
    labels: np.ndarray,
    cam_uv: Iterable[Iterable[float]],
    cell_size_m: float,
    out: Path,
    upscale: int = 6,
) -> None:
    """Top-down PNG showing the traversability classification + camera track.

    Each cell is rendered as an ``upscale``×``upscale`` block so the PNG is
    readable at a glance without forcing the viewer to zoom. The camera
    track is drawn as a thin polyline so reviewers can see where the
    capture happened relative to the inferred free space.
    """
    h, w = labels.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[labels == CELL_UNKNOWN] = _COLOUR_UNKN
    rgb[labels == CELL_TRAVERSABLE] = _COLOUR_TRAV
    rgb[labels == CELL_OBSTACLE] = _COLOUR_OBST

    img = Image.fromarray(rgb, mode="RGB").resize(
        (w * upscale, h * upscale),
        resample=Image.NEAREST,
    )

    draw = ImageDraw.Draw(img)
    cam_uv = np.asarray(cam_uv, dtype=np.float32)
    cell_px = float(upscale) / cell_size_m  # px per metre
    # cam_uv is already in (u_m, v_m) relative to the grid origin (because
    # the caller subtracted crop_origin_uv).
    pts_px = [(float(c[0]) * cell_px, float(c[1]) * cell_px) for c in cam_uv]
    if len(pts_px) >= 2:
        draw.line(pts_px, fill=_COLOUR_CAM_LINE, width=max(1, upscale // 2))
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, format="PNG", optimize=True)
