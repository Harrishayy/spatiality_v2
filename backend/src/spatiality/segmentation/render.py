"""Orbital novel-view rendering of point clouds for VLM input.

Lane B sends a 6-view orbital + 1 anchor crop grid to Claude. Doing this with
real Gaussian splat rasterization needs OpenGL/EGL headless setup that's
fragile in containers. A simple per-pixel point rasteriser is more than enough
for VLM consumption — the VLM doesn't need photo-realism, it needs a clear,
multi-angle view of the object.

This module:
  - filters points to a track's OBB (with margin)
  - generates camera poses on a sphere around the OBB center
  - rasterises each view (z-buffered, splat-as-disk)
  - returns RGB images suitable for direct VLM upload
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------- PLY loader


def load_points_ply(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read our binary points.ply (xyz f32 + rgb u8 + confidence f32).

    Returns (xyz Nx3 f32, rgb Nx3 u8, conf N f32).
    """
    with path.open("rb") as f:
        header_lines: list[str] = []
        while True:
            line = f.readline().decode("ascii", errors="ignore")
            header_lines.append(line)
            if line.strip() == "end_header":
                break
        n = 0
        for h in header_lines:
            if h.startswith("element vertex"):
                n = int(h.split()[-1])
                break
        dtype = np.dtype(
            [
                ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                ("r", "u1"), ("g", "u1"), ("b", "u1"),
                ("c", "<f4"),
            ]
        )
        rec = np.frombuffer(f.read(n * dtype.itemsize), dtype=dtype)

    xyz = np.stack([rec["x"], rec["y"], rec["z"]], axis=1).astype(np.float32)
    rgb = np.stack([rec["r"], rec["g"], rec["b"]], axis=1).astype(np.uint8)
    conf = rec["c"].astype(np.float32)
    return xyz, rgb, conf


# ---------------------------------------------------------------------------- camera maths


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Standard right-handed look-at, returns world→camera 3×4 [R|t]."""
    fwd = target - eye
    fwd /= max(1e-8, np.linalg.norm(fwd))
    right = np.cross(fwd, up)
    right /= max(1e-8, np.linalg.norm(right))
    new_up = np.cross(right, fwd)

    R = np.stack([right, -new_up, fwd], axis=0)  # OpenCV: x right, y down, z fwd
    t = -R @ eye
    return np.concatenate([R, t.reshape(3, 1)], axis=1)


def orbital_poses(
    centre: np.ndarray,
    radius: float,
    n: int = 6,
    elevations: tuple[float, float] = (-15.0, 30.0),
) -> list[np.ndarray]:
    """`n` viewpoints distributed around `centre` at the given `radius`."""
    poses: list[np.ndarray] = []
    azimuths = np.linspace(0, 360, n, endpoint=False)
    elevs = np.linspace(elevations[0], elevations[1], n)
    up_world = np.array([0, -1, 0], dtype=np.float32)  # OpenCV +y is down → world up is -y

    for az, el in zip(azimuths, elevs, strict=False):
        az_r = np.deg2rad(az)
        el_r = np.deg2rad(el)
        eye = centre + radius * np.array(
            [np.cos(el_r) * np.cos(az_r), -np.sin(el_r), np.cos(el_r) * np.sin(az_r)],
            dtype=np.float32,
        )
        poses.append(_look_at(eye, centre, up_world))
    return poses


# ---------------------------------------------------------------------------- point rasterizer


def render_view(
    xyz: np.ndarray,
    rgb: np.ndarray,
    extrinsic: np.ndarray,
    image_size: tuple[int, int] = (512, 512),
    fov_deg: float = 50.0,
    point_radius_px: int = 2,
    background: tuple[int, int, int] = (24, 24, 28),
) -> np.ndarray:
    """Z-buffered point rasterizer.

    Returns an HxWx3 uint8 RGB image. No OpenGL — pure numpy.
    """
    h, w = image_size
    fy = (h / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
    fx = fy
    cx, cy = w / 2.0, h / 2.0

    # World → camera.
    cam = (extrinsic[:, :3] @ xyz.T + extrinsic[:, 3:4]).T  # (N, 3)
    z = cam[:, 2]
    front = z > 1e-3
    cam = cam[front]
    rgb_f = rgb[front]
    z = z[front]
    if not len(cam):
        return np.full((h, w, 3), background, dtype=np.uint8)

    u = (cam[:, 0] * fx / z + cx).astype(np.int32)
    v = (cam[:, 1] * fy / z + cy).astype(np.int32)

    in_frame = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u = u[in_frame]; v = v[in_frame]; z = z[in_frame]; rgb_f = rgb_f[in_frame]
    if not len(u):
        return np.full((h, w, 3), background, dtype=np.uint8)

    # Z-buffer composite. Sort by descending depth so nearer pixels overwrite.
    order = np.argsort(z)[::-1]
    u = u[order]; v = v[order]; rgb_f = rgb_f[order]

    img = np.full((h, w, 3), background, dtype=np.uint8)
    r = max(0, point_radius_px)
    if r == 0:
        img[v, u] = rgb_f
    else:
        for du in range(-r, r + 1):
            for dv in range(-r, r + 1):
                if du * du + dv * dv > r * r:
                    continue
                vu = v + dv
                uu = u + du
                ok = (uu >= 0) & (uu < w) & (vu >= 0) & (vu < h)
                img[vu[ok], uu[ok]] = rgb_f[ok]
    return img


# ---------------------------------------------------------------------------- track-aware orbit


def render_track_orbit(
    points_path: Path,
    centroid: np.ndarray,
    obb_corners: np.ndarray,
    n_views: int = 6,
    image_size: tuple[int, int] = (512, 512),
    bbox_margin: float = 1.5,
) -> list[np.ndarray]:
    """Render `n_views` orbital views focused on a track's OBB.

    Filters the cloud to the OBB AABB (×margin) so the VLM isn't drowned in
    the rest of the room.
    """
    xyz, rgb, _ = load_points_ply(points_path)

    aabb_min = obb_corners.min(axis=0)
    aabb_max = obb_corners.max(axis=0)
    centre = (aabb_min + aabb_max) / 2.0
    half_extent = (aabb_max - aabb_min) / 2.0 * bbox_margin
    lo = centre - half_extent
    hi = centre + half_extent

    inside = (
        (xyz[:, 0] >= lo[0]) & (xyz[:, 0] <= hi[0]) &
        (xyz[:, 1] >= lo[1]) & (xyz[:, 1] <= hi[1]) &
        (xyz[:, 2] >= lo[2]) & (xyz[:, 2] <= hi[2])
    )
    if inside.sum() < 100:
        # Fall back to the whole cloud at distance — the OBB might be tiny.
        focus = xyz; focus_rgb = rgb
    else:
        focus = xyz[inside]; focus_rgb = rgb[inside]

    radius = float(np.linalg.norm(half_extent)) * 1.6
    poses = orbital_poses(centroid.astype(np.float32), radius=radius, n=n_views)
    return [render_view(focus, focus_rgb, p, image_size=image_size) for p in poses]


def composite_grid(images: list[np.ndarray], cols: int = 3) -> np.ndarray:
    """Stack images into a single RGB grid (rows = ceil(N/cols))."""
    n = len(images)
    rows = (n + cols - 1) // cols
    h, w, _ = images[0].shape
    grid = np.full((rows * h, cols * w, 3), 24, dtype=np.uint8)
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        grid[r * h: (r + 1) * h, c * w: (c + 1) * w] = im
    return grid


def crop_anchor(frame_path: Path, bbox_2d: tuple[int, int, int, int], pad: int = 32) -> np.ndarray:
    """Crop the original keyframe around the 2D mask bbox + a margin."""
    img = np.array(Image.open(frame_path).convert("RGB"))
    h, w = img.shape[:2]
    x0, y0, x1, y1 = bbox_2d
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(w, x1 + pad); y1 = min(h, y1 + pad)
    return img[y0:y1, x0:x1]
