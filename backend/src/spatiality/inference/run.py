"""Inference entrypoint.

Stage 1 of the pipeline. Reads frames from the inputs volume, runs
FlashVGGT/VGGT, and writes the geometry artefacts segmentation will consume.

Outputs (relative to ``$SPATIALITY_ARTEFACTS_ROOT/<input_id>/``):

    points.ply          # confidence-gated dense colour cloud (xyz+rgb+conf)
    cameras.json        # per-frame K, R, t
    depth/<frame>.npy   # per-frame depth maps
    depth_conf/<frame>.npy  # per-frame depth confidences
    manifest.json       # writes/updates the "poses" stage entry
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .flashvggt import points_from_results, run_inference

logger = logging.getLogger(__name__)


_FRAME_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def _list_frames(input_dir: Path) -> list[Path]:
    """Return sorted frame paths from ``<input_dir>/frames/`` or ``<input_dir>``."""
    frames_dir = input_dir / "frames"
    root = frames_dir if frames_dir.is_dir() else input_dir
    paths = [p for p in sorted(root.iterdir()) if p.suffix.lower() in _FRAME_EXTS]
    return paths


def _data_root() -> Path:
    return Path(os.environ.get("SPATIALITY_DATA_ROOT", "/inputs"))


def _artefact_root() -> Path:
    return Path(os.environ.get("SPATIALITY_ARTEFACTS_ROOT", "/outputs"))


def _write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray, conf: np.ndarray) -> None:
    """Write a binary little-endian PLY: xyz float + rgb uchar + confidence float.

    Matches the schema the web SplatViewer parser expects (see SplatViewer.tsx
    PLY streaming parser around lines 1600–1850).
    """
    n = xyz.shape[0]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property float confidence\n"
        "end_header\n"
    )

    dtype = np.dtype(
        [
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("r", "u1"), ("g", "u1"), ("b", "u1"),
            ("c", "<f4"),
        ]
    )
    rec = np.empty(n, dtype=dtype)
    rec["x"] = xyz[:, 0]
    rec["y"] = xyz[:, 1]
    rec["z"] = xyz[:, 2]
    rec["r"] = rgb[:, 0]
    rec["g"] = rgb[:, 1]
    rec["b"] = rgb[:, 2]
    rec["c"] = conf

    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        f.write(rec.tobytes(order="C"))


def _write_cameras(path: Path, frame_results: list, image_size_first: tuple[int, int]) -> None:
    """Persist per-frame K, R, t in a flat JSON shape segmentation can reload cheaply."""
    h, w = image_size_first
    payload = {
        "frame_size": {"height": h, "width": w},
        "convention": "opencv",  # +y down, +z forward (camera looks down +z)
        "frames": [
            {
                "frame_id": r.frame_id,
                "K": r.K.tolist(),
                "R": r.R.tolist(),
                "t": r.t.tolist(),
                "size": [r.image_rgb.shape[0], r.image_rgb.shape[1]],
            }
            for r in frame_results
        ],
    }
    path.write_text(json.dumps(payload))


def _update_manifest(scene_dir: Path, stage_entry: dict, top_status: str | None = None) -> None:
    """Merge a stage entry into the per-scene manifest.json.

    The frontend polls manifest.json and renders the stage waterfall from it.
    Other stages (segmentation lanes) update their own keys; we only touch
    `poses` here.
    """
    manifest_path = scene_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {
            "scene_id": scene_dir.name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "processing",
            "stages": {
                "capture": {"status": "complete"},
                "poses": {"status": "pending"},
                "splat": {"status": "complete"},  # we skip dedicated splat opt; points.ply doubles as the splat source
                "segmentation": {"status": "pending"},
            },
            "artifacts": {},
            "stats": {"frame_count": 0, "object_count": 0, "splat_size_mb": 0.0},
        }

    manifest["stages"]["poses"] = stage_entry
    if top_status is not None:
        manifest["status"] = top_status

    manifest["artifacts"].setdefault("splat_ply", "points.ply")
    manifest["artifacts"]["cameras_json"] = "cameras.json"

    manifest_path.write_text(json.dumps(manifest, indent=2))


def run(input_id: str, **kwargs) -> dict:
    """Entry point called from ``modal_inference.py::run_inference_one``.

    ``kwargs`` accepted:
      conf_threshold (float): drop points below this confidence in points.ply (default 0.05).
      chunk_size (int): override the model batch size (0 = no chunking).
    """
    conf_threshold = float(kwargs.get("conf_threshold", 0.05))
    chunk_size = int(kwargs.get("chunk_size", 0))

    in_dir = _data_root() / input_id
    out_dir = _artefact_root() / input_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "depth").mkdir(exist_ok=True)
    (out_dir / "depth_conf").mkdir(exist_ok=True)
    (out_dir / "frames").mkdir(exist_ok=True)

    frame_paths = _list_frames(in_dir)
    if not frame_paths:
        raise SystemExit(f"no frames found under {in_dir}")

    logger.info("running geometry on %d frames", len(frame_paths))

    t0 = time.time()
    results, meta = run_inference(frame_paths, chunk_size=chunk_size)

    # Persist per-frame depth + conf, copy/save the originals into frames/ so
    # the UI evidence gallery can serve them.
    for r in results:
        np.save(out_dir / "depth" / f"{r.frame_id}.npy", r.depth)
        np.save(out_dir / "depth_conf" / f"{r.frame_id}.npy", r.depth_conf)

        # Save originals as PNG (frontend evidence URLs accept any extension;
        # PNG keeps colour fidelity for VLM inputs).
        png_path = out_dir / "frames" / f"{r.frame_id}.png"
        if not png_path.exists():
            from PIL import Image  # noqa: PLC0415

            Image.fromarray(r.image_rgb).save(png_path)

    xyz, rgb, conf = points_from_results(results, conf_threshold=conf_threshold)
    _write_ply(out_dir / "points.ply", xyz, rgb, conf)

    first_size = (results[0].image_rgb.shape[0], results[0].image_rgb.shape[1])
    _write_cameras(out_dir / "cameras.json", results, first_size)

    duration = time.time() - t0
    stage_entry = {
        "status": "complete",
        "duration_s": duration,
        "method": meta["backend"],
        "frame_count": len(frame_paths),
        "iterations": 1,
    }
    _update_manifest(out_dir, stage_entry)

    logger.info(
        "geometry done in %.2fs (%s, %d points, %d frames)",
        duration, meta["backend"], xyz.shape[0], len(frame_paths),
    )
    return {
        "input_id": input_id,
        "status": "complete",
        "stage": "poses",
        "backend": meta["backend"],
        "frame_count": len(frame_paths),
        "point_count": int(xyz.shape[0]),
        "duration_s": duration,
    }
