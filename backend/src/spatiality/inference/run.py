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
from .frame_select import select_frames

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
      conf_min (float): drop pixels with VGGT depth_conf below this
        absolute floor. Default 0.15 — matches the old `spatiality` repo's
        VGGT_DEPTH_CONF_MIN=0.2 (slightly looser to keep textureless
        walls/floor that 0.2 wiped on handheld captures). depth_conf uses
        `expp1` activation, so values cluster >>1 on confident pixels and
        drift toward 0 on sky / blur / dark. Set to 0 to disable.
      pixel_stride (int): take every Nth pixel per frame before unprojection.
        Default 2 → 4× fewer points per frame; the target_count cap below
        keeps the final cloud size bounded regardless of stride.
      target_count (int): random-subsample the final cloud to at most this
        many points. Default 50,000,000 — three.js Points handles 50 M
        easily on M-series Macs / RTX 3070+ (300 MB GPU buffer; 800 MB
        on-disk PLY). Bump to 100_000_000 for "walk-in" demo captures
        (1.6 GB PLY; ~600 MB GPU; works on M2/M3 Pro/Max). Set None to
        disable the cap entirely.
      depth_grad_max (float): drop pixels where |∇depth|/depth > this.
        Default 0.06. Silhouette guard — kills floaters at object edges.
      depth_far_pct, depth_far_mult (float): drop pixels with depth above
        percentile_of_(depth_far_pct) × depth_far_mult per frame. Defaults
        95.0 and 1.5. Removes sky / unbounded background.
      blur_drop_pct (float): drop the bottom X fraction of frames by
        Laplacian variance before they hit the model. Default 0.20.
        Set to 0 to disable.
    """
    conf_min = float(kwargs.get("conf_min", 0.15))
    pixel_stride = int(kwargs.get("pixel_stride", 2))
    target_count = int(kwargs.get("target_count", 50_000_000))
    depth_grad_max = float(kwargs.get("depth_grad_max", 0.06))
    depth_far_pct = float(kwargs.get("depth_far_pct", 95.0))
    depth_far_mult = float(kwargs.get("depth_far_mult", 1.5))
    blur_drop_pct = float(kwargs.get("blur_drop_pct", 0.20))
    frames_max_kw = kwargs.get("frames_max")  # None → use whatever ffmpeg produced

    in_dir = _data_root() / input_id
    out_dir = _artefact_root() / input_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "depth").mkdir(exist_ok=True)
    (out_dir / "depth_conf").mkdir(exist_ok=True)
    (out_dir / "frames").mkdir(exist_ok=True)
    # Optional VGGT point-head outputs — directories created lazily below
    # if and only if the inference call actually populated them.

    frame_paths = _list_frames(in_dir)
    if not frame_paths:
        raise SystemExit(f"no frames found under {in_dir}")
    n_pre_select = len(frame_paths)

    # Blur filter — drop motion-blurred frames BEFORE they hit FlashVGGT.
    # This is the single highest-impact fix for handheld iPhone captures: the
    # pose head's robustness depends on the encoder seeing in-focus content,
    # and a single blurry frame mid-sequence is enough to crash the global
    # feature bank into a wrong attractor (visible as ghost-duplicates of
    # objects in the resulting point cloud — see analysis at frames 480-491
    # of the prior IMG_7531 run, where pose ΔR exceeded 30° in single steps).
    # Ported from the old `spatiality` repo's `_frame_select.py`.
    if blur_drop_pct > 0.0:
        frame_paths = select_frames(
            frame_paths,
            frames_max=int(frames_max_kw) if frames_max_kw else len(frame_paths),
            blur_drop_pct=blur_drop_pct,
            log_prefix="[stage:poses] blur_filter",
        )
    print(f"[stage:poses] input_id={input_id} frames={len(frame_paths)}/{n_pre_select} "
          f"conf_min={conf_min} pixel_stride={pixel_stride} "
          f"target_count={target_count} depth_grad_max={depth_grad_max} "
          f"depth_far_pct={depth_far_pct} depth_far_mult={depth_far_mult} "
          f"blur_drop_pct={blur_drop_pct}", flush=True)

    # Crash-safety: stash the raw forward-pass tensors here. If anything
    # downstream of the GPU forward fails (pose decode, K rescale, file I/O),
    # the next retry resumes from this file instead of redoing 12 min of A100.
    forward_ckpt = out_dir / "_forward_preds.pt"

    t0 = time.time()
    results, meta = run_inference(frame_paths, checkpoint_path=forward_ckpt)

    # Persist per-frame depth + conf, copy/save the originals into frames/ so
    # the UI evidence gallery can serve them.
    print(f"[stage:poses] writing {len(results)} depth + conf + frame copies …", flush=True)
    t_write = time.time()
    # World-points stride for storage. Stage 3's lift only consumes
    # per-mask-pixel lookups, so a half-resolution copy is plenty and cuts
    # disk by 4× (1474×1472×6 bytes/frame × 573 frames ≈ 7.5 GB → ~1.9 GB
    # at stride-2). Stride-1 is exact; stride-4 is acceptable for very
    # large scenes if disk pressure rises.
    _WP_STRIDE = 2

    have_wp = any(r.world_points is not None for r in results)
    have_wpc = any(r.world_points_conf is not None for r in results)
    if have_wp:
        (out_dir / "world_points").mkdir(exist_ok=True)
    if have_wpc:
        (out_dir / "world_points_conf").mkdir(exist_ok=True)

    for i, r in enumerate(results):
        np.save(out_dir / "depth" / f"{r.frame_id}.npy", r.depth)
        np.save(out_dir / "depth_conf" / f"{r.frame_id}.npy", r.depth_conf)
        if r.world_points is not None:
            np.save(
                out_dir / "world_points" / f"{r.frame_id}.npy",
                r.world_points[::_WP_STRIDE, ::_WP_STRIDE].astype(np.float16),
            )
        if r.world_points_conf is not None:
            np.save(
                out_dir / "world_points_conf" / f"{r.frame_id}.npy",
                r.world_points_conf[::_WP_STRIDE, ::_WP_STRIDE].astype(np.float16),
            )
        png_path = out_dir / "frames" / f"{r.frame_id}.png"
        if not png_path.exists():
            from PIL import Image  # noqa: PLC0415
            Image.fromarray(r.image_rgb).save(png_path)
        if (i + 1) % 100 == 0 or (i + 1) == len(results):
            print(f"[stage:poses]   wrote {i+1}/{len(results)} "
                  f"({time.time()-t_write:.1f}s elapsed)", flush=True)
    if have_wp:
        print(f"[stage:poses]   world_points saved at stride-{_WP_STRIDE} "
              f"({len(results)} frames)", flush=True)

    print("[stage:poses] building points.ply …", flush=True)
    t_pts = time.time()
    xyz, rgb, conf = points_from_results(
        results,
        conf_min=conf_min,
        pixel_stride=pixel_stride,
        target_count=target_count,
        depth_grad_max=depth_grad_max,
        depth_far_pct=depth_far_pct,
        depth_far_mult=depth_far_mult,
    )
    _write_ply(out_dir / "points.ply", xyz, rgb, conf)
    print(f"[stage:poses] points.ply: {xyz.shape[0]:,} points "
          f"({(out_dir/'points.ply').stat().st_size/1e6:.1f} MB) "
          f"in {time.time()-t_pts:.1f}s", flush=True)

    first_size = (results[0].image_rgb.shape[0], results[0].image_rgb.shape[1])
    _write_cameras(out_dir / "cameras.json", results, first_size)
    print(f"[stage:poses] cameras.json written ({len(results)} K/R/t entries)", flush=True)

    duration = time.time() - t0
    stage_entry = {
        "status": "complete",
        "duration_s": duration,
        "method": meta["backend"],
        "frame_count": len(frame_paths),
        "iterations": 1,
    }
    _update_manifest(out_dir, stage_entry)

    # Final artefacts are on disk → drop the forward checkpoint to free the volume.
    if forward_ckpt.exists():
        try:
            forward_ckpt.unlink()
            print(f"[stage:poses] cleaned up forward checkpoint", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[stage:poses] WARN: could not delete {forward_ckpt}: {e}", flush=True)

    print(f"[stage:poses] DONE in {duration:.1f}s "
          f"(backend={meta['backend']}, {xyz.shape[0]:,} points, "
          f"{len(frame_paths)} frames)", flush=True)
    return {
        "input_id": input_id,
        "status": "complete",
        "stage": "poses",
        "backend": meta["backend"],
        "frame_count": len(frame_paths),
        "point_count": int(xyz.shape[0]),
        "duration_s": duration,
    }
