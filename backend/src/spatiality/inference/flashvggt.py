"""FlashVGGT (preferred) / base VGGT (fallback) wrapper.

The two share an API: a single forward over N images returns dense per-pixel
depth, per-pixel confidence, per-frame camera pose encoding, and (optionally) a
point map. We wrap the loading + inference logic so the rest of the pipeline
doesn't care which backend is in use.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class FrameResult:
    """Per-frame outputs from VGGT/FlashVGGT inference."""

    frame_id: str
    depth: np.ndarray          # (H, W) float32, metres in arbitrary scale
    depth_conf: np.ndarray     # (H, W) float32 in [0, 1]
    K: np.ndarray              # (3, 3) intrinsics
    R: np.ndarray              # (3, 3) extrinsics rotation, world→cam
    t: np.ndarray              # (3,)   extrinsics translation, world→cam
    image_rgb: np.ndarray      # (H, W, 3) uint8


def _try_load_flashvggt() -> tuple[object, str] | None:
    """Try to import and load FlashVGGT; return (model, name) or None."""
    try:
        from flashvggt.models.flashvggt import FlashVGGT  # type: ignore[attr-defined]

        model = FlashVGGT.from_pretrained("wzpscott/FlashVGGT")
        return model, "flashvggt"
    except Exception as e:  # noqa: BLE001
        logger.warning("FlashVGGT unavailable (%s); will try base VGGT", e)
        return None


def _try_load_vggt() -> tuple[object, str] | None:
    try:
        from vggt.models.vggt import VGGT  # type: ignore[attr-defined]

        model = VGGT.from_pretrained("facebook/VGGT-1B")
        return model, "vggt"
    except Exception as e:  # noqa: BLE001
        logger.error("Base VGGT load failed too (%s)", e)
        return None


def load_model(prefer: str = "flashvggt") -> tuple[object, str]:
    """Load FlashVGGT (preferred) with fallback to base VGGT.

    Returns (model, backend_name). Raises if neither loads.
    """
    if prefer == "flashvggt":
        attempts = [_try_load_flashvggt, _try_load_vggt]
    else:
        attempts = [_try_load_vggt, _try_load_flashvggt]

    for fn in attempts:
        result = fn()
        if result is not None:
            model, name = result
            logger.info("loaded geometry backbone: %s", name)
            return model, name

    raise RuntimeError("No geometry backbone available — install flashvggt or vggt")


def _load_and_preprocess_images(
    image_paths: Sequence[Path],
    target_long_side: int = 518,
) -> tuple[torch.Tensor, list[np.ndarray]]:
    """Match the canonical VGGT preprocessing: resize so long side = 518, square-pad.

    Returns:
      - tensor (N, 3, H, W) in [0, 1] float32
      - list of original RGB uint8 arrays (for points colouring)
    """
    from PIL import Image  # noqa: PLC0415

    tensors: list[torch.Tensor] = []
    originals: list[np.ndarray] = []

    for path in image_paths:
        with Image.open(path) as im:
            im = im.convert("RGB")
            originals.append(np.asarray(im))

            w, h = im.size
            scale = target_long_side / max(w, h)
            new_w, new_h = int(round(w * scale)), int(round(h * scale))
            im = im.resize((new_w, new_h), Image.Resampling.BILINEAR)

            arr = np.asarray(im, dtype=np.float32) / 255.0  # H, W, 3

            # Square pad to (target_long_side, target_long_side).
            pad_h = target_long_side - new_h
            pad_w = target_long_side - new_w
            arr = np.pad(
                arr,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode="constant",
                constant_values=0,
            )

            tensor = torch.from_numpy(arr).permute(2, 0, 1)  # C, H, W
            tensors.append(tensor)

    return torch.stack(tensors, dim=0), originals


def _decode_pose_enc(pose_enc: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert VGGT pose encoding (Nx9 = quat[4] + t[3] + fov[2]) to K/R/t.

    Both FlashVGGT and base VGGT use the same 9-D pose encoding; the helper
    `pose_encoding_to_extri_intri` is the canonical decoder.
    """
    try:
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        from flashvggt.utils.pose_enc import pose_encoding_to_extri_intri  # type: ignore[attr-defined]

    extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc)
    extrinsics = extrinsics.detach().cpu().numpy()  # (N, 4, 4) or (N, 3, 4)
    intrinsics = intrinsics.detach().cpu().numpy()  # (N, 3, 3)

    if extrinsics.shape[-2:] == (4, 4):
        R = extrinsics[..., :3, :3]
        t = extrinsics[..., :3, 3]
    else:
        R = extrinsics[..., :3, :3]
        t = extrinsics[..., :3, 3]
    return intrinsics, R, t


def run_inference(
    image_paths: Sequence[Path],
    device: str | None = None,
    chunk_size: int = 0,
) -> tuple[list[FrameResult], dict]:
    """Run geometry inference over a list of frames.

    Args:
      image_paths: ordered list of frame image paths (e.g. 0001.png, 0002.png, ...).
      device: "cuda" / "cpu". Auto-detect when None.
      chunk_size: 0 = let the model handle the full batch (FlashVGGT scales
        to 1k+ frames; base VGGT may OOM and need chunking).

    Returns:
      (frame_results, meta) where meta includes backend_name, duration_s, n_frames.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, backend_name = load_model()
    model = model.to(device).eval()

    images, originals = _load_and_preprocess_images(image_paths)
    images = images.to(device)

    t0 = time.time()
    with torch.inference_mode():
        # FlashVGGT and VGGT share this call signature: single batch in, dict out.
        if chunk_size and chunk_size < len(image_paths):
            preds = _chunked_forward(model, images, chunk_size)
        else:
            preds = model(images.unsqueeze(0))  # (1, N, 3, H, W) -> dict
    duration = time.time() - t0

    # Both repos return tensors keyed by "depth", "depth_conf", "pose_enc".
    depth = preds["depth"].squeeze(0).detach().cpu().numpy()       # (N, H, W) or (N, 1, H, W)
    if depth.ndim == 4:
        depth = depth[:, 0]
    depth_conf = preds["depth_conf"].squeeze(0).detach().cpu().numpy()
    if depth_conf.ndim == 4:
        depth_conf = depth_conf[:, 0]
    pose_enc = preds["pose_enc"].squeeze(0)                        # (N, 9)

    K_all, R_all, t_all = _decode_pose_enc(pose_enc)

    # The model operates on padded square 518×518 images; rescale depth back to
    # original frame dims so downstream unprojection lines up with the VLM
    # crops it'll generate from the same originals.
    results: list[FrameResult] = []
    for i, path in enumerate(image_paths):
        rgb = originals[i]
        h, w = rgb.shape[:2]
        d = _resize_to(depth[i], (h, w))
        c = _resize_to(depth_conf[i], (h, w))
        # Intrinsics produced for the 518×518 input frame; rescale K to the
        # original resolution so unprojection in lift.py matches the RGB pixels.
        scale_x = w / depth[i].shape[1]
        scale_y = h / depth[i].shape[0]
        K = K_all[i].copy()
        K[0, 0] *= scale_x; K[0, 2] *= scale_x
        K[1, 1] *= scale_y; K[1, 2] *= scale_y

        results.append(
            FrameResult(
                frame_id=path.stem,
                depth=d.astype(np.float32),
                depth_conf=c.astype(np.float32),
                K=K.astype(np.float32),
                R=R_all[i].astype(np.float32),
                t=t_all[i].astype(np.float32),
                image_rgb=rgb,
            )
        )

    meta = {
        "backend": backend_name,
        "duration_s": duration,
        "n_frames": len(image_paths),
        "device": device,
    }
    return results, meta


def _chunked_forward(model, images: torch.Tensor, chunk: int) -> dict:
    """Naive chunking for base VGGT on long sequences. FlashVGGT shouldn't need this."""
    n = images.shape[0]
    parts: list[dict] = []
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        parts.append(model(images[s:e].unsqueeze(0)))

    keys = parts[0].keys()
    out: dict = {}
    for k in keys:
        out[k] = torch.cat([p[k] for p in parts], dim=1)
    return out


def _resize_to(arr: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Bilinear resize a (H, W) float array to target_hw without bringing in cv2 explicitly."""
    import cv2  # noqa: PLC0415

    h, w = target_hw
    return cv2.resize(arr.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


def points_from_results(
    results: list[FrameResult],
    conf_threshold: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Confidence-gated unprojection over all frames.

    Returns (points_xyz, colors_rgb_uint8, confidences) flattened across all frames.
    """
    pts_all: list[np.ndarray] = []
    col_all: list[np.ndarray] = []
    conf_all: list[np.ndarray] = []

    for r in results:
        h, w = r.depth.shape
        ys, xs = np.where(r.depth_conf > conf_threshold)
        if len(xs) == 0:
            continue

        ds = r.depth[ys, xs]
        # Pixel → camera coords: x_cam = (x - cx) * d / fx
        fx, fy = r.K[0, 0], r.K[1, 1]
        cx, cy = r.K[0, 2], r.K[1, 2]
        x_cam = (xs.astype(np.float32) - cx) * ds / fx
        y_cam = (ys.astype(np.float32) - cy) * ds / fy
        z_cam = ds

        cam = np.stack([x_cam, y_cam, z_cam], axis=1)         # (N, 3)
        # Camera→world: world = R^T (cam - t)
        world = (r.R.T @ (cam - r.t).T).T
        pts_all.append(world.astype(np.float32))

        col_all.append(r.image_rgb[ys, xs])  # (N, 3)
        conf_all.append(r.depth_conf[ys, xs].astype(np.float32))

    if not pts_all:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8), np.empty((0,), np.float32)
    return (
        np.concatenate(pts_all, axis=0),
        np.concatenate(col_all, axis=0),
        np.concatenate(conf_all, axis=0),
    )
