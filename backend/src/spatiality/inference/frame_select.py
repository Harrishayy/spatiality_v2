"""Frame pre-selection for FlashVGGT inference.

Drop motion-blurred frames BEFORE they hit the pose head. Ported verbatim
from the old `spatiality` repo's `_frame_select.py` — handheld iPhone
captures of the kind we feed FlashVGGT have plenty of motion blur, and
blurry frames are the #1 cause of pose-estimation noise. The old pipeline
silently dropped the bottom 20% blurriest frames here; the new pipeline
was missing this step entirely, which directly caused the ghost-duplicate
artifacts we were seeing (same physical object reconstructed twice
because the pose head got confused by blurry frames mid-sequence).

Pipeline: stride down to ~2× target → drop the blurriest 20% by Laplacian
variance → even-spacing cap to budget. Cheap (cv2 releases the GIL during
imread + Laplacian, so a thread pool gives near-linear speedup).
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

# JPEG/PNG decode + Laplacian are CPU-bound but cv2 releases the GIL, so a
# thread pool gives near-linear speedup. Modal containers run with cpu=8
# so 8 workers is reasonable.
_FRAME_SELECT_WORKERS = int(
    os.environ.get("FRAME_SELECT_WORKERS", str(min(8, (os.cpu_count() or 4))))
)


def _laplacian_variance(path: Path) -> float:
    """Cheap motion-blur proxy: variance of the Laplacian of a small grayscale
    crop of the frame. Higher = sharper; lower = blurrier.

    Resizes to long-side 256 first so a 1552×2064 iPhone frame is graded in
    ~1 ms. The Laplacian variance metric is a standard sharpness estimator
    (used in OpenCV docs, photogrammetry pipelines, autofocus algorithms).
    """
    import cv2  # noqa: PLC0415

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    h, w = img.shape
    scale = 256.0 / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def _laplacian_variance_parallel(paths: list[Path]) -> np.ndarray:
    """Parallel scoring across a thread pool. cv2 releases the GIL, so
    threading scales well even on Modal's containerised CPU.
    """
    if not paths:
        return np.empty(0, dtype=np.float64)
    if _FRAME_SELECT_WORKERS <= 1 or len(paths) <= 1:
        return np.array([_laplacian_variance(p) for p in paths], dtype=np.float64)
    with ThreadPoolExecutor(max_workers=_FRAME_SELECT_WORKERS) as ex:
        # `map` preserves input order — critical so scores align with `paths`.
        return np.fromiter(
            ex.map(_laplacian_variance, paths), dtype=np.float64, count=len(paths)
        )


def select_frames(
    frame_paths: list[Path],
    *,
    frames_max: int,
    frames_min: int = 16,
    blur_drop_pct: float = 0.20,
    log_prefix: str = "frame_select",
) -> list[Path]:
    """Three-stage frame pre-filter.

    1. Stride down to ~2× target so the blur-scoring step doesn't have to
       grade every single frame (saves wall-time on long captures).
    2. Drop the bottom `blur_drop_pct` by Laplacian variance — kills the
       motion-blurred frames that VGGT/FlashVGGT's pose head can't recover
       reliable poses from.
    3. Even-spacing cap to `frames_max` so temporal coverage stays uniform.

    `frames_min` is a floor — for short captures we'd rather keep the whole
    thing than drop into a regime where blur filtering is over-aggressive.
    """
    n = len(frame_paths)
    if n == 0:
        return frame_paths

    # 1. Stride down to ~2× target candidates.
    if n > frames_max * 2:
        stride = max(1, n // (frames_max * 2))
        frame_paths = frame_paths[::stride]
        print(f"{log_prefix}: strided {n} → {len(frame_paths)} candidates (stride={stride})", flush=True)
        n = len(frame_paths)

    # 2. Drop blurriest by Laplacian variance.
    if n > frames_min and blur_drop_pct > 0.0:
        scores = _laplacian_variance_parallel(frame_paths)
        keep_count = max(frames_min, int(round(n * (1.0 - blur_drop_pct))))
        if keep_count < n:
            order = np.argsort(scores)[::-1]  # sharpest first
            keep_idx = sorted(order[:keep_count].tolist())
            dropped_scores = scores[sorted(order[keep_count:].tolist())]
            kept_scores = scores[keep_idx]
            frame_paths = [frame_paths[i] for i in keep_idx]
            print(
                f"{log_prefix}: dropped {n - keep_count} blurriest "
                f"({blur_drop_pct:.0%}) — "
                f"dropped Laplacian-var range [{dropped_scores.min():.1f}, {dropped_scores.max():.1f}], "
                f"kept range [{kept_scores.min():.1f}, {kept_scores.max():.1f}]",
                flush=True,
            )

    # 3. Even-spacing cap to budget.
    if len(frame_paths) > frames_max:
        idx = np.linspace(0, len(frame_paths) - 1, frames_max).round().astype(int)
        frame_paths = [frame_paths[i] for i in idx.tolist()]

    print(f"{log_prefix}: selected {len(frame_paths)} / {n} frames", flush=True)
    return frame_paths
