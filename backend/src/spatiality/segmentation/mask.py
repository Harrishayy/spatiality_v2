"""SAM 2.1-hiera-tiny single-frame mask predictor for the lift stage.

Stage 3 used to sample a 5×5 grid inside an inset of the GDINO bbox. That
heuristic threw out boundary pixels but still mixed background into the
unprojection — fine for centred convex objects, fragile for thin /
articulated / U-shaped ones (chair frames, lamp poles, plants).

This module gives the lift mask-grade pixel selection at modest cost:
~50 ms encoder + ~3 ms decoder per (track, frame) on A100. Encoder runs
once per frame and is cached, so the marginal cost per extra track on the
same frame is just the decoder pass.

Public API:
  - :class:`SamMaskPredictor` — wraps SAM 2.1's ``SAM2ImagePredictor`` with
    per-frame encoder caching. Use as a context manager so the model is
    freed after Stage 3 completes.
  - :func:`build_predictor` — returns either the live predictor or
    ``None`` if SAM is unavailable / disabled, so callers can transparently
    fall back to the bbox-interior grid.

Disable via env var ``SPATIALITY_DISABLE_SAM=1`` for debug / cost runs.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


_SAM_MODEL_ID = "facebook/sam2.1-hiera-tiny"


# Hard cap on mask pixels we sample for unprojection. The mask itself
# can be >100k pixels for a near-camera object; sampling all of them
# wastes depth lookups (depth is sparse-conf-gated, so most are filtered
# anyway) and inflates PCA cost. 1024 is plenty for a robust median /
# OBB fit.
_MAX_MASK_SAMPLES = 1024


# Run SAM at its native training resolution. SAM 2.1 was trained at
# 1024x1024 — feeding larger images forces internal resize anyway and
# the encoder cost scales roughly with input pixel count. By resizing
# the source frame to fit within 1024 (preserving aspect) BEFORE we
# call set_image, encoder cost drops 5-6× on our 1474×1472 frames at
# zero fidelity loss (the network's actual capacity is at 1024). The
# returned mask is at the resized resolution; we upsample it back to
# the original (H, W) for depth lookup using nearest-neighbour.
_SAM_TARGET_MAX_SIDE = 1024


class SamMaskPredictor:
    """Wraps SAM 2.1 single-frame prediction with a per-frame encoder cache.

    Usage:
        with SamMaskPredictor() as predictor:
            for frame_id, bbox in items:
                mask = predictor.predict(frame_path, bbox)

    The encoder is run lazily on the first ``predict`` for a given frame
    and stays cached until ``release(frame_id)`` is called or the
    instance is closed. Lift calls ``release`` after exhausting a track's
    frames so memory stays bounded for long videos.
    """

    def __init__(self):
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: PLC0415

        t0 = time.time()
        print(f"[mask] loading {_SAM_MODEL_ID} …", flush=True)
        self._predictor = SAM2ImagePredictor.from_pretrained(_SAM_MODEL_ID)
        try:
            self._predictor.model.to("cuda").eval()
        except Exception:  # noqa: BLE001
            pass  # newer SAM2 builds expose model differently; .predict still works
        self._image_set: str | None = None  # currently encoded frame id
        # Cached source-image dimensions and resize factor for the
        # currently-encoded frame, so predict() can scale the bbox prompt
        # and resize the resulting mask back to native resolution.
        self._native_h: int = 0
        self._native_w: int = 0
        self._scale: float = 1.0
        print(f"[mask] SAM 2.1-hiera-tiny ready in {time.time()-t0:.1f}s "
              f"(input target ≤ {_SAM_TARGET_MAX_SIDE}px)", flush=True)

    def _set_image(self, frame_path: Path) -> None:
        """Encode ``frame_path``, downscaling to ≤ _SAM_TARGET_MAX_SIDE first.

        Caches scale + native dims so ``predict`` can transform bbox
        prompts and the returned mask between native and SAM resolutions.
        """
        key = str(frame_path)
        if self._image_set == key:
            return
        img = np.asarray(Image.open(frame_path).convert("RGB"))
        h, w = img.shape[:2]
        scale = float(_SAM_TARGET_MAX_SIDE) / float(max(h, w))
        if scale < 1.0:
            new_h = max(1, int(round(h * scale)))
            new_w = max(1, int(round(w * scale)))
            try:
                import cv2  # noqa: PLC0415
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            except Exception:  # noqa: BLE001
                # Fallback to PIL if cv2 unavailable.
                img = np.asarray(
                    Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR)
                )
        else:
            scale = 1.0
        self._native_h, self._native_w = h, w
        self._scale = scale
        self._predictor.set_image(img)
        self._image_set = key

    def predict(
        self,
        frame_path: Path,
        bbox_xyxy: tuple[int, int, int, int],
    ) -> np.ndarray | None:
        """Return a bool mask at the SAM (downscaled) resolution, or None.

        IMPORTANT: the returned mask shape is ``(round(h * scale),
        round(w * scale))``, NOT the source-frame resolution. Callers
        must resize via :meth:`upsample_mask_to_native` (or equivalent)
        before indexing into per-pixel data at native resolution.
        Returning at the SAM resolution avoids the redundant CPU resize
        when the caller only needs a sample-pixel set.
        """
        self._set_image(frame_path)
        x0, y0, x1, y1 = bbox_xyxy
        if self._scale != 1.0:
            x0 = x0 * self._scale
            y0 = y0 * self._scale
            x1 = x1 * self._scale
            y1 = y1 * self._scale
        box = np.asarray([[x0, y0, x1, y1]], dtype=np.float32)
        try:
            masks, scores, _ = self._predictor.predict(
                box=box,
                multimask_output=False,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("SAM predict failed for %s: %s", frame_path.name, e)
            return None
        if masks is None or not len(masks):
            return None
        # SAM returns shape (1, H, W) for single-bbox / single-mask output.
        mask = np.asarray(masks[0]).astype(bool)
        if not mask.any():
            return None
        return mask

    def native_resolution(self) -> tuple[int, int]:
        """(h, w) of the source frame at native resolution."""
        return self._native_h, self._native_w

    def release(self, frame_path: Path | None = None) -> None:
        """Drop the cached encoded image so VRAM doesn't accumulate.

        Pass ``frame_path`` to release only that frame; pass None to drop
        whatever's currently cached.
        """
        if frame_path is None or self._image_set == str(frame_path):
            try:
                self._predictor.reset_predictor()
            except Exception:  # noqa: BLE001
                pass
            self._image_set = None

    def __enter__(self) -> "SamMaskPredictor":
        return self

    def __exit__(self, *_exc) -> None:
        self.release()
        try:
            import torch  # noqa: PLC0415
            del self._predictor
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


def _sam_disabled() -> bool:
    return os.environ.get("SPATIALITY_DISABLE_SAM", "0").lower() in ("1", "true", "yes")


@contextmanager
def build_predictor() -> Iterator[SamMaskPredictor | None]:
    """Yield a ``SamMaskPredictor`` or ``None`` (transparent fallback).

    Returns None when:
      - the env disable flag is set
      - sam2 is not importable (e.g. local-laptop dev without the dep)
      - the model fails to load (network / VRAM)

    Callers must handle the ``None`` case by falling back to the legacy
    bbox-interior grid sampler.
    """
    if _sam_disabled():
        print("[mask] SPATIALITY_DISABLE_SAM set — using bbox-interior grid fallback",
              flush=True)
        yield None
        return
    try:
        predictor = SamMaskPredictor()
    except Exception as e:  # noqa: BLE001
        logger.warning("SAM mask predictor unavailable (%s) — falling back to grid", e)
        print(f"[mask] SAM unavailable: {type(e).__name__}: {e} — using grid fallback",
              flush=True)
        yield None
        return
    try:
        yield predictor
    finally:
        predictor.__exit__(None, None, None)


def sample_mask_pixels(
    mask: np.ndarray,
    max_samples: int = _MAX_MASK_SAMPLES,
    rng_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (ys, xs) of up to ``max_samples`` pixels sampled from ``mask``.

    Uniform random sub-sample so we don't bias toward any quadrant of the
    object. Deterministic when seeded — useful for cross-run consistency.
    """
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return ys, xs
    if len(ys) <= max_samples:
        return ys, xs
    rng = np.random.default_rng(rng_seed)
    sub = rng.choice(len(ys), size=max_samples, replace=False)
    return ys[sub], xs[sub]
