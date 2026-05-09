"""VLM scene scout — discover what's actually in the video before SAM 3.1.

Replaces the static 40-phrase vocabulary that used to drive SAM 3.1's
open-vocabulary detector with a per-scene list discovered by Gemini 2.5
Flash. The scout chops the timeline into temporal slices, fires one Flash
call per slice in parallel (via asyncio.gather), then merges + dedupes
the phrases.

Why per-slice batches instead of one global pass:
  - 6 evenly-spaced frames out of a 500-frame walkthrough is fine for "what
    objects are in this single room" but bad for multi-room walkthroughs or
    briefly-framed objects (Roomba on the floor, a cat passing through, a
    shelf the camera lingers on for ~30 frames before panning away). Those
    miss the global samples entirely.
  - Gemini Flash legibility drops past ~10–15 images per call (attention
    spreads thin, instances merge, detail is dropped). So "more frames" has
    to mean "more calls", not "one bigger call".
  - One call per ~50 frames + 6 images per call → ~12% temporal sampling
    on a typical clip. All calls go out concurrently with asyncio.gather,
    so wall-clock is bounded by the slowest single call (~3–8s).

Why this matters for SAM 3.1 cost:
  - Each phrase the scout emits costs SAM one bidirectional propagation
    across the full clip. So phrase quantity directly bounds SAM wall-clock.
  - We dedupe aggressively (case-insensitive, whitespace-trimmed) before
    handing the list to SAM, and cap at `_MAX_PROMPTS` so SAM time stays
    bounded even on visually busy scenes.

The scout never returns regions ("kitchen"), materials ("wood"), or
abstractions ("lighting") — only segmentable noun phrases.
"""

from __future__ import annotations

import asyncio
import logging
import math
from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from .vlm import call_vlm_async

logger = logging.getLogger(__name__)


# Universal segmentable categories VLMs sometimes drop from enumerations
# (people walking through, doors/windows that frame the scene). Always
# added; SAM 3.1 only finds them if visible, so adding them costs at most
# three extra propagations on scenes where they're absent.
_SAFETY_NET: list[str] = ["person", "door", "window"]

# Each prompt costs one bidirectional propagation, so this directly bounds
# SAM 3.1 wall-clock. Bumped from 25 → 35 because the multi-batch scout
# produces a richer list and we don't want the cap to throw recall away.
_MAX_PROMPTS = 35

# Roughly one Flash call per this many frames. 50 = ~10 batches on a
# 500-frame walkthrough. Calls are async-parallel so this only changes
# coverage, not wall-clock.
_FRAMES_PER_BATCH = 50

# How many evenly-spaced frames each batch sends to Gemini. 6 is the
# Flash legibility sweet spot: more than that and the model starts merging
# instances across images.
_IMAGES_PER_BATCH = 6


class SceneInventory(BaseModel):
    """Structured response from one scout VLM call (one temporal slice)."""

    phrases: list[str] = Field(
        description=(
            "Concrete, segmentable noun phrases for every distinct physical "
            "object class visible in the frames. Examples: 'office chair', "
            "'ceramic coffee mug', 'guitar amplifier', 'wall outlet'. NOT "
            "regions ('kitchen'), materials ('wood'), or abstractions "
            "('lighting'). Up to 20 entries, ordered by visual prominence."
        )
    )
    reasoning: str = Field(default="", description="One short sentence.")


_PROMPT = """\
You are looking at {n} frames sampled from a portion of a video walkthrough \
of a real indoor scene. Your job is to enumerate every distinct physical \
object class visible in THESE frames so a downstream segmentation model \
can detect and track each one.

Rules:
- Use concrete noun phrases that can be outlined as a 2D mask (e.g. \
"office chair", "ceramic mug", "guitar amplifier", "ceiling fan", \
"power strip").
- Do NOT include regions ("kitchen", "living room"), materials ("wood \
floor", "marble"), parts of larger objects you've already named ("chair \
leg" when you said "chair"), or abstractions ("lighting", "shadow").
- Be specific where you can ("electric bass guitar" beats "guitar"; \
"track lighting" beats "lights"). Specificity helps the segmentation \
model find the right object.
- Include each class once even if multiple instances are visible — the \
downstream model finds every instance per phrase.
- Up to 20 entries, ordered by visual prominence (biggest / most central \
first). Don't pad — only list what's actually in these frames.\
"""


def _slice_boundaries(n_total: int, frames_per_slice: int) -> list[tuple[int, int]]:
    """Partition [0, n_total) into roughly equal consecutive [start, end) slices."""
    n_slices = max(1, math.ceil(n_total / frames_per_slice))
    boundaries = np.linspace(0, n_total, n_slices + 1).astype(int)
    return [(int(boundaries[i]), int(boundaries[i + 1])) for i in range(n_slices)]


def _within_slice_indices(start: int, end: int, n_pick: int) -> list[int]:
    span = end - start
    if span <= n_pick:
        return list(range(start, end))
    return [int(round(i)) for i in np.linspace(start, end - 1, n_pick)]


def _load_frame(path: Path, max_side: int = 1024) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return np.asarray(img)


async def _scout_one_batch(
    frame_paths: list[Path],
    indices: list[int],
    batch_idx: int,
    n_batches: int,
    vlm_model: str,
) -> list[str]:
    """Run one Gemini Flash call covering one temporal slice of the video."""
    keyframes = [_load_frame(frame_paths[i]) for i in indices]
    prompt = _PROMPT.format(n=len(keyframes))
    try:
        result = await call_vlm_async(prompt, keyframes, SceneInventory, model=vlm_model)
        phrases = [p.strip() for p in result.phrases if p and p.strip()]
        print(
            f"[scout]   batch {batch_idx + 1}/{n_batches} (frames "
            f"{indices[0]}..{indices[-1]}): {len(phrases)} phrases",
            flush=True,
        )
        return phrases
    except Exception as e:  # noqa: BLE001
        logger.warning("scout batch %d/%d failed: %s", batch_idx + 1, n_batches, e)
        print(
            f"[scout]   batch {batch_idx + 1}/{n_batches} FAILED "
            f"({type(e).__name__}: {e})",
            flush=True,
        )
        return []


async def _scout_all_batches(
    frame_paths: list[Path],
    vlm_model: str,
) -> list[list[str]]:
    slices = _slice_boundaries(len(frame_paths), _FRAMES_PER_BATCH)
    n_batches = len(slices)
    print(
        f"[scout] dispatching {n_batches} parallel Flash calls "
        f"(~{_FRAMES_PER_BATCH} frames per slice, {_IMAGES_PER_BATCH} images per call)",
        flush=True,
    )

    tasks = [
        _scout_one_batch(
            frame_paths,
            _within_slice_indices(start, end, _IMAGES_PER_BATCH),
            i,
            n_batches,
            vlm_model,
        )
        for i, (start, end) in enumerate(slices)
    ]
    return await asyncio.gather(*tasks)


def discover_scene_prompts(
    frames_dir: Path,
    vlm_model: str = "gemini-2.5-flash",
    n_frames: int = 6,  # kept for API back-compat; now per-batch, not global
) -> list[str]:
    """Return the concrete-noun-phrase list SAM 3.1 should look for.

    Fans out one Gemini Flash call per ~50-frame slice, runs them in parallel
    via asyncio.gather, then merges + dedupes. On total VLM failure (every
    batch errors) returns just the safety-net list so the pipeline still
    runs in degraded mode.
    """
    frame_paths = sorted(
        p for p in frames_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )
    if not frame_paths:
        raise SystemExit(f"no frames under {frames_dir}")

    print(
        f"[scout] {len(frame_paths)} frames available, model={vlm_model}",
        flush=True,
    )

    try:
        per_batch = asyncio.run(_scout_all_batches(frame_paths, vlm_model))
    except Exception as e:  # noqa: BLE001
        logger.warning("scout fan-out failed: %s", e)
        print(
            f"[scout] fan-out FAILED ({type(e).__name__}: {e}) — "
            f"falling back to safety net only",
            flush=True,
        )
        per_batch = []

    # Merge: preserve first-seen order across batches, dedupe case-insensitively.
    seen: set[str] = set()
    merged: list[str] = []
    for batch in per_batch:
        for phrase in batch:
            key = phrase.lower().strip()
            if key and key not in seen:
                seen.add(key)
                merged.append(phrase.strip())

    # Append safety net, then cap.
    for phrase in _SAFETY_NET:
        key = phrase.lower().strip()
        if key not in seen:
            seen.add(key)
            merged.append(phrase)

    if len(merged) > _MAX_PROMPTS:
        print(
            f"[scout] dedup yielded {len(merged)} phrases — capping at "
            f"{_MAX_PROMPTS} (dropping lowest-prominence tail)",
            flush=True,
        )
        merged = merged[:_MAX_PROMPTS]

    print(
        f"[scout] final SAM 3.1 prompt list ({len(merged)} phrases): {merged}",
        flush=True,
    )
    return merged
