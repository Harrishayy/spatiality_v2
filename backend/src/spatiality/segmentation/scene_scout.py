"""VLM scene scout — discover what's in each temporal slice of the video.

Replaces the static 40-phrase vocabulary that used to drive open-vocab
detection with a per-scene, per-slice list discovered by Gemini 2.5
Flash. The scout chops the timeline into ~20 temporal slices, fires one
Flash call per slice in parallel (asyncio.gather), then returns each
phrase tagged with the frame range it was discovered in so the GDINO
sweep only fires that phrase over the relevant frames instead of the
whole video.

Why per-slice scoping matters:
  - GDINO runs once per frame in a dot-separated multi-phrase query;
    keeping each phrase's frame range tight lets the downstream scope
    filter discard out-of-window detections without re-running GDINO.
  - When the same phrase appears in multiple slices we union the slice
    ranges so the linker keeps a single track with stable identity
    across those slices (avoids cross-track stitching for the easy case).

Why a tiny global safety net is still useful:
  - "person" is universal — humans walk through scenes briefly enough
    that scout often misses them in any given slice. Running this as a
    global propagation costs one full-video pass and catches them
    regardless of when they appear.
  - Door / window etc. don't go on the safety net: they're static and
    one slice discovery is enough.

The scout never returns regions ("kitchen"), materials ("wood"), or
abstractions ("lighting") — only segmentable noun phrases.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from .vlm import call_vlm_async

logger = logging.getLogger(__name__)


# Closed-class safety-net taxonomy (GLIP / OWL-ViT-2 hybrid scouting,
# Li et al. CVPR 2022; Minderer et al. NeurIPS 2023). Open-vocab scouts
# miss common objects depending on which slice gets sampled — laptops
# absent in slice 4 may still appear in slices 6-8, and so on. Pairing
# scout's open-vocab discovery with a fixed closed-class list ensures
# these recurrent indoor classes are always candidate phrases for GDINO.
# Each entry adds ~one phrase to the multi-phrase GDINO query, which
# scales sub-linearly in the text encoder — wall-clock cost is negligible.
_GLOBAL_SAFETY_NET: list[str] = [
    "door",
    "clothes rack",
    "closet",
    "laundry bag",
    "ceiling light",
]

# How many temporal slices to chop the video into. ~20 is the sweet spot:
# enough that each slice covers ~25-30 frames (small enough to run cheap
# slice-bounded SAM passes), few enough that each Gemini call still sees
# 6 well-spread frames (Flash legibility holds up).
_TARGET_N_SLICES = 20

# Don't slice below this — tiny slices give Gemini too little context.
_MIN_FRAMES_PER_SLICE = 8

# Open-vocab discovery cap. 70 scoped phrases + global safety net keeps
# the joined dot-separated GDINO text query comfortably under the 256-token
# BERT cap that the GDINO text branch enforces. The closed-class safety
# net above operates orthogonally to this cap, so the lower-bound on
# coverage is the safety-net length even if scout discovers nothing.
_MAX_SCOPED_PROMPTS = 70

# How many evenly-spaced frames each batch sends to Gemini. 6 is the
# Flash legibility sweet spot: more than that and the model starts merging
# instances across images.
_IMAGES_PER_BATCH = 6

# Frames of padding added on each side of a phrase's slice range before
# handing it to the GDINO sweep. Catches objects that drift across slice
# boundaries without scout flagging the adjacent slice.
_RANGE_PADDING_FRAMES = 15


@dataclass
class ScopedPrompt:
    """A phrase + the absolute frame range GDINO should detect it in.

    `frame_range` is ``None`` for safety-net (global) prompts that should
    fire across the entire video. Otherwise it's a half-open
    ``[start, end)`` interval over absolute video frame indices, already
    padded for entry/exit slack.
    """

    phrase: str
    frame_range: tuple[int, int] | None


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


def _slice_boundaries(n_total: int, target_n_slices: int) -> list[tuple[int, int]]:
    """Partition [0, n_total) into roughly equal consecutive [start, end) slices.

    Aims for ``target_n_slices`` slices but never lets a slice get smaller
    than ``_MIN_FRAMES_PER_SLICE``.
    """
    n_slices = max(1, min(target_n_slices, n_total // _MIN_FRAMES_PER_SLICE))
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
    slices: list[tuple[int, int]],
    vlm_model: str,
) -> list[list[str]]:
    n_batches = len(slices)
    print(
        f"[scout] dispatching {n_batches} parallel Flash calls "
        f"(target ~{_TARGET_N_SLICES} slices, {_IMAGES_PER_BATCH} images per call)",
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


def _checkpoint_path(frames_dir: Path) -> Path:
    """Where to cache scout output. Sits next to the scene's geometry artefacts."""
    return frames_dir.parent / "scout_prompts.json"


def _serialise(scoped: list[ScopedPrompt]) -> list[dict]:
    return [{"phrase": s.phrase, "frame_range": list(s.frame_range) if s.frame_range else None}
            for s in scoped]


def _deserialise(payload: list[dict]) -> list[ScopedPrompt]:
    out: list[ScopedPrompt] = []
    for p in payload:
        fr = p.get("frame_range")
        out.append(ScopedPrompt(
            phrase=p["phrase"],
            frame_range=tuple(fr) if fr is not None else None,
        ))
    return out


def discover_scene_prompts(
    frames_dir: Path,
    vlm_model: str = "gemini-2.5-flash",
    n_frames: int = 6,  # kept for API back-compat; now per-batch, not global
) -> list[ScopedPrompt]:
    """Return per-slice scoped prompts the GDINO sweep should look for.

    Returns a list of ``ScopedPrompt`` objects, each tagged with the
    absolute frame range over which GDINO should detect that phrase.
    Phrases discovered in multiple slices have their ranges unioned (so
    the linker keeps single-track identity across runs of the same
    object class). Safety-net phrases get ``frame_range=None`` and fire
    across the full video.

    On total VLM failure (every batch errors) returns just the safety net.
    """
    # Resume shortcut — scout is the only stage where the same call costs
    # 60+ seconds AND the result is deterministic enough to cache. If we
    # already have a checkpoint, return it directly.
    ckpt = _checkpoint_path(frames_dir)
    if ckpt.exists():
        try:
            scoped = _deserialise(json.loads(ckpt.read_text()))
            print(f"[scout] resuming from {ckpt.name} ({len(scoped)} prompts)", flush=True)
            return scoped
        except Exception as e:  # noqa: BLE001
            logger.warning("could not parse %s (%s); re-running scout", ckpt.name, e)

    frame_paths = sorted(
        p for p in frames_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )
    if not frame_paths:
        raise SystemExit(f"no frames under {frames_dir}")

    n_total = len(frame_paths)
    slices = _slice_boundaries(n_total, _TARGET_N_SLICES)
    print(
        f"[scout] {n_total} frames → {len(slices)} slices "
        f"(~{n_total // max(1, len(slices))} frames each), model={vlm_model}",
        flush=True,
    )

    try:
        per_batch = asyncio.run(_scout_all_batches(frame_paths, slices, vlm_model))
    except Exception as e:  # noqa: BLE001
        logger.warning("scout fan-out failed: %s", e)
        print(
            f"[scout] fan-out FAILED ({type(e).__name__}: {e}) — "
            f"falling back to safety net only",
            flush=True,
        )
        per_batch = [[] for _ in slices]

    # Track which slices each phrase appeared in (case-insensitive key,
    # preserving the casing of the first occurrence).
    first_casing: dict[str, str] = {}
    phrase_to_slice_idxs: dict[str, list[int]] = {}
    for batch_idx, batch_phrases in enumerate(per_batch):
        for phrase in batch_phrases:
            key = phrase.lower().strip()
            if not key:
                continue
            if key not in first_casing:
                first_casing[key] = phrase.strip()
            phrase_to_slice_idxs.setdefault(key, []).append(batch_idx)

    # Build scoped prompts, range-unioned + padded, capped to MAX_SCOPED.
    scoped: list[ScopedPrompt] = []
    # Order phrases by first-appearance slice index, then by appearance count
    # within that slice — keeps prominent objects ahead of long-tail filler.
    keys_ordered = sorted(
        phrase_to_slice_idxs.keys(),
        key=lambda k: (min(phrase_to_slice_idxs[k]), -len(phrase_to_slice_idxs[k])),
    )
    for key in keys_ordered:
        slice_idxs = phrase_to_slice_idxs[key]
        union_start = min(slices[i][0] for i in slice_idxs)
        union_end = max(slices[i][1] for i in slice_idxs)
        padded_start = max(0, union_start - _RANGE_PADDING_FRAMES)
        padded_end = min(n_total, union_end + _RANGE_PADDING_FRAMES)
        scoped.append(
            ScopedPrompt(
                phrase=first_casing[key],
                frame_range=(padded_start, padded_end),
            )
        )

    if len(scoped) > _MAX_SCOPED_PROMPTS:
        print(
            f"[scout] dedup yielded {len(scoped)} scoped phrases — capping at "
            f"{_MAX_SCOPED_PROMPTS} (dropping lowest-prominence tail)",
            flush=True,
        )
        scoped = scoped[:_MAX_SCOPED_PROMPTS]

    # Append safety-net phrases as global prompts (frame_range=None).
    seen_keys = {p.phrase.lower().strip() for p in scoped}
    for phrase in _GLOBAL_SAFETY_NET:
        if phrase.lower().strip() not in seen_keys:
            scoped.append(ScopedPrompt(phrase=phrase, frame_range=None))

    n_scoped = sum(1 for p in scoped if p.frame_range is not None)
    n_global = sum(1 for p in scoped if p.frame_range is None)
    print(
        f"[scout] final GDINO prompt list: {n_scoped} scoped + {n_global} global "
        f"(total {len(scoped)})",
        flush=True,
    )
    for p in scoped:
        if p.frame_range is None:
            print(f"[scout]   • '{p.phrase}' [global, all {n_total} frames]", flush=True)
        else:
            s, e = p.frame_range
            print(f"[scout]   • '{p.phrase}' [{s}..{e}) ({e - s} frames)", flush=True)

    # Persist for instant resume on retry. Keeps Stage 3.2/3.5 GPU iterations
    # cheap if a downstream stage fails — scout never gets re-run.
    try:
        ckpt.write_text(json.dumps(_serialise(scoped), indent=2))
        print(f"[scout] checkpoint → {ckpt.name}", flush=True)
    except Exception as e:  # noqa: BLE001
        logger.warning("could not write %s: %s", ckpt.name, e)

    return scoped
