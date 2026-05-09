"""SAM 3.1 detection + video tracking via the unified text-prompted predictor.

Rewritten against `facebookresearch/sam3` v1 actual API after the original
code (written against an imagined `build_sam3_image_predictor` /
`predict_everything`) was found to import nothing real.

How SAM 3.1 actually works:
  - One unified video predictor: ``build_sam3_predictor(version="sam3.1")``.
  - Open-vocabulary text prompts: ``add_prompt(text="chair")`` finds every
    instance of the phrase in the video and assigns each its own obj_id.
  - Mask + box prompts also supported, but text is what makes SAM 3.1
    different from SAM 2.
  - ``handle_stream_request({"type": "propagate_in_video", ...})`` yields one
    output per frame: ``{"frame_index": int, "outputs": {"out_obj_ids", ...,
    "out_binary_masks", ..., "out_boxes_xywh", ...}}``.

This module:
  1. starts a session on the frames dir
  2. loops over a default indoor-scene vocabulary (extensible) and
     ``add_prompt`` for each phrase at frame 0
  3. streams ``propagate_in_video`` (forward + backward from frame 0) and
     records per-frame masks for every obj_id the model produces
  4. filters tracks by length / area, saves PNG masks, writes
     ``sam3/tracks.json`` + ``masks/<track_id>/<frame_id>.png``

Outputs:
  outputs/<id>/sam3/tracks.json
  outputs/<id>/masks/<track_id>/<frame>.png
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFile

logger = logging.getLogger(__name__)


# At least one frame in the inputs volume has a truncated PNG tail (likely
# from an incomplete upload during ffmpeg → modal sync). SAM 3.1 owns the
# frame loader inside `start_session`, so the cleanest fix is the global
# PIL toggle: PIL fills truncated bytes with what it has, the model gets
# slightly degraded data on that frame, and the rest of the 500-frame
# sequence runs to completion instead of crashing.
ImageFile.LOAD_TRUNCATED_IMAGES = True


def _amp_ctx():
    """Autocast context for SAM 3.1.

    SAM 3.1's tracker uses flash-attn v3 (`flash_attn_interface`). Ampere /
    Ada (A100, 4090, …) flash-attn-3 only accepts fp16/bf16 — calling it
    with the model's default fp32 weights raises ``mha_fwd … only supports
    fp16 and bf16 data type`` for every prompt. Hopper (H100, H200) does
    accept fp32, so on those GPUs we skip autocast.

    We pick bf16 over fp16 because SAM 3.1 follows the SAM 2 / DINO recipe
    (large dynamic range; fp16 underflows in the cross-attention softmax)
    and the upstream FlashVGGT demo also defaults to bf16 on capable GPUs.
    """
    if not torch.cuda.is_available():
        return contextlib.nullcontext()
    major, _ = torch.cuda.get_device_capability(0)
    if major >= 9:  # Hopper+ supports fp32 flash-attn — no cast needed
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=torch.bfloat16)


# Fallback vocabulary used only when no `text_prompts` are passed AND the
# VLM scene-scout pass (`scene_scout.discover_scene_prompts`) is bypassed.
# Each phrase costs one bidirectional propagation across the full clip, so
# this list is intentionally short — anything missing should come from the
# scout, which sees the actual frames.
_FALLBACK_PROMPTS: list[str] = [
    "chair", "table", "sofa", "bed", "desk",
    "person", "door", "window",
]


# ---------------------------------------------------------------------------- types


@dataclass
class TrackFrame:
    frame_id: str
    mask_path: str          # relative to the artefacts root for this scene
    score: float
    bbox_2d: tuple[int, int, int, int]  # (x0, y0, x1, y1)


@dataclass
class Track:
    track_id: str
    frames: list[TrackFrame] = field(default_factory=list)
    text_prompt: str | None = None
    source: str = "text"


# ---------------------------------------------------------------------------- helpers


def _save_mask(mask: np.ndarray, out_path: Path) -> None:
    """Persist a binary mask as a single-channel PNG (255 inside, 0 outside)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(out_path)


def _bbox_of_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _to_numpy(x: Any) -> np.ndarray:
    """Tolerate either torch.Tensor or numpy.ndarray inputs from SAM3."""
    if x is None:
        return np.empty((0,))
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _patch_fa3_for_ampere() -> bool:
    """Swap SAM 3.1's FP8 flash-attn entry for a bf16 variant on Ampere.

    Root cause of the "FlashAttention on Ampere/Ada cards only supports
    fp16 and bf16 data type" wall: `sam3.perflib.fa3.flash_attn_func`
    upcasts q/k/v to ``torch.float8_e4m3fn`` before calling
    `flash_attn_interface.flash_attn_func`. FP8 is Hopper-only — A100's
    flash-attn-3 wheel rejects it with that exact message.

    Every SAM 3.1 attention site (sam/transformer.py, model/vitdet.py,
    model/decoder.py, model/model_misc.py) imports `flash_attn_func`
    *lazily* inside the forward, so replacing the module attribute before
    the first forward swaps the FP8 cast for bf16 across all of them. We
    only do this on Ampere (SM<9.0); Hopper keeps FP8 untouched.

    Returns True if the patch was applied.
    """
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] >= 9:
        return False

    import sam3.perflib.fa3 as _fa3  # type: ignore[import-not-found]

    def _flash_attn_func_bf16(q, k, v):
        from flash_attn_interface import flash_attn_func as _fa3_kernel  # type: ignore[import-not-found]

        qb = q.to(torch.bfloat16)
        kb = k.to(torch.bfloat16)
        vb = v.to(torch.bfloat16)
        return _fa3_kernel(qb, kb, vb).to(q.dtype)

    _fa3.flash_attn_func = _flash_attn_func_bf16
    print("[sam3] patched sam3.perflib.fa3.flash_attn_func: FP8 → bf16 (Ampere)", flush=True)
    return True


def _build_predictor():
    """Construct the unified SAM 3.1 video predictor.

    `build_sam3_predictor(version="sam3.1")` auto-downloads weights from
    HuggingFace (gated; needs HF_TOKEN with the license accepted on
    facebook/sam3.1). Returns a Sam3VideoPredictorMultiGPU-wrapped instance
    on the current GPU.

    Workaround for SAM3 v1: their `Sam3BasePredictor.start_session()`
    unconditionally passes `offload_state_to_cpu` to `model.init_state()`,
    but SAM 3.1's multiplex model (`Sam3MultiplexTrackingWithInteractivity`)
    doesn't accept that kwarg → TypeError. Their `add_prompt` filters kwargs
    by inspecting `model.add_prompt`'s signature; `start_session` doesn't.
    We monkey-patch `model.init_state` on the constructed predictor to
    accept-and-drop unknown kwargs, so unsupported `offload_*` flags pass
    through harmlessly.
    """
    import inspect as _inspect

    # Must run BEFORE build_sam3_predictor — the predictor's submodules
    # bind `flash_attn_func` lazily inside their forwards, so as long as the
    # module-level attribute is patched before any forward runs, every
    # attention path picks up the bf16 variant.
    _patch_fa3_for_ampere()

    from sam3.model_builder import build_sam3_predictor  # type: ignore[import-not-found]

    print("[sam3] building SAM 3.1 unified predictor (version=sam3.1) …", flush=True)
    predictor = build_sam3_predictor(version="sam3.1")

    _orig_init_state = predictor.model.init_state
    _accepted = set(_inspect.signature(_orig_init_state).parameters.keys())

    def _filtered_init_state(*args, **kwargs):
        dropped = [k for k in kwargs if k not in _accepted]
        if dropped:
            kwargs = {k: v for k, v in kwargs.items() if k in _accepted}
            print(f"[sam3] init_state: dropped unsupported kwargs {dropped}", flush=True)
        return _orig_init_state(*args, **kwargs)

    predictor.model.init_state = _filtered_init_state
    return predictor


# ---------------------------------------------------------------------------- main


def run_sam3(
    frames_dir: Path,
    out_dir: Path,
    seed_stride: int = 25,             # kept for API back-compat; unused with text prompts
    reprompt_stride: int = 100,        # ditto
    min_track_frames: int = 3,
    min_track_score: float = 0.0,      # SAM 3.1 doesn't always emit per-mask scores
    text_prompts: list[str] | None = None,
    extra_text_prompts: list[str] | None = None,
) -> list[Track]:
    """Run SAM 3.1 over the frames directory and return one Track per detected object.

    Args:
      frames_dir: directory of ordered frame PNGs (sorted = time order).
      out_dir:   where to write masks and tracks.json.
      min_track_frames: drop tracks shorter than this (in frames).
      min_track_score:  drop tracks with mean score below this.
      text_prompts: phrases to feed SAM 3.1's open-vocab detector. When
        provided (typically by the VLM scene-scout pass), this REPLACES the
        fallback vocabulary entirely. When None, the short fallback list is
        used so the pipeline still runs in scout-disabled / debug paths.
      extra_text_prompts: appended to whichever vocabulary is used. Useful
        for passing scene-specific phrases through the CLI.
    """
    frame_paths = sorted(frames_dir.iterdir())
    frame_paths = [p for p in frame_paths if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    if not frame_paths:
        raise SystemExit(f"no frames under {frames_dir}")

    if text_prompts:
        prompts = list(text_prompts)
        source_label = "scout"
    else:
        prompts = list(_FALLBACK_PROMPTS)
        source_label = "fallback"
    if extra_text_prompts:
        # Dedupe while preserving order — case-insensitive.
        seen = {p.lower() for p in prompts}
        for p in extra_text_prompts:
            if p and p.lower() not in seen:
                prompts.append(p)
                seen.add(p.lower())
    print(f"[sam3] {len(frame_paths)} frames, {len(prompts)} text prompts "
          f"(source={source_label}"
          f"{f', +{len(extra_text_prompts)} extra' if extra_text_prompts else ''})",
          flush=True)
    text_prompts = prompts

    masks_dir = out_dir / "masks"
    sam_dir = out_dir / "sam3"
    sam_dir.mkdir(parents=True, exist_ok=True)

    t = time.time()
    predictor = _build_predictor()
    print(f"[sam3] predictor built in {time.time()-t:.1f}s", flush=True)

    with _amp_ctx():
        # ── 1. start session ─────────────────────────────────────────────────
        t = time.time()
        sess = predictor.handle_request({
            "type": "start_session",
            "resource_path": str(frames_dir),
        })
        session_id = sess["session_id"]
        print(f"[sam3] session started (id={session_id[:8]}…) in {time.time()-t:.1f}s", flush=True)

        # ── 2. one prompt per session (SAM 3.1 requires reset between phrases) ─
        # Per the upstream SAM 3.1 video predictor notebook: "in case you
        # already ran one text prompt and now want to switch to another text
        # prompt it's required to reset the session first (otherwise the
        # results would be wrong)". So we propagate-per-prompt, reset, repeat.
        t = time.time()
        obj_id_to_label: dict[int, str] = {}
        per_frame_outputs: dict[int, dict] = {}  # frame_idx -> outputs (merged across prompts)
        next_global_oid = 1
        for i, phrase in enumerate(text_prompts):
            try:
                resp = predictor.handle_request({
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": 0,
                    "text": phrase,
                })
            except Exception as e:  # noqa: BLE001
                print(f"[sam3]   prompt {i+1}/{len(text_prompts)} '{phrase}' FAILED: {e}", flush=True)
                # reset session before continuing so the next prompt isn't polluted
                try:
                    predictor.handle_request({"type": "reset_session", "session_id": session_id})
                except Exception:  # noqa: BLE001
                    pass
                continue

            seed_outputs = resp.get("outputs") or {}
            seed_obj_ids = _to_numpy(seed_outputs.get("out_obj_ids")).astype(np.int64).tolist()
            if not seed_obj_ids:
                # nothing matched this phrase — skip propagation for it
                try:
                    predictor.handle_request({"type": "reset_session", "session_id": session_id})
                except Exception:  # noqa: BLE001
                    pass
                if (i + 1) % 5 == 0 or (i + 1) == len(text_prompts):
                    print(f"[sam3]   prompt {i+1}/{len(text_prompts)} ('{phrase}'): "
                          f"+0 instances, total objs={len(obj_id_to_label)} "
                          f"({time.time()-t:.1f}s)", flush=True)
                continue

            # remap per-prompt obj_ids → globally-unique ids so different
            # phrases don't collide in per_frame_outputs
            local_to_global = {}
            for local_oid in seed_obj_ids:
                gid = next_global_oid
                next_global_oid += 1
                local_to_global[int(local_oid)] = gid
                obj_id_to_label[gid] = phrase

            # propagate this single phrase across the timeline, both directions
            for prop_resp in predictor.handle_stream_request({
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": "both",
            }):
                fidx = int(prop_resp["frame_index"])
                outs = prop_resp["outputs"]
                f_obj_ids = _to_numpy(outs.get("out_obj_ids")).astype(np.int64).tolist()
                f_masks = _to_numpy(outs.get("out_binary_masks"))
                f_boxes = _to_numpy(outs.get("out_boxes_xywh"))
                f_scores = _to_numpy(outs.get("out_scores"))

                bucket = per_frame_outputs.setdefault(
                    fidx, {"out_obj_ids": [], "out_binary_masks": [],
                           "out_boxes_xywh": [], "out_scores": []}
                )
                for j, local_oid in enumerate(f_obj_ids):
                    gid = local_to_global.get(int(local_oid))
                    if gid is None:
                        continue
                    if j >= len(f_masks):
                        continue
                    bucket["out_obj_ids"].append(gid)
                    bucket["out_binary_masks"].append(f_masks[j])
                    if j < len(f_boxes):
                        bucket["out_boxes_xywh"].append(f_boxes[j])
                    if j < len(f_scores):
                        bucket["out_scores"].append(float(f_scores[j]))

            # reset between phrases per the upstream guidance
            try:
                predictor.handle_request({"type": "reset_session", "session_id": session_id})
            except Exception as e:  # noqa: BLE001
                logger.warning("reset_session failed after '%s': %s", phrase, e)

            if (i + 1) % 5 == 0 or (i + 1) == len(text_prompts):
                print(f"[sam3]   prompt {i+1}/{len(text_prompts)} ('{phrase}'): "
                      f"+{len(seed_obj_ids)} instances, total objs={len(obj_id_to_label)} "
                      f"({time.time()-t:.1f}s)", flush=True)

        n_objs = len(obj_id_to_label)
        print(f"[sam3] prompts + propagation done in {time.time()-t:.1f}s — "
              f"{n_objs} objects across {len(per_frame_outputs)} frames", flush=True)
        if n_objs == 0:
            print("[sam3] no objects detected by any text prompt — closing session and returning empty", flush=True)
            try:
                predictor.handle_request({"type": "close_session", "session_id": session_id})
            except Exception:  # noqa: BLE001
                pass
            (sam_dir / "tracks.json").write_text(json.dumps({
                "tracks": [], "n_input_frames": len(frame_paths), "n_tracks": 0,
            }, indent=2))
            return []

        # ── 3. close session (frees GPU memory) ──────────────────────────────
        try:
            predictor.handle_request({"type": "close_session", "session_id": session_id})
        except Exception as e:  # noqa: BLE001
            logger.warning("close_session failed: %s", e)

    # ── 5. assemble Track objects + write masks ──────────────────────────────
    t = time.time()
    tracks: dict[int, Track] = {}
    n_mask_writes = 0
    for fidx, outputs in per_frame_outputs.items():
        frame_path = frame_paths[fidx]
        frame_id = frame_path.stem

        out_obj_ids = _to_numpy(outputs.get("out_obj_ids")).astype(np.int64).tolist()
        out_masks = _to_numpy(outputs.get("out_binary_masks"))
        out_boxes = _to_numpy(outputs.get("out_boxes_xywh"))
        out_scores = _to_numpy(outputs.get("out_scores"))  # SAM 3.1 may emit this

        for i, oid in enumerate(out_obj_ids):
            if i >= len(out_masks):
                continue
            mask = np.asarray(out_masks[i], dtype=bool)
            if not mask.any():
                continue

            # bbox: prefer model-emitted xywh, fall back to mask bbox
            if i < len(out_boxes) and out_boxes.ndim == 2 and out_boxes.shape[1] == 4:
                x, y, w, h = out_boxes[i]
                bbox_2d = (int(x), int(y), int(x + w), int(y + h))
            else:
                bb = _bbox_of_mask(mask)
                if bb is None:
                    continue
                bbox_2d = bb

            score = float(out_scores[i]) if i < len(out_scores) else 1.0

            track_id = f"obj_{oid:04d}"
            mask_path = masks_dir / track_id / f"{frame_id}.png"
            _save_mask(mask, mask_path)
            n_mask_writes += 1

            track = tracks.setdefault(
                oid,
                Track(
                    track_id=track_id,
                    text_prompt=obj_id_to_label.get(oid),
                    source="text",
                ),
            )
            track.frames.append(
                TrackFrame(
                    frame_id=frame_id,
                    mask_path=str(mask_path.relative_to(out_dir)),
                    score=score,
                    bbox_2d=bbox_2d,
                )
            )

    print(f"[sam3] mask writes: {n_mask_writes} across {len(tracks)} raw tracks "
          f"({time.time()-t:.1f}s)", flush=True)

    # ── 6. filter tracks ─────────────────────────────────────────────────────
    final: list[Track] = []
    for tr in tracks.values():
        if len(tr.frames) < min_track_frames:
            continue
        if tr.frames:
            mean_score = sum(f.score for f in tr.frames) / len(tr.frames)
            if mean_score < min_track_score:
                continue
        tr.frames.sort(key=lambda f: f.frame_id)
        final.append(tr)
    final.sort(key=lambda t: t.track_id)
    print(f"[sam3] after filter (min_frames={min_track_frames}, "
          f"min_score={min_track_score}): {len(tracks)} → {len(final)} tracks", flush=True)

    # ── 7. write tracks.json ─────────────────────────────────────────────────
    tracks_path = sam_dir / "tracks.json"
    tracks_path.write_text(json.dumps({
        "tracks": [
            {
                "track_id": t.track_id,
                "source": t.source,
                "text_prompt": t.text_prompt,
                "frames": [
                    {
                        "frame_id": f.frame_id,
                        "mask_path": f.mask_path,
                        "score": f.score,
                        "bbox_2d": list(f.bbox_2d),
                    }
                    for f in t.frames
                ],
            }
            for t in final
        ],
        "n_input_frames": len(frame_paths),
        "n_tracks": len(final),
    }, indent=2))
    print(f"[sam3] wrote {tracks_path.relative_to(out_dir)} "
          f"({len(final)} tracks, {len(frame_paths)} frames)", flush=True)

    return final
