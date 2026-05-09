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
    """Try to import and load FlashVGGT; return (model, name) or None.

    Upstream publishes weights to ``ZipW/FlashVGGT`` as raw ``.pt`` files
    (``flashvggt.pt``, ``flashvggt_stream.pt``) — there is no safetensors /
    config.json pair, so ``PyTorchModelHubMixin.from_pretrained`` 404s. We
    instead pull the checkpoint via ``hf_hub_download`` and feed it to the
    repo's own ``model.load_ckpt(...)``. Constructor defaults match the
    upstream demo (``demo_o3d.py``: kv_downfactor=4, keyframe_every=200).
    """
    try:
        from flashvggt.models.flash_vggt import FlashVGGT  # type: ignore[attr-defined]
        from huggingface_hub import hf_hub_download

        ckpt_path = hf_hub_download(repo_id="ZipW/FlashVGGT", filename="flashvggt.pt")
        model = FlashVGGT(kv_downfactor=4, keyframe_every=200)
        model.load_ckpt(ckpt_path)
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
    target_size: int = 518,
) -> tuple[torch.Tensor, list[np.ndarray], list[tuple[int, int, int, int]]]:
    """Canonical VGGT/FlashVGGT preprocessing — `mode="crop"`, BICUBIC, no padding.

    This MUST match the reference implementation in
    `flashvggt.utils.load_fn.load_and_preprocess_images(mode="crop")`
    (which is verbatim VGGT's preprocessor) — otherwise the pose head reads
    a training-distribution-mismatched feature bank and produces noisy
    extrinsics. The previous version of this function was a custom
    reimplementation that:
      (a) used "pad" mode instead of the canonical default "crop",
      (b) padded with BLACK (value=0.0) when canon pads with WHITE (1.0),
      (c) used BILINEAR resize when canon uses BICUBIC.
    All three are training-distribution mismatches that polluted attention
    and made the pose head produce ghost-duplicates of real objects in 3D.

    Crop mode (verbatim from canon):
      new_w = 518
      new_h_resized = round(h_orig * (518 / w_orig) / 14) * 14   ← multiple of 14
      resize BICUBIC → (new_w, new_h_resized)
      if new_h_resized > 518: center-crop top/bottom → final 518×518

    Returns:
      - tensor (N, 3, final_h, 518) in [0, 1] float32  — model input
      - list of original RGB uint8 arrays  — for color sampling at original res
      - list of (start_y_orig, end_y_orig, w_orig, h_orig) per frame  — the
        vertical band of the original image that the cropped 518×final_h
        view corresponds to. Used downstream to (1) sample colors from the
        un-cropped middle of the original image, and (2) rescale K from
        518×final_h coords back to (w_orig, end_y_orig - start_y_orig)
        coords for the unprojection.

    Note: with all input frames at the same aspect ratio (typical for a
    single video source), `final_h` is identical across frames, so the
    returned tensor is rectangular not jagged.
    """
    from PIL import Image  # noqa: PLC0415

    tensors: list[torch.Tensor] = []
    originals: list[np.ndarray] = []
    crop_info: list[tuple[int, int, int, int]] = []

    for path in image_paths:
        with Image.open(path) as im:
            if im.mode == "RGBA":
                bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
                im = Image.alpha_composite(bg, im)
            im = im.convert("RGB")
            originals.append(np.asarray(im))

            w_orig, h_orig = im.size
            new_w = target_size
            # Height: maintain aspect ratio, then round to nearest multiple of
            # 14 (ViT patch size). This matches canon EXACTLY; do not change.
            new_h = round(h_orig * (new_w / w_orig) / 14) * 14
            im = im.resize((new_w, new_h), Image.Resampling.BICUBIC)
            arr = np.asarray(im, dtype=np.float32) / 255.0  # (new_h, new_w, 3)

            # Center-crop height to target_size if it overflows. NEVER pad —
            # if new_h < target_size (landscape inputs) we just keep the
            # smaller height (canon's behavior in crop mode for landscape).
            if new_h > target_size:
                start_y = (new_h - target_size) // 2
                arr = arr[start_y : start_y + target_size, :, :]
                final_h = target_size
            else:
                start_y = 0
                final_h = new_h

            # Map the cropped vertical band back to ORIGINAL pixel coords —
            # used downstream to sample original-resolution colors only from
            # this band (we drop ~12% top + ~12% bottom of the original).
            scale_y_to_orig = h_orig / new_h  # resize factor we just inverted
            start_y_orig = int(round(start_y * scale_y_to_orig))
            end_y_orig = int(round((start_y + final_h) * scale_y_to_orig))

            tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (3, final_h, new_w)
            tensors.append(tensor)
            crop_info.append((start_y_orig, end_y_orig, w_orig, h_orig))

    return torch.stack(tensors, dim=0), originals, crop_info


def _decode_pose_enc(
    pose_enc: torch.Tensor,
    image_size_hw: tuple[int, int] = (518, 518),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert VGGT pose encoding (Nx9 = quat[4] + t[3] + fov[2]) to K/R/t.

    `pose_encoding_to_extri_intri` allocates `intrinsics` as
    ``torch.zeros(pose_enc.shape[:2] + (3, 3))`` — so it expects pose_enc as
    `[B, S, 9]` (it interprets the first two axes as batch + sequence).
    We accept either `[N, 9]` or `[B, N, 9]` and normalise to `[1, N, 9]`
    on the way in, then squeeze the leading 1 back out so callers see
    consistent `[N, ...]` shapes.

    `image_size_hw` is REQUIRED by VGGT — it converts FoV (fov_h, fov_w) into
    pixel-space focal lengths. Defaults to the (518, 518) square pad that
    `_load_and_preprocess_images` produces. Caller in `run_inference` then
    rescales K to each frame's original H/W via depth-shape ratio.
    """
    try:
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        from flashvggt.utils.pose_enc import pose_encoding_to_extri_intri  # type: ignore[attr-defined]

    if pose_enc.dim() == 2:
        pose_enc_in = pose_enc.unsqueeze(0)   # [N, 9] -> [1, N, 9]
        squeeze_after = True
    else:
        pose_enc_in = pose_enc
        squeeze_after = False

    extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc_in, image_size_hw=image_size_hw)
    if squeeze_after:
        extrinsics = extrinsics.squeeze(0)
        intrinsics = intrinsics.squeeze(0)
    extrinsics = extrinsics.detach().cpu().numpy()   # (N, 3, 4) or (N, 4, 4)
    intrinsics = intrinsics.detach().cpu().numpy()   # (N, 3, 3)

    R = extrinsics[..., :3, :3]
    t = extrinsics[..., :3, 3]
    return intrinsics, R, t


def run_inference(
    image_paths: Sequence[Path],
    device: str | None = None,
    checkpoint_path: Path | None = None,
) -> tuple[list[FrameResult], dict]:
    """Run geometry inference over a list of frames in a SINGLE forward pass.

    We deliberately do NOT chunk. Chunking each VGGT/FlashVGGT forward over a
    sub-window produces poses anchored at that chunk's first frame, in
    chunk-local world coordinates — concatenating those windows naively gives
    you N disjoint reconstructions overlapping at the origin. FlashVGGT's
    compressed-descriptor attention scales to 1k+ frames in a single forward
    on A100-80GB, so chunking has no benefit and breaks geometry. If the
    sequence is too long for one forward, raise the GPU class — don't chunk.

    Args:
      image_paths: ordered list of frame image paths (e.g. 0001.png, 0002.png, ...).
      device: "cuda" / "cpu". Auto-detect when None.
      checkpoint_path: if given, the raw forward-pass tensors (depth,
        depth_conf, pose_enc) are saved here right after the forward
        completes — and reloaded on retry, skipping the GPU work entirely.
        Makes a downstream crash (pose decode, rescale, file I/O) NOT throw
        away ~4 min of A100 time. Caller deletes the file once final
        artefacts are written.

    Returns:
      (frame_results, meta) where meta includes backend_name, duration_s, n_frames.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    n_frames = len(image_paths)
    print(f"[inference] device={device} frames={n_frames} (single forward, no chunking)", flush=True)

    # Resume-from-checkpoint short-circuit.
    if checkpoint_path is not None and checkpoint_path.exists():
        print(f"[inference] resuming from forward-pass checkpoint: {checkpoint_path}", flush=True)
        t_ckpt = time.time()
        preds = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        backend_name = "vggt-checkpoint"
        images, originals, crop_info = _load_and_preprocess_images(image_paths)
        duration = 0.0
        print(f"[inference] checkpoint loaded in {time.time()-t_ckpt:.1f}s "
              f"(skipped model load + forward pass)", flush=True)
    else:
        t_load = time.time()
        print("[inference] loading model …", flush=True)
        model, backend_name = load_model()
        model = model.to(device).eval()
        print(f"[inference] model loaded: backend={backend_name} in {time.time()-t_load:.1f}s", flush=True)

        t_pre = time.time()
        print(f"[inference] preprocessing {n_frames} frames (resize→518, center square-pad) …", flush=True)
        images, originals, crop_info = _load_and_preprocess_images(image_paths)
        images = images.to(device)
        print(f"[inference] preprocessing done in {time.time()-t_pre:.1f}s "
              f"(tensor {tuple(images.shape)}, {images.element_size()*images.nelement()/1e6:.1f} MB)", flush=True)

        # Per-submodule forward hooks — fire as each top-level block of the
        # model finishes its forward, giving us a real-time progress signal
        # through the otherwise-opaque `model(images)` call. We also spin a
        # watchdog thread that prints elapsed wallclock + GPU memory every
        # 10s so the user sees liveness even when no submodule has fired.
        hook_handles = _attach_progress_hooks(model)
        stop_watchdog = _start_forward_watchdog(device, n_frames, every=10.0)

        t0 = time.time()
        try:
            with torch.inference_mode():
                print(f"[inference] running SINGLE forward on all {n_frames} frames", flush=True)
                preds = model(images.unsqueeze(0))
        finally:
            stop_watchdog()
            for h in hook_handles:
                h.remove()
        duration = time.time() - t0
        print(f"[inference] forward pass done in {duration:.1f}s ({duration/n_frames*1000:.0f}ms/frame)", flush=True)

        # Persist the raw preds NOW, before any downstream code can crash.
        if checkpoint_path is not None:
            t_save = time.time()
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            preds_cpu = {
                k: v.detach().cpu()
                for k, v in preds.items()
                if isinstance(v, torch.Tensor)
            }
            torch.save(preds_cpu, checkpoint_path)
            size_mb = checkpoint_path.stat().st_size / 1e6
            print(f"[inference] forward checkpoint saved → {checkpoint_path.name} "
                  f"({size_mb:.0f} MB, {time.time()-t_save:.1f}s) — downstream is now crash-safe", flush=True)

    # Both repos return tensors keyed by "depth", "depth_conf", "pose_enc".
    raw_keys = {k: tuple(v.shape) for k, v in preds.items() if isinstance(v, torch.Tensor)}
    print(f"[inference] preds tensor shapes: {raw_keys}", flush=True)

    # VGGT's actual output shapes (verified on this run):
    #   depth        (1, N, H, W, 1)   ← channel dim is TRAILING
    #   depth_conf   (1, N, H, W)      ← no channel dim
    #   pose_enc     (1, N, 9)
    # We want depth and depth_conf as (N, H, W).
    depth = _strip_to_3d(preds["depth"])           # (N, H, W)
    depth_conf = _strip_to_3d(preds["depth_conf"]) # (N, H, W)
    pose_enc = preds["pose_enc"].squeeze(0)        # (N, 9)
    print(f"[inference] post-squeeze: depth {depth.shape}, depth_conf "
          f"{depth_conf.shape}, pose_enc {tuple(pose_enc.shape)}", flush=True)

    # Decode K, R, t using the model's actual input resolution (final_h, final_w).
    # In crop mode, all frames share the same (final_h, final_w) — read it from
    # the input tensor we just preprocessed.
    final_h, final_w = images.shape[-2], images.shape[-1]
    K_all, R_all, t_all = _decode_pose_enc(pose_enc, image_size_hw=(final_h, final_w))

    # Crop mode: the model saw the un-padded vertical band of each original
    # image, namely original[start_y_orig:end_y_orig, :, :] resized to
    # (final_h, final_w). To unproject in the original-image frame we:
    #   1) Resize the model's depth/conf maps from (model_h, model_w) to the
    #      cropped band's pixel size (band_h, w_orig). Depth is always at the
    #      model's output resolution which equals (final_h, final_w) for
    #      VGGT/FlashVGGT, so the model_h/model_w branch handles any future
    #      mismatch (e.g. if the model output is downsampled from input).
    #   2) Scale K from (final_h, final_w) coords to (band_h, w_orig) coords.
    #      No pad offsets — crop mode never pads.
    #   3) Color is sampled from the cropped band of the original.
    results: list[FrameResult] = []
    for i, path in enumerate(image_paths):
        rgb = originals[i]
        H_orig, W_orig = rgb.shape[:2]
        start_y_orig, end_y_orig, _, _ = crop_info[i]
        band_h = end_y_orig - start_y_orig
        # The cropped band of the original — what the model actually attended to.
        rgb_band = rgb[start_y_orig:end_y_orig, :, :]

        depth_model = depth[i]
        conf_model = depth_conf[i]
        model_h, model_w = depth_model.shape[:2]

        # Resize depth from model resolution → band_h × W_orig.
        d = _resize_to(depth_model, (band_h, W_orig))
        c = _resize_to(conf_model, (band_h, W_orig))

        # Scale K from (model_h, model_w) coords to (band_h, W_orig) coords.
        # No pad offset to subtract — crop mode never pads.
        K = K_all[i].copy()
        sx_o = W_orig / model_w
        sy_o = band_h / model_h
        K[0, 0] *= sx_o; K[0, 2] *= sx_o
        K[1, 1] *= sy_o; K[1, 2] *= sy_o

        results.append(
            FrameResult(
                frame_id=path.stem,
                depth=d.astype(np.float32),
                depth_conf=c.astype(np.float32),
                K=K.astype(np.float32),
                R=R_all[i].astype(np.float32),
                t=t_all[i].astype(np.float32),
                image_rgb=rgb_band,  # only the band the model actually saw
            )
        )

    meta = {
        "backend": backend_name,
        "duration_s": duration,
        "n_frames": len(image_paths),
        "device": device,
    }
    return results, meta


def _attach_progress_hooks(model) -> list:
    """Attach per-submodule forward hooks that print as each top-level block fires.

    FlashVGGT's `forward` is one opaque call from the outside — we'd see
    silence between "running SINGLE forward" and "forward pass done" except
    for the convergence print that happens deep in the aggregator. By
    hooking each *named* top-level submodule (aggregator, depth_head,
    camera_head, point_head, …) we get a per-stage timestamp showing which
    block of the model is currently running.

    Returns a list of handles the caller must `.remove()` after the forward
    pass to avoid leaking hooks across runs.
    """
    handles = []
    t_start = time.time()
    last_t = [t_start]

    for name, module in model.named_children():
        def _make_hook(n=name):
            def _hook(_mod, _inp, _out):
                now = time.time()
                dt = now - last_t[0]
                last_t[0] = now
                gb = (
                    torch.cuda.max_memory_allocated() / 1e9
                    if torch.cuda.is_available()
                    else 0.0
                )
                print(
                    f"[inference]   ✓ {n}: stage done in {dt:.1f}s "
                    f"(elapsed {now - t_start:.1f}s, peak GPU {gb:.1f} GB)",
                    flush=True,
                )
            return _hook
        handles.append(module.register_forward_hook(_make_hook()))
    print(
        f"[inference] attached progress hooks to {len(handles)} submodules: "
        f"{[n for n, _ in model.named_children()]}",
        flush=True,
    )
    return handles


def _start_forward_watchdog(device: str, n_frames: int, every: float = 10.0):
    """Spin a daemon thread that prints liveness + GPU memory every `every` seconds.

    Returns a `stop()` callable the caller invokes from a `finally:` to
    cleanly tear down the thread once the forward returns. The watchdog
    only prints if no submodule hook has fired in the last `every` seconds,
    so we don't double-spam when hooks are firing fast.
    """
    import threading

    t_start = time.time()
    stop_event = threading.Event()

    def _loop():
        # First tick at `every` seconds in, not immediately.
        while not stop_event.wait(every):
            elapsed = time.time() - t_start
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / 1e9
                peak = torch.cuda.max_memory_allocated() / 1e9
                print(
                    f"[inference]   … forward in flight: {elapsed:.0f}s elapsed, "
                    f"GPU {alloc:.1f}/{peak:.1f} GB (alloc/peak), {n_frames} frames",
                    flush=True,
                )
            else:
                print(f"[inference]   … forward in flight: {elapsed:.0f}s elapsed", flush=True)

    t = threading.Thread(target=_loop, daemon=True, name="forward-watchdog")
    t.start()
    return stop_event.set


def _strip_to_3d(t: torch.Tensor) -> np.ndarray:
    """Reduce a VGGT depth-like tensor to (N, H, W) numpy.

    Handles both ``(B, N, H, W, 1)`` (depth, with trailing channel) and
    ``(B, N, H, W)`` (depth_conf, no channel) by removing only the singleton
    leading batch dim and any trailing singleton channel dim. Never touches
    the spatial dims.
    """
    arr = t.detach().cpu().numpy()
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 4 and arr.shape[1] == 1:
        # (N, 1, H, W) channel-first variant — squeeze that one channel
        arr = arr[:, 0]
    if arr.ndim != 3:
        raise RuntimeError(f"_strip_to_3d: cannot reduce shape {tuple(t.shape)} to (N, H, W)")
    return arr


def _resize_to(arr: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Bilinear resize a (H, W) float array to target_hw without bringing in cv2 explicitly."""
    import cv2  # noqa: PLC0415

    h, w = target_hw
    return cv2.resize(arr.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


def _smooth_depth(depth: np.ndarray, d: int = 5, sigma_s: float = 50.0, sigma_r: float = 0.05) -> np.ndarray:
    """Edge-preserving bilateral filter on a depth map.

    Cleans VGGT's per-pixel prediction noise inside continuous surfaces
    without smearing across depth discontinuities (which would create
    floater points at silhouette edges). Standard step in photogrammetry
    and surfel-rendering pipelines (cf. Splatt3R, MVSplat). Ported from
    the old `spatiality` repo's `_smooth_depth` — was missing entirely
    in the new pipeline. Set d<=0 to disable.
    """
    if d <= 0:
        return depth
    import cv2  # noqa: PLC0415

    # cv2.bilateralFilter requires float32 contiguous, finite values. NaN-clean
    # to 0 first; the gates downstream (depth > 0, isfinite) drop them anyway.
    arr = np.where(np.isfinite(depth), depth, 0.0).astype(np.float32, copy=False)
    return cv2.bilateralFilter(arr, d=d, sigmaColor=sigma_r, sigmaSpace=sigma_s)


def points_from_results(
    results: list[FrameResult],
    conf_min: float = 0.15,
    pixel_stride: int = 2,
    target_count: int | None = 50_000_000,
    depth_grad_max: float = 0.06,
    depth_far_pct: float = 95.0,
    depth_far_mult: float = 1.5,
    bilateral_d: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Confidence-gated unprojection across all frames, with the four
    quality filters the old `spatiality` repo had but the new repo dropped.

    The web viewer can only reasonably handle a few million points; full
    500-frame VGGT output at 1552×2064 is 1.6 *billion* points before any
    filtering. So we cascade six filters per frame:

    1. **Bilateral depth smoothing** (`bilateral_d`): edge-preserving
       smoothing of the depth map kills VGGT prediction noise inside flat
       surfaces. Ported from old `_smooth_depth`. Set 0 to disable.
    2. **Stride sampling** (`pixel_stride`): every Nth pixel; stride=4
       drops pixel count 16×.
    3. **Absolute confidence floor** (`conf_min`): drop pixels where
       VGGT depth_conf < this. Mirrors the old spatiality repo's
       `VGGT_DEPTH_CONF_MIN = 0.2`; we ship slightly looser (0.15) so
       textureless walls/floor survive. depth_conf uses `expp1`
       activation, so values are typically >>1 on confident pixels and
       drift toward 0 on sky/blur/dark.
    4. **Far-cap** (`depth_far_pct`, `depth_far_mult`): drop pixels with
       depth above (per-frame percentile_pct(depth) × mult). Catches the
       enormous low-confidence depths VGGT predicts for sky/distant
       background that would otherwise project as huge floaters behind
       everything. Ported from old `_frame_surfels`.
    5. **Depth-gradient silhouette guard** (`depth_grad_max`): drop pixels
       where |∇depth| / depth > GRAD_MAX. These are silhouette edges
       where VGGT depth is unreliable — they project as floaters between
       foreground and background. Ported from old `_frame_surfels`.
    6. **Global random subsample to `target_count`**: caps total points
       so points.ply stays browser-loadable. Set None to disable.

    Returns (points_xyz, colors_rgb_uint8, confidences) flattened across all frames.
    """
    pts_all: list[np.ndarray] = []
    col_all: list[np.ndarray] = []
    conf_all: list[np.ndarray] = []

    n_pre = 0
    n_after_stride = 0
    n_after_conf = 0
    n_after_far = 0
    n_after_grad = 0

    for r in results:
        h, w = r.depth.shape
        n_pre += h * w

        # 1. Bilateral smooth the depth map first — every downstream filter
        #    (stride, conf, far, grad) sees the cleaned signal.
        d_smooth = _smooth_depth(r.depth, d=bilateral_d)

        # Per-frame far-cap from the SMOOTHED depth.
        valid_mask = np.isfinite(d_smooth) & (d_smooth > 1e-5)
        valid_depths = d_smooth[valid_mask]
        if valid_depths.size > 0:
            far_cap = float(np.percentile(valid_depths, depth_far_pct)) * depth_far_mult
        else:
            far_cap = float("inf")

        # Pixel-space depth gradient (relative to local depth) — central
        # differences. Borders are NaN (no neighbors → drop them).
        d_pad = np.where(valid_mask, d_smooth, np.nan)
        grad_u = np.full_like(d_pad, np.nan)
        grad_v = np.full_like(d_pad, np.nan)
        grad_u[:, 1:-1] = (d_pad[:, 2:] - d_pad[:, :-2]) * 0.5
        grad_v[1:-1, :] = (d_pad[2:, :] - d_pad[:-2, :]) * 0.5
        rel_grad = np.sqrt(grad_u ** 2 + grad_v ** 2) / (d_pad + 1e-9)

        # 2. stride sampling — strict subgrid in (y, x).
        ys_grid, xs_grid = np.meshgrid(
            np.arange(0, h, pixel_stride, dtype=np.int64),
            np.arange(0, w, pixel_stride, dtype=np.int64),
            indexing="ij",
        )
        ys = ys_grid.reshape(-1)
        xs = xs_grid.reshape(-1)
        if not len(xs):
            continue
        n_after_stride += len(xs)

        conf_strided = r.depth_conf[ys, xs]
        ds = d_smooth[ys, xs]
        rel_grad_strided = rel_grad[ys, xs]

        # 3. absolute confidence floor — matches the old `spatiality` repo's
        #    VGGT_DEPTH_CONF_MIN (0.2 there). VGGT's depth_conf uses `expp1`
        #    activation so values are typically >>1; the floor exists to drop
        #    pixels where the model is genuinely uncertain (sky, dark, blur)
        #    rather than thinning a per-frame percentile that wipes legitimate
        #    low-texture surfaces. Set conf_min=0 to disable the gate.
        if conf_min > 0:
            keep_conf = (conf_strided >= conf_min) & valid_mask[ys, xs]
        else:
            keep_conf = valid_mask[ys, xs]
        n_after_conf += int(keep_conf.sum())

        # 4. far-cap.
        keep_far = ds <= far_cap
        # 5. depth-gradient silhouette guard. NaN comparisons → False, which
        #    naturally drops border pixels with bad neighbors.
        keep_grad = (rel_grad_strided <= depth_grad_max) & np.isfinite(rel_grad_strided)

        keep = keep_conf & keep_far & keep_grad
        n_after_far += int((keep_conf & keep_far).sum())
        n_after_grad += int(keep.sum())

        if not keep.any():
            continue
        ys, xs, ds, conf_strided = ys[keep], xs[keep], ds[keep], conf_strided[keep]

        # Pixel → camera coords.
        fx, fy = r.K[0, 0], r.K[1, 1]
        cx, cy = r.K[0, 2], r.K[1, 2]
        x_cam = (xs.astype(np.float32) - cx) * ds / fx
        y_cam = (ys.astype(np.float32) - cy) * ds / fy
        z_cam = ds.astype(np.float32)
        cam = np.stack([x_cam, y_cam, z_cam], axis=1)

        # Camera → world: world = R^T (cam - t)
        world = (r.R.T @ (cam - r.t).T).T

        pts_all.append(world.astype(np.float32))
        col_all.append(r.image_rgb[ys, xs])
        conf_all.append(conf_strided.astype(np.float32))

    if not pts_all:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8), np.empty((0,), np.float32)

    pts = np.concatenate(pts_all, axis=0)
    cols = np.concatenate(col_all, axis=0)
    confs = np.concatenate(conf_all, axis=0)

    print(f"[points] funnel: pre={n_pre:,} → stride={n_after_stride:,} "
          f"→ conf(>={conf_min:.2f})={n_after_conf:,} "
          f"→ far_cap={n_after_far:,} → grad_guard={n_after_grad:,}", flush=True)

    # 6. global random subsample to target_count.
    if target_count is not None and len(pts) > target_count:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(pts), size=target_count, replace=False)
        pts, cols, confs = pts[idx], cols[idx], confs[idx]
        print(f"[points] subsampled to {len(pts):,} points (target={target_count:,})", flush=True)

    return pts, cols, confs
