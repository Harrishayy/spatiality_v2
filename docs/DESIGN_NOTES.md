# Design notes (short)

The five decisions that materially shape the output. Each is one paragraph; the long-form rationale + every alternative I considered and rejected is in [`DESIGN_DECISIONS.md`](DESIGN_DECISIONS.md).

### 1. FlashVGGT over COLMAP / DUSt3R / vanilla VGGT

FlashVGGT (Dec 2025, compressed-descriptor attention) does dense per-pixel depth + per-frame intrinsics + extrinsics in a single forward pass on an A100-80GB. COLMAP fails on textureless interiors (blank walls, kitchen counters), DUSt3R is pair-based and degrades on long sequences with little parallax, and vanilla VGGT is ~10× slower at our 500-frame target. FlashVGGT runs the full sequence as one pass (no chunking — chunked solves pin each chunk's first frame at the world origin and produce N disjoint reconstructions overlapping at zero), so the world frame is globally consistent without any extra alignment step. VGGT-1B stays as a fallback for short clips.

### 2. Blur-filter the frames *before* the model sees them

A single blurry frame can push FlashVGGT's pose head off by `> 30° ΔR` because the chunked-attention feature bank caches descriptors across the whole sequence. Filtering noisy frames *after* the model is too late. Dropping the bottom 20 % by Laplacian variance up-front is the single highest-impact fix for handheld phone captures — costs almost nothing, prevents the most expensive failure mode. The orchestrator ffmpeg-oversamples by 1.30× so we land at the target frame count post-filter.

### 3. Scoped Gemini scout, not a fixed taxonomy

Open-vocab detection with Grounding DINO is great in principle and terrible in practice if you query "all the objects" — false-positive rate becomes the dominant failure. Closed-class detectors (COCO etc.) miss everything indoor-specific. The middle path: a VLM scout looks at temporal slices of the video, proposes only the noun phrases it actually sees, and GDINO fires those phrases only within their slice windows (+15-frame padding). Per-slice scoping plus cross-phrase NMS at IoU 0.7 lands the precision back where you want it without sacrificing open-vocab recall.

### 4. Per-track checkpoint flush + multi-view consistency for the 3D lift

Two operational decisions that together kill the failure modes I actually saw. Per-track flush: Lane B writes its annotation immediately after each Gemini response — an earlier version wrote at end-of-loop and lost 24 labels to a single cancellation. Multi-view consistency in the lift: each pixel's world point is reprojected into other frames and only kept if it lands inside the SAM 2.1 mask in ≥ 50 % of views. Same idea as COLMAP-style visibility but applied to mask-grade segmentation — fixes the "floor bleed" mode where unmasked floor pixels otherwise get assigned to whichever object is closest.

### 5. Gemini 2.5 Flash (via PydanticAI)

Lanes B and C are heavy on multi-image grids (3×3 anchor frames + orbital novel views per track). Gemini 2.5 Flash is roughly 5–10× cheaper than Claude Sonnet at our typical token mix, with structured-output reliability that's good enough via PydanticAI's `BaseModel` decoding. The pipeline can swap to Claude or to OpenAI by setting `SPATIALITY_VLM_MODEL` — `vlm.py` is the only file that knows the model id. The constraint on a portfolio piece is cost per scene; Flash makes a full run land at ~$0.05–$0.15 in Gemini fees instead of $0.50+.

---

Stage 4 (capture map — top-down density of above-floor surfaces) intentionally uses **no model**. Points + cameras already encode all the geometry the layer needs; floor extraction is robust statistics (mode of the lower percentile after a histogram pass). An earlier humanoid-traversability framing was dropped because handheld captures rarely contain enough floor pixels to support that inference honestly — the capture map is what every run produces meaningfully.

Full alternatives catalogue (SAM 3.1 / SAM 2 video / SpatialLM / OWL-ViT / Anthropic VLMs / Lanes E + F as scene graph + room layout, etc.) and the prior failure modes that informed each choice: [`../DESIGN_DECISIONS.md`](../DESIGN_DECISIONS.md).
