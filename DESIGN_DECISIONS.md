# spatiality_v2 — Design decisions, rejects, and tradeoffs

Companion to [`PIPELINE.md`](./PIPELINE.md).

That doc describes what the pipeline *is*. This one describes everything we tried, what broke, why we rejected it, and which fallbacks we kept around when the primary path is unavailable.

&nbsp;

---

&nbsp;

## 0. The high-level design philosophy

A few principles that ended up driving most of these decisions:

1. **Simple beats clever when the simple thing actually works.** When given a complex-precise vs simple-fast choice, we keep picking simpler. Many of the "rejected complex" entries below are not bad approaches in absolute terms; they were just outpaced by a simpler approach that hit the same accuracy at a fraction of the wall-clock or operational cost.

2. **Single forward over the whole sequence beats chunking.** Every chunked geometry approach we tried produced N disjoint mini-reconstructions overlapping at the world origin. We pay for a bigger GPU instead.

3. **Per-unit checkpointing, never per-stage.** Lane B's first version flushed at the end of the loop; one cancellation lost 24 labels. Now every track flushes the moment its annotation lands.

4. **Keep VLMs on the labelling and judgement steps, not the geometry steps.** They're remarkable at noun-phrase scoping and at noticing scene-level inconsistencies. They are not useful for 3D math, and they are expensive enough that you don't want to call them inside per-pixel loops.

5. **Class-conditional priors beat scene-relative ones.** A "chair" with a 6 m diagonal is wrong regardless of how big the room is.

&nbsp;

---

&nbsp;

## 1. Geometry backbone: FlashVGGT vs VGGT vs the alternatives

### What we ship

**FlashVGGT** (Dec 2025, `ZipW/FlashVGGT`) is the primary; **base VGGT-1B** (`facebook/VGGT-1B`) is the wrapped fallback.

See `inference/flashvggt.py::load_model`.

&nbsp;

### Why FlashVGGT

- **Single-forward scaling.** FlashVGGT's compressed-descriptor attention scales to 1 k+ frames in one shot on an A100-80GB. Base VGGT topped out for us around ~300 frames before VRAM became the binding constraint.

- **Faster.** ~10× speedup at scale on the same hardware (compressed KV cache plus downsampled descriptors).

- **Better fine detail on long sequences.** Base VGGT's attention diffuses over many frames; FlashVGGT preserves per-frame texture better.

- **Ships a `point_head` we can use.** `world_points` and `world_points_conf` come out of the same forward and let the lift skip its manual unprojection. This eliminates the convention-error class entirely (no more "did I forget to negate y?").

&nbsp;

### Why we kept base VGGT as a fallback

- **Short clips don't need the heavy machinery.** For a 30-frame test capture, base VGGT loads faster (~3 GB vs ~5 GB weights) and the output is indistinguishable.

- **Insurance against an upstream FlashVGGT breakage.** FlashVGGT's distribution is rough around the edges: the upstream `pyproject.toml` has broken `include` entries, no `__init__.py` files, and weights distributed as raw `.pt` files (no `safetensors` or `config.json`, so `PyTorchModelHubMixin.from_pretrained` 404s). We had to ship a corrected `pyproject.toml` from `patches/` and call `model.load_ckpt(...)` directly. If any of that drifts upstream, base VGGT keeps the pipeline alive.

&nbsp;

### Things we tried and dropped

| Tried | Why we dropped it |
|---|---|
| **Chunked VGGT inference** (windows of 64 frames) | Each chunk's first frame is pinned at the world origin, so naive concatenation produces N disjoint reconstructions overlapping at the origin. The result looks like a starburst of identical rooms. We chose "raise the GPU class" over "chunk". |
| **DUSt3R / MASt3R** | The pairwise plus global alignment workflow is much more code surface than a single-forward VGGT. Quality on indoor walkthroughs was on par; FlashVGGT won on engineering simplicity. |
| **COLMAP-style SfM** | Slow (10s of minutes per scene), brittle on textureless walls and motion blur, and the dense MVS pass on top is a separate engineering project. VGGT gives us per-pixel depth plus cameras in 4 minutes. |
| **VGGT preprocessing in "pad" mode with black plus bilinear resize** | Three training-distribution mismatches stacked: pad instead of crop, black pad instead of white, bilinear instead of bicubic. Result: the pose head produced ghost duplicates of real objects in 3D. Fix was to follow the canonical `mode="crop"` path verbatim (see `flashvggt.py::_load_and_preprocess_images`). |

&nbsp;

### Fallback chain in `load_model()`

```python
def load_model(prefer="flashvggt"):
    if prefer == "flashvggt":
        attempts = [_try_load_flashvggt, _try_load_vggt]
    else:
        attempts = [_try_load_vggt, _try_load_flashvggt]
    for fn in attempts:
        result = fn()
        if result is not None: return result
    raise RuntimeError("No geometry backbone available")
```

If FlashVGGT's import or weight fetch fails for *any* reason (network blip, breaking change upstream, gated weight without HF token), the pipeline transparently falls back to base VGGT. The only visible difference is `meta["backend"] = "vggt"` in the `manifest.json`.

&nbsp;

---

&nbsp;

## 2. Frame selection: blur filter

### What we ship

A Laplacian-variance blur filter that drops the bottom 20% of frames *before* they hit FlashVGGT (`inference/frame_select.py`).

&nbsp;

### Why this matters more than it looks like it should

Handheld iPhone captures of indoor walkthroughs have a lot of motion blur. Anything from a slow pan has 2 to 5 obviously blurry frames, and a single blurry frame mid-sequence is enough to crash the pose head's global feature bank into a wrong attractor.

The visible failure mode is **ghost duplicates**: the same physical object reconstructed twice in 3D, ~40 cm apart, at frames where the camera made a sudden orientation change with motion blur masking the transition.

The fix was ported verbatim from the original `spatiality` repo's `_frame_select.py`, which we had dropped during the v2 rewrite and only re-added after seeing the regression on `IMG_7531`.

&nbsp;

### Tradeoffs

- **Why Laplacian variance and not optical flow or PiQ?** Laplacian is ~1 ms per frame and accurate enough; we only need to find the *bottom* 20%, not score every frame to four decimals.

- **Why 20%?** Empirically fits a bell curve of handheld capture quality. Going to 30% started dropping frames the model could handle fine.

- **Why oversample by 1.30× upstream in ffmpeg?** So the blur drop lands the model at exactly its target frame count instead of 80% of it. See `backend/main.py::_EXTRACT_OVERSAMPLE`.

&nbsp;

---

&nbsp;

## 3. Detection: Grounding DINO vs SAM 3.1 vs the rest

### What we ship

**Grounding DINO base** (`IDEA-Research/grounding-dino-base`). Open-vocab object detection from a free-form text query. Ships per-frame bboxes labelled with the matching phrase.

See `segmentation/gdino.py`.

&nbsp;

### What we tried first: SAM 3.1

We initially built around **SAM 3.1**, which combines text-prompted detection with mask propagation through a video. The plan was:

1. Scout discovers per-slice noun phrases.
2. SAM 3.1 detects, masks, and propagates each phrase through its slice.
3. The resulting masks feed the lift directly.

This is what `_GLOBAL_SAFETY_NET`'s "scoping" comment in `scene_scout.py` was originally written against: `propagate_in_video` runs the full attention pipeline once per frame in a session, so per-slice scoping was the wallclock optimisation that made SAM 3.1 viable at all.

&nbsp;

### Why we dropped SAM 3.1 (and SAM 2 video) entirely

The hard truth: **the web UI never rendered masks**, and the lift only consumes per-pixel depth lookups inside a 2D bbox. Once that's true, mask propagation is paying a substantial wallclock cost to produce data nobody uses.

- SAM 3.1 video propagation: ~10 minutes per scene on A100-40GB for a 500-frame clip.

- Mask outputs were only consumed by a previous mask-pixel lift implementation that we then replaced with bbox-depth unprojection.

- Bbox-center depth gives ~5 to 10 cm centroid accuracy, well within the "geometrically coherent" target.

- The image build was simpler: no SAM 2 / SAM 3.1 git+ install, no tricky CUDA flags.

So we pulled `lane_e.py`, `lane_f.py`, and `sam3.py` (the deleted files in the working tree) and replaced the whole detection stage with: **GDINO once → IoU-link bboxes → lift bboxes**.

&nbsp;

### Why we kept a tiny SAM 2.1 surface

SAM 2.1-hiera-tiny *does* still load in `lift.py` to give us mask-grade pixel selection inside each bbox. It's an opportunistic per-frame call with an encoder cache (~50 ms encoder plus ~3 ms decoder per (track, frame) on A100). The reasons:

- For a chair frame or a houseplant, the 5×5 bbox-interior grid does pull in some leg or floor pixels that drag the centroid; SAM mask sampling tightens the lift noticeably for thin, articulated, or U-shaped objects.

- It also enables the multi-view consistency filter in `lift.py`. That filter projects each pixel's world point into other frames and checks it lands in *that frame's mask*. Without masks the test degrades and we skip it.

- Set `SPATIALITY_DISABLE_SAM=1` to force the bbox-interior grid fallback, e.g. for cost or debug runs.

&nbsp;

### Why GDINO over alternatives

| Alternative | Why not |
|---|---|
| **OWL-ViT v2** | Fine accuracy on COCO categories, weak on the mid-tail vocabulary scout actually generates ("Dyson cyclone vacuum", "table lamp with woven shade"). GDINO's text encoder handles long, descriptive phrases better. |
| **MM-Grounding-DINO** | Marginally better numbers but only available via OpenMMLab `mmdet`; not registered as an HF architecture, which is a Modal-image-build burden we didn't want. The migration path is in a TODO at `gdino.py:31-33` for when MM-GDINO lands as a registered HF arch. |
| **YOLO-World** | Closed taxonomy out of the box. The open-vocab variant exists but its text encoder is weaker than GDINO's at descriptive multi-word phrases. |
| **GLIP** | Predecessor to GDINO; strictly worse on grounded detection. |

&nbsp;

### Fallback vocabulary

`gdino.py::_FALLBACK_PROMPTS` is a hand-picked 8-item list (`chair`, `table`, `sofa`, `bed`, `desk`, `person`, `door`, `window`) used when no scout output is available AND the scout pass is bypassed.

Real runs always use the scout list. The fallback is for offline debug.

&nbsp;

---

&nbsp;

## 4. The scene scout: why discover, not enumerate

### What we ship

A 20-way parallel Gemini 2.5 Flash fan-out that discovers per-scene, per-slice noun phrases, scoped to the temporal slice they were seen in.

See `segmentation/scene_scout.py`.

&nbsp;

### What we replaced

A static 40-phrase indoor taxonomy ("chair", "desk", "monitor", …) that drove SAM 3.1's open-vocab head.

&nbsp;

### Why discovery wins

- **Coverage.** A static list misses the long tail (the Stitch plush toy, the green succulent in a terracotta pot, the gymnastics chalk bag). Scout enumerates whatever is actually in *this* scene.

- **Scoping.** Per-slice tagging means GDINO's downstream scope filter can drop detections that fire in irrelevant frames (the "couch" scout saw in slice 3 doesn't match the cushion in slice 17 even if GDINO would otherwise match it).

- **Cheap.** 20 Gemini Flash calls in parallel = ~10 to 20 s wallclock. Compared to the 4-min FlashVGGT step or the 5-min lift, this is rounding error.

&nbsp;

### Why a closed-class safety net on top

Open-vocab discovery is non-deterministic. `person` may be missed in any given slice if the human walks through quickly.

A 12-item safety net (`person`, `laptop`, `door`, `wardrobe`, `book`, `remote control`, `ceiling light`, …) gets unioned in with `frame_range=None` so these always propagate over the whole video. Each safety-net item adds one phrase to the multi-phrase GDINO query, which scales sub-linearly in the text encoder.

The list is intentionally *short and conservative*. It doesn't include "wall" or "floor" because those are scene labels we explicitly ban in Lane B's prompt.

&nbsp;

---

&nbsp;

## 5. Tracker: IoU + DINOv2 appearance vs SAM 2 video propagation

### What we ship

A SORT-style greedy linker (`gdino.py::_group_into_tracklets`) scoring `α · IoU + (1 - α) · cosine` (α = 0.6) where the cosine term comes from per-detection DINOv2-small embeddings.

Falls back to pure IoU when the encoder is unavailable.

&nbsp;

### What we replaced

SAM 2's mask-propagation, which gave us implicit cross-frame identity but at the cost of an entire video-state pass per phrase.

&nbsp;

### Why IoU + appearance is enough

- IoU links are correct in the typical case (slow pan, contiguous frames).

- DINOv2 cosine rescues tracklets across:
  - **fast camera motion** (no IoU overlap between consecutive frames),
  - **partial occlusion** (object briefly leaves view),
  - **gaps up to `gap_tolerance = 3`** (without appearance, the linker would re-cut).

- α = 0.6 keeps the linker geometry-led for the common case where IoU is reliable, while letting appearance disambiguate the hard cases.

&nbsp;

### Tradeoffs

- DINOv2-small is 384-dim and ~85 MB on disk; total embedding cost is ~5 ms per detection batched 64-wide on A100. For a 500-frame scene with 3 to 8 k detections that's +20 to 40 s on Stage 2.

- We chose DINOv2-small over OpenCLIP-ViT-B/32 because the indoor-furniture domain is closer to ImageNet-style natural images than to CLIP's web-scraped distribution. DINOv2's instance-level features cluster crops of the *same physical chair* tighter than CLIP does.

- Set `SPATIALITY_DISABLE_REID=1` to skip embeddings entirely. Linker drops to IoU-only: degraded but functional.

&nbsp;

### Cross-phrase NMS (a small but important step)

GDINO's multi-phrase query produces one detection per `(phrase, bbox)`. A single physical chair often gets detected under several scout synonyms ("chair", "office chair", "swivel chair") at almost the same bbox in the same frame.

Without cross-phrase NMS @ IoU ≥ 0.7 these fan out into parallel tracklets and downstream the dedupe has to clean them up. Doing it pre-linker is much cleaner.

&nbsp;

---

&nbsp;

## 6. The lift: bbox-depth unprojection with multi-view consistency

### What we ship

For each (track, frame): SAM-mask-sample (or 5×5 inset grid fallback) → confidence gate → finite-depth gate → unproject.

Then once per track: multi-view consistency → DBSCAN coherence → reprojection sanity → confidence-weighted PCA OBB.

See `segmentation/lift.py`.

&nbsp;

### The failure modes this design specifically protects against

| Failure mode | The fix |
|---|---|
| **Floor bleed** ("the laptop is placed near the bed"). 5×5 grid corners land on background pixels visible behind the object, pull their depth, and project to wherever the background sits in 3D. | (a) 25% inset on the bbox so the grid samples the inner 50%×50%; (b) SAM-mask sampling when SAM is available; (c) **multi-view consistency filter** as the actual catch-all. |
| **Curtain fusion**. Tracker bridged two physically distinct objects (curtains on opposite walls) into one tracklet. | **DBSCAN 3D-coherence filter** with eps=0.3 m; drops the track if the largest cluster < 70% of points. |
| **Wrong-side-of-wall centroid**. GMM front/back detection failed and the centroid landed behind a transparent or reflective surface. | **Reprojection inlier check**: the lifted centroid must project inside the source bbox in ≥ 50% of frames. |
| **Synonym-phrase tracker fragmentation**. The same physical chair lifted three times under three synonyms. | **3D OBB merge** in `merge_lifted_tracks` (AABB-IoU ≥ 0.5 OR centroid < 0.5 × min_diag with class-compatible labels). |
| **OBB stretched by outliers**. Even with SAM masks, a few seam pixels project to free space and PCA's eigenvalue is dragged metres along the major axis. | **Robust extents**: 5/95 weighted percentiles instead of min/max. |

&nbsp;

### Why multi-view consistency is the centrepiece

For each pixel's lifted world-point, we project it back into every *other* frame in the tracklet, look up that frame's (dilated by 2 px) SAM mask, and tally hits / total in-frustum frames. Keep the pixel iff:

- it's in-frustum in ≥ 3 other frames, AND
- ≥ 50% of those frames have the mask covering its projected location.

This is a standard MVS view-selection pattern (Schönberger et al., ECCV 2016, COLMAP).

It's a *sample-decision* rule, not a population CI. We considered Wilson-LB but the calibration choices weren't worth the lost interpretability for what is fundamentally a per-pixel operational decision.

The `min_other_frames = 3` floor was raised from 2 specifically to reject the n=2, k=1 ambiguous-evidence case (one frame agrees, one disagrees, no useful signal). We lose unanimous-on-2-frames evidence but those tracks are typically short and noisy anyway.

&nbsp;

### Why we don't use SOR (statistical outlier removal) or DBSCAN at the per-frame level

- Bbox-interior plus confidence gate plus multi-view filter is a much sharper outlier reject than SOR.

- DBSCAN at the per-frame level is overkill. We only run it once per track on the assembled point cloud, which is the natural granularity for "did the tracker drift between two physical objects".

&nbsp;

### The `world_points` shortcut

When VGGT's point head is present, we look up world-XYZ directly from the saved `world_points/` arrays instead of doing manual unprojection. This eliminates the convention-error class entirely: no risk of sign mistakes in the `R, t` math.

We still keep the manual path as a fallback for older inference runs that pre-date the point-head save (and as a sanity check, since both methods should agree to within numerical noise).

&nbsp;

---

&nbsp;

## 7. The labelling VLM: Gemini 2.5 Flash via PydanticAI

### What we ship

Gemini 2.5 Flash for both Lane B (per-track labels) and Lane C (whole-scene coherence), through PydanticAI for structured outputs.

Auth flows through the Pydantic AI Gateway in production (Modal `pydantic-gateway` Secret) and direct `GEMINI_API_KEY` in local dev.

&nbsp;

### Why Gemini Flash and not Anthropic / OpenAI

This is a deliberate, recorded decision (memory: "VLM choice — Gemini via PydanticAI"). The reasoning:

1. **Latency under fan-out.** Lane B fans out 16-way under `asyncio.gather`. Flash sustains that concurrency with sub-3-second per-call latency on a 9-image grid. We never hit a rate cap during normal runs.

2. **Cost.** Flash is roughly an order of magnitude cheaper per call than Sonnet or GPT-4o for what is fundamentally a "look at a grid and pick a label" task. The pipeline runs ~25 to 60 calls per scene (1 scout fan-out × 20, 1 Lane C, 1 per labelled track) so cost matters.

3. **Multi-image input.** Flash accepts arbitrary image lists in a single call without rendering them into one composite, which is convenient for the orbital plus anchor grid (we still pre-composite for prompt tidiness, but the path is open if we later want true multi-image).

4. **Structured output via PydanticAI.** Flash supports JSON mode with a Pydantic schema, so the boundary between model and Python is sharp: no regex parsing of free-form text.

5. **Gateway routing.** PydanticAI's gateway handles provider failover, retries, and observability for free. Switching to a different provider is one env var.

&nbsp;

### Why Flash and not Flash-Lite

We tried Flash-Lite for Lane B and saw a measurable drop in the alternative-noun quality and reasoning coherence. The cost difference wasn't material at our call volume.

Flash-Lite is exposed via `SPATIALITY_VLM_MODEL=gemini-2.5-flash-lite` for users who want it.

&nbsp;

### Things we tried and dropped

| Tried | Why we dropped it |
|---|---|
| **Anthropic Sonnet for labels** | More expensive, no measurable accuracy gain on the orbital plus anchor grid task. Where Sonnet shines (long-context reasoning, code) is irrelevant here. |
| **GPT-4o** | Same conclusion as Sonnet on accuracy; per-call latency under 16-way fan-out was inconsistent in our testing. |
| **Single anchor crop instead of three** | Single anchor often shows a partial or awkward view (half a bed framed as if it were a sofa). Three temporally-spread anchors give Gemini the angular diversity to spot tracker drift and disambiguate. |
| **Orbital views only (no anchors)** | Orbital point-cloud renders are stylised. Gemini guesses shape from blurry blobs and gets it wrong (we saw "stroller" returned for what was actually a metal gymnastics bar). The anchor crops are the photographic ground truth Gemini explicitly trusts in the prompt. |
| **Single Lane B call with no Lane C review** | Per-track labels get the per-track view right but miss cross-object inconsistencies ("a stroller in a closed bedroom"). One Gemini call seeing the whole top-down plus the JSON inventory catches these for ~15 s of wallclock. |

&nbsp;

---

&nbsp;

## 8. Lanes E and F: what we built and then deleted

### Lane E: ConceptGraphs-style scene relations

**What it was:** a third Gemini call per scene that ran SigLIP embeddings on each track's anchor crop, clustered them in feature space, and proposed `(subject, relation, object)` triples (`monitor on desk`, `chair under table`).

**Why we deleted it (2026-05-10):** we weren't using its output. The web UI never rendered the relation graph; Lane B's per-track labels and Lane C's optional `scene_relations` field cover the actual user-facing story. Lane E was paying ~30 s of wallclock plus a SigLIP encode pass per scene for data nobody consumed.

&nbsp;

### Lane F: SpatialLM layout (walls, doors, windows)

**What it was:** a separate model pass extracting room-layout primitives (wall planes, door rectangles, window rectangles) using SpatialLM-Llama-1B.

**Why we deleted it (2026-05-10):** the weights are gated and the upstream Modal image build was fragile. The layout output also wasn't consumed by the UI.

If room layout becomes a real product requirement, the cleanest revival path is Splatlands-style plane fitting on the existing point cloud rather than re-importing SpatialLM.

&nbsp;

### What's left in the code

`segmentation/run.py:11-12` documents the deletion. The defaults in `run.py:121` are now `lanes = ["b", "c"]`. Lane E and F are gone from the dispatch table entirely.

&nbsp;

---

&nbsp;

## 9. Postprocess: class-conditional priors vs scene-relative caps

### What we ship

A class-conditional 3D OBB-diagonal prior table (`postprocess.py::_CLASS_OBB_RANGES_M`) with real-world dimension ranges per object class:

```python
"chair":     (0.4, 1.5),     # metres, OBB diagonal
"bed":       (1.5, 3.5),
"laptop":    (0.2, 0.6),
"outlet":    (0.05, 0.3),
...
```

Anything wildly out of range is dropped. Unrecognised classes fall back to `0.85 × scene_diagonal`.

&nbsp;

### What we replaced

A single global `_MAX_OBB_DIAG_FRACTION = 0.6` cap relative to the scene diagonal. Replaced because it had an obvious failure mode in both directions:

- **False negatives in big rooms:** a chair in a 12 m × 12 m office passed the cap easily, even when the lift had drifted it to a 4 m AABB.

- **False positives in small rooms:** in a small bedroom, a normal-size bed could exceed `0.6 × scene_diag` and get dropped as "scene".

The class-conditional prior makes both decisions independent of scene size. Reference: nuScenes (Caesar et al. CVPR 2020), same pattern used for proposal regression and post-processing NMS.

&nbsp;

### Instance-aware dedupe

We also rewrote the dedupe step (`postprocess.py::_cluster_and_dedupe`):

- **Bucket by class first** so distinct classes (e.g. "chair" + "table") can never merge.

- **Within each class**, single-link cluster on `centroid_distance < 0.5 m OR AABB-IoU > 0.3` with a hard "disjoint AABBs" veto.

- The 0.5 m eps was tightened from 1.5 m once we no longer needed permissive distance to swallow cross-class label noise (the bucket-by-class step does that explicitly).

This was the fix for "three chairs around a table fuse into one". They're all class `chair`, all within ~0.8 m of each other, but their AABBs are disjoint.

&nbsp;

---

&nbsp;

## 10. Lane C: the whole-scene coherence pass

### Why we have it at all

Lane B labels each track in isolation. Single-track labels are right ~85% of the time but get wrong in *exactly* the cases a human reviewer would flag immediately:

- A "stroller" floating in mid-air next to a "ceiling fan". The label is plausible *for that bbox in isolation* but obviously wrong given the spatial neighbours.

- Two tracks describing the same physical object that survived Lane B's per-class dedupe (e.g. one as "office chair", the other as "swivel chair", both at the same centroid).

- A "wall" that slipped past the scene-label deny-list.

Lane C is one Gemini call that sees the whole scene (top-down render plus JSON inventory) and is allowed to relabel, drop, merge, or attach relations.

&nbsp;

### Why a programmatic guard on merges

Gemini occasionally emits cross-class merges despite the prompt explicitly banning them ("merge 'desk' and 'chest of drawers' because they look stacked").

We added `_apply_corrections` → `_labels_compatible` as a hard guard that rejects any merge whose drop label fails the class-equivalence check (last-noun match OR substring match like "chair" / "office chair").

This is engineering safety, not theoretical rigor. The prompt is the first line of defence, the guard is the second.

&nbsp;

### Idempotency

Lane C is **idempotent** on `annotations.c.json`. If the file exists from a prior run, we return it directly without re-invoking the VLM.

This matters because Lane C is the cheapest stage to re-run (~15 s) but it's also the one most often interrupted by users tweaking earlier params and re-running, and we don't want to pay the call cost every time.

&nbsp;

---

&nbsp;

## 11. Confidence calibration: VLM × track length × depth confidence

### What we ship

A simple linear corroboration function in Lane B:

```python
corroboration = 0.5 + 0.5 · (0.5 · min(1, n_frames / 30) + 0.5 · clip(mean_depth_conf, 0, 1))
calibrated_conf = clip(vlm_conf · corroboration, 0, 1)
```

The VLM's confidence remains primary; we only down-weight when corroboration is weak.

&nbsp;

### Why this and not a learned calibrator (Platt / isotonic / temperature scaling)

We have no held-out labelled data. Anything learned would be calibrated on an ad-hoc evaluation set that wouldn't generalise to the next user's bedroom.

The hand-coded corroboration function is interpretable, the failure modes are obvious from inspection, and it covers the two signals that empirically correlate with "this label is right":

- **Track length.** A 30+ frame track has cross-view evidence; a 5-frame track is one fast pan and a guess.

- **Depth confidence.** VGGT's `depth_conf` is well-calibrated on textured surfaces. Low mean depth-conf means the lift was working with sky, blur, or glare and the centroid isn't trustworthy.

&nbsp;

### Constants

- `n_frames / 30` ceiling: empirically, gains plateau around 30 frames of evidence.

- `0.5 + 0.5 · …` shape ensures `corroboration ∈ [0.5, 1.0]`. The VLM never gets *boosted* above its self-reported number, but it can be halved when corroboration is fully zero. A track Gemini scored 0.9 with no corroboration ends up at 0.45 (still survives the 0.30 floor); a track at 0.2 with full corroboration ends up at 0.2 (no spurious promotions).

&nbsp;

---

&nbsp;

## 12. Resumability: the per-unit checkpoint pattern

### What we ship

Six independent checkpoints, each at the granularity of a single user-visible unit of work:

| Checkpoint | Granularity | Recovers |
|---|---|---|
| `_forward_preds.pt` | 1 forward pass | the ~4 min A100 forward |
| `scout_prompts.json` | 1 scout fan-out | 20 Gemini calls |
| `tracks.json` | 1 GDINO sweep | the multi-phrase detection sweep |
| `_lifted_tracks_v2.pkl` | full lift output | every per-track lift plus multi-view filter |
| `annotations.b.raw.json` | per *track* | every Lane B Gemini call individually |
| `annotations.c.json` | 1 coherence pass | the whole-scene Gemini call |

&nbsp;

### Why per-track and not per-stage flushing in Lane B

This is recorded in memory ("Pipeline stages should checkpoint per-unit") because we got it wrong the first time. Lane B v1 did:

```python
results = await asyncio.gather(*coros)
write_to_disk(results)
```

A single retry being cancelled mid-way through `gather` lost 24 already-completed labels. Lane B v2 does:

```python
async def _flush(annotation):
    async with write_lock:
        done[annotation["id"]] = annotation
        raw_path.write_text(json.dumps(list(done.values()), indent=2))
```

The cost is ~16 disk writes per second under full fan-out (negligible). The benefit is that any cancellation, exception, or container restart loses at most the in-flight track for each of the 16 workers.

&nbsp;

### Why a `_v2` suffix on the lifted-tracks pickle

Schema migration safety. The pre-2026 lifted-tracks dataclass carried a `siglip_feat` field (Lane E's appearance vector). Old pickles loaded into the new dataclass would silently set `siglip_feat=None` and the downstream code might or might not handle that gracefully.

The `_v2` suffix makes it impossible to load a stale pickle into a new dataclass. We'd rather re-run the lift than mysteriously crash three steps later.

&nbsp;

---

&nbsp;

## 13. Things considered and explicitly deferred

For completeness, here's what we discussed and chose *not* to build:

| Idea | Why deferred |
|---|---|
| **LangSplat / GS-Splat** for the splat representation | The web viewer uses three.js Points on the dense PLY directly. Real Gaussian-splat rasterisation would give nicer renders, but the engineering surface (training a 3DGS, headless EGL renderer, custom shader) is large for a quality bump that doesn't change the labelling story. Documented in `docs/future_work_langsplat.md`. |
| **Re-running lift with SAM 3 masks** | Would tighten OBBs but only after re-introducing the SAM 3 install we just deleted. Marginal accuracy gain doesn't justify the ~10 min extra wallclock per scene. |
| **A per-scene class-prior learner** | Update the `_CLASS_OBB_RANGES_M` table from observed data over time. Premature without a database of labelled scenes; the static table is fine for now. |
| **Anchor frame ordering in the prompt** | The grid currently feeds orbital views first, then anchors, in temporal order. Shuffling or score-ordering the anchors made no measurable difference in Lane B accuracy in our testing. |
| **A real ANN search for tracklet linking** (Faiss) | The greedy O(N²)-per-frame linker is plenty fast at our detection counts. Faiss would matter at 100 k+ detections per scene, not 8 k. |

&nbsp;

---

&nbsp;

## 14. What this all adds up to

The accuracy story compared to the v1 pipeline (the original `spatiality` repo):

- **Geometry.** Ghost-duplicates eliminated by the blur filter, canonical-crop preprocessing, and single-forward-pass commitment.

- **Detection.** Higher recall on the long-tail vocabulary (scout discovery vs static taxonomy) and fewer parallel tracklets (cross-phrase NMS plus class-bucketed dedupe).

- **3D position.** Centroid accuracy went from "roughly the right room" to "5 to 10 cm of the actual object" via bbox-inset sampling, multi-view consistency, DBSCAN coherence, and the reprojection sanity check. This is the change that prompted the "the accuracy of the annotation to the location is amazing now" observation.

- **Labels.** Lane B's three-anchor plus detector-phrase prompt is far less prone to render-shape guessing; Lane C catches the inter-object inconsistencies a per-track call can't.

&nbsp;

The wallclock story:

- Pipeline end-to-end: **~8 to 14 minutes** for a 500-frame indoor capture, vs **~25 to 35 minutes** for the SAM-3.1-based predecessor.

- Most of the savings are from deleting SAM 2/3.1 video propagation. The rest is from per-slice scope filtering on GDINO and from caching scout output on resume.

&nbsp;

The operational story:

- Two Modal apps plus a local FastAPI orchestrator. No persistent job queue. No durable state outside the on-disk manifest and the Modal volumes. A failed run flips one JSON field and stops the frontend poll.

- Six checkpoints make any partial failure recoverable in seconds, not minutes.
