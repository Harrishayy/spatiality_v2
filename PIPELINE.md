# spatiality_v2: Pipeline

A walkthrough of how a phone video becomes a labelled 3D scene.

---

## TL;DR

```
phone video (.mp4 / .mov)
   │
   ▼
[ 1   ] capture       ─ ffmpeg even-cadence frame extract       (local CPU)
   │
   ▼
[ 2   ] poses         ─ FlashVGGT  → depth + camera + 3D points (Modal A100-80GB)
   │                    • blur filter     → depth, depth_conf
   │                    • single forward  → cameras.json, points.ply
   │                                      → world_points (per-pixel XYZ)
   ▼
[ 3   ] segmentation                                            (Modal A100-40GB)
   │
   │  [ 3.1 ] scene scout   ─ Gemini 2.5 Flash, 20 parallel calls
   │           • watches 6 frames per temporal slice
   │           • returns a per-slice noun-phrase list  (scout_prompts.json)
   │
   │  [ 3.2 ] detection     ─ Grounding DINO (open-vocab)
   │           • single multi-phrase query, all frames
   │           • scope filter → cross-phrase NMS
   │
   │  [ 3.3 ] re-ID         ─ DINOv2 appearance embeddings
   │  [ 3.4 ] linking       ─ IoU + appearance tracklet linking  (tracks.json)
   │
   │  [ 3.5 ] 3D lift       ─ SAM 2.1 mask + bbox-depth unprojection
   │           • multi-view consistency filter
   │           • DBSCAN coherence + reproj sanity
   │           • PCA OBB merge                       (_lifted_tracks_v2.pkl)
   │
   │  [ 3.6 ] Lane B labels ─ 6 orbital + 3 anchor crops → Gemini
   │           • 16-way concurrent, per-track flush
   │           • postprocess clean & dedupe         (annotations.b.json)
   │
   │  [ 3.7 ] Lane C review ─ top-down render + JSON inventory → Gemini
   │           • drops, merges, relabels, relations (annotations.c.json)
   ▼
[ 4   ] capture map   ─ top-down 2D footprint of captured surfaces (Modal CPU)
                        • density heatmap of above-floor points
                        • tight bbox via 5th-percentile threshold
                        • non-fatal: labels ship even if this fails
                                                    (capture_map.json/.png)
   │
   ▼
                   3D viewer in the web UI
```

The whole run takes roughly **8–14 minutes** for a 500-frame indoor capture.

Each stage produces files on disk, every stage is resumable from its checkpoint, and any failure flips `manifest.json` so the frontend stops polling.

&nbsp;

---

&nbsp;

## 1. Where the code lives

```
backend/
  main.py                      FastAPI orchestrator (laptop, port 8765)
  src/spatiality/
    inference/
      run.py                   Stage 2 entrypoint
      flashvggt.py             FlashVGGT / VGGT wrapper
      frame_select.py          blur filter
    segmentation/
      run.py                   Stage 3 orchestrator (also triggers Stage 4)
      scene_scout.py           VLM scene scout
      gdino.py                 Grounding DINO + IoU linker
      reid.py                  DINOv2 appearance embeddings
      mask.py                  SAM 2.1 mask predictor wrapper
      lift.py                  3D lift + multi-view filter + OBB merge
      lane_b.py                per-track VLM labelling
      lane_c.py                whole-scene coherence pass
      postprocess.py           class-conditional dedupe + size sanity
      render.py                point-cloud orbital / top-down rasteriser
      vlm.py                   Gemini wrapper (PydanticAI)
    nav/
      capture_map.py           Stage 4: top-down density map of captured scene

backend/modal/inference.py     Modal app: spatiality-inference
backend/modal/segmentation.py  Modal app: spatiality-segmentation
```

The laptop runs FastAPI; the GPU work runs in two Modal apps that get invoked via `modal.Function.from_name(...).remote(...)`.

Files round-trip through two named Modal Volumes (`spatiality-inputs`, `spatiality-outputs`).

&nbsp;

---

&nbsp;

## 2. Stage 1: Capture (laptop, ffmpeg)

**Goal:** turn an arbitrary-length video into ~500 evenly-spaced PNGs.

In `backend/main.py::_extract_frames`:

1. `ffprobe` reads the video duration.

2. We oversample by **1.30×** to compensate for the 20% blur drop in stage 2 (so we land on the target ~500 frames going *into* the pose head).

3. `ffmpeg -vf fps=...` writes `0001.png … NNNN.png` into `backend/data/inputs/<scene_id>/frames/`.

4. The whole input directory is mirrored to the `spatiality-inputs` Modal volume via `Volume.batch_upload(force=True)`.

The frame count default is `DEFAULT_FRAMES = 500` and is overridable per job.

&nbsp;

---

&nbsp;

## 3. Stage 2: Poses & geometry (Modal, A100-80GB)

**Goal:** for every frame, recover a depth map, a camera (intrinsics + extrinsics), and an optional per-pixel XYZ map.

Together these let us produce a coloured 3D point cloud (`points.ply`).

The model is **FlashVGGT** (Dec 2025, compressed-descriptor attention), with **base VGGT-1B** as fallback.

See `backend/src/spatiality/inference/flashvggt.py::load_model`.

&nbsp;

### 3.1  Blur pre-filter (`frame_select.py`)

Before the model sees anything, we drop the bottom 20% of frames by Laplacian variance, a cheap motion-blur proxy where blurry frames have small Laplacian responses.

This is the single highest-impact fix for handheld iPhone captures. A single blurry frame mid-sequence is enough to send the pose head's global feature bank into a wrong attractor and produce **ghost-duplicates** of objects in the resulting cloud.

See the comment block at `inference/run.py:200-218` for the exact failure case we caught.

&nbsp;

### 3.2  Canonical 518-px crop preprocessing

`flashvggt.py::_load_and_preprocess_images` resizes each frame so width = 518 and height is the nearest multiple of 14 (the ViT patch size), then center-crops the height to 518 if it overflows.

This is *verbatim* what FlashVGGT was trained on. Using the wrong resize mode (e.g. pad-with-black + bilinear) is a training-distribution mismatch that pollutes attention and was the second-largest source of ghost duplicates in earlier runs.

We track the cropped vertical band so the rest of the pipeline samples colours from exactly the part of the original image the model attended to.

&nbsp;

### 3.3  Single forward pass

```python
preds = model(images.unsqueeze(0))   # one shot, no chunking
```

We deliberately **never chunk**.

Chunked VGGT/FlashVGGT solves are chunk-local, each chunk's first frame is pinned at the world origin, so naively concatenating chunks gives N disjoint reconstructions overlapping at the origin.

FlashVGGT's compressed-descriptor attention scales to 1 k+ frames in a single forward on A100-80GB. If the sequence is too long, we'd raise the GPU class, not chunk.

To keep the user informed during this otherwise-opaque ~4-minute call, we attach per-submodule forward hooks that print as each top-level block fires (`aggregator`, `depth_head`, `camera_head`, `point_head`, …) and run a watchdog thread that prints elapsed wallclock + GPU memory every 10 s.

A **forward checkpoint** (`_forward_preds.pt`) is dropped to disk *immediately* after the GPU work, so any downstream crash (pose decode, K rescale, file I/O) doesn't throw away ~4 min of A100 time. It's deleted once the final artefacts land.

&nbsp;

### 3.4  What comes out of the model

```
preds["depth"]              (1, N, H, W, 1)    per-pixel scalar depth
preds["depth_conf"]         (1, N, H, W)       confidence in the depth
preds["pose_enc"]           (1, N, 9)          quaternion[4] + t[3] + fov[2]
preds["world_points"]       (1, N, H, W, 3)    per-pixel XYZ in world coords
preds["world_points_conf"]  (1, N, H, W)       confidence in those XYZ
```

`pose_enc` is decoded into per-frame `K, R, t` via VGGT's `pose_encoding_to_extri_intri`.

Because we rendered the model at 518-pixel cropped width, K is then **rescaled** back to original pixel coordinates so downstream lift math works in the source-frame frame.

&nbsp;

### 3.5  Building `points.ply`

`flashvggt.py::points_from_results` runs a six-stage funnel per frame:

| # | Filter | What it does |
|---|---|---|
| 1 | bilateral depth smooth | edge-preserving cleanup of per-pixel noise without smearing across silhouettes |
| 2 | stride sampling (default 2) | take every Nth pixel, 4× fewer points per frame |
| 3 | `conf_min` floor (default 0.15) | drop pixels with `depth_conf < 0.15` (sky, dark, blur) |
| 4 | far-cap (95th pct × 1.5) | drop the long sky/background tail that VGGT predicts as huge floaters |
| 5 | depth-gradient guard (≤ 0.06) | drop silhouette pixels where `\|∇depth\|/depth` is high |
| 6 | random subsample to `target_count` (default 50 M) | bound the on-disk PLY at ~800 MB so the web viewer stays loadable |

The result is a binary little-endian PLY with `xyz f32 + rgb u8 + confidence f32` per vertex, the exact schema the web `SplatViewer` parses.

`cameras.json` is written alongside; the per-frame depth + confidence maps land in `depth/` and `depth_conf/`; the per-frame world-point XYZ maps go to `world_points/` (saved at stride-2 to cut disk 4×).

The `manifest.json` `poses` stage entry is updated and the forward checkpoint is deleted.

&nbsp;

---

&nbsp;

## 4. Stage 3: Segmentation (Modal, A100-40GB)

This is the long stage. It turns the geometry from Stage 2 into a labelled object inventory.

Orchestrated by `backend/src/spatiality/segmentation/run.py`.

The whole stage is **resumable**:

- lifted tracks are pickled to `_lifted_tracks_v2.pkl` after the lift,
- Lane B per-track flushes to `annotations.b.raw.json`,
- Lane C is idempotent on `annotations.c.json`.

A failure mid-run can be retried and only redoes the unfinished slice.

&nbsp;

### 4.1  Stage 3.1: Scene scout (`scene_scout.py`)

**Goal:** instead of asking GDINO to look for a fixed 40-phrase taxonomy, *discover* what's in this particular video first.

**How:**

1. Chop the timeline into **~20 temporal slices** (≥ 8 frames each).

2. For each slice, sample 6 evenly-spaced frames and send them to **Gemini 2.5 Flash** with a strict prompt: list concrete, segmentable noun phrases, no regions ("kitchen") or materials ("wood") or abstractions ("lighting").

3. The 20 calls fan out in parallel via `asyncio.gather`, total wallclock ~10–20 s.

4. Each phrase is tagged with the slice range it was discovered in, so GDINO can later only *propagate* that phrase over those frames (+ 15-frame padding on each side).

5. A **closed-class safety net** (`person`, `laptop`, `door`, `wardrobe`, `table lamp`, `book`, `remote control`, `ceiling light`, …) is appended with `frame_range=None` so common indoor objects are always candidate phrases even if scout missed them in any one slice.

6. Output is capped at 40 scoped phrases plus the safety net and written to `scout_prompts.json` for cheap resume.

The structured response uses PydanticAI:

```python
class SceneInventory(BaseModel):
    phrases: list[str]
    reasoning: str
```

&nbsp;

### 4.2  Stage 3.2: Detection (`gdino.py`)

**Model:** `IDEA-Research/grounding-dino-base` via Hugging Face's `AutoModelForZeroShotObjectDetection`.

GDINO accepts a free-form text query and returns per-frame bounding boxes labelled with the phrase that matched.

We send a **single dot-separated multi-phrase query** containing every scout phrase (e.g. `"office chair. ceramic coffee mug. guitar amplifier. ..."`) and run it over every frame in batches of 8.

Several filters then run in sequence:

1. **Stage-1 presence filter**: drop any frame missing a camera, depth map, or depth-conf map (they were filtered out by the blur pre-filter).

2. **0-byte / unreadable PNG filter**: defence against a partial-disk write upstream.

3. **Label canonicalisation** (`_canonicalize_label`): map GDINO's raw `text_labels` (which may span tokens across phrases) back to the closest scout phrase by longest-substring match.

4. **Scope filter** (`_filter_by_scope`): drop detections whose absolute frame index falls outside the phrase's scout-assigned `frame_range`. This is the per-slice scoping payoff. Phrases stay confined to where the scout actually saw them.

5. **Cross-phrase NMS @ IoU ≥ 0.7** (`_apply_cross_phrase_nms`): one physical chair often gets detected under several scout synonyms ("chair", "office chair", "swivel chair") at almost the same bbox in the same frame; suppress all but the highest-scoring per frame.

What we have at this point is, for every frame, a list of bounding boxes; each box tagged with the scout phrase it matched and a confidence score.

But GDINO has no concept of "the same object across frames". The chair in frame 100 and the chair in frame 101 are two independent detections to it. We need to **link** these per-frame boxes into per-object **tracklets**. A tracklet is one physical instance followed across consecutive frames.

The next two sub-stages do exactly that.

&nbsp;

### 4.3  Stage 3.3: Re-ID embeddings (`reid.py`)

**Goal:** give every single detection a numerical "appearance fingerprint" so the linker (4.4) can tell whether two boxes in different frames are actually the same physical object.

**Why we need this at all.**

The naive way to link per-frame boxes into tracklets is **IoU only**: if box A in frame 100 and box B in frame 101 overlap a lot, assume they're the same object. This works *most* of the time, but it falls apart in three specific cases that are very common in handheld phone captures:

| Failure mode | Example | What IoU alone does |
|---|---|---|
| **Fast camera motion** | The user pans across a desk in 200 ms. | The chair's bbox in frame N has zero overlap with its bbox in frame N+1 (IoU = 0), so the linker decides they're two different chairs. |
| **Brief occlusion** | A person walks in front of the laptop for a few frames. | The laptop's bbox vanishes, then reappears slightly offset. IoU sees no continuity and starts a new tracklet. |
| **Pose change of the object** | The plush toy gets bumped and rotates. | Bbox shape changes, IoU drops, tracker re-cuts. |

In all three cases we end up with the **same physical object split into multiple short tracklets**. Downstream, that means the same chair gets lifted into 3D twice, sent to Gemini twice, and shows up in the final annotations as two confidence-0.6 entries instead of one confidence-0.9 entry.

The fix is to add a *second* signal that doesn't depend on geometric overlap: **does box A look like box B?** If two boxes in different frames depict the same chair, their image content should be similar even if they don't overlap.

**How re-ID works.**

For every bounding box GDINO produced, we:

1. Crop the source frame around the box, with **15% padding** on each side so the model gets some surrounding context (an object at the edge of its bbox is hard to recognise without a hint of what's around it).

2. Resize the crop to **224 × 224** (DINOv2's training resolution).

3. Run it through **DINOv2-small** (`facebook/dinov2-small`), a 22M-parameter vision transformer trained on ImageNet-scale images with self-supervision.

4. Take the **CLS token** from the last hidden layer: a single 384-dimensional vector that summarises the whole image patch.

5. **L2-normalise** the vector so cosine similarity becomes a clean dot product.

The output is, for every detection, a 384-dim unit vector. Two crops of the same physical chair will have vectors with cosine similarity ~0.85+. Two crops of different chairs will sit around 0.5–0.7. Two crops of completely different objects (a chair vs a houseplant) will be < 0.4.

**Why DINOv2-small specifically.**

- It's *self-supervised*: it learned features that distinguish *instances* (this specific chair) better than category-level classifiers like CLIP, which were trained to bucket "chair" together regardless of which chair.
- It's **small (~85 MB)** and **fast**: about 5 ms per detection batched 64-wide on A100. For a 500-frame scene with ~5,000 surviving detections, that's ~30 s of wallclock on top of the GDINO sweep.
- Indoor furniture is closer to ImageNet's natural-image distribution than to CLIP's web-scraped distribution, so DINOv2 outperforms CLIP on our specific task.

**Failure modes and fallback.**

If the encoder fails to load (network blip, OOM, disabled via `SPATIALITY_DISABLE_REID=1`), we silently skip the embedding pass entirely and the linker falls back to pure IoU. Degraded but functional, never a hard failure.

What we have now: every detection from 4.2 is decorated with a 384-dim appearance vector. On to the linker.

&nbsp;

### 4.4  Stage 3.4: IoU + appearance linking

**Goal:** chain per-frame detections into **tracklets** (one tracklet per physical object), using *both* geometric overlap (IoU) and appearance similarity (cosine of the DINOv2 embeddings from 4.3).

**The algorithm.**

A SORT-style greedy linker walks frames in time order. It maintains a list of "active" tracklets, tracklets that received a detection recently and might still be ongoing.

For each frame, for each new detection, the linker scores it against every active tracklet and picks the best match. The score is:

```
score = α · IoU(detection, last_box_in_tracklet)
      + (1 - α) · cosine(detection.embed, last_detection.embed)

with α = 0.6
```

In English: 60% of the score is "do the boxes overlap geometrically", 40% is "do they look like the same object". Both terms are bounded in [0, 1], so the combined score is also in [0, 1] and directly comparable against the threshold.

If `score ≥ 0.3`, the detection is **appended to the existing tracklet**. Otherwise, it **starts a new tracklet**.

**Why α = 0.6.**

- IoU is the more reliable signal in the *common* case (slow pan, contiguous frames). We don't want to dilute it.
- But we want appearance to have enough weight to *rescue* the failure cases from 4.3's table: if IoU drops to 0 because of a fast pan, a cosine of 0.85 still gives `0.6·0 + 0.4·0.85 = 0.34`, which clears the threshold and keeps the tracklet alive.

**Gap tolerance.**

A tracklet stays "active" for **3 frames** after its last detection. This means brief occlusions (a person walks past for 1–2 frames) don't kill the tracklet. When the object reappears, the linker can still match it back to the same tracklet ID.

After 3 frames with no match, the tracklet is **closed** and moved out of the active list. It can no longer absorb new detections, even if a similar-looking object appears later. This prevents the linker from incorrectly stitching together two different chairs that briefly looked similar.

**Graceful degradation.**

When a detection has no embedding (re-ID was disabled or failed for that crop), the score formula collapses to pure IoU. This is exactly the same behaviour we'd have without 4.3 at all. Re-ID is purely additive.

**Per-phrase pruning.**

After the linker has produced its raw tracklets, we apply three sanity filters per phrase:

- **Drop tracklets whose peak-confidence detection has min(side) < 16 px.** A 12×12 px detection is almost certainly noise on textured background, not a real object.

- **Drop tracklets shorter than `min_run_frames = 8`.** A real physical object typically appears in many consecutive frames as the camera moves around. An object that "exists" for only 5 frames is usually a flicker, a momentary mis-detection on a wall pattern, not a real instance.

- **Cap each phrase at the top-K = 6 longest tracklets.** Sometimes a single noisy phrase like "upholstered furniture" matches dozens of slightly-shifted bboxes across the video. Without a cap, that one phrase could monopolise the entire track budget. Keeping only the 6 longest tracklets per phrase keeps things balanced.

**What's written.**

`tracks.json` contains one entry per surviving tracklet:

```jsonc
{
  "track_id": "obj_0042",
  "text_prompt": "office chair",
  "source": "text",
  "frames": [
    { "frame_id": "0118", "score": 0.61, "bbox_2d": [340, 220, 612, 540] },
    { "frame_id": "0119", "score": 0.58, "bbox_2d": [342, 222, 614, 542] },
    ...
  ]
}
```

This is the input to the 3D lift in 4.5.

&nbsp;

### 4.5  Stage 3.5: 3D lift (`lift.py`)

**Goal:** turn each 2D tracklet (a sequence of bboxes) into a single 3D `LiftedTrack` with a centroid, an oriented bounding box, and provenance back to the source frames.

**For each track:**

1. **Frame stride / cap.** Sub-sample to at most `_MAX_FRAMES_PER_TRACK = 16` frames, evenly spaced. This caps the lift's O(N²) multi-view filter cost.

2. **Mask sampling per frame.**
   - If SAM 2.1-hiera-tiny loaded successfully, prompt it with the bbox and get a binary mask. Sample up to 1024 random pixels from inside the mask.
   - If SAM is unavailable, fall back to a 5×5 grid sampled inside the *inner* 50%×50% of the bbox (25% inset on each side). The inset is critical, the 5×5 corners on a raw GDINO bbox often land on background (the bed, wall, or floor visible behind the object) and pull their depth, which is exactly the failure mode that produced "the laptop is placed near the bed".

3. **Confidence gate (depth_conf > 0.5)** with adaptive 0.3 fallback if the strict gate strips the frame to fewer than 50 samples.

4. **Finite-depth gate.**

5. **GMM front-surface filter** (only for short tracks with < 3 posed frames). Fits a 2-component Gaussian Mixture; if BIC says the depth distribution is bimodal, keep only pixels whose posterior assigns them to the *front* mode. This catches the case where a near-camera object's mask still grabs background through holes.

6. **Unproject to world coordinates.** Prefer VGGT's `world_points` head when present (avoids any convention risk in our manual `R, t` math); otherwise compute via standard `world = R.T @ (cam - t)`. Combine `depth_conf × world_points_conf` for downstream weighting.

**After collecting per-frame world-points across the track:**

7. **Multi-view geometric consistency filter** (`_multiview_visibility_keep`).

   For each pixel's world point, project it into every *other* frame in the track and check whether it lands inside that frame's (dilated) SAM mask. Keep a pixel iff it's in-frustum in ≥ 3 other frames AND ≥ 50% of those frames have the mask covering its projected location.

   This is the actual fix for the floor-bleed failure mode, based on COLMAP's MVS view-selection logic (Schönberger et al., ECCV 2016).

8. **DBSCAN 3D-coherence filter** (`_largest_coherent_cluster`).

   Run DBSCAN with `eps=0.3 m, min_samples=5`. Drop the track if its largest cluster holds < 70% of points. Catches the catastrophic-drift failure mode where a tracker bridged two distinct physical objects (e.g. curtains on opposite walls fused into one tracklet).

9. **Confidence-weighted centroid + PCA OBB.**

   Per-axis weighted median for the centroid (so noisy boundary pixels don't pull it); weighted covariance + 5/95 percentiles for OBB extents (rejects ~5% outliers per side without rejecting legitimate object shape).

10. **Reprojection sanity check.**

    Project the lifted centroid back into every source frame and require it to land inside the source bbox in ≥ 50% of frames. If not, the lift latched onto background depth, drop.

**3D OBB merge.**

After all tracks are lifted, `merge_lifted_tracks` collapses duplicates via single-link clustering on:

- **AABB-IoU ≥ 0.5** (high-overlap → almost certainly the same physical object), OR
- **centroid distance < 0.5 × `min(diag_i, diag_j)`** AND labels are class-compatible (same last-noun or substring).

Survivor of each cluster is the highest-mean-confidence track; the cluster's OBB is re-fit on the union of all members' corners.

The lifted-tracks list is pickled to `_lifted_tracks_v2.pkl` immediately so Lanes B and C are crash-safe.

&nbsp;

### 4.6  Stage 3.6: Lane B, per-track VLM labels (`lane_b.py`)

**Goal:** turn each lifted track into an `Annotation { id, label, alternatives, confidence, centroid, bbox, color, ... }` that the web UI can render.

**Per track** (16-way concurrent under a `Semaphore`):

1. **Render 6 orbital novel views** of the splat focused on the track's OBB (`render.py::render_track_orbit`).

   The point cloud is filtered to the OBB AABB × 1.5 margin so the VLM isn't drowned in the rest of the room; cameras are placed on a sphere around the centroid at `radius = 1.6 × diag`. This is a pure-numpy z-buffered point rasteriser, no OpenGL, because the VLM doesn't need photo-realism.

2. **Pick 3 anchor RGB frames** spanning the track's lifetime (`_pick_anchor_frames`).

   The track is split temporally into 3 windows and the highest-evidence frame (`bbox_area × score`) is taken from each. This gives the VLM angular + temporal variety so it can spot drift, partial views, and pose changes.

3. **Composite into one 3-column grid** (orbital views + anchor crops, all resized to the same height).

4. **Send to Gemini 2.5 Flash** with this prompt structure:

   > Images 1–6: orbital point-cloud renders of one 3D region (use for shape/scale only).
   >
   > Images 7–9: 3 real photographs of that same region.
   >
   > Trust photographs over renders. Detector candidate: "<scout phrase>", verify or override.
   >
   > Identify the SINGLE physical object…
   >
   > Return label="unknown" with confidence ≤ 0.3 if [no whole-object view, region is wall/floor/scene, photographs disagree, ambiguous].

   Forbidden labels are explicitly listed: `room`, `wall`, `floor`, `ceiling`, `scene`, `area`, `space`, `background`, `interior`, `environment`.

   Structured output via PydanticAI:

   ```python
   class LabelOutput(BaseModel):
       label: str
       alternatives: list[str]
       confidence: float
       reasoning: str
   ```

5. **Calibrate the VLM's confidence** by combining it with track length and depth confidence:

   ```
   corroboration = 0.5 + 0.5 · (0.5 · min(1, n_frames/30) + 0.5 · clip(mean_depth_conf, 0, 1))
   final_conf    = clip(vlm_conf · corroboration, 0, 1)
   ```

   The VLM remains the primary signal; we only down-weight when corroboration is weak.

6. **Flush to `annotations.b.raw.json`** under an asyncio lock as soon as the call returns.

   This is the per-unit checkpointing that keeps an interrupted run from losing 24 labels.

**Postprocess.**

After all tracks are processed, `postprocess.cleanup_lane_b_annotations` runs:

- **Scene-label filter**: regex deny-list (`\broom\b`, `\bscene\b`, `\bworkspace\b`, `\bbackground\b`, …) drops anything Gemini emitted that's actually a region.

- **Confidence floor**: drop `< 0.30`.

- **Class-conditional 3D size sanity**: each label looks up its expected OBB-diagonal range from a real-world prior table (e.g. `"chair": (0.4, 1.5) m`, `"bed": (1.5, 3.5) m`). Anything wildly out of range gets dropped. A "chair" with a 6.5 m diagonal is almost certainly a mislabelled wall. Unrecognised classes fall back to `0.85 × scene_diagonal`.

- **Instance-aware dedupe**: bucket by class first (so distinct classes never merge), then within each class run single-link clustering on `centroid_distance < 0.5 m OR AABB-IoU > 0.3`, with a hard "disjoint AABBs" veto so stacked or neighbouring objects can't collapse.

The cleaned list is written to `annotations.b.json`; the raw flush stays on disk for debugging.

&nbsp;

### 4.7  Stage 3.7: Lane C, whole-scene coherence (`lane_c.py`)

Lane B labels each track in isolation. Lane C is **one Gemini call** that sees the whole scene and is allowed to:

- **Relabel** an annotation if its neighbours make it implausible ("a 'stroller' inside a closed bedroom" → "metal gymnastics bar").

- **Drop** an annotation that the whole-scene context reveals as background or tracker drift.

- **Merge** two annotations describing the same physical object (with a programmatic class-equivalence guard that rejects cross-class merges Gemini occasionally proposes despite the prompt).

- **Propose up to 8 high-confidence parent–child relations** (`on`, `under`, `contains`, `supports`, `next-to`, `behind`, `in-front-of`).

**Inputs to the call:**

- a top-down render of the point cloud framed to enclose every annotation centroid,
- a compact JSON inventory of `(id, label, centroid, extents, confidence)` for every Lane B annotation.

**Output** is the structured `LaneCCorrections` Pydantic model. Edits are applied in order: drops → merges → relabels → relations bound to surviving annotations.

The result is written to `annotations.c.json` (the file the frontend prefers, falling back to `annotations.b.json` if Lane C is disabled or failed).

The stage is **idempotent**: if `annotations.c.json` already exists, it's returned directly without re-invoking the VLM, so a downstream failure never re-pays the call cost.

&nbsp;

---

&nbsp;

## 5. Stage 4: Capture map (Modal CPU)

Stage 4 emits a top-down 2D map of the captured space — a density heatmap of above-floor surfaces showing what was observed and how much of the room was covered. It runs at the end of segmentation on CPU (numpy + Pillow), uses no models, and is non-fatal — if it raises, Lanes B/C labels still ship as the primary product.

This used to be a humanoid free-space / traversability layer. We dropped that framing because handheld captures rarely contain enough floor pixels to support "where can a robot stand?" — the previous algorithm was returning 0 m² traversable on every desk-centric scene. The capture map is the artefact every run can produce meaningfully.

**Inputs:** `points.ply`, `cameras.json`.

**What it does** (`backend/src/spatiality/nav/capture_map.py`):

1. **Recover scene up.** Each camera's image-y axis (`[0, -1, 0]` in camera space) is mapped through `Rᵀ` into world coordinates. Averaging across all cameras gives a stable gravity estimate — more robust than either "−y in world" (only true if frame 0 is level) or PCA of the cloud (gets confused by tall obstacles).
2. **Pick an in-plane basis.** Two orthonormal axes `(u, v)` perpendicular to `up` so the grid's rows/cols are aligned with the captured space's natural orientation.
3. **Estimate the floor.** Project every point onto `up`, then take the densest 5 cm band near the 2nd-percentile height. Single-point outliers under the floor don't move the estimate.
4. **Drop the floor itself.** Points within 5 cm of the floor are removed — they're either the floor (uninteresting for "what's in the room") or under-floor noise. What remains is everything *standing on* the floor.
5. **Rasterise above-floor density.** XY footprint at 5 cm cells. Per cell, count above-floor points; log-bin to uint8 because the raw count distribution is heavy-tailed (a high-coverage shelf can have 100× the points of a typical cell) and a linear ramp washes everything else into the background.
6. **Tighten the grid** via a 5th-percentile density threshold + 3-cell breathing-room margin. The percentile filter drops long-tail single-point cells at the periphery that would otherwise drag the displayed extent out by 30-40%.

**Outputs:**

- `capture_map.json` — `{cell_size_m, grid_shape, tight_extent_m, origin_uv_m, floor_height_world, up_axis_world, u_axis_world, v_axis_world, camera_center_uv_m, stats: {coverage_m2, coverage_cells, n_frames}, density_b64: base64(uint8 grid)}`. The basis vectors are preserved verbatim from the previous schema so the viewer's leveling code keeps working — gravity-aligned cloud, floor at Y=0.
- `capture_map.png` — top-down preview: amber density heatmap (sparse → bright as cells get denser). This is what the viewer's "Capture map" toggle surfaces.

**Why CPU and why now**: the points + camera poses already encode all the geometry; the only computer-visiony step (floor extraction) is robust statistics. Wall-clock is ~5–15 s on a 50 M-point cloud; cost is zero (no API calls). Wired into `segmentation.run` after Lane C so it never blocks labels' completion.

The frontend reads `manifest.artifacts.capture_map_png` / `capture_map_json`; absence means the stage didn't run (e.g. older scenes) and the viewer falls back to "Stage 4 hasn't run" copy in the Capture-map card.

&nbsp;

---

&nbsp;

## 6. Coordinate conventions

We standardise on **OpenCV camera convention** end-to-end:

- `+y` is down in the image and in camera space,
- `+z` points forward (the camera looks down `+z`),
- `R, t` are world → camera (so `world = Rᵀ · (cam − t)`).

The web `SplatViewer` knows this and negates `y/z` while parsing, do not pre-flip on the backend.

&nbsp;

---

&nbsp;

## 7. Manifest, status, and the FastAPI frontend contract

`backend/data/outputs/<scene_id>/manifest.json` is the single source of truth the Next.js client polls.

**Schema:**

```jsonc
{
  "scene_id": "abc123",
  "created_at": "2026-05-10T...",
  "status": "processing" | "ready" | "failed",
  "stages": {
    "capture":      { "status": "complete" },
    "poses":        { "status": "running" | "complete" | "failed", ... },
    "splat":        { "status": "complete" },        // points.ply doubles as the splat source
    "segmentation": { "status": "running" | "complete" | "failed", ... }
  },
  "artifacts": {
    "splat_ply":          "points.ply",
    "cameras_json":       "cameras.json",
    "annotations_b_json": "annotations.b.json",
    "annotations_c_json": "annotations.c.json",
    "annotations_json":   "annotations.c.json"  // frontend's canonical pointer
  },
  "stats": { "frame_count": 487, "object_count": 31, "splat_size_mb": 812.4 },
  "errors": [ /* if any stage failed */ ]
}
```

`backend/main.py::_recompute_stats` refreshes `splat_size_mb` / `frame_count` / `object_count` from disk after every Modal pull, so the frontend never sees a stale `splat_size_mb = 0` and refuses to mount the viewer.

&nbsp;

---

&nbsp;

## 8. On-disk artefact layout

```
backend/data/outputs/<scene_id>/
  manifest.json
  points.ply                         dense colour cloud (~800 MB at 50 M pts)
  cameras.json                       per-frame K, R, t
  frames/0001.png …                  the cropped band the model attended to
  depth/0001.npy …                   per-frame depth float32 (H, W)
  depth_conf/0001.npy …              per-frame depth confidence float32 (H, W)
  world_points/0001.npy …            optional, per-frame XYZ float16 (H/2, W/2, 3)
  world_points_conf/0001.npy …       optional, per-frame XYZ confidence float16
  scout_prompts.json                 cached scout output (resume)
  tracks.json                        post-link tracklets w/ per-frame bboxes
  _lifted_tracks_v2.pkl              lifted tracks checkpoint (resume)
  annotations.b.raw.json             Lane B per-track flushes (resume)
  annotations.b.json                 Lane B cleaned + deduped
  annotations.c.json                 Lane C coherence-reviewed (canonical)
```

Anything starting with `_` is a checkpoint that gets cleaned up on a successful end-to-end run.

&nbsp;

---

&nbsp;

## 9. Resumability summary

| Stage | Checkpoint | What resume skips |
|---|---|---|
| Poses | `_forward_preds.pt` | the whole forward pass (~4 min A100) |
| Scene scout | `scout_prompts.json` | 20 Gemini calls (~15 s) |
| GDINO | `tracks.json` | the multi-phrase sweep (~30 s) |
| Lift | `_lifted_tracks_v2.pkl` | every per-track lift + multi-view filter |
| Lane B | `annotations.b.raw.json` | per-track Gemini calls (16-way × ~3 s each) |
| Lane C | `annotations.c.json` | the whole-scene Gemini call (~15 s) |

Per-unit checkpoints (Lane B per track, lift per track) are crucial. Lane B's previous "write at end of loop" lost 24 labels when a single retry was cancelled.

&nbsp;

---

&nbsp;

## 10. Failure handling

Every Modal call is wrapped in a try/except in `backend/main.py::_run_pipeline`. On any exception:

```python
_bump_manifest(scene_id, "<stage>", "failed", top="failed", error=f"{type(exc).__name__}: {exc}")
```

The frontend's poll sees `status: "failed"`, stops spinning, and shows the error string. No silent hangs.

&nbsp;

---

&nbsp;

## 11. Running the pipeline

Two execution paths. Path A is the supported one; Path B exists so anyone with their own CUDA box can skip Modal.

### Path A — Modal (the path the rest of this doc describes)

The laptop runs the local services:

```bash
# backend (port 8765)
uvicorn backend.main:app --host 0.0.0.0 --port 8765 --reload

# web
cd web && pnpm dev
```

Then upload a video through the UI. The lifecycle is:

```
POST /api/uploads/local                  → returns { scene_id, ... }
POST /api/jobs { scene_id }              → kicks off _run_pipeline in a daemon thread
GET  /api/jobs/{scene_id}                ← polled by the frontend (manifest)
GET  /artifacts/scenes/<id>/<rel_path>   ← serves points.ply, frames, annotations
```

Direct GPU re-runs (skipping the upload path) are also possible:

```bash
modal run backend/modal/inference.py::main    --input-id <scene_id>
modal run backend/modal/segmentation.py::main --input-id <scene_id> [--lanes b,c]
```

Both `--local_entrypoint`s pull the outputs back into a fresh `backend/data/outputs/<scene_id>_<timestamp>/` so prior runs are never overwritten.

### Path B — Local CUDA GPU (no Modal) — ⚠️ experimental, untested

For someone with their own GPU who doesn't want a Modal account. **This path was authored on macOS and has not been smoke-tested on real CUDA hardware** — every dependency and env-var choice is inferred from the working Modal image builds in [`../backend/modal/`](../backend/modal/). If anything errors, treat those two files as the source of truth.

```bash
# one-time install (FlashVGGT applies the patched pyproject from patches/)
bash scripts/install_local_gpu.sh

# put your video at backend/data/inputs/<scene_id>/source.mp4
python scripts/run_local_gpu.py <scene_id>

# (optional) view in the web UI — the FastAPI server just serves files
uvicorn backend.main:app --host 0.0.0.0 --port 8765 --reload
cd web && pnpm dev  # http://localhost:3000/scenes/<scene_id>
```

The local-GPU runner sets `SPATIALITY_DATA_ROOT=backend/data/inputs` and `SPATIALITY_ARTEFACTS_ROOT=backend/data/outputs`, then calls `spatiality.inference.run` and `spatiality.segmentation.run` in-process — the same entry points Modal's `run_inference_one` / `run_segmentation_one` wrappers delegate to. The web viewer is identical either way: it just reads `backend/data/outputs/<scene_id>/`.

&nbsp;

---

&nbsp;

## 12. The model stack at a glance

| Stage | Model | Where | Why |
|---|---|---|---|
| Geometry | **FlashVGGT** (fallback: VGGT-1B) | A100-80GB | best per-pixel depth + camera in one forward, scales to 1k+ frames |
| Scene scout | **Gemini 2.5 Flash** via PydanticAI | per-call API | strong at structured noun-phrase enumeration from frames |
| Detection | **Grounding DINO base** (`IDEA-Research/grounding-dino-base`) | A100-40GB | open-vocab boxes over scout phrases in one query |
| Re-ID | **DINOv2-small** | A100-40GB | cheap appearance vector for the IoU+appearance linker |
| Lift mask | **SAM 2.1-hiera-tiny** | A100-40GB | mask-grade pixel selection for the 3D lift (graceful fallback to bbox-interior grid) |
| Labelling | **Gemini 2.5 Flash** | per-call API | structured `{label, alternatives, confidence, reasoning}` from 9-image grid |
| Coherence | **Gemini 2.5 Flash** | per-call API | one whole-scene call; relabels / drops / merges / relations |

For *why* each of these was chosen over the alternatives we tried (SAM 3.1, SAM 2 video propagation, Anthropic VLMs, SpatialLM, ConceptGraphs Lane E, …), see [`DESIGN_DECISIONS.md`](./DESIGN_DECISIONS.md).
