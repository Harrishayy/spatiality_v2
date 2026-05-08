# Future work: LangSplat-style 3D language field (Lane G)

This is a deferred 4th lane that turns the splat itself into a queryable language field. Users type plain English into a search bar, the backend cosine-matches the query against per-Gaussian language embeddings, and the viewer paints a 3D heatmap on the splat showing where the matching content lives.

We deferred it because the per-scene build cost is high relative to v2's ASAP timeline:

| Variant | Per-scene build cost (700-frame bedroom video) | Notes |
|---|---|---|
| Vanilla LangSplat (CVPR'24) | ~33 hours | Trains a per-scene autoencoder over CLIP — not viable |
| **Gen-LangSplat** ([arXiv 2510.22930](https://arxiv.org/abs/2510.22930), Oct 2025) | **~1.5–2 hours on A100, ~$2–4 Modal compute** | Pre-trained autoencoder on ScanNet → no per-scene training step |
| Object-level SigLIP (lite alternative) | ~5 seconds | Already saved by Stage 3 (`sigliP_feat` per track); covers "find the chair", misses parts and undetected concepts |
| Lazy SAM 3.1 at query time | 0 upfront, slow per-query | Run SAM 3.1 with text prompt over stored keyframes on demand |

If we revisit, **Gen-LangSplat is the path** because it eliminates the per-scene autoencoder.

## Pipeline sketch

1. **Reuse Stage 1 output**: `splat.ply` from FlashVGGT.
2. **Reuse Stage 2 output**: SAM 3.1 masks per keyframe (already produced).
3. **Per-pixel SigLIP features**: extract dense SigLIP features for each keyframe (or per-mask region — see "granularity" below).
4. **Compress via pre-trained autoencoder**: Gen-LangSplat ships a generalized encoder trained on ScanNet that compresses CLIP/SigLIP features to 16-D. No per-scene training needed.
5. **Distill into Gaussians**: 30K-iteration optimization that adds a 16-D channel to each Gaussian in `splat.ply` (~30 min on A100).
6. **Persist**: write `splat_lang.ply` with the extra channel; write `siglip_meta.json` (model id, latent dim, query-time encoder URL/path).
7. **Backend `/api/scenes/{id}/query` endpoint**:
   - Accepts `{ "text": "the red mug" }`
   - Embeds the text via the same SigLIP model used at build time
   - Compresses to 16-D using the same generalized encoder
   - Cosine-similarity against every Gaussian's stored 16-D vector
   - Returns either a heatmap PNG, an array of `{gaussian_index, score}`, or a thresholded mask of "active" Gaussians
8. **Frontend search bar**:
   - Renders a small `<input>` in the viewer header
   - On submit, calls `/api/scenes/{id}/query`, receives scores, and applies a per-Gaussian tint in `SplatViewer.tsx`
   - "Clear" returns to RGB rendering

## Granularity choice

Three options for how dense the language field should be:

| Granularity | What it captures | Build cost | Storage |
|---|---|---|---|
| Per-mask | Object-level + part-level via SAM 3.1 hierarchy | Low (one feature per mask) | Tens of KB |
| Per-pixel within masks | All visible regions — including parts, supports queries like "leg of chair" | Medium | Few MB |
| Dense per-pixel | Background + parts + everything | High (the LangSplat default) | Tens of MB |

Recommendation: start with **per-pixel within masks** — it's the LangSplat sweet spot for room scenes and is much cheaper than dense.

## Failure modes to handle

- **Out-of-distribution query**: SigLIP doesn't know specialized vocab (medical, brand names). Fall back to lazy SAM 3.1.
- **Multiple matches**: e.g. "chair" with 4 chairs in the room — return all 4 highlighted, let the user click to disambiguate.
- **Empty match**: cosine threshold below floor → message "no match"; do not paint random Gaussians.

## Why this is on the roadmap, not in v2

The trade-off is build time vs. capability. v2's three-lane story (B verified labels, E scene graph, F SpatialLM layout) is already a strong portfolio piece for the Humanoid internship submission. LangSplat is a banger demo addition but the ~2-hour-per-scene cost would slow iteration during the submission sprint.

Once v2 is shipped, this lane is a clean drop-in:
- Stage 1 already saves `splat.ply` we'd need
- Stage 2 already saves the SAM 3.1 masks we'd need
- Stage 3 already saves a per-track `sigliP_feat` (object-level), so the lite alternative could ship in a single afternoon if we want a stop-gap before tackling Gen-LangSplat
