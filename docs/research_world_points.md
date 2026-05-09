# Research note: should we use VGGT's `world_points` directly?

**TL;DR — No.** VGGT's authors explicitly say depth-based unprojection is more
accurate than the point-map head. Our current pipeline already does the
recommended thing. Keep it.

## What `world_points` is

VGGT has four parallel prediction heads. We saw all of them in the
chunked-forward concat output:

```
[inference]   concat output keys: ['pose_enc', 'depth', 'depth_conf',
                                   'world_points', 'world_points_conf', 'images']
```

| Head        | Output                        | Shape                          |
|-------------|-------------------------------|--------------------------------|
| camera_head | `pose_enc` (FoV + quat + T)   | `[B, S, 9]`                    |
| depth_head  | `depth`, `depth_conf`         | `[B, S, H, W]`                 |
| **point_head**  | **`world_points`, `world_points_conf`** | `[B, S, H, W, 3]`              |
| track_head  | track / vis / conf (opt-in)   | `[B, S, N, 2]`                 |

The `point_head` is a separate, jointly-trained branch that *directly*
predicts a 3D world coordinate per pixel. It is NOT computed by
post-processing the depth head's output — it has its own DPT decoder fed by
the same aggregated tokens. So in principle it's an alternative path to a
3D point cloud that bypasses the camera head entirely.

## Why we'd want to use it

Naïvely, the point head looks attractive:

- Bypasses the camera head's `pose_enc → K, R, t` chain (which had multiple
  bugs we already fixed: missing `image_size_hw`, asymmetric center-pad,
  K-rescale not accounting for pad offset).
- Drops a class of math from `inference/run.py:points_from_results` and from
  `segmentation/lift.py` — instead of unprojecting per-mask pixels through
  K, R, t, we'd just index `world_points[ys, xs]`.
- Has its own confidence channel `world_points_conf`, separate from
  `depth_conf` — could be more reliable for "is this point trustworthy in
  3D" filtering than depth-only confidence.

## Why we shouldn't

Both checked against the upstream VGGT repo (`facebookresearch/vggt@main`):

**1. Author's inline comment in their reference code** — emphasis added:

```python
# Predict Point Maps
point_map, point_conf = model.point_head(aggregated_tokens_list, images, ps_idx)

# Construct 3D Points from Depth Maps and Cameras
# which usually leads to MORE ACCURATE 3D points than point map branch
point_map_by_unprojection = unproject_depth_map_to_point_map(
    depth_map.squeeze(0), extrinsic.squeeze(0), intrinsic.squeeze(0)
)
```

(`README.md`, "Detailed Usage" section.)

**2. Their demo viewer ships depth-based as the default**:

> You can set `--use_point_map` to use the point cloud from the point map
> branch, *instead of* the depth-based point cloud.

— `README.md`, Viser viewer section. The opt-in flag confirms depth-based is
the recommended path.

The plausible reason: depth + camera are tightly coupled during training
(the loss enforces `unproject(depth, K, R, t) ≈ ground_truth_points`) so
depth-derived points stay self-consistent with the camera head's pose. The
point head, by contrast, is a separate decoder that can drift slightly —
small per-pixel errors that accumulate when you stitch points from many
frames into one cloud.

## What we currently do (= the recommended path)

Inference:
1. `depth_head` → `depth, depth_conf`
2. `camera_head` → `pose_enc → K, R, t` (with the center-pad-aware K
   rescale we just fixed)
3. `points_from_results(depth, K, R, t, conf_threshold=0.05)` unprojects
   confidence-gated pixels into a single world cloud → `points.ply`

Segmentation `lift_track`:
1. For each mask pixel (ys, xs), index `depth[ys, xs]`, `conf[ys, xs]`
2. Unproject through that frame's K, R, t
3. SOR + DBSCAN + median centroid

Both paths use depth + K + R + t. Both match VGGT's recommendation.

## Where `world_points` IS still useful

Two niche uses worth bookmarking:

- **Sanity check on first run.** A 1-line diff between
  `unproject(depth, K, R, t)` and `world_points` is a free correctness
  signal: if they disagree by more than a few cm, our K rescale or center-pad
  is broken. Adding a single-frame consistency log on the first run after
  any preprocessor change would catch issues before they bake into a
  500-frame `points.ply`.

- **Fallback if pose decode fails.** `pose_encoding_to_extri_intri` choked
  with `image_size_hw=None` until we fixed it; if a future VGGT version
  changes that API again, falling back to `world_points` keeps the pipeline
  producing *some* cloud while we patch.

## Recommendation

Keep current depth-based unprojection. Optionally add a per-run
`assert_close(unproject(...), world_points, atol_m=0.10)` consistency
check during the first frame's processing in `run_inference` — it costs
nothing, surfaces preprocessor bugs cheaply, and keeps `world_points` as a
diagnostic without changing the production path.

## Sources

- `vggt/models/vggt.py` (lines 71–83): both heads, both outputs
- `README.md`: "Detailed Usage" snippet — recommends depth+camera over
  point map
- `README.md`: Viser viewer — `--use_point_map` is opt-in
