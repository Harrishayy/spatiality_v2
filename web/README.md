# /web — Spatiality point cloud viewer

Next.js 16 (App Router) + React 18 + TypeScript strict + Three.js + Tailwind.

## What ships out of the box

- **PointCloudViewer** — drag/orbit/zoom Three.js viewer that streams `points.ply` (xyz + uchar rgb + optional confidence). Cycles between RGB / depth / confidence colorings. Dynamically imported with `ssr: false` since it owns a WebGL context.
- **AnnotationOverlay** — HTML billboard pins anchored to each annotation centroid; tap to select, double-tap to isolate.
- **PipelineProgress / PipelineOverview** — auto-poll `manifest.json` every 2 s until the splat stage completes, then render the cloud; segmentation continues in the background.
- **SidePanel** — Pipeline / Objects / Evidence drawers, plus a chat input wired to `/api/agent/chat`.
- **CaptureMapCard** — top-left overlay showing the Stage 4 capture map (top-down density of above-floor surfaces).

## Routes

- `/` — landing page.
- `/upload` — upload a clip and start the pipeline.
- `/scenes/[id]` — viewer for a given scene.

## Run locally

The web app expects the agent backend running at `http://localhost:8765`. Next rewrites `/api/*` and `/artifacts/*` to that origin.

```
pnpm install
pnpm dev      # → http://localhost:5173
```

To point at a different agent host, set `SPATIALITY_API_URL`:

```
SPATIALITY_API_URL=http://localhost:9000 pnpm dev
```

## File contracts (read; don't drift)

- `manifest.json` — schema in [`../shared/shared/schemas/manifest.py`](../shared/shared/schemas/manifest.py)
- `annotations.b.json` — Lane B (VLM-verified labels); schema in [`../shared/shared/schemas/annotations.py`](../shared/shared/schemas/annotations.py)
- `points.ply` — dense colour point cloud from the poses stage; parsed by `PointCloudViewer`
