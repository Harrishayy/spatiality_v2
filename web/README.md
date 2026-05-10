# /web — Spatiality 3D mesh viewer

Next.js 16 (App Router) + React 18 + TypeScript strict + Three.js + `@mkkellogg/gaussian-splats-3d` + Tailwind.

Schema mirror lives at [`app/lib/types.ts`](./app/lib/types.ts) — keep in sync with `shared/shared/schemas/*.py`.

## What ships out of the box

- **SplatViewer** — drag/orbit/zoom, loads `splat.ply` via `gaussian-splats-3d`. When the splat is empty, falls back to a Three.js placeholder with annotation bboxes rendered as wireframes so the rest of the UI is exercisable. Dynamically imported with `ssr: false` since it owns a WebGL context.
- **AnnotationOverlay** — HTML billboard pins anchored to each annotation centroid; tap to select, double-tap to isolate (Module 04 Path A).
- **PipelineProgress** — auto-polls `manifest.json` every 2 s until `status === "ready"`, then loads splat + annotations.
- **ChatPanel** — messages list + input. Talks to `/api/agent/chat`.
- **WhereAmIButton** — frustum-filters annotations, POSTs to `/api/agent/locate`, shows the answer back in chat.
- **Object isolation** — Tap pin or sidebar row to select; double-tap (or sidebar ◉) to toggle isolation. Hidden cluster annotations dim to 30% opacity. Once a real splat with `cluster_gaussian_indices` is wired, the same toggle hides those Gaussians too — that's the only edit needed in `SplatViewer.tsx`.

## Routes

- `/` — landing page.
- `/demos` — gallery of available scenes.
- `/upload` — upload a clip and start the pipeline.
- `/scenes/[id]` — viewer for a given scene.

## Run locally

The web app expects the agent backend running at `http://localhost:8765`. Next rewrites `/api/*` and `/artifacts/*` to that origin.

```
pnpm install
pnpm dev      # → http://localhost:5173
```

To point at a different agent host, set `NEXT_PUBLIC_AGENT_URL`:

```
NEXT_PUBLIC_AGENT_URL=http://localhost:9000 pnpm dev
```

## File contracts (read; don't drift)

- `manifest.json` — schema in [`../shared/shared/schemas/manifest.py`](../shared/shared/schemas/manifest.py)
- `annotations.json` — schema in [`../shared/shared/schemas/annotations.py`](../shared/shared/schemas/annotations.py); rendered as billboards
- `splat.ply` — fetched URL passed to `gaussian-splats-3d`'s `addSplatScene`
