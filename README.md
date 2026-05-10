# spatiality_v2

Phone video → labelled 3D scene. Built for Humanoid's Perception & Spatial AI internship submission.

## Documentation

- **[`PIPELINE.md`](./PIPELINE.md)** — end-to-end architecture: capture → FlashVGGT geometry → scene scout → Grounding DINO → 3D lift → Lane B/C VLM labelling.
- **[`DESIGN_DECISIONS.md`](./DESIGN_DECISIONS.md)** — every alternative we tried (SAM 3.1, SAM 2 video, DUSt3R, COLMAP, OWL-ViT, Anthropic / OpenAI VLMs, Lanes E/F, …) and why we landed where we did.

## Layout

```
backend/
  main.py                  FastAPI orchestrator (laptop, port 8765)
  src/spatiality/          inference + segmentation packages
modal_inference.py         Modal app — FlashVGGT geometry (A100-80GB)
modal_segmentation.py      Modal app — GDINO + lift + Lane B/C (A100-40GB)
web/                       Next.js frontend
docs/                      research notes
patches/                   upstream-fix carry (FlashVGGT pyproject)
```

## Running locally

The two GPU stages run on Modal; everything else runs on the laptop.

```bash
# backend (port 8765)
uvicorn backend.main:app --host 0.0.0.0 --port 8765 --reload

# web
cd web && pnpm dev
```

Then upload a video through the UI. See `PIPELINE.md` for what happens next.
