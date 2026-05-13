# TODO — before submitting

Punch list of what's still open. Ranked by impact-per-effort.

## P0 — submission-blocking

- [ ] **Hero GIF/video** at top of `README.md`. There's a `<!-- TODO -->`
      comment placeholder. Record a 6–10 s loop of phone-video → labelled
      point cloud rotating in the viewer. Save at `docs/hero.gif` (≤ 5 MB)
      and uncomment the `![hero]` line. First thing a reviewer sees.

- [x] **Bake the demo bundle.** Run:
      `python scripts/build_demo_scene.py`
      No data is written to git. The script produces two artefacts under
      `dist/` (gitignored):
        - [x] `dist/demo_piece_r2/`       — flat directory mirroring the bucket
          root layout. Upload this to R2.
        - [x] `dist/demo_piece_full.zip`  — same payload as a single ≈ 1.3 GB
          zip for the offline-download path. (1,372 MB)

- [x] **Upload to Cloudflare R2.** Create a public bucket; sync
      `dist/demo_piece_r2/` to the bucket root:
        `rclone copy dist/demo_piece_r2/ r2:<bucket>/`
      (or `aws s3 sync` against R2's S3-compatible endpoint). Allow
      public read on the bucket and note the bucket's public URL
      (`https://<id>.r2.dev`). Configure both env vars per the section
      below before the next deploy.
      *Done: bucket `spatiality`, 2,844 / 2,844 keys at
      `demo_piece/`, public URL `https://pub-82e8a39203604293831950cd4f40d8ce.r2.dev`.
      `NEXT_PUBLIC_DEMO_CDN_URL` + `NEXT_PUBLIC_DEMO_ONLY` set for
      **Production** only — add them to **Preview** too if you want
      preview builds to render the demo.*

### Vercel environment variables (paste before deploying)

The hosted build needs two env vars. Set both for **Production**
and **Preview**:

| Key | Value | Effect |
|---|---|---|
| `NEXT_PUBLIC_DEMO_CDN_URL` | `https://<id>.r2.dev`  (no trailing slash) | `web/next.config.mjs` rewrites all `/api/jobs/demo_piece` + `/artifacts/scenes/demo_piece/*` to the R2 bucket. |
| `NEXT_PUBLIC_DEMO_ONLY`   | `1` | Disables the upload dropzone, swaps the landing CTA to "View demo scene", and links the upload page to a "Run it yourself ↗" GitHub card. |

How to set them:

```bash
# Via the Vercel CLI (recommended — easier to keep in sync)
vercel env add NEXT_PUBLIC_DEMO_CDN_URL production
# paste:  https://<id>.r2.dev
vercel env add NEXT_PUBLIC_DEMO_CDN_URL preview
# paste:  https://<id>.r2.dev
vercel env add NEXT_PUBLIC_DEMO_ONLY production
# paste:  1
vercel env add NEXT_PUBLIC_DEMO_ONLY preview
# paste:  1

# Or via the dashboard:
#   Vercel project → Settings → Environment Variables → Add
#   (do this once per key per env)

# Pull them down for local Vercel-style runs:
vercel env pull web/.env.local
```

To run the demo-mode UI locally for testing (without deploying):

```bash
cd web
NEXT_PUBLIC_DEMO_CDN_URL=https://<id>.r2.dev \
NEXT_PUBLIC_DEMO_ONLY=1 \
pnpm dev
# → http://localhost:3000/  shows "View demo scene" CTA
# → /upload                 shows the demo-only notice card
# → /scenes/demo_piece      streams artefacts from R2
```

Unset both vars in your local shell to get the full upload UI back.

- [ ] **Attach `dist/demo_piece_full.zip` to a GitHub Release** so
      reviewers can grab the full data for offline viewing. Paste the
      release URL into the README's "Download the full demo scene" line.

- [x] **Fill in `scripts/demo.sh`'s `SAMPLE_URL`**. *(Done: removed the
      placeholder and made `SAMPLE_URL` an optional override. Users drop
      their own `.mp4` at `backend/data/inputs/demo/source.mp4`; if
      `SAMPLE_URL` is set, the script fetches from there instead.)*

## P1 — high-impact bonuses

- [ ] **Hosted demo on Vercel.** Once the demo scene is baked (P0
      above), the R2 bucket is populated, and the two env vars are
      configured per the table above, deploy `web/` and paste the URL
      into the README's "Hosted demo" line. The deployed
      `/scenes/demo_piece` will stream from R2; `/upload` will show the
      demo-only notice card; the landing page CTA will read
      "View demo scene".

- [ ] **Loom walkthrough (≤ 3 min)** embedded in README under the hero GIF.
      Phone video → upload → pipeline overview → viewer with labels →
      free-space toggle. Don't narrate every detail.

## P2 — nice to have

- [ ] **One smoke test** for `inference.flashvggt._rescale_K()` at
      `tests/test_smoke.py`. Catches regressions on the K rescale that
      every downstream stage depends on. 5 min of work.

- [ ] **`scripts/eval.py`** that surfaces simple metrics on the demo
      scene (object count, mean OBB diagonal, pose RMS reprojection).
      Makes the "running experiments" line from the JD literal.

- [ ] **Smoke-test the local-GPU path on real CUDA.** Can't be done on
      a Mac. If you can grab a Colab A100, run
      `bash scripts/install_local_gpu.sh && python scripts/run_local_gpu.py <id>`
      and fix whatever breaks. The path is currently marked
      experimental/untested for a reason.

## Submission

- [ ] **Application form**: repo URL + CV + short design-choices note.
      `docs/DESIGN_NOTES.md` already serves as the design note; paste
      from it or link it.
