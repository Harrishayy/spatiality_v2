"use client";

import { DEFAULT_SETTINGS } from "@/components/SettingsPanel";

type Param = { name: string; value: string };

type Stage = {
  index: string;
  title: string;
  where: string;
  summary: string;
  params: Param[];
};

const STAGES: Stage[] = [
  {
    index: "1",
    title: "Capture",
    where: "laptop · ffmpeg",
    summary:
      "Decode the upload and resample to ~500 evenly-spaced frames. Oversample to absorb the blur drop in stage 2 before mirroring to a Modal volume.",
    params: [
      { name: "max_frames", value: String(DEFAULT_SETTINGS.max_frames) },
      { name: "target_long_side", value: `${DEFAULT_SETTINGS.target_long_side} px` },
      { name: "oversample", value: "1.30×" },
    ],
  },
  {
    index: "2",
    title: "Poses & geometry",
    where: "Modal · A100-80GB · FlashVGGT",
    summary:
      "Single forward pass recovers per-pixel depth, per-frame camera (K, R, t), and a world-point map. A six-stage funnel turns the depth maps into a coloured PLY.",
    params: [
      { name: "blur drop", value: "bottom 20%" },
      { name: "crop", value: "518 px (VGGT canonical)" },
      { name: "depth_conf floor", value: "0.15" },
      { name: "stride", value: "2 (4× fewer px)" },
      { name: "far-cap", value: "p95 × 1.5" },
      { name: "depth-gradient guard", value: "≤ 0.06" },
      { name: "target point count", value: "50 M" },
    ],
  },
  {
    index: "3",
    title: "Scene scout",
    where: "Modal · Gemini 2.5 Flash",
    summary:
      "Discover what is in this particular video before detection. Chops the timeline into temporal slices and asks the VLM for segmentable noun phrases per slice.",
    params: [
      { name: "slices", value: "~20" },
      { name: "frames per slice", value: "6" },
      { name: "parallel calls", value: "asyncio.gather(20)" },
      { name: "phrase cap", value: "40 + closed-class safety net" },
      { name: "scope padding", value: "±15 frames" },
    ],
  },
  {
    index: "4",
    title: "Detection",
    where: "Modal · Grounding DINO base",
    summary:
      "One multi-phrase open-vocab query over every frame, followed by scope filtering (per-slice) and cross-phrase NMS to collapse synonym duplicates.",
    params: [
      { name: "batch size", value: "8 frames" },
      { name: "query", value: "single dot-separated multi-phrase" },
      { name: "NMS IoU", value: "≥ 0.7" },
      { name: "label canon.", value: "longest-substring → scout phrase" },
    ],
  },
  {
    index: "5",
    title: "Re-ID embeddings",
    where: "Modal · DINOv2-small",
    summary:
      "Appearance fingerprint per detection so the linker survives fast pans, brief occlusions, and pose changes that pure IoU loses.",
    params: [
      { name: "bbox padding", value: "15%" },
      { name: "input size", value: "224 × 224" },
      { name: "embedding", value: "CLS · 384-d · L2-normalised" },
      { name: "disable flag", value: "SPATIALITY_DISABLE_REID=1" },
    ],
  },
  {
    index: "6",
    title: "IoU + appearance linker",
    where: "CPU · SORT-style greedy",
    summary:
      "Score = α · IoU(last bbox) + (1−α) · cos(appearance). Per-phrase pruning then keeps only the most plausible tracklets.",
    params: [
      { name: "α (IoU weight)", value: "0.6" },
      { name: "match threshold", value: "≥ 0.30" },
      { name: "gap tolerance", value: "3 frames" },
      { name: "min bbox side", value: "16 px" },
      { name: "min run length", value: "8 frames" },
      { name: "top-K per phrase", value: "6" },
    ],
  },
  {
    index: "7",
    title: "3D lift",
    where: "Modal · SAM 2.1-hiera-tiny",
    summary:
      "Each 2D tracklet becomes a single LiftedTrack with centroid and PCA OBB. Multi-view consistency, DBSCAN coherence, and reprojection sanity catch background bleed and tracker drift.",
    params: [
      { name: "frames per track", value: "≤ 16 (evenly spaced)" },
      { name: "mask px sampled", value: "1024 (or 5×5 inset grid)" },
      { name: "depth_conf gate", value: "0.50 (fallback 0.30)" },
      { name: "multi-view need", value: "in ≥ 3 frames · ≥ 50% mask cover" },
      { name: "DBSCAN", value: "eps 0.30 m · min_samples 5" },
      { name: "largest-cluster keep", value: "≥ 70%" },
      { name: "OBB extents", value: "weighted 5/95 percentiles" },
      { name: "reprojection sanity", value: "centroid in bbox ≥ 50% frames" },
    ],
  },
  {
    index: "8",
    title: "OBB merge",
    where: "CPU · single-link clustering",
    summary:
      "Collapse 3D duplicates from over-segmented or split tracks before the VLM ever sees them.",
    params: [
      { name: "AABB-IoU merge", value: "≥ 0.50" },
      { name: "centroid merge", value: "< 0.5 × min(diag_i, diag_j)" },
      { name: "class guard", value: "same last-noun / substring" },
    ],
  },
  {
    index: "9",
    title: "Lane B — per-track labels",
    where: "Modal CPU + Gemini 2.5 Flash",
    summary:
      "Render 6 orbital novel views of each lifted track plus 3 anchor RGB crops, send as one 3-column grid to the VLM for a structured label/alternatives/confidence/reasoning.",
    params: [
      { name: "concurrency", value: "16-way semaphore" },
      { name: "orbit radius", value: "1.6 × diag · 6 views" },
      { name: "anchors", value: "3 temporally-spread RGB frames" },
      { name: "scene-label deny", value: "wall/floor/room/scene/…" },
      { name: "confidence floor", value: "0.30" },
      { name: "class size sanity", value: "real-world OBB-diagonal priors" },
      { name: "calibration", value: "vlm × (track length × depth_conf)" },
      { name: "checkpoint", value: "per-track flush · annotations.b.raw.json" },
    ],
  },
  {
    index: "10",
    title: "Lane C — coherence pass",
    where: "Gemini 2.5 Flash · one call",
    summary:
      "One whole-scene VLM call sees a top-down render plus the Lane B inventory. Allowed to drop, merge, relabel, and propose spatial relations.",
    params: [
      { name: "input", value: "top-down render + JSON inventory" },
      { name: "merge guard", value: "class-equivalence required" },
      { name: "relations cap", value: "≤ 8 (on, under, contains, …)" },
      { name: "idempotency", value: "annotations.c.json short-circuits" },
    ],
  },
];

export function PipelineOverview() {
  return (
    <section className="lp-surface lp-surface--tight min-h-0 min-w-0 max-w-full flex-1 overflow-hidden">
      <header className="flex min-w-0 shrink-0 flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
        <div className="min-w-0">
          <h2 className="lp-surface-title">Pipeline overview</h2>
          <p className="lp-surface-sub break-words">
            Ten stages from video to labelled 3D scene. Hyperparameters below
            are the live defaults used for this run.
          </p>
        </div>
        <span className="lp-eyebrow-mono whitespace-nowrap">10 stages</span>
      </header>

      <ol className="grid min-h-0 min-w-0 flex-1 auto-rows-min grid-cols-1 content-start gap-3 overflow-y-auto pr-1 sm:grid-cols-2 xl:grid-cols-3">
        {STAGES.map((stage) => (
          <li
            key={stage.index}
            className="flex min-w-0 max-w-full flex-col gap-2 overflow-hidden rounded-[10px] border border-[rgba(255,235,220,0.08)] bg-[rgba(255,255,255,0.015)] p-3"
          >
            <div className="flex min-w-0 flex-col gap-0.5">
              <div className="flex min-w-0 items-baseline gap-2">
                <span className="lp-eyebrow-mono opacity-70">
                  {stage.index.padStart(2, "0")}
                </span>
                <span className="min-w-0 flex-1 break-words text-[13.5px] font-semibold tracking-[-0.01em] text-[var(--ink-100)]">
                  {stage.title}
                </span>
              </div>
              <span className="lp-eyebrow-mono block min-w-0 break-words text-[10.5px] opacity-80">
                {stage.where}
              </span>
            </div>

            <p className="break-words text-[11.5px] leading-[1.5] text-[var(--ink-500)]">
              {stage.summary}
            </p>

            <dl className="mt-0.5 grid min-w-0 grid-cols-2 gap-x-2 gap-y-1.5">
              {stage.params.map((p) => (
                <div
                  key={p.name}
                  className="flex min-w-0 flex-col gap-0.5 overflow-hidden rounded-[6px] bg-[rgba(255,255,255,0.02)] px-2 py-1"
                >
                  <dt className="min-w-0 truncate font-mono text-[9.5px] uppercase tracking-[0.06em] text-[var(--ink-500)]">
                    {p.name}
                  </dt>
                  <dd className="min-w-0 break-words font-mono text-[11px] leading-[1.35] text-[var(--ink-100)]">
                    {p.value}
                  </dd>
                </div>
              ))}
            </dl>
          </li>
        ))}
      </ol>
    </section>
  );
}
