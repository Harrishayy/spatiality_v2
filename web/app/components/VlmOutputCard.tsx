// Specialised renderers for VLM evidence attributes attached to spans.
// `vlm_proposals` (proposer.frame), `vlm_labels` (labeler.batch), and
// `annotations` (segmentation.vlm_label) come back as JSON-encoded strings
// because attach_payload truncates + JSON.stringifies them server-side.

"use client";

import { useMemo, useState } from "react";

import { frameUrl } from "@/lib/api";
import type { TraceTreeNode } from "@/lib/types";

interface ProposalRow {
  phrase?: string;
  fallback?: string;
  bbox_norm?: number[];
  confidence?: number;
  frame?: string;
}

interface LabelRow {
  label?: string;
  confidence?: number;
  alternatives?: string[];
  provenance?: string[];
  id?: string;
}

function safeParse<T>(v: unknown): T | null {
  if (v == null) return null;
  if (typeof v === "object") return v as T;
  if (typeof v === "string") {
    try {
      return JSON.parse(v) as T;
    } catch {
      return null;
    }
  }
  return null;
}

function pct(v: number | undefined): string {
  if (v == null || !Number.isFinite(v)) return "—";
  return `${Math.round(v * 100)}%`;
}

export function VlmOutputCard({
  node,
  sceneId,
}: {
  node: TraceTreeNode;
  sceneId?: string;
}) {
  const a = node.attributes;
  const proposals = safeParse<ProposalRow[]>(a.vlm_proposals);
  const labels = safeParse<Record<string, LabelRow>>(a.vlm_labels);
  const annotations = safeParse<LabelRow[]>(a.annotations);
  const raw = typeof a.vlm_response_raw === "string" ? a.vlm_response_raw : null;

  // The proposer attaches `frame_name` directly on the span; the proposer
  // summary span has per-proposal frames embedded in the rows. Use the span
  // attr first, then fall back to whatever proposals carry.
  const spanFrame =
    typeof a.frame_name === "string" ? a.frame_name : undefined;
  const frameForProposals =
    spanFrame ??
    proposals?.find((p) => typeof p.frame === "string" && p.frame.length > 0)?.frame;

  return (
    <div className="space-y-3">
      {proposals && proposals.length > 0 && sceneId && frameForProposals && (
        <Section title={`Keyframe · ${frameForProposals}`}>
          <KeyframeWithBboxes
            sceneId={sceneId}
            frameName={frameForProposals}
            proposals={proposals.filter(
              (p) => !p.frame || p.frame === frameForProposals,
            )}
          />
        </Section>
      )}

      {proposals && proposals.length > 0 && (
        <Section title={`Proposals (${proposals.length})`}>
          <div className="space-y-1">
            {proposals.map((p, i) => (
              <div
                key={i}
                className="grid grid-cols-[1fr_auto] items-baseline gap-3 rounded border border-ink-800 bg-ink-900/40 px-3 py-2"
              >
                <div className="min-w-0">
                  <div className="truncate text-sm text-ink-100">{p.phrase}</div>
                  <div className="truncate font-mono text-[10px] text-ink-500">
                    {p.fallback ?? ""}
                    {p.bbox_norm
                      ? ` · bbox [${p.bbox_norm.map((n) => n.toFixed(2)).join(", ")}]`
                      : ""}
                    {p.frame && p.frame !== frameForProposals
                      ? ` · ${p.frame}`
                      : ""}
                  </div>
                </div>
                <span className="font-mono text-[11px] tabular-nums text-accent-300">
                  {pct(p.confidence)}
                </span>
              </div>
            ))}
          </div>
        </Section>
      )}

      {labels && Object.keys(labels).length > 0 && (
        <Section title={`Cluster labels (${Object.keys(labels).length})`}>
          <div className="space-y-1">
            {Object.entries(labels).map(([cid, l]) => {
              const dup = (l.alternatives ?? []).find((x) =>
                String(x).startsWith("duplicate_of:"),
              );
              const rejected = (l.label ?? "").toLowerCase() === "none";
              return (
                <div
                  key={cid}
                  className={[
                    "grid grid-cols-[auto_1fr_auto] items-baseline gap-3 rounded border px-3 py-2",
                    rejected
                      ? "border-ink-800 bg-ink-950/40 text-ink-500"
                      : dup
                        ? "border-amber-500/40 bg-amber-500/5"
                        : "border-ink-800 bg-ink-900/40",
                  ].join(" ")}
                >
                  <span className="font-mono text-[11px] text-ink-500">{cid}</span>
                  <div className="min-w-0">
                    <div className="truncate text-sm">
                      {l.label || "—"}
                      {dup && (
                        <span className="ml-2 rounded bg-amber-500/20 px-1.5 py-0.5 font-mono text-[10px] text-amber-300">
                          {dup}
                        </span>
                      )}
                      {rejected && (
                        <span className="ml-2 rounded bg-ink-800 px-1.5 py-0.5 font-mono text-[10px] text-ink-500">
                          rejected
                        </span>
                      )}
                    </div>
                    {(l.alternatives ?? []).filter((x) => !String(x).startsWith("duplicate_of:")).length >
                      0 && (
                      <div className="font-mono text-[10px] text-ink-500">
                        alts:{" "}
                        {(l.alternatives ?? [])
                          .filter((x) => !String(x).startsWith("duplicate_of:"))
                          .join(", ")}
                      </div>
                    )}
                  </div>
                  <span className="font-mono text-[11px] tabular-nums text-accent-300">
                    {pct(l.confidence)}
                  </span>
                </div>
              );
            })}
          </div>
        </Section>
      )}

      {annotations && annotations.length > 0 && (
        <Section title={`Final annotations (${annotations.length})`}>
          <div className="space-y-1">
            {annotations.map((a, i) => (
              <div
                key={a.id ?? i}
                className="grid grid-cols-[auto_1fr_auto] items-baseline gap-3 rounded border border-emerald-500/30 bg-emerald-500/5 px-3 py-2"
              >
                <span className="font-mono text-[11px] text-ink-500">{a.id ?? `#${i}`}</span>
                <div className="min-w-0 truncate text-sm text-ink-100">{a.label}</div>
                <span className="font-mono text-[11px] tabular-nums text-emerald-300">
                  {pct(a.confidence)}
                </span>
              </div>
            ))}
          </div>
        </Section>
      )}

      {raw && (
        <Section title="Raw VLM response">
          <pre className="scroll-thin max-h-64 overflow-auto whitespace-pre-wrap break-words rounded border border-ink-800 bg-ink-950/50 p-3 font-mono text-[11px] text-ink-300">
            {raw}
          </pre>
        </Section>
      )}
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <h4 className="mb-1 text-[10px] uppercase tracking-wider text-ink-500">{title}</h4>
      {children}
    </div>
  );
}

// Keyframe thumbnail with proposal bboxes overlaid as labelled rectangles.
// Renders the actual image the VLM proposer looked at + a deterministic
// color-per-proposal so the user can match the row in the table to the box
// on the image.
function KeyframeWithBboxes({
  sceneId,
  frameName,
  proposals,
}: {
  sceneId: string;
  frameName: string;
  proposals: ProposalRow[];
}) {
  const url = useMemo(() => frameUrl(sceneId, frameName), [sceneId, frameName]);
  const [hovered, setHovered] = useState<number | null>(null);
  const [errored, setErrored] = useState(false);

  if (errored) {
    return (
      <div className="rounded border border-ink-800 bg-ink-900/40 p-3 text-[11px] text-ink-500">
        Couldn&apos;t load <code>{frameName}</code>.
      </div>
    );
  }

  return (
    <div className="relative overflow-hidden rounded border border-ink-800 bg-ink-950">
      {/* Plain img — Next/Image gives no optimisation benefit because these
          frames are already JPEG-quality-compressed by ffmpeg. */}
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={url}
        alt={frameName}
        className="block w-full"
        onError={() => setErrored(true)}
      />
      <svg
        viewBox="0 0 1 1"
        preserveAspectRatio="none"
        className="pointer-events-none absolute inset-0 h-full w-full"
      >
        {proposals.map((p, i) => {
          if (!p.bbox_norm || p.bbox_norm.length < 4) return null;
          const [x, y, w, h] = p.bbox_norm;
          const color = bboxColor(i);
          const isHovered = hovered === i;
          return (
            <g key={i}>
              <rect
                x={x}
                y={y}
                width={w}
                height={h}
                fill="none"
                stroke={color}
                strokeWidth={isHovered ? 0.006 : 0.003}
                vectorEffect="non-scaling-stroke"
                opacity={hovered != null && !isHovered ? 0.35 : 1}
              />
            </g>
          );
        })}
      </svg>
      {/* Numbered legend below the image so the SVG can stay pointer-events-none. */}
      <div className="flex flex-wrap gap-1 border-t border-ink-800 bg-ink-900/60 px-2 py-1.5">
        {proposals.map((p, i) => (
          <button
            key={i}
            type="button"
            onMouseEnter={() => setHovered(i)}
            onMouseLeave={() => setHovered(null)}
            className="flex items-center gap-1.5 rounded px-1.5 py-0.5 font-mono text-[10px] text-ink-300 hover:bg-ink-800"
          >
            <span
              className="size-2 rounded-sm"
              style={{ background: bboxColor(i) }}
            />
            <span className="max-w-[14ch] truncate">{p.phrase}</span>
            <span className="tabular-nums text-ink-500">{pct(p.confidence)}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

// Deterministic palette — distinct, high-contrast on dark backgrounds.
const BBOX_PALETTE = [
  "#22d3ee", "#f472b6", "#fbbf24", "#a3e635", "#fb7185",
  "#60a5fa", "#c084fc", "#34d399", "#f97316", "#e879f9",
];

function bboxColor(i: number): string {
  return BBOX_PALETTE[i % BBOX_PALETTE.length];
}
