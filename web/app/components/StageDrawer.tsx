// Pipeline drill-down drawer.
//
// Click a stage in PipelineProgress -> drawer opens on the right with a
// time-axis waterfall of every Logfire span tagged with this scene_id, scoped
// to the clicked stage. Selecting a span shows its raw + parsed VLM output
// (when present) and the full attribute payload.
//
// Data path: useTrace -> /api/trace/:scene_id (agent) -> Logfire Read API.

"use client";

import { useEffect, useMemo, useState } from "react";

import { SpanWaterfall } from "./SpanWaterfall";
import { VlmOutputCard } from "./VlmOutputCard";
import { useTrace } from "./useTrace";
import {
  nodeDurationS,
  type ManifestStatus,
  type TraceTreeNode,
} from "@/lib/types";
import { useUI, type DrillStage } from "@/store/ui";

const STAGE_TITLE: Record<DrillStage, string> = {
  capture: "Capture",
  poses: "Reconstruction (VGGT)",
  segmentation: "Segmentation",
};

const LOGFIRE_PROJECT_URL = process.env.NEXT_PUBLIC_LOGFIRE_PROJECT_URL ?? "";

// Build a deep-link to the Logfire dashboard for a given scene/trace/span.
// Logfire's UI wraps `?q=...` as a subquery (effectively
// `SELECT * FROM records WHERE span_id IN (<q>)`), so the inner SELECT must
// return exactly one column — `SELECT *` triggers a "Too many columns" error.
function logfireSceneUrl(sceneId: string): string | null {
  if (!LOGFIRE_PROJECT_URL) return null;
  const sql = `SELECT span_id FROM records WHERE attributes->>'scene_id' = '${sceneId.replace(/'/g, "''")}'`;
  return `${LOGFIRE_PROJECT_URL}?q=${encodeURIComponent(sql)}`;
}

// Format an ISO-8601 timestamp for display. Snapshotted spans from R2 can be
// missing/null in the wild — guard rather than crash the drawer.
function prettyTimestamp(ts: string | null | undefined): string {
  if (!ts) return "—";
  return ts.replace("T", " ").replace("Z", "");
}

function logfireSpanUrl(traceId: string | undefined): string | null {
  // Older agent builds don't return trace_id; without it the per-span deep
  // link is unconstructable. Skip rather than crash on undefined.replace().
  if (!LOGFIRE_PROJECT_URL || !traceId) return null;
  return `${LOGFIRE_PROJECT_URL}?q=${encodeURIComponent(`SELECT span_id FROM records WHERE trace_id = '${traceId.replace(/'/g, "''")}'`)}`;
}

// Which span name prefixes belong to which drill stage. The lookup is fuzzy
// (startsWith) so future sub-spans (e.g. inference.vggt.frame) drop in
// without touching this map.
const STAGE_PREFIXES: Record<DrillStage, string[]> = {
  capture: ["capture.", "modal.process_video", "modal.ffprobe", "modal.ffmpeg_extract"],
  poses: ["inference.", "modal.run_inference", "modal.subprocess"],
  segmentation: ["segmentation.", "modal.run_segmentation"],
};

function flatten(roots: TraceTreeNode[]): TraceTreeNode[] {
  const out: TraceTreeNode[] = [];
  const walk = (n: TraceTreeNode) => {
    out.push(n);
    for (const c of n.children) walk(c);
  };
  for (const r of roots) walk(r);
  return out;
}

// Filter the full trace tree down to the spans relevant to this stage.
// Matching is by span_name prefix; we keep all descendants of any match so
// the waterfall keeps its hierarchy.
function filterByStage(roots: TraceTreeNode[], stage: DrillStage): TraceTreeNode[] {
  const prefixes = STAGE_PREFIXES[stage];
  const matches = (name: string) => prefixes.some((p) => name.startsWith(p));
  // Include any span that matches OR has an ancestor that matched. We keep
  // descendants by traversing into matched subtrees verbatim.
  const out: TraceTreeNode[] = [];
  const walk = (n: TraceTreeNode) => {
    if (matches(n.span_name)) {
      out.push(n);
      return; // children are already inside n.children — kept as-is.
    }
    for (const c of n.children) walk(c);
  };
  for (const r of roots) walk(r);
  return out;
}

interface Props {
  sceneId: string;
  manifestStatus: ManifestStatus | null;
}

export function StageDrawer({ sceneId, manifestStatus }: Props) {
  const openStage = useUI((s) => s.openStage);
  const setOpenStage = useUI((s) => s.setOpenStage);
  const trace = useTrace(sceneId, manifestStatus);
  const [selectedSpanId, setSelectedSpanId] = useState<string | null>(null);

  // Reset selection when the stage changes so we don't carry a span from
  // segmentation into the poses drawer.
  useEffect(() => {
    setSelectedSpanId(null);
  }, [openStage]);

  const stageRoots = useMemo(() => {
    if (!openStage || !trace.data) return [];
    return filterByStage(trace.data.tree, openStage);
  }, [openStage, trace.data]);

  const selectedSpan = useMemo(() => {
    if (!selectedSpanId) return null;
    return flatten(stageRoots).find((n) => n.span_id === selectedSpanId) ?? null;
  }, [selectedSpanId, stageRoots]);

  // Esc closes the drawer.
  useEffect(() => {
    if (!openStage) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpenStage(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [openStage, setOpenStage]);

  if (!openStage) return null;

  return (
    <div className="fixed inset-0 z-40 flex">
      <button
        type="button"
        aria-label="Close drill-down"
        onClick={() => setOpenStage(null)}
        className="flex-1 bg-ink-950/60 backdrop-blur-sm"
      />
      <div className="flex h-full w-full max-w-3xl flex-col border-l border-ink-800 bg-ink-950 shadow-2xl">
        <header className="flex items-baseline justify-between border-b border-ink-800 px-5 py-3">
          <div>
            <h2 className="text-base font-semibold tracking-tight">
              {STAGE_TITLE[openStage]}
            </h2>
            <p className="mt-0.5 font-mono text-[10px] uppercase tracking-wider text-ink-500">
              live trace · scene {sceneId}
              {trace.loading && " · loading…"}
              {trace.error && ` · ${trace.error.slice(0, 80)}`}
            </p>
          </div>
          <div className="flex items-center gap-3">
            {trace.data && (
              <CostBadge
                totalUsd={trace.data.cost.total_usd}
                callCount={trace.data.cost.call_count}
              />
            )}
            {logfireSceneUrl(sceneId) && (
              <a
                href={logfireSceneUrl(sceneId) ?? "#"}
                target="_blank"
                rel="noopener noreferrer"
                title="Open canonical Logfire view, filtered to this scene"
                className="rounded-md border border-emerald-500/30 bg-emerald-500/5 px-2 py-1 font-mono text-[10px] uppercase tracking-wider text-emerald-300 hover:border-emerald-400/60"
              >
                logfire ↗
              </a>
            )}
            <button
              type="button"
              onClick={() => setOpenStage(null)}
              className="rounded-md border border-ink-700 bg-ink-900 px-2 py-1 font-mono text-[10px] uppercase tracking-wider text-ink-400 hover:border-ink-500 hover:text-ink-100"
            >
              Esc · close
            </button>
          </div>
        </header>

        <div className="grid min-h-0 flex-1 grid-cols-[1fr_minmax(0,22rem)] gap-0">
          <div className="scroll-thin min-h-0 overflow-y-auto border-r border-ink-800 p-4">
            <SpanWaterfall
              roots={stageRoots}
              onSelect={(n) => setSelectedSpanId(n.span_id)}
              selectedId={selectedSpanId}
            />
          </div>
          <div className="scroll-thin min-h-0 overflow-y-auto p-4">
            {selectedSpan ? (
              <SpanDetail node={selectedSpan} sceneId={sceneId} />
            ) : (
              <div className="text-xs italic text-ink-500">
                Select a span to inspect its attributes and VLM evidence.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function CostBadge({
  totalUsd,
  callCount,
}: {
  totalUsd: number;
  callCount: number;
}) {
  const pretty =
    totalUsd >= 0.01
      ? `$${totalUsd.toFixed(3)}`
      : totalUsd > 0
        ? `$${totalUsd.toFixed(5)}`
        : "$0";
  return (
    <span className="rounded-full border border-emerald-500/30 bg-emerald-500/10 px-3 py-1 font-mono text-[10px] uppercase tracking-wider text-emerald-200">
      {pretty} · {callCount} calls
    </span>
  );
}

function SpanDetail({ node, sceneId }: { node: TraceTreeNode; sceneId: string }) {
  const lfUrl = logfireSpanUrl(node.trace_id);
  const durSec = nodeDurationS(node);
  return (
    <div className="space-y-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold text-ink-100">{node.span_name}</h3>
          <p className="mt-1 font-mono text-[10px] text-ink-500">
            {durSec != null ? `${durSec.toFixed(3)}s` : "—"}
            {" · "}
            {prettyTimestamp(node.start_timestamp)}
          </p>
        </div>
        {lfUrl && (
          <a
            href={lfUrl}
            target="_blank"
            rel="noopener noreferrer"
            title="Open this trace in the canonical Logfire UI"
            className="shrink-0 rounded border border-emerald-500/30 bg-emerald-500/5 px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider text-emerald-300 hover:border-emerald-400/60"
          >
            trace ↗
          </a>
        )}
      </div>

      <VlmOutputCard node={node} sceneId={sceneId} />

      <details>
        <summary className="cursor-pointer text-[10px] uppercase tracking-wider text-ink-500 hover:text-ink-300">
          all attributes
        </summary>
        <pre className="scroll-thin mt-2 max-h-96 overflow-auto whitespace-pre-wrap break-words rounded border border-ink-800 bg-ink-950/60 p-3 font-mono text-[10px] text-ink-300">
          {JSON.stringify(node.attributes, null, 2)}
        </pre>
      </details>
    </div>
  );
}
