// Span waterfall — renders a tree of spans as nested horizontal bars over a
// shared time axis. No D3, no chart lib: plain divs with absolute-positioned
// bars. The shared axis is computed once at the top of the visible subtree.

"use client";

import { nodeDurationS, type TraceTreeNode } from "@/lib/types";

interface Props {
  /** The root span(s) of the subtree to render — typically the result of
   *  filterByStage(tree, stage) so we only show the relevant slice. */
  roots: TraceTreeNode[];
  /** Click handler — receives the clicked span. Used to populate the detail
   *  pane in the drawer. */
  onSelect: (node: TraceTreeNode) => void;
  /** Currently-selected span_id, for highlight. */
  selectedId: string | null;
}

interface Bounds {
  minMs: number;
  maxMs: number;
}

function bounds(roots: TraceTreeNode[]): Bounds {
  let minMs = Infinity;
  let maxMs = -Infinity;
  const visit = (n: TraceTreeNode) => {
    const start = Date.parse(n.start_timestamp);
    const end = Date.parse(n.end_timestamp);
    if (Number.isFinite(start)) minMs = Math.min(minMs, start);
    if (Number.isFinite(end)) maxMs = Math.max(maxMs, end);
    for (const c of n.children) visit(c);
  };
  for (const r of roots) visit(r);
  if (!Number.isFinite(minMs) || !Number.isFinite(maxMs) || maxMs <= minMs) {
    return { minMs: 0, maxMs: 1 };
  }
  return { minMs, maxMs };
}

// Color the bar by span family — visual cue distinguishing model calls,
// modal wrappers, and pipeline stages at a glance.
function barColor(name: string): string {
  if (name.includes("vlm_label") || name.includes("vlm_proposal")) return "bg-fuchsia-500/70";
  if (name.includes("sam3")) return "bg-cyan-500/70";
  if (name.startsWith("inference.")) return "bg-emerald-500/70";
  if (name.startsWith("modal.")) return "bg-slate-500/60";
  if (name.startsWith("segmentation.lift")) return "bg-amber-500/60";
  if (name.startsWith("capture.")) return "bg-blue-500/70";
  return "bg-ink-600";
}

function fmt(seconds: number | null): string {
  if (seconds == null || !Number.isFinite(seconds)) return "—";
  if (seconds < 0.001) return `${(seconds * 1_000_000).toFixed(0)}μs`;
  if (seconds < 1) return `${(seconds * 1000).toFixed(0)}ms`;
  if (seconds < 60) return `${seconds.toFixed(2)}s`;
  return `${Math.floor(seconds / 60)}m ${(seconds % 60).toFixed(0)}s`;
}

export function SpanWaterfall({ roots, onSelect, selectedId }: Props) {
  if (roots.length === 0) {
    return (
      <div className="rounded border border-ink-800 bg-ink-900/30 p-4 text-center text-xs text-ink-500">
        No spans yet for this stage. Once the pipeline reaches this stage and
        the next manifest tick fires, spans will stream in here.
      </div>
    );
  }
  const b = bounds(roots);
  const totalMs = b.maxMs - b.minMs;
  return (
    <div className="space-y-0.5 overflow-x-hidden font-mono text-[11px]">
      {roots.map((r) => (
        <Row
          key={r.span_id}
          node={r}
          depth={0}
          bounds={b}
          totalMs={totalMs}
          onSelect={onSelect}
          selectedId={selectedId}
        />
      ))}
    </div>
  );
}

function Row({
  node,
  depth,
  bounds,
  totalMs,
  onSelect,
  selectedId,
}: {
  node: TraceTreeNode;
  depth: number;
  bounds: Bounds;
  totalMs: number;
  onSelect: (n: TraceTreeNode) => void;
  selectedId: string | null;
}) {
  const start = Date.parse(node.start_timestamp);
  const end = Date.parse(node.end_timestamp);
  const leftPct = ((start - bounds.minMs) / totalMs) * 100;
  const widthPct = Math.max(0.3, ((end - start) / totalMs) * 100);
  const selected = selectedId === node.span_id;
  return (
    <>
      <button
        type="button"
        onClick={() => onSelect(node)}
        className={[
          "group grid w-full grid-cols-[1fr_auto] items-center gap-3 rounded px-2 py-1 text-left",
          selected
            ? "bg-accent-500/15 ring-1 ring-accent-400/60"
            : "hover:bg-ink-800/60",
        ].join(" ")}
      >
        <div className="flex min-w-0 items-center gap-2">
          <span
            className="shrink-0 text-ink-700"
            style={{ marginLeft: depth * 12 }}
          >
            {node.children.length > 0 ? "▾" : "·"}
          </span>
          <span className="truncate text-ink-200">{node.span_name}</span>
        </div>
        <div className="relative h-3 w-48 shrink-0 rounded-sm bg-ink-900/80 ring-1 ring-ink-800">
          <div
            className={`absolute top-0 h-full rounded-sm ${barColor(node.span_name)}`}
            style={{ left: `${leftPct}%`, width: `${widthPct}%` }}
          />
          <span className="absolute -right-12 top-1/2 -translate-y-1/2 text-[10px] tabular-nums text-ink-400">
            {fmt(nodeDurationS(node))}
          </span>
        </div>
      </button>
      {node.children.map((c) => (
        <Row
          key={c.span_id}
          node={c}
          depth={depth + 1}
          bounds={bounds}
          totalMs={totalMs}
          onSelect={onSelect}
          selectedId={selectedId}
        />
      ))}
    </>
  );
}
